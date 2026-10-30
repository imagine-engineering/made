use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativeResolvedPayload, NativeStoredPayload,
    NativeValue, NativeValueType, NodeRegistry, built_in_source_schema,
    native_value_matches_input_schema, native_value_types_for_input_schema,
};
use sim_model::{
    GENERATED_LATENT_FORMATS, LatentFormatDefinition, LatentFormatRegistry, LatentTensorLayout,
    LatentTransform, VaeTilePlan, process_latent_in, process_latent_out,
};
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, Layout,
    MemoryFormatReference, NativeLatentBundle, NativeLatentMetadata, NativeLatentSamples,
    NativeTensorPayload, NativeTensorRole, Scalar, ScalarSide, Tensor, TensorBackend,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_06::round_method_with_context_exact_native,
    generated_elementwise_or_runtime_operation_09::clamp_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_native_diffusion::conv2d,
    generated_shape_layout_transform_02::{
        tensor_reshape_with_context_exact_native, torch_cat_with_context_exact_native,
        torch_movedim_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        InterpolateConfiguration, InterpolateMode, interpolate_tensor_with_context_exact_native,
    },
    generated_storage_dtype_device_01::clone_with_context_exact_native,
    generated_tensor_creation_01::ones_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "VAEDecodeTiled",
    "VAEEncode",
    "VAEEncodeForInpaint",
    "VAEEncodeTiled",
    "VOIDWarpedNoiseSource",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const FEATURES: [&str; 5] = [
    "COMFY-NODE-0714",
    "COMFY-NODE-0716",
    "COMFY-NODE-0718",
    "COMFY-NODE-0719",
    "COMFY-NODE-0745",
];
struct VaeNode {
    class_type: &'static str,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let registry = NodeRegistry::built_in().map_err(schema_error)?;
    NODE_DESCRIPTOR_IDS
        .iter()
        .zip(FEATURES)
        .map(|(class_type, feature_id)| {
            let catalog = built_in_source_schema(class_type).map_err(schema_error)?;
            let row = registry
                .descriptor(class_type)
                .ok_or_else(|| schema_error("Missing VAE row"))?;
            let names: Vec<_> = catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect();
            let decode = *class_type == "VAEDecodeTiled";
            let noise = *class_type == "VOIDWarpedNoiseSource";
            let output_name = if noise {
                "noise"
            } else if decode {
                "image"
            } else {
                "latent"
            };
            let output_type = if noise {
                NativeHandleType::new(NativeHandleKind::StructuredCompute, "NOISE")?
            } else if decode {
                NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?
            } else {
                NativeHandleType::new(NativeHandleKind::Latent, "LATENT")?
            };
            let source_schema = catalog
                .bind_execution_ports(&names, &[], &[output_name.to_owned()])
                .map_err(schema_error)?;
            let inputs = catalog
                .inputs
                .iter()
                .map(|input| {
                    Ok(NativeInputDescriptor {
                        name: input.schema.name.clone(),
                        accepted_types: native_value_types_for_input_schema(&input.schema)
                            .map_err(schema_error)?,
                        required: input.requirement == NativeInputRequirement::Required,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: !matches!(
                            input.schema.name.as_str(),
                            "pixels" | "samples" | "vae" | "mask" | "warped_noise"
                        ),
                    })
                })
                .collect::<Result<_, NativeNodeContractError>>()?;
            let node = VaeNode { class_type };
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
                        name: output_name.to_owned(),
                        produced_type: NativeValueType::Handle(output_type),
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
                    search_aliases: if *class_type == "VAEEncode" {
                        ["encode", "encode image", "image to latent"]
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
fn schema_error(error: impl std::fmt::Display) -> NativeNodeContractError {
    NativeNodeContractError::InvalidSourceSchema(error.to_string())
}
impl VaeNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        if inputs.keys().any(|name| {
            !catalog
                .inputs
                .iter()
                .any(|input| input.schema.name == *name)
        }) {
            return Err(invalid("Unassigned VAE input"));
        }
        for input in &catalog.inputs {
            let value = inputs
                .get(&input.schema.name)
                .ok_or_else(|| invalid(format!("Missing {}", input.schema.name)))?;
            value.validate().map_err(invalid)?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid("Required mapped VAE inputs cannot be empty"));
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
impl NativeNode for VaeNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        if self.class_type == "VOIDWarpedNoiseSource" {
            "source-latent-noise-part-seven-canonical-v1"
        } else {
            "source-image-vae-part-seven-canonical-v1"
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
            if self.class_type == "VOIDWarpedNoiseSource" {
                let latent = resolve(&context, &inputs, "warped_noise")?;
                let NativeStoredPayload::Latent(bundle) = latent.as_ref() else {
                    return Err(invalid("Canonical warped LATENT required"));
                };
                let samples = bundle
                    .samples()
                    .tensor()
                    .ok_or_else(|| invalid("Original Noise_FromLatent requires tensor samples"))?;
                let noise = sim_sampler::NativeNoisePayload::latent_source(samples.clone())
                    .map_err(|error| failure(&context, error))?;
                return publish(&context, NativeStoredPayload::Noise(Arc::new(noise)));
            }
            let model = resolve(&context, &inputs, "vae")?;
            let NativeStoredPayload::Model(model) = model.as_ref() else {
                return Err(invalid("Canonical VAE model resource required"));
            };
            let vae = model
                .model_payload()
                .vae()
                .ok_or_else(|| invalid("Canonical NativeVae resource required"))?;
            let formats =
                LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS).map_err(invalid)?;
            let format = formats
                .get(&vae.descriptor().latent_format().identity)
                .ok_or_else(|| invalid("VAE latent format unavailable"))?;
            validate_source_latent_transform(format)?;
            if format.dimensions != 2 || format.layout != LatentTensorLayout::ChannelsFirst {
                return Err(invalid(
                    "Source audio/video VAE wrapper geometry is not yet bound",
                ));
            }
            let payload = if self.class_type == "VAEDecodeTiled" {
                let latent = resolve(&context, &inputs, "samples")?;
                let NativeStoredPayload::Latent(bundle) = latent.as_ref() else {
                    return Err(invalid("Canonical LATENT required"));
                };
                let samples = match bundle.samples() {
                    NativeLatentSamples::Tensor(samples) => samples,
                    NativeLatentSamples::AudioVideo { .. } => {
                        return Err(invalid("Source tiled decode requires a tensor latent"));
                    }
                };
                let source_shape = samples.descriptor().shape();
                let mut overlap = integer(&inputs, "overlap")?;
                let tile_size = integer(&inputs, "tile_size")?;
                if tile_size
                    < overlap
                        .checked_mul(4)
                        .ok_or_else(|| invalid("VAE tile overlap overflow"))?
                {
                    overlap = tile_size / 4;
                }
                let compression = vae
                    .descriptor()
                    .source_image_decode_upscale_ratio()
                    .map_err(invalid)?;
                let extent = tile_size
                    .checked_div(compression)
                    .ok_or_else(|| invalid("Invalid source decode compression"))?;
                let overlap = overlap
                    .checked_div(compression)
                    .ok_or_else(|| invalid("Invalid source decode compression"))?;
                // NativeVae consumes model-space latents; workflow samples retain the source-unscaled values.
                let samples = process_latent_in(format, compute.backend(), samples, &execution)
                    .map_err(|error| failure(&context, error))?;
                let plan = vae
                    .plan_decode_tiles(&samples, vec![extent, extent], vec![overlap, overlap])
                    .map_err(|error| failure(&context, error))?;
                validate_source_decode_canvas(
                    &plan,
                    source_shape,
                    compression,
                    vae.descriptor().pixel_channels(),
                )?;
                let images = vae
                    .decode_tiled(compute.backend(), &samples, &plan, &execution)
                    .map_err(|error| failure(&context, error))?;
                let images =
                    torch_movedim_exact_native(&images, &[1], &[-1], &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                let images = clone_with_context_exact_native(
                    compute.backend(),
                    &images,
                    MemoryFormatReference::Layout(Layout::Contiguous),
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
                let images =
                    ImageTensor::from_tensor(images).map_err(|error| failure(&context, error))?;
                NativeStoredPayload::Tensor(Arc::new(
                    NativeTensorPayload::from_image(NativeTensorRole::Image, images)
                        .map_err(|error| failure(&context, error))?,
                ))
            } else {
                let image = resolve(&context, &inputs, "pixels")?;
                let NativeStoredPayload::Tensor(image) = image.as_ref() else {
                    return Err(invalid("Canonical IMAGE tensor required"));
                };
                if image.role() != NativeTensorRole::Image {
                    return Err(invalid("IMAGE tensor role required"));
                }
                let prepared = if self.class_type == "VAEEncodeForInpaint" {
                    let mask = resolve(&context, &inputs, "mask")?;
                    let NativeStoredPayload::Tensor(mask) = mask.as_ref() else {
                        return Err(invalid("Canonical MASK tensor required"));
                    };
                    if mask.role() != NativeTensorRole::Mask {
                        return Err(invalid("MASK tensor role required"));
                    }
                    Some(prepare_source_inpaint(
                        &context,
                        compute.backend(),
                        image.tensor(),
                        mask.tensor(),
                        vae.descriptor()
                            .source_image_encode_downscale_ratio()
                            .map_err(invalid)?,
                        integer(&inputs, "grow_mask_by")?,
                        &execution,
                    )?)
                } else {
                    None
                };
                let source = prepared
                    .as_ref()
                    .map_or(image.tensor(), |(pixels, _)| pixels);
                let channels = *source
                    .descriptor()
                    .shape()
                    .last()
                    .ok_or_else(|| invalid("Image channel axis absent"))?;
                let pixels = narrow_method_exact_native(
                    source,
                    3,
                    0,
                    channels.min(vae.descriptor().pixel_channels()),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
                let pixels =
                    torch_movedim_exact_native(&pixels, &[-1], &[1], &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                let tiled = self.class_type == "VAEEncodeTiled";
                let plan = if tiled {
                    let tile_size = integer(&inputs, "tile_size")?;
                    let overlap = integer(&inputs, "overlap")?;
                    vae.plan_encode_tiles(
                        &pixels,
                        vec![tile_size, tile_size],
                        vec![overlap, overlap],
                    )
                } else {
                    vae.plan_encode(&pixels)
                }
                .map_err(|error| failure(&context, error))?;
                validate_source_encode_crop_and_canvas(
                    &plan,
                    pixels.descriptor().shape(),
                    vae.descriptor()
                        .source_image_encode_downscale_ratio()
                        .map_err(invalid)?,
                    tiled,
                    format.channels,
                )?;
                let encoded = vae
                    .encode_tiled(compute.backend(), &pixels, &plan, &execution)
                    .map_err(|error| failure(&context, error))?;
                // NativeVae returns model-space latents; the original node publishes the raw first-stage samples.
                let samples = process_latent_out(format, compute.backend(), &encoded, &execution)
                    .map_err(|error| failure(&context, error))?;
                NativeStoredPayload::Latent(Arc::new(
                    NativeLatentBundle::single(
                        samples,
                        prepared.map(|(_, mask)| mask),
                        None,
                        NativeLatentMetadata::default(),
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?,
                ))
            };
            publish(&context, payload)
        })
    }
}
pub fn prepare_source_inpaint(
    context: &NativeNodeContext,
    backend: &CpuBackend,
    pixels: &Tensor,
    mask: &Tensor,
    compression: u64,
    grow: u64,
    execution: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), NativeNodeFailure> {
    let [_, height, width, channels] = pixels.descriptor().shape() else {
        return Err(invalid("Source inpaint requires BHWC pixels"));
    };
    let [mask_batch, mask_height, mask_width] = mask.descriptor().shape() else {
        return Err(invalid(
            "Source inpaint MASK leading dimensions are not yet bound",
        ));
    };
    if *channels < 3
        || pixels.descriptor().dtype() != DType::F32
        || mask.descriptor().dtype() != DType::F32
    {
        return Err(invalid(
            "Source inpaint requires canonical F32 RGB pixels and F32 masks",
        ));
    }
    let output_height = height
        .checked_div(compression)
        .and_then(|value| value.checked_mul(compression))
        .ok_or_else(|| invalid("Invalid source inpaint compression"))?;
    let output_width = width
        .checked_div(compression)
        .and_then(|value| value.checked_mul(compression))
        .ok_or_else(|| invalid("Invalid source inpaint compression"))?;
    let height_offset = height
        .checked_rem(compression)
        .ok_or_else(|| invalid("Invalid source inpaint compression"))?
        / 2;
    let width_offset = width
        .checked_rem(compression)
        .ok_or_else(|| invalid("Invalid source inpaint compression"))?
        / 2;
    let height_offset = i64::try_from(height_offset).map_err(invalid)?;
    let width_offset = i64::try_from(width_offset).map_err(invalid)?;
    let shape = [*mask_batch, 1, *mask_height, *mask_width]
        .into_iter()
        .map(|extent| i64::try_from(extent).map_err(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    let mask = tensor_reshape_with_context_exact_native(backend, mask, &shape, execution)
        .map_err(|error| failure(context, error))?;
    let mask = interpolate_tensor_with_context_exact_native(
        backend,
        &mask,
        &InterpolateConfiguration {
            output_size: Some(vec![
                usize::try_from(*height).map_err(invalid)?,
                usize::try_from(*width).map_err(invalid)?,
            ]),
            scale_factor: None,
            mode: InterpolateMode::Bilinear,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let mask = narrow_method_exact_native(
        &mask,
        2,
        height_offset,
        output_height,
        &context.cancellation,
    )
    .and_then(|mask| {
        narrow_method_exact_native(&mask, 3, width_offset, output_width, &context.cancellation)
    })
    .map_err(|error| failure(context, error))?;
    let rounded = round_method_with_context_exact_native(backend, &mask, 0, execution)
        .map_err(|error| failure(context, error))?;
    let grown = if grow == 0 {
        mask
    } else {
        let kernel = ones_with_context_exact_native(
            backend,
            &[1, 1, grow, grow],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            execution,
        )
        .map_err(|error| failure(context, error))?;
        let padding = usize::try_from(grow / 2).map_err(invalid)?;
        let grown = conv2d(backend, &rounded, &kernel, None, 1, padding, execution)
            .map_err(|error| failure(context, error))?;
        clamp_with_context_exact_native(
            backend,
            &grown,
            Some(Scalar::Float(0.0)),
            Some(Scalar::Float(1.0)),
            execution,
        )
        .map_err(|error| failure(context, error))?
    };
    // Even-sized source kernels produce an extra row and column, cropped from the top left.
    let grown = narrow_method_exact_native(&grown, 2, 0, output_height, &context.cancellation)
        .and_then(|mask| {
            narrow_method_exact_native(&mask, 3, 0, output_width, &context.cancellation)
        })
        .map_err(|error| failure(context, error))?;
    let noise_mask = round_method_with_context_exact_native(backend, &grown, 0, execution)
        .map_err(|error| failure(context, error))?;
    let rounded_descriptor = TensorDescriptor::contiguous(
        rounded.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| failure(context, error))?;
    let (keep, _) = backend
        .binary_scalar(
            BinaryOperation::Subtract,
            &rounded,
            Scalar::Float(1.0),
            ScalarSide::Left,
            rounded_descriptor,
            execution,
        )
        .map_err(|error| failure(context, error))?;
    let keep = torch_movedim_exact_native(&keep, &[1], &[-1], &context.cancellation)
        .map_err(|error| failure(context, error))?;
    let pixels = narrow_method_exact_native(
        pixels,
        1,
        height_offset,
        output_height,
        &context.cancellation,
    )
    .and_then(|pixels| {
        narrow_method_exact_native(
            &pixels,
            2,
            width_offset,
            output_width,
            &context.cancellation,
        )
    })
    .map_err(|error| failure(context, error))?;
    let rgb = narrow_method_exact_native(&pixels, 3, 0, 3, &context.cancellation)
        .map_err(|error| failure(context, error))?;
    let rgb_descriptor = TensorDescriptor::contiguous(
        rgb.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| failure(context, error))?;
    let (centered, _) = backend
        .binary_scalar(
            BinaryOperation::Subtract,
            &rgb,
            Scalar::Float(0.5),
            ScalarSide::Right,
            rgb_descriptor.clone(),
            execution,
        )
        .map_err(|error| failure(context, error))?;
    // The output descriptor retains the receiver batch, enforcing the source in-place broadcast constraint.
    let (masked, _) = backend
        .binary(
            BinaryOperation::Multiply,
            &centered,
            &keep,
            rgb_descriptor.clone(),
            execution,
        )
        .map_err(|error| failure(context, error))?;
    let (rgb, _) = backend
        .binary_scalar(
            BinaryOperation::Add,
            &masked,
            Scalar::Float(0.5),
            ScalarSide::Right,
            rgb_descriptor,
            execution,
        )
        .map_err(|error| failure(context, error))?;
    let pixels = if *channels == 3 {
        rgb
    } else {
        let remaining =
            narrow_method_exact_native(&pixels, 3, 3, channels - 3, &context.cancellation)
                .map_err(|error| failure(context, error))?;
        torch_cat_with_context_exact_native(backend, &[rgb, remaining], 3, execution)
            .map_err(|error| failure(context, error))?
    };
    Ok((pixels, noise_mask))
}

fn validate_source_latent_transform(
    format: &LatentFormatDefinition,
) -> Result<(), NativeNodeFailure> {
    if format.transform != LatentTransform::Identity {
        return Err(NativeNodeFailure {
            code: "source_latent_transform_unbound".to_owned(),
            message: "Canonical model-space conversion cannot prove unchanged source latent bits"
                .to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        });
    }
    Ok(())
}

fn source_geometry_unbound(message: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "source_image_geometry_unbound".to_owned(),
        message: message.to_owned(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn validate_source_encode_crop_and_canvas(
    plan: &VaeTilePlan,
    pixels: &[u64],
    compression: u64,
    tiled: bool,
    channels: u64,
) -> Result<(), NativeNodeFailure> {
    let [batch, _, height, width] = pixels else {
        return Err(invalid("Source image encoding requires BCHW pixels"));
    };
    let source_height = height
        .checked_div(compression)
        .ok_or_else(|| invalid("Invalid source encode compression"))?;
    let source_width = width
        .checked_div(compression)
        .ok_or_else(|| invalid("Invalid source encode compression"))?;
    let effective = [
        source_height
            .checked_mul(compression)
            .ok_or_else(|| invalid("Source crop height overflow"))?,
        source_width
            .checked_mul(compression)
            .ok_or_else(|| invalid("Source crop width overflow"))?,
    ];
    let offsets = [
        height
            .checked_rem(compression)
            .ok_or_else(|| invalid("Invalid source encode compression"))?
            / 2,
        width
            .checked_rem(compression)
            .ok_or_else(|| invalid("Invalid source encode compression"))?
            / 2,
    ];
    if plan.effective_input_spatial_shape() != effective || plan.input_spatial_offsets() != offsets
    {
        return Err(source_geometry_unbound(
            "Canonical encoder crop differs from the pinned source wrapper",
        ));
    }
    if tiled && plan.output_shape() != [*batch, channels, source_height, source_width] {
        return Err(source_geometry_unbound(
            "Canonical tiled encoder canvas differs from the pinned source wrapper",
        ));
    }
    Ok(())
}
fn validate_source_decode_canvas(
    plan: &VaeTilePlan,
    samples: &[u64],
    compression: u64,
    channels: u64,
) -> Result<(), NativeNodeFailure> {
    let [batch, _, height, width] = samples else {
        return Err(invalid("Source image decoding requires BCHW samples"));
    };
    let height = height
        .checked_mul(compression)
        .ok_or_else(|| invalid("Source decode canvas height overflow"))?;
    let width = width
        .checked_mul(compression)
        .ok_or_else(|| invalid("Source decode canvas width overflow"))?;
    if plan.output_shape() != [*batch, channels, height, width] {
        return Err(source_geometry_unbound(
            "Canonical tiled decoder canvas differs from the pinned source wrapper",
        ));
    }
    Ok(())
}

fn integer(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).map_err(invalid),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value),
        _ => Err(invalid(format!("Integer {name} required"))),
    }
}
fn resolve(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
        return Err(invalid(format!("Handle {name} required")));
    };
    context
        .handle_store()
        .resolve(value, value.handle_type(), &context.cancellation)
        .map_err(|error| failure(context, error))
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
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_vae_inputs".to_owned(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonidentity_formats_are_unbound_without_a_lossless_source_vae_facade()
    -> Result<(), Box<dyn std::error::Error>> {
        let raw = f32::from_bits(0x3e45614c);
        let scale = 0.18215_f32;
        assert_ne!((raw * scale / scale).to_bits(), raw.to_bits());
        for format in GENERATED_LATENT_FORMATS {
            let result = validate_source_latent_transform(format);
            if format.transform == LatentTransform::Identity {
                result?;
            } else {
                let error = result
                    .err()
                    .ok_or("Nonidentity source transform accepted")?;
                assert_eq!(error.code, "source_latent_transform_unbound");
                assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            }
        }
        Ok(())
    }
}
