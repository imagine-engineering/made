use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativeResolvedPayload,
    NativeStoredModelPayload, NativeStoredPayload, NativeValue, NativeValueType, NodeRegistry,
    built_in_source_schema, generated_model_latent_01::resize_latent,
    native_value_matches_input_schema, native_value_types_for_input_schema,
};
use sim_model::{NativeLatentOperation, NativeModelPayload};
use sim_tensor::{
    CpuBackend, ExecutionContext, Layout, MemoryFormatReference, NativeLatentBundle,
    NativeLatentNoiseMask, NativeLatentSamples, ResizeCrop, ResizeMode, ResizeSpec, Scalar, Tensor,
    TensorBackend, TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, real_multiply_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_04::nan_to_num_with_context_exact_native,
    generated_elementwise_or_runtime_operation_05::div_with_context_exact_native,
    generated_elementwise_or_runtime_operation_08::rot90_with_context_exact_native,
    generated_elementwise_or_runtime_operation_16::add_method_with_context_exact_native,
    generated_elementwise_or_runtime_operation_20::flip_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_linear_algebra_01::vector_norm_with_context_exact_native,
    generated_shape_layout_transform_02::{
        tensor_repeat_with_context_exact_native, tensor_reshape_with_context_exact_native,
        torch_movedim_exact_native,
    },
    generated_spatial_functional_kernel_01::bislerp_tensor_with_context_exact_native,
    generated_storage_dtype_device_01::clone_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "LatentFlip",
    "LatentFromBatch",
    "LatentInterpolate",
    "LatentMultiply",
    "LatentOperationSharpen",
    "LatentOperationTonemapReinhard",
    "LatentRotate",
    "LatentSubtract",
    "LatentUpscale",
    "LatentUpscaleBy",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
struct LatentNode {
    class_type: &'static str,
}
const FEATURES: [&str; 10] = [
    "COMFY-NODE-0322",
    "COMFY-NODE-0323",
    "COMFY-NODE-0324",
    "COMFY-NODE-0325",
    "COMFY-NODE-0326",
    "COMFY-NODE-0327",
    "COMFY-NODE-0328",
    "COMFY-NODE-0329",
    "COMFY-NODE-0330",
    "COMFY-NODE-0331",
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
            let source_schema = catalog
                .bind_execution_ports(
                    &names,
                    &[],
                    &[if operation_node(class_type) {
                        "operation"
                    } else {
                        "latent"
                    }
                    .to_owned()],
                )
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
                        name: if operation_node(class_type) {
                            "operation"
                        } else {
                            "latent"
                        }
                        .to_owned(),
                        produced_type: NativeValueType::Handle(if operation_node(class_type) {
                            NativeHandleType::new(NativeHandleKind::Model, "LATENT_OPERATION")?
                        } else {
                            latent_type()?
                        }),
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
                    output_names: vec![
                        if operation_node(class_type) {
                            "operation"
                        } else {
                            "latent"
                        }
                        .to_owned(),
                    ],
                    search_aliases: match *class_type {
                        "LatentFlip" => vec!["mirror latent"],
                        "LatentFromBatch" => {
                            vec!["select from batch", "pick latent", "batch subset"]
                        }
                        "LatentInterpolate" => {
                            vec!["blend latent", "mix latent", "lerp latent", "transition"]
                        }
                        "LatentMultiply" => vec!["scale latent", "amplify latent", "latent gain"],
                        "LatentOperationTonemapReinhard" => vec!["hdr latent"],
                        "LatentSubtract" => vec!["difference latent", "remove features"],
                        "LatentUpscale" => vec!["enlarge latent", "resize latent"],
                        "LatentUpscaleBy" => {
                            vec!["enlarge latent", "resize latent", "scale latent"]
                        }
                        _ => Vec::new(),
                    }
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
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
        "source-latent-part-four-v1"
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
            let payload = if operation_node(self.class_type) {
                let operation = if self.class_type == "LatentOperationTonemapReinhard" {
                    NativeLatentOperation::tonemap_reinhard(number(&inputs, "multiplier")? as f32)
                } else {
                    NativeLatentOperation::sharpen(
                        u32::try_from(input_integer(&inputs, "sharpen_radius")?)
                            .map_err(invalid)?,
                        number(&inputs, "sigma")? as f32,
                        number(&inputs, "alpha")? as f32,
                    )
                }
                .map_err(invalid)?;
                NativeStoredPayload::Model(Arc::new(
                    NativeStoredModelPayload::model_resource(Arc::new(
                        NativeModelPayload::latent_operation(Arc::new(operation))
                            .map_err(invalid)?,
                    ))
                    .map_err(invalid)?,
                ))
            } else {
                let source_name =
                    if matches!(self.class_type, "LatentSubtract" | "LatentInterpolate") {
                        "samples1"
                    } else {
                        "samples"
                    };
                let retained = resolve(&context, &inputs, source_name)?;
                let first = latent(&retained)?;
                let samples = first
                    .samples()
                    .tensor()
                    .ok_or_else(|| invalid("This source node requires tensor LATENT samples"))?;
                if self.class_type == "LatentUpscale"
                    && input_integer(&inputs, "width")? == 0
                    && input_integer(&inputs, "height")? == 0
                {
                    check(&context)?;
                    return outcome(handle(input(&inputs, source_name)?)?.clone());
                }
                let output = match self.class_type {
                    "LatentFlip" => {
                        let method = string(input(&inputs, "flip_method")?)?;
                        flip_with_context_exact_native(
                            compute.backend(),
                            samples,
                            &[if method.starts_with('x') { 2 } else { 3 }],
                            &execution,
                        )
                        .map_err(invalid)?
                    }
                    "LatentRotate" => {
                        let rotation = string(input(&inputs, "rotation")?)?;
                        let turns = if rotation.starts_with("90") {
                            1
                        } else if rotation.starts_with("180") {
                            2
                        } else if rotation.starts_with("270") {
                            3
                        } else {
                            0
                        };
                        rot90_with_context_exact_native(
                            compute.backend(),
                            samples,
                            turns,
                            [3, 2],
                            &execution,
                        )
                        .map_err(invalid)?
                    }
                    "LatentMultiply" => scaled(
                        samples,
                        number(&inputs, "multiplier")?,
                        compute.backend(),
                        &execution,
                    )?,
                    "LatentFromBatch" => {
                        let (start, length) = batch_range(
                            samples,
                            signed(&inputs, "batch_index")?,
                            input_integer(&inputs, "length")?,
                        )?;
                        let view = narrow_method_exact_native(
                            samples,
                            0,
                            start,
                            length,
                            &context.cancellation,
                        )
                        .map_err(invalid)?;
                        cloned(&view, compute.backend(), &execution)?
                    }
                    "LatentSubtract" | "LatentInterpolate" => {
                        let second = resolve(&context, &inputs, "samples2")?;
                        let second = latent(&second)?
                            .samples()
                            .tensor()
                            .ok_or_else(|| invalid("Tensor LATENT required"))?;
                        let second =
                            matching_second(samples, second, compute.backend(), &execution)?;
                        if self.class_type == "LatentSubtract" {
                            add_method_with_context_exact_native(
                                compute.backend(),
                                samples,
                                ElementwiseOperand::Tensor(&second),
                                -1.0,
                                &execution,
                            )
                            .map_err(invalid)?
                        } else {
                            interpolate(
                                samples,
                                &second,
                                number(&inputs, "ratio")?,
                                compute.backend(),
                                &execution,
                            )?
                        }
                    }
                    "LatentUpscale" | "LatentUpscaleBy" => {
                        let shape = samples.descriptor().shape();
                        let (height, width) = spatial_size(shape)?;
                        let (target_width, target_height) = if self.class_type == "LatentUpscaleBy"
                        {
                            let scale = number(&inputs, "scale_by")?;
                            (
                                (width as f64 * scale).round_ties_even() as u64,
                                (height as f64 * scale).round_ties_even() as u64,
                            )
                        } else {
                            let mut target_width = input_integer(&inputs, "width")?;
                            let mut target_height = input_integer(&inputs, "height")?;
                            if target_width == 0 {
                                target_height = target_height.max(64);
                                target_width = ((width as f64 * target_height as f64
                                    / height as f64)
                                    .round_ties_even()
                                    as u64)
                                    .max(64);
                            } else if target_height == 0 {
                                target_width = target_width.max(64);
                                target_height = ((height as f64 * target_width as f64
                                    / width as f64)
                                    .round_ties_even()
                                    as u64)
                                    .max(64);
                            } else {
                                target_width = target_width.max(64);
                                target_height = target_height.max(64);
                            }
                            (target_width / 8, target_height / 8)
                        };
                        let crop = if self.class_type == "LatentUpscaleBy" {
                            "disabled"
                        } else {
                            string(input(&inputs, "crop")?)?
                        };
                        upscale(
                            samples,
                            target_width,
                            target_height,
                            string(input(&inputs, "upscale_method")?)?,
                            crop,
                            compute.backend(),
                            &execution,
                        )?
                    }
                    _ => return Err(invalid("Unknown latent node")),
                };
                let bundle = if self.class_type == "LatentFromBatch" {
                    let (start, length) = batch_range(
                        samples,
                        signed(&inputs, "batch_index")?,
                        input_integer(&inputs, "length")?,
                    )?;
                    let end = usize::try_from(start)
                        .map_err(invalid)?
                        .checked_add(usize::try_from(length).map_err(invalid)?)
                        .ok_or_else(|| invalid("Batch slice overflow"))?;
                    let indices = if let Some(indices) = first.batch_indices() {
                        let start = usize::try_from(start).map_err(invalid)?.min(indices.len());
                        Some(
                            indices
                                .get(start..end.min(indices.len()))
                                .ok_or_else(|| invalid("Batch index slice"))?
                                .to_vec(),
                        )
                    } else {
                        Some(
                            (u64::try_from(start).map_err(invalid)?
                                ..u64::try_from(start)
                                    .map_err(invalid)?
                                    .checked_add(length)
                                    .ok_or_else(|| invalid("Batch index overflow"))?)
                                .collect(),
                        )
                    };
                    let mask = match first.noise_mask() {
                        None => None,
                        Some(NativeLatentNoiseMask::Tensor(mask)) => {
                            let batch = *mask
                                .descriptor()
                                .shape()
                                .first()
                                .ok_or_else(|| invalid("Missing mask batch"))?;
                            if batch == 0 {
                                return Err(invalid("Cannot repeat empty mask batch"));
                            }
                            let mask = if batch == 1 {
                                cloned(mask, compute.backend(), &execution)?
                            } else {
                                let total = *samples
                                    .descriptor()
                                    .shape()
                                    .first()
                                    .ok_or_else(|| invalid("Missing batch"))?;
                                let repeated = if batch < total {
                                    tensor_repeat_with_context_exact_native(
                                        compute.backend(),
                                        mask,
                                        &[
                                            i64::try_from(total.div_ceil(batch))
                                                .map_err(invalid)?,
                                            1,
                                            1,
                                            1,
                                        ],
                                        &execution,
                                    )
                                    .map_err(invalid)?
                                } else {
                                    mask.clone()
                                };
                                let repeated_batch = *repeated
                                    .descriptor()
                                    .shape()
                                    .first()
                                    .ok_or_else(|| invalid("Missing mask batch"))?;
                                let available = repeated_batch.min(total);
                                let start = u64::try_from(start).map_err(invalid)?.min(available);
                                let amount = length.min(available - start);
                                let view = narrow_method_exact_native(
                                    &repeated,
                                    0,
                                    i64::try_from(start).map_err(invalid)?,
                                    amount,
                                    &context.cancellation,
                                )
                                .map_err(invalid)?;
                                cloned(&view, compute.backend(), &execution)?
                            };
                            Some(mask)
                        }
                        Some(_) => return Err(invalid("Tensor noise mask required")),
                    };
                    first
                        .replaced_samples_mask_and_indices_source_dictionary(
                            NativeLatentSamples::Tensor(output),
                            mask.map(NativeLatentNoiseMask::Tensor),
                            indices,
                            &execution,
                        )
                        .map_err(invalid)?
                } else {
                    first
                        .replaced_samples_source_dictionary(
                            NativeLatentSamples::Tensor(output),
                            &execution,
                        )
                        .map_err(invalid)?
                };
                NativeStoredPayload::Latent(Arc::new(bundle))
            };
            check(&context)?;
            let published = context
                .handle_store()
                .publish(payload, &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| outcome(published.clone()));
            if result.is_err() {
                context
                    .handle_store()
                    .revoke(&published, &sim_types::CancellationToken::default())
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

fn operation_node(name: &str) -> bool {
    matches!(
        name,
        "LatentOperationSharpen" | "LatentOperationTonemapReinhard"
    )
}
fn number(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<f64, NativeNodeFailure> {
    match input(inputs, name)? {
        NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        } => Ok(*value),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => Ok(*value as f64),
        _ => Err(invalid("Number required")),
    }
}
fn signed(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<i64, NativeNodeFailure> {
    match input(inputs, name)? {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => Ok(*value),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => i64::try_from(*value).map_err(invalid),
        _ => Err(invalid("Integer required")),
    }
}
fn batch_range(samples: &Tensor, index: i64, length: u64) -> Result<(i64, u64), NativeNodeFailure> {
    let batch = i64::try_from(
        *samples
            .descriptor()
            .shape()
            .first()
            .ok_or_else(|| invalid("Batch required"))?,
    )
    .map_err(invalid)?;
    if batch == 0 {
        return Err(invalid("Cannot slice empty batch"));
    }
    let index = if index < 0 {
        index.saturating_add(batch)
    } else {
        index
    };
    let index = index.clamp(0, batch - 1);
    Ok((
        index,
        length.min(u64::try_from(batch - index).map_err(invalid)?),
    ))
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
fn repeat_batch(
    input: &Tensor,
    batch: u64,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let source = *input
        .descriptor()
        .shape()
        .first()
        .ok_or_else(|| invalid("Batch required"))?;
    if source == 0 {
        return Err(invalid("Empty source batch"));
    }
    let repeated = if source < batch {
        let mut repeats = vec![1; input.descriptor().rank()];
        *repeats
            .first_mut()
            .ok_or_else(|| invalid("Rank required"))? =
            i64::try_from(batch.div_ceil(source)).map_err(invalid)?;
        tensor_repeat_with_context_exact_native(backend, input, &repeats, execution)
            .map_err(invalid)?
    } else {
        input.clone()
    };
    if source != batch {
        narrow_method_exact_native(&repeated, 0, 0, batch, execution.cancellation).map_err(invalid)
    } else {
        Ok(repeated)
    }
}
fn matching_second(
    first: &Tensor,
    second: &Tensor,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let target = first.descriptor().shape();
    let second = if second.descriptor().shape().get(1..) != target.get(1..) {
        resize_latent(second, target, backend, execution)?
    } else {
        second.clone()
    };
    repeat_batch(
        &second,
        *target.first().ok_or_else(|| invalid("Batch required"))?,
        backend,
        execution,
    )
}
fn scaled(
    input: &Tensor,
    scale: f64,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    real_multiply_with_context_exact_native(
        backend,
        input,
        ElementwiseOperand::Scalar(Scalar::Float(scale)),
        execution,
    )
    .map_err(invalid)
}
fn interpolate(
    first: &Tensor,
    second: &Tensor,
    ratio: f64,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let norm = |input: &Tensor| {
        vector_norm_with_context_exact_native(backend, input, 2.0, &[1], false, None, execution)
            .map_err(invalid)
    };
    let normalize = |input: &Tensor, magnitude: &Tensor| {
        let divided = div_with_context_exact_native(
            backend,
            input,
            ElementwiseOperand::Tensor(magnitude),
            execution,
        )
        .map_err(invalid)?;
        nan_to_num_with_context_exact_native(backend, &divided, None, None, None, execution)
            .map_err(invalid)
    };
    let first_magnitude = norm(first)?;
    let second_magnitude = norm(second)?;
    let first = scaled(
        &normalize(first, &first_magnitude)?,
        ratio,
        backend,
        execution,
    )?;
    let second = scaled(
        &normalize(second, &second_magnitude)?,
        1.0 - ratio,
        backend,
        execution,
    )?;
    let mixed = add_method_with_context_exact_native(
        backend,
        &first,
        ElementwiseOperand::Tensor(&second),
        1.0,
        execution,
    )
    .map_err(invalid)?;
    let normalized = normalize(&mixed, &norm(&mixed)?)?;
    let magnitude = add_method_with_context_exact_native(
        backend,
        &scaled(&first_magnitude, ratio, backend, execution)?,
        ElementwiseOperand::Tensor(&scaled(&second_magnitude, 1.0 - ratio, backend, execution)?),
        1.0,
        execution,
    )
    .map_err(invalid)?;
    real_multiply_with_context_exact_native(
        backend,
        &normalized,
        ElementwiseOperand::Tensor(&magnitude),
        execution,
    )
    .map_err(invalid)
}
fn spatial_size(shape: &[u64]) -> Result<(u64, u64), NativeNodeFailure> {
    let height = *shape
        .get(
            shape
                .len()
                .checked_sub(2)
                .ok_or_else(|| invalid("Spatial rank"))?,
        )
        .ok_or_else(|| invalid("Height required"))?;
    let width = *shape.last().ok_or_else(|| invalid("Width required"))?;
    if width == 0 || height == 0 {
        return Err(invalid("Empty spatial dimension"));
    }
    Ok((height, width))
}
fn reshape(
    input: &Tensor,
    shape: &[u64],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let shape = shape
        .iter()
        .map(|v| i64::try_from(*v).map_err(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    tensor_reshape_with_context_exact_native(backend, input, &shape, execution).map_err(invalid)
}
fn upscale(
    input: &Tensor,
    width: u64,
    height: u64,
    method: &str,
    crop: &str,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let shape = input.descriptor().shape();
    if !(4..=5).contains(&shape.len()) {
        return Err(invalid("Image or video LATENT required"));
    }
    let (batch, channels) = (
        *shape.first().ok_or_else(|| invalid("Batch"))?,
        *shape.get(1).ok_or_else(|| invalid("Channels"))?,
    );
    let (source_height, source_width) = spatial_size(shape)?;
    let input = if let Some(frames) = shape.get(2).filter(|_| shape.len() == 5) {
        let moved = torch_movedim_exact_native(input, &[2], &[1], execution.cancellation)
            .map_err(invalid)?;
        reshape(
            &moved,
            &[
                batch
                    .checked_mul(*frames)
                    .ok_or_else(|| invalid("Frame overflow"))?,
                channels,
                source_height,
                source_width,
            ],
            backend,
            execution,
        )?
    } else {
        input.clone()
    };
    let crop = match crop {
        "center" => ResizeCrop::Center,
        "disabled" => ResizeCrop::Disabled,
        _ => return Err(invalid("Unknown crop")),
    };
    let output = if method == "bislerp" {
        let input = if crop == ResizeCrop::Center {
            if width == 0 || height == 0 {
                return Err(invalid("Empty output"));
            }
            let old = source_width as f64 / source_height as f64;
            let new = width as f64 / height as f64;
            let x = if old > new {
                ((source_width as f64 - source_width as f64 * new / old) / 2.0).round_ties_even()
                    as u64
            } else {
                0
            };
            let y = if old < new {
                ((source_height as f64 - source_height as f64 * old / new) / 2.0).round_ties_even()
                    as u64
            } else {
                0
            };
            let rows = narrow_method_exact_native(
                &input,
                -2,
                i64::try_from(y).map_err(invalid)?,
                source_height
                    .checked_sub(y.checked_mul(2).ok_or_else(|| invalid("Crop overflow"))?)
                    .ok_or_else(|| invalid("Crop overflow"))?,
                execution.cancellation,
            )
            .map_err(invalid)?;
            narrow_method_exact_native(
                &rows,
                -1,
                i64::try_from(x).map_err(invalid)?,
                source_width
                    .checked_sub(x.checked_mul(2).ok_or_else(|| invalid("Crop overflow"))?)
                    .ok_or_else(|| invalid("Crop overflow"))?,
                execution.cancellation,
            )
            .map_err(invalid)?
        } else {
            input
        };
        bislerp_tensor_with_context_exact_native(backend, &input, width, height, execution)
            .map_err(invalid)?
    } else {
        let mode = match method {
            "nearest-exact" => ResizeMode::NearestExact,
            "bilinear" => ResizeMode::Bilinear,
            "area" => ResizeMode::Area,
            "bicubic" => ResizeMode::Bicubic,
            _ => return Err(invalid("Unknown resize mode")),
        };
        let mut target = input.descriptor().shape().to_vec();
        let len = target.len();
        *target.get_mut(len - 2).ok_or_else(|| invalid("Height"))? = height;
        *target.get_mut(len - 1).ok_or_else(|| invalid("Width"))? = width;
        let descriptor = TensorDescriptor::contiguous(
            target,
            input.descriptor().dtype(),
            input.descriptor().device(),
            input.descriptor().stream(),
        )
        .map_err(invalid)?;
        backend
            .resize(
                ResizeSpec {
                    width,
                    height,
                    mode,
                    crop,
                    antialias: false,
                    align_corners: false,
                },
                &input,
                descriptor,
                execution,
            )
            .map_err(invalid)?
            .0
    };
    if shape.len() == 5 {
        let frames = *shape.get(2).ok_or_else(|| invalid("Frames"))?;
        let output = reshape(
            &output,
            &[batch, frames, channels, height, width],
            backend,
            execution,
        )?;
        let output = torch_movedim_exact_native(&output, &[1], &[2], execution.cancellation)
            .map_err(invalid)?;
        // The canonical LATENT payload owner requires contiguous tensor storage.
        cloned(&output, backend, execution)
    } else {
        Ok(output)
    }
}
