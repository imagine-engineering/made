use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_tensor::{
    ImageTensor, Layout, MemoryFormatReference, NativeTensorPayload, NativeTensorRole,
    NumpyRandomState,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["ShuffleImageTextDataset", "SplitImageToTileList"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const SHUFFLE_FEATURE_ID: &str = "COMFY-NODE-0621";
const SHUFFLE_CLASS_TYPE: &str = "ShuffleImageTextDataset";
const SHUFFLE_IMPLEMENTATION_VERSION: &str = "source-3b27465f-v2";
const SHUFFLE_CACHE_TOKEN: &str = "shuffle-image-text-dataset-source-3b27465f-v2";
const SHUFFLE_RNG_PHASE: &str = "training-and-data-order";
const SPLIT_FEATURE_ID: &str = "COMFY-NODE-0631";
const SPLIT_CLASS_TYPE: &str = "SplitImageToTileList";
const SPLIT_IMPLEMENTATION_VERSION: &str = "source-a57638bf-v1";
const SPLIT_CACHE_TOKEN: &str = "split-image-to-tile-list-source-a57638bf-v1";
const MAX_RESOLUTION: u64 = 16_384;
const MAX_LIST_VALUES: usize = 1_000_000;

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let image_type = image_type()?;
    let shuffle_source = built_in_source_schema(SHUFFLE_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["images".to_owned(), "texts".to_owned(), "seed".to_owned()],
            &[],
            &["images".to_owned(), "texts".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let split_source = built_in_source_schema(SPLIT_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &[
                "image".to_owned(),
                "tile_width".to_owned(),
                "tile_height".to_owned(),
                "overlap".to_owned(),
            ],
            &[],
            &["tiles".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;

    Ok(vec![
        NativeNodeBinding::Executable {
            feature_id: SHUFFLE_FEATURE_ID.to_owned(),
            descriptor: NativeNodeDescriptor {
                schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                class_type: SHUFFLE_CLASS_TYPE.to_owned(),
                implementation_version: SHUFFLE_IMPLEMENTATION_VERSION.to_owned(),
                source_schema: Some(shuffle_source),
                inputs: vec![
                    input(
                        "images",
                        NativeValueType::Handle(image_type.clone()),
                        NativePortCardinality::List,
                        false,
                    )?,
                    input(
                        "texts",
                        NativeValueType::Primitive(NativePrimitiveType::String),
                        NativePortCardinality::List,
                        false,
                    )?,
                    input(
                        "seed",
                        NativeValueType::Primitive(NativePrimitiveType::Integer),
                        NativePortCardinality::List,
                        true,
                    )?,
                ],
                dynamic_inputs: Vec::new(),
                outputs: vec![
                    NativeOutputDescriptor {
                        name: "images".to_owned(),
                        produced_type: NativeValueType::Handle(image_type.clone()),
                        is_list: true,
                    },
                    NativeOutputDescriptor {
                        name: "texts".to_owned(),
                        produced_type: NativeValueType::Primitive(NativePrimitiveType::String),
                        is_list: true,
                    },
                ],
                output_node: false,
                effect: NativeEffectClass::Pure,
                cache: NativeCachePolicy::InputIdentity,
            },
            presentation: NativeNodePresentation {
                display_name: "Shuffle Pairs of Image-Text".to_owned(),
                category: "image/batch".to_owned(),
                description: "Randomly shuffle the order of pairs of image-text in a list."
                    .to_owned(),
                output_names: vec!["images".to_owned(), "texts".to_owned()],
                search_aliases: vec![
                    "shuffle".to_owned(),
                    "randomize".to_owned(),
                    "mix".to_owned(),
                ],
                is_deprecated: false,
                is_experimental: true,
            },
            node: Arc::new(ShuffleImageTextDataset),
        },
        NativeNodeBinding::Executable {
            feature_id: SPLIT_FEATURE_ID.to_owned(),
            descriptor: NativeNodeDescriptor {
                schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                class_type: SPLIT_CLASS_TYPE.to_owned(),
                implementation_version: SPLIT_IMPLEMENTATION_VERSION.to_owned(),
                source_schema: Some(split_source),
                inputs: vec![
                    input(
                        "image",
                        NativeValueType::Handle(image_type.clone()),
                        NativePortCardinality::Scalar,
                        false,
                    )?,
                    input(
                        "tile_width",
                        NativeValueType::Primitive(NativePrimitiveType::Integer),
                        NativePortCardinality::Scalar,
                        true,
                    )?,
                    input(
                        "tile_height",
                        NativeValueType::Primitive(NativePrimitiveType::Integer),
                        NativePortCardinality::Scalar,
                        true,
                    )?,
                    input(
                        "overlap",
                        NativeValueType::Primitive(NativePrimitiveType::Integer),
                        NativePortCardinality::Scalar,
                        true,
                    )?,
                ],
                dynamic_inputs: Vec::new(),
                outputs: vec![NativeOutputDescriptor {
                    name: "tiles".to_owned(),
                    produced_type: NativeValueType::Handle(image_type),
                    is_list: true,
                }],
                output_node: false,
                effect: NativeEffectClass::Pure,
                cache: NativeCachePolicy::InputIdentity,
            },
            presentation: NativeNodePresentation {
                display_name: "Split Image into List of Tiles".to_owned(),
                category: "image/batch".to_owned(),
                description:
                    "Splits an image into a batched list of tiles with a specified overlap."
                        .to_owned(),
                output_names: vec!["tiles".to_owned()],
                search_aliases: vec![
                    "split image".to_owned(),
                    "tile image".to_owned(),
                    "slice image".to_owned(),
                ],
                is_deprecated: false,
                is_experimental: false,
            },
            node: Arc::new(SplitImageToTileList),
        },
    ])
}

fn input(
    name: &str,
    value_type: NativeValueType,
    cardinality: NativePortCardinality,
    allows_literal: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required: true,
        hidden: false,
        lazy: false,
        cardinality,
        allows_literal,
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct ShuffleImageTextDataset;

impl NativeNode for ShuffleImageTextDataset {
    fn class_type(&self) -> &str {
        SHUFFLE_CLASS_TYPE
    }
    fn implementation_version(&self) -> &str {
        SHUFFLE_IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        shuffle_inputs(inputs)?;
        Ok(SHUFFLE_CACHE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, SHUFFLE_CLASS_TYPE)?;
        shuffle_inputs(inputs)?;
        Ok(NativeCacheDependencies {
            rng_phase: Some(SHUFFLE_RNG_PHASE.to_owned()),
            ..Default::default()
        })
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, SHUFFLE_CLASS_TYPE)?;
            let (images, texts, seed) = shuffle_inputs(&inputs)?;
            let expected_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            for handle in &images {
                let payload = context
                    .handle_store()
                    .resolve(handle, &expected_type, &context.cancellation)
                    .map_err(|error| handle_failure(error, SHUFFLE_CLASS_TYPE))?;
                require_image_payload(&payload)?;
            }
            let permutation = numpy_permutation(images.len(), seed, &context.cancellation)?;
            let mut shuffled_images = Vec::with_capacity(images.len());
            let mut shuffled_texts = Vec::with_capacity(images.len());
            for index in permutation {
                check_cancellation(&context, SHUFFLE_CLASS_TYPE)?;
                let handle = images
                    .get(index)
                    .ok_or_else(|| native_failure("permutation index exceeded the image list"))?;
                let text = texts.get(index).ok_or_else(|| {
                    invalid_inputs("texts must contain at least one entry for every image")
                })?;
                shuffled_images.push(NativeValue::Handle {
                    value: handle.clone(),
                });
                shuffled_texts.push(NativeValue::Primitive {
                    value: NativePrimitive::String(text.clone()),
                });
            }
            values_outcome(vec![
                NativeValue::List {
                    values: shuffled_images,
                },
                NativeValue::List {
                    values: shuffled_texts,
                },
            ])
        })
    }
}

#[derive(Debug)]
struct SplitImageToTileList;

impl NativeNode for SplitImageToTileList {
    fn class_type(&self) -> &str {
        SPLIT_CLASS_TYPE
    }
    fn implementation_version(&self) -> &str {
        SPLIT_IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        split_inputs(inputs)?;
        Ok(SPLIT_CACHE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, SPLIT_CLASS_TYPE)?;
        split_inputs(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, SPLIT_CLASS_TYPE)?;
            let (handle, tile_width, tile_height, overlap) = split_inputs(&inputs)?;
            let expected_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved = context
                .handle_store()
                .resolve(&handle, &expected_type, &context.cancellation)
                .map_err(|error| handle_failure(error, SPLIT_CLASS_TYPE))?;
            let image = require_image_payload(&resolved)?;
            let (_, height, width, _) = image.dimensions().map_err(native_failure)?;
            let coordinates = tile_coordinates(width, height, tile_width, tile_height, overlap)?;
            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute
                .execution_context(&context)
                .map_err(compute_failure)?;
            let mut payloads = Vec::with_capacity(coordinates.len());
            for (x_start, x_end, y_start, y_end) in coordinates {
                check_cancellation(&context, SPLIT_CLASS_TYPE)?;
                let height_view = narrow_method_exact_native(
                    image.tensor(),
                    1,
                    i64::try_from(y_start).map_err(native_failure)?,
                    y_end - y_start,
                    &context.cancellation,
                )
                .map_err(native_failure)?;
                let tile_view = narrow_method_exact_native(
                    &height_view,
                    2,
                    i64::try_from(x_start).map_err(native_failure)?,
                    x_end - x_start,
                    &context.cancellation,
                )
                .map_err(native_failure)?;
                let tensor = contiguous_with_context_exact_native(
                    compute.backend(),
                    &tile_view,
                    MemoryFormatReference::Layout(Layout::Contiguous),
                    &execution,
                )
                .map_err(native_failure)?;
                let image = ImageTensor::from_tensor(tensor).map_err(native_failure)?;
                let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                    .map_err(native_failure)?;
                payloads.push(NativeStoredPayload::Tensor(Arc::new(payload)));
            }
            drop(resolved);
            let mut published = Vec::with_capacity(payloads.len());
            for payload in payloads {
                if let Err(failure) = check_cancellation(&context, SPLIT_CLASS_TYPE) {
                    rollback_published(&context, &published)?;
                    return Err(failure);
                }
                match context
                    .handle_store()
                    .publish(payload, &context.cancellation)
                {
                    Ok(handle) => published.push(handle),
                    Err(error) => {
                        rollback_published(&context, &published)?;
                        return Err(handle_failure(error, SPLIT_CLASS_TYPE));
                    }
                }
            }
            if let Err(failure) = check_cancellation(&context, SPLIT_CLASS_TYPE) {
                rollback_published(&context, &published)?;
                return Err(failure);
            }
            values_outcome(vec![NativeValue::List {
                values: published
                    .into_iter()
                    .map(|value| NativeValue::Handle { value })
                    .collect(),
            }])
        })
    }
}

fn shuffle_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(Vec<NativeOpaqueHandle>, Vec<String>, u64), NativeNodeFailure> {
    if inputs.len() != 3 {
        return Err(invalid_inputs(
            "ShuffleImageTextDataset requires images, texts, and seed",
        ));
    }
    let images = handle_list(inputs.get("images"), "images")?;
    let texts = string_list(inputs.get("texts"), "texts")?;
    let seeds = integer_list(inputs.get("seed"), "seed")?;
    let seed = seeds
        .first()
        .copied()
        .ok_or_else(|| invalid_inputs("seed list must not be empty"))?;
    if texts.len() < images.len() {
        return Err(invalid_inputs(
            "texts must contain at least one entry for every image",
        ));
    }
    Ok((images, texts, seed))
}

fn numpy_permutation(
    length: usize,
    seed: u64,
    cancellation: &CancellationToken,
) -> Result<Vec<usize>, NativeNodeFailure> {
    let mut permutation = (0..length).collect::<Vec<_>>();
    let mut random_state = NumpyRandomState::from_seed(seed);
    for upper in (1..permutation.len()).rev() {
        cancellation
            .check()
            .map_err(|_| interrupted_failure(SHUFFLE_CLASS_TYPE))?;
        let high_exclusive = u32::try_from(upper + 1)
            .map_err(|_| invalid_inputs("image list exceeds NumPy's permutation range"))?;
        let selected = random_state
            .randint(0, high_exclusive, cancellation)
            .map_err(native_failure)?;
        permutation.swap(upper, usize::try_from(selected).map_err(native_failure)?);
    }
    Ok(permutation)
}

fn split_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(NativeOpaqueHandle, u64, u64, u64), NativeNodeFailure> {
    if inputs.len() != 4 {
        return Err(invalid_inputs(
            "SplitImageToTileList requires image, tile_width, tile_height, and overlap",
        ));
    }
    let Some(NativeValue::Handle { value: image }) = inputs.get("image") else {
        return Err(invalid_inputs("image must be an IMAGE handle"));
    };
    if image.handle_type().kind != NativeHandleKind::Image || image.handle_type().type_id != "IMAGE"
    {
        return Err(invalid_inputs("image must be an exact IMAGE handle"));
    }
    let tile_width = bounded_integer(inputs.get("tile_width"), "tile_width", 64, MAX_RESOLUTION)?;
    let tile_height =
        bounded_integer(inputs.get("tile_height"), "tile_height", 64, MAX_RESOLUTION)?;
    let overlap = bounded_integer(inputs.get("overlap"), "overlap", 0, 4_096)?;
    Ok((image.clone(), tile_width, tile_height, overlap))
}

fn handle_list(
    value: Option<&NativeValue>,
    name: &str,
) -> Result<Vec<NativeOpaqueHandle>, NativeNodeFailure> {
    let Some(NativeValue::List { values }) = value else {
        return Err(invalid_inputs(format!("{name} must be a list")));
    };
    if values.len() > MAX_LIST_VALUES {
        return Err(invalid_inputs(format!("{name} exceeds the native list limit")));
    }
    values
        .iter()
        .map(|value| match value {
            NativeValue::Handle { value }
                if value.handle_type().kind == NativeHandleKind::Image
                    && value.handle_type().type_id == "IMAGE" =>
            {
                Ok(value.clone())
            }
            _ => Err(invalid_inputs(format!(
                "{name} must contain only exact IMAGE handles"
            ))),
        })
        .collect()
}

fn string_list(value: Option<&NativeValue>, name: &str) -> Result<Vec<String>, NativeNodeFailure> {
    let Some(NativeValue::List { values }) = value else {
        return Err(invalid_inputs(format!("{name} must be a list")));
    };
    if values.len() > MAX_LIST_VALUES {
        return Err(invalid_inputs(format!("{name} exceeds the native list limit")));
    }
    values
        .iter()
        .map(|value| match value {
            NativeValue::Primitive {
                value: NativePrimitive::String(value),
            } => Ok(value.clone()),
            _ => Err(invalid_inputs(format!("{name} must contain only strings"))),
        })
        .collect()
}

fn integer_list(value: Option<&NativeValue>, name: &str) -> Result<Vec<u64>, NativeNodeFailure> {
    let Some(NativeValue::List { values }) = value else {
        return Err(invalid_inputs(format!("{name} must be a list")));
    };
    if values.len() > MAX_LIST_VALUES {
        return Err(invalid_inputs(format!("{name} exceeds the native list limit")));
    }
    values
        .iter()
        .map(|value| primitive_u64(value, name))
        .collect()
}

fn primitive_u64(value: &NativeValue, name: &str) -> Result<u64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => u64::try_from(*value)
            .map_err(|_| invalid_inputs(format!("{name} must be non-negative"))),
        _ => Err(invalid_inputs(format!("{name} must be an integer"))),
    }
}

fn bounded_integer(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = value.ok_or_else(|| invalid_inputs(format!("missing {name}")))?;
    let value = primitive_u64(value, name)?;
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn require_image_payload(payload: &NativeStoredPayload) -> Result<&ImageTensor, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = payload else {
        return Err(native_failure(
            "IMAGE handle did not resolve to a tensor payload",
        ));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(native_failure(
            "IMAGE handle resolved to the wrong tensor role",
        ));
    }
    payload
        .image()
        .ok_or_else(|| native_failure("IMAGE handle did not resolve to canonical image storage"))
}

fn tile_coordinates(
    width: u64,
    height: u64,
    tile_width: u64,
    tile_height: u64,
    overlap: u64,
) -> Result<Vec<(u64, u64, u64, u64)>, NativeNodeFailure> {
    let stride_x = rounded_max_quarter(tile_width, tile_width.saturating_sub(overlap));
    let stride_y = rounded_max_quarter(tile_height, tile_height.saturating_sub(overlap));
    let x_coordinates = axis_coordinates(width, tile_width, stride_x)?;
    let y_coordinates = axis_coordinates(height, tile_height, stride_y)?;
    let count = x_coordinates
        .len()
        .checked_mul(y_coordinates.len())
        .ok_or_else(|| invalid_inputs("tile count overflowed"))?;
    if count > MAX_LIST_VALUES {
        return Err(invalid_inputs("tile count exceeds the native list limit"));
    }
    let mut coordinates = Vec::with_capacity(count);
    for &(y_start, y_end) in &y_coordinates {
        for &(x_start, x_end) in &x_coordinates {
            coordinates.push((x_start, x_end, y_start, y_end));
        }
    }
    Ok(coordinates)
}

fn rounded_max_quarter(tile: u64, other: u64) -> u64 {
    let rounded_quarter = match (tile / 4, tile % 4) {
        (quotient, 0 | 1) => quotient,
        (quotient, 2) if quotient % 2 == 0 => quotient,
        (quotient, _) => quotient + 1,
    };
    rounded_quarter.max(other)
}

fn axis_coordinates(
    size: u64,
    tile: u64,
    stride: u64,
) -> Result<Vec<(u64, u64)>, NativeNodeFailure> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if stride == 0 {
        return Err(invalid_inputs("tile stride must be non-zero"));
    }
    let mut coordinates = Vec::new();
    let mut start = 0_u64;
    loop {
        let end = start.saturating_add(tile).min(size);
        let adjusted_start = end.saturating_sub(tile);
        coordinates.push((adjusted_start, end));
        if end >= size {
            break;
        }
        if coordinates.len() >= MAX_LIST_VALUES {
            return Err(invalid_inputs("tile count exceeds the native list limit"));
        }
        start = start
            .checked_add(stride)
            .ok_or_else(|| invalid_inputs("tile coordinate overflowed"))?;
    }
    Ok(coordinates)
}

fn rollback_published(
    context: &NativeNodeContext,
    published: &[NativeOpaqueHandle],
) -> Result<(), NativeNodeFailure> {
    let cleanup = CancellationToken::default();
    for handle in published.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &cleanup)
            .map_err(|error| NativeNodeFailure {
                code: "native_image_batch_rollback_failed".to_owned(),
                message: format!("failed to revoke a partially published image tile: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
    }
    Ok(())
}

fn values_outcome(outputs: Vec<NativeValue>) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs,
        ui: None,
        effects: Vec::new(),
    };
    outcome
        .validate()
        .map_err(|error| invalid_inputs(error.to_string()))?;
    Ok(outcome)
}

fn check_cancellation(
    context: &NativeNodeContext,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(class_type))
}

fn handle_failure(error: NativeHandleStoreError, class_type: &str) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(class_type)
    } else {
        NativeNodeFailure {
            code: "invalid_image_handle".to_owned(),
            message: format!("{class_type} IMAGE handle is not available: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: NativeNodeContractError) -> NativeNodeFailure {
    native_failure(error)
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn native_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("cancelled") {
        return NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message,
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        };
    }
    NativeNodeFailure {
        code: "native_image_batch_failed".to_owned(),
        message,
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(class_type: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{class_type} execution was interrupted"),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession,
        NativeNodeServiceIdentity, NativeNodeServices, NativeResolvedPayload,
        NativeResolvedPayloadRetention,
    };
    use sim_tensor::{CpuBackend, CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, NodeId, PromptId};
    use serde_json::Value;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-batch-comfy-node-0621/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_generation: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(attempt_id: AttemptId) -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x39201),
                    Uuid::from_u128(0x39202),
                )?,
                attempt_id,
                next_generation: AtomicU64::new(1),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values
                .lock()
                .map(|values| values.len())
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))
        }
    }

    impl NativeHandleStore for TestStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }

        fn attempt_id(&self) -> AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            handle: &NativeOpaqueHandle,
            expected_type: &NativeHandleType,
            cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            if handle.store_identity() != self.identity {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            let values = self.values.lock().map_err(|_| {
                NativeHandleStoreError::Rejected("test store is poisoned".to_owned())
            })?;
            let payload = values
                .get(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if payload.digest_sha256() != handle.digest_sha256().unwrap_or_default() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            Ok(NativeResolvedPayload::checked(
                payload.clone(),
                Arc::new(TestRetention),
            )?)
        }

        fn publish(
            &self,
            payload: NativeStoredPayload,
            cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            payload.validate()?;
            let handle_type = payload.handle_type()?;
            let digest = payload.digest_sha256();
            let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
            let identifier = format!("image-{generation}");
            let handle = NativeOpaqueHandle::new(
                handle_type,
                self.identity,
                identifier.clone(),
                generation,
                Some(digest),
            )?;
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .insert(identifier, Arc::new(payload));
            Ok(handle)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            let removed = self
                .values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .remove(handle.identifier());
            if removed.is_none() {
                return Err(NativeHandleStoreError::Missing(
                    handle.identifier().to_owned(),
                ));
            }
            Ok(())
        }
    }

    struct TestExecution {
        backend: Arc<CpuBackend>,
        store: Arc<TestStore>,
        context: NativeNodeContext,
    }

    fn test_execution(
        node_id: &str,
        cancellation: CancellationToken,
    ) -> Result<TestExecution, Box<dyn std::error::Error>> {
        let attempt_id = AttemptId(Uuid::from_u128(0x39203));
        let node_id = NodeId(node_id.to_owned());
        let store = TestStore::new(attempt_id)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(16 * 1024 * 1024)?;
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x39204),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let services = NativeNodeServices::checked(None, None, Some(compute))?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x39205)),
            attempt_id,
            node_id,
            cancellation,
            scratch,
            store.clone(),
            services,
        )?;
        Ok(TestExecution {
            backend,
            store,
            context,
        })
    }

    fn publish_image(
        execution: &TestExecution,
        height: u64,
        width: u64,
        values: Vec<f32>,
    ) -> Result<NativeOpaqueHandle, Box<dyn std::error::Error>> {
        let compute = execution.context.compute_session()?;
        let tensor_context = compute.execution_context(&execution.context)?;
        let image = ImageTensor::from_f32(
            &execution.backend,
            &tensor_context,
            1,
            height,
            width,
            1,
            &values,
        )?;
        let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
        Ok(execution.store.publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &CancellationToken::default(),
        )?)
    }

    fn node(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn std::error::Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == class_type => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{class_type} binding is absent").into())
    }

    #[test]
    fn descriptors_preserve_source_contracts() -> Result<(), Box<dyn std::error::Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 2);
        for binding in bindings {
            let NativeNodeBinding::Executable { descriptor, .. } = binding else {
                return Err("binding was not executable".into());
            };
            descriptor.validate_exact_schema_v2()?;
        }
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture["task_id"],
            "comfy-parity-native-nodes-image-batch-comfy-node-0621"
        );
        assert_eq!(
            fixture["nodes"][0]["source_file_sha256"],
            "3b27465fec391509083bd1837895c09abc489c04d81afae5ffe631abd6a4e772"
        );
        Ok(())
    }

    #[test]
    fn shuffle_matches_numpy_random_state_permutation_oracles()
    -> Result<(), NativeNodeFailure> {
        let cancellation = CancellationToken::default();
        assert_eq!(
            numpy_permutation(10, 7, &cancellation)?,
            vec![8, 5, 0, 2, 1, 9, 7, 3, 6, 4]
        );
        assert_eq!(
            numpy_permutation(5, 42, &cancellation)?,
            vec![1, 4, 2, 0, 3]
        );
        Ok(())
    }

    #[test]
    fn source_grid_is_row_major_and_uses_bankers_rounding() -> Result<(), NativeNodeFailure> {
        assert_eq!(rounded_max_quarter(66, 0), 16);
        assert_eq!(rounded_max_quarter(70, 0), 18);
        assert_eq!(
            tile_coordinates(160, 96, 64, 64, 16)?,
            vec![
                (0, 64, 0, 64),
                (48, 112, 0, 64),
                (96, 160, 0, 64),
                (0, 64, 32, 96),
                (48, 112, 32, 96),
                (96, 160, 32, 96)
            ]
        );
        assert_eq!(tile_coordinates(32, 48, 64, 64, 0)?, vec![(0, 32, 0, 48)]);
        Ok(())
    }

    #[test]
    fn list_and_bounds_validation_match_source_edges() {
        let empty = NativeValue::List { values: Vec::new() };
        assert!(integer_list(Some(&empty), "seed").is_ok());
        assert!(
            bounded_integer(
                Some(&NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(63)
                }),
                "tile_width",
                64,
                MAX_RESOLUTION
            )
            .is_err()
        );
        assert!(
            bounded_integer(
                Some(&NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(4_097)
                }),
                "overlap",
                0,
                4_096
            )
            .is_err()
        );
    }

    #[test]
    fn shuffle_execution_is_deterministic_and_preserves_pairs()
    -> Result<(), Box<dyn std::error::Error>> {
        let execution = test_execution("shuffle-test", CancellationToken::default())?;
        let handles = (0..5)
            .map(|value| publish_image(&execution, 1, 1, vec![value as f32]))
            .collect::<Result<Vec<_>, _>>()?;
        let texts = (0..5)
            .map(|value| format!("text-{value}"))
            .collect::<Vec<_>>();
        let inputs = BTreeMap::from([
            (
                "images".to_owned(),
                NativeValue::List {
                    values: handles
                        .iter()
                        .cloned()
                        .map(|value| NativeValue::Handle { value })
                        .collect(),
                },
            ),
            (
                "texts".to_owned(),
                NativeValue::List {
                    values: texts
                        .iter()
                        .cloned()
                        .map(|value| NativeValue::Primitive {
                            value: NativePrimitive::String(value),
                        })
                        .collect(),
                },
            ),
            (
                "seed".to_owned(),
                NativeValue::List {
                    values: vec![NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(42),
                    }],
                },
            ),
        ]);
        let shuffle = node(SHUFFLE_CLASS_TYPE)?;
        let first = futures::executor::block_on(
            shuffle.execute(execution.context.clone(), inputs.clone()),
        )?;
        let second = futures::executor::block_on(shuffle.execute(execution.context, inputs))?;
        assert_eq!(first, second);
        let NativeNodeOutcome::Values { outputs, .. } = first else {
            return Err("shuffle did not return values".into());
        };
        let (
            NativeValue::List {
                values: shuffled_images,
            },
            NativeValue::List {
                values: shuffled_texts,
            },
        ) = (&outputs[0], &outputs[1])
        else {
            return Err("shuffle output cardinality changed".into());
        };
        let mut actual_permutation = Vec::with_capacity(shuffled_images.len());
        for (image, text) in shuffled_images.iter().zip(shuffled_texts) {
            let NativeValue::Handle { value: image } = image else {
                return Err("shuffle image output is not a handle".into());
            };
            let NativeValue::Primitive {
                value: NativePrimitive::String(text),
            } = text
            else {
                return Err("shuffle text output is not a string".into());
            };
            let index = handles
                .iter()
                .position(|candidate| candidate == image)
                .ok_or("shuffle returned an unknown image")?;
            actual_permutation.push(index);
            assert_eq!(text, &texts[index]);
        }
        assert_eq!(actual_permutation, vec![1, 4, 2, 0, 3]);
        assert_eq!(execution.store.count()?, 5);
        Ok(())
    }

    #[test]
    fn split_execution_materializes_source_order_and_cancellation_is_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let execution = test_execution("split-test", CancellationToken::default())?;
        let values = (0..96 * 160).map(|value| value as f32).collect::<Vec<_>>();
        let image = publish_image(&execution, 96, 160, values)?;
        let inputs = BTreeMap::from([
            ("image".to_owned(), NativeValue::Handle { value: image }),
            (
                "tile_width".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(64),
                },
            ),
            (
                "tile_height".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(64),
                },
            ),
            (
                "overlap".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(16),
                },
            ),
        ]);
        let outcome = futures::executor::block_on(
            node(SPLIT_CLASS_TYPE)?.execute(execution.context, inputs),
        )?;
        let NativeNodeOutcome::Values { outputs, .. } = outcome else {
            return Err("split did not return values".into());
        };
        let Some(NativeValue::List { values: tiles }) = outputs.first() else {
            return Err("split output cardinality changed".into());
        };
        assert_eq!(tiles.len(), 6);
        for (tile, expected_first) in tiles
            .iter()
            .zip([0.0_f32, 48.0, 96.0, 5_120.0, 5_168.0, 5_216.0])
        {
            let NativeValue::Handle { value: tile } = tile else {
                return Err("split tile is not an IMAGE handle".into());
            };
            let payload =
                execution
                    .store
                    .resolve(tile, &image_type()?, &CancellationToken::default())?;
            let image = require_image_payload(&payload)?;
            assert_eq!(image.dimensions()?, (1, 64, 64, 1));
            assert_eq!(image.as_f32_slice()?.first(), Some(&expected_first));
        }
        assert_eq!(execution.store.count()?, 7);

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let cancelled = test_execution("split-cancel-test", cancellation)?;
        let before = cancelled.store.count()?;
        let failure = futures::executor::block_on(
            node(SPLIT_CLASS_TYPE)?.execute(cancelled.context, BTreeMap::new()),
        )
        .expect_err("cancelled split must fail before parsing inputs");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(cancelled.store.count()?, before);
        Ok(())
    }
}
