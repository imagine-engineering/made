use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeHandleKind, NativeHandleStoreError,
    NativeHandleType, NativeInputDescriptor, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativePrimitiveType, NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use sim_tensor::{
    ImageTensor, Layout, MemoryFormatReference, NativeTensorPayload, NativeTensorRole, ResizeCrop,
    NumpyRandomState, ResizeMode,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "BatchImagesNode",
    "ImageBatch",
    "ImageDeduplication",
    "ImageFromBatch",
    "ImageGrid",
    "ImageMergeTileList",
    "MergeImageLists",
    "RebatchImages",
    "RepeatImageBatch",
    "ShuffleDataset",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const MAX_AUTOGROW_INPUTS: u32 = 50;
const MAX_LIST_VALUES: usize = 1_000_000;
const MAX_RESOLUTION: u64 = 16_384;
const SHUFFLE_RNG_PHASE: &str = "training-and-data-order";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchKind {
    BatchImages,
    ImageBatch,
    Deduplicate,
    FromBatch,
    Grid,
    MergeTiles,
    MergeLists,
    Rebatch,
    Repeat,
    Shuffle,
}

impl BatchKind {
    const fn feature_id(self) -> &'static str {
        match self {
            Self::BatchImages => "COMFY-NODE-0017",
            Self::ImageBatch => "COMFY-NODE-0241",
            Self::Deduplicate => "COMFY-NODE-0249",
            Self::FromBatch => "COMFY-NODE-0251",
            Self::Grid => "COMFY-NODE-0252",
            Self::MergeTiles => "COMFY-NODE-0255",
            Self::MergeLists => "COMFY-NODE-0405",
            Self::Rebatch => "COMFY-NODE-0506",
            Self::Repeat => "COMFY-NODE-0535",
            Self::Shuffle => "COMFY-NODE-0620",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::BatchImages => "BatchImagesNode",
            Self::ImageBatch => "ImageBatch",
            Self::Deduplicate => "ImageDeduplication",
            Self::FromBatch => "ImageFromBatch",
            Self::Grid => "ImageGrid",
            Self::MergeTiles => "ImageMergeTileList",
            Self::MergeLists => "MergeImageLists",
            Self::Rebatch => "RebatchImages",
            Self::Repeat => "RepeatImageBatch",
            Self::Shuffle => "ShuffleDataset",
        }
    }

    const fn source_version(self) -> &'static str {
        match self {
            Self::BatchImages => "source-96ec39e8-v1",
            Self::ImageBatch => "source-b8dfdde1-v1",
            Self::Deduplicate | Self::Grid | Self::MergeLists | Self::Shuffle => {
                "source-3b27465f-v1"
            }
            Self::FromBatch | Self::MergeTiles | Self::Repeat => "source-a57638bf-v1",
            Self::Rebatch => "source-2ebbc41c-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::BatchImages => "Batch Images",
            Self::ImageBatch => "Batch Images (DEPRECATED)",
            Self::Deduplicate => "Deduplicate Images",
            Self::FromBatch => "Get Image from Batch",
            Self::Grid => "Make Image Grid",
            Self::MergeTiles => "Merge List of Tiles to Image",
            Self::MergeLists => "Merge Image Lists (DEPRECATED)",
            Self::Rebatch => "Rebatch Images",
            Self::Repeat => "Repeat Image Batch",
            Self::Shuffle => "Shuffle Images List",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Deduplicate => "Remove duplicate or very similar images from a list.",
            Self::Grid => "Arrange multiple images into a grid layout.",
            Self::MergeLists => "Concatenate multiple image lists into one.",
            Self::Shuffle => "Randomly shuffle the order of images in a list.",
            _ => "",
        }
    }

    fn search_aliases(self) -> Vec<String> {
        let aliases: &[&str] = match self {
            Self::BatchImages => &[
                "batch",
                "image batch",
                "batch images",
                "combine images",
                "merge images",
                "stack images",
            ],
            Self::ImageBatch => &["combine images", "merge images", "stack images"],
            Self::Deduplicate => &["deduplicate", "remove duplicates", "similarity filter"],
            Self::FromBatch => &["select image", "pick from batch", "extract image"],
            Self::Grid => &["grid", "collage", "combine"],
            Self::MergeTiles => &["split image", "tile image", "slice image"],
            Self::MergeLists => &["list", "merge list", "make list"],
            Self::Rebatch => &[],
            Self::Repeat => &["duplicate image", "clone image"],
            Self::Shuffle => &["shuffle", "randomize", "mix"],
        };
        aliases.iter().map(|alias| (*alias).to_owned()).collect()
    }

    const fn output_is_list(self) -> bool {
        matches!(
            self,
            Self::Deduplicate | Self::MergeLists | Self::Rebatch | Self::Shuffle
        )
    }

    const fn is_experimental(self) -> bool {
        matches!(
            self,
            Self::Deduplicate | Self::Grid | Self::MergeLists | Self::Shuffle
        )
    }

    fn cache_token(self) -> String {
        format!(
            "{}-{}-input-identity",
            self.class_type().to_ascii_lowercase(),
            self.source_version()
        )
    }
}

const ALL_KINDS: [BatchKind; 10] = [
    BatchKind::BatchImages,
    BatchKind::ImageBatch,
    BatchKind::Deduplicate,
    BatchKind::FromBatch,
    BatchKind::Grid,
    BatchKind::MergeTiles,
    BatchKind::MergeLists,
    BatchKind::Rebatch,
    BatchKind::Repeat,
    BatchKind::Shuffle,
];

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    ALL_KINDS.into_iter().map(native_node_binding).collect()
}

fn native_node_binding(kind: BatchKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let image_type = image_type()?;
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let input_names = input_names(kind);
    let output_name = if kind.output_is_list() {
        "images"
    } else {
        "image"
    };
    let source_schema = catalog_schema
        .bind_execution_ports(&input_names, &dynamic_schema, &[output_name.to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = input_descriptors(kind, &image_type)?;
    let dynamic_inputs = if kind == BatchKind::BatchImages {
        vec![NativeDynamicInputDescriptor {
            name_template: "image{index}".to_owned(),
            start_index: 1,
            minimum_count: 1,
            maximum_count: MAX_AUTOGROW_INPUTS,
            input: handle_input(
                "image",
                image_type.clone(),
                NativePortCardinality::Scalar,
            )?,
        }]
    } else {
        Vec::new()
    };
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: kind.source_version().to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs: vec![NativeOutputDescriptor {
                name: output_name.to_owned(),
                produced_type: NativeValueType::Handle(image_type),
                is_list: kind.output_is_list(),
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: "image/batch".to_owned(),
            description: kind.description().to_owned(),
            output_names: vec![output_name.to_owned()],
            search_aliases: kind.search_aliases(),
            is_deprecated: kind == BatchKind::ImageBatch,
            is_experimental: kind.is_experimental(),
        },
        node: Arc::new(ImageBatchFamilyNode { kind }),
    })
}

fn input_names(kind: BatchKind) -> Vec<String> {
    let names: &[&str] = match kind {
        BatchKind::BatchImages => &[],
        BatchKind::ImageBatch => &["image1", "image2"],
        BatchKind::Deduplicate => &["images", "similarity_threshold"],
        BatchKind::FromBatch => &["image", "batch_index", "length"],
        BatchKind::Grid => &["images", "columns", "cell_width", "cell_height", "padding"],
        BatchKind::MergeTiles => &["image_list", "final_width", "final_height", "overlap"],
        BatchKind::MergeLists => &["images"],
        BatchKind::Rebatch => &["images", "batch_size"],
        BatchKind::Repeat => &["image", "amount"],
        BatchKind::Shuffle => &["images", "seed"],
    };
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn input_descriptors(
    kind: BatchKind,
    image_type: &NativeHandleType,
) -> Result<Vec<NativeInputDescriptor>, NativeNodeContractError> {
    let scalar = NativePortCardinality::Scalar;
    let list = NativePortCardinality::List;
    let inputs = match kind {
        BatchKind::BatchImages => Vec::new(),
        BatchKind::ImageBatch => vec![
            handle_input("image1", image_type.clone(), scalar)?,
            handle_input("image2", image_type.clone(), scalar)?,
        ],
        BatchKind::Deduplicate => vec![
            handle_input("images", image_type.clone(), list)?,
            number_input("similarity_threshold", list)?,
        ],
        BatchKind::FromBatch => vec![
            handle_input("image", image_type.clone(), scalar)?,
            integer_input("batch_index", scalar)?,
            integer_input("length", scalar)?,
        ],
        BatchKind::Grid => vec![
            handle_input("images", image_type.clone(), list)?,
            integer_input("columns", list)?,
            integer_input("cell_width", list)?,
            integer_input("cell_height", list)?,
            integer_input("padding", list)?,
        ],
        BatchKind::MergeTiles => vec![
            handle_input("image_list", image_type.clone(), list)?,
            integer_input("final_width", list)?,
            integer_input("final_height", list)?,
            integer_input("overlap", list)?,
        ],
        BatchKind::MergeLists => vec![handle_input("images", image_type.clone(), list)?],
        BatchKind::Rebatch => vec![
            handle_input("images", image_type.clone(), list)?,
            integer_input("batch_size", list)?,
        ],
        BatchKind::Repeat => vec![
            handle_input("image", image_type.clone(), scalar)?,
            integer_input("amount", scalar)?,
        ],
        BatchKind::Shuffle => vec![
            handle_input("images", image_type.clone(), list)?,
            integer_input("seed", list)?,
        ],
    };
    Ok(inputs)
}

fn handle_input(
    name: &str,
    handle_type: NativeHandleType,
    cardinality: NativePortCardinality,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    input(
        name,
        NativeValueType::Handle(handle_type),
        cardinality,
        false,
    )
}

fn integer_input(
    name: &str,
    cardinality: NativePortCardinality,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    input(
        name,
        NativeValueType::Primitive(NativePrimitiveType::Integer),
        cardinality,
        true,
    )
}

fn number_input(
    name: &str,
    cardinality: NativePortCardinality,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    input(
        name,
        NativeValueType::Primitive(NativePrimitiveType::Number),
        cardinality,
        true,
    )
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
struct ImageBatchFamilyNode {
    kind: BatchKind,
}

impl NativeNode for ImageBatchFamilyNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        self.kind.source_version()
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_input_shape(self.kind, inputs)?;
        Ok(self.kind.cache_token())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, self.class_type())?;
        validate_input_shape(self.kind, inputs)?;
        Ok(NativeCacheDependencies {
            rng_phase: (self.kind == BatchKind::Shuffle).then(|| SHUFFLE_RNG_PHASE.to_owned()),
            ..Default::default()
        })
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.class_type())?;
            match self.kind {
                BatchKind::BatchImages => execute_batch_images(&context, &inputs),
                BatchKind::ImageBatch => execute_image_batch(&context, &inputs),
                BatchKind::Deduplicate => execute_deduplicate(&context, &inputs),
                BatchKind::FromBatch => execute_from_batch(&context, &inputs),
                BatchKind::Grid => execute_grid(&context, &inputs),
                BatchKind::MergeTiles => execute_merge_tiles(&context, &inputs),
                BatchKind::MergeLists => execute_merge_lists(&context, &inputs),
                BatchKind::Rebatch => execute_rebatch(&context, &inputs),
                BatchKind::Repeat => execute_repeat(&context, &inputs),
                BatchKind::Shuffle => execute_shuffle(&context, &inputs),
            }
        })
    }
}

fn validate_input_shape(
    kind: BatchKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if kind == BatchKind::BatchImages {
        dynamic_image_handles(inputs)?;
        return Ok(());
    }
    let expected = input_names(kind);
    if inputs.len() != expected.len() || expected.iter().any(|name| !inputs.contains_key(name)) {
        return Err(invalid_inputs(format!(
            "{} requires exactly {}",
            kind.class_type(),
            expected.join(", ")
        )));
    }
    Ok(())
}

fn execute_batch_images(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let handles = dynamic_image_handles(inputs)?;
    batch_handles(context, &handles, BatchKind::BatchImages)
}

fn execute_image_batch(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::ImageBatch, inputs)?;
    let handles = vec![
        scalar_image_handle(inputs.get("image1"), "image1")?,
        scalar_image_handle(inputs.get("image2"), "image2")?,
    ];
    batch_handles(context, &handles, BatchKind::ImageBatch)
}

fn batch_handles(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
    kind: BatchKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let resolved = resolve_images(context, handles, kind.class_type())?;
    let images = resolved
        .iter()
        .map(|payload| require_image_payload(payload.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let first = images
        .first()
        .ok_or_else(|| invalid_inputs("at least one IMAGE input is required"))?;
    let (_, target_height, target_width, _) = first.dimensions().map_err(native_failure)?;
    let max_channels = images
        .iter()
        .map(|image| {
            image
                .dimensions()
                .map(|(_, _, _, channels)| channels)
                .map_err(native_failure)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| invalid_inputs("at least one IMAGE input is required"))?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let mut prepared = Vec::with_capacity(images.len());
    for image in images {
        check_cancellation(context, kind.class_type())?;
        let image = pad_channels(image, max_channels, compute.backend(), &execution)?;
        let (_, height, width, _) = image.dimensions().map_err(native_failure)?;
        let image = if height != target_height || width != target_width {
            image
                .resize(
                    target_width,
                    target_height,
                    ResizeMode::Bilinear,
                    ResizeCrop::Center,
                    compute.backend(),
                    &execution,
                )
                .map_err(native_failure)?
        } else {
            image
        };
        prepared.push(image);
    }
    let output = concatenate_images(&prepared, compute.backend(), &execution)?;
    drop(resolved);
    publish_image(context, output, kind.class_type())
}

fn execute_from_batch(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::FromBatch, inputs)?;
    let handle = scalar_image_handle(inputs.get("image"), "image")?;
    let batch_index = scalar_i64(inputs.get("batch_index"), "batch_index")?;
    if !(-i64::try_from(MAX_RESOLUTION).map_err(native_failure)?
        ..=i64::try_from(MAX_RESOLUTION).map_err(native_failure)?)
        .contains(&batch_index)
    {
        return Err(invalid_inputs("batch_index is outside the source bounds"));
    }
    let length = bounded_scalar_u64(inputs.get("length"), "length", 1, 4_096)?;
    let resolved = resolve_images(context, std::slice::from_ref(&handle), "ImageFromBatch")?;
    let image = require_image_payload(
        resolved
            .first()
            .ok_or_else(|| native_failure("IMAGE resolution was empty"))?,
    )?;
    let (batch, _, _, _) = image.dimensions().map_err(native_failure)?;
    if batch == 0 {
        return Err(native_failure("ImageFromBatch cannot select from an empty batch"));
    }
    let batch_i64 = i64::try_from(batch).map_err(native_failure)?;
    let adjusted = if batch_index < 0 {
        batch_index.saturating_add(batch_i64)
    } else {
        batch_index
    };
    let start = adjusted.clamp(0, batch_i64 - 1);
    let start = u64::try_from(start).map_err(native_failure)?;
    let length = length.min(batch - start);
    let view = narrow_method_exact_native(
        image.tensor(),
        0,
        i64::try_from(start).map_err(native_failure)?,
        length,
        &context.cancellation,
    )
    .map_err(native_failure)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let tensor = contiguous_with_context_exact_native(
        compute.backend(),
        &view,
        MemoryFormatReference::Layout(Layout::Contiguous),
        &execution,
    )
    .map_err(native_failure)?;
    let output = ImageTensor::from_tensor(tensor).map_err(native_failure)?;
    drop(resolved);
    publish_image(context, output, "ImageFromBatch")
}

fn execute_repeat(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::Repeat, inputs)?;
    let handle = scalar_image_handle(inputs.get("image"), "image")?;
    let amount = bounded_scalar_u64(inputs.get("amount"), "amount", 1, 4_096)?;
    let resolved = resolve_images(context, std::slice::from_ref(&handle), "RepeatImageBatch")?;
    let image = require_image_payload(
        resolved
            .first()
            .ok_or_else(|| native_failure("IMAGE resolution was empty"))?,
    )?;
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    let output_batch = batch
        .checked_mul(amount)
        .ok_or_else(|| invalid_inputs("repeated image batch is too large"))?;
    let source = image.as_f32_slice().map_err(native_failure)?;
    let output_count = usize::try_from(
        output_batch
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| invalid_inputs("repeated image storage is too large"))?,
    )
    .map_err(|_| invalid_inputs("repeated image storage is too large"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|error| native_failure(error.to_string()))?;
    for _ in 0..amount {
        check_cancellation(context, "RepeatImageBatch")?;
        values.extend_from_slice(source);
    }
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let output = ImageTensor::from_f32(
        compute.backend(),
        &execution,
        output_batch,
        height,
        width,
        channels,
        &values,
    )
    .map_err(native_failure)?;
    drop(resolved);
    publish_image(context, output, "RepeatImageBatch")
}

fn execute_merge_lists(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::MergeLists, inputs)?;
    let handles = image_handle_list(inputs.get("images"), "images")?;
    let images = flatten_images(context, &handles, "MergeImageLists")?;
    publish_image_list(context, images, "MergeImageLists")
}

fn execute_rebatch(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::Rebatch, inputs)?;
    let handles = image_handle_list(inputs.get("images"), "images")?;
    let batch_size = bounded_list_scalar_u64(inputs.get("batch_size"), "batch_size", 1, 4_096)?;
    let images = flatten_images(context, &handles, "RebatchImages")?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let mut batches = Vec::new();
    for chunk in images.chunks(usize::try_from(batch_size).map_err(native_failure)?) {
        check_cancellation(context, "RebatchImages")?;
        batches.push(concatenate_images(chunk, compute.backend(), &execution)?);
    }
    publish_image_list(context, batches, "RebatchImages")
}

fn execute_shuffle(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::Shuffle, inputs)?;
    let handles = image_handle_list(inputs.get("images"), "images")?;
    let seed = list_scalar_u64(inputs.get("seed"), "seed")?;
    let images = flatten_images(context, &handles, "ShuffleDataset")?;
    let permutation = numpy_permutation(images.len(), seed, &context.cancellation)?;
    let mut shuffled = Vec::with_capacity(images.len());
    for source in permutation {
        check_cancellation(context, "ShuffleDataset")?;
        shuffled.push(
            images
                .get(source)
                .ok_or_else(|| native_failure("permutation index exceeded the image list"))?
                .clone(),
        );
    }
    publish_image_list(context, shuffled, "ShuffleDataset")
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
            .map_err(|_| interrupted_failure("ShuffleDataset"))?;
        let high_exclusive = u32::try_from(upper + 1)
            .map_err(|_| invalid_inputs("shuffle list exceeds NumPy's permutation range"))?;
        let selected = random_state
            .randint(0, high_exclusive, cancellation)
            .map_err(native_failure)?;
        permutation.swap(upper, usize::try_from(selected).map_err(native_failure)?);
    }
    Ok(permutation)
}

fn execute_deduplicate(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::Deduplicate, inputs)?;
    let handles = image_handle_list(inputs.get("images"), "images")?;
    let threshold = bounded_list_scalar_f32(
        inputs.get("similarity_threshold"),
        "similarity_threshold",
        0.0,
        1.0,
    )?;
    let images = flatten_images(context, &handles, "ImageDeduplication")?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let mut hashes = Vec::with_capacity(images.len());
    for image in &images {
        check_cancellation(context, "ImageDeduplication")?;
        hashes.push(perceptual_hash(
            image,
            compute.backend(),
            &execution,
        )?);
    }
    let mut kept = Vec::new();
    let mut kept_hashes = Vec::new();
    for (image, hash) in images.into_iter().zip(hashes) {
        check_cancellation(context, "ImageDeduplication")?;
        let duplicate = kept_hashes.iter().any(|kept_hash| {
            let distance = (hash ^ kept_hash).count_ones();
            let similarity = 1.0 - (distance as f32 / 64.0);
            similarity >= threshold
        });
        if !duplicate {
            kept.push(image);
            kept_hashes.push(hash);
        }
    }
    publish_image_list(context, kept, "ImageDeduplication")
}

fn execute_grid(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::Grid, inputs)?;
    let handles = image_handle_list(inputs.get("images"), "images")?;
    let columns = bounded_list_scalar_u64(inputs.get("columns"), "columns", 1, 20)?;
    let cell_width = bounded_list_scalar_u64(inputs.get("cell_width"), "cell_width", 32, 2_048)?;
    let cell_height =
        bounded_list_scalar_u64(inputs.get("cell_height"), "cell_height", 32, 2_048)?;
    let padding = bounded_list_scalar_u64(inputs.get("padding"), "padding", 0, 50)?;
    let images = flatten_images(context, &handles, "ImageGrid")?;
    if images.is_empty() {
        return Err(invalid_inputs("Cannot create grid from empty image list"));
    }
    let image_count = u64::try_from(images.len()).map_err(native_failure)?;
    let rows = image_count
        .checked_add(columns - 1)
        .ok_or_else(|| invalid_inputs("grid row count overflowed"))?
        / columns;
    let grid_width = columns
        .checked_mul(cell_width)
        .and_then(|value| value.checked_add((columns - 1).checked_mul(padding)?))
        .ok_or_else(|| invalid_inputs("grid width overflowed"))?;
    let grid_height = rows
        .checked_mul(cell_height)
        .and_then(|value| value.checked_add((rows - 1).checked_mul(padding)?))
        .ok_or_else(|| invalid_inputs("grid height overflowed"))?;
    let output_count = usize::try_from(
        grid_width
            .checked_mul(grid_height)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| invalid_inputs("grid storage is too large"))?,
    )
    .map_err(|_| invalid_inputs("grid storage is too large"))?;
    let mut output = vec![0.0_f32; output_count];
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    for (index, image) in images.iter().enumerate() {
        check_cancellation(context, "ImageGrid")?;
        let resized = image
            .resize(
                cell_width,
                cell_height,
                ResizeMode::Lanczos,
                ResizeCrop::Disabled,
                compute.backend(),
                &execution,
            )
            .map_err(native_failure)?;
        let rgb = source_compatible_rgb(&resized)?;
        let index = u64::try_from(index).map_err(native_failure)?;
        let row = index / columns;
        let column = index % columns;
        let x = column
            .checked_mul(cell_width + padding)
            .ok_or_else(|| invalid_inputs("grid x coordinate overflowed"))?;
        let y = row
            .checked_mul(cell_height + padding)
            .ok_or_else(|| invalid_inputs("grid y coordinate overflowed"))?;
        paste_rgb(
            &mut output,
            grid_width,
            grid_height,
            &rgb,
            cell_width,
            cell_height,
            x,
            y,
        )?;
    }
    let image = ImageTensor::from_f32(
        compute.backend(),
        &execution,
        1,
        grid_height,
        grid_width,
        3,
        &output,
    )
    .map_err(native_failure)?;
    publish_image(context, image, "ImageGrid")
}

fn execute_merge_tiles(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    validate_input_shape(BatchKind::MergeTiles, inputs)?;
    let handles = image_handle_list(inputs.get("image_list"), "image_list")?;
    let final_width =
        bounded_list_scalar_u64(inputs.get("final_width"), "final_width", 64, 32_768)?;
    let final_height =
        bounded_list_scalar_u64(inputs.get("final_height"), "final_height", 64, 32_768)?;
    let overlap = bounded_list_scalar_u64(inputs.get("overlap"), "overlap", 0, 4_096)?;
    let resolved = resolve_images(context, &handles, "ImageMergeTileList")?;
    let images = resolved
        .iter()
        .map(|payload| require_image_payload(payload))
        .collect::<Result<Vec<_>, _>>()?;
    let first = images
        .first()
        .ok_or_else(|| invalid_inputs("image_list must not be empty"))?;
    let (batch, tile_height, tile_width, channels) = first.dimensions().map_err(native_failure)?;
    let coordinates = tile_coordinates(
        final_width,
        final_height,
        tile_width,
        tile_height,
        overlap,
    )?;
    let count = usize::try_from(
        batch
            .checked_mul(final_height)
            .and_then(|value| value.checked_mul(final_width))
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| invalid_inputs("merged tile storage is too large"))?,
    )
    .map_err(|_| invalid_inputs("merged tile storage is too large"))?;
    let weight_count = usize::try_from(
        batch
            .checked_mul(final_height)
            .and_then(|value| value.checked_mul(final_width))
            .ok_or_else(|| invalid_inputs("merged tile weights are too large"))?,
    )
    .map_err(|_| invalid_inputs("merged tile weights are too large"))?;
    let mut canvas = vec![0.0_f32; count];
    let mut weights = vec![0.0_f32; weight_count];
    let mask = tile_weight_mask(tile_width, tile_height, overlap)?;
    for (tile, coordinate) in images.iter().zip(coordinates) {
        check_cancellation(context, "ImageMergeTileList")?;
        let (tile_batch, height, width, tile_channels) = tile.dimensions().map_err(native_failure)?;
        if tile_batch != 1 && tile_batch != batch
            || tile_channels != 1 && tile_channels != channels
        {
            return Err(invalid_inputs(
                "every tile must broadcast to the first tile's batch and channel dimensions",
            ));
        }
        merge_tile_values(
            &mut canvas,
            &mut weights,
            final_width,
            final_height,
            batch,
            channels,
            tile,
            height,
            width,
            tile_width,
            tile_height,
            coordinate,
            &mask,
        )?;
    }
    for batch_index in 0..batch {
        for y in 0..final_height {
            check_cancellation(context, "ImageMergeTileList")?;
            for x in 0..final_width {
                let weight_index = usize::try_from(
                    batch_index
                        .checked_mul(final_height)
                        .and_then(|value| value.checked_add(y))
                        .and_then(|value| value.checked_mul(final_width))
                        .and_then(|value| value.checked_add(x))
                        .ok_or_else(|| native_failure("tile weight index overflowed"))?,
                )
                .map_err(native_failure)?;
                let weight = weights
                    .get(weight_index)
                    .copied()
                    .ok_or_else(|| native_failure("tile weight index exceeded storage"))?;
                let weight = if weight == 0.0 { 1.0 } else { weight };
                for channel in 0..channels {
                    let output_index = image_offset(
                        batch_index,
                        y,
                        x,
                        channel,
                        final_height,
                        final_width,
                        channels,
                    )?;
                    let value = canvas
                        .get_mut(output_index)
                        .ok_or_else(|| native_failure("tile canvas index exceeded storage"))?;
                    *value /= weight;
                }
            }
        }
    }
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let image = ImageTensor::from_f32(
        compute.backend(),
        &execution,
        batch,
        final_height,
        final_width,
        channels,
        &canvas,
    )
    .map_err(native_failure)?;
    drop(images);
    drop(resolved);
    publish_image(context, image, "ImageMergeTileList")
}

fn dynamic_image_handles(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<Vec<NativeOpaqueHandle>, NativeNodeFailure> {
    if inputs.is_empty() || inputs.len() > MAX_AUTOGROW_INPUTS as usize {
        return Err(invalid_inputs(
            "BatchImagesNode requires between 1 and 50 dynamic IMAGE inputs",
        ));
    }
    let mut indexed = inputs
        .iter()
        .map(|(name, value)| {
            let suffix = name
                .strip_prefix("image")
                .ok_or_else(|| invalid_inputs("dynamic IMAGE names must match image{index}"))?;
            let index = suffix
                .parse::<u32>()
                .map_err(|_| invalid_inputs("dynamic IMAGE names must match image{index}"))?;
            if !(1..=MAX_AUTOGROW_INPUTS).contains(&index) {
                return Err(invalid_inputs("dynamic IMAGE index is outside 1 through 50"));
            }
            Ok((index, scalar_image_handle(Some(value), name)?))
        })
        .collect::<Result<Vec<_>, NativeNodeFailure>>()?;
    indexed.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in (1_u32..).zip(&indexed) {
        if expected != *actual {
            return Err(invalid_inputs("dynamic IMAGE inputs must be contiguous"));
        }
    }
    Ok(indexed.into_iter().map(|(_, handle)| handle).collect())
}

fn scalar_image_handle(
    value: Option<&NativeValue>,
    name: &str,
) -> Result<NativeOpaqueHandle, NativeNodeFailure> {
    match value {
        Some(NativeValue::Handle { value })
            if value.handle_type().kind == NativeHandleKind::Image
                && value.handle_type().type_id == "IMAGE" =>
        {
            Ok(value.clone())
        }
        _ => Err(invalid_inputs(format!(
            "{name} must be an exact IMAGE handle"
        ))),
    }
}

fn image_handle_list(
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
        .map(|value| scalar_image_handle(Some(value), name))
        .collect()
}

fn scalar_i64(value: Option<&NativeValue>, name: &str) -> Result<i64, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => i64::try_from(*value).map_err(|_| invalid_inputs(format!("{name} is too large"))),
        _ => Err(invalid_inputs(format!("{name} must be an integer"))),
    }
}

fn scalar_u64(value: Option<&NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    let value = scalar_i64(value, name)?;
    u64::try_from(value).map_err(|_| invalid_inputs(format!("{name} must be non-negative")))
}

fn list_scalar_u64(value: Option<&NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    let Some(NativeValue::List { values }) = value else {
        return Err(invalid_inputs(format!("{name} must be a list")));
    };
    scalar_u64(values.first(), name)
}

fn bounded_scalar_u64(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = scalar_u64(value, name)?;
    bounded_u64(value, name, minimum, maximum)
}

fn bounded_list_scalar_u64(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = list_scalar_u64(value, name)?;
    bounded_u64(value, name, minimum, maximum)
}

fn bounded_u64(
    value: u64,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn bounded_list_scalar_f32(
    value: Option<&NativeValue>,
    name: &str,
    minimum: f32,
    maximum: f32,
) -> Result<f32, NativeNodeFailure> {
    let Some(NativeValue::List { values }) = value else {
        return Err(invalid_inputs(format!("{name} must be a list")));
    };
    let value = match values.first() {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) if value.is_finite() => *value as f32,
        _ => return Err(invalid_inputs(format!("{name} must be a finite number"))),
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn resolve_images(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
    class_type: &str,
) -> Result<Vec<crate::NativeResolvedPayload>, NativeNodeFailure> {
    let expected = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    handles
        .iter()
        .map(|handle| {
            context
                .handle_store()
                .resolve(handle, &expected, &context.cancellation)
                .map_err(|error| handle_failure(error, class_type))
        })
        .collect()
}

fn require_image_payload(
    payload: &NativeStoredPayload,
) -> Result<&ImageTensor, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = payload else {
        return Err(native_failure("IMAGE handle did not resolve to tensor storage"));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(native_failure("IMAGE handle resolved to the wrong tensor role"));
    }
    payload
        .image()
        .ok_or_else(|| native_failure("IMAGE handle did not resolve to canonical image storage"))
}

fn flatten_images(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
    class_type: &str,
) -> Result<Vec<ImageTensor>, NativeNodeFailure> {
    let resolved = resolve_images(context, handles, class_type)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let mut images = Vec::new();
    for payload in &resolved {
        let image = require_image_payload(payload)?;
        let (batch, _, _, _) = image.dimensions().map_err(native_failure)?;
        for index in 0..batch {
            check_cancellation(context, class_type)?;
            if images.len() >= MAX_LIST_VALUES {
                return Err(invalid_inputs("flattened IMAGE list exceeds the native limit"));
            }
            let view = narrow_method_exact_native(
                image.tensor(),
                0,
                i64::try_from(index).map_err(native_failure)?,
                1,
                &context.cancellation,
            )
            .map_err(native_failure)?;
            let tensor = contiguous_with_context_exact_native(
                compute.backend(),
                &view,
                MemoryFormatReference::Layout(Layout::Contiguous),
                &execution,
            )
            .map_err(native_failure)?;
            images.push(ImageTensor::from_tensor(tensor).map_err(native_failure)?);
        }
    }
    Ok(images)
}

fn pad_channels(
    image: &ImageTensor,
    channels: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, input_channels) = image.dimensions().map_err(native_failure)?;
    if input_channels == channels {
        return Ok(image.clone());
    }
    if input_channels.checked_add(1) != Some(channels) {
        return Err(native_failure(
            "source channel padding supports exactly one added alpha channel",
        ));
    }
    let source = image.as_f32_slice().map_err(native_failure)?;
    let pixels = usize::try_from(
        batch
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| native_failure("image pixel count overflowed"))?,
    )
    .map_err(native_failure)?;
    let input_channels = usize::try_from(input_channels).map_err(native_failure)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            pixels
                .checked_mul(usize::try_from(channels).map_err(native_failure)?)
                .ok_or_else(|| native_failure("padded image storage overflowed"))?,
        )
        .map_err(|error| native_failure(error.to_string()))?;
    for pixel in source.chunks_exact(input_channels) {
        execution.check().map_err(native_failure)?;
        values.extend_from_slice(pixel);
        values.push(1.0);
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(native_failure)
}

fn concatenate_images(
    images: &[ImageTensor],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let first = images
        .first()
        .ok_or_else(|| invalid_inputs("cannot concatenate an empty IMAGE list"))?;
    let (_, height, width, channels) = first.dimensions().map_err(native_failure)?;
    let mut batch = 0_u64;
    let mut values = Vec::new();
    for image in images {
        execution.check().map_err(native_failure)?;
        let (image_batch, image_height, image_width, image_channels) =
            image.dimensions().map_err(native_failure)?;
        if (image_height, image_width, image_channels) != (height, width, channels) {
            return Err(invalid_inputs(
                "images must share height, width, and channels before concatenation",
            ));
        }
        batch = batch
            .checked_add(image_batch)
            .ok_or_else(|| invalid_inputs("image batch size overflowed"))?;
        values.extend_from_slice(image.as_f32_slice().map_err(native_failure)?);
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(native_failure)
}

fn perceptual_hash(
    image: &ImageTensor,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<u64, NativeNodeFailure> {
    let resized = image
        .resize(
            8,
            8,
            ResizeMode::Lanczos,
            ResizeCrop::Disabled,
            backend,
            execution,
        )
        .map_err(native_failure)?;
    let (_, _, _, channels) = resized.dimensions().map_err(native_failure)?;
    let channels = usize::try_from(channels).map_err(native_failure)?;
    let mut grayscale = Vec::with_capacity(64);
    for pixel in resized.as_f32_slice().map_err(native_failure)?.chunks_exact(channels) {
        let luminance = match pixel {
            [value] => quantize_u8(*value),
            [red, green, blue, ..] => {
                let red = u32::from(quantize_u8(*red));
                let green = u32::from(quantize_u8(*green));
                let blue = u32::from(quantize_u8(*blue));
                u8::try_from((299 * red + 587 * green + 114 * blue + 500) / 1_000)
                    .map_err(native_failure)?
            }
            _ => return Err(native_failure("IMAGE has an unsupported channel count")),
        };
        grayscale.push(luminance);
    }
    if grayscale.len() != 64 {
        return Err(native_failure("perceptual hash requires one 8x8 image"));
    }
    let average = grayscale.iter().map(|value| u64::from(*value)).sum::<u64>() as f64 / 64.0;
    Ok(grayscale
        .iter()
        .enumerate()
        .fold(0_u64, |hash, (index, value)| {
            if f64::from(*value) > average {
                hash | (1_u64 << index)
            } else {
                hash
            }
        }))
}

fn source_compatible_rgb(image: &ImageTensor) -> Result<Vec<f32>, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    if batch != 1 {
        return Err(invalid_inputs("grid entries must contain exactly one image"));
    }
    let count = usize::try_from(
        height
            .checked_mul(width)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| native_failure("RGB image storage overflowed"))?,
    )
    .map_err(native_failure)?;
    let channels = usize::try_from(channels).map_err(native_failure)?;
    let mut rgb = Vec::with_capacity(count);
    for pixel in image.as_f32_slice().map_err(native_failure)?.chunks_exact(channels) {
        match pixel {
            [value] => {
                let value = f32::from(quantize_u8(*value)) / 255.0;
                rgb.extend_from_slice(&[value, value, value]);
            }
            [red, green, blue, ..] => rgb.extend_from_slice(&[
                f32::from(quantize_u8(*red)) / 255.0,
                f32::from(quantize_u8(*green)) / 255.0,
                f32::from(quantize_u8(*blue)) / 255.0,
            ]),
            _ => return Err(native_failure("IMAGE has an unsupported channel count")),
        }
    }
    Ok(rgb)
}

fn quantize_u8(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).trunc() as u8
}

#[allow(clippy::too_many_arguments)]
fn paste_rgb(
    output: &mut [f32],
    output_width: u64,
    output_height: u64,
    source: &[f32],
    source_width: u64,
    source_height: u64,
    x: u64,
    y: u64,
) -> Result<(), NativeNodeFailure> {
    for source_y in 0..source_height {
        for source_x in 0..source_width {
            for channel in 0..3_u64 {
                let source_index = image_offset(
                    0,
                    source_y,
                    source_x,
                    channel,
                    source_height,
                    source_width,
                    3,
                )?;
                let output_index = image_offset(
                    0,
                    y + source_y,
                    x + source_x,
                    channel,
                    output_height,
                    output_width,
                    3,
                )?;
                let source_value = source
                    .get(source_index)
                    .copied()
                    .ok_or_else(|| native_failure("grid source index exceeded storage"))?;
                *output
                    .get_mut(output_index)
                    .ok_or_else(|| native_failure("grid output index exceeded storage"))? =
                    source_value;
            }
        }
    }
    Ok(())
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
            coordinates.push((x_start, y_start, x_end, y_end));
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
    if size == 0 || tile == 0 || stride == 0 {
        return Err(invalid_inputs("tile dimensions and stride must be non-zero"));
    }
    let mut coordinates = Vec::new();
    let mut start = 0_u64;
    loop {
        let end = start.saturating_add(tile).min(size);
        coordinates.push((end.saturating_sub(tile), end));
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

fn tile_weight_mask(
    width: u64,
    height: u64,
    overlap: u64,
) -> Result<Vec<f32>, NativeNodeFailure> {
    let count = usize::try_from(
        width
            .checked_mul(height)
            .ok_or_else(|| native_failure("tile weight mask is too large"))?,
    )
    .map_err(native_failure)?;
    if overlap == 0 {
        return Ok(vec![1.0; count]);
    }
    let mut mask = Vec::with_capacity(count);
    for y in 0..height {
        let y_fraction = if height == 1 {
            0.0
        } else {
            y as f32 / (height - 1) as f32
        };
        let y_weight = (std::f32::consts::PI * y_fraction).sin().max(1.0e-5);
        for x in 0..width {
            let x_fraction = if width == 1 {
                0.0
            } else {
                x as f32 / (width - 1) as f32
            };
            let x_weight = (std::f32::consts::PI * x_fraction).sin().max(1.0e-5);
            mask.push(y_weight * x_weight);
        }
    }
    Ok(mask)
}

#[allow(clippy::too_many_arguments)]
fn merge_tile_values(
    canvas: &mut [f32],
    weights: &mut [f32],
    final_width: u64,
    final_height: u64,
    batch: u64,
    channels: u64,
    tile: &ImageTensor,
    tile_height: u64,
    tile_width: u64,
    mask_width: u64,
    mask_height: u64,
    coordinate: (u64, u64, u64, u64),
    mask: &[f32],
) -> Result<(), NativeNodeFailure> {
    let (x_start, y_start, x_end, y_end) = coordinate;
    let real_height = (y_end - y_start).min(tile_height);
    let real_width = (x_end - x_start).min(tile_width);
    let (tile_batch, _, _, tile_channels) = tile.dimensions().map_err(native_failure)?;
    let values = tile.as_f32_slice().map_err(native_failure)?;
    for batch_index in 0..batch {
        for y in 0..real_height {
            for x in 0..real_width {
                let mask_index = usize::try_from(
                    y.checked_mul(mask_width)
                        .and_then(|value| value.checked_add(x))
                        .ok_or_else(|| native_failure("tile mask index overflowed"))?,
                )
                .map_err(native_failure)?;
                if y >= mask_height {
                    return Err(native_failure("tile mask height changed"));
                }
                let weight = *mask
                    .get(mask_index)
                    .ok_or_else(|| native_failure("tile mask index exceeded storage"))?;
                let weight_index = usize::try_from(
                    batch_index
                        .checked_mul(final_height)
                        .and_then(|value| value.checked_add(y_start + y))
                        .and_then(|value| value.checked_mul(final_width))
                        .and_then(|value| value.checked_add(x_start + x))
                        .ok_or_else(|| native_failure("tile weight index overflowed"))?,
                )
                .map_err(native_failure)?;
                *weights
                    .get_mut(weight_index)
                    .ok_or_else(|| native_failure("tile weight index exceeded storage"))? +=
                    weight;
                for channel in 0..channels {
                    let source_index = image_offset(
                        if tile_batch == 1 { 0 } else { batch_index },
                        y,
                        x,
                        if tile_channels == 1 { 0 } else { channel },
                        tile_height,
                        tile_width,
                        tile_channels,
                    )?;
                    let output_index = image_offset(
                        batch_index,
                        y_start + y,
                        x_start + x,
                        channel,
                        final_height,
                        final_width,
                        channels,
                    )?;
                    let source = values
                        .get(source_index)
                        .copied()
                        .ok_or_else(|| native_failure("tile source index exceeded storage"))?;
                    *canvas
                        .get_mut(output_index)
                        .ok_or_else(|| native_failure("tile canvas index exceeded storage"))? +=
                        source * weight;
                }
            }
        }
    }
    Ok(())
}

fn image_offset(
    batch: u64,
    y: u64,
    x: u64,
    channel: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<usize, NativeNodeFailure> {
    usize::try_from(
        batch
            .checked_mul(height)
            .and_then(|value| value.checked_add(y))
            .and_then(|value| value.checked_mul(width))
            .and_then(|value| value.checked_add(x))
            .and_then(|value| value.checked_mul(channels))
            .and_then(|value| value.checked_add(channel))
            .ok_or_else(|| native_failure("IMAGE index overflowed"))?,
    )
    .map_err(native_failure)
}

fn publish_image(
    context: &NativeNodeContext,
    image: ImageTensor,
    class_type: &str,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
        .map_err(native_failure)?;
    check_cancellation(context, class_type)?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure(error, class_type))?;
    if let Err(failure) = check_cancellation(context, class_type) {
        revoke_output(context, &handle)?;
        return Err(failure);
    }
    values_outcome(vec![NativeValue::Handle { value: handle }])
}

fn publish_image_list(
    context: &NativeNodeContext,
    images: Vec<ImageTensor>,
    class_type: &str,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let payloads = images
        .into_iter()
        .map(|image| {
            NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                .map(|payload| NativeStoredPayload::Tensor(Arc::new(payload)))
                .map_err(native_failure)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut published = Vec::with_capacity(payloads.len());
    for payload in payloads {
        if let Err(failure) = check_cancellation(context, class_type) {
            rollback_published(context, &published)?;
            return Err(failure);
        }
        match context
            .handle_store()
            .publish(payload, &context.cancellation)
        {
            Ok(handle) => published.push(handle),
            Err(error) => {
                rollback_published(context, &published)?;
                return Err(handle_failure(error, class_type));
            }
        }
    }
    if let Err(failure) = check_cancellation(context, class_type) {
        rollback_published(context, &published)?;
        return Err(failure);
    }
    values_outcome(vec![NativeValue::List {
        values: published
            .into_iter()
            .map(|value| NativeValue::Handle { value })
            .collect(),
    }])
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
                message: format!("failed to revoke a partially published IMAGE: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
    }
    Ok(())
}

fn revoke_output(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<(), NativeNodeFailure> {
    context
        .handle_store()
        .revoke(handle, &CancellationToken::default())
        .map_err(|error| NativeNodeFailure {
            code: "native_image_batch_rollback_failed".to_owned(),
            message: format!("failed to revoke a cancelled IMAGE output: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        })
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
        "/../sim_test_support/fixtures/nodes/image-batch-comfy-node-0017/fixture.json"
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
                    Uuid::from_u128(0x49601),
                    Uuid::from_u128(0x49602),
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

        fn payload(
            &self,
            handle: &NativeOpaqueHandle,
        ) -> Result<Arc<NativeStoredPayload>, NativeHandleStoreError> {
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))
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
            let payload = self.payload(handle)?;
            if payload.digest_sha256() != handle.digest_sha256().unwrap_or_default() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            Ok(NativeResolvedPayload::checked(
                payload,
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

    struct Harness {
        backend: Arc<CpuBackend>,
        store: Arc<TestStore>,
        context: NativeNodeContext,
    }

    impl Harness {
        fn new(node_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x49603));
            let node_id = NodeId(node_id.to_owned());
            let store = TestStore::new(attempt_id)?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
            let backend = Arc::new(backend);
            let scratch = authority.authorize_workspace(64 * 1024 * 1024)?;
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x49604),
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
                PromptId(Uuid::from_u128(0x49605)),
                attempt_id,
                node_id,
                CancellationToken::default(),
                scratch,
                store.clone(),
                services,
            )?;
            Ok(Self {
                backend,
                store,
                context,
            })
        }

        fn image(
            &self,
            batch: u64,
            height: u64,
            width: u64,
            channels: u64,
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn std::error::Error>> {
            let compute = self.context.compute_session()?;
            let execution = compute.execution_context(&self.context)?;
            let image = ImageTensor::from_f32(
                &self.backend,
                &execution,
                batch,
                height,
                width,
                channels,
                values,
            )?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &CancellationToken::default(),
            )?)
        }

        fn output_image(
            &self,
            outcome: NativeNodeOutcome,
        ) -> Result<ImageTensor, Box<dyn std::error::Error>> {
            let NativeNodeOutcome::Values { outputs, .. } = outcome else {
                return Err("node did not return values".into());
            };
            let Some(NativeValue::Handle { value }) = outputs.first() else {
                return Err("node did not return an IMAGE handle".into());
            };
            self.image_for_handle(value)
        }

        fn output_list(
            &self,
            outcome: NativeNodeOutcome,
        ) -> Result<Vec<ImageTensor>, Box<dyn std::error::Error>> {
            let NativeNodeOutcome::Values { outputs, .. } = outcome else {
                return Err("node did not return values".into());
            };
            let Some(NativeValue::List { values }) = outputs.first() else {
                return Err("node did not return an IMAGE list".into());
            };
            values
                .iter()
                .map(|value| {
                    let NativeValue::Handle { value } = value else {
                        return Err("IMAGE list contains a non-handle".into());
                    };
                    self.image_for_handle(value)
                })
                .collect()
        }

        fn image_for_handle(
            &self,
            handle: &NativeOpaqueHandle,
        ) -> Result<ImageTensor, Box<dyn std::error::Error>> {
            let payload = self.store.payload(handle)?;
            let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
                return Err("IMAGE handle payload is not a tensor".into());
            };
            Ok(payload
                .image()
                .ok_or("IMAGE handle payload is not canonical")?
                .clone())
        }
    }

    fn executable(kind: BatchKind) -> Result<Arc<dyn NativeNode>, Box<dyn std::error::Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == kind.class_type() => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{} executable is absent", kind.class_type()).into())
    }

    fn handle_value(handle: NativeOpaqueHandle) -> NativeValue {
        NativeValue::Handle { value: handle }
    }

    fn integer(value: i64) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }
    }

    fn number(value: f64) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }
    }

    fn list(values: Vec<NativeValue>) -> NativeValue {
        NativeValue::List { values }
    }

    #[test]
    fn descriptors_and_fixture_cover_exact_assigned_rows() -> Result<(), Box<dyn std::error::Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture["stable_task_id"],
            "comfy-parity-native-nodes-image-batch-comfy-node-0017"
        );
        let fixture_nodes = fixture["nodes"].as_array().ok_or("fixture nodes")?;
        assert_eq!(fixture_nodes.len(), ALL_KINDS.len());
        for (fixture_node, kind) in fixture_nodes.iter().zip(ALL_KINDS) {
            assert_eq!(fixture_node["feature_id"], kind.feature_id());
            assert_eq!(fixture_node["class_type"], kind.class_type());
        }
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        for ((binding, class_type), kind) in bindings
            .iter()
            .zip(NODE_DESCRIPTOR_IDS)
            .zip(ALL_KINDS)
        {
            binding.validate()?;
            assert_eq!(binding.feature_id(), kind.feature_id());
            assert_eq!(binding.descriptor().class_type, *class_type);
            assert_eq!(binding.descriptor().effect, NativeEffectClass::Pure);
            assert_eq!(binding.descriptor().cache, NativeCachePolicy::InputIdentity);
            assert_eq!(binding.descriptor().outputs[0].is_list, kind.output_is_list());
        }
        let batch = &bindings[0].descriptor();
        assert!(batch.inputs.is_empty());
        assert_eq!(batch.dynamic_inputs.len(), 1);
        assert_eq!(batch.dynamic_inputs[0].name_template, "image{index}");
        assert_eq!(batch.dynamic_inputs[0].maximum_count, 50);
        assert!(bindings[1].presentation().is_deprecated);
        Ok(())
    }

    #[test]
    fn batching_selection_and_repeat_preserve_source_order() -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new("batching")?;
        let first = harness.image(1, 1, 1, 3, &[0.1, 0.2, 0.3])?;
        let second = harness.image(1, 1, 1, 4, &[0.4, 0.5, 0.6, 0.7])?;
        let outcome = futures::executor::block_on(executable(BatchKind::ImageBatch)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image1".to_owned(), handle_value(first.clone())),
                ("image2".to_owned(), handle_value(second.clone())),
            ]),
        ))?;
        let batched = harness.output_image(outcome)?;
        assert_eq!(batched.dimensions()?, (2, 1, 1, 4));
        assert_eq!(
            batched.as_f32_slice()?,
            &[0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 0.7]
        );

        let dynamic = futures::executor::block_on(executable(BatchKind::BatchImages)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image1".to_owned(), handle_value(first)),
                ("image2".to_owned(), handle_value(second)),
            ]),
        ))?;
        assert_eq!(
            harness.output_image(dynamic)?.as_f32_slice()?,
            &[0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 0.7]
        );

        let batched_handle = harness.image(2, 1, 1, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
        let selected = futures::executor::block_on(executable(BatchKind::FromBatch)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image".to_owned(), handle_value(batched_handle.clone())),
                ("batch_index".to_owned(), integer(-1)),
                ("length".to_owned(), integer(4)),
            ]),
        ))?;
        assert_eq!(
            harness.output_image(selected)?.as_f32_slice()?,
            &[0.0, 1.0, 0.0]
        );

        let repeated = futures::executor::block_on(executable(BatchKind::Repeat)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image".to_owned(), handle_value(batched_handle)),
                ("amount".to_owned(), integer(2)),
            ]),
        ))?;
        let repeated = harness.output_image(repeated)?;
        assert_eq!(repeated.dimensions()?, (4, 1, 1, 3));
        assert_eq!(
            repeated.as_f32_slice()?,
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn list_grid_tile_dedup_rebatch_and_shuffle_are_source_exact()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new("lists")?;
        let red_values = [1.0, 0.0, 0.0].repeat(64 * 64);
        let green_values = [0.0, 1.0, 0.0].repeat(64 * 64);
        let red = harness.image(1, 64, 64, 3, &red_values)?;
        let green = harness.image(1, 64, 64, 3, &green_values)?;

        let merged = futures::executor::block_on(executable(BatchKind::MergeLists)?.execute(
            harness.context.clone(),
            BTreeMap::from([(
                "images".to_owned(),
                list(vec![handle_value(red.clone()), handle_value(green.clone())]),
            )]),
        ))?;
        assert_eq!(harness.output_list(merged)?.len(), 2);

        let rebatched = futures::executor::block_on(executable(BatchKind::Rebatch)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "images".to_owned(),
                    list(vec![handle_value(red.clone()), handle_value(green.clone())]),
                ),
                ("batch_size".to_owned(), list(vec![integer(2)])),
            ]),
        ))?;
        assert_eq!(harness.output_list(rebatched)?[0].dimensions()?, (2, 64, 64, 3));

        let deduplicated = futures::executor::block_on(executable(BatchKind::Deduplicate)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "images".to_owned(),
                    list(vec![handle_value(red.clone()), handle_value(green.clone())]),
                ),
                (
                    "similarity_threshold".to_owned(),
                    list(vec![number(0.95)]),
                ),
            ]),
        ))?;
        assert_eq!(harness.output_list(deduplicated)?.len(), 1);

        let grid = futures::executor::block_on(executable(BatchKind::Grid)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "images".to_owned(),
                    list(vec![handle_value(red.clone()), handle_value(green.clone())]),
                ),
                ("columns".to_owned(), list(vec![integer(2)])),
                ("cell_width".to_owned(), list(vec![integer(32)])),
                ("cell_height".to_owned(), list(vec![integer(32)])),
                ("padding".to_owned(), list(vec![integer(0)])),
            ]),
        ))?;
        let grid = harness.output_image(grid)?;
        assert_eq!(grid.dimensions()?, (1, 32, 64, 3));
        assert_eq!(&grid.as_f32_slice()?[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&grid.as_f32_slice()?[32 * 3..32 * 3 + 3], &[0.0, 1.0, 0.0]);

        let tile = futures::executor::block_on(executable(BatchKind::MergeTiles)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image_list".to_owned(), list(vec![handle_value(red.clone())])),
                ("final_width".to_owned(), list(vec![integer(64)])),
                ("final_height".to_owned(), list(vec![integer(64)])),
                ("overlap".to_owned(), list(vec![integer(0)])),
            ]),
        ))?;
        assert_eq!(harness.output_image(tile)?.as_f32_slice()?, red_values);

        let first_shuffle = futures::executor::block_on(executable(BatchKind::Shuffle)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "images".to_owned(),
                    list(vec![handle_value(red.clone()), handle_value(green.clone())]),
                ),
                ("seed".to_owned(), list(vec![integer(7)])),
            ]),
        ))?;
        let second_shuffle = futures::executor::block_on(executable(BatchKind::Shuffle)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "images".to_owned(),
                    list(vec![handle_value(red), handle_value(green)]),
                ),
                ("seed".to_owned(), list(vec![integer(7)])),
            ]),
        ))?;
        let first = harness.output_list(first_shuffle)?;
        let second = harness.output_list(second_shuffle)?;
        assert_eq!(first.len(), 2);
        for (left, right) in first.iter().zip(second) {
            assert_eq!(left.as_f32_slice()?, right.as_f32_slice()?);
        }
        Ok(())
    }

    #[test]
    fn validation_cancellation_and_fresh_retry_publish_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new("recovery")?;
        let image = harness.image(1, 1, 1, 3, &[0.0, 0.5, 1.0])?;
        let count = harness.store.count()?;
        let failure = futures::executor::block_on(executable(BatchKind::Repeat)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image".to_owned(), handle_value(image.clone())),
                ("amount".to_owned(), integer(0)),
            ]),
        ))
        .expect_err("zero repeat amount must fail");
        assert_eq!(failure.code, "invalid_node_inputs");
        assert_eq!(harness.store.count()?, count);

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let mut cancelled_context = harness.context.clone();
        cancelled_context.cancellation = cancellation;
        let failure = futures::executor::block_on(executable(BatchKind::Repeat)?.execute(
            cancelled_context,
            BTreeMap::from([
                ("image".to_owned(), handle_value(image.clone())),
                ("amount".to_owned(), integer(2)),
            ]),
        ))
        .expect_err("cancelled repeat must fail");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, count);

        let recovered = futures::executor::block_on(executable(BatchKind::Repeat)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("image".to_owned(), handle_value(image)),
                ("amount".to_owned(), integer(2)),
            ]),
        ))?;
        assert_eq!(harness.output_image(recovered)?.dimensions()?, (2, 1, 1, 3));
        assert_eq!(harness.store.count()?, count + 1);
        Ok(())
    }

    #[test]
    fn shuffle_matches_numpy_random_state_permutation_oracle() -> Result<(), NativeNodeFailure> {
        assert_eq!(
            numpy_permutation(10, 7, &CancellationToken::default())?,
            vec![8, 5, 0, 2, 1, 9, 7, 3, 6, 4]
        );
        Ok(())
    }
}
