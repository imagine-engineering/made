use serde::{Deserialize, Serialize};
use sim_tensor::{
    DType, ExecutionContext, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
};
use std::collections::BTreeMap;
use thiserror::Error;

pub const LATENT_FORMAT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatentTensorLayout {
    ChannelsFirst,
    SequenceChannelsLast,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatentTransform {
    Identity,
    Affine,
    PerChannelAffine,
    HunyuanImage21Refiner,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreviewReshape {
    None,
    Flux2Spatial2x,
}

#[derive(Clone, Copy, Debug)]
pub struct LatentFormatDefinition {
    pub feature_id: &'static str,
    pub identifier: &'static str,
    pub channels: u64,
    pub dimensions: u8,
    pub spatial_downscale_ratio: u64,
    pub temporal_downscale_ratio: u64,
    pub scale_factor: f32,
    pub shift_factor: f32,
    pub channel_means: &'static [f32],
    pub channel_stds: &'static [f32],
    pub preview_factors: &'static [[f32; 3]],
    pub preview_bias: Option<[f32; 3]>,
    pub preview_reshape: PreviewReshape,
    pub decoder_name: Option<&'static str>,
    pub layout: LatentTensorLayout,
    pub transform: LatentTransform,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LatentFormatIdentity {
    feature_id: String,
    identifier: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LatentFormatIdentityWire {
    schema_version: u16,
    feature_id: String,
    identifier: String,
}

impl LatentFormatIdentity {
    pub fn new(
        feature_id: impl Into<String>,
        identifier: impl Into<String>,
    ) -> Result<Self, LatentFormatError> {
        let feature_id = feature_id.into();
        let identifier = identifier.into();
        validate_feature_id(&feature_id)?;
        validate_identifier(&identifier)?;
        Ok(Self {
            feature_id,
            identifier,
        })
    }

    pub fn feature_id(&self) -> &str {
        &self.feature_id
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn owned_resident_bytes(&self) -> Option<u64> {
        let bytes = self
            .feature_id
            .capacity()
            .checked_add(self.identifier.capacity())?;
        u64::try_from(bytes).ok()
    }
}

impl TryFrom<LatentFormatIdentityWire> for LatentFormatIdentity {
    type Error = LatentFormatError;

    fn try_from(value: LatentFormatIdentityWire) -> Result<Self, Self::Error> {
        if value.schema_version != LATENT_FORMAT_SCHEMA_VERSION {
            return Err(LatentFormatError::SchemaVersion(value.schema_version));
        }
        Self::new(value.feature_id, value.identifier)
    }
}

impl From<LatentFormatIdentity> for LatentFormatIdentityWire {
    fn from(value: LatentFormatIdentity) -> Self {
        Self {
            schema_version: LATENT_FORMAT_SCHEMA_VERSION,
            feature_id: value.feature_id,
            identifier: value.identifier,
        }
    }
}

impl Serialize for LatentFormatIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        LatentFormatIdentityWire::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LatentFormatIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = LatentFormatIdentityWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LatentFormatDescriptor {
    pub identity: LatentFormatIdentity,
    pub channels: u64,
    pub dimensions: u8,
    pub spatial_downscale_ratio: u64,
    pub temporal_downscale_ratio: u64,
    pub scale_factor: f32,
    pub shift_factor: f32,
    pub channel_means: Vec<f32>,
    pub channel_stds: Vec<f32>,
    pub preview_factors: Vec<[f32; 3]>,
    pub preview_bias: Option<[f32; 3]>,
    pub decoder_name: Option<String>,
}

impl LatentFormatDescriptor {
    pub fn checked(definition: &LatentFormatDefinition) -> Result<Self, LatentFormatError> {
        validate_definition(definition)?;
        Ok(Self {
            identity: LatentFormatIdentity::new(definition.feature_id, definition.identifier)?,
            channels: definition.channels,
            dimensions: definition.dimensions,
            spatial_downscale_ratio: definition.spatial_downscale_ratio,
            temporal_downscale_ratio: definition.temporal_downscale_ratio,
            scale_factor: definition.scale_factor,
            shift_factor: definition.shift_factor,
            channel_means: definition.channel_means.to_vec(),
            channel_stds: definition.channel_stds.to_vec(),
            preview_factors: definition.preview_factors.to_vec(),
            preview_bias: definition.preview_bias,
            decoder_name: definition.decoder_name.map(str::to_owned),
        })
    }

    pub fn owned_resident_bytes(&self) -> Option<u64> {
        let channel_means = self
            .channel_means
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())?;
        let channel_stds = self
            .channel_stds
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())?;
        let preview_factors = self
            .preview_factors
            .capacity()
            .checked_mul(std::mem::size_of::<[f32; 3]>())?;
        let bytes = channel_means
            .checked_add(channel_stds)?
            .checked_add(preview_factors)?
            .checked_add(self.decoder_name.as_ref().map_or(0, String::capacity))?;
        self.identity
            .owned_resident_bytes()?
            .checked_add(u64::try_from(bytes).ok()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentExtent {
    OneDimensional {
        batch: u64,
        length: u64,
    },
    TwoDimensional {
        batch: u64,
        width: u64,
        height: u64,
    },
    ThreeDimensional {
        batch: u64,
        frames: u64,
        width: u64,
        height: u64,
    },
}

#[derive(Clone, Debug)]
pub struct LatentFormatRegistry {
    formats: BTreeMap<LatentFormatIdentity, &'static LatentFormatDefinition>,
}

impl LatentFormatRegistry {
    pub fn checked(
        definitions: &'static [LatentFormatDefinition],
    ) -> Result<Self, LatentFormatError> {
        let mut formats = BTreeMap::new();
        let mut identifiers = BTreeMap::<&str, &str>::new();
        let mut feature_ids = BTreeMap::<&str, &str>::new();
        for definition in definitions {
            validate_definition(definition)?;
            if let Some(existing) = identifiers.insert(definition.identifier, definition.feature_id)
            {
                return Err(LatentFormatError::DuplicateIdentifier {
                    identifier: definition.identifier.to_owned(),
                    first_feature_id: existing.to_owned(),
                    second_feature_id: definition.feature_id.to_owned(),
                });
            }
            if let Some(existing) = feature_ids.insert(definition.feature_id, definition.identifier)
            {
                return Err(LatentFormatError::DuplicateFeatureId {
                    feature_id: definition.feature_id.to_owned(),
                    first_identifier: existing.to_owned(),
                    second_identifier: definition.identifier.to_owned(),
                });
            }
            let identity = LatentFormatIdentity::new(definition.feature_id, definition.identifier)?;
            formats.insert(identity, definition);
        }
        Ok(Self { formats })
    }

    pub fn get(&self, identity: &LatentFormatIdentity) -> Option<&'static LatentFormatDefinition> {
        self.formats.get(identity).copied()
    }

    pub fn len(&self) -> usize {
        self.formats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum LatentFormatError {
    #[error("unsupported latent-format schema version {0}")]
    SchemaVersion(u16),
    #[error("invalid latent-format feature id: {0}")]
    InvalidFeatureId(String),
    #[error("invalid latent-format identifier: {0}")]
    InvalidIdentifier(String),
    #[error("latent-format {identifier} has invalid {field}: {value}")]
    InvalidConstant {
        identifier: String,
        field: &'static str,
        value: String,
    },
    #[error("latent-format {identifier} requires {expected} dimensions but received {actual}")]
    ExtentDimensions {
        identifier: String,
        expected: u8,
        actual: u8,
    },
    #[error("latent-format {identifier} cannot construct an empty latent from {field}={value}")]
    InvalidExtent {
        identifier: String,
        field: &'static str,
        value: u64,
    },
    #[error("latent-format {identifier} expected shape {expected:?} but received {actual:?}")]
    InvalidShape {
        identifier: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("latent-format {identifier} has no RGB preview projection")]
    PreviewUnavailable { identifier: String },
    #[error(
        "duplicate latent-format identifier {identifier}: {first_feature_id} and {second_feature_id}"
    )]
    DuplicateIdentifier {
        identifier: String,
        first_feature_id: String,
        second_feature_id: String,
    },
    #[error(
        "duplicate latent-format feature id {feature_id}: {first_identifier} and {second_identifier}"
    )]
    DuplicateFeatureId {
        feature_id: String,
        first_identifier: String,
        second_identifier: String,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

pub fn empty_latent(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    extent: LatentExtent,
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LatentFormatError> {
    validate_definition(definition)?;
    context.check()?;
    let shape = empty_shape(definition, extent)?;
    let descriptor = TensorDescriptor::contiguous(shape, dtype, backend.device(), stream)?;
    let (tensor, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    Ok(tensor)
}

pub fn normalize_sampling_latent(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    spatial_downscale_ratio: Option<u32>,
    temporal_downscale_ratio: Option<u32>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LatentFormatError> {
    validate_definition(definition)?;
    validate_backend_input(backend, input, context)?;
    require_f32(input)?;
    let mut shape = input.descriptor().shape().to_vec();
    if shape.len() < 2 || shape.contains(&0) {
        return Err(LatentFormatError::InvalidShape {
            identifier: definition.identifier.to_owned(),
            expected: vec![0, definition.channels, 0],
            actual: shape,
        });
    }
    let mut empty = true;
    for index in 0..element_count(&shape)? {
        check_periodically(index, context)?;
        if read_f32_linear(input, index)? != 0.0 {
            empty = false;
            break;
        }
    }
    let resized_dimension = |size: u64, source_ratio: u32, target_ratio: u64, minimum: u64| {
        let scaled = (size as f64 * (f64::from(source_ratio) / target_ratio as f64))
            .round_ties_even()
            .max(minimum as f64);
        if !scaled.is_finite() || scaled < 1.0 || scaled >= u64::MAX as f64 {
            return Err(TensorError::ShapeOverflow);
        }
        Ok(scaled as u64)
    };
    if empty {
        *shape.get_mut(1).ok_or(TensorError::ShapeOverflow)? = definition.channels;
        if let Some(ratio) = spatial_downscale_ratio
            .filter(|ratio| u64::from(*ratio) != definition.spatial_downscale_ratio)
        {
            if shape.len() < 4 {
                return Err(LatentFormatError::InvalidShape {
                    identifier: definition.identifier.to_owned(),
                    expected: vec![0, definition.channels, 0, 0],
                    actual: shape,
                });
            }
            let rank = shape.len();
            for dimension in shape
                .get_mut(rank - 2..)
                .ok_or(TensorError::ShapeOverflow)?
            {
                *dimension =
                    resized_dimension(*dimension, ratio, definition.spatial_downscale_ratio, 0)?;
            }
        }
    }
    if definition.dimensions == 3 && shape.len() == 4 {
        shape.insert(2, 1);
    }
    if empty {
        if let Some(ratio) = temporal_downscale_ratio
            .filter(|ratio| u64::from(*ratio) != definition.temporal_downscale_ratio)
        {
            let temporal = shape.get_mut(2).ok_or(TensorError::ShapeOverflow)?;
            *temporal =
                resized_dimension(*temporal, ratio, definition.temporal_downscale_ratio, 1)?;
        }
    }
    context.check()?;
    if shape == input.descriptor().shape() {
        return Ok(input.clone());
    }
    if empty {
        let descriptor = TensorDescriptor::contiguous(
            shape,
            input.descriptor().dtype(),
            backend.device(),
            context.stream,
        )?;
        let (output, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
        Ok(output)
    } else {
        Ok(input.view(
            input.descriptor().reshaped_view(shape)?,
            sim_tensor::ViewAccess::ReadOnly,
        )?)
    }
}

pub fn process_latent_in(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LatentFormatError> {
    process(definition, backend, input, context, true)
}

pub fn process_latent_out(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LatentFormatError> {
    process(definition, backend, input, context, false)
}

pub fn project_latent_preview(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LatentFormatError> {
    validate_definition(definition)?;
    validate_backend_input(backend, input, context)?;
    require_f32(input)?;
    if definition.preview_factors.is_empty() {
        return Err(LatentFormatError::PreviewUnavailable {
            identifier: definition.identifier.to_owned(),
        });
    }
    validate_standard_input_shape(definition, input.descriptor().shape())?;
    let source_shape = input.descriptor().shape();
    let (preview_channels, output_shape) = match definition.preview_reshape {
        PreviewReshape::None => (
            definition.channels,
            preview_output_shape(source_shape, definition.layout)?,
        ),
        PreviewReshape::Flux2Spatial2x => {
            let batch = source_shape[0];
            let height = source_shape[source_shape.len() - 2]
                .checked_mul(2)
                .ok_or_else(|| invalid_constant(definition, "preview height", "overflow"))?;
            let width = source_shape[source_shape.len() - 1]
                .checked_mul(2)
                .ok_or_else(|| invalid_constant(definition, "preview width", "overflow"))?;
            (32, vec![batch, 3, height, width])
        }
    };
    if usize::try_from(preview_channels).ok() != Some(definition.preview_factors.len()) {
        return Err(invalid_constant(
            definition,
            "preview_factors",
            "channel count mismatch",
        ));
    }
    let descriptor = TensorDescriptor::contiguous(
        output_shape.clone(),
        DType::F32,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let output_count = element_count(&output_shape)?;
    let spatial = output_count
        / (usize::try_from(output_shape[0]).map_err(|_| TensorError::ShapeOverflow)? * 3);
    let batch_count = usize::try_from(output_shape[0]).map_err(|_| TensorError::ShapeOverflow)?;
    let mut write = output.write()?;
    for batch in 0..batch_count {
        for color in 0..3_usize {
            for position in 0..spatial {
                check_periodically(position, context)?;
                let mut value = definition.preview_bias.map_or(0.0, |bias| bias[color]);
                for channel in
                    0..usize::try_from(preview_channels).map_err(|_| TensorError::ShapeOverflow)?
                {
                    let source = preview_source_value(
                        definition,
                        input,
                        batch,
                        channel,
                        position,
                        &output_shape,
                    )?;
                    value = source.mul_add(definition.preview_factors[channel][color], value);
                }
                let index = (batch * 3 + color) * spatial + position;
                write_f32_linear(&mut write, &output_shape, index, value)?;
            }
        }
    }
    drop(write);
    Ok(output)
}

fn process(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    process_in: bool,
) -> Result<Tensor, LatentFormatError> {
    validate_definition(definition)?;
    validate_backend_input(backend, input, context)?;
    if definition.transform == LatentTransform::Identity {
        validate_standard_input_shape(definition, input.descriptor().shape())?;
        return Ok(input.clone());
    }
    require_f32(input)?;
    if definition.transform == LatentTransform::HunyuanImage21Refiner {
        return process_hunyuan_refiner(definition, backend, input, context, process_in);
    }
    validate_standard_input_shape(definition, input.descriptor().shape())?;
    let shape = input.descriptor().shape().to_vec();
    let descriptor = TensorDescriptor::contiguous(
        shape.clone(),
        DType::F32,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let count = element_count(&shape)?;
    let mut write = output.write()?;
    for index in 0..count {
        check_periodically(index, context)?;
        let value = read_f32_linear(input, index)?;
        let channel = channel_for_linear_index(&shape, definition.layout, index)?;
        let (mean, standard_deviation) = match definition.transform {
            LatentTransform::Identity => unreachable!(),
            LatentTransform::Affine => (definition.shift_factor, 1.0),
            LatentTransform::PerChannelAffine => (
                definition.channel_means[channel],
                definition.channel_stds[channel],
            ),
            LatentTransform::HunyuanImage21Refiner => unreachable!(),
        };
        let transformed = if process_in {
            (value - mean) * definition.scale_factor / standard_deviation
        } else {
            value * standard_deviation / definition.scale_factor + mean
        };
        write_f32_linear(&mut write, &shape, index, transformed)?;
    }
    drop(write);
    Ok(output)
}

fn process_hunyuan_refiner(
    definition: &LatentFormatDefinition,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    process_in: bool,
) -> Result<Tensor, LatentFormatError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 {
        return Err(LatentFormatError::InvalidShape {
            identifier: definition.identifier.to_owned(),
            expected: vec![0, definition.channels, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let (output_shape, input_channels, output_channels, output_frames) = if process_in {
        if shape[1] != definition.channels || shape[2] == 0 || shape[2].is_multiple_of(2) {
            return Err(LatentFormatError::InvalidShape {
                identifier: definition.identifier.to_owned(),
                expected: vec![0, definition.channels, 1, 0, 0],
                actual: shape.to_vec(),
            });
        }
        let frames = shape[2].checked_add(1).ok_or(TensorError::ShapeOverflow)? / 2;
        (
            vec![
                shape[0],
                definition.channels * 2,
                frames,
                shape[3],
                shape[4],
            ],
            definition.channels,
            definition.channels * 2,
            frames,
        )
    } else {
        if shape[1] != definition.channels * 2 || shape[2] == 0 {
            return Err(LatentFormatError::InvalidShape {
                identifier: definition.identifier.to_owned(),
                expected: vec![0, definition.channels * 2, 0, 0, 0],
                actual: shape.to_vec(),
            });
        }
        let frames = shape[2]
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(TensorError::ShapeOverflow)?;
        (
            vec![shape[0], definition.channels, frames, shape[3], shape[4]],
            definition.channels * 2,
            definition.channels,
            frames,
        )
    };
    let descriptor = TensorDescriptor::contiguous(
        output_shape.clone(),
        DType::F32,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let count = element_count(&output_shape)?;
    let output_spatial = usize::try_from(
        output_shape[3]
            .checked_mul(output_shape[4])
            .ok_or(TensorError::ShapeOverflow)?,
    )
    .map_err(|_| TensorError::ShapeOverflow)?;
    let output_channels =
        usize::try_from(output_channels).map_err(|_| TensorError::ShapeOverflow)?;
    let output_frames = usize::try_from(output_frames).map_err(|_| TensorError::ShapeOverflow)?;
    let input_channels = usize::try_from(input_channels).map_err(|_| TensorError::ShapeOverflow)?;
    let input_frames = usize::try_from(shape[2]).map_err(|_| TensorError::ShapeOverflow)?;
    let mut write = output.write()?;
    for index in 0..count {
        check_periodically(index, context)?;
        let position = index % output_spatial;
        let frame = (index / output_spatial) % output_frames;
        let channel = (index / output_spatial / output_frames) % output_channels;
        let batch = index / output_spatial / output_frames / output_channels;
        let (source_channel, source_frame) = if process_in {
            let duplicated_frame = frame * 2 + channel / input_channels;
            (channel % input_channels, duplicated_frame.saturating_sub(1))
        } else {
            let expanded_frame = frame + 1;
            (
                channel + (expanded_frame % 2) * output_channels,
                expanded_frame / 2,
            )
        };
        let source_index = ((batch * input_channels + source_channel) * input_frames
            + source_frame)
            * output_spatial
            + position;
        let value = read_f32_linear(input, source_index)?;
        let transformed = if process_in {
            value * definition.scale_factor
        } else {
            value / definition.scale_factor
        };
        write_f32_linear(&mut write, &output_shape, index, transformed)?;
    }
    drop(write);
    Ok(output)
}

fn empty_shape(
    definition: &LatentFormatDefinition,
    extent: LatentExtent,
) -> Result<Vec<u64>, LatentFormatError> {
    let actual_dimensions = match extent {
        LatentExtent::OneDimensional { .. } => 1,
        LatentExtent::TwoDimensional { .. } => 2,
        LatentExtent::ThreeDimensional { .. } => 3,
    };
    if actual_dimensions != definition.dimensions {
        return Err(LatentFormatError::ExtentDimensions {
            identifier: definition.identifier.to_owned(),
            expected: definition.dimensions,
            actual: actual_dimensions,
        });
    }
    let mut shape = match extent {
        LatentExtent::OneDimensional { batch, length } => {
            require_nonzero_extent(definition, "batch", batch)?;
            require_nonzero_extent(definition, "length", length)?;
            shape_for_layout(
                definition.layout,
                batch,
                &[ceil_div(
                    definition,
                    length,
                    definition.temporal_downscale_ratio,
                )?],
            )
        }
        LatentExtent::TwoDimensional {
            batch,
            width,
            height,
        } => {
            require_nonzero_extent(definition, "batch", batch)?;
            let width = spatial_extent(definition, "width", width)?;
            let height = spatial_extent(definition, "height", height)?;
            shape_for_layout(definition.layout, batch, &[height, width])
        }
        LatentExtent::ThreeDimensional {
            batch,
            frames,
            width,
            height,
        } => {
            require_nonzero_extent(definition, "batch", batch)?;
            require_nonzero_extent(definition, "frames", frames)?;
            let width = spatial_extent(definition, "width", width)?;
            let height = spatial_extent(definition, "height", height)?;
            shape_for_layout(
                definition.layout,
                batch,
                &[
                    ceil_div(definition, frames, definition.temporal_downscale_ratio)?,
                    height,
                    width,
                ],
            )
        }
    };
    let channel_index = if definition.layout == LatentTensorLayout::ChannelsFirst {
        1
    } else {
        shape.len() - 1
    };
    shape[channel_index] = definition.channels;
    Ok(shape)
}

fn validate_definition(definition: &LatentFormatDefinition) -> Result<(), LatentFormatError> {
    validate_feature_id(definition.feature_id)?;
    validate_identifier(definition.identifier)?;
    if definition.channels == 0 {
        return Err(invalid_constant(definition, "channels", "0"));
    }
    if !(1..=3).contains(&definition.dimensions) {
        return Err(invalid_constant(
            definition,
            "dimensions",
            definition.dimensions.to_string(),
        ));
    }
    if definition.spatial_downscale_ratio == 0 {
        return Err(invalid_constant(definition, "spatial_downscale_ratio", "0"));
    }
    if definition.temporal_downscale_ratio == 0 {
        return Err(invalid_constant(
            definition,
            "temporal_downscale_ratio",
            "0",
        ));
    }
    if !definition.scale_factor.is_finite() || definition.scale_factor == 0.0 {
        return Err(invalid_constant(
            definition,
            "scale_factor",
            definition.scale_factor.to_string(),
        ));
    }
    if !definition.shift_factor.is_finite() {
        return Err(invalid_constant(
            definition,
            "shift_factor",
            definition.shift_factor.to_string(),
        ));
    }
    if definition.transform == LatentTransform::PerChannelAffine {
        let channels = usize::try_from(definition.channels)
            .map_err(|_| invalid_constant(definition, "channels", "overflow"))?;
        if definition.channel_means.len() != channels || definition.channel_stds.len() != channels {
            return Err(invalid_constant(
                definition,
                "channel affine vectors",
                "channel count mismatch",
            ));
        }
    } else if !definition.channel_means.is_empty() || !definition.channel_stds.is_empty() {
        return Err(invalid_constant(
            definition,
            "channel affine vectors",
            "unexpected values",
        ));
    }
    if definition
        .channel_means
        .iter()
        .any(|value| !value.is_finite())
        || definition
            .channel_stds
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
        || definition
            .preview_factors
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || definition
            .preview_bias
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(invalid_constant(
            definition,
            "floating constants",
            "non-finite or zero standard deviation",
        ));
    }
    if definition.preview_reshape == PreviewReshape::Flux2Spatial2x {
        if definition.channels != 128
            || definition.preview_factors.len() != 32
            || definition.dimensions != 2
        {
            return Err(invalid_constant(
                definition,
                "Flux2 preview reshape",
                "requires 128 input channels, 32 RGB factors, and 2 dimensions",
            ));
        }
    } else if !definition.preview_factors.is_empty()
        && usize::try_from(definition.channels).ok() != Some(definition.preview_factors.len())
    {
        return Err(invalid_constant(
            definition,
            "preview_factors",
            "channel count mismatch",
        ));
    }
    Ok(())
}

fn validate_feature_id(value: &str) -> Result<(), LatentFormatError> {
    let suffix = value
        .strip_prefix("COMFY-MODEL-")
        .ok_or_else(|| LatentFormatError::InvalidFeatureId(value.to_owned()))?;
    if suffix.len() != 4 || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(LatentFormatError::InvalidFeatureId(value.to_owned()));
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), LatentFormatError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(LatentFormatError::InvalidIdentifier(value.to_owned()));
    }
    Ok(())
}

fn validate_backend_input(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), LatentFormatError> {
    context.check()?;
    if backend.device() != input.descriptor().device() {
        return Err(TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: input.descriptor().device(),
        }
        .into());
    }
    if context.stream != input.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: input.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn validate_standard_input_shape(
    definition: &LatentFormatDefinition,
    shape: &[u64],
) -> Result<(), LatentFormatError> {
    let expected_rank = usize::from(definition.dimensions) + 2;
    let channel_matches = match definition.layout {
        LatentTensorLayout::ChannelsFirst => shape.get(1) == Some(&definition.channels),
        LatentTensorLayout::SequenceChannelsLast => shape.last() == Some(&definition.channels),
    };
    if shape.len() != expected_rank
        || shape.first() == Some(&0)
        || !channel_matches
        || shape.contains(&0)
    {
        let mut expected = vec![0; expected_rank];
        match definition.layout {
            LatentTensorLayout::ChannelsFirst => expected[1] = definition.channels,
            LatentTensorLayout::SequenceChannelsLast => {
                expected[expected_rank - 1] = definition.channels
            }
        }
        return Err(LatentFormatError::InvalidShape {
            identifier: definition.identifier.to_owned(),
            expected,
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

fn preview_output_shape(
    shape: &[u64],
    layout: LatentTensorLayout,
) -> Result<Vec<u64>, LatentFormatError> {
    if layout != LatentTensorLayout::ChannelsFirst || shape.len() < 4 {
        return Err(LatentFormatError::PreviewUnavailable {
            identifier: "sequence-layout".to_owned(),
        });
    }
    let mut output = shape.to_vec();
    output[1] = 3;
    Ok(output)
}

fn preview_source_value(
    definition: &LatentFormatDefinition,
    input: &Tensor,
    batch: usize,
    channel: usize,
    position: usize,
    output_shape: &[u64],
) -> Result<f32, LatentFormatError> {
    let source_shape = input.descriptor().shape();
    let source_spatial = element_count(&source_shape[2..])?;
    let source_index = match definition.preview_reshape {
        PreviewReshape::None => {
            (batch
                * usize::try_from(definition.channels).map_err(|_| TensorError::ShapeOverflow)?
                + channel)
                * source_spatial
                + position
        }
        PreviewReshape::Flux2Spatial2x => {
            let output_width =
                usize::try_from(*output_shape.last().ok_or(TensorError::ShapeOverflow)?)
                    .map_err(|_| TensorError::ShapeOverflow)?;
            let output_y = position / output_width;
            let output_x = position % output_width;
            let source_width =
                usize::try_from(*source_shape.last().ok_or(TensorError::ShapeOverflow)?)
                    .map_err(|_| TensorError::ShapeOverflow)?;
            let source_height = usize::try_from(source_shape[source_shape.len() - 2])
                .map_err(|_| TensorError::ShapeOverflow)?;
            let source_channel = channel * 4 + (output_y % 2) * 2 + output_x % 2;
            ((batch * 128 + source_channel) * source_height + output_y / 2) * source_width
                + output_x / 2
        }
    };
    Ok(read_f32_linear(input, source_index)?)
}

fn channel_for_linear_index(
    shape: &[u64],
    layout: LatentTensorLayout,
    index: usize,
) -> Result<usize, LatentFormatError> {
    match layout {
        LatentTensorLayout::ChannelsFirst => {
            let inner = element_count(&shape[2..])?;
            let channels = usize::try_from(shape[1]).map_err(|_| TensorError::ShapeOverflow)?;
            Ok((index / inner) % channels)
        }
        LatentTensorLayout::SequenceChannelsLast => {
            let channels = usize::try_from(*shape.last().ok_or(TensorError::ShapeOverflow)?)
                .map_err(|_| TensorError::ShapeOverflow)?;
            Ok(index % channels)
        }
    }
}

fn read_f32_linear(tensor: &Tensor, index: usize) -> Result<f32, TensorError> {
    let bytes = tensor
        .linear_element_bytes(u64::try_from(index).map_err(|_| TensorError::ShapeOverflow)?)?;
    let value: [u8; 4] = bytes.try_into().map_err(|_| TensorError::DTypeMismatch {
        expected: DType::F32,
        actual: tensor.descriptor().dtype(),
    })?;
    Ok(f32::from_le_bytes(value))
}

fn write_f32_linear(
    write: &mut sim_tensor::TensorWrite<'_>,
    shape: &[u64],
    index: usize,
    value: f32,
) -> Result<(), TensorError> {
    let indices = linear_indices(shape, index)?;
    let destination = write.element_bytes_mut(&indices)?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn linear_indices(shape: &[u64], mut index: usize) -> Result<Vec<u64>, TensorError> {
    let mut indices = vec![0; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let size = usize::try_from(shape[dimension]).map_err(|_| TensorError::ShapeOverflow)?;
        if size == 0 {
            return Err(TensorError::ShapeOverflow);
        }
        indices[dimension] = u64::try_from(index % size).map_err(|_| TensorError::ShapeOverflow)?;
        index /= size;
    }
    Ok(indices)
}

fn require_f32(input: &Tensor) -> Result<(), LatentFormatError> {
    if input.descriptor().dtype() != DType::F32 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: input.descriptor().dtype(),
        }
        .into());
    }
    Ok(())
}

fn element_count(shape: &[u64]) -> Result<usize, TensorError> {
    let count = shape.iter().try_fold(1_u64, |count, value| {
        count.checked_mul(*value).ok_or(TensorError::ShapeOverflow)
    })?;
    usize::try_from(count).map_err(|_| TensorError::ShapeOverflow)
}

fn check_periodically(index: usize, context: &ExecutionContext<'_>) -> Result<(), TensorError> {
    if index.is_multiple_of(64) {
        context.check()?;
    }
    Ok(())
}

fn require_nonzero_extent(
    definition: &LatentFormatDefinition,
    field: &'static str,
    value: u64,
) -> Result<(), LatentFormatError> {
    if value == 0 {
        return Err(LatentFormatError::InvalidExtent {
            identifier: definition.identifier.to_owned(),
            field,
            value,
        });
    }
    Ok(())
}

fn spatial_extent(
    definition: &LatentFormatDefinition,
    field: &'static str,
    value: u64,
) -> Result<u64, LatentFormatError> {
    require_nonzero_extent(definition, field, value)?;
    let output = value / definition.spatial_downscale_ratio;
    if output == 0 {
        return Err(LatentFormatError::InvalidExtent {
            identifier: definition.identifier.to_owned(),
            field,
            value,
        });
    }
    Ok(output)
}

fn shape_for_layout(layout: LatentTensorLayout, batch: u64, dimensions: &[u64]) -> Vec<u64> {
    let mut shape = Vec::with_capacity(dimensions.len() + 2);
    shape.push(batch);
    if layout == LatentTensorLayout::ChannelsFirst {
        shape.push(0);
    }
    shape.extend_from_slice(dimensions);
    if layout == LatentTensorLayout::SequenceChannelsLast {
        shape.push(0);
    }
    shape
}

fn ceil_div(
    definition: &LatentFormatDefinition,
    value: u64,
    divisor: u64,
) -> Result<u64, LatentFormatError> {
    value
        .checked_sub(1)
        .and_then(|value| value.checked_div(divisor))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| LatentFormatError::InvalidExtent {
            identifier: definition.identifier.to_owned(),
            field: "extent",
            value,
        })
}

fn invalid_constant(
    definition: &LatentFormatDefinition,
    field: &'static str,
    value: impl Into<String>,
) -> LatentFormatError {
    LatentFormatError::InvalidConstant {
        identifier: definition.identifier.to_owned(),
        field,
        value: value.into(),
    }
}
