use crate::vae::{VaeError, VaeOperation};
use sim_tensor::{
    BinaryOperation, ExecutionContext, Scalar, ScalarSide, Tensor, TensorBackend, TensorDescriptor,
};

const MAX_TILE_COUNT: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaeTileAxisFormula {
    Linear {
        ratio: u64,
    },
    Causal {
        ratio: u64,
    },
    ResampledCausal {
        original_frequency: u64,
        new_frequency: u64,
        hop_length: u64,
        latent_downsample: u64,
        decode_ratio: u64,
        decode_subtract: u64,
    },
}

impl VaeTileAxisFormula {
    pub fn checked_linear(ratio: u64) -> Result<Self, VaeError> {
        if ratio == 0 {
            return Err(VaeError::InvalidTileScale { ratio });
        }
        Ok(Self::Linear { ratio })
    }

    pub fn checked_causal(ratio: u64) -> Result<Self, VaeError> {
        if ratio == 0 {
            return Err(VaeError::InvalidTileScale { ratio });
        }
        Ok(Self::Causal { ratio })
    }

    pub fn checked_resampled_causal(
        original_frequency: u64,
        new_frequency: u64,
        hop_length: u64,
        latent_downsample: u64,
        decode_ratio: u64,
        decode_subtract: u64,
    ) -> Result<Self, VaeError> {
        if original_frequency == 0
            || new_frequency == 0
            || hop_length == 0
            || latent_downsample == 0
            || decode_ratio == 0
            || decode_subtract >= decode_ratio
        {
            return Err(VaeError::InvalidTileScale {
                ratio: original_frequency
                    .min(new_frequency)
                    .min(hop_length)
                    .min(latent_downsample)
                    .min(decode_ratio),
            });
        }
        Ok(Self::ResampledCausal {
            original_frequency,
            new_frequency,
            hop_length,
            latent_downsample,
            decode_ratio,
            decode_subtract,
        })
    }

    pub fn output_extent(
        self,
        operation: VaeOperation,
        input_extent: u64,
    ) -> Result<u64, VaeError> {
        let extent = match (self, operation) {
            (Self::Linear { ratio }, VaeOperation::Encode) => {
                round_ratio_ties_even(input_extent, ratio)?
            }
            (Self::Linear { ratio }, VaeOperation::Decode) => input_extent
                .checked_mul(ratio)
                .ok_or(VaeError::ShapeOverflow)?,
            (Self::Causal { ratio }, VaeOperation::Encode) => {
                input_extent
                    .checked_add(ratio - 1)
                    .ok_or(VaeError::ShapeOverflow)?
                    / ratio
            }
            (Self::Causal { ratio }, VaeOperation::Decode) => input_extent
                .checked_mul(ratio)
                .and_then(|extent| extent.checked_sub(ratio - 1))
                .ok_or(VaeError::ShapeOverflow)?,
            (
                Self::ResampledCausal {
                    original_frequency,
                    new_frequency,
                    hop_length,
                    latent_downsample,
                    ..
                },
                VaeOperation::Encode,
            ) => {
                let resampled = input_extent
                    .checked_mul(new_frequency)
                    .and_then(|value| value.checked_add(original_frequency - 1))
                    .ok_or(VaeError::ShapeOverflow)?
                    / original_frequency;
                let frames = resampled
                    .checked_div(hop_length)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(VaeError::ShapeOverflow)?;
                frames
                    .checked_add(latent_downsample - 1)
                    .ok_or(VaeError::ShapeOverflow)?
                    / latent_downsample
            }
            (
                Self::ResampledCausal {
                    decode_ratio,
                    decode_subtract,
                    ..
                },
                VaeOperation::Decode,
            ) => input_extent
                .checked_mul(decode_ratio)
                .and_then(|extent| extent.checked_sub(decode_subtract))
                .ok_or(VaeError::ShapeOverflow)?,
        };
        if extent == 0 {
            return Err(VaeError::ZeroTileOutput);
        }
        Ok(extent)
    }

    pub fn output_position(
        self,
        operation: VaeOperation,
        input_position: u64,
    ) -> Result<u64, VaeError> {
        match (self, operation) {
            (Self::ResampledCausal { .. }, VaeOperation::Encode) => {
                if input_position == 0 {
                    Ok(0)
                } else {
                    self.output_extent(VaeOperation::Encode, input_position)?
                        .checked_sub(1)
                        .ok_or(VaeError::ShapeOverflow)
                }
            }
            (Self::Linear { ratio } | Self::Causal { ratio }, VaeOperation::Encode) => {
                round_ratio_ties_even(input_position, ratio)
            }
            (
                Self::Linear { ratio }
                | Self::Causal { ratio }
                | Self::ResampledCausal {
                    decode_ratio: ratio,
                    ..
                },
                VaeOperation::Decode,
            ) => input_position
                .checked_mul(ratio)
                .ok_or(VaeError::ShapeOverflow),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TileTensorLayout {
    ChannelsFirst,
    SequenceChannelsLast,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TilePass {
    tile_extent: Vec<u64>,
    overlap: Vec<u64>,
    tile_counts: Vec<u64>,
    tile_count: u64,
}

impl TilePass {
    fn checked(
        input_spatial_shape: &[u64],
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
    ) -> Result<Self, VaeError> {
        if tile_extent.len() != input_spatial_shape.len()
            || overlap.len() != input_spatial_shape.len()
        {
            return Err(VaeError::TileRank {
                expected: input_spatial_shape.len(),
                tile_extent: tile_extent.len(),
                overlap: overlap.len(),
            });
        }
        let mut tile_counts = Vec::new();
        tile_counts
            .try_reserve_exact(input_spatial_shape.len())
            .map_err(|error| VaeError::Allocation(error.to_string()))?;
        let mut tile_count = 1_u64;
        for (dimension, ((&size, &extent), &dimension_overlap)) in input_spatial_shape
            .iter()
            .zip(&tile_extent)
            .zip(&overlap)
            .enumerate()
        {
            if size == 0 || extent == 0 || dimension_overlap >= extent {
                return Err(VaeError::InvalidTileDimension {
                    dimension,
                    size,
                    extent,
                    overlap: dimension_overlap,
                });
            }
            let stride = extent - dimension_overlap;
            let count = if size <= extent {
                1
            } else {
                (size - dimension_overlap).div_ceil(stride)
            };
            tile_count = tile_count
                .checked_mul(count)
                .ok_or(VaeError::ShapeOverflow)?;
            if tile_count > MAX_TILE_COUNT {
                return Err(VaeError::TooManyTiles(tile_count));
            }
            tile_counts.push(count);
        }
        Ok(Self {
            tile_extent,
            overlap,
            tile_counts,
            tile_count,
        })
    }

    fn tile_bounds(
        &self,
        input_spatial_shape: &[u64],
        tile_index: u64,
    ) -> Result<(Vec<u64>, Vec<u64>), VaeError> {
        if tile_index >= self.tile_count {
            return Err(VaeError::ShapeOverflow);
        }
        let mut remainder = tile_index;
        let mut starts = vec![0; self.tile_counts.len()];
        let mut lengths = vec![0; self.tile_counts.len()];
        for dimension in (0..self.tile_counts.len()).rev() {
            let coordinate = remainder % self.tile_counts[dimension];
            remainder /= self.tile_counts[dimension];
            let stride = self.tile_extent[dimension] - self.overlap[dimension];
            let candidate = coordinate
                .checked_mul(stride)
                .ok_or(VaeError::ShapeOverflow)?;
            let maximum_start =
                input_spatial_shape[dimension].saturating_sub(self.overlap[dimension]);
            let start = candidate.min(maximum_start);
            let length = self.tile_extent[dimension].min(input_spatial_shape[dimension] - start);
            starts[dimension] = start;
            lengths[dimension] = length;
        }
        Ok((starts, lengths))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TileExecutionPlan {
    operation: VaeOperation,
    input_spatial_shape: Vec<u64>,
    input_spatial_offsets: Vec<u64>,
    output_spatial_shape: Vec<u64>,
    formulas: Vec<VaeTileAxisFormula>,
    input_layout: TileTensorLayout,
    output_layout: TileTensorLayout,
    passes: Vec<TilePass>,
    tile_count: u64,
    preserve_batch_group: bool,
}

impl TileExecutionPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        operation: VaeOperation,
        input_spatial_shape: Vec<u64>,
        input_spatial_offsets: Vec<u64>,
        output_spatial_shape: Vec<u64>,
        formulas: Vec<VaeTileAxisFormula>,
        input_layout: TileTensorLayout,
        output_layout: TileTensorLayout,
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
        three_pass_2d: bool,
        preserve_batch_group: bool,
    ) -> Result<Self, VaeError> {
        let dimensions = input_spatial_shape.len();
        if dimensions == 0
            || dimensions > 3
            || input_spatial_offsets.len() != dimensions
            || output_spatial_shape.len() != dimensions
            || formulas.len() != dimensions
            || tile_extent.len() != dimensions
            || overlap.len() != dimensions
        {
            return Err(VaeError::TileRank {
                expected: dimensions,
                tile_extent: tile_extent.len(),
                overlap: overlap.len(),
            });
        }
        for (formula, (&input, &expected)) in formulas
            .iter()
            .zip(input_spatial_shape.iter().zip(&output_spatial_shape))
        {
            let actual = formula.output_extent(operation, input)?;
            if actual != expected {
                return Err(VaeError::TileOutputGeometryMismatch {
                    expected: output_spatial_shape.clone(),
                    actual: formulas
                        .iter()
                        .zip(&input_spatial_shape)
                        .map(|(formula, input)| formula.output_extent(operation, *input))
                        .collect::<Result<Vec<_>, _>>()?,
                });
            }
        }

        let mut pass_extents = vec![tile_extent.clone()];
        if three_pass_2d {
            if dimensions != 2 {
                return Err(VaeError::ThreePassTileRank(dimensions));
            }
            let short_height = tile_extent[0] / 2;
            let short_width = tile_extent[1] / 2;
            let tall_height = tile_extent[0]
                .checked_mul(2)
                .ok_or(VaeError::ShapeOverflow)?;
            let wide_width = tile_extent[1]
                .checked_mul(2)
                .ok_or(VaeError::ShapeOverflow)?;
            match operation {
                VaeOperation::Encode => {
                    pass_extents.push(vec![short_height, wide_width]);
                    pass_extents.push(vec![tall_height, short_width]);
                }
                VaeOperation::Decode => {
                    pass_extents.clear();
                    pass_extents.push(vec![tall_height, short_width]);
                    pass_extents.push(vec![short_height, wide_width]);
                    pass_extents.push(tile_extent);
                }
            }
        }

        let mut passes = Vec::new();
        passes
            .try_reserve_exact(pass_extents.len())
            .map_err(|error| VaeError::Allocation(error.to_string()))?;
        let mut tile_count = 0_u64;
        for extent in pass_extents {
            let pass = TilePass::checked(&input_spatial_shape, extent, overlap.clone())?;
            if pass.tile_count > 1
                && formulas
                    .iter()
                    .any(|formula| matches!(formula, VaeTileAxisFormula::ResampledCausal { .. }))
            {
                return Err(VaeError::PhaseSensitiveTileRequiresWholeInput);
            }
            tile_count = tile_count
                .checked_add(pass.tile_count)
                .ok_or(VaeError::ShapeOverflow)?;
            if tile_count > MAX_TILE_COUNT {
                return Err(VaeError::TooManyTiles(tile_count));
            }
            passes.push(pass);
        }
        Ok(Self {
            operation,
            input_spatial_shape,
            input_spatial_offsets,
            output_spatial_shape,
            formulas,
            input_layout,
            output_layout,
            passes,
            tile_count,
            preserve_batch_group,
        })
    }

    pub(crate) fn tile_extent(&self) -> &[u64] {
        match (self.operation, self.passes.as_slice()) {
            (VaeOperation::Decode, [_, _, base]) => &base.tile_extent,
            (_, [base, ..]) => &base.tile_extent,
            _ => &[],
        }
    }

    pub(crate) fn input_spatial_shape(&self) -> &[u64] {
        &self.input_spatial_shape
    }

    pub(crate) fn input_spatial_offsets(&self) -> &[u64] {
        &self.input_spatial_offsets
    }

    pub(crate) fn overlap(&self) -> &[u64] {
        self.passes
            .first()
            .map_or(&[], |pass| pass.overlap.as_slice())
    }

    pub(crate) fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub(crate) fn tile_count(&self) -> u64 {
        self.tile_count
    }
}

pub(crate) fn execute_tiled_scale<Kernel>(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_shape: &[u64],
    output_channels: u64,
    plan: &TileExecutionPlan,
    mut kernel: Kernel,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError>
where
    Kernel: FnMut(&Tensor, &[u64], &ExecutionContext<'_>) -> Result<Tensor, VaeError>,
{
    context.check()?;
    let mut sum: Option<Tensor> = None;
    for pass in &plan.passes {
        let pass_output = execute_pass(
            backend,
            input,
            output_shape,
            output_channels,
            plan,
            pass,
            &mut kernel,
            context,
        )?;
        sum = Some(match sum {
            None => pass_output,
            Some(previous) => {
                let (next, event) = backend.binary(
                    BinaryOperation::Add,
                    &previous,
                    &pass_output,
                    previous.descriptor().clone(),
                    context,
                )?;
                backend.wait_event(event, context)?;
                next
            }
        });
    }
    let sum = sum.ok_or(VaeError::NoTilePasses)?;
    if plan.passes.len() == 1 {
        return Ok(sum);
    }
    let (averaged, event) = backend.binary_scalar(
        BinaryOperation::Divide,
        &sum,
        Scalar::Float(plan.passes.len() as f64),
        ScalarSide::Right,
        sum.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(averaged)
}

#[allow(clippy::too_many_arguments)]
fn execute_pass<Kernel>(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_shape: &[u64],
    output_channels: u64,
    plan: &TileExecutionPlan,
    pass: &TilePass,
    kernel: &mut Kernel,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError>
where
    Kernel: FnMut(&Tensor, &[u64], &ExecutionContext<'_>) -> Result<Tensor, VaeError>,
{
    let descriptor = TensorDescriptor::contiguous(
        output_shape.to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let (mut accumulated, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let mut divisor = if pass.tile_count == 1 {
        None
    } else {
        let divisor_shape = shape_with_channels(
            output_shape[0],
            1,
            &plan.output_spatial_shape,
            plan.output_layout,
        );
        let divisor_descriptor = TensorDescriptor::contiguous(
            divisor_shape,
            input.descriptor().dtype(),
            input.descriptor().device(),
            input.descriptor().stream(),
        )?;
        let (divisor, event) = backend.fill(Scalar::Float(0.0), divisor_descriptor, context)?;
        backend.wait_event(event, context)?;
        Some(divisor)
    };

    let input_batch = input.descriptor().shape()[0];
    let batch_groups = if plan.preserve_batch_group {
        1
    } else {
        input_batch
    };
    for batch_group in 0..batch_groups {
        let (batch_start, batch_length) = if plan.preserve_batch_group {
            (0, input_batch)
        } else {
            (batch_group, 1)
        };
        for tile_index in 0..pass.tile_count {
            context.check()?;
            let (starts, lengths) = pass.tile_bounds(&plan.input_spatial_shape, tile_index)?;
            let source_starts = starts
                .iter()
                .zip(&plan.input_spatial_offsets)
                .map(|(start, offset)| start.checked_add(*offset).ok_or(VaeError::ShapeOverflow))
                .collect::<Result<Vec<_>, _>>()?;
            let input_tile = narrow_tile(
                input,
                plan.input_layout,
                batch_start,
                batch_length,
                &source_starts,
                &lengths,
            )?;
            let mut output_starts = Vec::with_capacity(starts.len());
            let mut expected_spatial = Vec::with_capacity(lengths.len());
            for (dimension, ((formula, start), length)) in
                plan.formulas.iter().zip(&starts).zip(&lengths).enumerate()
            {
                let output_start = formula.output_position(plan.operation, *start)?;
                let maximum = plan.output_spatial_shape[dimension];
                if output_start >= maximum {
                    return Err(VaeError::TileOutputOutOfBounds {
                        dimension,
                        start: output_start,
                        size: maximum,
                    });
                }
                output_starts.push(output_start);
                expected_spatial.push(formula.output_extent(plan.operation, *length)?);
            }
            let produced = kernel(&input_tile, &expected_spatial, context)?;
            validate_tile_output(
                &produced,
                output_channels,
                &expected_spatial,
                plan.output_layout,
                batch_length,
                input,
            )?;
            let cropped_lengths = expected_spatial
                .iter()
                .zip(&output_starts)
                .zip(&plan.output_spatial_shape)
                .map(|((&length, &start), &maximum)| length.min(maximum - start))
                .collect::<Vec<_>>();
            let produced = crop_spatial(&produced, plan.output_layout, &cropped_lengths)?;
            let offsets = tensor_offsets(batch_start, 0, &output_starts, plan.output_layout);
            if pass.tile_count == 1 {
                let (updated, event) = backend.replace_rectangular_slice(
                    &accumulated,
                    &produced,
                    &offsets,
                    context,
                )?;
                backend.wait_event(event, context)?;
                accumulated = updated;
                continue;
            }
            let mask = feather_mask(
                backend,
                input,
                &cropped_lengths,
                &pass.overlap,
                &plan.formulas,
                plan.operation,
                plan.output_layout,
                context,
            )?;
            let (weighted, event) = backend.binary(
                BinaryOperation::Multiply,
                &produced,
                &mask,
                produced.descriptor().clone(),
                context,
            )?;
            backend.wait_event(event, context)?;

            let accumulated_slice = narrow_tile(
                &accumulated,
                plan.output_layout,
                batch_start,
                batch_length,
                &output_starts,
                &cropped_lengths,
            )?;
            let (updated_slice, event) = backend.binary(
                BinaryOperation::Add,
                &accumulated_slice,
                &weighted,
                accumulated_slice.descriptor().clone(),
                context,
            )?;
            backend.wait_event(event, context)?;
            let (updated, event) = backend.replace_rectangular_slice(
                &accumulated,
                &updated_slice,
                &offsets,
                context,
            )?;
            backend.wait_event(event, context)?;
            accumulated = updated;

            let Some(current_divisor) = divisor.as_ref() else {
                return Err(VaeError::NoTilePasses);
            };
            let divisor_slice = narrow_tile(
                current_divisor,
                plan.output_layout,
                batch_start,
                batch_length,
                &output_starts,
                &cropped_lengths,
            )?;
            let (updated_divisor_slice, event) = backend.binary(
                BinaryOperation::Add,
                &divisor_slice,
                &mask,
                divisor_slice.descriptor().clone(),
                context,
            )?;
            backend.wait_event(event, context)?;
            let (updated, event) = backend.replace_rectangular_slice(
                current_divisor,
                &updated_divisor_slice,
                &offsets,
                context,
            )?;
            backend.wait_event(event, context)?;
            divisor = Some(updated);
        }
    }
    let Some(divisor) = divisor else {
        return Ok(accumulated);
    };
    let (normalized, event) = backend.binary(
        BinaryOperation::Divide,
        &accumulated,
        &divisor,
        accumulated.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(normalized)
}

#[allow(clippy::too_many_arguments)]
fn feather_mask(
    backend: &dyn TensorBackend,
    reference: &Tensor,
    spatial_shape: &[u64],
    overlap: &[u64],
    formulas: &[VaeTileAxisFormula],
    operation: VaeOperation,
    layout: TileTensorLayout,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let element_count = checked_product(spatial_shape)?;
    let byte_count = element_count
        .checked_mul(4)
        .ok_or(VaeError::ShapeOverflow)?;
    let _workspace = backend.reserve_workspace(context, byte_count)?;
    let capacity = usize::try_from(element_count).map_err(|_| VaeError::ShapeOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for linear in 0..element_count {
        if linear % 1_024 == 0 {
            context.check()?;
        }
        let coordinates = coordinates_for_shape(linear, spatial_shape)?;
        let mut weight = 1.0_f32;
        for dimension in 0..spatial_shape.len() {
            let feather = formulas[dimension]
                .output_extent(operation, overlap[dimension])
                .unwrap_or(0);
            if feather >= spatial_shape[dimension] || feather == 0 {
                continue;
            }
            let coordinate = coordinates[dimension];
            if coordinate < feather {
                weight *= (coordinate + 1) as f32 / feather as f32;
            }
            let from_end = spatial_shape[dimension] - 1 - coordinate;
            if from_end < feather {
                weight *= (from_end + 1) as f32 / feather as f32;
            }
        }
        values.push(weight);
    }
    let shape = shape_with_channels(1, 1, spatial_shape, layout);
    let (mask, event) =
        backend.upload_f32_payload(&shape, &values, reference.descriptor().dtype(), context)?;
    backend.wait_event(event, context)?;
    Ok(mask)
}

fn validate_tile_output(
    output: &Tensor,
    output_channels: u64,
    expected_spatial: &[u64],
    layout: TileTensorLayout,
    expected_batch: u64,
    input: &Tensor,
) -> Result<(), VaeError> {
    let expected_shape =
        shape_with_channels(expected_batch, output_channels, expected_spatial, layout);
    if output.descriptor().shape() != expected_shape
        || output.descriptor().dtype() != input.descriptor().dtype()
        || output.descriptor().device() != input.descriptor().device()
        || output.descriptor().stream() != input.descriptor().stream()
    {
        return Err(VaeError::TileKernelOutputMismatch {
            expected_shape,
            actual_shape: output.descriptor().shape().to_vec(),
        });
    }
    Ok(())
}

fn narrow_tile(
    input: &Tensor,
    layout: TileTensorLayout,
    batch_start: u64,
    batch_length: u64,
    starts: &[u64],
    lengths: &[u64],
) -> Result<Tensor, VaeError> {
    let mut tile = input.narrow_read_only(0, i64::try_from(batch_start)?, batch_length)?;
    for dimension in 0..starts.len() {
        let axis = spatial_axis(tile.descriptor().rank(), dimension, layout)?;
        tile =
            tile.narrow_read_only(axis, i64::try_from(starts[dimension])?, lengths[dimension])?;
    }
    Ok(tile)
}

fn crop_spatial(
    input: &Tensor,
    layout: TileTensorLayout,
    lengths: &[u64],
) -> Result<Tensor, VaeError> {
    let mut cropped = input.clone();
    for (dimension, &length) in lengths.iter().enumerate() {
        let axis = spatial_axis(cropped.descriptor().rank(), dimension, layout)?;
        if cropped.descriptor().shape()[axis] != length {
            cropped = cropped.narrow_read_only(axis, 0, length)?;
        }
    }
    Ok(cropped)
}

fn spatial_axis(
    rank: usize,
    dimension: usize,
    layout: TileTensorLayout,
) -> Result<usize, VaeError> {
    let axis = match layout {
        TileTensorLayout::ChannelsFirst => dimension.checked_add(2),
        TileTensorLayout::SequenceChannelsLast => dimension.checked_add(1),
    }
    .ok_or(VaeError::ShapeOverflow)?;
    if axis >= rank {
        return Err(VaeError::ShapeOverflow);
    }
    Ok(axis)
}

fn shape_with_channels(
    batch: u64,
    channels: u64,
    spatial: &[u64],
    layout: TileTensorLayout,
) -> Vec<u64> {
    let mut shape = Vec::with_capacity(spatial.len() + 2);
    shape.push(batch);
    match layout {
        TileTensorLayout::ChannelsFirst => {
            shape.push(channels);
            shape.extend_from_slice(spatial);
        }
        TileTensorLayout::SequenceChannelsLast => {
            shape.extend_from_slice(spatial);
            shape.push(channels);
        }
    }
    shape
}

fn tensor_offsets(batch: u64, channel: u64, spatial: &[u64], layout: TileTensorLayout) -> Vec<u64> {
    shape_with_channels(batch, channel, spatial, layout)
}

fn checked_product(shape: &[u64]) -> Result<u64, VaeError> {
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(VaeError::ShapeOverflow)
    })
}

fn coordinates_for_shape(mut linear: u64, shape: &[u64]) -> Result<Vec<u64>, VaeError> {
    let mut coordinates = vec![0; shape.len()];
    for dimension in (0..shape.len()).rev() {
        if shape[dimension] == 0 {
            return Err(VaeError::ShapeOverflow);
        }
        coordinates[dimension] = linear % shape[dimension];
        linear /= shape[dimension];
    }
    Ok(coordinates)
}

fn round_ratio_ties_even(value: u64, divisor: u64) -> Result<u64, VaeError> {
    if divisor == 0 {
        return Err(VaeError::InvalidTileScale { ratio: divisor });
    }
    let quotient = value / divisor;
    let remainder = value % divisor;
    let doubled = remainder.checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    if doubled > divisor || (doubled == divisor && quotient % 2 == 1) {
        quotient.checked_add(1).ok_or(VaeError::ShapeOverflow)
    } else {
        Ok(quotient)
    }
}

impl From<std::num::TryFromIntError> for VaeError {
    fn from(_: std::num::TryFromIntError) -> Self {
        Self::ShapeOverflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, StreamId,
    };

    fn context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
        workspace: u64,
    ) -> Result<ExecutionContext<'a>, VaeError> {
        Ok(ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(workspace)?,
            rng_phase: None,
            cancellation,
        })
    }

    fn identity_plan(batch: u64) -> Result<(TileExecutionPlan, Vec<u64>), VaeError> {
        let output_shape = vec![batch, 1, 4];
        Ok((
            TileExecutionPlan::checked(
                VaeOperation::Encode,
                vec![4],
                vec![0],
                vec![4],
                vec![VaeTileAxisFormula::checked_linear(1)?],
                TileTensorLayout::ChannelsFirst,
                TileTensorLayout::ChannelsFirst,
                vec![2],
                vec![1],
                false,
                false,
            )?,
            output_shape,
        ))
    }

    fn upload(
        backend: &CpuBackend,
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let (tensor, event) =
            backend.upload_f32_payload(&[2, 1, 4], values, DType::F32, context)?;
        backend.wait_event(event, context)?;
        Ok(tensor)
    }

    fn copy_tile(
        backend: &CpuBackend,
        tile: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let descriptor = TensorDescriptor::contiguous(
            tile.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (copied, event) = backend.copy(tile, descriptor, context)?;
        backend.wait_event(event, context)?;
        Ok(copied)
    }

    #[test]
    fn val_vae_001_source_rounding_and_causal_formulas_are_exact() -> Result<(), VaeError> {
        let linear = VaeTileAxisFormula::checked_linear(4)?;
        assert!(matches!(
            linear.output_extent(VaeOperation::Encode, 2),
            Err(VaeError::ZeroTileOutput)
        ));
        assert_eq!(round_ratio_ties_even(2, 4)?, 0);
        assert_eq!(round_ratio_ties_even(6, 4)?, 2);
        assert_eq!(linear.output_extent(VaeOperation::Decode, 3)?, 12);

        let causal = VaeTileAxisFormula::checked_causal(4)?;
        assert_eq!(causal.output_extent(VaeOperation::Encode, 1)?, 1);
        assert_eq!(causal.output_extent(VaeOperation::Encode, 5)?, 2);
        assert_eq!(causal.output_extent(VaeOperation::Decode, 2)?, 5);
        assert_eq!(causal.output_position(VaeOperation::Decode, 2)?, 8);

        let ltx_audio =
            VaeTileAxisFormula::checked_resampled_causal(44_100, 16_000, 160, 4, 640, 480)?;
        assert_eq!(ltx_audio.output_extent(VaeOperation::Encode, 1_763)?, 2);
        assert_eq!(ltx_audio.output_extent(VaeOperation::Encode, 1_764)?, 2);
        assert_eq!(ltx_audio.output_extent(VaeOperation::Encode, 44_100)?, 26);
        assert_eq!(ltx_audio.output_extent(VaeOperation::Decode, 26)?, 16_160);
        assert_eq!(ltx_audio.output_position(VaeOperation::Encode, 1_764)?, 1);
        assert_eq!(ltx_audio.output_position(VaeOperation::Decode, 2)?, 1_280);
        assert!(matches!(
            TileExecutionPlan::checked(
                VaeOperation::Encode,
                vec![44_100],
                vec![0],
                vec![26],
                vec![ltx_audio],
                TileTensorLayout::ChannelsFirst,
                TileTensorLayout::ChannelsFirst,
                vec![22_050],
                vec![1_764],
                false,
                false,
            ),
            Err(VaeError::PhaseSensitiveTileRequiresWholeInput)
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_source_three_pass_order_is_exact() -> Result<(), VaeError> {
        let formula = VaeTileAxisFormula::checked_linear(1)?;
        let encode = TileExecutionPlan::checked(
            VaeOperation::Encode,
            vec![32, 48],
            vec![0, 0],
            vec![32, 48],
            vec![formula, formula],
            TileTensorLayout::ChannelsFirst,
            TileTensorLayout::ChannelsFirst,
            vec![16, 24],
            vec![2, 2],
            true,
            false,
        )?;
        assert_eq!(encode.passes[0].tile_extent, [16, 24]);
        assert_eq!(encode.passes[1].tile_extent, [8, 48]);
        assert_eq!(encode.passes[2].tile_extent, [32, 12]);

        let decode = TileExecutionPlan::checked(
            VaeOperation::Decode,
            vec![32, 48],
            vec![0, 0],
            vec![32, 48],
            vec![formula, formula],
            TileTensorLayout::ChannelsFirst,
            TileTensorLayout::ChannelsFirst,
            vec![16, 24],
            vec![2, 2],
            true,
            false,
        )?;
        assert_eq!(decode.passes[0].tile_extent, [32, 12]);
        assert_eq!(decode.passes[1].tile_extent, [8, 48]);
        assert_eq!(decode.passes[2].tile_extent, [16, 24]);
        Ok(())
    }

    #[test]
    fn val_vae_001_tiled_scale_executes_each_batch_tile_and_normalizes_feathers()
    -> Result<(), VaeError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let context = context(&authority, &cancellation, 1 << 20)?;
        let input_values = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let input = upload(&backend, &input_values, &context)?;
        let (plan, output_shape) = identity_plan(2)?;
        let mut calls = 0_u64;
        let output = execute_tiled_scale(
            &backend,
            &input,
            &output_shape,
            1,
            &plan,
            |tile, _, context| {
                calls += 1;
                copy_tile(&backend, tile, context)
            },
            &context,
        )?;
        assert_eq!(calls, 6);
        assert_eq!(output.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        let first_peak = context.scratch.peak_bytes();
        drop(output);

        let second = execute_tiled_scale(
            &backend,
            &input,
            &output_shape,
            1,
            &plan,
            |tile, _, context| copy_tile(&backend, tile, context),
            &context,
        )?;
        assert_eq!(second.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(context.scratch.peak_bytes(), first_peak);
        Ok(())
    }

    #[test]
    fn val_vae_001_tiled_scale_can_preserve_a_temporal_batch_group() -> Result<(), VaeError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let context = context(&authority, &cancellation, 1 << 20)?;
        let input_values = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let input = upload(&backend, &input_values, &context)?;
        let output_shape = vec![2, 1, 4];
        let plan = TileExecutionPlan::checked(
            VaeOperation::Decode,
            vec![4],
            vec![0],
            vec![4],
            vec![VaeTileAxisFormula::checked_linear(1)?],
            TileTensorLayout::ChannelsFirst,
            TileTensorLayout::ChannelsFirst,
            vec![2],
            vec![1],
            false,
            true,
        )?;
        let mut calls = 0_u64;
        let output = execute_tiled_scale(
            &backend,
            &input,
            &output_shape,
            1,
            &plan,
            |tile, _, context| {
                calls += 1;
                assert_eq!(tile.descriptor().shape()[0], 2);
                copy_tile(&backend, tile, context)
            },
            &context,
        )?;
        assert_eq!(calls, 3);
        assert_eq!(output.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn val_vae_001_single_tile_uses_the_source_direct_assignment_path() -> Result<(), VaeError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let upload_context = context(&authority, &cancellation, 1 << 20)?;
        let input = upload(
            &backend,
            &[0.1, -0.2, 0.3, -0.4, 1.1, -1.2, 1.3, -1.4],
            &upload_context,
        )?;
        let output_shape = vec![2, 1, 4];
        let plan = TileExecutionPlan::checked(
            VaeOperation::Encode,
            vec![4],
            vec![0],
            vec![4],
            vec![VaeTileAxisFormula::checked_linear(1)?],
            TileTensorLayout::ChannelsFirst,
            TileTensorLayout::ChannelsFirst,
            vec![4],
            vec![3],
            false,
            false,
        )?;
        let exact_context = context(&authority, &cancellation, 24)?;
        let mut calls = 0_u64;
        let output = execute_tiled_scale(
            &backend,
            &input,
            &output_shape,
            1,
            &plan,
            |tile, _, context| {
                calls += 1;
                copy_tile(&backend, tile, context)
            },
            &exact_context,
        )?;
        assert_eq!(calls, 2);
        assert_eq!(output.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(exact_context.scratch.in_use_bytes(), 0);
        assert_eq!(exact_context.scratch.peak_bytes(), 24);
        Ok(())
    }

    #[test]
    fn val_vae_001_cancellation_at_every_tile_and_underauthorization_publish_nothing()
    -> Result<(), VaeError> {
        let input_values = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let (plan, output_shape) = identity_plan(2)?;
        for cancel_at in 0..plan.tile_count() {
            let cancellation = CancellationToken::default();
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
            let context = context(&authority, &cancellation, 1 << 20)?;
            let input = upload(&backend, &input_values, &context)?;
            let mut calls = 0_u64;
            let result = execute_tiled_scale(
                &backend,
                &input,
                &output_shape,
                1,
                &plan,
                |tile, _, context| {
                    if calls == cancel_at {
                        cancellation.cancel();
                    }
                    calls += 1;
                    context.check()?;
                    copy_tile(&backend, tile, context)
                },
                &context,
            );
            assert!(matches!(
                result,
                Err(VaeError::Tensor(sim_tensor::TensorError::Cancelled))
            ));
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }

        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let upload_context = context(&authority, &cancellation, 1 << 20)?;
        let input = upload(&backend, &input_values, &upload_context)?;
        let constrained = context(&authority, &cancellation, 4)?;
        let result = execute_tiled_scale(
            &backend,
            &input,
            &output_shape,
            1,
            &plan,
            |tile, _, context| copy_tile(&backend, tile, context),
            &constrained,
        );
        assert!(matches!(
            result,
            Err(VaeError::Tensor(
                sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(constrained.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
