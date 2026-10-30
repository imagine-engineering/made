use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, Layout, StreamId, Tensor, TensorDescriptor, TensorError, ViewAccess,
    generated_sim_operator_indirection_01::{
        ConvolutionPaddingMode, map_padded_coordinate,
    },
    generated_elementwise_or_runtime_operation_17::{
        TensorSplitSpec, tensor_split_exact_native, tensor_split_jvp_exact_native,
        tensor_split_vjp_with_context_exact_native,
    },
    generated_shape_layout_transform_01::reshape_with_context_for_operation,
    generated_shape_layout_transform_02::{
        permute_for_operation, permute_vjp_for_operation,
    },
};
use thiserror::Error;

pub const TENSOR_PERMUTE_OPERATION_ID: &str = "COMFY-TENSOR-OP-FC2184DD3E3D";
pub const TENSOR_SPLIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-E1ADB7F2ED49";
pub const TENSOR_SQUEEZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-ED0C08B31410";
pub const TENSOR_TRANSPOSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-DF605ED35114";
pub const FUNCTIONAL_PAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-E867958E2F71";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FunctionalPadMode {
    #[default]
    Constant,
    Reflect,
    Replicate,
    Circular,
}

#[derive(Debug, Error)]
pub enum ShapeLayoutTransformPartThreeError {
    #[error("{0}")]
    Tensor(#[from] #[source] TensorError),
    #[error("shape/layout-transform part-three execution was cancelled")]
    Cancelled,
    #[error("operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} does not support dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed for operation {operation} while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
    #[error("operation {operation} failed in its canonical owner: {reason}")]
    CanonicalOwner {
        operation: &'static str,
        reason: String,
    },
}

impl From<sim_types::CancellationError> for ShapeLayoutTransformPartThreeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn tensor_permute_exact_native(
    input: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    permute_for_operation(input, dimensions, TENSOR_PERMUTE_OPERATION_ID, cancellation)
        .map_err(|error| canonical_error(TENSOR_PERMUTE_OPERATION_ID, error))
}

pub fn permute_vjp_exact_native(
    output_gradient: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    permute_vjp_for_operation(
        output_gradient,
        dimensions,
        TENSOR_PERMUTE_OPERATION_ID,
        cancellation,
    )
    .map_err(|error| canonical_error(TENSOR_PERMUTE_OPERATION_ID, error))
}

pub fn permute_jvp_exact_native(
    input_tangent: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    tensor_permute_exact_native(input_tangent, dimensions, cancellation)
}

pub fn tensor_split_exact_native_part_three(
    input: &Tensor,
    specification: &TensorSplitSpec,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    tensor_split_exact_native(input, specification, dimension, cancellation)
        .map_err(|error| canonical_error(TENSOR_SPLIT_OPERATION_ID, error))
}

pub fn split_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradients: &[Tensor],
    specification: &TensorSplitSpec,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    tensor_split_vjp_with_context_exact_native(
        backend,
        input,
        output_gradients,
        specification,
        dimension,
        context,
    )
    .map_err(|error| canonical_error(TENSOR_SPLIT_OPERATION_ID, error))
}

pub fn split_jvp_exact_native(
    input_tangent: &Tensor,
    specification: &TensorSplitSpec,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    tensor_split_jvp_exact_native(input_tangent, specification, dimension, cancellation)
        .map_err(|error| canonical_error(TENSOR_SPLIT_OPERATION_ID, error))
}

pub fn tensor_squeeze_exact_native(
    input: &Tensor,
    dimensions: Option<&[i64]>,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    squeeze_for_operation(input, dimensions, TENSOR_SQUEEZE_OPERATION_ID, cancellation)
}

pub fn squeeze_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    let shape = input_shape
        .iter()
        .map(|dimension| {
            i64::try_from(*dimension)
                .map_err(|_| overflow(TENSOR_SQUEEZE_OPERATION_ID, "input shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    reshape_with_context_for_operation(
        backend,
        output_gradient,
        &shape,
        TENSOR_SQUEEZE_OPERATION_ID,
        context,
    )
    .map_err(|error| canonical_error(TENSOR_SQUEEZE_OPERATION_ID, error))
}

pub fn squeeze_jvp_exact_native(
    input_tangent: &Tensor,
    dimensions: Option<&[i64]>,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    squeeze_for_operation(
        input_tangent,
        dimensions,
        TENSOR_SQUEEZE_OPERATION_ID,
        cancellation,
    )
}

pub fn tensor_transpose_exact_native(
    input: &Tensor,
    first_dimension: i64,
    second_dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    let rank = input.descriptor().rank();
    let first = normalize_axis(first_dimension, rank, TENSOR_TRANSPOSE_OPERATION_ID)?;
    let second = normalize_axis(second_dimension, rank, TENSOR_TRANSPOSE_OPERATION_ID)?;
    let mut permutation = (0..rank)
        .map(|axis| {
            i64::try_from(axis)
                .map_err(|_| overflow(TENSOR_TRANSPOSE_OPERATION_ID, "permutation"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    permutation.swap(first, second);
    permute_for_operation(
        input,
        &permutation,
        TENSOR_TRANSPOSE_OPERATION_ID,
        cancellation,
    )
    .map_err(|error| canonical_error(TENSOR_TRANSPOSE_OPERATION_ID, error))
}

pub fn transpose_vjp_exact_native(
    output_gradient: &Tensor,
    first_dimension: i64,
    second_dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    tensor_transpose_exact_native(
        output_gradient,
        first_dimension,
        second_dimension,
        cancellation,
    )
}

pub fn transpose_jvp_exact_native(
    input_tangent: &Tensor,
    first_dimension: i64,
    second_dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    tensor_transpose_exact_native(
        input_tangent,
        first_dimension,
        second_dimension,
        cancellation,
    )
}

pub fn functional_pad_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    padding: &[i64],
    mode: FunctionalPadMode,
    value: Option<DecodedScalar>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    require_cpu(input, FUNCTIONAL_PAD_OPERATION_ID)?;
    if mode != FunctionalPadMode::Constant && value.is_some() {
        return invalid(
            FUNCTIONAL_PAD_OPERATION_ID,
            "a padding value is valid only in constant mode",
        );
    }
    let geometry = PadGeometry::new(input.descriptor().shape(), padding, mode)?;
    let output_elements = element_count(geometry.output_shape())?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "dtype byte width"))?;
    let byte_count = output_elements
        .checked_mul(width)
        .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "output bytes"))?;
    let mut bytes = backend.workspace_vec::<u8>(context, byte_count)?;
    let constant = if mode == FunctionalPadMode::Constant {
        input.descriptor().dtype().encode_decoded_scalar(
            value.unwrap_or(DecodedScalar::Signed(0)),
            FUNCTIONAL_PAD_OPERATION_ID,
            DeviceId::CPU,
        )?
    } else {
        Vec::new()
    };
    for linear in 0..output_elements {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, geometry.output_shape())?;
        let source = geometry.source_coordinates(&output_indices)?;
        let element = match source.as_deref() {
            Some(indices) => input.element_bytes(indices)?,
            None => constant.as_slice(),
        };
        workspace_extend(&mut bytes, element)?;
    }
    context.cancellation.check()?;
    upload_bytes(
        backend,
        geometry.output_shape(),
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn pad_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    padding: &[i64],
    mode: FunctionalPadMode,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    require_f32_cpu(output_gradient, FUNCTIONAL_PAD_OPERATION_ID)?;
    let geometry = PadGeometry::new(input_shape, padding, mode)?;
    if output_gradient.descriptor().shape() != geometry.output_shape() {
        return invalid(
            FUNCTIONAL_PAD_OPERATION_ID,
            "output gradient shape does not match the padded shape",
        );
    }
    let input_elements = element_count(input_shape)?;
    let mut values = backend.workspace_vec::<f32>(context, input_elements)?;
    for _ in 0..input_elements {
        values.try_push(0.0)?;
    }
    let output_elements = element_count(geometry.output_shape())?;
    for linear in 0..output_elements {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, geometry.output_shape())?;
        let Some(source_indices) = geometry.source_coordinates(&output_indices)? else {
            continue;
        };
        let destination = ravel_index(&source_indices, input_shape)?;
        let value = read_f32(output_gradient, &output_indices)?;
        let destination = values
            .get_mut(destination)
            .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "gradient destination"))?;
        *destination += value;
    }
    context.cancellation.check()?;
    upload_f32(
        backend,
        input_shape,
        output_gradient.descriptor().stream(),
        &values,
        context,
    )
}

pub fn pad_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    padding: &[i64],
    mode: FunctionalPadMode,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    functional_pad_with_context_exact_native(
        backend,
        input_tangent,
        padding,
        mode,
        None,
        context,
    )
}

fn squeeze_for_operation(
    input: &Tensor,
    dimensions: Option<&[i64]>,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    cancellation.check()?;
    let rank = input.descriptor().rank();
    let remove = match dimensions {
        None => input
            .descriptor()
            .shape()
            .iter()
            .map(|extent| *extent == 1)
            .collect::<Vec<_>>(),
        Some(dimensions) => {
            let mut remove = vec![false; rank];
            for &dimension in dimensions {
                let axis = normalize_axis(dimension, rank, operation)?;
                let selected = remove
                    .get_mut(axis)
                    .ok_or_else(|| overflow(operation, "squeeze axis"))?;
                if *selected {
                    return invalid(operation, "squeeze dimensions must be unique");
                }
                *selected = input
                    .descriptor()
                    .shape()
                    .get(axis)
                    .copied()
                    .ok_or_else(|| overflow(operation, "squeeze extent"))?
                    == 1;
            }
            remove
        }
    };
    let mut shape = Vec::new();
    let mut strides = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| overflow(operation, "squeezed shape"))?;
    strides
        .try_reserve_exact(rank)
        .map_err(|_| overflow(operation, "squeezed strides"))?;
    for (axis, (&extent, &stride)) in input
        .descriptor()
        .shape()
        .iter()
        .zip(input.descriptor().strides())
        .enumerate()
    {
        if !remove
            .get(axis)
            .copied()
            .ok_or_else(|| overflow(operation, "squeeze selection"))?
        {
            shape.push(extent);
            strides.push(stride);
        }
    }
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        input.descriptor().offset_elements(),
        input.descriptor().dtype(),
        Layout::Strided,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
struct AxisPadding {
    crop_before: u64,
    effective_extent: u64,
    positive_before: u64,
}

#[derive(Clone, Debug)]
struct PadGeometry {
    axes: Vec<AxisPadding>,
    output_shape: Vec<u64>,
    mode: FunctionalPadMode,
}

impl PadGeometry {
    fn new(
        input_shape: &[u64],
        padding: &[i64],
        mode: FunctionalPadMode,
    ) -> Result<Self, ShapeLayoutTransformPartThreeError> {
        if !padding.len().is_multiple_of(2) {
            return invalid(FUNCTIONAL_PAD_OPERATION_ID, "padding length must be even");
        }
        if padding.len() > input_shape.len().saturating_mul(2) {
            return invalid(
                FUNCTIONAL_PAD_OPERATION_ID,
                "padding addresses more dimensions than the input rank",
            );
        }
        let mut axes = input_shape
            .iter()
            .map(|extent| AxisPadding {
                crop_before: 0,
                effective_extent: *extent,
                positive_before: 0,
            })
            .collect::<Vec<_>>();
        let mut output_shape = input_shape.to_vec();
        for pair in 0..padding.len() / 2 {
            let axis = input_shape
                .len()
                .checked_sub(pair + 1)
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding axis"))?;
            let padding_index = pair
                .checked_mul(2)
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding index"))?;
            let before = i128::from(
                *padding
                    .get(padding_index)
                    .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding before"))?,
            );
            let after = i128::from(
                *padding
                    .get(padding_index + 1)
                    .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding after"))?,
            );
            let extent = i128::from(
                *input_shape
                    .get(axis)
                    .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "input extent"))?,
            );
            let crop_before = (-before).max(0);
            let crop_after = (-after).max(0);
            if crop_before + crop_after > extent {
                return invalid(
                    FUNCTIONAL_PAD_OPERATION_ID,
                    "negative padding crops beyond the input extent",
                );
            }
            let effective = extent - crop_before - crop_after;
            let positive_before = before.max(0);
            let positive_after = after.max(0);
            let output = effective
                .checked_add(positive_before)
                .and_then(|value| value.checked_add(positive_after))
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padded extent"))?;
            if mode != FunctionalPadMode::Constant && effective == 0 {
                return invalid(
                    FUNCTIONAL_PAD_OPERATION_ID,
                    "non-constant padding requires a nonempty source extent",
                );
            }
            if mode == FunctionalPadMode::Reflect
                && (positive_before >= effective || positive_after >= effective)
                && (positive_before != 0 || positive_after != 0)
            {
                return invalid(
                    FUNCTIONAL_PAD_OPERATION_ID,
                    "reflection padding must be smaller than the source extent",
                );
            }
            if mode == FunctionalPadMode::Circular
                && (positive_before > effective || positive_after > effective)
            {
                return invalid(
                    FUNCTIONAL_PAD_OPERATION_ID,
                    "circular padding cannot wrap more than once",
                );
            }
            *axes
                .get_mut(axis)
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding axis"))? = AxisPadding {
                crop_before: u64::try_from(crop_before)
                    .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "crop"))?,
                effective_extent: u64::try_from(effective)
                    .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "source extent"))?,
                positive_before: u64::try_from(positive_before)
                    .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "positive padding"))?,
            };
            *output_shape
                .get_mut(axis)
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "output axis"))? =
                u64::try_from(output)
                    .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "output extent"))?;
        }
        Ok(Self {
            axes,
            output_shape,
            mode,
        })
    }

    fn output_shape(&self) -> &[u64] {
        &self.output_shape
    }

    fn source_coordinates(
        &self,
        output_indices: &[u64],
    ) -> Result<Option<Vec<u64>>, ShapeLayoutTransformPartThreeError> {
        if output_indices.len() != self.axes.len() {
            return invalid(
                FUNCTIONAL_PAD_OPERATION_ID,
                "output coordinate rank does not match padding geometry",
            );
        }
        let mut source = Vec::new();
        source
            .try_reserve_exact(output_indices.len())
            .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "source coordinates"))?;
        for (&coordinate, axis) in output_indices.iter().zip(&self.axes) {
            let coordinate = usize::try_from(coordinate)
                .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "output coordinate"))?;
            let padding = usize::try_from(axis.positive_before)
                .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "padding"))?;
            let extent = usize::try_from(axis.effective_extent)
                .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "source extent"))?;
            let mode = match self.mode {
                FunctionalPadMode::Constant => ConvolutionPaddingMode::Zeros,
                FunctionalPadMode::Reflect => ConvolutionPaddingMode::Reflect,
                FunctionalPadMode::Replicate => ConvolutionPaddingMode::Replicate,
                FunctionalPadMode::Circular => ConvolutionPaddingMode::Circular,
            };
            let Some(mapped) = map_padded_coordinate(coordinate, padding, extent, mode)
                .map_err(|error| canonical_error(FUNCTIONAL_PAD_OPERATION_ID, error))?
            else {
                return Ok(None);
            };
            let mapped = axis
                .crop_before
                .checked_add(
                    u64::try_from(mapped)
                        .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "source coordinate"))?,
                )
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "source coordinate"))?;
            source.push(mapped);
        }
        Ok(Some(source))
    }
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    context.cancellation.check()?;
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartThreeError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_extend(
    destination: &mut CpuWorkspaceVec<u8>,
    source: &[u8],
) -> Result<(), ShapeLayoutTransformPartThreeError> {
    for byte in source {
        destination.try_push(*byte)?;
    }
    Ok(())
}

fn read_f32(
    tensor: &Tensor,
    indices: &[u64],
) -> Result<f32, ShapeLayoutTransformPartThreeError> {
    let bytes: [u8; 4] = tensor
        .element_bytes(indices)?
        .try_into()
        .map_err(|_| TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: tensor.descriptor().dtype(),
        })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartThreeError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ShapeLayoutTransformPartThreeError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartThreeError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ShapeLayoutTransformPartThreeError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn normalize_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartThreeError> {
    if rank == 0 {
        return invalid(operation, "operation requires a tensor axis");
    }
    let rank = i64::try_from(rank).map_err(|_| overflow(operation, "rank"))?;
    let axis = if axis < 0 {
        rank.checked_add(axis)
            .ok_or_else(|| overflow(operation, "axis"))?
    } else {
        axis
    };
    if axis < 0 || axis >= rank {
        return invalid(operation, "axis is outside the input rank");
    }
    usize::try_from(axis).map_err(|_| overflow(operation, "axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ShapeLayoutTransformPartThreeError> {
    shape
        .iter()
        .try_fold(1_u64, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "element count"))
        })
        .and_then(|count| {
            usize::try_from(count)
                .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "element count"))
        })
}

fn unravel_index(
    linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ShapeLayoutTransformPartThreeError> {
    let mut remaining = u64::try_from(linear)
        .map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "linear index"))?;
    let mut indices = vec![0_u64; shape.len()];
    for (&extent, index) in shape.iter().zip(indices.iter_mut()).rev() {
        if extent != 0 {
            *index = remaining % extent;
            remaining /= extent;
        }
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ShapeLayoutTransformPartThreeError> {
    if indices.len() != shape.len() {
        return invalid(FUNCTIONAL_PAD_OPERATION_ID, "coordinate rank mismatch");
    }
    let linear = indices
        .iter()
        .zip(shape)
        .try_fold(0_u64, |linear, (index, extent)| {
            if index >= extent {
                return invalid(FUNCTIONAL_PAD_OPERATION_ID, "coordinate is out of bounds");
            }
            linear
                .checked_mul(*extent)
                .and_then(|linear| linear.checked_add(*index))
                .ok_or_else(|| overflow(FUNCTIONAL_PAD_OPERATION_ID, "linear index"))
        })?;
    usize::try_from(linear).map_err(|_| overflow(FUNCTIONAL_PAD_OPERATION_ID, "linear index"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ShapeLayoutTransformPartThreeError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn canonical_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> ShapeLayoutTransformPartThreeError {
    ShapeLayoutTransformPartThreeError::CanonicalOwner {
        operation,
        reason: error.to_string(),
    }
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ShapeLayoutTransformPartThreeError> {
    Err(ShapeLayoutTransformPartThreeError::Invalid {
        operation,
        reason: reason.into(),
    })
}

fn overflow(
    operation: &'static str,
    subject: &'static str,
) -> ShapeLayoutTransformPartThreeError {
    ShapeLayoutTransformPartThreeError::ShapeOverflow { operation, subject }
}

#[cfg(test)]
mod validation_tests {
    use std::collections::BTreeMap;

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            (
                "COMFY-TENSOR-OP-DF605ED35114",
                "54f9d7991b3890bb9f321c1f65ffa091150201036711c740eab1c8b04ef35b11",
            ),
            (
                "COMFY-TENSOR-OP-E1ADB7F2ED49",
                "3887e46e9b8cedd6cac61c057ad3b0813d18d71127aa23370e9e23160c422cd1",
            ),
            (
                "COMFY-TENSOR-OP-E867958E2F71",
                "25aa2a992217a9e753de41b4dc5748c5708f4769efe94560bf80604ffd7dc051",
            ),
            (
                "COMFY-TENSOR-OP-ED0C08B31410",
                "97969e53a77c176948f7273ac125a9b2dfc36bba96db3d3e6622f00a8f319875",
            ),
            (
                "COMFY-TENSOR-OP-FC2184DD3E3D",
                "6b6208d1d8588d27b8666487ac2b6401207c7a502ec266e6e6a4fd4ca7be3bb3",
            ),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation| (*operation, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-shape-layout-transform-03.json",
            "VAL-TENSOR-001",
            "Task 88 exact shape facades over descriptor, Task 60 split, canonical padding-coordinate, and CpuBackend owners",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-shape-layout-transform-03.json",
            "VAL-AUTOGRAD-001",
            "Task 88 analytical view, split, and padding VJP/JVP contracts",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
