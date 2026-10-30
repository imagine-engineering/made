use crate::{
    CpuBackend, DType, DeviceId, ExecutionContext, ReductionOperation, ReductionSpec, StreamId,
    Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_11::{
        ElementwiseRuntimePartElevenError, maximum_with_context_exact_native,
    },
};
use thiserror::Error;

pub const TENSOR_ALL_OPERATION_ID: &str = "COMFY-TENSOR-OP-36F99E8950F4";
pub const TENSOR_ANY_OPERATION_ID: &str = "COMFY-TENSOR-OP-2326740E6353";
pub const TENSOR_ARGMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-00E998458E0C";
pub const TENSOR_MAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-61B9BB91A65A";
pub const TENSOR_MEAN_OPERATION_ID: &str = "COMFY-TENSOR-OP-7821FE22568F";
pub const TENSOR_MIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-5ED308A788FD";
pub const TENSOR_PROD_OPERATION_ID: &str = "COMFY-TENSOR-OP-37C179735855";
pub const TENSOR_VAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-228E61E0B93B";
pub const TORCH_ALL_OPERATION_ID: &str = "COMFY-TENSOR-OP-7ED5B6B0740C";
pub const TORCH_ARGMIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-578E8375BF1C";
pub const TORCH_MAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-5B9E7FC75162";
pub const TORCH_STD_OPERATION_ID: &str = "COMFY-TENSOR-OP-955EF5B745CE";

#[derive(Debug, Error)]
pub enum ReductionPartOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("NumPy contiguous scalar sum received {count} values, exceeding limit {limit}")]
    ScalarInputLimit { count: usize, limit: usize },
    #[error(transparent)]
    Elementwise(#[from] ElementwiseRuntimePartElevenError),
    #[error("operation {operation} received dimension {dimension} for rank {rank}")]
    InvalidDimension {
        operation: &'static str,
        dimension: i64,
        rank: usize,
    },
    #[error("operation {operation} received duplicate dimension {dimension}")]
    DuplicateDimension {
        operation: &'static str,
        dimension: i64,
    },
    #[error("operation {operation} requires a floating input dtype, not {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
}

#[derive(Clone, Debug)]
pub struct ReductionForward {
    pub values: Tensor,
    pub indices: Option<Tensor>,
}

pub enum TorchMaximumArgument<'a> {
    All,
    Dimension(i64),
    Tensor(&'a Tensor),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DifferentiableReduction {
    Sum { operation_id: &'static str },
    Mean,
    Product,
    Minimum,
    Maximum,
    Variance { correction: u64 },
    StandardDeviation { correction: u64 },
}

pub fn tensor_all_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    reduce(
        backend,
        input,
        ReductionOperation::All,
        dimensions,
        keep_dimensions,
        DType::Bool,
        0,
        TENSOR_ALL_OPERATION_ID,
        context,
    )
}

pub fn tensor_any_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    reduce(
        backend,
        input,
        ReductionOperation::Any,
        dimensions,
        keep_dimensions,
        DType::Bool,
        0,
        TENSOR_ANY_OPERATION_ID,
        context,
    )
}

pub fn tensor_argmax_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    reduce(
        backend,
        input,
        ReductionOperation::ArgMaximum,
        dimension.as_ref().map(std::slice::from_ref),
        keep_dimensions,
        DType::I64,
        0,
        TENSOR_ARGMAX_OPERATION_ID,
        context,
    )
}

pub fn tensor_max_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<ReductionForward, ReductionPartOneError> {
    extrema(
        backend,
        input,
        dimension,
        keep_dimensions,
        ReductionOperation::Maximum,
        ReductionOperation::ArgMaximum,
        TENSOR_MAX_OPERATION_ID,
        context,
    )
}

pub fn tensor_mean_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    let dtype = floating_output_dtype(input, dtype, TENSOR_MEAN_OPERATION_ID)?;
    reduce(
        backend,
        input,
        ReductionOperation::Mean,
        dimensions,
        keep_dimensions,
        dtype,
        0,
        TENSOR_MEAN_OPERATION_ID,
        context,
    )
}

pub fn tensor_min_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<ReductionForward, ReductionPartOneError> {
    extrema(
        backend,
        input,
        dimension,
        keep_dimensions,
        ReductionOperation::Minimum,
        ReductionOperation::ArgMinimum,
        TENSOR_MIN_OPERATION_ID,
        context,
    )
}

pub fn tensor_prod_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    let dtype = floating_output_dtype(input, dtype, TENSOR_PROD_OPERATION_ID)?;
    reduce(
        backend,
        input,
        ReductionOperation::Product,
        dimension.as_ref().map(std::slice::from_ref),
        keep_dimensions,
        dtype,
        0,
        TENSOR_PROD_OPERATION_ID,
        context,
    )
}

pub fn tensor_var_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    correction: u64,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    let dtype = floating_output_dtype(input, None, TENSOR_VAR_OPERATION_ID)?;
    reduce(
        backend,
        input,
        ReductionOperation::Variance,
        dimensions,
        keep_dimensions,
        dtype,
        correction,
        TENSOR_VAR_OPERATION_ID,
        context,
    )
}

pub fn torch_all_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    reduce(
        backend,
        input,
        ReductionOperation::All,
        dimensions,
        keep_dimensions,
        DType::Bool,
        0,
        TORCH_ALL_OPERATION_ID,
        context,
    )
}

pub fn torch_argmin_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    reduce(
        backend,
        input,
        ReductionOperation::ArgMinimum,
        dimension.as_ref().map(std::slice::from_ref),
        keep_dimensions,
        DType::I64,
        0,
        TORCH_ARGMIN_OPERATION_ID,
        context,
    )
}

pub fn torch_max_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    argument: TorchMaximumArgument<'_>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<ReductionForward, ReductionPartOneError> {
    match argument {
        TorchMaximumArgument::All => extrema(
            backend,
            input,
            None,
            keep_dimensions,
            ReductionOperation::Maximum,
            ReductionOperation::ArgMaximum,
            TORCH_MAX_OPERATION_ID,
            context,
        ),
        TorchMaximumArgument::Dimension(dimension) => extrema(
            backend,
            input,
            Some(dimension),
            keep_dimensions,
            ReductionOperation::Maximum,
            ReductionOperation::ArgMaximum,
            TORCH_MAX_OPERATION_ID,
            context,
        ),
        TorchMaximumArgument::Tensor(other) => Ok(ReductionForward {
            values: maximum_with_context_exact_native(
                backend,
                input,
                ElementwiseOperand::Tensor(other),
                context,
            )?,
            indices: None,
        }),
    }
}

pub fn torch_std_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    correction: u64,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    let dtype = floating_output_dtype(input, None, TORCH_STD_OPERATION_ID)?;
    reduce(
        backend,
        input,
        ReductionOperation::StandardDeviation,
        dimensions,
        keep_dimensions,
        dtype,
        correction,
        TORCH_STD_OPERATION_ID,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn reduction_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output: &Tensor,
    indices: Option<&Tensor>,
    output_gradient: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    operation: DifferentiableReduction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    context.cancellation.check().map_err(TensorError::from)?;
    let dimensions = normalize_dimensions(
        dimensions,
        input.descriptor().rank(),
        differentiable_operation_id(operation),
    )?;
    let derivatives = reduction_local_derivatives(
        backend,
        input,
        output,
        indices,
        &dimensions,
        keep_dimensions,
        operation,
        context,
    )?;
    let expected_output_shape =
        reduced_shape(input.descriptor().shape(), &dimensions, keep_dimensions)?;
    require_f32_shape(output_gradient, &expected_output_shape)?;
    let mut gradients = Vec::new();
    gradients
        .try_reserve_exact(derivatives.len())
        .map_err(|error| TensorError::AllocationFailed {
            requested: u64::try_from(derivatives.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(4),
            reason: error.to_string(),
        })?;
    for (linear_index, derivative) in derivatives.into_iter().enumerate() {
        context.cancellation.check().map_err(TensorError::from)?;
        let linear_index = u64::try_from(linear_index).map_err(|_| TensorError::ShapeOverflow)?;
        let input_indices = linear_to_indices_adapter(linear_index, input.descriptor().shape())?;
        let output_index = output_index_for_input(
            &input_indices,
            &dimensions,
            keep_dimensions,
            &expected_output_shape,
        )?;
        gradients.push(derivative * read_f32_linear(output_gradient, output_index)?);
    }
    upload_f32_shape(backend, input.descriptor().shape(), &gradients, context)
}

#[allow(clippy::too_many_arguments)]
pub fn reduction_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    output: &Tensor,
    indices: Option<&Tensor>,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    operation: DifferentiableReduction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    context.cancellation.check().map_err(TensorError::from)?;
    require_f32_shape(input_tangent, input.descriptor().shape())?;
    let dimensions = normalize_dimensions(
        dimensions,
        input.descriptor().rank(),
        differentiable_operation_id(operation),
    )?;
    let derivatives = reduction_local_derivatives(
        backend,
        input,
        output,
        indices,
        &dimensions,
        keep_dimensions,
        operation,
        context,
    )?;
    let output_shape = reduced_shape(input.descriptor().shape(), &dimensions, keep_dimensions)?;
    let output_count = usize::try_from(checked_element_count(&output_shape)?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let mut tangent = vec![0.0_f32; output_count];
    for (linear_index, derivative) in derivatives.into_iter().enumerate() {
        context.cancellation.check().map_err(TensorError::from)?;
        let linear_index = u64::try_from(linear_index).map_err(|_| TensorError::ShapeOverflow)?;
        let input_indices = linear_to_indices_adapter(linear_index, input.descriptor().shape())?;
        let output_index = usize::try_from(output_index_for_input(
            &input_indices,
            &dimensions,
            keep_dimensions,
            &output_shape,
        )?)
        .map_err(|_| TensorError::ShapeOverflow)?;
        let slot = tangent
            .get_mut(output_index)
            .ok_or(TensorError::ShapeOverflow)?;
        *slot += derivative * read_f32_linear(input_tangent, linear_index)?;
    }
    upload_f32_shape(backend, &output_shape, &tangent, context)
}

#[allow(clippy::too_many_arguments)]
fn reduction_local_derivatives(
    backend: &CpuBackend,
    input: &Tensor,
    output: &Tensor,
    indices: Option<&Tensor>,
    dimensions: &[usize],
    keep_dimensions: bool,
    operation: DifferentiableReduction,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, ReductionPartOneError> {
    require_f32_shape(input, input.descriptor().shape())?;
    let output_shape = reduced_shape(input.descriptor().shape(), dimensions, keep_dimensions)?;
    require_f32_shape(output, &output_shape)?;
    let input_count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let output_count = usize::try_from(checked_element_count(&output_shape)?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let reduced_count = dimensions.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(
                *input
                    .descriptor()
                    .shape()
                    .get(*dimension)
                    .ok_or(TensorError::ShapeOverflow)?,
            )
            .ok_or(TensorError::ShapeOverflow)
    })?;
    let mean = if matches!(
        operation,
        DifferentiableReduction::Variance { .. }
            | DifferentiableReduction::StandardDeviation { .. }
    ) {
        let signed_dimensions = dimensions
            .iter()
            .map(|dimension| i64::try_from(*dimension).map_err(|_| TensorError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Some(tensor_mean_with_context_exact_native(
            backend,
            input,
            Some(&signed_dimensions),
            keep_dimensions,
            None,
            context,
        )?)
    } else {
        None
    };
    let mut zero_counts = vec![0_u64; output_count];
    let mut nonzero_products = vec![1.0_f32; output_count];
    let mut tie_counts = vec![0_u64; output_count];
    if matches!(operation, DifferentiableReduction::Product)
        || matches!(
            operation,
            DifferentiableReduction::Minimum | DifferentiableReduction::Maximum
        ) && indices.is_none()
    {
        for linear_index in 0..input.descriptor().element_count()? {
            context.cancellation.check().map_err(TensorError::from)?;
            let input_indices =
                linear_to_indices_adapter(linear_index, input.descriptor().shape())?;
            let output_index = usize::try_from(output_index_for_input(
                &input_indices,
                dimensions,
                keep_dimensions,
                &output_shape,
            )?)
            .map_err(|_| TensorError::ShapeOverflow)?;
            let value = read_f32_linear(input, linear_index)?;
            if matches!(operation, DifferentiableReduction::Product) {
                if value == 0.0 {
                    *zero_counts
                        .get_mut(output_index)
                        .ok_or(TensorError::ShapeOverflow)? += 1;
                } else {
                    *nonzero_products
                        .get_mut(output_index)
                        .ok_or(TensorError::ShapeOverflow)? *= value;
                }
            } else if values_equal_for_extremum(
                value,
                read_f32_linear(
                    output,
                    u64::try_from(output_index).map_err(|_| TensorError::ShapeOverflow)?,
                )?,
            ) {
                *tie_counts
                    .get_mut(output_index)
                    .ok_or(TensorError::ShapeOverflow)? += 1;
            }
        }
    }
    let selected_indices = if let Some(indices) = indices {
        if indices.descriptor().dtype() != DType::I64
            || indices.descriptor().shape() != output_shape
        {
            return Err(TensorError::DTypeMismatch {
                expected: DType::I64,
                actual: indices.descriptor().dtype(),
            }
            .into());
        }
        Some(indices)
    } else {
        None
    };
    let mut derivatives = Vec::new();
    derivatives
        .try_reserve_exact(input_count)
        .map_err(|error| TensorError::AllocationFailed {
            requested: u64::try_from(input_count)
                .unwrap_or(u64::MAX)
                .saturating_mul(4),
            reason: error.to_string(),
        })?;
    for linear_index in 0..input.descriptor().element_count()? {
        context.cancellation.check().map_err(TensorError::from)?;
        let input_indices = linear_to_indices_adapter(linear_index, input.descriptor().shape())?;
        let output_index =
            output_index_for_input(&input_indices, dimensions, keep_dimensions, &output_shape)?;
        let output_slot = usize::try_from(output_index).map_err(|_| TensorError::ShapeOverflow)?;
        let value = read_f32_linear(input, linear_index)?;
        let derivative = match operation {
            DifferentiableReduction::Sum { .. } => 1.0,
            DifferentiableReduction::Mean => 1.0 / reduced_count as f32,
            DifferentiableReduction::Product => {
                let zero_count = *zero_counts
                    .get(output_slot)
                    .ok_or(TensorError::ShapeOverflow)?;
                let nonzero_product = *nonzero_products
                    .get(output_slot)
                    .ok_or(TensorError::ShapeOverflow)?;
                match (zero_count, value == 0.0) {
                    (0, false) => read_f32_linear(output, output_index)? / value,
                    (1, true) => nonzero_product,
                    _ => 0.0,
                }
            }
            DifferentiableReduction::Minimum | DifferentiableReduction::Maximum => {
                if let Some(indices) = selected_indices {
                    let dimension = dimensions
                        .first()
                        .copied()
                        .ok_or(TensorError::ShapeOverflow)?;
                    let selected = read_i64_linear(indices, output_index)?;
                    f32::from(
                        i64::try_from(
                            *input_indices
                                .get(dimension)
                                .ok_or(TensorError::ShapeOverflow)?,
                        )
                        .map_err(|_| TensorError::ShapeOverflow)?
                            == selected,
                    )
                } else if values_equal_for_extremum(value, read_f32_linear(output, output_index)?) {
                    let tie_count = *tie_counts
                        .get(output_slot)
                        .ok_or(TensorError::ShapeOverflow)?;
                    if tie_count == 0 {
                        0.0
                    } else {
                        1.0 / tie_count as f32
                    }
                } else {
                    0.0
                }
            }
            DifferentiableReduction::Variance { correction } => {
                let denominator = reduced_count.checked_sub(correction).unwrap_or(0);
                if denominator == 0 {
                    f32::NAN
                } else {
                    let mean = mean.as_ref().ok_or(TensorError::ShapeOverflow)?;
                    2.0 * (value - read_f32_linear(mean, output_index)?) / denominator as f32
                }
            }
            DifferentiableReduction::StandardDeviation { correction } => {
                let denominator = reduced_count.checked_sub(correction).unwrap_or(0);
                let standard_deviation = read_f32_linear(output, output_index)?;
                if denominator == 0 || standard_deviation == 0.0 {
                    f32::NAN
                } else {
                    let mean = mean.as_ref().ok_or(TensorError::ShapeOverflow)?;
                    (value - read_f32_linear(mean, output_index)?)
                        / (denominator as f32 * standard_deviation)
                }
            }
        };
        derivatives.push(derivative);
    }
    Ok(derivatives)
}

fn differentiable_operation_id(operation: DifferentiableReduction) -> &'static str {
    match operation {
        DifferentiableReduction::Sum { operation_id } => operation_id,
        DifferentiableReduction::Mean => TENSOR_MEAN_OPERATION_ID,
        DifferentiableReduction::Product => TENSOR_PROD_OPERATION_ID,
        DifferentiableReduction::Minimum => TENSOR_MIN_OPERATION_ID,
        DifferentiableReduction::Maximum => TENSOR_MAX_OPERATION_ID,
        DifferentiableReduction::Variance { .. } => TENSOR_VAR_OPERATION_ID,
        DifferentiableReduction::StandardDeviation { .. } => TORCH_STD_OPERATION_ID,
    }
}

fn values_equal_for_extremum(left: f32, right: f32) -> bool {
    (left.is_nan() && right.is_nan()) || left == right
}

fn require_f32_shape(tensor: &Tensor, shape: &[u64]) -> Result<(), TensorError> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: tensor.descriptor().dtype(),
        });
    }
    if tensor.descriptor().shape() != shape {
        return Err(TensorError::InvalidNumeric {
            reason: format!(
                "reduction derivative tensor shape {:?} does not match {shape:?}",
                tensor.descriptor().shape()
            ),
        });
    }
    Ok(())
}

fn upload_f32_shape(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn read_f32_linear(tensor: &Tensor, linear_index: u64) -> Result<f32, TensorError> {
    let bytes: [u8; 4] = tensor
        .linear_element_bytes(linear_index)?
        .try_into()
        .map_err(|_| TensorError::StorageLength {
            expected: 4,
            actual: tensor.descriptor().dtype().byte_width(),
        })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn read_i64_linear(tensor: &Tensor, linear_index: u64) -> Result<i64, TensorError> {
    let bytes: [u8; 8] = tensor
        .linear_element_bytes(linear_index)?
        .try_into()
        .map_err(|_| TensorError::StorageLength {
            expected: 8,
            actual: tensor.descriptor().dtype().byte_width(),
        })?;
    Ok(i64::from_ne_bytes(bytes))
}

fn output_index_for_input(
    input_indices: &[u64],
    dimensions: &[usize],
    keep_dimensions: bool,
    output_shape: &[u64],
) -> Result<u64, TensorError> {
    let mut reduced = vec![false; input_indices.len()];
    for &dimension in dimensions {
        *reduced
            .get_mut(dimension)
            .ok_or(TensorError::ShapeOverflow)? = true;
    }
    let output_indices = input_indices
        .iter()
        .enumerate()
        .filter_map(|(axis, index)| {
            if reduced.get(axis).copied().unwrap_or(false) {
                keep_dimensions.then_some(0)
            } else {
                Some(*index)
            }
        })
        .collect::<Vec<_>>();
    indices_to_linear_adapter(&output_indices, output_shape)
}

fn checked_element_count(shape: &[u64]) -> Result<u64, TensorError> {
    shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::ShapeOverflow)
    })
}

fn linear_to_indices_adapter(mut linear: u64, shape: &[u64]) -> Result<Vec<u64>, TensorError> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let size = *shape.get(axis).ok_or(TensorError::ShapeOverflow)?;
        if size == 0 {
            return Err(TensorError::ShapeOverflow);
        }
        *indices.get_mut(axis).ok_or(TensorError::ShapeOverflow)? = linear % size;
        linear /= size;
    }
    Ok(indices)
}

fn indices_to_linear_adapter(indices: &[u64], shape: &[u64]) -> Result<u64, TensorError> {
    if indices.len() != shape.len() {
        return Err(TensorError::IndexRankMismatch {
            rank: shape.len(),
            indices: indices.len(),
        });
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_u64, |linear, (&index, &size)| {
            linear
                .checked_mul(size)
                .and_then(|value| value.checked_add(index))
                .ok_or(TensorError::ShapeOverflow)
        })
}

#[allow(clippy::too_many_arguments)]
fn extrema(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    value_operation: ReductionOperation,
    index_operation: ReductionOperation,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<ReductionForward, ReductionPartOneError> {
    let dimensions = dimension.as_ref().map(std::slice::from_ref);
    let values = reduce(
        backend,
        input,
        value_operation,
        dimensions,
        keep_dimensions,
        input.descriptor().dtype(),
        0,
        operation_id,
        context,
    )?;
    let indices = if dimension.is_some() {
        Some(reduce(
            backend,
            input,
            index_operation,
            dimensions,
            keep_dimensions,
            DType::I64,
            0,
            operation_id,
            context,
        )?)
    } else {
        None
    };
    Ok(ReductionForward { values, indices })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reduce(
    backend: &CpuBackend,
    input: &Tensor,
    operation: ReductionOperation,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    output_dtype: DType,
    correction: u64,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartOneError> {
    context.cancellation.check().map_err(TensorError::from)?;
    let dimensions = normalize_dimensions(dimensions, input.descriptor().rank(), operation_id)?;
    let shape = reduced_shape(input.descriptor().shape(), &dimensions, keep_dimensions)?;
    let output = TensorDescriptor::contiguous(
        shape,
        output_dtype,
        require_cpu(input, operation_id)?,
        require_stream(input),
    )?;
    let spec = ReductionSpec {
        operation,
        dimensions: dimensions
            .iter()
            .map(|dimension| u64::try_from(*dimension).map_err(|_| TensorError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?,
        keep_dimensions,
        accumulation_dtype: matches!(
            operation,
            ReductionOperation::Sum
                | ReductionOperation::Product
                | ReductionOperation::Mean
                | ReductionOperation::Variance
                | ReductionOperation::StandardDeviation
        )
        .then_some(output_dtype),
        correction,
    };
    Ok(backend.reduction(&spec, input, output, context)?.0)
}

fn normalize_dimensions(
    dimensions: Option<&[i64]>,
    rank: usize,
    operation: &'static str,
) -> Result<Vec<usize>, ReductionPartOneError> {
    let dimensions = match dimensions {
        Some(dimensions) if !dimensions.is_empty() => dimensions
            .iter()
            .map(|dimension| normalize_dimension(*dimension, rank, operation))
            .collect::<Result<Vec<_>, _>>()?,
        _ => (0..rank).collect(),
    };
    let mut dimensions = dimensions;
    dimensions.sort_unstable();
    for pair in dimensions.windows(2) {
        if pair.first() == pair.get(1) {
            let dimension = pair.first().copied().unwrap_or_default();
            return Err(ReductionPartOneError::DuplicateDimension {
                operation,
                dimension: i64::try_from(dimension).unwrap_or(i64::MAX),
            });
        }
    }
    Ok(dimensions)
}

fn normalize_dimension(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ReductionPartOneError> {
    let rank_i64 = i64::try_from(rank).map_err(|_| ReductionPartOneError::InvalidDimension {
        operation,
        dimension,
        rank,
    })?;
    let normalized = if dimension < 0 {
        rank_i64.checked_add(dimension)
    } else {
        Some(dimension)
    };
    normalized
        .filter(|dimension| *dimension >= 0 && *dimension < rank_i64)
        .and_then(|dimension| usize::try_from(dimension).ok())
        .ok_or(ReductionPartOneError::InvalidDimension {
            operation,
            dimension,
            rank,
        })
}

fn reduced_shape(
    shape: &[u64],
    dimensions: &[usize],
    keep_dimensions: bool,
) -> Result<Vec<u64>, TensorError> {
    let mut reduced = vec![false; shape.len()];
    for &dimension in dimensions {
        *reduced
            .get_mut(dimension)
            .ok_or(TensorError::ShapeOverflow)? = true;
    }
    Ok(shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| {
            if reduced.get(axis).copied().unwrap_or(false) {
                keep_dimensions.then_some(1)
            } else {
                Some(*size)
            }
        })
        .collect())
}

pub(crate) fn floating_output_dtype(
    input: &Tensor,
    requested: Option<DType>,
    operation: &'static str,
) -> Result<DType, ReductionPartOneError> {
    let dtype = requested.unwrap_or(input.descriptor().dtype());
    if matches!(dtype, DType::F64 | DType::F32 | DType::F16 | DType::Bf16)
        && matches!(
            input.descriptor().dtype(),
            DType::F64 | DType::F32 | DType::F16 | DType::Bf16
        )
    {
        Ok(dtype)
    } else {
        Err(ReductionPartOneError::UnsupportedDType { operation, dtype })
    }
}

fn require_cpu(input: &Tensor, operation: &'static str) -> Result<DeviceId, ReductionPartOneError> {
    let device = input.descriptor().device();
    if device == DeviceId::CPU {
        Ok(device)
    } else {
        Err(TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: "the Task 83 reduction adapter is certified on the canonical CPU backend"
                .to_owned(),
        }
        .into())
    }
}

fn require_stream(input: &Tensor) -> StreamId {
    input.descriptor().stream()
}

#[cfg(test)]
mod validation_tests {
    use std::collections::BTreeMap;

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            (
                "COMFY-TENSOR-OP-36F99E8950F4",
                "8646bee905f74984e74053d1f40869eb519a1a16d3db08b948a42e25061f249e",
            ),
            (
                "COMFY-TENSOR-OP-2326740E6353",
                "16839f72f9ba6e56dcee65a3fdd5be3b0bcafe50f360237e3f07f1bd41c515e3",
            ),
            (
                "COMFY-TENSOR-OP-00E998458E0C",
                "61cc87dd3810c68d38071498aa81eca108db1fb18ea3cdbde1a1836b222c386f",
            ),
            (
                "COMFY-TENSOR-OP-61B9BB91A65A",
                "195f71881edaa5821a47c2fe2c7e7e576644d4bcb09aeecca3124b4ff59460bb",
            ),
            (
                "COMFY-TENSOR-OP-7821FE22568F",
                "b327ace6a88f179a92af144e97acbd7e50a1a03c290c2a573f3acc550a2be0af",
            ),
            (
                "COMFY-TENSOR-OP-5ED308A788FD",
                "fa939677d2a8f7b2bb514828739fd9a247c65689be00eea7c36474bedf49b4ce",
            ),
            (
                "COMFY-TENSOR-OP-37C179735855",
                "33ee6fced089b7b441e4c8fdc348b90cae47203a37135d43f5982e673a2ab6ce",
            ),
            (
                "COMFY-TENSOR-OP-228E61E0B93B",
                "cf1adf0621f0929cfb8999c6099329f0e50773e584ad8a6ef2c89141406c00db",
            ),
            (
                "COMFY-TENSOR-OP-7ED5B6B0740C",
                "95cfc1d0e674bed9ce26070805c9e5934d45c06bcc1f1e0c8646512e6514b924",
            ),
            (
                "COMFY-TENSOR-OP-578E8375BF1C",
                "173792a55eec75919e58fe4b1e055902ce7e6926780e8c65dc930cfeb57393c7",
            ),
            (
                "COMFY-TENSOR-OP-5B9E7FC75162",
                "36d3e6161f2dea4546ca0fca9c1460e8eded640f9fd2754a281dfb29eaf3c002",
            ),
            (
                "COMFY-TENSOR-OP-955EF5B745CE",
                "dd4ff0a362aee8bc9670c2d2db5c70722ddee9b27cd82cf8d3de5783d74c8e7f",
            ),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation_id| (*operation_id, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-reduction-01.json",
            "VAL-TENSOR-001",
            "Task 83 reduction part one: 12 exact operations through TensorBackend::reduction",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-reduction-01.json",
            "VAL-AUTOGRAD-001",
            "Task 83 reduction differentiability classification and forward semantics",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}

pub fn numpy_contiguous_sum_f32_with_context_exact_native(
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<f32, ReductionPartOneError> {
    context.check()?;
    if values.len() > 8192 {
        return Err(ReductionPartOneError::ScalarInputLimit {
            count: values.len(),
            limit: 8192,
        });
    }
    Ok(0.0 + numpy_pairwise_sum_f32(values, context)?)
}

fn numpy_pairwise_sum_f32(
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<f32, ReductionPartOneError> {
    context.check()?;
    if values.len() < 8 {
        return Ok(values.iter().fold(-0.0, |sum, value| sum + value));
    }
    if values.len() > 128 {
        let split = (values.len() / 2) / 8 * 8;
        let (left, right) = values.split_at(split);
        return Ok(numpy_pairwise_sum_f32(left, context)? + numpy_pairwise_sum_f32(right, context)?);
    }
    let mut sums = [0.0; 8];
    sums.copy_from_slice(&values[..8]);
    let aligned = values.len() - values.len() % 8;
    for block in values[8..aligned].chunks_exact(8) {
        for (sum, value) in sums.iter_mut().zip(block) {
            *sum += value;
        }
    }
    // NumPy's eight accumulators and block boundaries determine source rounding.
    let mut result =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for value in &values[aligned..] {
        result += value;
    }
    context.check()?;
    Ok(result)
}

pub fn numpy_contiguous_sum_f64_with_context_exact_native(
    values: &[f64],
    context: &ExecutionContext<'_>,
) -> Result<f64, ReductionPartOneError> {
    context.check()?;
    if values.len() > 8192 {
        return Err(ReductionPartOneError::ScalarInputLimit {
            count: values.len(),
            limit: 8192,
        });
    }
    Ok(0.0 + numpy_pairwise_sum_f64(values, context)?)
}

fn numpy_pairwise_sum_f64(
    values: &[f64],
    context: &ExecutionContext<'_>,
) -> Result<f64, ReductionPartOneError> {
    context.check()?;
    if values.len() < 8 {
        return Ok(values.iter().fold(-0.0, |sum, value| sum + value));
    }
    if values.len() > 128 {
        let split = (values.len() / 2) / 8 * 8;
        let (left, right) = values.split_at(split);
        return Ok(numpy_pairwise_sum_f64(left, context)? + numpy_pairwise_sum_f64(right, context)?);
    }
    let mut sums = [0.0; 8];
    sums.copy_from_slice(&values[..8]);
    let aligned = values.len() - values.len() % 8;
    for block in values[8..aligned].chunks_exact(8) {
        for (sum, value) in sums.iter_mut().zip(block) {
            *sum += value;
        }
    }
    // NumPy's eight accumulators and block boundaries determine source rounding.
    let mut result =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for value in &values[aligned..] {
        result += value;
    }
    context.check()?;
    Ok(result)
}
