use crate::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar,
    DeviceId, ExecutionContext, LinearAlgebraOperation, Scalar, ScalarSide, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
        tensor_from_f32_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError,
        ceil_with_context_exact_native as canonical_ceil_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, full_like_with_context_exact_native,
    },
};
use thiserror::Error;

pub const FLOAT_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-AB3C563E635F";
pub const LONG_OPERATION_ID: &str = "COMFY-TENSOR-OP-AA014D6FD446";
pub const ATAN2_OPERATION_ID: &str = "COMFY-TENSOR-OP-AC4C80016C2B";
pub const FLASH_SDP_OPERATION_ID: &str = "COMFY-TENSOR-OP-AB6C1D5013D1";
pub const BADDBMM_OPERATION_ID: &str = "COMFY-TENSOR-OP-AA097B951CB6";
pub const CEIL_OPERATION_ID: &str = "COMFY-TENSOR-OP-A91E1CFDC489";
pub const DIAG_OPERATION_ID: &str = "COMFY-TENSOR-OP-A8B4A9E79500";
pub const FLIPLR_OPERATION_ID: &str = "COMFY-TENSOR-OP-AC979D604DAA";
pub const NPU_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-A69CEE614EB5";
pub const ONES_LIKE_OPERATION_ID: &str = "COMFY-TENSOR-OP-AA36FFD0433B";
pub const SEARCHSORTED_OPERATION_ID: &str = "COMFY-TENSOR-OP-AAB5F10B20F5";
pub const TAN_OPERATION_ID: &str = "COMFY-TENSOR-OP-A68AE691163C";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartFifteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Cast(#[from] OperatorIndirectionError),
    #[error(transparent)]
    CanonicalCeil(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    CanonicalFullLike(#[from] ElementwiseRuntimePartNineError),
    #[error("elementwise/runtime part-fifteen execution was cancelled")]
    Cancelled,
    #[error("operation {operation} requires CPU ordinal zero, got {device:?}")]
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
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartFifteenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct Atan2Gradients {
    pub input: Tensor,
    pub other: Tensor,
}

#[derive(Clone, Debug)]
pub struct BaddbmmGradients {
    pub input: Tensor,
    pub batch1: Tensor,
    pub batch2: Tensor,
}

pub fn float_tensor_with_context_exact_native(
    backend: &CpuBackend,
    values: &[f32],
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    if element_count(shape)? != values.len() {
        return Err(ElementwiseRuntimePartFifteenError::Invalid {
            operation: FLOAT_TENSOR_OPERATION_ID,
            reason: "shape does not match the supplied value count".to_owned(),
        });
    }
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

pub fn long_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    Ok(cast_to_with_context_exact_native(
        backend,
        input,
        DType::I64,
        DeviceId::CPU,
        false,
        false,
        context,
    )?)
}

pub fn atan2_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_binary_f32(input, other, ATAN2_OPERATION_ID)?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
    let descriptor = TensorDescriptor::contiguous(
        shape,
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .binary(BinaryOperation::Atan2, input, other, descriptor, context)?
        .0)
}

pub fn atan2_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Atan2Gradients, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_binary_f32(input, other, ATAN2_OPERATION_ID)?;
    require_f32_cpu(output_gradient, ATAN2_OPERATION_ID)?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
    require_shape_stream(
        output_gradient,
        &shape,
        input.descriptor().stream(),
        ATAN2_OPERATION_ID,
    )?;
    let mut input_values = zero_f32_values(backend, input.descriptor().shape(), context)?;
    let mut other_values = zero_f32_values(backend, other.descriptor().shape(), context)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let input_indices = broadcast_indices(&indices, input.descriptor().shape())?;
        let other_indices = broadcast_indices(&indices, other.descriptor().shape())?;
        let input_value = read_f32(input, &input_indices, ATAN2_OPERATION_ID)?;
        let other_value = read_f32(other, &other_indices, ATAN2_OPERATION_ID)?;
        let gradient = read_f32(output_gradient, &indices, ATAN2_OPERATION_ID)?;
        let denominator = input_value * input_value + other_value * other_value;
        input_values[ravel_index(&input_indices, input.descriptor().shape())?] +=
            gradient * other_value / denominator;
        other_values[ravel_index(&other_indices, other.descriptor().shape())?] +=
            -gradient * input_value / denominator;
    }
    Ok(Atan2Gradients {
        input: upload_f32(backend, input.descriptor().shape(), &input_values, context)?,
        other: upload_f32(backend, other.descriptor().shape(), &other_values, context)?,
    })
}

pub fn atan2_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_binary_f32(input, other, ATAN2_OPERATION_ID)?;
    require_tangent(input, input_tangent, ATAN2_OPERATION_ID)?;
    require_tangent(other, other_tangent, ATAN2_OPERATION_ID)?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, element_count(&shape)?)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let input_indices = broadcast_indices(&indices, input.descriptor().shape())?;
        let other_indices = broadcast_indices(&indices, other.descriptor().shape())?;
        let input_value = read_f32(input, &input_indices, ATAN2_OPERATION_ID)?;
        let other_value = read_f32(other, &other_indices, ATAN2_OPERATION_ID)?;
        let input_tangent = read_f32(input_tangent, &input_indices, ATAN2_OPERATION_ID)?;
        let other_tangent = read_f32(other_tangent, &other_indices, ATAN2_OPERATION_ID)?;
        values.try_push(
            (other_value * input_tangent - input_value * other_tangent)
                / (input_value * input_value + other_value * other_value),
        )?;
    }
    upload_f32(backend, &shape, &values, context)
}

pub fn baddbmm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    batch1: &Tensor,
    batch2: &Tensor,
    beta: f32,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    let geometry = BaddbmmGeometry::new(input, batch1, batch2)?;
    context.check()?;
    let descriptor = TensorDescriptor::contiguous(
        geometry.output_shape,
        DType::F32,
        DeviceId::CPU,
        batch1.descriptor().stream(),
    )?;
    let product = backend
        .linear_algebra(
            LinearAlgebraOperation::BatchMatrixMultiply,
            &[batch1.clone(), batch2.clone()],
            descriptor.clone(),
            context,
        )?
        .0;
    let product = scale_tensor(backend, &product, alpha, context)?;
    if beta == 0.0 {
        context.check()?;
        return Ok(product);
    }
    let input = scale_tensor(backend, input, beta, context)?;
    Ok(backend
        .binary(BinaryOperation::Add, &input, &product, descriptor, context)?
        .0)
}

#[allow(clippy::too_many_arguments)]
pub fn baddbmm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    batch1: &Tensor,
    batch2: &Tensor,
    beta: f32,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<BaddbmmGradients, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    let geometry = BaddbmmGeometry::new(input, batch1, batch2)?;
    require_shape_stream(
        output_gradient,
        &geometry.output_shape,
        batch1.descriptor().stream(),
        BADDBMM_OPERATION_ID,
    )?;
    require_f32_cpu(output_gradient, BADDBMM_OPERATION_ID)?;
    let mut input_gradient = zero_f32_values(backend, input.descriptor().shape(), context)?;
    let mut batch1_gradient = zero_f32_values(backend, batch1.descriptor().shape(), context)?;
    let mut batch2_gradient = zero_f32_values(backend, batch2.descriptor().shape(), context)?;
    for batch in 0..geometry.batch {
        for row in 0..geometry.rows {
            for column in 0..geometry.columns {
                check_periodically(
                    (batch * geometry.rows + row) * geometry.columns + column,
                    context.cancellation,
                )?;
                let output_indices = [as_u64(batch)?, as_u64(row)?, as_u64(column)?];
                let gradient = read_f32(output_gradient, &output_indices, BADDBMM_OPERATION_ID)?;
                let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
                input_gradient[ravel_index(&input_indices, input.descriptor().shape())?] +=
                    beta * gradient;
                for inner in 0..geometry.inner {
                    let batch1_indices = [as_u64(batch)?, as_u64(row)?, as_u64(inner)?];
                    let batch2_indices = [as_u64(batch)?, as_u64(inner)?, as_u64(column)?];
                    batch1_gradient[ravel_index(&batch1_indices, batch1.descriptor().shape())?] +=
                        alpha * gradient * read_f32(batch2, &batch2_indices, BADDBMM_OPERATION_ID)?;
                    batch2_gradient[ravel_index(&batch2_indices, batch2.descriptor().shape())?] +=
                        alpha * gradient * read_f32(batch1, &batch1_indices, BADDBMM_OPERATION_ID)?;
                }
            }
        }
    }
    Ok(BaddbmmGradients {
        input: upload_f32(
            backend,
            input.descriptor().shape(),
            &input_gradient,
            context,
        )?,
        batch1: upload_f32(
            backend,
            batch1.descriptor().shape(),
            &batch1_gradient,
            context,
        )?,
        batch2: upload_f32(
            backend,
            batch2.descriptor().shape(),
            &batch2_gradient,
            context,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn baddbmm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    batch1: &Tensor,
    batch2: &Tensor,
    input_tangent: &Tensor,
    batch1_tangent: &Tensor,
    batch2_tangent: &Tensor,
    beta: f32,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    let geometry = BaddbmmGeometry::new(input, batch1, batch2)?;
    require_tangent(input, input_tangent, BADDBMM_OPERATION_ID)?;
    require_tangent(batch1, batch1_tangent, BADDBMM_OPERATION_ID)?;
    require_tangent(batch2, batch2_tangent, BADDBMM_OPERATION_ID)?;
    let mut values = backend.workspace_vec(context, element_count(&geometry.output_shape)?)?;
    for batch in 0..geometry.batch {
        for row in 0..geometry.rows {
            for column in 0..geometry.columns {
                check_periodically(values.len(), context.cancellation)?;
                let output_indices = [as_u64(batch)?, as_u64(row)?, as_u64(column)?];
                let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
                let mut value =
                    beta * read_f32(input_tangent, &input_indices, BADDBMM_OPERATION_ID)?;
                for inner in 0..geometry.inner {
                    let batch1_indices = [as_u64(batch)?, as_u64(row)?, as_u64(inner)?];
                    let batch2_indices = [as_u64(batch)?, as_u64(inner)?, as_u64(column)?];
                    value += alpha
                        * (read_f32(batch1_tangent, &batch1_indices, BADDBMM_OPERATION_ID)?
                            * read_f32(batch2, &batch2_indices, BADDBMM_OPERATION_ID)?
                            + read_f32(batch1, &batch1_indices, BADDBMM_OPERATION_ID)?
                                * read_f32(batch2_tangent, &batch2_indices, BADDBMM_OPERATION_ID)?);
                }
                values.try_push(value)?;
            }
        }
    }
    upload_f32(backend, &geometry.output_shape, &values, context)
}

pub fn ceil_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    Ok(canonical_ceil_exact_native(backend, input, context)?)
}

pub fn diag_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_cpu(input, DIAG_OPERATION_ID)?;
    match input.descriptor().shape() {
        [length] => diag_from_vector(backend, input, *length, diagonal, context),
        [rows, columns] => diag_from_matrix(backend, input, *rows, *columns, diagonal, context),
        _ => Err(ElementwiseRuntimePartFifteenError::Invalid {
            operation: DIAG_OPERATION_ID,
            reason: "diag requires a rank-one or rank-two tensor".to_owned(),
        }),
    }
}

pub fn diag_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: i64,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    match input.descriptor().shape() {
        [length] => {
            let side = length.checked_add(diagonal.unsigned_abs()).ok_or(
                ElementwiseRuntimePartFifteenError::ShapeOverflow("diag VJP matrix side"),
            )?;
            require_tensor_contract(
                output_gradient,
                &[side, side],
                input.descriptor().dtype(),
                input.descriptor().stream(),
                DIAG_OPERATION_ID,
            )?;
            diag_with_context_exact_native(backend, output_gradient, diagonal, context)
        }
        [rows, columns] => {
            let (_, _, length) = diagonal_geometry(*rows, *columns, diagonal)?;
            require_tensor_contract(
                output_gradient,
                &[length],
                input.descriptor().dtype(),
                input.descriptor().stream(),
                DIAG_OPERATION_ID,
            )?;
            diag_scatter_to_matrix(
                backend,
                output_gradient,
                *rows,
                *columns,
                diagonal,
                input.descriptor().dtype(),
                context,
            )
        }
        _ => Err(ElementwiseRuntimePartFifteenError::Invalid {
            operation: DIAG_OPERATION_ID,
            reason: "diag VJP requires a rank-one or rank-two input".to_owned(),
        }),
    }
}

pub fn diag_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: i64,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_tangent(input, input_tangent, DIAG_OPERATION_ID)?;
    diag_with_context_exact_native(backend, input_tangent, diagonal, context)
}

pub fn fliplr_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    if input.descriptor().rank() < 2 {
        return Err(ElementwiseRuntimePartFifteenError::Invalid {
            operation: FLIPLR_OPERATION_ID,
            reason: "fliplr requires rank at least two".to_owned(),
        });
    }
    flip_dimensions_with_context_exact_native(backend, input, &[1], FLIPLR_OPERATION_ID, context)
}

pub fn flip_dimensions_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    let shape = input.descriptor().shape();
    let mut axes = Vec::new();
    axes.try_reserve_exact(dimensions.len())
        .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("flip dimensions"))?;
    for dimension in dimensions {
        let axis = normalize_axis(*dimension, shape.len(), operation)?;
        if axes.contains(&axis) {
            return Err(ElementwiseRuntimePartFifteenError::Invalid {
                operation,
                reason: format!("flip dimension {dimension} is repeated"),
            });
        }
        axes.push(axis);
    }
    let mut bytes = backend.workspace_vec(context, byte_len(input.descriptor())?)?;
    for linear in 0..element_count(shape)? {
        check_periodically(linear, context.cancellation)?;
        let mut indices = unravel_index(linear, shape)?;
        for axis in &axes {
            let index = indices.get_mut(*axis).ok_or_else(|| {
                ElementwiseRuntimePartFifteenError::Invalid {
                    operation,
                    reason: "flip rank changed after validation".to_owned(),
                }
            })?;
            *index = shape[*axis]
                .checked_sub(1)
                .and_then(|last| last.checked_sub(*index))
                .ok_or(ElementwiseRuntimePartFifteenError::ShapeOverflow(
                    "flip index",
                ))?;
        }
        for byte in input.element_bytes(&indices)? {
            bytes.try_push(*byte)?;
        }
    }
    upload_bytes(backend, shape, input.descriptor().dtype(), &bytes, context)
}

pub fn fliplr_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    fliplr_with_context_exact_native(backend, output_gradient, context)
}

pub fn fliplr_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    fliplr_with_context_exact_native(backend, input_tangent, context)
}

pub fn ones_like_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    Ok(full_like_with_context_exact_native(
        backend,
        input,
        Scalar::Float(1.0),
        dtype,
        context,
    )?)
}

pub fn searchsorted_with_context_exact_native(
    backend: &CpuBackend,
    sorted_sequence: &Tensor,
    values: &Tensor,
    right: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_binary_f32(sorted_sequence, values, SEARCHSORTED_OPERATION_ID)?;
    let geometry = SearchSortedGeometry::new(sorted_sequence, values)?;
    let byte_count = element_count(values.descriptor().shape())?
        .checked_mul(8)
        .ok_or(ElementwiseRuntimePartFifteenError::ShapeOverflow(
            "searchsorted output",
        ))?;
    let mut encoded = backend.workspace_vec(context, byte_count)?;
    for linear in 0..element_count(values.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let value_indices = unravel_index(linear, values.descriptor().shape())?;
        let value = read_f32(values, &value_indices, SEARCHSORTED_OPERATION_ID)?;
        let row = geometry.row_for_values(&value_indices)?;
        let index = search_row(
            sorted_sequence,
            &row,
            geometry.boundaries,
            value,
            right,
            context.cancellation,
        )?;
        for byte in DType::I64.encode_scalar(
            Scalar::Signed(i64::try_from(index).map_err(|_| {
                ElementwiseRuntimePartFifteenError::ShapeOverflow("searchsorted index")
            })?),
            SEARCHSORTED_OPERATION_ID,
            DeviceId::CPU,
        )? {
            encoded.try_push(byte)?;
        }
    }
    upload_bytes(
        backend,
        values.descriptor().shape(),
        DType::I64,
        &encoded,
        context,
    )
}

pub fn tan_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, TAN_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(UnaryOperation::Tangent, input, descriptor, context)?
        .0)
}

pub fn tan_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    require_tangent(input, output_gradient, TAN_OPERATION_ID)?;
    let mut values = backend.workspace_vec(context, element_count(input.descriptor().shape())?)?;
    for linear in 0..element_count(input.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        let cosine = read_f32(input, &indices, TAN_OPERATION_ID)?.cos();
        values
            .try_push(read_f32(output_gradient, &indices, TAN_OPERATION_ID)? / (cosine * cosine))?;
    }
    upload_f32(backend, input.descriptor().shape(), &values, context)
}

pub fn tan_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    context.cancellation.check()?;
    tan_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

#[derive(Clone, Debug)]
struct BaddbmmGeometry {
    batch: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    output_shape: Vec<u64>,
}

impl BaddbmmGeometry {
    fn new(
        input: &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
    ) -> Result<Self, ElementwiseRuntimePartFifteenError> {
        require_binary_f32(batch1, batch2, BADDBMM_OPERATION_ID)?;
        require_f32_cpu(input, BADDBMM_OPERATION_ID)?;
        if input.descriptor().stream() != batch1.descriptor().stream() {
            return Err(ElementwiseRuntimePartFifteenError::Invalid {
                operation: BADDBMM_OPERATION_ID,
                reason: "input and batch matrices must use one stream".to_owned(),
            });
        }
        let [batch, rows, inner] = batch1.descriptor().shape() else {
            return Err(invalid(BADDBMM_OPERATION_ID, "batch1 must be rank three"));
        };
        let [other_batch, other_inner, columns] = batch2.descriptor().shape() else {
            return Err(invalid(BADDBMM_OPERATION_ID, "batch2 must be rank three"));
        };
        if batch != other_batch || inner != other_inner {
            return Err(invalid(
                BADDBMM_OPERATION_ID,
                "batch matrix dimensions are incompatible",
            ));
        }
        let output_shape = vec![*batch, *rows, *columns];
        if binary_broadcast_shape(input.descriptor().shape(), &output_shape)? != output_shape {
            return Err(invalid(
                BADDBMM_OPERATION_ID,
                "input is not broadcastable to the batch product",
            ));
        }
        Ok(Self {
            batch: as_usize(*batch)?,
            rows: as_usize(*rows)?,
            inner: as_usize(*inner)?,
            columns: as_usize(*columns)?,
            output_shape,
        })
    }
}

#[derive(Clone, Debug)]
struct SearchSortedGeometry {
    boundaries: u64,
    row_shape: Vec<u64>,
    one_dimensional: bool,
}

impl SearchSortedGeometry {
    fn new(
        sorted_sequence: &Tensor,
        values: &Tensor,
    ) -> Result<Self, ElementwiseRuntimePartFifteenError> {
        let boundaries = sorted_sequence
            .descriptor()
            .shape()
            .last()
            .copied()
            .ok_or_else(|| invalid(SEARCHSORTED_OPERATION_ID, "boundaries must have rank"))?;
        let one_dimensional = sorted_sequence.descriptor().shape().len() == 1;
        let row_shape = sorted_sequence
            .descriptor()
            .shape()
            .get(..sorted_sequence.descriptor().shape().len() - 1)
            .ok_or_else(|| invalid(SEARCHSORTED_OPERATION_ID, "invalid boundary rank"))?
            .to_vec();
        if !one_dimensional {
            let value_prefix = values
                .descriptor()
                .shape()
                .get(..values.descriptor().shape().len().saturating_sub(1))
                .ok_or_else(|| invalid(SEARCHSORTED_OPERATION_ID, "invalid values rank"))?;
            if value_prefix != row_shape {
                return Err(invalid(
                    SEARCHSORTED_OPERATION_ID,
                    "multi-dimensional boundaries and values require equal leading dimensions",
                ));
            }
        }
        Ok(Self {
            boundaries,
            row_shape,
            one_dimensional,
        })
    }

    fn row_for_values(
        &self,
        value_indices: &[u64],
    ) -> Result<Vec<u64>, ElementwiseRuntimePartFifteenError> {
        if self.one_dimensional {
            return Ok(Vec::new());
        }
        value_indices
            .get(..self.row_shape.len())
            .map(ToOwned::to_owned)
            .ok_or_else(|| invalid(SEARCHSORTED_OPERATION_ID, "value index rank changed"))
    }
}

fn search_row(
    sorted_sequence: &Tensor,
    row: &[u64],
    boundaries: u64,
    value: f32,
    right: bool,
    cancellation: &CancellationToken,
) -> Result<u64, ElementwiseRuntimePartFifteenError> {
    let mut previous = None;
    for column in 0..boundaries {
        check_periodically(as_usize(column)?, cancellation)?;
        let mut indices = row.to_vec();
        indices.push(column);
        let boundary = read_f32(sorted_sequence, &indices, SEARCHSORTED_OPERATION_ID)?;
        if boundary.is_nan() || previous.is_some_and(|previous| boundary < previous) {
            return Err(invalid(
                SEARCHSORTED_OPERATION_ID,
                "boundaries must be nondecreasing and contain no NaN",
            ));
        }
        previous = Some(boundary);
    }
    if value.is_nan() {
        return Ok(boundaries);
    }
    let mut low = 0;
    let mut high = boundaries;
    while low < high {
        cancellation.check()?;
        let middle = low + (high - low) / 2;
        let mut indices = row.to_vec();
        indices.push(middle);
        let boundary = read_f32(sorted_sequence, &indices, SEARCHSORTED_OPERATION_ID)?;
        let advance = if right {
            boundary <= value
        } else {
            boundary < value
        };
        if advance {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

fn diag_from_vector(
    backend: &CpuBackend,
    input: &Tensor,
    length: u64,
    diagonal: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    let offset = diagonal.unsigned_abs();
    let side =
        length
            .checked_add(offset)
            .ok_or(ElementwiseRuntimePartFifteenError::ShapeOverflow(
                "diag matrix side",
            ))?;
    diag_scatter_to_matrix(
        backend,
        input,
        side,
        side,
        diagonal,
        input.descriptor().dtype(),
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn diag_scatter_to_matrix(
    backend: &CpuBackend,
    input: &Tensor,
    rows: u64,
    columns: u64,
    diagonal: i64,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    require_cpu(input, DIAG_OPERATION_ID)?;
    if input.descriptor().shape().len() != 1 || input.descriptor().dtype() != dtype {
        return Err(invalid(
            DIAG_OPERATION_ID,
            "matrix construction requires a matching rank-one tensor",
        ));
    }
    let (row_start, column_start, diagonal_length) = diagonal_geometry(rows, columns, diagonal)?;
    if input.descriptor().shape()[0] != diagonal_length {
        return Err(invalid(
            DIAG_OPERATION_ID,
            "diagonal gradient length does not match the target matrix",
        ));
    }
    let descriptor =
        TensorDescriptor::contiguous(vec![rows, columns], dtype, DeviceId::CPU, context.stream)?;
    let mut bytes = workspace_filled(backend, context, byte_len(&descriptor)?, 0_u8)?;
    let width = as_usize(dtype.byte_width())?;
    let columns_usize = as_usize(columns)?;
    for index in 0..as_usize(diagonal_length)? {
        check_periodically(index, context.cancellation)?;
        let row = as_usize(row_start)? + index;
        let column = as_usize(column_start)? + index;
        let destination = row
            .checked_mul(columns_usize)
            .and_then(|value| value.checked_add(column))
            .and_then(|value| value.checked_mul(width))
            .ok_or(ElementwiseRuntimePartFifteenError::ShapeOverflow(
                "diag offset",
            ))?;
        let target = bytes.get_mut(destination..destination + width).ok_or(
            ElementwiseRuntimePartFifteenError::ShapeOverflow("diag destination"),
        )?;
        target.copy_from_slice(input.element_bytes(&[as_u64(index)?])?);
    }
    upload_bytes(backend, &[rows, columns], dtype, &bytes, context)
}

fn diag_from_matrix(
    backend: &CpuBackend,
    input: &Tensor,
    rows: u64,
    columns: u64,
    diagonal: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    let (row_start, column_start, length) = diagonal_geometry(rows, columns, diagonal)?;
    let width = as_usize(input.descriptor().dtype().byte_width())?;
    let byte_count = as_usize(length)?.checked_mul(width).ok_or(
        ElementwiseRuntimePartFifteenError::ShapeOverflow("diag vector"),
    )?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for index in 0..as_usize(length)? {
        check_periodically(index, context.cancellation)?;
        for byte in
            input.element_bytes(&[row_start + as_u64(index)?, column_start + as_u64(index)?])?
        {
            bytes.try_push(*byte)?;
        }
    }
    upload_bytes(
        backend,
        &[length],
        input.descriptor().dtype(),
        &bytes,
        context,
    )
}

fn diagonal_geometry(
    rows: u64,
    columns: u64,
    diagonal: i64,
) -> Result<(u64, u64, u64), ElementwiseRuntimePartFifteenError> {
    if diagonal >= 0 {
        let column_start = u64::try_from(diagonal)
            .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("diag offset"))?;
        Ok((
            0,
            column_start,
            rows.min(columns.saturating_sub(column_start)),
        ))
    } else {
        let row_start = diagonal.unsigned_abs();
        Ok((row_start, 0, rows.saturating_sub(row_start).min(columns)))
    }
}

fn scale_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .binary_scalar(
            BinaryOperation::Multiply,
            input,
            Scalar::Float(f64::from(scale)),
            ScalarSide::Right,
            descriptor,
            context,
        )?
        .0)
}

fn require_binary_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(invalid(operation, "inputs must use one stream"));
    }
    Ok(())
}

fn require_tangent(
    input: &Tensor,
    tangent: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    require_binary_f32(input, tangent, operation)?;
    if input.descriptor().shape() != tangent.descriptor().shape() {
        return Err(invalid(operation, "tangent shape must match input shape"));
    }
    Ok(())
}

fn require_shape_stream(
    tensor: &Tensor,
    shape: &[u64],
    stream: StreamId,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    if tensor.descriptor().shape() != shape || tensor.descriptor().stream() != stream {
        return Err(invalid(operation, "tensor shape or stream does not match"));
    }
    Ok(())
}

fn require_tensor_contract(
    tensor: &Tensor,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().shape() != shape
        || tensor.descriptor().dtype() != dtype
        || tensor.descriptor().stream() != stream
    {
        return Err(invalid(
            operation,
            "tensor shape, dtype, or stream does not match the forward contract",
        ));
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartFifteenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartFifteenError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    Ok(())
}

fn read_f32(
    tensor: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartFifteenError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartFifteenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        }),
    }
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFifteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartFifteenError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn zero_f32_values(
    backend: &CpuBackend,
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ElementwiseRuntimePartFifteenError> {
    let count = element_count(shape)?;
    workspace_filled(backend, context, count, 0.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartFifteenError> {
    let mut values = backend.workspace_vec(context, count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(value)?;
    }
    Ok(values)
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ElementwiseRuntimePartFifteenError> {
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("tensor rank"))?;
    let normalized = if dimension < 0 {
        dimension.checked_add(rank)
    } else {
        Some(dimension)
    };
    let normalized = normalized
        .filter(|axis| *axis >= 0 && *axis < rank)
        .ok_or_else(|| ElementwiseRuntimePartFifteenError::Invalid {
            operation,
            reason: format!("dimension {dimension} is out of range for rank {rank}"),
        })?;
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("normalized dimension"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartFifteenError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count.checked_mul(as_usize(*dimension)?).ok_or(
            ElementwiseRuntimePartFifteenError::ShapeOverflow("element count"),
        )
    })
}

fn byte_len(descriptor: &TensorDescriptor) -> Result<usize, ElementwiseRuntimePartFifteenError> {
    usize::try_from(descriptor.byte_len()?)
        .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("tensor bytes"))
}

fn as_usize(value: u64) -> Result<usize, ElementwiseRuntimePartFifteenError> {
    usize::try_from(value)
        .map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("dimension"))
}

fn as_u64(value: usize) -> Result<u64, ElementwiseRuntimePartFifteenError> {
    u64::try_from(value).map_err(|_| ElementwiseRuntimePartFifteenError::ShapeOverflow("index"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartFifteenError> {
    let mut indices = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        let dimension = as_usize(*dimension)?;
        if dimension == 0 {
            return Ok(indices);
        }
        indices[axis] = as_u64(linear % dimension)?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartFifteenError> {
    if indices.len() != shape.len() {
        return Err(invalid(
            ATAN2_OPERATION_ID,
            "index rank does not match shape",
        ));
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            if index >= dimension {
                return Err(invalid(ATAN2_OPERATION_ID, "index exceeds shape"));
            }
            linear
                .checked_mul(as_usize(*dimension)?)
                .and_then(|value| value.checked_add(as_usize(*index).ok()?))
                .ok_or(ElementwiseRuntimePartFifteenError::ShapeOverflow(
                    "linear index",
                ))
        })
}

fn invalid(
    operation: &'static str,
    reason: impl Into<String>,
) -> ElementwiseRuntimePartFifteenError {
    ElementwiseRuntimePartFifteenError::Invalid {
        operation,
        reason: reason.into(),
    }
}
