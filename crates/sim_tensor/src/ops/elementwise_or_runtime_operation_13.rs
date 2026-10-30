use std::cmp::Ordering;

use crate::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec,
    DType, DecodedScalar, DeterministicAlgorithmsPolicy, DeviceId, DeviceKind, ExecutionContext,
    Scalar, ScalarSide, Tensor, TensorBackend, TensorDescriptor, TensorError,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_activation_normalization_functional_01::{
        FunctionalError, softmax_jvp_with_context_exact_native,
        softmax_vjp_with_context_exact_native, softmax_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, linear_with_context_exact_native, linear_jvp_with_context_exact_native,
        linear_vjp_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_11::{
        ElementwiseRuntimePartElevenError, isclose_with_context_exact_native,
        sign_jvp_with_context_exact_native, sign_vjp_with_context_exact_native,
        sign_with_context_exact_native,
    },
};
use thiserror::Error;

pub const LERP_OPERATION_ID: &str = "COMFY-TENSOR-OP-9C679FFC6CCF";
pub const SIGN_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-9547009E77B6";
pub const ADDMM_OPERATION_ID: &str = "COMFY-TENSOR-OP-9443F2A50F6D";
pub const ALLCLOSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-9285B877ECB7";
pub const CUDA_BF16_OPERATION_ID: &str = "COMFY-TENSOR-OP-9CC2489AFA14";
pub const EMPTY_LIKE_OPERATION_ID: &str = "COMFY-TENSOR-OP-9B68023167CF";
pub const MLU_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-97058675DD67";
pub const MLU_MEM_INFO_OPERATION_ID: &str = "COMFY-TENSOR-OP-8F0ACDA02879";
pub const QUANTILE_OPERATION_ID: &str = "COMFY-TENSOR-OP-8F6CC1A0A7AC";
pub const SOFTMAX_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-9BE377B59853";
pub const DETERMINISTIC_ALGORITHMS_OPERATION_ID: &str = "COMFY-TENSOR-OP-999B3107D90B";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartThirteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartEleven(#[from] ElementwiseRuntimePartElevenError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error("elementwise/runtime part-thirteen operation was cancelled")]
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
    #[error("elementwise/runtime part-thirteen input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartThirteenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Debug)]
pub struct LerpVjp {
    pub input: Tensor,
    pub end: Tensor,
}

#[derive(Debug)]
pub struct AddmmVjp {
    pub input: Tensor,
    pub matrix1: Tensor,
    pub matrix2: Tensor,
}

pub fn lerp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    weight: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, LERP_OPERATION_ID)?;
    require_f32_cpu(end, LERP_OPERATION_ID)?;
    require_same_stream(input, end, LERP_OPERATION_ID)?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), end.descriptor().shape())?;
    let descriptor = || {
        TensorDescriptor::contiguous(
            shape.clone(),
            DType::F32,
            DeviceId::CPU,
            input.descriptor().stream(),
        )
    };
    let difference = backend
        .binary(
            BinaryOperation::Subtract,
            end,
            input,
            descriptor()?,
            context,
        )?
        .0;
    let scaled = backend
        .binary_scalar(
            BinaryOperation::Multiply,
            &difference,
            Scalar::Float(f64::from(weight)),
            ScalarSide::Right,
            descriptor()?,
            context,
        )?
        .0;
    Ok(backend
        .binary(BinaryOperation::Add, input, &scaled, descriptor()?, context)?
        .0)
}

pub fn lerp_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    weight: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<LerpVjp, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), end.descriptor().shape())?;
    require_f32_cpu(output_gradient, LERP_OPERATION_ID)?;
    require_shape(output_gradient, &shape, LERP_OPERATION_ID)?;
    require_same_stream(input, output_gradient, LERP_OPERATION_ID)?;
    require_same_stream(end, output_gradient, LERP_OPERATION_ID)?;
    let mut input_gradient = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0,
    )?;
    let mut end_gradient = workspace_filled(
        backend,
        context,
        element_count(end.descriptor().shape())?,
        0.0,
    )?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let end_indices = broadcast_indices(&output_indices, end.descriptor().shape())?;
        let gradient = read_f32(output_gradient, &output_indices, LERP_OPERATION_ID)?;
        input_gradient[ravel_index(&input_indices, input.descriptor().shape())?] +=
            gradient * (1.0 - weight);
        end_gradient[ravel_index(&end_indices, end.descriptor().shape())?] += gradient * weight;
    }
    Ok(LerpVjp {
        input: upload_f32(
            backend,
            input.descriptor().shape(),
            &input_gradient,
            context,
        )?,
        end: upload_f32(backend, end.descriptor().shape(), &end_gradient, context)?,
    })
}

pub fn lerp_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    input_tangent: &Tensor,
    end_tangent: &Tensor,
    weight: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_matching_tangent(input, input_tangent, LERP_OPERATION_ID)?;
    require_matching_tangent(end, end_tangent, LERP_OPERATION_ID)?;
    lerp_with_context_exact_native(backend, input_tangent, end_tangent, weight, context)
}

pub fn tensor_sign_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    Ok(sign_with_context_exact_native(backend, input, context)?)
}

pub fn tensor_sign_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    Ok(sign_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn tensor_sign_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    Ok(sign_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn addmm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    matrix1: &Tensor,
    matrix2: &Tensor,
    beta: f32,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let geometry = AddmmGeometry::new(input, matrix1, matrix2)?;
    let matrix1_values = logical_f32_values(backend, matrix1, ADDMM_OPERATION_ID, context)?;
    let matrix2_values = logical_f32_values(backend, matrix2, ADDMM_OPERATION_ID, context)?;
    let transposed = transpose_matrix(
        backend,
        &matrix2_values,
        geometry.inner,
        geometry.columns,
        context,
    )?;
    let product = linear_with_context_exact_native(
        &matrix1_values,
        &[geometry.rows, geometry.inner],
        &transposed,
        &[geometry.columns, geometry.inner],
        None,
        DeviceId::CPU,
        context,
    )?;
    let mut output = product.values;
    for (linear, value) in output.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.output_shape)?;
        let input_indices = broadcast_indices(&indices, input.descriptor().shape())?;
        *value = alpha.mul_add(
            *value,
            beta * read_f32(input, &input_indices, ADDMM_OPERATION_ID)?,
        );
    }
    upload_f32(backend, &geometry.output_shape, &output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn addmm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    matrix1: &Tensor,
    matrix2: &Tensor,
    beta: f32,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddmmVjp, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let geometry = AddmmGeometry::new(input, matrix1, matrix2)?;
    require_f32_cpu(output_gradient, ADDMM_OPERATION_ID)?;
    require_shape(output_gradient, &geometry.output_shape, ADDMM_OPERATION_ID)?;
    require_same_stream(matrix1, output_gradient, ADDMM_OPERATION_ID)?;
    let matrix1_values = logical_f32_values(backend, matrix1, ADDMM_OPERATION_ID, context)?;
    let matrix2_values = logical_f32_values(backend, matrix2, ADDMM_OPERATION_ID, context)?;
    let output_gradient_values =
        logical_f32_values(backend, output_gradient, ADDMM_OPERATION_ID, context)?;
    let transposed = transpose_matrix(
        backend,
        &matrix2_values,
        geometry.inner,
        geometry.columns,
        context,
    )?;
    let mut linear_gradients = linear_vjp_with_context_exact_native(
        &matrix1_values,
        &[geometry.rows, geometry.inner],
        &transposed,
        &[geometry.columns, geometry.inner],
        None,
        &output_gradient_values,
        DeviceId::CPU,
        context,
    )?;
    let mut input_gradient = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0,
    )?;
    for (linear, gradient) in output_gradient_values.iter().copied().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.output_shape)?;
        let input_indices = broadcast_indices(&indices, input.descriptor().shape())?;
        input_gradient[ravel_index(&input_indices, input.descriptor().shape())?] += beta * gradient;
    }
    for (index, value) in linear_gradients.input.iter_mut().enumerate() {
        check_periodically(index, context.cancellation)?;
        *value *= alpha;
    }
    for (index, value) in linear_gradients.weight.iter_mut().enumerate() {
        check_periodically(index, context.cancellation)?;
        *value *= alpha;
    }
    let matrix2_gradient = transpose_matrix(
        backend,
        &linear_gradients.weight,
        geometry.columns,
        geometry.inner,
        context,
    )?;
    Ok(AddmmVjp {
        input: upload_f32(
            backend,
            input.descriptor().shape(),
            &input_gradient,
            context,
        )?,
        matrix1: upload_f32(
            backend,
            matrix1.descriptor().shape(),
            &linear_gradients.input,
            context,
        )?,
        matrix2: upload_f32(
            backend,
            matrix2.descriptor().shape(),
            &matrix2_gradient,
            context,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn addmm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    matrix1: &Tensor,
    matrix2: &Tensor,
    input_tangent: &Tensor,
    matrix1_tangent: &Tensor,
    matrix2_tangent: &Tensor,
    beta: f32,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let geometry = AddmmGeometry::new(input, matrix1, matrix2)?;
    require_matching_tangent(input, input_tangent, ADDMM_OPERATION_ID)?;
    require_matching_tangent(matrix1, matrix1_tangent, ADDMM_OPERATION_ID)?;
    require_matching_tangent(matrix2, matrix2_tangent, ADDMM_OPERATION_ID)?;
    let matrix1_values = logical_f32_values(backend, matrix1, ADDMM_OPERATION_ID, context)?;
    let matrix1_tangent_values =
        logical_f32_values(backend, matrix1_tangent, ADDMM_OPERATION_ID, context)?;
    let matrix2_values = logical_f32_values(backend, matrix2, ADDMM_OPERATION_ID, context)?;
    let matrix2_tangent_values =
        logical_f32_values(backend, matrix2_tangent, ADDMM_OPERATION_ID, context)?;
    let transposed = transpose_matrix(
        backend,
        &matrix2_values,
        geometry.inner,
        geometry.columns,
        context,
    )?;
    let transposed_tangent = transpose_matrix(
        backend,
        &matrix2_tangent_values,
        geometry.inner,
        geometry.columns,
        context,
    )?;
    let product_tangent = linear_jvp_with_context_exact_native(
        &matrix1_values,
        &matrix1_tangent_values,
        &[geometry.rows, geometry.inner],
        &transposed,
        &transposed_tangent,
        &[geometry.columns, geometry.inner],
        None,
        None,
        DeviceId::CPU,
        context,
    )?;
    let mut output = product_tangent.values;
    for (linear, value) in output.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.output_shape)?;
        let input_indices = broadcast_indices(&indices, input_tangent.descriptor().shape())?;
        *value = alpha.mul_add(
            *value,
            beta * read_f32(input_tangent, &input_indices, ADDMM_OPERATION_ID)?,
        );
    }
    upload_f32(backend, &geometry.output_shape, &output, context)
}

pub fn allclose_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    relative_tolerance: f32,
    absolute_tolerance: f32,
    equal_nan: bool,
    context: &ExecutionContext<'_>,
) -> Result<bool, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let close = isclose_with_context_exact_native(
        backend,
        input,
        other,
        relative_tolerance,
        absolute_tolerance,
        equal_nan,
        context,
    )?;
    for linear in 0..element_count(close.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, close.descriptor().shape())?;
        match close
            .descriptor()
            .dtype()
            .decode_scalar(close.element_bytes(&indices)?)?
        {
            DecodedScalar::Boolean(true) => {}
            DecodedScalar::Boolean(false) => return Ok(false),
            _ => {
                return Err(ElementwiseRuntimePartThirteenError::UnsupportedDType {
                    operation: ALLCLOSE_OPERATION_ID,
                    dtype: close.descriptor().dtype(),
                });
            }
        }
    }
    Ok(true)
}

pub fn cuda_is_bf16_supported_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartThirteenError> {
    cancellation.check()?;
    let device = capabilities.device();
    if device.kind() != DeviceKind::Cuda {
        return Err(ElementwiseRuntimePartThirteenError::UnsupportedDevice {
            operation: CUDA_BF16_OPERATION_ID,
            device,
        });
    }
    let supported = capabilities.supports_dtype(DType::Bf16);
    cancellation.check()?;
    Ok(supported)
}

pub fn empty_like_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: Option<DType>,
    device: Option<DeviceId>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let device = device.unwrap_or(input.descriptor().device());
    if device != DeviceId::CPU {
        return Err(ElementwiseRuntimePartThirteenError::UnsupportedDevice {
            operation: EMPTY_LIKE_OPERATION_ID,
            device,
        });
    }
    let descriptor = input
        .descriptor()
        .preserving_format_for(dtype.unwrap_or(input.descriptor().dtype()), device)?;
    Ok(backend.allocate(descriptor, context)?.0)
}

pub fn quantile_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    quantile: f32,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let plan = QuantilePlan::new(input, quantile, dimension)?;
    let selections = plan.selections(backend, input, context)?;
    let mut values = backend.workspace_vec(context, selections.len())?;
    for selection in selections.iter() {
        values.try_push(selection.value)?;
    }
    upload_f32(backend, &plan.output_shape, &values, context)
}

pub fn quantile_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    quantile: f32,
    dimension: Option<i64>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    let plan = QuantilePlan::new(input, quantile, dimension)?;
    require_f32_cpu(output_gradient, QUANTILE_OPERATION_ID)?;
    require_shape(output_gradient, &plan.output_shape, QUANTILE_OPERATION_ID)?;
    require_same_stream(input, output_gradient, QUANTILE_OPERATION_ID)?;
    let selections = plan.selections(backend, input, context)?;
    let mut gradient = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0,
    )?;
    for (group, selection) in selections.iter().enumerate() {
        check_periodically(group, context.cancellation)?;
        let output_indices = unravel_index(group, &plan.output_shape)?;
        let output_gradient = read_f32(output_gradient, &output_indices, QUANTILE_OPERATION_ID)?;
        gradient[ravel_index(&selection.lower_indices, input.descriptor().shape())?] +=
            output_gradient * (1.0 - selection.upper_weight);
        gradient[ravel_index(&selection.upper_indices, input.descriptor().shape())?] +=
            output_gradient * selection.upper_weight;
    }
    upload_f32(backend, input.descriptor().shape(), &gradient, context)
}

pub fn quantile_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    quantile: f32,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_matching_tangent(input, input_tangent, QUANTILE_OPERATION_ID)?;
    let plan = QuantilePlan::new(input, quantile, dimension)?;
    let selections = plan.selections(backend, input, context)?;
    let mut values = backend.workspace_vec(context, selections.len())?;
    for (index, selection) in selections.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        let lower = read_f32(
            input_tangent,
            &selection.lower_indices,
            QUANTILE_OPERATION_ID,
        )?;
        let upper = read_f32(
            input_tangent,
            &selection.upper_indices,
            QUANTILE_OPERATION_ID,
        )?;
        values.try_push(lower + selection.upper_weight * (upper - lower))?;
    }
    upload_f32(backend, &plan.output_shape, &values, context)
}

pub fn softmax_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, SOFTMAX_FUNCTION_OPERATION_ID)?;
    let shape = usize_shape(input.descriptor().shape())?;
    let values = logical_f32_values(backend, input, SOFTMAX_FUNCTION_OPERATION_ID, context)?;
    let output = softmax_with_context_exact_native(
        backend,
        &values,
        &shape,
        isize::try_from(dimension).map_err(|_| {
            ElementwiseRuntimePartThirteenError::Invalid(
                "softmax dimension is out of range".to_owned(),
            )
        })?,
        DeviceId::CPU,
        context,
    )?;
    upload_f32(backend, input.descriptor().shape(), &output, context)
}

pub fn softmax_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_matching_tangent(input, output_gradient, SOFTMAX_FUNCTION_OPERATION_ID)?;
    let shape = usize_shape(input.descriptor().shape())?;
    let values = logical_f32_values(backend, input, SOFTMAX_FUNCTION_OPERATION_ID, context)?;
    let gradient = logical_f32_values(
        backend,
        output_gradient,
        SOFTMAX_FUNCTION_OPERATION_ID,
        context,
    )?;
    let output = softmax_vjp_with_context_exact_native(
        backend,
        &values,
        &gradient,
        &shape,
        isize::try_from(dimension).map_err(|_| {
            ElementwiseRuntimePartThirteenError::Invalid(
                "softmax dimension is out of range".to_owned(),
            )
        })?,
        DeviceId::CPU,
        context,
    )?;
    upload_f32(backend, input.descriptor().shape(), &output, context)
}

pub fn softmax_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    context.cancellation.check()?;
    require_matching_tangent(input, input_tangent, SOFTMAX_FUNCTION_OPERATION_ID)?;
    let shape = usize_shape(input.descriptor().shape())?;
    let values = logical_f32_values(backend, input, SOFTMAX_FUNCTION_OPERATION_ID, context)?;
    let tangent = logical_f32_values(
        backend,
        input_tangent,
        SOFTMAX_FUNCTION_OPERATION_ID,
        context,
    )?;
    let output = softmax_jvp_with_context_exact_native(
        backend,
        &values,
        &tangent,
        &shape,
        isize::try_from(dimension).map_err(|_| {
            ElementwiseRuntimePartThirteenError::Invalid(
                "softmax dimension is out of range".to_owned(),
            )
        })?,
        DeviceId::CPU,
        context,
    )?;
    upload_f32(backend, input.descriptor().shape(), &output, context)
}

pub fn use_deterministic_algorithms_exact_native(
    enabled: bool,
    warn_only: bool,
    cancellation: &CancellationToken,
) -> Result<DeterministicAlgorithmsPolicy, ElementwiseRuntimePartThirteenError> {
    cancellation.check()?;
    let policy = DeterministicAlgorithmsPolicy::new(enabled, warn_only);
    cancellation.check()?;
    Ok(policy)
}

struct AddmmGeometry {
    rows: usize,
    inner: usize,
    columns: usize,
    output_shape: Vec<u64>,
}

impl AddmmGeometry {
    fn new(
        input: &Tensor,
        matrix1: &Tensor,
        matrix2: &Tensor,
    ) -> Result<Self, ElementwiseRuntimePartThirteenError> {
        for tensor in [input, matrix1, matrix2] {
            require_f32_cpu(tensor, ADDMM_OPERATION_ID)?;
            require_same_stream(matrix1, tensor, ADDMM_OPERATION_ID)?;
        }
        if matrix1.descriptor().rank() != 2 || matrix2.descriptor().rank() != 2 {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "addmm matrix operands must both have rank two".to_owned(),
            ));
        }
        let rows = usize::try_from(matrix1.descriptor().shape()[0])
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm rows"))?;
        let inner = usize::try_from(matrix1.descriptor().shape()[1])
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm inner"))?;
        let matrix2_inner = usize::try_from(matrix2.descriptor().shape()[0])
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm inner"))?;
        let columns = usize::try_from(matrix2.descriptor().shape()[1])
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm columns"))?;
        if inner != matrix2_inner {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "addmm inner dimensions do not match".to_owned(),
            ));
        }
        let output_shape = vec![
            u64::try_from(rows)
                .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm output"))?,
            u64::try_from(columns)
                .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("addmm output"))?,
        ];
        if binary_broadcast_shape(input.descriptor().shape(), &output_shape)? != output_shape {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "addmm input is not broadcast-compatible with the matrix product".to_owned(),
            ));
        }
        Ok(Self {
            rows,
            inner,
            columns,
            output_shape,
        })
    }
}

struct QuantilePlan {
    quantile: f32,
    axis: Option<usize>,
    output_shape: Vec<u64>,
    group_count: usize,
}

#[derive(Clone)]
struct QuantileSelection {
    value: f32,
    lower_indices: Vec<u64>,
    upper_indices: Vec<u64>,
    upper_weight: f32,
}

#[derive(Clone)]
struct QuantileEntry {
    value: f32,
    indices: Vec<u64>,
    order: usize,
}

impl QuantilePlan {
    fn new(
        input: &Tensor,
        quantile: f32,
        dimension: Option<i64>,
    ) -> Result<Self, ElementwiseRuntimePartThirteenError> {
        require_f32_cpu(input, QUANTILE_OPERATION_ID)?;
        if !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "quantile must be finite and in the inclusive range [0, 1]".to_owned(),
            ));
        }
        if input.descriptor().element_count()? == 0 {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "quantile requires a non-empty input".to_owned(),
            ));
        }
        let axis = dimension
            .map(|dimension| normalize_axis(dimension, input.descriptor().rank()))
            .transpose()?;
        let mut output_shape = input.descriptor().shape().to_vec();
        if let Some(axis) = axis {
            output_shape.remove(axis);
        } else {
            output_shape.clear();
        }
        let group_count = element_count(&output_shape)?;
        Ok(Self {
            quantile,
            axis,
            output_shape,
            group_count,
        })
    }

    fn selections(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<CpuWorkspaceVec<QuantileSelection>, ElementwiseRuntimePartThirteenError> {
        let mut selections = backend.workspace_vec(context, self.group_count)?;
        for group in 0..self.group_count {
            check_periodically(group, context.cancellation)?;
            let output_indices = unravel_index(group, &self.output_shape)?;
            let mut entries = self.entries(backend, input, &output_indices, context)?;
            if let Some(entry) = entries.iter().find(|entry| entry.value.is_nan()) {
                selections.try_push(QuantileSelection {
                    value: f32::NAN,
                    lower_indices: entry.indices.clone(),
                    upper_indices: entry.indices.clone(),
                    upper_weight: 0.0,
                })?;
                continue;
            }
            entries.sort_by(|left, right| {
                left.value
                    .partial_cmp(&right.value)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.order.cmp(&right.order))
            });
            let last = entries.len().checked_sub(1).ok_or_else(|| {
                ElementwiseRuntimePartThirteenError::Invalid(
                    "quantile reduction group is empty".to_owned(),
                )
            })?;
            let position = self.quantile * last as f32;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            let upper_weight = position - lower as f32;
            let lower_entry = entries.get(lower).ok_or_else(|| {
                ElementwiseRuntimePartThirteenError::Invalid(
                    "quantile lower rank is outside the group".to_owned(),
                )
            })?;
            let upper_entry = entries.get(upper).ok_or_else(|| {
                ElementwiseRuntimePartThirteenError::Invalid(
                    "quantile upper rank is outside the group".to_owned(),
                )
            })?;
            selections.try_push(QuantileSelection {
                value: lower_entry.value + upper_weight * (upper_entry.value - lower_entry.value),
                lower_indices: lower_entry.indices.clone(),
                upper_indices: upper_entry.indices.clone(),
                upper_weight,
            })?;
        }
        Ok(selections)
    }

    fn entries(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        output_indices: &[u64],
        context: &ExecutionContext<'_>,
    ) -> Result<CpuWorkspaceVec<QuantileEntry>, ElementwiseRuntimePartThirteenError> {
        let width = self.axis.map_or_else(
            || element_count(input.descriptor().shape()),
            |axis| {
                usize::try_from(input.descriptor().shape()[axis]).map_err(|_| {
                    ElementwiseRuntimePartThirteenError::ShapeOverflow("quantile axis")
                })
            },
        )?;
        let mut entries = backend.workspace_vec(context, width)?;
        for order in 0..width {
            check_periodically(order, context.cancellation)?;
            let indices = if let Some(axis) = self.axis {
                let mut indices = output_indices.to_vec();
                indices.insert(
                    axis,
                    u64::try_from(order).map_err(|_| {
                        ElementwiseRuntimePartThirteenError::ShapeOverflow("quantile index")
                    })?,
                );
                indices
            } else {
                unravel_index(order, input.descriptor().shape())?
            };
            entries.try_push(QuantileEntry {
                value: read_f32(input, &indices, QUANTILE_OPERATION_ID)?,
                indices,
                order,
            })?;
        }
        Ok(entries)
    }
}

fn transpose_matrix(
    backend: &CpuBackend,
    values: &[f32],
    rows: usize,
    columns: usize,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ElementwiseRuntimePartThirteenError> {
    if values.len()
        != rows
            .checked_mul(columns)
            .ok_or(ElementwiseRuntimePartThirteenError::ShapeOverflow(
                "matrix transpose",
            ))?
    {
        return Err(ElementwiseRuntimePartThirteenError::Invalid(
            "matrix transpose input length does not match its shape".to_owned(),
        ));
    }
    let mut output = workspace_filled(backend, context, values.len(), 0.0)?;
    for row in 0..rows {
        check_periodically(row, context.cancellation)?;
        for column in 0..columns {
            output[column * rows + row] = values[row * columns + column];
        }
    }
    Ok(output)
}

fn logical_f32_values(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ElementwiseRuntimePartThirteenError> {
    require_f32_cpu(input, operation)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        values.try_push(read_f32(input, &indices, operation)?)?;
    }
    Ok(values)
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThirteenError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartThirteenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartThirteenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_shape(
    input: &Tensor,
    shape: &[u64],
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThirteenError> {
    if input.descriptor().shape() != shape {
        return Err(ElementwiseRuntimePartThirteenError::Invalid(format!(
            "operation {operation} expected shape {shape:?}, got {:?}",
            input.descriptor().shape()
        )));
    }
    Ok(())
}

fn require_same_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThirteenError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(ElementwiseRuntimePartThirteenError::Invalid(format!(
            "operation {operation} requires matching streams"
        )));
    }
    Ok(())
}

fn require_matching_tangent(
    input: &Tensor,
    tangent: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThirteenError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(tangent, operation)?;
    require_same_stream(input, tangent, operation)?;
    require_shape(tangent, input.descriptor().shape(), operation)
}

fn read_f32(
    input: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartThirteenError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartThirteenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        }),
    }
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartThirteenError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(ElementwiseRuntimePartThirteenError::ShapeOverflow(
            "element count",
        ))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartThirteenError> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let width = usize::try_from(shape[axis])
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("unravel dimension"))?;
        if width == 0 {
            return Ok(indices);
        }
        indices[axis] = u64::try_from(linear % width)
            .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("unravel index"))?;
        linear /= width;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartThirteenError> {
    if indices.len() != shape.len() {
        return Err(ElementwiseRuntimePartThirteenError::Invalid(
            "index rank does not match tensor rank".to_owned(),
        ));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(ElementwiseRuntimePartThirteenError::Invalid(
                "tensor index is outside its dimension".to_owned(),
            ));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or(ElementwiseRuntimePartThirteenError::ShapeOverflow(
                "ravel index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("ravel index"))
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
) -> Result<usize, ElementwiseRuntimePartThirteenError> {
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("tensor rank"))?;
    let normalized = if dimension < 0 {
        rank.checked_add(dimension)
    } else {
        Some(dimension)
    }
    .ok_or_else(|| {
        ElementwiseRuntimePartThirteenError::Invalid("dimension is out of range".to_owned())
    })?;
    if normalized < 0 || normalized >= rank {
        return Err(ElementwiseRuntimePartThirteenError::Invalid(
            "dimension is out of range".to_owned(),
        ));
    }
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("dimension"))
}

fn usize_shape(shape: &[u64]) -> Result<Vec<usize>, ElementwiseRuntimePartThirteenError> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| ElementwiseRuntimePartThirteenError::ShapeOverflow("usize shape"))
        })
        .collect()
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartThirteenError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThirteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartThirteenError> {
    let mut values = backend.workspace_vec(context, count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(value)?;
    }
    Ok(values)
}
