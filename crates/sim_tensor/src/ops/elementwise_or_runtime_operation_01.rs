use crate::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec,
    DType, DecodedScalar, DeviceId, ExecutionContext, FloatingPointInfo, NativeDeviceProperties,
    Scalar, ScalarSide, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const ABS_OPERATION_ID: &str = "COMFY-TENSOR-OP-0B05AB07BE66";
pub const DIM_OPERATION_ID: &str = "COMFY-TENSOR-OP-0DB870AE36B5";
pub const SUB_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0BDEE629B8C6";
pub const CUDA_GET_DEVICE_PROPERTIES_OPERATION_ID: &str = "COMFY-TENSOR-OP-0546BABACDB9";
pub const DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0B36DDFC0CD6";
pub const FINFO_OPERATION_ID: &str = "COMFY-TENSOR-OP-0A5DBFB907FD";
pub const MESHGRID_OPERATION_ID: &str = "COMFY-TENSOR-OP-07B99A0B13EF";
pub const SIGNBIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-015751FC6965";
pub const TRIU_OPERATION_ID: &str = "COMFY-TENSOR-OP-0678D863EEBA";
pub const VANDER_OPERATION_ID: &str = "COMFY-TENSOR-OP-010917B0D872";
pub const XPU_CURRENT_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-01475E433DB3";
pub const XPU_IS_BF16_SUPPORTED_OPERATION_ID: &str = "COMFY-TENSOR-OP-04A23E8A6156";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ElementwiseRuntimeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("elementwise/runtime operation was cancelled")]
    Cancelled,
    #[error("elementwise/runtime operation has invalid input: {0}")]
    Invalid(&'static str),
    #[error("elementwise/runtime shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
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
}

impl From<sim_types::CancellationError> for ElementwiseRuntimeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<crate::generated_sim_operator_indirection_01::OperatorIndirectionError>
    for ElementwiseRuntimeError
{
    fn from(
        error: crate::generated_sim_operator_indirection_01::OperatorIndirectionError,
    ) -> Self {
        map_operator_error(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshgridIndexing {
    Ij,
    Xy,
}

#[derive(Clone, Debug)]
pub struct SubtractVjp {
    pub input: Tensor,
    pub other: Tensor,
}

pub fn abs_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_cpu_tensor(ABS_OPERATION_ID, input)?;
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimeError::UnsupportedDType {
            operation: ABS_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let descriptor = contiguous_like(input, DType::F32)?;
    let (output, _event) = backend.unary(UnaryOperation::Absolute, input, descriptor, context)?;
    context.check()?;
    Ok(output)
}

pub fn abs_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_same_f32_shape(input, output_gradient, ABS_OPERATION_ID)?;
    let input_values = tensor_to_f32_workspace(backend, input, context)?;
    let gradient_values = tensor_to_f32_workspace(backend, output_gradient, context)?;
    let mut result = temporary_vec(backend, context, input_values.len(), "absolute gradient")?;
    for (index, (value, gradient)) in input_values.iter().zip(gradient_values.iter()).enumerate() {
        check_periodically(index, context.cancellation)?;
        let derivative = if *value == 0.0 { 0.0 } else { value.signum() };
        result.try_push(derivative * gradient)?;
    }
    upload_f32(backend, input.descriptor().shape(), &result, context)
}

pub fn abs_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    abs_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn dim_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<usize, ElementwiseRuntimeError> {
    cancellation.check()?;
    let rank = input.descriptor().rank();
    cancellation.check()?;
    Ok(rank)
}

pub fn subtract_in_place_with_context_exact_native<'a>(
    backend: &CpuBackend,
    input: &'a mut Tensor,
    other: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<&'a Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_same_f32_shape(input, other, SUB_IN_PLACE_OPERATION_ID)?;
    if !alpha.is_finite() {
        return Err(ElementwiseRuntimeError::Invalid(
            "subtraction alpha must be finite",
        ));
    }
    let scaled_other = if alpha == 1.0 {
        other.clone()
    } else {
        let descriptor = contiguous_like(other, DType::F32)?;
        let (scaled, _event) = backend.binary_scalar(
            BinaryOperation::Multiply,
            other,
            Scalar::Float(f64::from(alpha)),
            ScalarSide::Right,
            descriptor,
            context,
        )?;
        scaled
    };
    let descriptor = contiguous_like(input, DType::F32)?;
    let (staged, _event) = backend.binary(
        BinaryOperation::Subtract,
        input,
        &scaled_other,
        descriptor,
        context,
    )?;
    context.check()?;
    input.commit_in_place(staged)?;
    Ok(input)
}

pub fn subtract_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<SubtractVjp, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_cpu_tensor(SUB_IN_PLACE_OPERATION_ID, output_gradient)?;
    if output_gradient.descriptor().dtype() != DType::F32 || !alpha.is_finite() {
        return Err(ElementwiseRuntimeError::Invalid(
            "subtraction gradient requires f32 and finite alpha",
        ));
    }
    let descriptor = contiguous_like(output_gradient, DType::F32)?;
    let (other, _event) = backend.binary_scalar(
        BinaryOperation::Multiply,
        output_gradient,
        Scalar::Float(f64::from(-alpha)),
        ScalarSide::Right,
        descriptor,
        context,
    )?;
    let input = copy_tensor(backend, output_gradient, context)?;
    context.check()?;
    Ok(SubtractVjp { input, other })
}

pub fn subtract_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    let mut output = copy_tensor(backend, input_tangent, context)?;
    subtract_in_place_with_context_exact_native(
        backend,
        &mut output,
        other_tangent,
        alpha,
        context,
    )?;
    Ok(output)
}

pub fn cuda_get_device_properties_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeDeviceProperties, ElementwiseRuntimeError> {
    cancellation.check()?;
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm)
        || capabilities.device() != device
    {
        return Err(ElementwiseRuntimeError::UnsupportedDevice {
            operation: CUDA_GET_DEVICE_PROPERTIES_OPERATION_ID,
            device,
        });
    }
    let properties = capabilities.device_properties().cloned().ok_or(
        ElementwiseRuntimeError::UnsupportedDevice {
            operation: CUDA_GET_DEVICE_PROPERTIES_OPERATION_ID,
            device,
        },
    )?;
    cancellation.check()?;
    Ok(properties)
}

pub fn device_exact_native(
    source: &str,
    cancellation: &CancellationToken,
) -> Result<DeviceId, ElementwiseRuntimeError> {
    cancellation.check()?;
    let device = DeviceId::from_source_device(source)?;
    cancellation.check()?;
    Ok(device)
}

pub fn finfo_exact_native(
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<FloatingPointInfo, ElementwiseRuntimeError> {
    cancellation.check()?;
    let info = dtype.floating_point_info()?;
    cancellation.check()?;
    Ok(info)
}

pub fn meshgrid_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    indexing: MeshgridIndexing,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ElementwiseRuntimeError> {
    let geometry = MeshgridGeometry::new(inputs, indexing, context.cancellation)?;
    let output_count = geometry.output_element_count()?;
    let element_width = usize::try_from(geometry.dtype.byte_width())
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid element width"))?;
    let byte_count =
        output_count
            .checked_mul(element_width)
            .ok_or(ElementwiseRuntimeError::ShapeOverflow(
                "meshgrid output bytes",
            ))?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(inputs.len())
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid outputs"))?;
    for (input_index, input) in inputs.iter().enumerate() {
        let mut bytes = temporary_vec(backend, context, byte_count, "meshgrid output allocation")?;
        for linear_index in 0..output_count {
            check_periodically(linear_index, context.cancellation)?;
            let output_indices = unravel_index(linear_index, &geometry.output_shape)?;
            let source_index = geometry.source_index(input_index, &output_indices)?;
            let source_index = u64::try_from(source_index)
                .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid source index"))?;
            let source_indices = if input.descriptor().rank() == 0 {
                &[][..]
            } else {
                std::slice::from_ref(&source_index)
            };
            workspace_extend(&mut bytes, input.element_bytes(source_indices)?)?;
        }
        outputs.push(upload_contiguous(
            backend,
            &geometry.output_shape,
            geometry.dtype,
            geometry.stream,
            &bytes,
            context,
        )?);
    }
    context.check()?;
    Ok(outputs)
}

pub fn meshgrid_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangents: &[Tensor],
    indexing: MeshgridIndexing,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ElementwiseRuntimeError> {
    meshgrid_with_context_exact_native(backend, input_tangents, indexing, context)
}

pub fn meshgrid_vjp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    output_gradients: &[Tensor],
    indexing: MeshgridIndexing,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ElementwiseRuntimeError> {
    let geometry = MeshgridGeometry::new(inputs, indexing, context.cancellation)?;
    if inputs.len() != output_gradients.len() || geometry.dtype != DType::F32 {
        return Err(ElementwiseRuntimeError::Invalid(
            "meshgrid VJP requires one f32 output gradient per input",
        ));
    }
    let output_count = geometry.output_element_count()?;
    let mut results = Vec::new();
    results
        .try_reserve_exact(inputs.len())
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid VJP outputs"))?;
    for (input_index, (input, output_gradient)) in inputs.iter().zip(output_gradients).enumerate() {
        if output_gradient.descriptor().shape() != geometry.output_shape
            || output_gradient.descriptor().dtype() != DType::F32
        {
            return Err(ElementwiseRuntimeError::Invalid(
                "meshgrid output gradient descriptor mismatch",
            ));
        }
        let gradient_values = tensor_to_f32_workspace(backend, output_gradient, context)?;
        let input_length = geometry.input_lengths[input_index];
        let mut reduced = temporary_vec(backend, context, input_length, "meshgrid VJP reduction")?;
        for _ in 0..input_length {
            reduced.try_push(0.0)?;
        }
        for (linear_index, gradient) in gradient_values.iter().enumerate() {
            check_periodically(linear_index, context.cancellation)?;
            let output_indices = unravel_index(linear_index, &geometry.output_shape)?;
            let source_index = geometry.source_index(input_index, &output_indices)?;
            let target = reduced
                .get_mut(source_index)
                .ok_or(ElementwiseRuntimeError::Invalid(
                    "meshgrid VJP source index is out of range",
                ))?;
            *target += *gradient;
        }
        let shape = if input.descriptor().rank() == 0 {
            &[][..]
        } else {
            input.descriptor().shape()
        };
        results.push(upload_f32(backend, shape, &reduced, context)?);
        if output_count != gradient_values_len(output_gradient)? {
            return Err(ElementwiseRuntimeError::Invalid(
                "meshgrid output gradient element count mismatch",
            ));
        }
    }
    context.check()?;
    Ok(results)
}

pub fn signbit_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_cpu_tensor(SIGNBIT_OPERATION_ID, input)?;
    if input.descriptor().dtype().class() == crate::NumericClass::Complex {
        return Err(ElementwiseRuntimeError::UnsupportedDType {
            operation: SIGNBIT_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let element_count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("signbit values"))?;
    let mut bytes = temporary_vec(backend, context, element_count, "signbit output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let negative = match input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?
        {
            DecodedScalar::Signed(value) => value < 0,
            DecodedScalar::Real(value) => value.is_sign_negative(),
            DecodedScalar::Boolean(_) | DecodedScalar::Unsigned(_) => false,
            DecodedScalar::Complex { .. } => {
                return Err(ElementwiseRuntimeError::UnsupportedDType {
                    operation: SIGNBIT_OPERATION_ID,
                    dtype: input.descriptor().dtype(),
                });
            }
        };
        bytes.try_push(u8::from(negative))?;
    }
    upload_contiguous(
        backend,
        input.descriptor().shape(),
        DType::Bool,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn triu_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    triangular_mask_with_context_exact_native(backend, input, diagonal, true, context)
}

pub fn triangular_mask_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: isize,
    upper: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_cpu_tensor(TRIU_OPERATION_ID, input)?;
    let rank = input.descriptor().rank();
    if rank < 2 {
        return Err(ElementwiseRuntimeError::Invalid(
            "triu input rank must be at least two",
        ));
    }
    let element_count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("triu values"))?;
    let element_width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("triu element width"))?;
    let byte_count = element_count
        .checked_mul(element_width)
        .ok_or(ElementwiseRuntimeError::ShapeOverflow("triu output bytes"))?;
    let zero = input.descriptor().dtype().encode_scalar(
        Scalar::Float(0.0),
        TRIU_OPERATION_ID,
        DeviceId::CPU,
    )?;
    let mut bytes = temporary_vec(backend, context, byte_count, "triu output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let row = i128::from(indices[rank - 2]);
        let column = i128::from(indices[rank - 1]);
        let diagonal = diagonal as i128;
        let retain = if upper {
            column - row >= diagonal
        } else {
            column - row <= diagonal
        };
        if retain {
            workspace_extend(&mut bytes, input.element_bytes(&indices)?)?;
        } else {
            workspace_extend(&mut bytes, &zero)?;
        }
    }
    upload_contiguous(
        backend,
        input.descriptor().shape(),
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn triu_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    triu_with_context_exact_native(backend, output_gradient, diagonal, context)
}

pub fn triu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    triu_with_context_exact_native(backend, input_tangent, diagonal, context)
}

pub fn vander_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    columns: Option<usize>,
    increasing: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_cpu_tensor(VANDER_OPERATION_ID, input)?;
    if input.descriptor().rank() != 1
        || input.descriptor().dtype().class() != crate::NumericClass::FloatingPoint
    {
        return Err(ElementwiseRuntimeError::Invalid(
            "vander requires a one-dimensional floating-point tensor",
        ));
    }
    let rows = usize::try_from(input.descriptor().shape()[0])
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander rows"))?;
    let columns = columns.unwrap_or(rows);
    let value_count = rows
        .checked_mul(columns)
        .ok_or(ElementwiseRuntimeError::ShapeOverflow("vander values"))?;
    let element_width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander element width"))?;
    let byte_count =
        value_count
            .checked_mul(element_width)
            .ok_or(ElementwiseRuntimeError::ShapeOverflow(
                "vander output bytes",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_count, "vander output")?;
    for row in 0..rows {
        let row_index = u64::try_from(row)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander row index"))?;
        let value = match input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&[row_index])?)?
        {
            DecodedScalar::Real(value) => value,
            _ => {
                return Err(ElementwiseRuntimeError::UnsupportedDType {
                    operation: VANDER_OPERATION_ID,
                    dtype: input.descriptor().dtype(),
                });
            }
        };
        for column in 0..columns {
            let linear_index = row
                .checked_mul(columns)
                .and_then(|value| value.checked_add(column))
                .ok_or(ElementwiseRuntimeError::ShapeOverflow("vander index"))?;
            check_periodically(linear_index, context.cancellation)?;
            let exponent = if increasing {
                column
            } else {
                columns - 1 - column
            };
            let exponent = i32::try_from(exponent)
                .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander exponent"))?;
            workspace_extend(
                &mut bytes,
                &input.descriptor().dtype().encode_scalar(
                    Scalar::Float(value.powi(exponent)),
                    VANDER_OPERATION_ID,
                    DeviceId::CPU,
                )?,
            )?;
        }
    }
    let output_shape = [
        u64::try_from(rows).map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander rows"))?,
        u64::try_from(columns)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander columns"))?,
    ];
    upload_contiguous(
        backend,
        &output_shape,
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn vander_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    columns: Option<usize>,
    increasing: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    if input.descriptor().dtype() != DType::F32
        || output_gradient.descriptor().dtype() != DType::F32
        || input.descriptor().rank() != 1
    {
        return Err(ElementwiseRuntimeError::Invalid(
            "vander VJP requires f32 input and output gradient",
        ));
    }
    let rows = usize::try_from(input.descriptor().shape()[0])
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander VJP rows"))?;
    let columns = columns.unwrap_or(rows);
    let expected_shape = [
        u64::try_from(rows)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander VJP rows"))?,
        u64::try_from(columns)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander VJP columns"))?,
    ];
    if output_gradient.descriptor().shape() != expected_shape {
        return Err(ElementwiseRuntimeError::Invalid(
            "vander output gradient shape mismatch",
        ));
    }
    let input_values = tensor_to_f32_workspace(backend, input, context)?;
    let output_gradients = tensor_to_f32_workspace(backend, output_gradient, context)?;
    let mut input_gradients = temporary_vec(backend, context, rows, "vander VJP output")?;
    for _ in 0..rows {
        input_gradients.try_push(0.0)?;
    }
    for row in 0..rows {
        for column in 0..columns {
            let exponent = if increasing {
                column
            } else {
                columns - 1 - column
            };
            if exponent == 0 {
                continue;
            }
            let linear_index = row
                .checked_mul(columns)
                .and_then(|value| value.checked_add(column))
                .ok_or(ElementwiseRuntimeError::ShapeOverflow("vander VJP index"))?;
            check_periodically(linear_index, context.cancellation)?;
            let exponent_i32 = i32::try_from(exponent - 1)
                .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander VJP exponent"))?;
            input_gradients[row] += output_gradients[linear_index]
                * exponent as f32
                * input_values[row].powi(exponent_i32);
        }
    }
    upload_f32(
        backend,
        input.descriptor().shape(),
        &input_gradients,
        context,
    )
}

pub fn vander_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    columns: Option<usize>,
    increasing: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    context.cancellation.check()?;
    require_same_f32_shape(input, input_tangent, VANDER_OPERATION_ID)?;
    if input.descriptor().rank() != 1 {
        return Err(ElementwiseRuntimeError::Invalid(
            "vander JVP input rank must be one",
        ));
    }
    let rows = usize::try_from(input.descriptor().shape()[0])
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander JVP rows"))?;
    let columns = columns.unwrap_or(rows);
    let input_values = tensor_to_f32_workspace(backend, input, context)?;
    let tangent_values = tensor_to_f32_workspace(backend, input_tangent, context)?;
    let value_count = rows
        .checked_mul(columns)
        .ok_or(ElementwiseRuntimeError::ShapeOverflow("vander JVP values"))?;
    let mut values = temporary_vec(backend, context, value_count, "vander JVP output")?;
    for row in 0..rows {
        for column in 0..columns {
            let exponent = if increasing {
                column
            } else {
                columns - 1 - column
            };
            let value = if exponent == 0 {
                0.0
            } else {
                let exponent_i32 = i32::try_from(exponent - 1)
                    .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander JVP exponent"))?;
                exponent as f32 * input_values[row].powi(exponent_i32) * tangent_values[row]
            };
            values.try_push(value)?;
        }
    }
    let shape = [
        u64::try_from(rows)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander JVP rows"))?,
        u64::try_from(columns)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("vander JVP columns"))?,
    ];
    upload_f32(backend, &shape, &values, context)
}

pub fn xpu_current_device_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<u32, ElementwiseRuntimeError> {
    cancellation.check()?;
    let device = capabilities.device();
    if device.kind() != DeviceKind::Xpu {
        return Err(ElementwiseRuntimeError::UnsupportedDevice {
            operation: XPU_CURRENT_DEVICE_OPERATION_ID,
            device,
        });
    }
    cancellation.check()?;
    Ok(device.ordinal())
}

pub fn xpu_is_bf16_supported_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimeError> {
    cancellation.check()?;
    let device = capabilities.device();
    if device.kind() != DeviceKind::Xpu {
        return Err(ElementwiseRuntimeError::UnsupportedDevice {
            operation: XPU_IS_BF16_SUPPORTED_OPERATION_ID,
            device,
        });
    }
    let supported = capabilities.supports_dtype(DType::Bf16);
    cancellation.check()?;
    Ok(supported)
}

struct MeshgridGeometry {
    input_lengths: Vec<usize>,
    output_shape: Vec<u64>,
    dtype: DType,
    stream: crate::StreamId,
    indexing: MeshgridIndexing,
}

impl MeshgridGeometry {
    fn new(
        inputs: &[Tensor],
        indexing: MeshgridIndexing,
        cancellation: &CancellationToken,
    ) -> Result<Self, ElementwiseRuntimeError> {
        cancellation.check()?;
        let first = inputs.first().ok_or(ElementwiseRuntimeError::Invalid(
            "meshgrid inputs are empty",
        ))?;
        require_cpu_tensor(MESHGRID_OPERATION_ID, first)?;
        let dtype = first.descriptor().dtype();
        let stream = first.descriptor().stream();
        let mut input_lengths = Vec::new();
        input_lengths
            .try_reserve_exact(inputs.len())
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid input lengths"))?;
        for (index, input) in inputs.iter().enumerate() {
            check_periodically(index, cancellation)?;
            require_cpu_tensor(MESHGRID_OPERATION_ID, input)?;
            if input.descriptor().dtype() != dtype || input.descriptor().stream() != stream {
                return Err(ElementwiseRuntimeError::Invalid(
                    "meshgrid inputs must share dtype and stream",
                ));
            }
            let length = match input.descriptor().shape() {
                [] => 1,
                [length] => usize::try_from(*length)
                    .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid input"))?,
                _ => {
                    return Err(ElementwiseRuntimeError::Invalid(
                        "meshgrid inputs must be scalar or one-dimensional",
                    ));
                }
            };
            input_lengths.push(length);
        }
        let mut output_shape = input_lengths
            .iter()
            .map(|length| {
                u64::try_from(*length)
                    .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid shape"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if indexing == MeshgridIndexing::Xy && output_shape.len() >= 2 {
            output_shape.swap(0, 1);
        }
        Ok(Self {
            input_lengths,
            output_shape,
            dtype,
            stream,
            indexing,
        })
    }

    fn output_element_count(&self) -> Result<usize, ElementwiseRuntimeError> {
        self.output_shape.iter().try_fold(1_usize, |count, length| {
            let length = usize::try_from(*length)
                .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid values"))?;
            count
                .checked_mul(length)
                .ok_or(ElementwiseRuntimeError::ShapeOverflow("meshgrid values"))
        })
    }

    fn source_index(
        &self,
        input_index: usize,
        output_indices: &[u64],
    ) -> Result<usize, ElementwiseRuntimeError> {
        let output_dimension = match (self.indexing, input_index) {
            (MeshgridIndexing::Xy, 0) if self.input_lengths.len() >= 2 => 1,
            (MeshgridIndexing::Xy, 1) if self.input_lengths.len() >= 2 => 0,
            _ => input_index,
        };
        let source_index = output_indices.get(output_dimension).copied().ok_or(
            ElementwiseRuntimeError::Invalid("meshgrid output index rank mismatch"),
        )?;
        usize::try_from(source_index)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("meshgrid source index"))
    }
}

fn require_cpu_tensor(
    operation: &'static str,
    tensor: &Tensor,
) -> Result<(), ElementwiseRuntimeError> {
    if tensor.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimeError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        })
    }
}

fn require_same_f32_shape(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimeError> {
    require_cpu_tensor(operation, left)?;
    require_cpu_tensor(operation, right)?;
    if left.descriptor().dtype() != DType::F32 || right.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimeError::UnsupportedDType {
            operation,
            dtype: left.descriptor().dtype(),
        });
    }
    if left.descriptor().shape() != right.descriptor().shape()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(ElementwiseRuntimeError::Invalid(
            "tensor descriptors must share shape and stream",
        ));
    }
    Ok(())
}

fn contiguous_like(input: &Tensor, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        dtype,
        input.descriptor().device(),
        input.descriptor().stream(),
    )
}

fn upload_contiguous(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: crate::StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    let (output, _event) = backend.upload_bytes(descriptor, bytes, context)?;
    context.check()?;
    Ok(output)
}

fn copy_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    require_cpu_tensor(SUB_IN_PLACE_OPERATION_ID, input)?;
    let descriptor = contiguous_like(input, input.descriptor().dtype())?;
    let (output, _event) = backend.copy(input, descriptor, context)?;
    context.check()?;
    Ok(output)
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimeError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    let (output, _event) = backend.upload_f32(descriptor, values, context)?;
    context.check()?;
    Ok(output)
}

fn tensor_to_f32_workspace(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, ElementwiseRuntimeError> {
    require_cpu_tensor(ABS_OPERATION_ID, input)?;
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimeError::UnsupportedDType {
            operation: ABS_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("decoded f32 values"))?;
    let mut values = temporary_vec(backend, context, count, "decoded f32 values")?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let encoded: [u8; 4] = input
            .element_bytes(&indices)?
            .try_into()
            .map_err(|_| ElementwiseRuntimeError::Invalid("unaligned f32 tensor bytes"))?;
        values.try_push(f32::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _allocation: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimeError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn workspace_extend(
    values: &mut TemporaryVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimeError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimeError> {
    let mut indices = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimeError::Invalid(
                "cannot unravel an index into an empty tensor",
            ));
        }
        indices[axis] = u64::try_from(linear_index % dimension)
            .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("tensor index"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimeError> {
    if index & 1023 == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn gradient_values_len(tensor: &Tensor) -> Result<usize, ElementwiseRuntimeError> {
    usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimeError::ShapeOverflow("gradient values"))
}

fn map_operator_error(
    error: crate::generated_sim_operator_indirection_01::OperatorIndirectionError,
) -> ElementwiseRuntimeError {
    use crate::generated_sim_operator_indirection_01::OperatorIndirectionError;

    match error {
        OperatorIndirectionError::Tensor(error) => ElementwiseRuntimeError::Tensor(error),
        OperatorIndirectionError::Cancelled => ElementwiseRuntimeError::Cancelled,
        OperatorIndirectionError::ShapeOverflow(context) => {
            ElementwiseRuntimeError::ShapeOverflow(context)
        }
        OperatorIndirectionError::UnsupportedDevice { operation, device } => {
            ElementwiseRuntimeError::UnsupportedDevice { operation, device }
        }
        OperatorIndirectionError::Attention(_)
        | OperatorIndirectionError::ValueCount { .. }
        | OperatorIndirectionError::Invalid(_) => {
            ElementwiseRuntimeError::Invalid("canonical tensor conversion failed")
        }
    }
}
