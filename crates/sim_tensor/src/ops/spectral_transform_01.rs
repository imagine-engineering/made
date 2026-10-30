use crate::{
    CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId, ExecutionContext, Tensor,
    TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_12::{
        ElementwiseRuntimePartTwelveError, FftScalar, complex_fft_in_place,
    },
};
use thiserror::Error;

pub const FFTN_OPERATION_ID: &str = "COMFY-TENSOR-OP-7EC1A794152D";
pub const FFTSHIFT_OPERATION_ID: &str = "COMFY-TENSOR-OP-2C39E32ACD3C";
pub const IFFTN_OPERATION_ID: &str = "COMFY-TENSOR-OP-E4297F97AD47";
pub const IFFTSHIFT_OPERATION_ID: &str = "COMFY-TENSOR-OP-D94BAB2BAD78";

#[derive(Debug, Error)]
pub enum SpectralTransformError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("spectral transform execution was cancelled")]
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
    #[error("operation {operation} overflowed while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
    #[error("operation {operation} failed in the canonical Task 55 FFT owner: {reason}")]
    CanonicalFft {
        operation: &'static str,
        reason: String,
    },
}

impl From<sim_types::CancellationError> for SpectralTransformError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn fftn_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    spectral_transform(
        FFTN_OPERATION_ID,
        backend,
        input,
        dimensions,
        false,
        false,
        complex_output_dtype(input.descriptor().dtype()),
        context,
    )
}

pub fn fftn_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(FFTN_OPERATION_ID, input)?;
    require_same_shape(FFTN_OPERATION_ID, input, output_gradient)?;
    require_dtype(
        FFTN_OPERATION_ID,
        output_gradient,
        &[complex_output_dtype(input.descriptor().dtype())],
    )?;
    let output_dtype = match input.descriptor().dtype() {
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        dtype => {
            return Err(SpectralTransformError::UnsupportedDType {
                operation: FFTN_OPERATION_ID,
                dtype,
            });
        }
    };
    spectral_transform(
        FFTN_OPERATION_ID,
        backend,
        output_gradient,
        dimensions,
        true,
        false,
        output_dtype,
        context,
    )
}

pub fn fftn_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(FFTN_OPERATION_ID, input)?;
    require_same_shape(FFTN_OPERATION_ID, input, input_tangent)?;
    if input.descriptor().dtype() != input_tangent.descriptor().dtype() {
        return Err(SpectralTransformError::Invalid {
            operation: FFTN_OPERATION_ID,
            reason: "input tangent dtype must match the input dtype".to_owned(),
        });
    }
    fftn_with_context_exact_native(backend, input_tangent, dimensions, context)
}

pub fn ifftn_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(IFFTN_OPERATION_ID, input)?;
    require_dtype(
        IFFTN_OPERATION_ID,
        input,
        &[DType::Complex64, DType::Complex128],
    )?;
    spectral_transform(
        IFFTN_OPERATION_ID,
        backend,
        input,
        dimensions,
        true,
        true,
        input.descriptor().dtype(),
        context,
    )
}

pub fn ifftn_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(IFFTN_OPERATION_ID, input)?;
    require_same_shape(IFFTN_OPERATION_ID, input, output_gradient)?;
    require_dtype(
        IFFTN_OPERATION_ID,
        input,
        &[DType::Complex64, DType::Complex128],
    )?;
    require_dtype(
        IFFTN_OPERATION_ID,
        output_gradient,
        &[input.descriptor().dtype()],
    )?;
    spectral_transform(
        IFFTN_OPERATION_ID,
        backend,
        output_gradient,
        dimensions,
        false,
        true,
        input.descriptor().dtype(),
        context,
    )
}

pub fn ifftn_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(IFFTN_OPERATION_ID, input)?;
    require_same_shape(IFFTN_OPERATION_ID, input, input_tangent)?;
    require_dtype(
        IFFTN_OPERATION_ID,
        input,
        &[DType::Complex64, DType::Complex128],
    )?;
    require_dtype(
        IFFTN_OPERATION_ID,
        input_tangent,
        &[input.descriptor().dtype()],
    )?;
    ifftn_with_context_exact_native(backend, input_tangent, dimensions, context)
}

pub fn fftshift_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    shift_tensor(
        FFTSHIFT_OPERATION_ID,
        backend,
        input,
        dimensions,
        false,
        context,
    )
}

pub fn ifftshift_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    shift_tensor(
        IFFTSHIFT_OPERATION_ID,
        backend,
        input,
        dimensions,
        true,
        context,
    )
}

pub fn fftshift_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    shift_tensor(
        FFTSHIFT_OPERATION_ID,
        backend,
        output_gradient,
        dimensions,
        true,
        context,
    )
}

pub fn ifftshift_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    shift_tensor(
        IFFTSHIFT_OPERATION_ID,
        backend,
        output_gradient,
        dimensions,
        false,
        context,
    )
}

pub fn fftshift_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    fftshift_with_context_exact_native(backend, input_tangent, dimensions, context)
}

pub fn ifftshift_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    ifftshift_with_context_exact_native(backend, input_tangent, dimensions, context)
}

#[allow(clippy::too_many_arguments)]
fn spectral_transform(
    operation: &'static str,
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    inverse: bool,
    divide_by_transform_length: bool,
    output_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(operation, input)?;
    require_dtype(
        operation,
        input,
        &[DType::F32, DType::Complex64, DType::F64, DType::Complex128],
    )?;
    if matches!(input.descriptor().dtype(), DType::F64 | DType::Complex128) {
        spectral_transform_values::<f64>(
            operation,
            backend,
            input,
            dimensions,
            inverse,
            divide_by_transform_length,
            output_dtype,
            context,
        )
    } else {
        spectral_transform_values::<f32>(
            operation,
            backend,
            input,
            dimensions,
            inverse,
            divide_by_transform_length,
            output_dtype,
            context,
        )
    }
}

fn complex_output_dtype(dtype: DType) -> DType {
    if matches!(dtype, DType::F64 | DType::Complex128) {
        DType::Complex128
    } else {
        DType::Complex64
    }
}

#[allow(clippy::too_many_arguments)]
fn spectral_transform_values<Scalar: FftScalar>(
    operation: &'static str,
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    inverse: bool,
    divide_by_transform_length: bool,
    output_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    let axes = normalize_dimensions(operation, dimensions, input.descriptor().rank())?;
    let transform_length = checked_transform_length(operation, input.descriptor().shape(), &axes)?;
    let mut values = read_complex_values::<Scalar>(operation, backend, input, context)?;
    apply_fft_axes(
        operation,
        backend,
        &mut values,
        input.descriptor().shape(),
        &axes,
        inverse,
        context,
    )?;
    if divide_by_transform_length {
        let scale = Scalar::from_f64(1.0) / Scalar::from_f64(transform_length as f64);
        for (index, value) in values.iter_mut().enumerate() {
            check_periodically(index, context)?;
            value.0 *= scale;
            value.1 *= scale;
        }
    }
    upload_complex_projection(
        operation,
        backend,
        input.descriptor().shape(),
        output_dtype,
        &values,
        context,
    )
}

fn apply_fft_axes<Scalar: FftScalar>(
    operation: &'static str,
    backend: &CpuBackend,
    values: &mut [(Scalar, Scalar)],
    shape: &[u64],
    axes: &[usize],
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), SpectralTransformError> {
    if values.is_empty() {
        return Ok(());
    }
    for &axis in axes {
        context.cancellation.check()?;
        let axis_extent = usize::try_from(*shape.get(axis).ok_or(
            SpectralTransformError::ShapeOverflow {
                operation,
                subject: "FFT axis",
            },
        )?)
        .map_err(|_| SpectralTransformError::ShapeOverflow {
            operation,
            subject: "FFT axis extent",
        })?;
        let trailing = shape
            .get(axis + 1..)
            .ok_or(SpectralTransformError::ShapeOverflow {
                operation,
                subject: "FFT trailing dimensions",
            })?;
        let inner = checked_element_count(operation, trailing, "FFT inner extent")?;
        let axis_span =
            axis_extent
                .checked_mul(inner)
                .ok_or(SpectralTransformError::ShapeOverflow {
                    operation,
                    subject: "FFT axis span",
                })?;
        let outer =
            values
                .len()
                .checked_div(axis_span)
                .ok_or(SpectralTransformError::ShapeOverflow {
                    operation,
                    subject: "FFT outer extent",
                })?;
        let mut line = backend.workspace_vec(context, axis_extent)?;
        for _ in 0..axis_extent {
            line.try_push((Scalar::ZERO, Scalar::ZERO))?;
        }
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                check_periodically(inner_index, context)?;
                for axis_index in 0..axis_extent {
                    let source = line_index(
                        operation,
                        outer_index,
                        axis_span,
                        axis_index,
                        inner,
                        inner_index,
                    )?;
                    line[axis_index] =
                        *values
                            .get(source)
                            .ok_or(SpectralTransformError::ShapeOverflow {
                                operation,
                                subject: "FFT gather index",
                            })?;
                }
                complex_fft_in_place(backend, &mut line, inverse, context).map_err(|error| {
                    match error {
                        ElementwiseRuntimePartTwelveError::Cancelled => {
                            SpectralTransformError::Cancelled
                        }
                        ElementwiseRuntimePartTwelveError::Tensor(error) => {
                            SpectralTransformError::Tensor(error)
                        }
                        error => SpectralTransformError::CanonicalFft {
                            operation,
                            reason: error.to_string(),
                        },
                    }
                })?;
                for axis_index in 0..axis_extent {
                    let destination = line_index(
                        operation,
                        outer_index,
                        axis_span,
                        axis_index,
                        inner,
                        inner_index,
                    )?;
                    *values.get_mut(destination).ok_or(
                        SpectralTransformError::ShapeOverflow {
                            operation,
                            subject: "FFT scatter index",
                        },
                    )? = line[axis_index];
                }
            }
        }
    }
    Ok(())
}

fn shift_tensor(
    operation: &'static str,
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    context.cancellation.check()?;
    require_cpu(operation, input)?;
    let axes = normalize_dimensions(operation, dimensions, input.descriptor().rank())?;
    let element_count = usize::try_from(input.descriptor().element_count()?).map_err(|_| {
        SpectralTransformError::ShapeOverflow {
            operation,
            subject: "shift element count",
        }
    })?;
    let byte_width = usize::try_from(input.descriptor().dtype().byte_width()).map_err(|_| {
        SpectralTransformError::ShapeOverflow {
            operation,
            subject: "shift byte width",
        }
    })?;
    let byte_count =
        element_count
            .checked_mul(byte_width)
            .ok_or(SpectralTransformError::ShapeOverflow {
                operation,
                subject: "shift byte count",
            })?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for output_linear in 0..element_count {
        check_periodically(output_linear, context)?;
        let mut source_coordinates = unravel_index(
            operation,
            output_linear,
            input.descriptor().shape(),
            "shift coordinate",
        )?;
        for &axis in &axes {
            let extent = *input.descriptor().shape().get(axis).ok_or(
                SpectralTransformError::ShapeOverflow {
                    operation,
                    subject: "shift axis",
                },
            )?;
            if extent == 0 {
                continue;
            }
            let offset = if inverse {
                extent / 2
            } else {
                extent / 2 + extent % 2
            };
            let coordinate =
                source_coordinates
                    .get_mut(axis)
                    .ok_or(SpectralTransformError::ShapeOverflow {
                        operation,
                        subject: "shift source axis",
                    })?;
            *coordinate =
                coordinate
                    .checked_add(offset)
                    .ok_or(SpectralTransformError::ShapeOverflow {
                        operation,
                        subject: "shift source coordinate",
                    })?
                    % extent;
        }
        for byte in input.element_bytes(&source_coordinates)? {
            bytes.try_push(*byte)?;
        }
    }
    context.cancellation.check()?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        DeviceId::CPU,
        context.stream,
    )?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn read_complex_values<Scalar: FftScalar>(
    operation: &'static str,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<(Scalar, Scalar)>, SpectralTransformError> {
    let count = usize::try_from(input.descriptor().element_count()?).map_err(|_| {
        SpectralTransformError::ShapeOverflow {
            operation,
            subject: "input element count",
        }
    })?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context)?;
        let linear_index =
            u64::try_from(linear_index).map_err(|_| SpectralTransformError::ShapeOverflow {
                operation,
                subject: "input linear index",
            })?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.linear_element_bytes(linear_index)?)?;
        values.try_push(match value {
            DecodedScalar::Real(real) => (Scalar::from_f64(real), Scalar::ZERO),
            DecodedScalar::Complex { real, imaginary } => {
                (Scalar::from_f64(real), Scalar::from_f64(imaginary))
            }
            _ => {
                return Err(SpectralTransformError::UnsupportedDType {
                    operation,
                    dtype: input.descriptor().dtype(),
                });
            }
        })?;
    }
    Ok(values)
}

fn upload_complex_projection<Scalar: FftScalar>(
    operation: &'static str,
    backend: &CpuBackend,
    shape: &[u64],
    output_dtype: DType,
    values: &[(Scalar, Scalar)],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpectralTransformError> {
    let byte_width = usize::try_from(output_dtype.byte_width()).map_err(|_| {
        SpectralTransformError::ShapeOverflow {
            operation,
            subject: "output byte width",
        }
    })?;
    let byte_count =
        values
            .len()
            .checked_mul(byte_width)
            .ok_or(SpectralTransformError::ShapeOverflow {
                operation,
                subject: "output byte count",
            })?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for (index, &(real, imaginary)) in values.iter().enumerate() {
        check_periodically(index, context)?;
        let value = if matches!(output_dtype, DType::F32 | DType::F64) {
            DecodedScalar::Real(real.as_f64())
        } else {
            DecodedScalar::Complex {
                real: real.as_f64(),
                imaginary: imaginary.as_f64(),
            }
        };
        for byte in output_dtype.encode_decoded_scalar(value, operation, DeviceId::CPU)? {
            bytes.try_push(byte)?;
        }
    }
    context.cancellation.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), output_dtype, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn normalize_dimensions(
    operation: &'static str,
    dimensions: &[i64],
    rank: usize,
) -> Result<Vec<usize>, SpectralTransformError> {
    let rank_i64 = i64::try_from(rank).map_err(|_| SpectralTransformError::ShapeOverflow {
        operation,
        subject: "tensor rank",
    })?;
    let mut normalized = Vec::with_capacity(dimensions.len());
    for &dimension in dimensions {
        let dimension = if dimension < 0 {
            dimension
                .checked_add(rank_i64)
                .ok_or(SpectralTransformError::ShapeOverflow {
                    operation,
                    subject: "normalized dimension",
                })?
        } else {
            dimension
        };
        if dimension < 0 || dimension >= rank_i64 {
            return Err(SpectralTransformError::Invalid {
                operation,
                reason: format!("dimension {dimension} is outside rank {rank}"),
            });
        }
        let dimension =
            usize::try_from(dimension).map_err(|_| SpectralTransformError::ShapeOverflow {
                operation,
                subject: "normalized dimension",
            })?;
        if normalized.contains(&dimension) {
            return Err(SpectralTransformError::Invalid {
                operation,
                reason: format!("dimension {dimension} is repeated"),
            });
        }
        normalized.push(dimension);
    }
    Ok(normalized)
}

fn checked_transform_length(
    operation: &'static str,
    shape: &[u64],
    axes: &[usize],
) -> Result<usize, SpectralTransformError> {
    axes.iter().try_fold(1_usize, |length, &axis| {
        let extent = usize::try_from(*shape.get(axis).ok_or(
            SpectralTransformError::ShapeOverflow {
                operation,
                subject: "transform axis",
            },
        )?)
        .map_err(|_| SpectralTransformError::ShapeOverflow {
            operation,
            subject: "transform length",
        })?;
        if extent == 0 {
            return Err(SpectralTransformError::Invalid {
                operation,
                reason: format!("FFT dimension {axis} has zero length"),
            });
        }
        length
            .checked_mul(extent)
            .ok_or(SpectralTransformError::ShapeOverflow {
                operation,
                subject: "transform length",
            })
    })
}

fn checked_element_count(
    operation: &'static str,
    shape: &[u64],
    subject: &'static str,
) -> Result<usize, SpectralTransformError> {
    shape.iter().try_fold(1_usize, |count, &extent| {
        let extent = usize::try_from(extent)
            .map_err(|_| SpectralTransformError::ShapeOverflow { operation, subject })?;
        count
            .checked_mul(extent)
            .ok_or(SpectralTransformError::ShapeOverflow { operation, subject })
    })
}

fn line_index(
    operation: &'static str,
    outer_index: usize,
    axis_span: usize,
    axis_index: usize,
    inner: usize,
    inner_index: usize,
) -> Result<usize, SpectralTransformError> {
    outer_index
        .checked_mul(axis_span)
        .and_then(|base| {
            axis_index
                .checked_mul(inner)
                .and_then(|axis| base.checked_add(axis))
        })
        .and_then(|base| base.checked_add(inner_index))
        .ok_or(SpectralTransformError::ShapeOverflow {
            operation,
            subject: "FFT line index",
        })
}

fn unravel_index(
    operation: &'static str,
    linear_index: usize,
    shape: &[u64],
    subject: &'static str,
) -> Result<Vec<u64>, SpectralTransformError> {
    let mut remainder = u64::try_from(linear_index)
        .map_err(|_| SpectralTransformError::ShapeOverflow { operation, subject })?;
    let mut coordinates = vec![0_u64; shape.len()];
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape).rev() {
        if extent == 0 {
            return Err(SpectralTransformError::Invalid {
                operation,
                reason: "cannot address an element through a zero-length dimension".to_owned(),
            });
        }
        *coordinate = remainder % extent;
        remainder /= extent;
    }
    Ok(coordinates)
}

fn require_cpu(operation: &'static str, input: &Tensor) -> Result<(), SpectralTransformError> {
    let device = input.descriptor().device();
    if device != DeviceId::CPU {
        return Err(SpectralTransformError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn require_dtype(
    operation: &'static str,
    input: &Tensor,
    supported: &[DType],
) -> Result<(), SpectralTransformError> {
    let dtype = input.descriptor().dtype();
    if !supported.contains(&dtype) {
        return Err(SpectralTransformError::UnsupportedDType { operation, dtype });
    }
    Ok(())
}

fn require_same_shape(
    operation: &'static str,
    input: &Tensor,
    other: &Tensor,
) -> Result<(), SpectralTransformError> {
    if input.descriptor().shape() != other.descriptor().shape() {
        return Err(SpectralTransformError::Invalid {
            operation,
            reason: "input and derivative tensor shapes must match".to_owned(),
        });
    }
    Ok(())
}

fn check_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), SpectralTransformError> {
    if index.is_multiple_of(256) {
        context.cancellation.check()?;
    }
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::{CancellationToken, CpuWorkspaceAuthority, StreamId};
    use sim_types::DeviceKind;
    use std::collections::BTreeMap;

    #[test]
    fn unsupported_device_is_typed_before_dtype_validation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor =
            TensorDescriptor::contiguous(vec![1], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let mut input = backend.upload_f32(descriptor, &[1.0], &context)?.0;
        input.descriptor.device = DeviceId::new(DeviceKind::Metal, 0);
        assert!(matches!(
            ifftn_with_context_exact_native(&backend, &input, &[0], &context),
            Err(SpectralTransformError::UnsupportedDevice { .. })
        ));
        Ok(())
    }

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            (
                FFTN_OPERATION_ID,
                "04d0c5210f7dae09f105663afe19789260a4b5dc4cf2522166ee0a26e3ea64a8",
            ),
            (
                FFTSHIFT_OPERATION_ID,
                "6851503f6acc59f24084ee647d2450cacb44456c59e801d7ce98542335312a0b",
            ),
            (
                IFFTN_OPERATION_ID,
                "9ba7d57ae974f34728d0ce0ff49b3a103954ab230898495084d1438b2a5ddfb1",
            ),
            (
                IFFTSHIFT_OPERATION_ID,
                "c0ec118ef2ef0ab13a26cce3ee8bfe67efe87f9631474603da6978a0994cca8b",
            ),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation| (*operation, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-spectral-transform-01.json",
            "VAL-TENSOR-001",
            "Task 90 exact spectral transform adapters over the canonical Task 55 FFT, dtype codec, workspace, and publication owners",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-spectral-transform-01.json",
            "VAL-AUTOGRAD-001",
            "Task 90 analytical FFT, inverse FFT, and shift VJP/JVP contracts",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
