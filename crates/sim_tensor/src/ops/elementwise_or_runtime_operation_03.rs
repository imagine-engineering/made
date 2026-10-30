use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, GradientMode, NumericClass, Scalar, StreamId, Tensor, TensorDescriptor,
    TensorError, UnaryOperation,
    cpu_backend::{apply_unary_scalar, binary_broadcast_shape, broadcast_indices},
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionVjp, OperatorIndirectionError, TensorValues,
        convolution_with_context_exact_native, convolution_jvp_with_context_exact_native, convolution_vjp_with_context_exact_native,
    },
};
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

pub const ADD_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-1ED3CF790B68";
pub const CLAMP_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-26EF6B18C684";
pub const DATA_PTR_OPERATION_ID: &str = "COMFY-TENSOR-OP-1F39246F0FAD";
pub const CUDA_MEMORY_SUMMARY_OPERATION_ID: &str = "COMFY-TENSOR-OP-1B60D420F7C7";
pub const CUDNN_CONVOLUTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-231238FDA88D";
pub const FLOOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-1C55B11AD08B";
pub const GREATER_OPERATION_ID: &str = "COMFY-TENSOR-OP-22DF6A4C26CC";
pub const NO_GRAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-2673CE820FAC";
pub const TORCH_SAVE_OPERATION_ID: &str = "COMFY-TENSOR-OP-2464198E16CB";
pub const SIGMOID_OPERATION_ID: &str = "COMFY-TENSOR-OP-1917B7227A5C";
pub const EXPM1_OPERATION_ID: &str = "COMFY-TENSOR-OP-263D166C9D1F";
pub const XPU_DEVICE_COUNT_OPERATION_ID: &str = "COMFY-TENSOR-OP-2255F11A43BA";
pub const REAL_ADD_OPERATION_ID: &str = "ZED-TENSOR-REAL-ADD-V1";
pub const REAL_MULTIPLY_OPERATION_ID: &str = "ZED-TENSOR-REAL-MULTIPLY-V1";
pub const REAL_LERP_OPERATION_ID: &str = "ZED-TENSOR-REAL-LERP-V1";

const MAXIMUM_ARCHIVE_DEPTH: usize = 64;
const MAXIMUM_ARCHIVE_NODES: usize = 1_000_000;
const MAXIMUM_ARCHIVE_STRING_BYTES: usize = 1 << 20;
const MAXIMUM_ARCHIVE_BYTES: usize = u32::MAX as usize;

#[derive(Clone, Copy, Debug)]
pub enum ElementwiseOperand<'a> {
    Tensor(&'a Tensor),
    Scalar(Scalar),
}

#[derive(Clone, Debug)]
pub enum TorchArchiveValue {
    None,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Tensor(Tensor),
    List(Vec<TorchArchiveValue>),
    Tuple(Vec<TorchArchiveValue>),
    Map(BTreeMap<String, TorchArchiveValue>),
}

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartThreeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Convolution(#[from] OperatorIndirectionError),
    #[error("elementwise/runtime part-three operation was cancelled")]
    Cancelled,
    #[error("elementwise/runtime part-three input is invalid: {0}")]
    Invalid(&'static str),
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
    #[error("shape or allocation arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("torch archive exceeds the {kind} limit of {limit}")]
    ArchiveLimit { kind: &'static str, limit: usize },
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartThreeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn add_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    other: ElementwiseOperand<'_>,
    alpha: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    require_cpu(input, ADD_IN_PLACE_OPERATION_ID)?;
    let dtype = input.descriptor().dtype();
    let alpha = decode_scalar_for_dtype(dtype, alpha, ADD_IN_PLACE_OPERATION_ID)?;
    let operand = checked_operand(input, other, ADD_IN_PLACE_OPERATION_ID)?;
    if operand.broadcast_shape(input.descriptor().shape())? != input.descriptor().shape() {
        return Err(ElementwiseRuntimePartThreeError::Invalid(
            "in-place operand broadcasts beyond the receiver shape",
        ));
    }
    let source = input.clone();
    stage_in_place_with_context(backend, input, context, |indices| {
        let left = dtype.decode_scalar(source.element_bytes(indices)?)?;
        let right = operand.value(indices)?;
        encode_decoded(
            dtype,
            add_decoded(dtype, left, right, alpha)?,
            ADD_IN_PLACE_OPERATION_ID,
        )
    })
}

pub fn clamp_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    minimum: Option<Scalar>,
    maximum: Option<Scalar>,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    require_cpu(input, CLAMP_IN_PLACE_OPERATION_ID)?;
    if minimum.is_none() && maximum.is_none() {
        return Err(ElementwiseRuntimePartThreeError::Invalid(
            "clamp requires a minimum, a maximum, or both",
        ));
    }
    let dtype = input.descriptor().dtype();
    let minimum = minimum
        .map(|value| decode_scalar_for_dtype(dtype, value, CLAMP_IN_PLACE_OPERATION_ID))
        .transpose()?;
    let maximum = maximum
        .map(|value| decode_scalar_for_dtype(dtype, value, CLAMP_IN_PLACE_OPERATION_ID))
        .transpose()?;
    let source = input.clone();
    stage_in_place_with_context(backend, input, context, |indices| {
        let value = dtype.decode_scalar(source.element_bytes(indices)?)?;
        encode_decoded(
            dtype,
            clamp_decoded(value, minimum, maximum)?,
            CLAMP_IN_PLACE_OPERATION_ID,
        )
    })
}

pub fn data_ptr_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<usize, ElementwiseRuntimePartThreeError> {
    cancellation.check()?;
    require_cpu(input, DATA_PTR_OPERATION_ID)?;
    let bytes = input.host_storage_bytes()?;
    let byte_offset = input
        .descriptor()
        .offset_elements()
        .checked_mul(input.descriptor().dtype().byte_width())
        .ok_or(ElementwiseRuntimePartThreeError::Overflow(
            "tensor host address offset",
        ))?;
    let byte_offset = usize::try_from(byte_offset)
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("tensor host address offset"))?;
    if byte_offset > bytes.len() {
        return Err(TensorError::StorageBounds {
            required: u64::try_from(byte_offset)
                .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("host address bound"))?,
            actual: input.storage_byte_len(),
        }
        .into());
    }
    let address = bytes.as_ptr().wrapping_add(byte_offset) as usize;
    cancellation.check()?;
    Ok(address)
}

#[allow(clippy::too_many_arguments)]
pub fn cudnn_convolution_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    padding: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    benchmark: bool,
    deterministic: bool,
    allow_tf32: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    validate_cudnn_source_flags(benchmark, deterministic, allow_tf32)?;
    let geometry =
        ConvolutionGeometry::new(3, stride, padding, dilation, groups, false, vec![0; 3])?;
    Ok(convolution_with_context_exact_native(
        input,
        input_shape,
        weight,
        weight_shape,
        None,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn cudnn_convolution_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    output_gradient: &[f32],
    padding: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    let geometry =
        ConvolutionGeometry::new(3, stride, padding, dilation, groups, false, vec![0; 3])?;
    Ok(convolution_vjp_with_context_exact_native(
        input,
        input_shape,
        weight,
        weight_shape,
        None,
        output_gradient,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn cudnn_convolution_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    padding: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    let geometry =
        ConvolutionGeometry::new(3, stride, padding, dilation, groups, false, vec![0; 3])?;
    Ok(convolution_jvp_with_context_exact_native(
        input,
        input_tangent,
        input_shape,
        weight,
        weight_tangent,
        weight_shape,
        None,
        None,
        &geometry,
        device,
        context,
    )?)
}

pub fn floor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    map_real_preserving_dtype(
        backend,
        input,
        FLOOR_OPERATION_ID,
        true,
        f64::floor,
        context,
    )
}

pub fn floor_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    zeros_like_f32(backend, input, FLOOR_OPERATION_ID, context)
}

pub fn floor_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    floor_vjp_with_context_exact_native(backend, input, context)
}

pub fn greater_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    require_cpu(input, GREATER_OPERATION_ID)?;
    let dtype = input.descriptor().dtype();
    let operand = checked_operand(input, other, GREATER_OPERATION_ID)?;
    let output_shape = operand.broadcast_shape(input.descriptor().shape())?;
    let element_count = checked_element_count(&output_shape, "greater output")?;
    let mut bytes = temporary_vec(backend, context, element_count, "greater output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let left = dtype.decode_scalar(input.element_bytes(&input_indices)?)?;
        let right = operand.value(&output_indices)?;
        bytes.try_push(u8::from(decoded_greater(left, right)?))?;
    }
    upload_bytes(
        backend,
        &output_shape,
        DType::Bool,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn no_grad_exact_native(
    cancellation: &CancellationToken,
) -> Result<GradientMode, ElementwiseRuntimePartThreeError> {
    cancellation.check()?;
    Ok(GradientMode::NoGrad)
}

pub fn torch_save_exact_native(
    value: &TorchArchiveValue,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError> {
    cancellation.check()?;
    let mut encoder = PickleEncoder::new(cancellation);
    encoder.append(&[0x80, 0x02])?;
    encoder.encode_value(value, 0)?;
    encoder.append(b".")?;
    let (pickle, storages) = encoder.finish();
    let mut entries = reserved_vec(storages.len() + 4, "torch archive entries")?;
    entries.push(ZipEntry::new("archive/data.pkl", pickle));
    entries.push(ZipEntry::new("archive/byteorder", b"little".to_vec()));
    entries.push(ZipEntry::new("archive/version", b"3\n".to_vec()));
    for storage in storages {
        entries.push(ZipEntry::new(
            format!("archive/data/{}", storage.key),
            storage.bytes,
        ));
    }
    entries.push(ZipEntry::new(
        "archive/.data/serialization_id",
        b"zed-native-sim-tensor-v1".to_vec(),
    ));
    cancellation.check()?;
    write_stored_zip(&entries, cancellation)
}

pub fn sigmoid_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    map_real_preserving_dtype(
        backend,
        input,
        SIGMOID_OPERATION_ID,
        false,
        |value| f64::from(apply_unary_scalar(UnaryOperation::Sigmoid, value as f32)),
        context,
    )
}

pub fn real_add_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    real_binary_preserving_dtype(
        backend,
        input,
        ElementwiseOperand::Tensor(other),
        REAL_ADD_OPERATION_ID,
        |left, right| left + right,
        context,
    )
}

pub fn real_multiply_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    real_binary_preserving_dtype(
        backend,
        input,
        other,
        REAL_MULTIPLY_OPERATION_ID,
        |left, right| left * right,
        context,
    )
}

pub fn real_lerp_tensor_weight_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    weight: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    context.check()?;
    require_supported_real(input, REAL_LERP_OPERATION_ID)?;
    let end = checked_operand(
        input,
        ElementwiseOperand::Tensor(end),
        REAL_LERP_OPERATION_ID,
    )?;
    let weight = checked_operand(
        input,
        ElementwiseOperand::Tensor(weight),
        REAL_LERP_OPERATION_ID,
    )?;
    let output_shape = end.broadcast_shape(input.descriptor().shape())?;
    let output_shape = weight.broadcast_shape(&output_shape)?;
    let dtype = input.descriptor().dtype();
    let element_count = checked_element_count(&output_shape, "real lerp output")?;
    let byte_len = encoded_byte_len(dtype, element_count, "real lerp output")?;
    let mut bytes = temporary_vec(backend, context, byte_len, "real lerp output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let start = read_real_f32(input, &input_indices, REAL_LERP_OPERATION_ID)?;
        let end = decoded_real_f32(end.value(&output_indices)?, REAL_LERP_OPERATION_ID, dtype)?;
        let weight =
            decoded_real_f32(weight.value(&output_indices)?, REAL_LERP_OPERATION_ID, dtype)?;
        temporary_extend(
            &mut bytes,
            &encode_decoded(
                dtype,
                DecodedScalar::Real(f64::from(start + weight * (end - start))),
                REAL_LERP_OPERATION_ID,
            )?,
        )?;
    }
    upload_bytes(
        backend,
        &output_shape,
        dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn sigmoid_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    f32_binary_map(
        backend,
        input,
        output_gradient,
        SIGMOID_OPERATION_ID,
        |value, gradient| {
            let sigmoid = apply_unary_scalar(UnaryOperation::Sigmoid, value);
            gradient * sigmoid * (1.0 - sigmoid)
        },
        context,
    )
}

pub fn sigmoid_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    sigmoid_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn expm1_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    map_real_preserving_dtype(
        backend,
        input,
        EXPM1_OPERATION_ID,
        false,
        f64::exp_m1,
        context,
    )
}

pub fn expm1_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    f32_binary_map(
        backend,
        input,
        output_gradient,
        EXPM1_OPERATION_ID,
        |value, gradient| gradient * value.exp(),
        context,
    )
}

pub fn expm1_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    expm1_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

enum CheckedOperand {
    Tensor(Tensor),
    Scalar(DecodedScalar),
}

impl CheckedOperand {
    fn broadcast_shape(
        &self,
        input_shape: &[u64],
    ) -> Result<Vec<u64>, ElementwiseRuntimePartThreeError> {
        match self {
            Self::Tensor(tensor) => Ok(binary_broadcast_shape(
                input_shape,
                tensor.descriptor().shape(),
            )?),
            Self::Scalar(_) => Ok(input_shape.to_vec()),
        }
    }

    fn value(
        &self,
        output_indices: &[u64],
    ) -> Result<DecodedScalar, ElementwiseRuntimePartThreeError> {
        match self {
            Self::Tensor(tensor) => {
                let indices = broadcast_indices(output_indices, tensor.descriptor().shape())?;
                Ok(tensor
                    .descriptor()
                    .dtype()
                    .decode_scalar(tensor.element_bytes(&indices)?)?)
            }
            Self::Scalar(value) => Ok(*value),
        }
    }
}

fn checked_operand(
    input: &Tensor,
    operand: ElementwiseOperand<'_>,
    operation: &'static str,
) -> Result<CheckedOperand, ElementwiseRuntimePartThreeError> {
    match operand {
        ElementwiseOperand::Tensor(tensor) => {
            require_cpu(tensor, operation)?;
            if tensor.descriptor().dtype() != input.descriptor().dtype() {
                return Err(TensorError::DTypeMismatch {
                    expected: input.descriptor().dtype(),
                    actual: tensor.descriptor().dtype(),
                }
                .into());
            }
            if tensor.descriptor().stream() != input.descriptor().stream() {
                return Err(TensorError::StreamMismatch {
                    expected: input.descriptor().stream(),
                    actual: tensor.descriptor().stream(),
                }
                .into());
            }
            Ok(CheckedOperand::Tensor(tensor.clone()))
        }
        ElementwiseOperand::Scalar(scalar) => Ok(CheckedOperand::Scalar(decode_scalar_for_dtype(
            input.descriptor().dtype(),
            scalar,
            operation,
        )?)),
    }
}

fn stage_in_place_with_context(
    backend: &CpuBackend,
    input: &mut Tensor,
    context: &ExecutionContext<'_>,
    mut value: impl FnMut(&[u64]) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError>,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    let shape = input.descriptor().shape().to_vec();
    let element_count = checked_element_count(&shape, "staged in-place update")?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("staged element width"))?;
    let byte_count =
        element_count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartThreeError::Overflow(
                "staged in-place bytes",
            ))?;
    let mut staged = temporary_vec(backend, context, byte_count, "staged in-place bytes")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, &shape)?;
        let bytes = value(&indices)?;
        if bytes.len() != width {
            return Err(ElementwiseRuntimePartThreeError::Invalid(
                "staged element width differs from receiver dtype",
            ));
        }
        temporary_extend(&mut staged, &bytes)?;
    }
    context.check()?;
    let mut candidate = input.clone();
    {
        let mut write = candidate.write()?;
        for (linear_index, bytes) in staged.chunks_exact(width).enumerate() {
            let indices = unravel_index(linear_index, &shape)?;
            write.element_bytes_mut(&indices)?.copy_from_slice(bytes);
        }
    }
    context.check()?;
    input.commit_in_place(candidate)?;
    Ok(())
}

fn add_decoded(
    dtype: DType,
    left: DecodedScalar,
    right: DecodedScalar,
    alpha: DecodedScalar,
) -> Result<DecodedScalar, ElementwiseRuntimePartThreeError> {
    match (left, right, alpha) {
        (DecodedScalar::Real(left), DecodedScalar::Real(right), DecodedScalar::Real(alpha)) => {
            Ok(DecodedScalar::Real(alpha.mul_add(right, left)))
        }
        (
            DecodedScalar::Signed(left),
            DecodedScalar::Signed(right),
            DecodedScalar::Signed(alpha),
        ) => {
            let value = i128::from(alpha)
                .checked_mul(i128::from(right))
                .and_then(|value| value.checked_add(i128::from(left)))
                .ok_or(ElementwiseRuntimePartThreeError::Overflow(
                    "signed in-place add",
                ))?;
            Ok(DecodedScalar::Signed(wrap_signed(dtype, value)?))
        }
        (
            DecodedScalar::Unsigned(left),
            DecodedScalar::Unsigned(right),
            DecodedScalar::Unsigned(alpha),
        ) => {
            let value = u128::from(alpha)
                .checked_mul(u128::from(right))
                .and_then(|value| value.checked_add(u128::from(left)))
                .ok_or(ElementwiseRuntimePartThreeError::Overflow(
                    "unsigned in-place add",
                ))?;
            Ok(DecodedScalar::Unsigned(wrap_unsigned(dtype, value)?))
        }
        (
            DecodedScalar::Complex {
                real: left_real,
                imaginary: left_imaginary,
            },
            DecodedScalar::Complex {
                real: right_real,
                imaginary: right_imaginary,
            },
            DecodedScalar::Complex {
                real: alpha_real,
                imaginary: 0.0,
            },
        ) => Ok(DecodedScalar::Complex {
            real: alpha_real.mul_add(right_real, left_real),
            imaginary: alpha_real.mul_add(right_imaginary, left_imaginary),
        }),
        _ => Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
            operation: ADD_IN_PLACE_OPERATION_ID,
            dtype,
        }),
    }
}

fn wrap_signed(dtype: DType, value: i128) -> Result<i64, ElementwiseRuntimePartThreeError> {
    Ok(match dtype {
        DType::I8 => i64::from(value as i8),
        DType::I16 => i64::from(value as i16),
        DType::I32 => i64::from(value as i32),
        DType::I64 => value as i64,
        _ => {
            return Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
                operation: ADD_IN_PLACE_OPERATION_ID,
                dtype,
            });
        }
    })
}

fn wrap_unsigned(dtype: DType, value: u128) -> Result<u64, ElementwiseRuntimePartThreeError> {
    Ok(match dtype {
        DType::U8 => u64::from(value as u8),
        DType::U16 => u64::from(value as u16),
        DType::U32 => u64::from(value as u32),
        DType::U64 => value as u64,
        _ => {
            return Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
                operation: ADD_IN_PLACE_OPERATION_ID,
                dtype,
            });
        }
    })
}

fn clamp_decoded(
    value: DecodedScalar,
    minimum: Option<DecodedScalar>,
    maximum: Option<DecodedScalar>,
) -> Result<DecodedScalar, ElementwiseRuntimePartThreeError> {
    if matches!(
        value,
        DecodedScalar::Complex { .. } | DecodedScalar::Boolean(_)
    ) {
        return Err(ElementwiseRuntimePartThreeError::Invalid(
            "clamp accepts real or integer tensors",
        ));
    }
    let value = match minimum {
        Some(minimum) if decoded_greater(minimum, value)? => minimum,
        _ => value,
    };
    Ok(match maximum {
        Some(maximum) if decoded_greater(value, maximum)? => maximum,
        _ => value,
    })
}

fn decoded_greater(
    left: DecodedScalar,
    right: DecodedScalar,
) -> Result<bool, ElementwiseRuntimePartThreeError> {
    Ok(match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left & !right,
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left > right,
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left > right,
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => left > right,
        _ => {
            return Err(ElementwiseRuntimePartThreeError::Invalid(
                "comparison operands must have the same ordered numeric class",
            ));
        }
    })
}

fn decode_scalar_for_dtype(
    dtype: DType,
    value: Scalar,
    operation: &'static str,
) -> Result<DecodedScalar, ElementwiseRuntimePartThreeError> {
    let bytes = dtype.encode_scalar(value, operation, DeviceId::CPU)?;
    Ok(dtype.decode_scalar(&bytes)?)
}

fn encode_decoded(
    dtype: DType,
    value: DecodedScalar,
    operation: &'static str,
) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError> {
    Ok(dtype.encode_decoded_scalar(value, operation, DeviceId::CPU)?)
}

fn map_real_preserving_dtype(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    copy_integers: bool,
    function: impl Fn(f64) -> f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    require_cpu(input, operation)?;
    let dtype = input.descriptor().dtype();
    if matches!(dtype.class(), NumericClass::Complex | NumericClass::Boolean) {
        return Err(ElementwiseRuntimePartThreeError::UnsupportedDType { operation, dtype });
    }
    let element_count = checked_element_count(input.descriptor().shape(), "unary output")?;
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("unary dtype width"))?;
    let byte_len = element_count
        .checked_mul(width)
        .ok_or(ElementwiseRuntimePartThreeError::Overflow("unary output"))?;
    let mut bytes = temporary_vec(backend, context, byte_len, "unary output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let source = input.element_bytes(&indices)?;
        match dtype.decode_scalar(source)? {
            DecodedScalar::Real(value) => temporary_extend(
                &mut bytes,
                &encode_decoded(dtype, DecodedScalar::Real(function(value)), operation)?,
            )?,
            DecodedScalar::Signed(_) | DecodedScalar::Unsigned(_) if copy_integers => {
                temporary_extend(&mut bytes, source)?
            }
            _ => {
                return Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
                    operation,
                    dtype,
                });
            }
        }
    }
    upload_bytes(
        backend,
        input.descriptor().shape(),
        dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

fn real_binary_preserving_dtype(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    operation: &'static str,
    function: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    context.cancellation.check()?;
    context.check()?;
    require_supported_real(input, operation)?;
    let other = checked_operand(input, other, operation)?;
    let output_shape = other.broadcast_shape(input.descriptor().shape())?;
    let dtype = input.descriptor().dtype();
    let element_count = checked_element_count(&output_shape, "real binary output")?;
    let byte_len = encoded_byte_len(dtype, element_count, "real binary output")?;
    let mut bytes = temporary_vec(backend, context, byte_len, "real binary output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let left = read_real_f32(input, &input_indices, operation)?;
        let right = decoded_real_f32(other.value(&output_indices)?, operation, dtype)?;
        temporary_extend(
            &mut bytes,
            &encode_decoded(
                dtype,
                DecodedScalar::Real(f64::from(function(left, right))),
                operation,
            )?,
        )?;
    }
    upload_bytes(
        backend,
        &output_shape,
        dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

fn require_supported_real(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    require_cpu(input, operation)?;
    if matches!(input.descriptor().dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn read_real_f32(
    input: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartThreeError> {
    decoded_real_f32(
        input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(indices)?)?,
        operation,
        input.descriptor().dtype(),
    )
}

fn decoded_real_f32(
    value: DecodedScalar,
    operation: &'static str,
    dtype: DType,
) -> Result<f32, ElementwiseRuntimePartThreeError> {
    match value {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartThreeError::UnsupportedDType { operation, dtype }),
    }
}

fn encoded_byte_len(
    dtype: DType,
    element_count: usize,
    context: &'static str,
) -> Result<usize, ElementwiseRuntimePartThreeError> {
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow(context))?;
    element_count
        .checked_mul(width)
        .ok_or(ElementwiseRuntimePartThreeError::Overflow(context))
}

fn f32_binary_map(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
    function: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    require_f32(left, operation)?;
    require_f32(right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartThreeError::Invalid(
            "gradient tensors must share shape and stream",
        ));
    }
    let left_values = tensor_f32_workspace(backend, left, context)?;
    let right_values = tensor_f32_workspace(backend, right, context)?;
    let mut output = temporary_vec(backend, context, left_values.len(), "f32 gradient output")?;
    for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
        check_periodically(index, context.cancellation)?;
        output.try_push(function(*left, *right))?;
    }
    upload_f32(backend, left.descriptor().shape(), &output, context)
}

fn zeros_like_f32(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    require_f32(input, operation)?;
    let count = checked_element_count(input.descriptor().shape(), "zero gradient")?;
    let mut values = temporary_vec(backend, context, count, "zero gradient")?;
    for _ in 0..count {
        values.try_push(0.0)?;
    }
    upload_f32(backend, input.descriptor().shape(), &values, context)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    let (output, _) = backend.upload_bytes(descriptor, bytes, context)?;
    context.check()?;
    Ok(output)
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartThreeError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn tensor_f32_workspace(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, ElementwiseRuntimePartThreeError> {
    require_f32(input, "SIM-TENSOR-CONVERSION-F32")?;
    let count = checked_element_count(input.descriptor().shape(), "decoded f32 values")?;
    let mut values = temporary_vec(backend, context, count, "decoded f32 values")?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let encoded: [u8; 4] = input
            .element_bytes(&indices)?
            .try_into()
            .map_err(|_| ElementwiseRuntimePartThreeError::Invalid("unaligned f32 tensor bytes"))?;
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
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartThreeError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_extend(
    values: &mut TemporaryVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimePartThreeError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}

fn validate_cudnn_source_flags(
    benchmark: bool,
    deterministic: bool,
    allow_tf32: bool,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    if benchmark || deterministic || !allow_tf32 {
        return Err(ElementwiseRuntimePartThreeError::Invalid(
            "the observed Comfy cudnn_convolution path requires benchmark=false, deterministic=false, and allow_tf32=true",
        ));
    }
    Ok(())
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartThreeError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn checked_element_count(
    shape: &[u64],
    context: &'static str,
) -> Result<usize, ElementwiseRuntimePartThreeError> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
    usize::try_from(count.ok_or(ElementwiseRuntimePartThreeError::Overflow(context))?)
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow(context))
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartThreeError> {
    let mut indices = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartThreeError::Invalid(
                "cannot index an empty tensor",
            ));
        }
        indices[axis] = u64::try_from(linear_index % dimension)
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("tensor index"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartThreeError> {
    if index & 1023 == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn reserved_vec<T>(
    capacity: usize,
    context: &'static str,
) -> Result<Vec<T>, ElementwiseRuntimePartThreeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow(context))?;
    Ok(values)
}

fn reserved_bytes(
    capacity: usize,
    context: &'static str,
) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError> {
    reserved_vec(capacity, context)
}

struct ArchiveStorage {
    key: usize,
    bytes: Vec<u8>,
}

struct PickleEncoder<'a> {
    bytes: Vec<u8>,
    storages: Vec<ArchiveStorage>,
    storage_keys: HashMap<(u64, DType), usize>,
    nodes: usize,
    cancellation: &'a CancellationToken,
}

impl<'a> PickleEncoder<'a> {
    fn new(cancellation: &'a CancellationToken) -> Self {
        Self {
            bytes: Vec::new(),
            storages: Vec::new(),
            storage_keys: HashMap::new(),
            nodes: 0,
            cancellation,
        }
    }

    fn finish(self) -> (Vec<u8>, Vec<ArchiveStorage>) {
        (self.bytes, self.storages)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<(), ElementwiseRuntimePartThreeError> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or(ElementwiseRuntimePartThreeError::Overflow("pickle bytes"))?;
        if next > MAXIMUM_ARCHIVE_BYTES {
            return Err(ElementwiseRuntimePartThreeError::ArchiveLimit {
                kind: "archive bytes",
                limit: MAXIMUM_ARCHIVE_BYTES,
            });
        }
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("pickle bytes"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn encode_value(
        &mut self,
        value: &TorchArchiveValue,
        depth: usize,
    ) -> Result<(), ElementwiseRuntimePartThreeError> {
        self.cancellation.check()?;
        if depth > MAXIMUM_ARCHIVE_DEPTH {
            return Err(ElementwiseRuntimePartThreeError::ArchiveLimit {
                kind: "value depth",
                limit: MAXIMUM_ARCHIVE_DEPTH,
            });
        }
        self.nodes =
            self.nodes
                .checked_add(1)
                .ok_or(ElementwiseRuntimePartThreeError::Overflow(
                    "archive node count",
                ))?;
        if self.nodes > MAXIMUM_ARCHIVE_NODES {
            return Err(ElementwiseRuntimePartThreeError::ArchiveLimit {
                kind: "value nodes",
                limit: MAXIMUM_ARCHIVE_NODES,
            });
        }
        match value {
            TorchArchiveValue::None => self.append(b"N"),
            TorchArchiveValue::Boolean(true) => self.append(&[0x88]),
            TorchArchiveValue::Boolean(false) => self.append(&[0x89]),
            TorchArchiveValue::Integer(value) => self.encode_integer(*value),
            TorchArchiveValue::Float(value) => {
                self.append(b"G")?;
                self.append(&value.to_be_bytes())
            }
            TorchArchiveValue::String(value) => self.encode_string(value),
            TorchArchiveValue::Tensor(tensor) => self.encode_tensor(tensor),
            TorchArchiveValue::List(values) => {
                self.append(b"]")?;
                if !values.is_empty() {
                    self.append(b"(")?;
                    for value in values {
                        self.encode_value(value, depth + 1)?;
                    }
                    self.append(b"e")?;
                }
                Ok(())
            }
            TorchArchiveValue::Tuple(values) => self.encode_tuple_values(values, depth + 1),
            TorchArchiveValue::Map(values) => {
                self.append(b"}")?;
                if !values.is_empty() {
                    self.append(b"(")?;
                    for (key, value) in values {
                        self.encode_string(key)?;
                        self.encode_value(value, depth + 1)?;
                    }
                    self.append(b"u")?;
                }
                Ok(())
            }
        }
    }

    fn encode_string(&mut self, value: &str) -> Result<(), ElementwiseRuntimePartThreeError> {
        if value.len() > MAXIMUM_ARCHIVE_STRING_BYTES {
            return Err(ElementwiseRuntimePartThreeError::ArchiveLimit {
                kind: "string bytes",
                limit: MAXIMUM_ARCHIVE_STRING_BYTES,
            });
        }
        let length = u32::try_from(value.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("pickle string length"))?;
        self.append(b"X")?;
        self.append(&length.to_le_bytes())?;
        self.append(value.as_bytes())
    }

    fn encode_integer(&mut self, value: i64) -> Result<(), ElementwiseRuntimePartThreeError> {
        if let Ok(value) = i32::try_from(value) {
            self.append(b"J")?;
            return self.append(&value.to_le_bytes());
        }
        let mut bytes = value.to_le_bytes().to_vec();
        while bytes.len() > 1 {
            let last = bytes[bytes.len() - 1];
            let previous = bytes[bytes.len() - 2];
            if (last == 0 && previous & 0x80 == 0) || (last == 0xff && previous & 0x80 != 0) {
                bytes.pop();
            } else {
                break;
            }
        }
        let length = u8::try_from(bytes.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("pickle integer length"))?;
        self.append(&[0x8a, length])?;
        self.append(&bytes)
    }

    fn encode_tuple_values(
        &mut self,
        values: &[TorchArchiveValue],
        depth: usize,
    ) -> Result<(), ElementwiseRuntimePartThreeError> {
        if values.is_empty() {
            return self.append(b")");
        }
        self.append(b"(")?;
        for value in values {
            self.encode_value(value, depth)?;
        }
        self.append(b"t")
    }

    fn encode_u64_tuple(&mut self, values: &[u64]) -> Result<(), ElementwiseRuntimePartThreeError> {
        if values.is_empty() {
            return self.append(b")");
        }
        self.append(b"(")?;
        for value in values {
            let value = i64::try_from(*value)
                .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("tensor tuple value"))?;
            self.encode_integer(value)?;
        }
        self.append(b"t")
    }

    fn encode_i64_tuple(&mut self, values: &[i64]) -> Result<(), ElementwiseRuntimePartThreeError> {
        if values.is_empty() {
            return self.append(b")");
        }
        self.append(b"(")?;
        for value in values {
            self.encode_integer(*value)?;
        }
        self.append(b"t")
    }

    fn encode_global(
        &mut self,
        module: &str,
        name: &str,
    ) -> Result<(), ElementwiseRuntimePartThreeError> {
        self.append(b"c")?;
        self.append(module.as_bytes())?;
        self.append(b"\n")?;
        self.append(name.as_bytes())?;
        self.append(b"\n")
    }

    fn encode_tensor(&mut self, tensor: &Tensor) -> Result<(), ElementwiseRuntimePartThreeError> {
        require_cpu(tensor, TORCH_SAVE_OPERATION_ID)?;
        let dtype = tensor.descriptor().dtype();
        let storage_type = torch_storage_type(dtype)?;
        let storage_identity = (tensor.storage_id().get(), dtype);
        let storage_key = if let Some(key) = self.storage_keys.get(&storage_identity) {
            *key
        } else {
            let bytes = copy_bytes(tensor.host_storage_bytes()?, "torch tensor storage")?;
            if bytes.len()
                % usize::try_from(dtype.byte_width()).map_err(|_| {
                    ElementwiseRuntimePartThreeError::Overflow("torch storage dtype width")
                })?
                != 0
            {
                return Err(ElementwiseRuntimePartThreeError::Invalid(
                    "tensor storage byte length is not divisible by dtype width",
                ));
            }
            let key = self.storages.len();
            self.storages.push(ArchiveStorage { key, bytes });
            self.storage_keys.insert(storage_identity, key);
            key
        };
        let storage_elements = tensor.storage_byte_len() / dtype.byte_width();
        self.encode_global("torch._utils", "_rebuild_tensor_v2")?;
        self.append(b"(")?;
        self.append(b"(")?;
        self.encode_string("storage")?;
        self.encode_global("torch", storage_type)?;
        self.encode_string(&storage_key.to_string())?;
        self.encode_string("cpu")?;
        self.encode_integer(
            i64::try_from(storage_elements).map_err(|_| {
                ElementwiseRuntimePartThreeError::Overflow("torch storage elements")
            })?,
        )?;
        self.append(b"tQ")?;
        self.encode_integer(
            i64::try_from(tensor.descriptor().offset_elements()).map_err(|_| {
                ElementwiseRuntimePartThreeError::Overflow("torch tensor storage offset")
            })?,
        )?;
        self.encode_u64_tuple(tensor.descriptor().shape())?;
        self.encode_i64_tuple(tensor.descriptor().strides())?;
        self.append(&[0x89])?;
        self.encode_global("collections", "OrderedDict")?;
        self.append(b")RtR")
    }
}

fn torch_storage_type(dtype: DType) -> Result<&'static str, ElementwiseRuntimePartThreeError> {
    Ok(match dtype {
        DType::Bool => "BoolStorage",
        DType::U8 => "ByteStorage",
        DType::I8 => "CharStorage",
        DType::I16 => "ShortStorage",
        DType::I32 => "IntStorage",
        DType::I64 => "LongStorage",
        DType::F16 => "HalfStorage",
        DType::Bf16 => "BFloat16Storage",
        DType::F32 => "FloatStorage",
        DType::F64 => "DoubleStorage",
        DType::Complex64 => "ComplexFloatStorage",
        DType::Complex128 => "ComplexDoubleStorage",
        _ => {
            return Err(ElementwiseRuntimePartThreeError::UnsupportedDType {
                operation: TORCH_SAVE_OPERATION_ID,
                dtype,
            });
        }
    })
}

fn copy_bytes(
    bytes: &[u8],
    context: &'static str,
) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError> {
    let mut copied = reserved_bytes(bytes.len(), context)?;
    copied.extend_from_slice(bytes);
    Ok(copied)
}

struct ZipEntry {
    name: String,
    data: Vec<u8>,
}

impl ZipEntry {
    fn new(name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }
}

struct CentralEntry {
    name: String,
    crc32: u32,
    size: u32,
    offset: u32,
}

fn write_stored_zip(
    entries: &[ZipEntry],
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, ElementwiseRuntimePartThreeError> {
    let entry_count = u16::try_from(entries.len())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip entry count"))?;
    let total = entries.iter().try_fold(22_usize, |total, entry| {
        total
            .checked_add(76)
            .and_then(|value| value.checked_add(entry.name.len().checked_mul(2)?))
            .and_then(|value| value.checked_add(entry.data.len()))
    });
    let total = total.ok_or(ElementwiseRuntimePartThreeError::Overflow(
        "zip archive size",
    ))?;
    if total > MAXIMUM_ARCHIVE_BYTES {
        return Err(ElementwiseRuntimePartThreeError::ArchiveLimit {
            kind: "archive bytes",
            limit: MAXIMUM_ARCHIVE_BYTES,
        });
    }
    let mut archive = reserved_bytes(total, "zip archive")?;
    let mut central = reserved_vec(entries.len(), "zip central directory")?;
    for (index, entry) in entries.iter().enumerate() {
        check_periodically(index, cancellation)?;
        let name = entry.name.as_bytes();
        let name_length = u16::try_from(name.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip entry name"))?;
        let size = u32::try_from(entry.data.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip entry size"))?;
        let offset = u32::try_from(archive.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip entry offset"))?;
        let crc32 = crc32(&entry.data);
        push_u32(&mut archive, 0x0403_4b50);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, crc32);
        push_u32(&mut archive, size);
        push_u32(&mut archive, size);
        push_u16(&mut archive, name_length);
        push_u16(&mut archive, 0);
        archive.extend_from_slice(name);
        archive.extend_from_slice(&entry.data);
        central.push(CentralEntry {
            name: entry.name.clone(),
            crc32,
            size,
            offset,
        });
    }
    cancellation.check()?;
    let central_offset = u32::try_from(archive.len())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip central offset"))?;
    for entry in &central {
        let name = entry.name.as_bytes();
        let name_length = u16::try_from(name.len())
            .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip central name"))?;
        push_u32(&mut archive, 0x0201_4b50);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, entry.crc32);
        push_u32(&mut archive, entry.size);
        push_u32(&mut archive, entry.size);
        push_u16(&mut archive, name_length);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, 0);
        push_u32(&mut archive, entry.offset);
        archive.extend_from_slice(name);
    }
    let central_size = u32::try_from(archive.len())
        .map_err(|_| ElementwiseRuntimePartThreeError::Overflow("zip central size"))?
        .checked_sub(central_offset)
        .ok_or(ElementwiseRuntimePartThreeError::Overflow(
            "zip central size",
        ))?;
    push_u32(&mut archive, 0x0605_4b50);
    push_u16(&mut archive, 0);
    push_u16(&mut archive, 0);
    push_u16(&mut archive, entry_count);
    push_u16(&mut archive, entry_count);
    push_u32(&mut archive, central_size);
    push_u32(&mut archive, central_offset);
    push_u16(&mut archive, 0);
    cancellation.check()?;
    Ok(archive)
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}
