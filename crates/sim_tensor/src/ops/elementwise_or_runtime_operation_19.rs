use std::sync::Arc;

use crate::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar,
    DeviceId, ExecutionContext, NativeStream, NativeStreamRegistry, Tensor, TensorDescriptor,
    TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError,
        div_with_context_exact_native as canonical_div_with_context,
    },
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError,
        cos_jvp_with_context_exact_native as canonical_cos_jvp_with_context,
        cos_vjp_with_context_exact_native as canonical_cos_vjp_with_context,
        cos_with_context_exact_native as canonical_cos_with_context,
    },
    generated_elementwise_or_runtime_operation_14::{
        ElementwiseRuntimePartFourteenError,
        argsort_with_context_exact_native as canonical_argsort_with_context,
    },
    generated_elementwise_or_runtime_operation_15::{
        ElementwiseRuntimePartFifteenError,
        flip_dimensions_with_context_exact_native as canonical_flip_with_context,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const SIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-CC875F3A9DF9";
pub const COS_OPERATION_ID: &str = "COMFY-TENSOR-OP-CD54624C2360";
pub const DIV_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-C9CC06A648EC";
pub const FLIP_OPERATION_ID: &str = "COMFY-TENSOR-OP-C9C8310F80B5";
pub const CUDA_CURRENT_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-C93863D94FF9";
pub const CUDA_EMPTY_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-CA8F43C066B1";
pub const CUDA_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-C9765FFEEB7F";
pub const EQUAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-C905902EB028";
pub const MLU_DEVICE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-CC4DC3D17ADD";
pub const SORT_OPERATION_ID: &str = "COMFY-TENSOR-OP-C8BA6CE3159C";
pub const DIRECTML_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-CE66E20937C0";
pub const EXTERNAL_MATH_ATAN2_OPERATION_ID: &str = "COMFY-TENSOR-OP-CA83DE14D96E";

#[derive(Debug)]
pub struct NativeSortResult {
    pub values: Tensor,
    pub indices: Tensor,
}

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartNineteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartFive(#[from] ElementwiseRuntimePartFiveError),
    #[error(transparent)]
    PartEight(#[from] ElementwiseRuntimePartEightError),
    #[error(transparent)]
    PartFourteen(#[from] ElementwiseRuntimePartFourteenError),
    #[error(transparent)]
    PartFifteen(#[from] ElementwiseRuntimePartFifteenError),
    #[error("elementwise/runtime part-nineteen execution was cancelled")]
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
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartNineteenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _label: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartNineteenError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

pub fn tensor_size_exact_native(
    dimensions: &[u64],
    cancellation: &CancellationToken,
) -> Result<Arc<[u64]>, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    let size = Arc::<[u64]>::from(dimensions);
    cancellation.check()?;
    Ok(size)
}

pub fn cos_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    Ok(canonical_cos_with_context(backend, input, context)?)
}

pub fn cos_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    Ok(canonical_cos_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn cos_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    Ok(canonical_cos_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn div_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    let staged = canonical_div_with_context(backend, input, other, context)?;
    if staged.descriptor().shape() != input.descriptor().shape() {
        return invalid(
            DIV_IN_PLACE_OPERATION_ID,
            "div_ broadcast output must match the receiver shape",
        );
    }
    context.cancellation.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

pub fn flip_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    Ok(canonical_flip_with_context(
        backend,
        input,
        dimensions,
        FLIP_OPERATION_ID,
        context,
    )?)
}

pub fn flip_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    flip_with_context_exact_native(backend, output_gradient, dimensions, context)
}

pub fn flip_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    flip_with_context_exact_native(backend, input_tangent, dimensions, context)
}

pub fn cuda_current_device_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<u32, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    let device = capabilities.device();
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation: CUDA_CURRENT_DEVICE_OPERATION_ID,
            device,
        });
    }
    cancellation.check()?;
    Ok(device.ordinal())
}

pub fn cuda_stream_exact_native(
    registry: &NativeStreamRegistry,
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    priority: i32,
    cancellation: &CancellationToken,
) -> Result<NativeStream, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation: CUDA_STREAM_OPERATION_ID,
            device,
        });
    }
    Ok(registry.create(capabilities, device, priority, cancellation)?)
}

pub fn equal_exact_native(
    input: &Tensor,
    other: &Tensor,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    if input.descriptor().device() != other.descriptor().device() {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation: EQUAL_OPERATION_ID,
            device: other.descriptor().device(),
        });
    }
    if input.descriptor().shape() != other.descriptor().shape() {
        return Ok(false);
    }
    let count = element_count(input.descriptor().shape())?;
    for linear in 0..count {
        check_periodically(linear, cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        let left = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        let right = other
            .descriptor()
            .dtype()
            .decode_scalar(other.element_bytes(&indices)?)?;
        if !decoded_numeric_equal(left, right) {
            return Ok(false);
        }
    }
    cancellation.check()?;
    Ok(true)
}

pub fn mlu_get_device_name_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<String, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    Ok(crate::native_device_name_exact(
        capabilities,
        device,
        DeviceKind::Mlu,
        MLU_DEVICE_NAME_OPERATION_ID,
        cancellation,
    )?)
}

pub fn sort_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<NativeSortResult, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    sort_impl(backend, input, dimension, descending, stable, context)
}
fn sort_impl(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<NativeSortResult, ElementwiseRuntimePartNineteenError> {
    let axis = normalize_axis(dimension, input.descriptor().rank(), SORT_OPERATION_ID)?;
    let indices =
        canonical_argsort_with_context(backend, input, dimension, descending, stable, context)?;
    let values = gather_sorted_values(backend, input, &indices, axis, context)?;
    context.cancellation.check()?;
    Ok(NativeSortResult { values, indices })
}

pub fn sort_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    sort_vjp_impl(
        backend,
        input,
        output_gradient,
        dimension,
        descending,
        stable,
        context,
    )
}
fn sort_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    require_matching_tensor(input, output_gradient, SORT_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), SORT_OPERATION_ID)?;
    let indices =
        canonical_argsort_with_context(backend, input, dimension, descending, stable, context)?;
    scatter_sorted_values(backend, output_gradient, &indices, axis, context)
}

pub fn sort_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    context.cancellation.check()?;
    sort_jvp_impl(
        backend,
        input,
        input_tangent,
        dimension,
        descending,
        stable,
        context,
    )
}
fn sort_jvp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    require_matching_tensor(input, input_tangent, SORT_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), SORT_OPERATION_ID)?;
    let indices =
        canonical_argsort_with_context(backend, input, dimension, descending, stable, context)?;
    gather_sorted_values(backend, input_tangent, &indices, axis, context)
}

pub fn directml_device_exact_native(
    capabilities: &BackendCapabilityMatrix,
    ordinal: u32,
    cancellation: &CancellationToken,
) -> Result<DeviceId, ElementwiseRuntimePartNineteenError> {
    cancellation.check()?;
    let device = DeviceId::new(DeviceKind::DirectMl, ordinal);
    if capabilities.device() != device {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation: DIRECTML_DEVICE_OPERATION_ID,
            device,
        });
    }
    cancellation.check()?;
    Ok(device)
}

fn gather_sorted_values(
    backend: &CpuBackend,
    input: &Tensor,
    indices: &Tensor,
    axis: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    require_cpu(input, SORT_OPERATION_ID)?;
    let shape = input.descriptor().shape();
    let mut bytes = temporary_vec(
        backend,
        context,
        byte_len(input.descriptor())?,
        "sorted values",
    )?;
    for linear in 0..element_count(shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, shape)?;
        let source_axis = sorted_source_axis(indices, &output_indices, shape[axis])?;
        let mut source_indices = output_indices;
        source_indices[axis] = source_axis;
        for byte in input.element_bytes(&source_indices)? {
            bytes.try_push(*byte)?;
        }
    }
    upload_bytes(backend, input, &bytes, context)
}

fn scatter_sorted_values(
    backend: &CpuBackend,
    sorted: &Tensor,
    indices: &Tensor,
    axis: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    require_cpu(sorted, SORT_OPERATION_ID)?;
    let shape = sorted.descriptor().shape();
    let element_width = usize::try_from(sorted.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("element width"))?;
    let byte_count = byte_len(sorted.descriptor())?;
    let mut bytes = temporary_vec(backend, context, byte_count, "sort gradient")?;
    for _ in 0..byte_count {
        bytes.try_push(0)?;
    }
    for linear in 0..element_count(shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, shape)?;
        let source_axis = sorted_source_axis(indices, &output_indices, shape[axis])?;
        let mut source_indices = output_indices.clone();
        source_indices[axis] = source_axis;
        let source_linear = ravel_index(&source_indices, shape)?;
        let byte_offset = source_linear.checked_mul(element_width).ok_or(
            ElementwiseRuntimePartNineteenError::ShapeOverflow("sort gradient offset"),
        )?;
        let byte_end = byte_offset.checked_add(element_width).ok_or(
            ElementwiseRuntimePartNineteenError::ShapeOverflow("sort gradient end"),
        )?;
        let destination = bytes.get_mut(byte_offset..byte_end).ok_or(
            ElementwiseRuntimePartNineteenError::ShapeOverflow("sort gradient range"),
        )?;
        destination.copy_from_slice(sorted.element_bytes(&output_indices)?);
    }
    upload_bytes(backend, sorted, &bytes, context)
}

fn sorted_source_axis(
    indices: &Tensor,
    output_indices: &[u64],
    axis_length: u64,
) -> Result<u64, ElementwiseRuntimePartNineteenError> {
    if indices.descriptor().dtype() != DType::I64 {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDType {
            operation: SORT_OPERATION_ID,
            dtype: indices.descriptor().dtype(),
        });
    }
    let bytes: [u8; 8] = indices
        .element_bytes(output_indices)?
        .try_into()
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("sort index width"))?;
    let value = i64::from_ne_bytes(bytes);
    let value = u64::try_from(value).map_err(|_| ElementwiseRuntimePartNineteenError::Invalid {
        operation: SORT_OPERATION_ID,
        reason: "canonical argsort returned a negative index".to_owned(),
    })?;
    if value >= axis_length {
        return invalid(
            SORT_OPERATION_ID,
            "canonical argsort returned an out-of-range index",
        );
    }
    Ok(value)
}

fn upload_bytes(
    backend: &CpuBackend,
    template: &Tensor,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineteenError> {
    let descriptor = TensorDescriptor::contiguous(
        template.descriptor().shape().to_vec(),
        template.descriptor().dtype(),
        DeviceId::CPU,
        template.descriptor().stream(),
    )?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartNineteenError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    Ok(())
}

fn require_matching_tensor(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartNineteenError> {
    if input.descriptor().shape() != other.descriptor().shape() {
        return invalid(operation, "tensor shapes must match");
    }
    if input.descriptor().dtype() != other.descriptor().dtype() {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDType {
            operation,
            dtype: other.descriptor().dtype(),
        });
    }
    if input.descriptor().device() != other.descriptor().device()
        || input.descriptor().stream() != other.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartNineteenError::UnsupportedDevice {
            operation,
            device: other.descriptor().device(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum RealScalar {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

fn decoded_numeric_equal(left: DecodedScalar, right: DecodedScalar) -> bool {
    match (left, right) {
        (
            DecodedScalar::Complex {
                real: left_real,
                imaginary: left_imaginary,
            },
            DecodedScalar::Complex {
                real: right_real,
                imaginary: right_imaginary,
            },
        ) => left_real == right_real && left_imaginary == right_imaginary,
        (DecodedScalar::Complex { real, imaginary }, right) => {
            imaginary == 0.0
                && real_scalar(right)
                    .is_some_and(|right| real_scalar_equal(RealScalar::Float(real), right))
        }
        (left, DecodedScalar::Complex { real, imaginary }) => {
            imaginary == 0.0
                && real_scalar(left)
                    .is_some_and(|left| real_scalar_equal(left, RealScalar::Float(real)))
        }
        (left, right) => real_scalar(left)
            .zip(real_scalar(right))
            .is_some_and(|(left, right)| real_scalar_equal(left, right)),
    }
}

fn real_scalar(value: DecodedScalar) -> Option<RealScalar> {
    match value {
        DecodedScalar::Boolean(value) => Some(RealScalar::Unsigned(u64::from(value))),
        DecodedScalar::Signed(value) => Some(RealScalar::Signed(value)),
        DecodedScalar::Unsigned(value) => Some(RealScalar::Unsigned(value)),
        DecodedScalar::Real(value) => Some(RealScalar::Float(value)),
        DecodedScalar::Complex { .. } => None,
    }
}

fn real_scalar_equal(left: RealScalar, right: RealScalar) -> bool {
    match (left, right) {
        (RealScalar::Signed(left), RealScalar::Signed(right)) => left == right,
        (RealScalar::Unsigned(left), RealScalar::Unsigned(right)) => left == right,
        (RealScalar::Signed(left), RealScalar::Unsigned(right))
        | (RealScalar::Unsigned(right), RealScalar::Signed(left)) => {
            left >= 0 && u64::try_from(left).is_ok_and(|left| left == right)
        }
        (RealScalar::Float(left), RealScalar::Float(right)) => left == right,
        (RealScalar::Signed(integer), RealScalar::Float(real))
        | (RealScalar::Float(real), RealScalar::Signed(integer)) => {
            real.is_finite()
                && real.fract() == 0.0
                && real >= -9_223_372_036_854_775_808.0
                && real < 9_223_372_036_854_775_808.0
                && (real as i64) == integer
        }
        (RealScalar::Unsigned(integer), RealScalar::Float(real))
        | (RealScalar::Float(real), RealScalar::Unsigned(integer)) => {
            real.is_finite()
                && real.fract() == 0.0
                && real >= 0.0
                && real < 18_446_744_073_709_551_616.0
                && (real as u64) == integer
        }
    }
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ElementwiseRuntimePartNineteenError> {
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("tensor rank"))?;
    let normalized = if dimension < 0 {
        dimension.checked_add(rank)
    } else {
        Some(dimension)
    };
    let normalized = normalized
        .filter(|axis| *axis >= 0 && *axis < rank)
        .ok_or_else(|| ElementwiseRuntimePartNineteenError::Invalid {
            operation,
            reason: format!("dimension {dimension} is out of range for rank {rank}"),
        })?;
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("normalized dimension"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartNineteenError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartNineteenError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(
                usize::try_from(*dimension)
                    .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("dimension"))?,
            )
            .ok_or(ElementwiseRuntimePartNineteenError::ShapeOverflow(
                "element count",
            ))
    })
}

fn byte_len(descriptor: &TensorDescriptor) -> Result<usize, ElementwiseRuntimePartNineteenError> {
    usize::try_from(descriptor.byte_len()?)
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("tensor bytes"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartNineteenError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.len())
        .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("tensor indices"))?;
    indices.resize(shape.len(), 0);
    for axis in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[axis])
            .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("dimension"))?;
        if dimension == 0 {
            return invalid(SORT_OPERATION_ID, "cannot index an empty tensor");
        }
        indices[axis] = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartNineteenError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartNineteenError> {
    if indices.len() != shape.len() {
        return invalid(SORT_OPERATION_ID, "index rank does not match tensor rank");
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            if index >= dimension {
                return invalid(SORT_OPERATION_ID, "tensor index is out of bounds");
            }
            linear
                .checked_mul(
                    usize::try_from(*dimension).map_err(|_| {
                        ElementwiseRuntimePartNineteenError::ShapeOverflow("dimension")
                    })?,
                )
                .and_then(|value| value.checked_add(usize::try_from(*index).ok()?))
                .ok_or(ElementwiseRuntimePartNineteenError::ShapeOverflow(
                    "linear tensor index",
                ))
        })
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ElementwiseRuntimePartNineteenError> {
    Err(ElementwiseRuntimePartNineteenError::Invalid {
        operation,
        reason: reason.into(),
    })
}
