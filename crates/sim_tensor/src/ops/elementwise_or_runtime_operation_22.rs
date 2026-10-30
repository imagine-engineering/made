use crate::{
    AutocastPolicy, BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DeviceId,
    ExecutionContext, Layout, MemoryFormatReference, Scalar, Tensor, TensorBackend,
    TensorDescriptor, TensorError, ViewAccess,
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError,
        acos_jvp_with_context_exact_native as canonical_acos_jvp_with_context,
        acos_vjp_with_context_exact_native as canonical_acos_vjp_with_context,
        acos_with_context_exact_native as canonical_acos_with_context,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseRuntimePartThreeError,
        floor_jvp_with_context_exact_native as canonical_floor_jvp_with_context,
        floor_vjp_with_context_exact_native as canonical_floor_vjp_with_context,
        floor_with_context_exact_native as canonical_floor_with_context,
    },
    generated_elementwise_or_runtime_operation_13::{
        ElementwiseRuntimePartThirteenError,
        lerp_jvp_with_context_exact_native as canonical_lerp_jvp_with_context,
        lerp_vjp_with_context_exact_native as canonical_lerp_vjp_with_context,
        lerp_with_context_exact_native as canonical_lerp_with_context,
    },
    generated_elementwise_or_runtime_operation_14::{
        ElementwiseRuntimePartFourteenError,
        argsort_with_context_exact_native as canonical_argsort_with_context,
        autocast_exact_native as canonical_autocast,
    },
    generated_elementwise_or_runtime_operation_20::{
        ElementwiseRuntimePartTwentyError, swapaxes_exact_native as canonical_swapaxes,
    },
    generated_elementwise_or_runtime_operation_21::{
        ElementwiseRuntimePartTwentyOneError,
        exp_jvp_with_context_exact_native as canonical_exp_jvp_with_context,
        exp_vjp_with_context_exact_native as canonical_exp_vjp_with_context,
        exp_with_context_exact_native as canonical_exp_with_context,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const ARGSORT_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-FB884955DE1E";
pub const FLOOR_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-F2D7AE6E8F48";
pub const T_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-F1F71360D559";
pub const ARCCOS_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-F56BA497ED13";
pub const CUDA_AMP_AUTOCAST_OPERATION_ID: &str = "COMFY-TENSOR-OP-FBA06A1411DE";
pub const CUDA_GET_DEVICE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-F9ED42F7BFDF";
pub const CUDA_MEMORY_STATS_OPERATION_ID: &str = "COMFY-TENSOR-OP-F18E1AE1B857";
pub const EXP_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-FA7DD244B7CA";
pub const LERP_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-F27D07B4E10D";
pub const NPU_MEM_GET_INFO_OPERATION_ID: &str = "COMFY-TENSOR-OP-F15A2D8A6BD4";
pub const XPU_SET_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-FACB7FC5B252";
pub const ZEROS_LIKE_OPERATION_ID: &str = "COMFY-TENSOR-OP-F3D0014DD82A";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTwentyTwoError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    PartThree(#[from] ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    PartThirteen(#[from] ElementwiseRuntimePartThirteenError),
    #[error(transparent)]
    PartFourteen(#[from] ElementwiseRuntimePartFourteenError),
    #[error(transparent)]
    PartTwenty(#[from] ElementwiseRuntimePartTwentyError),
    #[error(transparent)]
    PartTwentyOne(#[from] ElementwiseRuntimePartTwentyOneError),
    #[error("elementwise/runtime part-twenty-two execution was cancelled")]
    Cancelled,
    #[error("operation {operation} does not support device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: &'static str,
    },
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwentyTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn argsort_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_argsort_with_context(
        backend, input, dimension, descending, stable, context,
    )?)
}

pub fn floor_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_floor_with_context(backend, input, context)?)
}

pub fn floor_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    require_same_geometry(input, output_gradient, FLOOR_METHOD_OPERATION_ID)?;
    Ok(canonical_floor_vjp_with_context(backend, input, context)?)
}

pub fn floor_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    require_same_geometry(input, input_tangent, FLOOR_METHOD_OPERATION_ID)?;
    Ok(canonical_floor_jvp_with_context(backend, input, context)?)
}

pub fn t_method_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    let output = match input.descriptor().rank() {
        0 | 1 => input.view(input.descriptor().clone(), ViewAccess::ReadOnly)?,
        2 => canonical_swapaxes(input, 0, 1, cancellation)?,
        _ => {
            return Err(ElementwiseRuntimePartTwentyTwoError::Invalid {
                operation: T_METHOD_OPERATION_ID,
                reason: "t() expects a tensor with at most two dimensions",
            });
        }
    };
    cancellation.check()?;
    Ok(output)
}

pub fn t_method_vjp_exact_native(
    output_gradient: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    t_method_exact_native(output_gradient, cancellation)
}

pub fn t_method_jvp_exact_native(
    input_tangent: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    t_method_exact_native(input_tangent, cancellation)
}

pub fn arccos_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_acos_with_context(backend, input, context)?)
}

pub fn arccos_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_acos_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn arccos_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_acos_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn cuda_amp_autocast_exact_native(
    enabled: bool,
    dtype: Option<DType>,
    cache_enabled: Option<bool>,
    cancellation: &CancellationToken,
) -> Result<AutocastPolicy, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    Ok(canonical_autocast(
        DeviceKind::Cuda,
        dtype,
        enabled,
        cache_enabled,
        cancellation,
    )?)
}

pub fn cuda_get_device_name_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<String, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    Ok(crate::native_device_name_exact_for_kinds(
        capabilities,
        device,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        CUDA_GET_DEVICE_NAME_OPERATION_ID,
        cancellation,
    )?)
}

pub fn exp_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    out: Option<&mut Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    let result = canonical_exp_with_context(backend, input, context)?;
    context.check()?;
    if let Some(out) = out {
        out.commit_in_place(result)?;
        return Ok(out.clone());
    }
    Ok(result)
}

pub fn exp_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_exp_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn exp_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_exp_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn lerp_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    weight: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_lerp_with_context(
        backend, input, end, weight, context,
    )?)
}

pub fn lerp_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    weight: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<
    crate::generated_elementwise_or_runtime_operation_13::LerpVjp,
    ElementwiseRuntimePartTwentyTwoError,
> {
    context.cancellation.check()?;
    Ok(canonical_lerp_vjp_with_context(
        backend,
        input,
        end,
        weight,
        output_gradient,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn lerp_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    end: &Tensor,
    input_tangent: &Tensor,
    end_tangent: &Tensor,
    weight: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    Ok(canonical_lerp_jvp_with_context(
        backend,
        input,
        end,
        input_tangent,
        end_tangent,
        weight,
        context,
    )?)
}

pub fn xpu_set_device_exact_native<'a>(
    available: &'a [BackendCapabilityMatrix],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<&'a BackendCapabilityMatrix, ElementwiseRuntimePartTwentyTwoError> {
    cancellation.check()?;
    Ok(crate::native_select_device_exact(
        available,
        device,
        &[DeviceKind::Xpu],
        XPU_SET_DEVICE_OPERATION_ID,
        cancellation,
    )?)
}

pub fn zeros_like_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: Option<DType>,
    device: Option<DeviceId>,
    memory_format: Option<MemoryFormatReference>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyTwoError> {
    context.cancellation.check()?;
    context.check()?;
    let dtype = dtype.unwrap_or(input.descriptor().dtype());
    let device = device.unwrap_or(input.descriptor().device());
    if device != DeviceId::CPU {
        return Err(ElementwiseRuntimePartTwentyTwoError::UnsupportedDevice {
            operation: ZEROS_LIKE_OPERATION_ID,
            device,
        });
    }
    let descriptor = match memory_format.unwrap_or(MemoryFormatReference::PreserveFormat) {
        MemoryFormatReference::PreserveFormat => {
            input.descriptor().preserving_format_for(dtype, device)?
        }
        MemoryFormatReference::Layout(Layout::Contiguous) => TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            dtype,
            device,
            input.descriptor().stream(),
        )?,
        MemoryFormatReference::Layout(Layout::ChannelsLast) => TensorDescriptor::channels_last(
            input.descriptor().shape().to_vec(),
            dtype,
            device,
            input.descriptor().stream(),
        )?,
        MemoryFormatReference::Layout(Layout::ChannelsLast3d) => {
            TensorDescriptor::channels_last_3d(
                input.descriptor().shape().to_vec(),
                dtype,
                device,
                input.descriptor().stream(),
            )?
        }
        MemoryFormatReference::Layout(Layout::Strided) => {
            return Err(ElementwiseRuntimePartTwentyTwoError::Invalid {
                operation: ZEROS_LIKE_OPERATION_ID,
                reason: "an arbitrary strided layout is not a named memory format",
            });
        }
    };
    let output = backend.fill(Scalar::Unsigned(0), descriptor, context)?.0;
    context.check()?;
    Ok(output)
}

fn require_same_geometry(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyTwoError> {
    if input.descriptor().shape() != other.descriptor().shape()
        || input.descriptor().dtype() != other.descriptor().dtype()
        || input.descriptor().device() != other.descriptor().device()
        || input.descriptor().stream() != other.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartTwentyTwoError::Invalid {
            operation,
            reason: "primal and derivative tensor geometry must match",
        });
    }
    Ok(())
}
