use std::collections::BTreeSet;

use crate::{
    AutocastPolicy, BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend,
    CpuWorkspaceVec, CustomKernelId, DType, DecodedScalar, DeviceId, ExecutionContext,
    NativeStream, NativeStreamRegistry, Scalar, ScalarSide, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, log_softmax_vjp_with_context_exact_native as canonical_log_softmax_vjp,
        log_softmax_jvp_with_context_exact_native as canonical_log_softmax_jvp,
        log_softmax_with_context_exact_native as canonical_log_softmax,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, tensor_constructor_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_12::{
        ElementwiseRuntimePartTwelveError, numel_exact_native as canonical_numel,
    },
    generated_elementwise_or_runtime_operation_16::{
        AddGradients, ElementwiseRuntimePartSixteenError,
        add_method_jvp_with_context_exact_native as canonical_add_method_jvp_with_context,
        add_method_vjp_with_context_exact_native as canonical_add_method_vjp_with_context,
        add_method_with_context_exact_native as canonical_add_method_with_context,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const BYTE_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-C579143F7B56";
pub const ELEMENT_SIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-C83ECA429710";
pub const SUB_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-C1FAC5999B98";
pub const ADD_OPERATION_ID: &str = "COMFY-TENSOR-OP-C575200CD790";
pub const BINCOUNT_OPERATION_ID: &str = "COMFY-TENSOR-OP-C7A255E21877";
pub const BUCKETIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-C7B72CD0ABE7";
pub const COMPILER_IS_COMPILING_OPERATION_ID: &str = "COMFY-TENSOR-OP-C0D4EF19ED71";
pub const IS_AUTOCAST_ENABLED_OPERATION_ID: &str = "COMFY-TENSOR-OP-C64E630A756E";
pub const LIBRARY_CUSTOM_OP_OPERATION_ID: &str = "COMFY-TENSOR-OP-C63B343CD3EF";
pub const LOG_SOFTMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-C46EB25624FB";
pub const NUMEL_OPERATION_ID: &str = "COMFY-TENSOR-OP-C1BBE55AA3A0";
pub const XPU_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-C05FE0730305";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCustomOperatorDeclaration {
    qualified_name: String,
    kernel: CustomKernelId,
    mutates_arguments: Vec<usize>,
}

impl NativeCustomOperatorDeclaration {
    pub fn qualified_name(&self) -> &str {
        &self.qualified_name
    }

    pub fn kernel(&self) -> &CustomKernelId {
        &self.kernel
    }

    pub fn mutates_arguments(&self) -> &[usize] {
        &self.mutates_arguments
    }
}

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartEighteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error(transparent)]
    PartTwelve(#[from] ElementwiseRuntimePartTwelveError),
    #[error(transparent)]
    PartSixteen(#[from] ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error("elementwise/runtime part-eighteen execution was cancelled")]
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

impl From<sim_types::CancellationError> for ElementwiseRuntimePartEighteenError {
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
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartEighteenError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

pub fn byte_tensor_with_context_exact_native(
    backend: &CpuBackend,
    values: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    let mut scalars = temporary_vec(backend, context, values.len(), "byte tensor values")?;
    for (index, value) in values.iter().copied().enumerate() {
        check_periodically(index, context.cancellation)?;
        scalars.try_push(Scalar::Unsigned(u64::from(value)))?;
    }
    Ok(tensor_constructor_with_context_exact_native(
        backend,
        &scalars,
        &[u64::try_from(values.len()).map_err(|_| {
            ElementwiseRuntimePartEighteenError::ShapeOverflow("byte tensor length")
        })?],
        DType::U8,
        context.stream,
        context,
    )?)
}

pub fn element_size_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<u64, ElementwiseRuntimePartEighteenError> {
    cancellation.check()?;
    Ok(input.descriptor().dtype().byte_width())
}

pub fn sub_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_finite(alpha, SUB_METHOD_OPERATION_ID, "alpha")?;
    Ok(canonical_add_method_with_context(
        backend, input, other, -alpha, context,
    )?)
}

pub fn sub_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddGradients, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_finite(alpha, SUB_METHOD_OPERATION_ID, "alpha")?;
    Ok(canonical_add_method_vjp_with_context(
        backend,
        input,
        other,
        -alpha,
        output_gradient,
        context,
    )?)
}

pub fn sub_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_finite(alpha, SUB_METHOD_OPERATION_ID, "alpha")?;
    Ok(canonical_add_method_jvp_with_context(
        backend,
        input_tangent,
        other_tangent,
        -alpha,
        context,
    )?)
}

pub fn add_scalar_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: f32,
    other: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_f32_cpu(other, ADD_OPERATION_ID)?;
    require_finite(input, ADD_OPERATION_ID, "input")?;
    require_finite(alpha, ADD_OPERATION_ID, "alpha")?;
    let descriptor = contiguous_like(other, DType::F32)?;
    let scaled = backend
        .binary_scalar(
            BinaryOperation::Multiply,
            other,
            Scalar::Float(f64::from(alpha)),
            ScalarSide::Right,
            descriptor.clone(),
            context,
        )?
        .0;
    Ok(backend
        .binary_scalar(
            BinaryOperation::Add,
            &scaled,
            Scalar::Float(f64::from(input)),
            ScalarSide::Left,
            descriptor,
            context,
        )?
        .0)
}

pub fn add_scalar_tensor_vjp_with_context_exact_native(
    backend: &CpuBackend,
    other: &Tensor,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_f32_cpu(other, ADD_OPERATION_ID)?;
    require_same_shape_and_stream(other, output_gradient, ADD_OPERATION_ID)?;
    scale_f32(backend, output_gradient, alpha, ADD_OPERATION_ID, context)
}

pub fn add_scalar_tensor_jvp_with_context_exact_native(
    backend: &CpuBackend,
    other_tangent: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    scale_f32(backend, other_tangent, alpha, ADD_OPERATION_ID, context)
}

pub fn bincount_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minlength: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    bincount_impl(backend, input, minlength, context)
}
fn bincount_impl(
    backend: &CpuBackend,
    input: &Tensor,
    minlength: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    require_cpu(input, BINCOUNT_OPERATION_ID)?;
    if input.descriptor().rank() != 1 {
        return invalid(
            BINCOUNT_OPERATION_ID,
            "bincount input must be one-dimensional",
        );
    }
    let count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("bincount input"))?;
    let mut indices = temporary_vec(backend, context, count, "bincount indices")?;
    let mut maximum = None;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        let value = integer_index(
            input,
            u64::try_from(index).map_err(|_| {
                ElementwiseRuntimePartEighteenError::ShapeOverflow("bincount index")
            })?,
            BINCOUNT_OPERATION_ID,
        )?;
        maximum = Some(maximum.map_or(value, |current: u64| current.max(value)));
        indices.try_push(value)?;
    }
    let output_length = maximum
        .map(|value| {
            value
                .checked_add(1)
                .ok_or(ElementwiseRuntimePartEighteenError::ShapeOverflow(
                    "bincount output",
                ))
        })
        .transpose()?
        .unwrap_or(0)
        .max(minlength);
    let output_length_usize = usize::try_from(output_length)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("bincount output"))?;
    let mut counts = temporary_vec(backend, context, output_length_usize, "bincount output")?;
    for index in 0..output_length_usize {
        check_periodically(index, context.cancellation)?;
        counts.try_push(0_i64)?;
    }
    for (position, index) in indices.iter().copied().enumerate() {
        check_periodically(position, context.cancellation)?;
        let slot =
            counts
                .get_mut(usize::try_from(index).map_err(|_| {
                    ElementwiseRuntimePartEighteenError::ShapeOverflow("bincount slot")
                })?)
                .ok_or(ElementwiseRuntimePartEighteenError::ShapeOverflow(
                    "bincount slot",
                ))?;
        *slot = slot
            .checked_add(1)
            .ok_or(ElementwiseRuntimePartEighteenError::ShapeOverflow(
                "bincount value",
            ))?;
    }
    upload_i64_with_context(
        backend,
        &[output_length],
        &counts,
        input.descriptor().stream(),
        context,
    )
}

pub fn bucketize_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    boundaries: &Tensor,
    right: bool,
    out_int32: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    bucketize_impl(backend, input, boundaries, right, out_int32, context)
}
fn bucketize_impl(
    backend: &CpuBackend,
    input: &Tensor,
    boundaries: &Tensor,
    right: bool,
    out_int32: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    require_f32_cpu(input, BUCKETIZE_OPERATION_ID)?;
    require_f32_cpu(boundaries, BUCKETIZE_OPERATION_ID)?;
    require_same_stream(input, boundaries, BUCKETIZE_OPERATION_ID)?;
    if boundaries.descriptor().rank() != 1 {
        return invalid(
            BUCKETIZE_OPERATION_ID,
            "bucket boundaries must be one-dimensional",
        );
    }
    let boundary_count = usize::try_from(boundaries.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("bucket boundaries"))?;
    let mut boundary_values = temporary_vec(backend, context, boundary_count, "bucket boundaries")?;
    for index in 0..boundary_count {
        check_periodically(index, context.cancellation)?;
        let value = f32_at_linear(boundaries, index, BUCKETIZE_OPERATION_ID)?;
        if value.is_nan()
            || boundary_values
                .last()
                .is_some_and(|previous| *previous > value)
        {
            return invalid(
                BUCKETIZE_OPERATION_ID,
                "bucket boundaries must be sorted and non-NaN",
            );
        }
        boundary_values.try_push(value)?;
    }
    let count = element_count(input.descriptor().shape())?;
    if out_int32 {
        let mut values = temporary_vec(backend, context, count, "i32 bucket output")?;
        for index in 0..count {
            check_periodically(index, context.cancellation)?;
            let value = f32_at_linear(input, index, BUCKETIZE_OPERATION_ID)?;
            let bucket = if value.is_nan() {
                boundary_values.len()
            } else if right {
                boundary_values.partition_point(|boundary| *boundary <= value)
            } else {
                boundary_values.partition_point(|boundary| *boundary < value)
            };
            values.try_push(i32::try_from(bucket).map_err(|_| {
                ElementwiseRuntimePartEighteenError::ShapeOverflow("i32 bucket index")
            })?)?;
        }
        upload_i32_with_context(
            backend,
            input.descriptor().shape(),
            &values,
            input.descriptor().stream(),
            context,
        )
    } else {
        let mut values = temporary_vec(backend, context, count, "i64 bucket output")?;
        for index in 0..count {
            check_periodically(index, context.cancellation)?;
            let value = f32_at_linear(input, index, BUCKETIZE_OPERATION_ID)?;
            let bucket = if value.is_nan() {
                boundary_values.len()
            } else if right {
                boundary_values.partition_point(|boundary| *boundary <= value)
            } else {
                boundary_values.partition_point(|boundary| *boundary < value)
            };
            values.try_push(i64::try_from(bucket).map_err(|_| {
                ElementwiseRuntimePartEighteenError::ShapeOverflow("i64 bucket index")
            })?)?;
        }
        upload_i64_with_context(
            backend,
            input.descriptor().shape(),
            &values,
            input.descriptor().stream(),
            context,
        )
    }
}

pub fn is_autocast_enabled_exact_native(
    policy: &AutocastPolicy,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartEighteenError> {
    cancellation.check()?;
    Ok(policy.enabled())
}

pub fn library_custom_op_exact_native(
    qualified_name: &str,
    mutates_arguments: &[usize],
    cancellation: &CancellationToken,
) -> Result<NativeCustomOperatorDeclaration, ElementwiseRuntimePartEighteenError> {
    cancellation.check()?;
    validate_custom_operator_name(qualified_name)?;
    let unique = mutates_arguments.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != mutates_arguments.len() {
        return invalid(
            LIBRARY_CUSTOM_OP_OPERATION_ID,
            "custom operator mutation arguments must be unique",
        );
    }
    let kernel = CustomKernelId::new(qualified_name.replace("::", "."))?;
    cancellation.check()?;
    Ok(NativeCustomOperatorDeclaration {
        qualified_name: qualified_name.to_owned(),
        kernel,
        mutates_arguments: mutates_arguments.to_vec(),
    })
}

pub fn log_softmax_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, LOG_SOFTMAX_OPERATION_ID)?;
    let shape = usize_shape(input.descriptor().shape())?;
    let values = f32_values_with_context(backend, input, LOG_SOFTMAX_OPERATION_ID, context)?;
    let dimension = isize::try_from(dimension)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("log softmax axis"))?;
    let output =
        canonical_log_softmax(backend, &values, &shape, dimension, DeviceId::CPU, context)?;
    upload_f32_with_context(backend, input.descriptor().shape(), &output, context)
}

pub fn log_softmax_vjp_exact_native(
    backend: &CpuBackend,
    output: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    log_softmax_derivative(backend, output, output_gradient, dimension, context)
}

pub fn log_softmax_jvp_exact_native(
    backend: &CpuBackend,
    output: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.cancellation.check()?;
    log_softmax_jvp_derivative(backend, output, input_tangent, dimension, context)
}

pub fn numel_function_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<u64, ElementwiseRuntimePartEighteenError> {
    cancellation.check()?;
    Ok(canonical_numel(input, cancellation)?)
}

pub fn xpu_stream_exact_native(
    registry: &NativeStreamRegistry,
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    priority: i32,
    cancellation: &CancellationToken,
) -> Result<NativeStream, ElementwiseRuntimePartEighteenError> {
    cancellation.check()?;
    if device.kind() != DeviceKind::Xpu {
        return Err(ElementwiseRuntimePartEighteenError::UnsupportedDevice {
            operation: XPU_STREAM_OPERATION_ID,
            device,
        });
    }
    Ok(registry.create(capabilities, device, priority, cancellation)?)
}

fn log_softmax_derivative(
    backend: &CpuBackend,
    output: &Tensor,
    tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    require_f32_cpu(output, LOG_SOFTMAX_OPERATION_ID)?;
    require_f32_cpu(tangent, LOG_SOFTMAX_OPERATION_ID)?;
    require_same_shape_and_stream(output, tangent, LOG_SOFTMAX_OPERATION_ID)?;
    let shape = usize_shape(output.descriptor().shape())?;
    let output_values =
        f32_values_with_context(backend, output, LOG_SOFTMAX_OPERATION_ID, context)?;
    let tangent_values =
        f32_values_with_context(backend, tangent, LOG_SOFTMAX_OPERATION_ID, context)?;
    let dimension = isize::try_from(dimension)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("log softmax axis"))?;
    let values = canonical_log_softmax_vjp(
        backend,
        &output_values,
        &tangent_values,
        &shape,
        dimension,
        DeviceId::CPU,
        context,
    )?;
    upload_f32_with_context(backend, output.descriptor().shape(), &values, context)
}

fn log_softmax_jvp_derivative(
    backend: &CpuBackend,
    output: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    require_f32_cpu(output, LOG_SOFTMAX_OPERATION_ID)?;
    require_f32_cpu(input_tangent, LOG_SOFTMAX_OPERATION_ID)?;
    require_same_shape_and_stream(output, input_tangent, LOG_SOFTMAX_OPERATION_ID)?;
    let shape = usize_shape(output.descriptor().shape())?;
    let output_values =
        f32_values_with_context(backend, output, LOG_SOFTMAX_OPERATION_ID, context)?;
    let tangent_values =
        f32_values_with_context(backend, input_tangent, LOG_SOFTMAX_OPERATION_ID, context)?;
    let dimension = isize::try_from(dimension)
        .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("log softmax axis"))?;
    let values = canonical_log_softmax_jvp(
        backend,
        &output_values,
        &tangent_values,
        &shape,
        dimension,
        DeviceId::CPU,
        context,
    )?;
    upload_f32_with_context(backend, output.descriptor().shape(), &values, context)
}

fn validate_custom_operator_name(
    qualified_name: &str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    let mut segments = qualified_name.split("::");
    let namespace = segments.next().unwrap_or_default();
    let name = segments.next().unwrap_or_default();
    if segments.next().is_some() || !valid_identifier(namespace) || !valid_identifier(name) {
        return invalid(
            LIBRARY_CUSTOM_OP_OPERATION_ID,
            "custom operator name must be namespace::identifier",
        );
    }
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn integer_index(
    input: &Tensor,
    index: u64,
    operation: &'static str,
) -> Result<u64, ElementwiseRuntimePartEighteenError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(&[index])?)?
    {
        DecodedScalar::Signed(value) if value >= 0 => Ok(u64::try_from(value)
            .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("integer index"))?),
        DecodedScalar::Unsigned(value) => Ok(value),
        DecodedScalar::Signed(_) => invalid(operation, "integer indices must be non-negative"),
        _ => Err(ElementwiseRuntimePartEighteenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        }),
    }
}

fn scale_f32(
    backend: &CpuBackend,
    input: &Tensor,
    scale: f32,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    require_f32_cpu(input, operation)?;
    require_finite(scale, operation, "scale")?;
    Ok(backend
        .binary_scalar(
            BinaryOperation::Multiply,
            input,
            Scalar::Float(f64::from(scale)),
            ScalarSide::Right,
            contiguous_like(input, DType::F32)?,
            context,
        )?
        .0)
}

fn f32_at_linear(
    input: &Tensor,
    linear: usize,
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartEighteenError> {
    let indices = unravel_index(linear, input.descriptor().shape())?;
    let bytes: [u8; 4] = input.element_bytes(&indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartEighteenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        }
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn f32_values_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ElementwiseRuntimePartEighteenError> {
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(f32_at_linear(input, index, operation)?)?;
    }
    Ok(values)
}

fn usize_shape(shape: &[u64]) -> Result<Vec<usize>, ElementwiseRuntimePartEighteenError> {
    shape
        .iter()
        .copied()
        .map(|dimension| {
            usize::try_from(dimension)
                .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("shape"))
        })
        .collect()
}

fn upload_i64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, stream)?;
    let mut bytes =
        temporary_vec(
            backend,
            context,
            values.len().checked_mul(8).ok_or(
                ElementwiseRuntimePartEighteenError::ShapeOverflow("i64 upload"),
            )?,
            "i64 upload",
        )?;
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        for byte in value.to_ne_bytes() {
            bytes.try_push(byte)?;
        }
    }
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn upload_i32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i32],
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I32, DeviceId::CPU, stream)?;
    let mut bytes =
        temporary_vec(
            backend,
            context,
            values.len().checked_mul(4).ok_or(
                ElementwiseRuntimePartEighteenError::ShapeOverflow("i32 upload"),
            )?,
            "i32 upload",
        )?;
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        for byte in value.to_ne_bytes() {
            bytes.try_push(byte)?;
        }
    }
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEighteenError> {
    context.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartEighteenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartEighteenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_same_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return invalid(operation, "tensor streams must match");
    }
    Ok(())
}

fn require_same_shape_and_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    require_same_stream(left, right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return invalid(operation, "tensor shapes must match");
    }
    Ok(())
}

fn require_finite(
    value: f32,
    operation: &'static str,
    field: &'static str,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    if !value.is_finite() {
        return invalid(operation, &format!("{field} must be finite"));
    }
    Ok(())
}

fn contiguous_like(
    input: &Tensor,
    dtype: DType,
) -> Result<TensorDescriptor, ElementwiseRuntimePartEighteenError> {
    Ok(TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        dtype,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?)
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartEighteenError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(
                usize::try_from(*dimension)
                    .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("dimension"))?,
            )
            .ok_or(ElementwiseRuntimePartEighteenError::ShapeOverflow(
                "element count",
            ))
    })
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartEighteenError> {
    let mut indices = vec![0_u64; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let size = usize::try_from(shape[dimension])
            .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("index dimension"))?;
        if size == 0 {
            return invalid(LOG_SOFTMAX_OPERATION_ID, "cannot index an empty tensor");
        }
        indices[dimension] = u64::try_from(linear % size)
            .map_err(|_| ElementwiseRuntimePartEighteenError::ShapeOverflow("index"))?;
        linear /= size;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartEighteenError> {
    if index.is_multiple_of(1_024) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: &str,
) -> Result<T, ElementwiseRuntimePartEighteenError> {
    Err(ElementwiseRuntimePartEighteenError::Invalid {
        operation,
        reason: reason.to_owned(),
    })
}
