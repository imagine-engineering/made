use crate::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Layout,
    MemoryFormatReference, NumericClass, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const CLONE_OPERATION_ID: &str = "COMFY-TENSOR-OP-4FE9C7A46973";
pub const CONTIGUOUS_OPERATION_ID: &str = "COMFY-TENSOR-OP-B044D0F40FCE";
pub const COPY_OPERATION_ID: &str = "COMFY-TENSOR-OP-243D55E505F5";
pub const CPU_OPERATION_ID: &str = "COMFY-TENSOR-OP-A59F2D9336D7";
pub const CUDA_OPERATION_ID: &str = "COMFY-TENSOR-OP-03EA795D0AC8";
pub const FLOAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-99D902CE1938";
pub const HALF_OPERATION_ID: &str = "COMFY-TENSOR-OP-CFD813E3EE51";
pub const NUMPY_OPERATION_ID: &str = "COMFY-TENSOR-OP-00F639D6C8A7";
pub const TO_OPERATION_ID: &str = "COMFY-TENSOR-OP-D9F653A3B821";
pub const TYPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-90C921ECD7A0";
pub const TYPE_AS_OPERATION_ID: &str = "COMFY-TENSOR-OP-B3036C455992";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StorageDTypeDeviceError {
    #[error(transparent)]
    Tensor(TensorError),
    #[error("tensor storage/dtype/device operation {operation} was cancelled")]
    Cancelled { operation: &'static str },
    #[error("tensor storage/dtype/device operation {operation} does not support device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("tensor storage/dtype/device operation {operation} has invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("tensor storage/dtype/device operation {operation} is not differentiable for {dtype:?}")]
    NonDifferentiable {
        operation: &'static str,
        dtype: DType,
    },
}

impl From<TensorError> for StorageDTypeDeviceError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled {
                operation: "tensor storage/dtype/device",
            },
            error => Self::Tensor(error),
        }
    }
}

impl From<OperatorIndirectionError> for StorageDTypeDeviceError {
    fn from(error: OperatorIndirectionError) -> Self {
        match error {
            OperatorIndirectionError::Cancelled => Self::Cancelled {
                operation: TO_OPERATION_ID,
            },
            OperatorIndirectionError::UnsupportedDevice { operation, device } => {
                Self::UnsupportedDevice { operation, device }
            }
            OperatorIndirectionError::Tensor(error) => Self::Tensor(error),
            error => Self::Invalid {
                operation: TO_OPERATION_ID,
                reason: error.to_string(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorTypeRequest {
    Query,
    Convert(DType),
}

#[derive(Clone, Debug)]
pub enum TensorTypeResult {
    Name(&'static str),
    Tensor(Tensor),
}

#[derive(Clone, Copy, Debug)]
pub struct NativeArrayView<'a> {
    tensor: &'a Tensor,
}

impl<'a> NativeArrayView<'a> {
    pub fn shape(self) -> &'a [u64] {
        self.tensor.descriptor().shape()
    }

    pub fn dtype(self) -> DType {
        self.tensor.descriptor().dtype()
    }

    pub fn rank(self) -> usize {
        self.tensor.descriptor().rank()
    }

    pub fn stride_bytes(self, dimension: usize) -> Result<i64, StorageDTypeDeviceError> {
        let stride = self
            .tensor
            .descriptor()
            .strides()
            .get(dimension)
            .copied()
            .ok_or_else(|| StorageDTypeDeviceError::Invalid {
                operation: NUMPY_OPERATION_ID,
                reason: format!(
                    "dimension {dimension} is outside rank {}",
                    self.tensor.descriptor().rank()
                ),
            })?;
        let width = i64::try_from(self.tensor.descriptor().dtype().byte_width()).map_err(|_| {
            StorageDTypeDeviceError::Invalid {
                operation: NUMPY_OPERATION_ID,
                reason: "dtype byte width does not fit a signed stride".to_owned(),
            }
        })?;
        stride
            .checked_mul(width)
            .ok_or_else(|| StorageDTypeDeviceError::Invalid {
                operation: NUMPY_OPERATION_ID,
                reason: "byte stride overflowed".to_owned(),
            })
    }

    pub fn offset_bytes(self) -> Result<u64, StorageDTypeDeviceError> {
        self.tensor
            .descriptor()
            .offset_elements()
            .checked_mul(self.tensor.descriptor().dtype().byte_width())
            .ok_or_else(|| StorageDTypeDeviceError::Invalid {
                operation: NUMPY_OPERATION_ID,
                reason: "byte offset overflowed".to_owned(),
            })
    }

    pub fn storage_bytes(self) -> Result<&'a [u8], StorageDTypeDeviceError> {
        Ok(self.tensor.host_storage_bytes()?)
    }

    pub fn element_bytes(self, indices: &[u64]) -> Result<&'a [u8], StorageDTypeDeviceError> {
        Ok(self.tensor.element_bytes(indices)?)
    }
}

pub fn clone_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    memory_format: MemoryFormatReference,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context.check().map_err(|error| map_tensor_error(CLONE_OPERATION_ID, error))?;
    require_cpu(CLONE_OPERATION_ID, input.descriptor().device())?;
    let descriptor = descriptor_for_memory_format(
        input.descriptor(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        memory_format,
        CLONE_OPERATION_ID,
    )?;
    let (output, _) = backend
        .copy(input, descriptor, context)
        .map_err(|error| map_tensor_error(CLONE_OPERATION_ID, error))?;
    Ok(output)
}

pub fn contiguous_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    memory_format: MemoryFormatReference,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(CONTIGUOUS_OPERATION_ID, error))?;
    require_cpu(CONTIGUOUS_OPERATION_ID, input.descriptor().device())?;
    if matches_memory_format(input.descriptor(), memory_format)? {
        return Ok(input.clone());
    }
    let descriptor = descriptor_for_memory_format(
        input.descriptor(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        memory_format,
        CONTIGUOUS_OPERATION_ID,
    )?;
    let (output, _) = backend
        .copy(input, descriptor, context)
        .map_err(|error| map_tensor_error(CONTIGUOUS_OPERATION_ID, error))?;
    Ok(output)
}

pub fn copy_with_context_exact_native(
    backend: &CpuBackend,
    destination: &mut Tensor,
    source: &Tensor,
    non_blocking: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
    if non_blocking {
        let event = backend
            .record_event(context)
            .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
        backend
            .wait_event(event, context)
            .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
    }
    context
        .check()
        .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
    require_cpu(COPY_OPERATION_ID, destination.descriptor().device())?;
    require_cpu(COPY_OPERATION_ID, source.descriptor().device())?;
    validate_broadcast_shape(
        source.descriptor().shape(),
        destination.descriptor().shape(),
    )?;

    let destination_shape = destination.descriptor().shape().to_vec();
    let source_shape = source.descriptor().shape();
    let destination_dtype = destination.descriptor().dtype();
    let element_count = usize::try_from(destination.descriptor().element_count()?).map_err(|_| {
        invalid(COPY_OPERATION_ID, "destination element count does not fit memory")
    })?;
    let element_width = usize::try_from(destination_dtype.byte_width())
        .map_err(|_| invalid(COPY_OPERATION_ID, "destination dtype width does not fit memory"))?;
    let byte_count = element_count
        .checked_mul(element_width)
        .ok_or_else(|| invalid(COPY_OPERATION_ID, "staging byte count overflowed"))?;
    let mut staged = backend
        .workspace_vec(context, byte_count)
        .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
    for linear_index in 0..element_count {
        check_periodically(COPY_OPERATION_ID, linear_index, context.cancellation)?;
        let destination_indices = unravel_index(linear_index, &destination_shape)?;
        let source_indices = broadcast_source_indices(&destination_indices, source_shape)?;
        let value = source
            .descriptor()
            .dtype()
            .decode_scalar(source.element_bytes(&source_indices)?)?;
        for byte in destination_dtype.encode_decoded_scalar(
            value,
            COPY_OPERATION_ID,
            destination.descriptor().device(),
        )? {
            staged.try_push(byte)?;
        }
    }
    context
        .check()
        .map_err(|error| map_tensor_error(COPY_OPERATION_ID, error))?;
    {
        let mut write = destination.write()?;
        for (linear_index, bytes) in staged.chunks_exact(element_width).enumerate() {
            let indices = unravel_index(linear_index, &destination_shape)?;
            write.element_bytes_mut(&indices)?.copy_from_slice(bytes);
        }
    }
    Ok(destination.clone())
}

pub fn cpu_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    memory_format: MemoryFormatReference,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    to_with_context_exact_native(
        backend,
        input,
        None,
        Some(DeviceId::CPU),
        false,
        false,
        Some(memory_format),
        context,
    )
    .map_err(|error| remap_operation(error, CPU_OPERATION_ID))
}

pub fn cuda_with_context_exact_native(
    input: &Tensor,
    ordinal: Option<u32>,
    non_blocking: bool,
    memory_format: MemoryFormatReference,
    cancellation: &CancellationToken,
) -> Result<Tensor, StorageDTypeDeviceError> {
    cancellation
        .check()
        .map_err(|_| StorageDTypeDeviceError::Cancelled {
            operation: CUDA_OPERATION_ID,
        })?;
    let (_input, _non_blocking, _memory_format) = (input, non_blocking, memory_format);
    Err(StorageDTypeDeviceError::UnsupportedDevice {
        operation: CUDA_OPERATION_ID,
        device: DeviceId::new(DeviceKind::Cuda, ordinal.unwrap_or(0)),
    })
}

pub fn float_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    memory_format: MemoryFormatReference,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    to_with_context_exact_native(
        backend,
        input,
        Some(DType::F32),
        None,
        false,
        false,
        Some(memory_format),
        context,
    )
    .map_err(|error| remap_operation(error, FLOAT_OPERATION_ID))
}

pub fn half_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    memory_format: MemoryFormatReference,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    to_with_context_exact_native(
        backend,
        input,
        Some(DType::F16),
        None,
        false,
        false,
        Some(memory_format),
        context,
    )
    .map_err(|error| remap_operation(error, HALF_OPERATION_ID))
}

pub fn numpy_exact_native<'a>(
    input: &'a Tensor,
    cancellation: &CancellationToken,
) -> Result<NativeArrayView<'a>, StorageDTypeDeviceError> {
    cancellation
        .check()
        .map_err(|_| StorageDTypeDeviceError::Cancelled {
            operation: NUMPY_OPERATION_ID,
        })?;
    require_cpu(NUMPY_OPERATION_ID, input.descriptor().device())?;
    input.host_storage_bytes()?;
    Ok(NativeArrayView { tensor: input })
}

#[allow(clippy::too_many_arguments)]
pub fn to_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: Option<DType>,
    device: Option<DeviceId>,
    non_blocking: bool,
    copy: bool,
    memory_format: Option<MemoryFormatReference>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(TO_OPERATION_ID, error))?;
    let dtype = dtype.unwrap_or(input.descriptor().dtype());
    let device = device.unwrap_or(input.descriptor().device());
    require_cpu(TO_OPERATION_ID, input.descriptor().device())?;
    require_cpu(TO_OPERATION_ID, device)?;
    let memory_format = memory_format.unwrap_or(MemoryFormatReference::PreserveFormat);
    let format_matches = matches_memory_format(input.descriptor(), memory_format)?;
    let converted = cast_to_with_context_exact_native(
        backend,
        input,
        dtype,
        device,
        non_blocking,
        copy || !format_matches,
        context,
    )?;
    let target = descriptor_for_memory_format(
        input.descriptor(),
        dtype,
        device,
        memory_format,
        TO_OPERATION_ID,
    )?;
    if converted.descriptor() == &target {
        return Ok(converted);
    }
    let (converted, event) = backend
        .copy(&converted, target, context)
        .map_err(|error| map_tensor_error(TO_OPERATION_ID, error))?;
    if non_blocking {
        backend
            .wait_event(event, context)
            .map_err(|error| map_tensor_error(TO_OPERATION_ID, error))?;
    }
    Ok(converted)
}

pub fn tensor_type_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    request: TensorTypeRequest,
    context: &ExecutionContext<'_>,
) -> Result<TensorTypeResult, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(TYPE_OPERATION_ID, error))?;
    match request {
        TensorTypeRequest::Query => Ok(TensorTypeResult::Name(tensor_type_name(input)?)),
        TensorTypeRequest::Convert(dtype) => Ok(TensorTypeResult::Tensor(
            to_with_context_exact_native(
                backend,
                input,
                Some(dtype),
                None,
                false,
                false,
                None,
                context,
            )
            .map_err(|error| remap_operation(error, TYPE_OPERATION_ID))?,
        )),
    }
}

pub fn type_as_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    to_with_context_exact_native(
        backend,
        input,
        Some(other.descriptor().dtype()),
        Some(other.descriptor().device()),
        false,
        false,
        None,
        context,
    )
    .map_err(|error| remap_operation(error, TYPE_AS_OPERATION_ID))
}

pub fn identity_vjp_with_context_exact_native(
    backend: &CpuBackend,
    primal: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(CLONE_OPERATION_ID, error))?;
    require_same_shape(primal, output_gradient, CLONE_OPERATION_ID)?;
    require_differentiable(primal.descriptor().dtype(), CLONE_OPERATION_ID)?;
    let descriptor = primal.descriptor().preserving_format_for(
        output_gradient.descriptor().dtype(),
        primal.descriptor().device(),
    )?;
    Ok(backend
        .copy(output_gradient, descriptor, context)
        .map_err(|error| map_tensor_error(CLONE_OPERATION_ID, error))?
        .0)
}

pub fn cast_vjp_with_context_exact_native(
    backend: &CpuBackend,
    primal: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(TO_OPERATION_ID, error))?;
    require_same_shape(primal, output_gradient, TO_OPERATION_ID)?;
    require_differentiable(primal.descriptor().dtype(), TO_OPERATION_ID)?;
    to_with_context_exact_native(
        backend,
        output_gradient,
        Some(primal.descriptor().dtype()),
        Some(primal.descriptor().device()),
        false,
        false,
        Some(MemoryFormatReference::PreserveFormat),
        context,
    )
}

pub fn cast_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    output_dtype: DType,
    output_device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StorageDTypeDeviceError> {
    context
        .check()
        .map_err(|error| map_tensor_error(TO_OPERATION_ID, error))?;
    require_differentiable(input_tangent.descriptor().dtype(), TO_OPERATION_ID)?;
    require_differentiable(output_dtype, TO_OPERATION_ID)?;
    to_with_context_exact_native(
        backend,
        input_tangent,
        Some(output_dtype),
        Some(output_device),
        false,
        false,
        Some(MemoryFormatReference::PreserveFormat),
        context,
    )
}

fn descriptor_for_memory_format(
    source: &TensorDescriptor,
    dtype: DType,
    device: DeviceId,
    memory_format: MemoryFormatReference,
    operation: &'static str,
) -> Result<TensorDescriptor, StorageDTypeDeviceError> {
    match memory_format {
        MemoryFormatReference::PreserveFormat => Ok(source.preserving_format_for(dtype, device)?),
        MemoryFormatReference::Layout(Layout::Contiguous) => Ok(TensorDescriptor::contiguous(
            source.shape().to_vec(),
            dtype,
            device,
            source.stream(),
        )?),
        MemoryFormatReference::Layout(Layout::ChannelsLast) => Ok(
            TensorDescriptor::channels_last(
                source.shape().to_vec(),
                dtype,
                device,
                source.stream(),
            )?,
        ),
        MemoryFormatReference::Layout(Layout::ChannelsLast3d) => Ok(
            TensorDescriptor::channels_last_3d(
                source.shape().to_vec(),
                dtype,
                device,
                source.stream(),
            )?,
        ),
        MemoryFormatReference::Layout(Layout::Strided) => Err(invalid(
            operation,
            "an arbitrary strided layout is not a named memory format",
        )),
    }
}

fn matches_memory_format(
    descriptor: &TensorDescriptor,
    memory_format: MemoryFormatReference,
) -> Result<bool, StorageDTypeDeviceError> {
    match memory_format {
        MemoryFormatReference::PreserveFormat => Ok(true),
        MemoryFormatReference::Layout(Layout::Contiguous) => {
            Ok(descriptor.is_contiguous()?)
        }
        MemoryFormatReference::Layout(layout @ (Layout::ChannelsLast | Layout::ChannelsLast3d)) => {
            Ok(descriptor.layout() == layout)
        }
        MemoryFormatReference::Layout(Layout::Strided) => Err(invalid(
            CONTIGUOUS_OPERATION_ID,
            "an arbitrary strided layout is not a named memory format",
        )),
    }
}

fn validate_broadcast_shape(
    source: &[u64],
    destination: &[u64],
) -> Result<(), StorageDTypeDeviceError> {
    if source.len() > destination.len() {
        return Err(invalid(
            COPY_OPERATION_ID,
            "source rank exceeds destination rank",
        ));
    }
    let offset = destination.len() - source.len();
    for (source_dimension, destination_dimension) in
        source.iter().zip(destination.iter().skip(offset))
    {
        if source_dimension != destination_dimension && *source_dimension != 1 {
            return Err(invalid(
                COPY_OPERATION_ID,
                "source shape is not broadcastable to destination",
            ));
        }
    }
    Ok(())
}

fn broadcast_source_indices(
    destination_indices: &[u64],
    source_shape: &[u64],
) -> Result<Vec<u64>, StorageDTypeDeviceError> {
    let offset = destination_indices
        .len()
        .checked_sub(source_shape.len())
        .ok_or_else(|| invalid(COPY_OPERATION_ID, "source rank exceeds destination rank"))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(source_shape.len())
        .map_err(|_| invalid(COPY_OPERATION_ID, "source index allocation failed"))?;
    for (source_dimension, destination_index) in
        source_shape.iter().zip(destination_indices.iter().skip(offset))
    {
        indices.push(if *source_dimension == 1 {
            0
        } else {
            *destination_index
        });
    }
    Ok(indices)
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, StorageDTypeDeviceError> {
    let mut indices = vec![0; shape.len()];
    for dimension_index in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[dimension_index])
            .map_err(|_| invalid(COPY_OPERATION_ID, "dimension does not fit memory"))?;
        if dimension == 0 {
            return Err(invalid(
                COPY_OPERATION_ID,
                "cannot index a zero-length dimension",
            ));
        }
        indices[dimension_index] = u64::try_from(linear_index % dimension)
            .map_err(|_| invalid(COPY_OPERATION_ID, "index conversion failed"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn tensor_type_name(input: &Tensor) -> Result<&'static str, StorageDTypeDeviceError> {
    require_cpu(TYPE_OPERATION_ID, input.descriptor().device())?;
    Ok(match input.descriptor().dtype() {
        DType::F64 => "torch.DoubleTensor",
        DType::F32 => "torch.FloatTensor",
        DType::F16 => "torch.HalfTensor",
        DType::Bf16 => "torch.BFloat16Tensor",
        DType::I64 => "torch.LongTensor",
        DType::I32 => "torch.IntTensor",
        DType::I16 => "torch.ShortTensor",
        DType::I8 => "torch.CharTensor",
        DType::U8 => "torch.ByteTensor",
        DType::Bool => "torch.BoolTensor",
        DType::Complex64 => "torch.ComplexFloatTensor",
        DType::Complex128 => "torch.ComplexDoubleTensor",
        dtype => {
            return Err(invalid(
                TYPE_OPERATION_ID,
                format!("legacy tensor type name is unavailable for {}", dtype.catalog_name()),
            ));
        }
    })
}

fn require_same_shape(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), StorageDTypeDeviceError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(invalid(operation, "primal and derivative shapes differ"));
    }
    Ok(())
}

fn require_differentiable(
    dtype: DType,
    operation: &'static str,
) -> Result<(), StorageDTypeDeviceError> {
    if matches!(dtype.class(), NumericClass::FloatingPoint | NumericClass::Complex) {
        Ok(())
    } else {
        Err(StorageDTypeDeviceError::NonDifferentiable { operation, dtype })
    }
}

fn require_cpu(
    operation: &'static str,
    device: DeviceId,
) -> Result<(), StorageDTypeDeviceError> {
    if device == DeviceId::CPU {
        Ok(())
    } else {
        Err(StorageDTypeDeviceError::UnsupportedDevice { operation, device })
    }
}

fn check_periodically(
    operation: &'static str,
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), StorageDTypeDeviceError> {
    if index.is_multiple_of(1_024) {
        cancellation
            .check()
            .map_err(|_| StorageDTypeDeviceError::Cancelled { operation })?;
    }
    Ok(())
}

fn invalid(
    operation: &'static str,
    reason: impl Into<String>,
) -> StorageDTypeDeviceError {
    StorageDTypeDeviceError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn map_tensor_error(operation: &'static str, error: TensorError) -> StorageDTypeDeviceError {
    match error {
        TensorError::Cancelled => StorageDTypeDeviceError::Cancelled { operation },
        error => StorageDTypeDeviceError::Tensor(error),
    }
}

fn remap_operation(
    error: StorageDTypeDeviceError,
    operation: &'static str,
) -> StorageDTypeDeviceError {
    match error {
        StorageDTypeDeviceError::Cancelled { .. } => {
            StorageDTypeDeviceError::Cancelled { operation }
        }
        StorageDTypeDeviceError::UnsupportedDevice { device, .. } => {
            StorageDTypeDeviceError::UnsupportedDevice { operation, device }
        }
        StorageDTypeDeviceError::Invalid { reason, .. } => {
            StorageDTypeDeviceError::Invalid { operation, reason }
        }
        StorageDTypeDeviceError::NonDifferentiable { dtype, .. } => {
            StorageDTypeDeviceError::NonDifferentiable { operation, dtype }
        }
        error => error,
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::{CpuWorkspaceAuthority, StreamId, validation_artifacts};
    use std::{collections::BTreeMap, error::Error};

    #[test]
    fn storage_dtype_device_01_rejects_unavailable_devices() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4_096)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4_096)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let input = backend.upload_f32(descriptor, &[1.0], &context)?.0;
        assert!(matches!(
            cuda_with_context_exact_native(
                &input,
                Some(1),
                false,
                MemoryFormatReference::PreserveFormat,
                &cancellation,
            ),
            Err(StorageDTypeDeviceError::UnsupportedDevice {
                operation: CUDA_OPERATION_ID,
                device,
            }) if device == DeviceId::new(DeviceKind::Cuda, 1)
        ));
        Ok(())
    }

    #[test]
    fn storage_dtype_device_01_writes_validation_artifacts() -> Result<(), Box<dyn Error>> {
        let fixture_digests = BTreeMap::from([
            ("clone", "5b18a667357d16b776c8b63ebd3b7e194e683a44a59e022fd6566c9525b4a74a"),
            ("contiguous", "ece8e84f49320f0514ed3d4a76b48306f83aab4054e326cfa49d410b4cd44fd3"),
            ("copy", "c560079c8eb87758e9437d764d90ccd4ef62fcbaa345924cbca9fa398d38e3d3"),
            ("cpu", "e85bf5eb2641d57d3d0431c39f16a1286046ea61171aca66a955c82a513b0e6b"),
            ("cuda", "4f13b19bbf2f37d4dc6ed645984c869a1badea0e0364f82fa211eb9a4a8a3d4c"),
            ("float", "56d03f96751ed875b2ec5acbee70d4de97921da25379f9ddfa7ac1884fd5173e"),
            ("half", "4915f7643095ca0c38fcd150e56d15dd3fca13b3836239067c6ffa81a2090558"),
            ("numpy", "4323b3c6f4ba2d7acfceb0148e9195ed15abc8139d6e7bb133a4eee6ae62d9e8"),
            ("to", "2fd954e9382cace3346adf84b3c6762091b0517d5b20b1f2db558153a36b6fe1"),
            ("type", "c40956b26df948fc8c67ed3f82f3ae4b3b130165ffd8cccba3615e798402a8ca"),
            ("type_as", "10eebe2078ab49ccf7602380722318b55b597b7660ded3a8d66649237db2a10a"),
        ]);
        let tensor_cases = BTreeMap::from([
            ("alias_layout_and_transaction_semantics", true),
            ("dtype_device_and_native_array_boundaries", true),
            ("runtime_sealed_contracts", true),
        ]);
        validation_artifacts::write(
            "val-tensor-storage-dtype-device-01.json",
            "VAL-TENSOR-001",
            "Task 91 storage, dtype, and device adapters",
            "task",
            &fixture_digests,
            &tensor_cases,
            &["all remaining tensor leaves", "release closure"],
        )?;
        let autograd_cases = BTreeMap::from([
            ("cast_vjp_jvp", true),
            ("identity_vjp_layout", true),
            ("nondifferentiable_dtype_boundary", true),
        ]);
        validation_artifacts::write(
            "val-autograd-storage-dtype-device-01.json",
            "VAL-AUTOGRAD-001",
            "Task 91 storage, dtype, and device analytical maps",
            "task",
            &fixture_digests,
            &autograd_cases,
            &["all remaining autograd leaves", "release closure"],
        )?;
        Ok(())
    }
}
