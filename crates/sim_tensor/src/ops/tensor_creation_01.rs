use crate::{
    AutogradError, AutogradTape, BackendStorage, CancellationToken, CpuBackend, DType,
    DecodedScalar, DeviceId, ExecutionContext, Layout, LeafId, NumericClass, Scalar, StreamId,
    Tensor, TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    generated_storage_dtype_device_01::{
        StorageDTypeDeviceError, cast_jvp_with_context_exact_native,
        cast_vjp_with_context_exact_native, identity_vjp_with_context_exact_native,
        to_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use std::{fmt, sync::Arc};
use thiserror::Error;

pub const ARANGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-583EFBAFDD35";
pub const AS_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-1D3DBB9A17A7";
pub const EMPTY_OPERATION_ID: &str = "COMFY-TENSOR-OP-9FE84462B357";
pub const EYE_OPERATION_ID: &str = "COMFY-TENSOR-OP-CA2E738EA0EF";
pub const FROM_NUMPY_OPERATION_ID: &str = "COMFY-TENSOR-OP-AB3BD8938087";
pub const FULL_OPERATION_ID: &str = "COMFY-TENSOR-OP-80ED45644147";
pub const LINSPACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-00009BB729DF";
pub const ONES_OPERATION_ID: &str = "COMFY-TENSOR-OP-67746161BCEF";
pub const TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-C82C17577549";
pub const ZEROS_OPERATION_ID: &str = "COMFY-TENSOR-OP-4D196D1FC26D";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum TensorCreationPartOneError {
    #[error("tensor-creation operation {operation} was cancelled")]
    Cancelled { operation: &'static str },
    #[error("tensor-creation operation {operation} does not support device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("tensor-creation operation {operation} does not support layout {layout:?}")]
    UnsupportedLayout {
        operation: &'static str,
        layout: Layout,
    },
    #[error("tensor-creation operation {operation} does not support dtype {dtype:?}: {reason}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
        reason: &'static str,
    },
    #[error("tensor-creation operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("tensor-creation operation {operation} failed in canonical tensor storage: {source}")]
    Tensor {
        operation: &'static str,
        source: TensorError,
    },
    #[error("tensor-creation operation {operation} failed in the canonical cast adapter: {source}")]
    Cast {
        operation: &'static str,
        source: StorageDTypeDeviceError,
    },
    #[error("tensor-creation operation {operation} failed in the canonical autograd tape: {source}")]
    Autograd {
        operation: &'static str,
        source: AutogradError,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum NativeTensorInput<'a> {
    Tensor(&'a Tensor),
    Literal {
        values: &'a [Scalar],
        shape: &'a [u64],
    },
    NativeArray(&'a NativeArray),
}

#[derive(Clone)]
pub struct NativeArray {
    descriptor: TensorDescriptor,
    bytes: Arc<[u8]>,
    byte_len: u64,
}

impl fmt::Debug for NativeArray {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeArray")
            .field("descriptor", &self.descriptor)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

impl NativeArray {
    pub fn new(
        bytes: Arc<[u8]>,
        shape: Vec<u64>,
        strides: Vec<i64>,
        offset_elements: u64,
        dtype: DType,
        stream: StreamId,
    ) -> Result<Self, TensorCreationPartOneError> {
        let descriptor = TensorDescriptor::new_strided(
            shape,
            strides,
            offset_elements,
            dtype,
            Layout::Strided,
            DeviceId::CPU,
            stream,
        )
        .map_err(|source| map_tensor_error(FROM_NUMPY_OPERATION_ID, source))?;
        let byte_len = u64::try_from(bytes.len()).map_err(|_| {
            invalid(
                FROM_NUMPY_OPERATION_ID,
                "native-array storage length does not fit u64",
            )
        })?;
        if descriptor
            .storage_span_bytes()
            .map_err(|source| map_tensor_error(FROM_NUMPY_OPERATION_ID, source))?
            .is_some_and(|range| range.end > byte_len)
        {
            return Err(invalid(
                FROM_NUMPY_OPERATION_ID,
                "native-array descriptor exceeds its immutable byte storage",
            ));
        }
        Ok(Self {
            descriptor,
            bytes,
            byte_len,
        })
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn storage_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone)]
struct NativeArrayStorage {
    bytes: Arc<[u8]>,
    byte_len: u64,
}

impl fmt::Debug for NativeArrayStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeArrayStorage")
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for NativeArrayStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn byte_len(&self) -> u64 {
        self.byte_len
    }

    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError> {
        Err(TensorError::ReadOnlyView)
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub fn arange_with_context_exact_native(
    backend: &CpuBackend,
    start: Scalar,
    end: Scalar,
    step: Scalar,
    dtype: Option<DType>,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(ARANGE_OPERATION_ID, context)?;
    require_factory_options(
        ARANGE_OPERATION_ID,
        layout,
        device,
        dtype.unwrap_or_else(|| infer_range_dtype(start, end, step)),
        requires_grad,
        autograd_registration.is_some(),
    )?;
    let dtype = dtype.unwrap_or_else(|| infer_range_dtype(start, end, step));
    require_real_sequence_dtype(ARANGE_OPERATION_ID, dtype)?;
    if matches!(start, Scalar::Float(_))
        || matches!(end, Scalar::Float(_))
        || matches!(step, Scalar::Float(_))
    {
        let start = scalar_to_f64(start);
        let end = scalar_to_f64(end);
        let step = scalar_to_f64(step);
        if !start.is_finite() || !end.is_finite() || !step.is_finite() || step == 0.0 {
            return Err(invalid(
                ARANGE_OPERATION_ID,
                "floating start, end, and nonzero step must be finite",
            ));
        }
        let length = floating_range_length(start, end, step)?;
        let output = upload_generated(
            backend,
            &[u64::try_from(length).map_err(|_| {
                invalid(ARANGE_OPERATION_ID, "range length does not fit a tensor dimension")
            })?],
            dtype,
            ARANGE_OPERATION_ID,
            context,
            |index| Ok(Scalar::Float((index as f64).mul_add(step, start))),
        )?;
        return publish_factory_leaf(
            output,
            ARANGE_OPERATION_ID,
            autograd_registration,
            context,
        );
    }

    let start = scalar_to_i128(start, ARANGE_OPERATION_ID)?;
    let end = scalar_to_i128(end, ARANGE_OPERATION_ID)?;
    let step = scalar_to_i128(step, ARANGE_OPERATION_ID)?;
    if step == 0 {
        return Err(invalid(ARANGE_OPERATION_ID, "step must be nonzero"));
    }
    let length = integer_range_length(start, end, step)?;
    let output = upload_generated(
        backend,
        &[u64::try_from(length).map_err(|_| {
            invalid(ARANGE_OPERATION_ID, "range length does not fit a tensor dimension")
        })?],
        dtype,
        ARANGE_OPERATION_ID,
        context,
        |index| {
            let value = i128::try_from(index)
                .ok()
                .and_then(|index| index.checked_mul(step))
                .and_then(|offset| start.checked_add(offset))
                .ok_or_else(|| invalid(ARANGE_OPERATION_ID, "range value overflowed"))?;
            scalar_from_i128(value, ARANGE_OPERATION_ID)
        },
    )?;
    publish_factory_leaf(
        output,
        ARANGE_OPERATION_ID,
        autograd_registration,
        context,
    )
}

pub fn as_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: NativeTensorInput<'_>,
    dtype: Option<DType>,
    device: Option<DeviceId>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(AS_TENSOR_OPERATION_ID, context)?;
    let tensor = match input {
        NativeTensorInput::Tensor(input) => input.clone(),
        NativeTensorInput::Literal { values, shape } => {
            let dtype = dtype.unwrap_or_else(|| infer_literal_dtype(values));
            require_cpu(AS_TENSOR_OPERATION_ID, device.unwrap_or(DeviceId::CPU))?;
            return upload_literals(
                backend,
                values,
                shape,
                dtype,
                AS_TENSOR_OPERATION_ID,
                context,
            );
        }
        NativeTensorInput::NativeArray(array) => {
            from_numpy_exact_native(array, context.cancellation)
                .map_err(|error| remap_creation_error(AS_TENSOR_OPERATION_ID, error))?
        }
    };
    let dtype = dtype.unwrap_or(tensor.descriptor().dtype());
    let device = device.unwrap_or(tensor.descriptor().device());
    require_cpu(AS_TENSOR_OPERATION_ID, device)?;
    map_cast_result(
        AS_TENSOR_OPERATION_ID,
        to_with_context_exact_native(
            backend,
            &tensor,
            Some(dtype),
            Some(device),
            false,
            false,
            None,
            context,
        ),
    )
}

pub fn empty_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(EMPTY_OPERATION_ID, context)?;
    require_factory_options(
        EMPTY_OPERATION_ID,
        layout,
        device,
        dtype,
        requires_grad,
        autograd_registration.is_some(),
    )?;
    let descriptor = contiguous_descriptor(shape, dtype, device, context, EMPTY_OPERATION_ID)?;
    let output = backend
        .allocate(descriptor, context)
        .map_err(|source| map_tensor_error(EMPTY_OPERATION_ID, source))?
        .0;
    publish_factory_leaf(output, EMPTY_OPERATION_ID, autograd_registration, context)
}

#[allow(clippy::too_many_arguments)]
pub fn eye_with_context_exact_native(
    backend: &CpuBackend,
    rows: u64,
    columns: Option<u64>,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(EYE_OPERATION_ID, context)?;
    require_factory_options(
        EYE_OPERATION_ID,
        layout,
        device,
        dtype,
        requires_grad,
        autograd_registration.is_some(),
    )?;
    let columns = columns.unwrap_or(rows);
    let output = upload_generated(
        backend,
        &[rows, columns],
        dtype,
        EYE_OPERATION_ID,
        context,
        |index| {
            let columns = usize::try_from(columns)
                .map_err(|_| invalid(EYE_OPERATION_ID, "column count does not fit memory"))?;
            let row = index
                .checked_div(columns)
                .ok_or_else(|| invalid(EYE_OPERATION_ID, "zero-column index is unreachable"))?;
            let column = index
                .checked_rem(columns)
                .ok_or_else(|| invalid(EYE_OPERATION_ID, "zero-column index is unreachable"))?;
            Ok(Scalar::Unsigned(u64::from(row == column)))
        },
    )?;
    publish_factory_leaf(output, EYE_OPERATION_ID, autograd_registration, context)
}

pub fn from_numpy_exact_native(
    array: &NativeArray,
    cancellation: &CancellationToken,
) -> Result<Tensor, TensorCreationPartOneError> {
    check(FROM_NUMPY_OPERATION_ID, cancellation)?;
    let storage = NativeArrayStorage {
        bytes: array.bytes.clone(),
        byte_len: array.byte_len,
    };
    let tensor = Tensor::from_backend_storage(
        array.descriptor.clone(),
        Box::new(storage),
        ViewAccess::ReadOnly,
    )
    .map_err(|source| map_tensor_error(FROM_NUMPY_OPERATION_ID, source))?;
    check(FROM_NUMPY_OPERATION_ID, cancellation)?;
    Ok(tensor)
}

pub fn full_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    fill_value: Scalar,
    dtype: Option<DType>,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    let dtype = dtype.unwrap_or_else(|| infer_literal_dtype(std::slice::from_ref(&fill_value)));
    fill_factory(
        backend,
        shape,
        fill_value,
        dtype,
        layout,
        device,
        requires_grad,
        autograd_registration,
        FULL_OPERATION_ID,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn linspace_with_context_exact_native(
    backend: &CpuBackend,
    start: f64,
    end: f64,
    steps: u64,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(LINSPACE_OPERATION_ID, context)?;
    require_factory_options(
        LINSPACE_OPERATION_ID,
        layout,
        device,
        dtype,
        requires_grad,
        autograd_registration.is_some(),
    )?;
    require_real_sequence_dtype(LINSPACE_OPERATION_ID, dtype)?;
    if !start.is_finite() || !end.is_finite() {
        return Err(invalid(
            LINSPACE_OPERATION_ID,
            "start and end must be finite",
        ));
    }
    let steps_usize = usize::try_from(steps)
        .map_err(|_| invalid(LINSPACE_OPERATION_ID, "steps does not fit memory"))?;
    let output = upload_generated(
        backend,
        &[steps],
        dtype,
        LINSPACE_OPERATION_ID,
        context,
        |index| linspace_f64_value(start, end, steps_usize, index).map(Scalar::Float),
    )?;
    publish_factory_leaf(
        output,
        LINSPACE_OPERATION_ID,
        autograd_registration,
        context,
    )
}

pub fn linspace_f64_value(
    start: f64,
    end: f64,
    steps: usize,
    index: usize,
) -> Result<f64, TensorCreationPartOneError> {
    if !start.is_finite() || !end.is_finite() || index >= steps {
        return Err(invalid(
            LINSPACE_OPERATION_ID,
            "finite endpoints and an index within the sequence are required",
        ));
    }
    Ok(match steps {
        1 => start,
        _ if index + 1 == steps => end,
        _ => {
            let weight = index as f64 / (steps - 1) as f64;
            start * (1.0 - weight) + end * weight
        }
    })
}

pub fn linspace_f32_value(
    start: f32,
    end: f32,
    steps: usize,
    index: usize,
) -> Result<f32, TensorCreationPartOneError> {
    if !start.is_finite() || !end.is_finite() || index >= steps {
        return Err(invalid(
            LINSPACE_OPERATION_ID,
            "finite float32 endpoints and an index within the sequence are required",
        ));
    }
    if steps == 1 {
        return Ok(start);
    }
    let step = (end - start) / (steps - 1) as f32;
    // The source evaluates each half from its nearest endpoint; double-precision
    // interpolation changes float32 ramps enough to affect high-rho schedules.
    Ok(if index < steps / 2 {
        start + step * index as f32
    } else {
        end - step * (steps - index - 1) as f32
    })
}

pub fn ones_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    fill_factory(
        backend,
        shape,
        Scalar::Unsigned(1),
        dtype,
        layout,
        device,
        requires_grad,
        autograd_registration,
        ONES_OPERATION_ID,
        context,
    )
}

pub fn tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: NativeTensorInput<'_>,
    dtype: Option<DType>,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(TENSOR_OPERATION_ID, context)?;
    require_cpu(TENSOR_OPERATION_ID, device)?;
    require_autograd_registration(
        TENSOR_OPERATION_ID,
        requires_grad,
        autograd_registration.is_some(),
    )?;
    let output = match input {
        NativeTensorInput::Literal { values, shape } => {
            let dtype = dtype.unwrap_or_else(|| infer_literal_dtype(values));
            require_requires_grad(TENSOR_OPERATION_ID, dtype, requires_grad)?;
            upload_literals(backend, values, shape, dtype, TENSOR_OPERATION_ID, context)
        }
        NativeTensorInput::Tensor(input) => {
            let dtype = dtype.unwrap_or(input.descriptor().dtype());
            require_requires_grad(TENSOR_OPERATION_ID, dtype, requires_grad)?;
            map_cast_result(
                TENSOR_OPERATION_ID,
                to_with_context_exact_native(
                    backend,
                    input,
                    Some(dtype),
                    Some(device),
                    false,
                    true,
                    None,
                    context,
                ),
            )
        }
        NativeTensorInput::NativeArray(array) => {
            let input = from_numpy_exact_native(array, context.cancellation)
                .map_err(|error| remap_creation_error(TENSOR_OPERATION_ID, error))?;
            let dtype = dtype.unwrap_or(input.descriptor().dtype());
            require_requires_grad(TENSOR_OPERATION_ID, dtype, requires_grad)?;
            map_cast_result(
                TENSOR_OPERATION_ID,
                to_with_context_exact_native(
                    backend,
                    &input,
                    Some(dtype),
                    Some(device),
                    false,
                    true,
                    None,
                    context,
                ),
            )
        }
    }?;
    publish_factory_leaf(
        output,
        TENSOR_OPERATION_ID,
        autograd_registration,
        context,
    )
}

pub fn zeros_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    fill_factory(
        backend,
        shape,
        Scalar::Unsigned(0),
        dtype,
        layout,
        device,
        requires_grad,
        autograd_registration,
        ZEROS_OPERATION_ID,
        context,
    )
}

pub fn as_tensor_vjp_with_context_exact_native(
    backend: &CpuBackend,
    primal: &Tensor,
    output_gradient: &Tensor,
    output_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(AS_TENSOR_OPERATION_ID, context)?;
    if output_gradient.descriptor().dtype() != output_dtype
        || output_gradient.descriptor().device() != primal.descriptor().device()
    {
        return Err(invalid(
            AS_TENSOR_OPERATION_ID,
            "output gradient dtype and device must match the as_tensor output",
        ));
    }
    let result = if primal.descriptor().dtype() == output_dtype {
        identity_vjp_with_context_exact_native(backend, primal, output_gradient, context)
    } else {
        cast_vjp_with_context_exact_native(backend, primal, output_gradient, context)
    };
    map_cast_result(AS_TENSOR_OPERATION_ID, result)
}

pub fn as_tensor_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    output_dtype: DType,
    output_device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(AS_TENSOR_OPERATION_ID, context)?;
    if input_tangent.descriptor().dtype() == output_dtype
        && input_tangent.descriptor().device() == output_device
    {
        return Ok(input_tangent.clone());
    }
    map_cast_result(
        AS_TENSOR_OPERATION_ID,
        cast_jvp_with_context_exact_native(
            backend,
            input_tangent,
            output_dtype,
            output_device,
            context,
        ),
    )
}

pub fn linspace_vjp_exact_native(
    output_gradient: &Tensor,
    cancellation: &CancellationToken,
) -> Result<[f64; 2], TensorCreationPartOneError> {
    check(LINSPACE_OPERATION_ID, cancellation)?;
    require_differentiable_real_dtype(
        LINSPACE_OPERATION_ID,
        output_gradient.descriptor().dtype(),
    )?;
    if output_gradient.descriptor().rank() != 1 {
        return Err(invalid(
            LINSPACE_OPERATION_ID,
            "linspace output gradient must have rank one",
        ));
    }
    let steps = usize::try_from(output_gradient.descriptor().element_count().map_err(|source| {
        map_tensor_error(LINSPACE_OPERATION_ID, source)
    })?)
    .map_err(|_| invalid(LINSPACE_OPERATION_ID, "gradient length does not fit memory"))?;
    let mut start_gradient = 0.0;
    let mut end_gradient = 0.0;
    for index in 0..steps {
        check_periodically(LINSPACE_OPERATION_ID, index, cancellation)?;
        let value = decoded_real(
            output_gradient
                .descriptor()
                .dtype()
                .decode_scalar(
                    output_gradient
                        .linear_element_bytes(u64::try_from(index).map_err(|_| {
                            invalid(LINSPACE_OPERATION_ID, "gradient index does not fit u64")
                        })?)
                        .map_err(|source| map_tensor_error(LINSPACE_OPERATION_ID, source))?,
                )
                .map_err(|source| map_tensor_error(LINSPACE_OPERATION_ID, source))?,
            LINSPACE_OPERATION_ID,
        )?;
        let end_weight = match steps {
            0 | 1 => 0.0,
            _ => index as f64 / (steps - 1) as f64,
        };
        start_gradient += value * (1.0 - end_weight);
        end_gradient += value * end_weight;
    }
    check(LINSPACE_OPERATION_ID, cancellation)?;
    Ok([start_gradient, end_gradient])
}

#[allow(clippy::too_many_arguments)]
pub fn linspace_jvp_with_context_exact_native(
    backend: &CpuBackend,
    start_tangent: f64,
    end_tangent: f64,
    steps: u64,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(LINSPACE_OPERATION_ID, context)?;
    require_differentiable_real_dtype(LINSPACE_OPERATION_ID, dtype)?;
    linspace_with_context_exact_native(
        backend,
        start_tangent,
        end_tangent,
        steps,
        dtype,
        layout,
        device,
        false,
        None,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn fill_factory(
    backend: &CpuBackend,
    shape: &[u64],
    value: Scalar,
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    requires_grad: bool,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(operation, context)?;
    require_factory_options(
        operation,
        layout,
        device,
        dtype,
        requires_grad,
        autograd_registration.is_some(),
    )?;
    let descriptor = contiguous_descriptor(shape, dtype, device, context, operation)?;
    let output = backend
        .fill(value, descriptor, context)
        .map_err(|source| map_tensor_error(operation, source))?
        .0;
    publish_factory_leaf(output, operation, autograd_registration, context)
}

fn upload_literals(
    backend: &CpuBackend,
    values: &[Scalar],
    shape: &[u64],
    dtype: DType,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    let expected = element_count(shape, operation)?;
    if values.len() != expected {
        return Err(invalid(
            operation,
            format!(
                "literal value count must equal the shape element count: expected {expected}, got {}",
                values.len()
            ),
        ));
    }
    upload_generated(backend, shape, dtype, operation, context, |index| {
        values
            .get(index)
            .copied()
            .ok_or_else(|| invalid(operation, "literal index exceeded its validated length"))
    })
}

fn upload_generated(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    mut value: impl FnMut(usize) -> Result<Scalar, TensorCreationPartOneError>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(operation, context)?;
    require_cpu(operation, DeviceId::CPU)?;
    let count = element_count(shape, operation)?;
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| invalid(operation, "dtype byte width does not fit memory"))?;
    let capacity = count
        .checked_mul(width)
        .ok_or_else(|| invalid(operation, "output byte count overflowed"))?;
    let mut bytes = backend
        .workspace_vec(context, capacity)
        .map_err(|source| map_tensor_error(operation, source))?;
    for index in 0..count {
        check_periodically(operation, index, context.cancellation)?;
        for byte in dtype
            .encode_scalar(value(index)?, operation, DeviceId::CPU)
            .map_err(|source| map_tensor_error(operation, source))?
        {
            bytes
                .try_push(byte)
                .map_err(|source| map_tensor_error(operation, source))?;
        }
    }
    let descriptor = contiguous_descriptor(shape, dtype, DeviceId::CPU, context, operation)?;
    let output = backend
        .upload_bytes(descriptor, &bytes, context)
        .map_err(|source| map_tensor_error(operation, source))?
        .0;
    check_context(operation, context)?;
    Ok(output)
}

fn contiguous_descriptor(
    shape: &[u64],
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
    operation: &'static str,
) -> Result<TensorDescriptor, TensorCreationPartOneError> {
    TensorDescriptor::contiguous(shape.to_vec(), dtype, device, context.stream)
        .map_err(|source| map_tensor_error(operation, source))
}

fn element_count(
    shape: &[u64],
    operation: &'static str,
) -> Result<usize, TensorCreationPartOneError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension)
    });
    usize::try_from(
        count.ok_or_else(|| invalid(operation, "shape element count overflowed"))?,
    )
    .map_err(|_| invalid(operation, "shape element count does not fit memory"))
}

fn infer_range_dtype(start: Scalar, end: Scalar, step: Scalar) -> DType {
    if [start, end, step]
        .into_iter()
        .any(|value| matches!(value, Scalar::Float(_)))
    {
        DType::F32
    } else {
        DType::I64
    }
}

fn infer_literal_dtype(values: &[Scalar]) -> DType {
    if values.is_empty()
        || values
            .iter()
            .any(|value| matches!(value, Scalar::Float(_)))
    {
        DType::F32
    } else if values
        .iter()
        .any(|value| matches!(value, Scalar::Signed(_) | Scalar::Unsigned(_)))
    {
        DType::I64
    } else {
        DType::Bool
    }
}

fn require_factory_options(
    operation: &'static str,
    layout: Layout,
    device: DeviceId,
    dtype: DType,
    requires_grad: bool,
    has_autograd_registration: bool,
) -> Result<(), TensorCreationPartOneError> {
    require_cpu(operation, device)?;
    if layout != Layout::Strided {
        return Err(TensorCreationPartOneError::UnsupportedLayout { operation, layout });
    }
    require_requires_grad(operation, dtype, requires_grad)?;
    require_autograd_registration(operation, requires_grad, has_autograd_registration)
}

fn require_autograd_registration(
    operation: &'static str,
    requires_grad: bool,
    has_autograd_registration: bool,
) -> Result<(), TensorCreationPartOneError> {
    if requires_grad != has_autograd_registration {
        return Err(invalid(
            operation,
            if requires_grad {
                "requires_grad=true requires a canonical AutogradTape and checked LeafId"
            } else {
                "an autograd leaf registration requires requires_grad=true"
            },
        ));
    }
    Ok(())
}

fn publish_factory_leaf(
    output: Tensor,
    operation: &'static str,
    autograd_registration: Option<(&mut AutogradTape, LeafId)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, TensorCreationPartOneError> {
    check_context(operation, context)?;
    if let Some((tape, leaf)) = autograd_registration {
        tape.set_requires_grad(&output, Some(leaf), true, context.cancellation)
            .map_err(|source| TensorCreationPartOneError::Autograd { operation, source })?;
    }
    Ok(output)
}

fn require_requires_grad(
    operation: &'static str,
    dtype: DType,
    requires_grad: bool,
) -> Result<(), TensorCreationPartOneError> {
    if requires_grad
        && !matches!(
            dtype.class(),
            NumericClass::FloatingPoint | NumericClass::Complex
        )
    {
        return Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype,
            reason: "requires_grad is valid only for floating-point or complex tensors",
        });
    }
    Ok(())
}

fn require_real_sequence_dtype(
    operation: &'static str,
    dtype: DType,
) -> Result<(), TensorCreationPartOneError> {
    if matches!(dtype.class(), NumericClass::Boolean | NumericClass::Complex) {
        return Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype,
            reason: "this real-valued sequence requires an integer or floating-point dtype",
        });
    }
    Ok(())
}

fn require_differentiable_real_dtype(
    operation: &'static str,
    dtype: DType,
) -> Result<(), TensorCreationPartOneError> {
    if dtype.class() != NumericClass::FloatingPoint {
        return Err(TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype,
            reason: "analytical derivatives require a floating-point dtype",
        });
    }
    Ok(())
}

fn require_cpu(
    operation: &'static str,
    device: DeviceId,
) -> Result<(), TensorCreationPartOneError> {
    if device.kind() != DeviceKind::Cpu || device.ordinal() != 0 {
        return Err(TensorCreationPartOneError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn floating_range_length(
    start: f64,
    end: f64,
    step: f64,
) -> Result<usize, TensorCreationPartOneError> {
    if (step > 0.0 && start >= end) || (step < 0.0 && start <= end) {
        return Ok(0);
    }
    let length = ((end - start) / step).ceil();
    if !length.is_finite() || length < 0.0 || length > usize::MAX as f64 {
        return Err(invalid(
            ARANGE_OPERATION_ID,
            "floating range length is not representable",
        ));
    }
    Ok(length as usize)
}

fn integer_range_length(
    start: i128,
    end: i128,
    step: i128,
) -> Result<usize, TensorCreationPartOneError> {
    let distance = if step > 0 {
        if start >= end {
            return Ok(0);
        }
        end.checked_sub(start)
    } else {
        if start <= end {
            return Ok(0);
        }
        start.checked_sub(end)
    }
    .ok_or_else(|| invalid(ARANGE_OPERATION_ID, "integer range distance overflowed"))?;
    let step = step
        .checked_abs()
        .ok_or_else(|| invalid(ARANGE_OPERATION_ID, "integer step magnitude overflowed"))?;
    let length = distance
        .checked_add(step - 1)
        .and_then(|distance| distance.checked_div(step))
        .ok_or_else(|| invalid(ARANGE_OPERATION_ID, "integer range length overflowed"))?;
    usize::try_from(length)
        .map_err(|_| invalid(ARANGE_OPERATION_ID, "integer range length does not fit memory"))
}

fn scalar_to_f64(value: Scalar) -> f64 {
    match value {
        Scalar::Boolean(value) => f64::from(u8::from(value)),
        Scalar::Signed(value) => value as f64,
        Scalar::Unsigned(value) => value as f64,
        Scalar::Float(value) => value,
    }
}

fn scalar_to_i128(
    value: Scalar,
    operation: &'static str,
) -> Result<i128, TensorCreationPartOneError> {
    match value {
        Scalar::Boolean(value) => Ok(i128::from(u8::from(value))),
        Scalar::Signed(value) => Ok(i128::from(value)),
        Scalar::Unsigned(value) => Ok(i128::from(value)),
        Scalar::Float(_) => Err(invalid(
            operation,
            "floating range scalars must use the floating sequence path",
        )),
    }
}

fn scalar_from_i128(
    value: i128,
    operation: &'static str,
) -> Result<Scalar, TensorCreationPartOneError> {
    if let Ok(value) = i64::try_from(value) {
        Ok(Scalar::Signed(value))
    } else if let Ok(value) = u64::try_from(value) {
        Ok(Scalar::Unsigned(value))
    } else {
        Err(invalid(
            operation,
            "integer range value exceeds the canonical scalar domain",
        ))
    }
}

fn decoded_real(
    value: DecodedScalar,
    operation: &'static str,
) -> Result<f64, TensorCreationPartOneError> {
    match value {
        DecodedScalar::Signed(value) => Ok(value as f64),
        DecodedScalar::Unsigned(value) => Ok(value as f64),
        DecodedScalar::Real(value) => Ok(value),
        DecodedScalar::Boolean(_) | DecodedScalar::Complex { .. } => Err(invalid(
            operation,
            "derivative input must be real numeric",
        )),
    }
}

fn check(
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<(), TensorCreationPartOneError> {
    cancellation
        .check()
        .map_err(|_| TensorCreationPartOneError::Cancelled { operation })
}

fn check_context(
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), TensorCreationPartOneError> {
    context
        .check()
        .map_err(|source| map_tensor_error(operation, source))
}

fn check_periodically(
    operation: &'static str,
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), TensorCreationPartOneError> {
    if index.is_multiple_of(1_024) {
        check(operation, cancellation)?;
    }
    Ok(())
}

fn map_tensor_error(
    operation: &'static str,
    source: TensorError,
) -> TensorCreationPartOneError {
    match source {
        TensorError::Cancelled => TensorCreationPartOneError::Cancelled { operation },
        source => TensorCreationPartOneError::Tensor { operation, source },
    }
}

fn map_cast_result(
    operation: &'static str,
    result: Result<Tensor, StorageDTypeDeviceError>,
) -> Result<Tensor, TensorCreationPartOneError> {
    result.map_err(|source| match source {
        StorageDTypeDeviceError::Cancelled { .. } => {
            TensorCreationPartOneError::Cancelled { operation }
        }
        source => TensorCreationPartOneError::Cast { operation, source },
    })
}

fn remap_creation_error(
    operation: &'static str,
    error: TensorCreationPartOneError,
) -> TensorCreationPartOneError {
    match error {
        TensorCreationPartOneError::Cancelled { .. } => {
            TensorCreationPartOneError::Cancelled { operation }
        }
        TensorCreationPartOneError::UnsupportedDevice { device, .. } => {
            TensorCreationPartOneError::UnsupportedDevice { operation, device }
        }
        TensorCreationPartOneError::UnsupportedLayout { layout, .. } => {
            TensorCreationPartOneError::UnsupportedLayout { operation, layout }
        }
        TensorCreationPartOneError::UnsupportedDType { dtype, reason, .. } => {
            TensorCreationPartOneError::UnsupportedDType {
                operation,
                dtype,
                reason,
            }
        }
        TensorCreationPartOneError::Invalid { reason, .. } => {
            TensorCreationPartOneError::Invalid { operation, reason }
        }
        TensorCreationPartOneError::Tensor { source, .. } => {
            TensorCreationPartOneError::Tensor { operation, source }
        }
        TensorCreationPartOneError::Cast { source, .. } => {
            TensorCreationPartOneError::Cast { operation, source }
        }
        TensorCreationPartOneError::Autograd { source, .. } => {
            TensorCreationPartOneError::Autograd { operation, source }
        }
    }
}

fn invalid(
    operation: &'static str,
    reason: impl Into<String>,
) -> TensorCreationPartOneError {
    TensorCreationPartOneError::Invalid {
        operation,
        reason: reason.into(),
    }
}
