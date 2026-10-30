use crate::{
    AutocastPolicy, AutogradError, BackendCapabilityMatrix, CancellationToken, CheckpointRecord,
    CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, GradientMode,
    NativeDeviceProperties, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    ViewAccess,
    generated_sim_operator_indirection_01::OperatorIndirectionError,
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError,
        acos_jvp_with_context_exact_native as canonical_acos_jvp_with_context,
        acos_vjp_with_context_exact_native as canonical_acos_vjp_with_context,
        acos_with_context_exact_native as canonical_acos_with_context,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, ElementwiseRuntimePartThreeError,
        floor_with_context_exact_native as canonical_floor_with_context,
        sigmoid_jvp_with_context_exact_native as canonical_sigmoid_jvp_with_context,
        sigmoid_vjp_with_context_exact_native as canonical_sigmoid_vjp_with_context,
        sigmoid_with_context_exact_native as canonical_sigmoid_with_context,
    },
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError,
        div_jvp_with_context_exact_native as canonical_div_jvp_with_context,
        div_vjp_with_context_exact_native as canonical_div_vjp_with_context,
        div_with_context_exact_native as canonical_div_with_context,
        sqrt_jvp_with_context_exact_native as canonical_sqrt_jvp_with_context,
        sqrt_vjp_with_context_exact_native as canonical_sqrt_vjp_with_context,
        sqrt_with_context_exact_native as canonical_sqrt_with_context,
        zero_in_place_with_context_exact_native as canonical_zero_in_place_with_context,
    },
};
use sim_types::DeviceKind;
use std::cmp::Ordering;
use thiserror::Error;

pub const ACOS_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-4B4746D5885A";
pub const BOOL_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-472F133627A1";
pub const ROUND_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-4B6925D60ACD";
pub const SIGMOID_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-51695C0FE8D8";
pub const UNIQUE_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-4685F95970C6";
pub const DIV_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-4D087B722410";
pub const INFERENCE_MODE_OPERATION_ID: &str = "COMFY-TENSOR-OP-42AC47EA61EE";
pub const JIT_IS_TRACING_OPERATION_ID: &str = "COMFY-TENSOR-OP-49949B24BFD5";
pub const SINC_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-4BE7FEEFD9EF";
pub const SQRT_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-4130D690D4B2";
pub const CHECKPOINT_OPERATION_ID: &str = "COMFY-TENSOR-OP-5278A14360E3";
pub const XPU_GET_DEVICE_PROPERTIES_OPERATION_ID: &str = "COMFY-TENSOR-OP-48C9CD534224";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartSixError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    PartThree(#[from] ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    PartFive(#[from] ElementwiseRuntimePartFiveError),
    #[error("elementwise/runtime part-six operation was cancelled")]
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
    #[error("elementwise/runtime part-six input is invalid: {0}")]
    Invalid(&'static str),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartSixError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn acos_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_acos_with_context(backend, input, context)?)
}

pub fn acos_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_acos_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn acos_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_acos_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn bool_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_cpu(input, BOOL_METHOD_OPERATION_ID)?;
    let count = element_count(input.descriptor().shape())?;
    let mut bytes = backend.workspace_vec::<u8>(context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        bytes.try_push(u8::from(value.is_nonzero()))?;
    }
    upload_bytes_with_context(
        backend,
        input.descriptor().shape(),
        DType::Bool,
        input,
        &bytes,
        context,
    )
}

pub fn sigmoid_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_with_context(backend, input, context)?)
}

pub fn sigmoid_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn sigmoid_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

#[derive(Clone, Debug)]
pub struct UniqueResult {
    pub values: Tensor,
    pub inverse_indices: Option<Tensor>,
    pub counts: Option<Tensor>,
}

pub fn round_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    decimals: i32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_f32_cpu(input, ROUND_METHOD_OPERATION_ID)?;
    if decimals == 0 {
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            input.descriptor().stream(),
        )?;
        return Ok(backend
            .unary(UnaryOperation::Round, input, descriptor, context)?
            .0);
    }
    let factor = 10_f64.powi(decimals);
    if !factor.is_finite() || factor == 0.0 {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "round decimals exceed the native finite scaling range",
        ));
    }
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = f64::from(read_f32(input, &indices)?);
        values.try_push(((value * factor).round_ties_even() / factor) as f32)?;
    }
    upload_f32_with_context(backend, input, &values, context)
}

pub fn round_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_matching_f32(input, output_gradient, ROUND_METHOD_OPERATION_ID)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(0.0)?;
    }
    upload_f32_with_context(backend, input, &values, context)
}

pub fn round_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    round_method_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn unique_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    _sorted: bool,
    return_inverse: bool,
    return_counts: bool,
    context: &ExecutionContext<'_>,
) -> Result<UniqueResult, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    unique_flat_with_context_exact_native(
        backend,
        input,
        return_inverse,
        return_counts,
        UNIQUE_METHOD_OPERATION_ID,
        context,
    )
}

pub(crate) fn unique_flat_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    return_inverse: bool,
    return_counts: bool,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<UniqueResult, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    if input.descriptor().dtype().class() == crate::NumericClass::Complex {
        return Err(ElementwiseRuntimePartSixError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    let count = element_count(input.descriptor().shape())?;
    let mut entries = backend.workspace_vec::<(DecodedScalar, usize)>(context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        entries.try_push((value, linear_index))?;
    }
    entries.sort_by(|left, right| scalar_order(left.0, right.0).then(left.1.cmp(&right.1)));
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("unique bytes"))?;
    let byte_capacity =
        count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartSixError::ShapeOverflow(
                "unique bytes",
            ))?;
    let mut unique_bytes = backend.workspace_vec::<u8>(context, byte_capacity)?;
    let mut inverse = backend.workspace_vec::<i64>(context, count)?;
    for _ in 0..count {
        inverse.try_push(0)?;
    }
    let mut counts = backend.workspace_vec::<i64>(context, count)?;
    let mut previous = None;
    for (sorted_index, (value, original_index)) in entries.iter().copied().enumerate() {
        check_periodically(sorted_index, context.cancellation)?;
        if previous.is_none_or(|previous| !scalar_equal(previous, value)) {
            let original_indices = unravel_index(original_index, input.descriptor().shape())?;
            for byte in input.element_bytes(&original_indices)? {
                unique_bytes.try_push(*byte)?;
            }
            counts.try_push(0)?;
            previous = Some(value);
        }
        let unique_index =
            counts
                .len()
                .checked_sub(1)
                .ok_or(ElementwiseRuntimePartSixError::ShapeOverflow(
                    "unique index",
                ))?;
        inverse[original_index] = i64::try_from(unique_index)
            .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("unique inverse"))?;
        counts[unique_index] = counts[unique_index].checked_add(1).ok_or(
            ElementwiseRuntimePartSixError::ShapeOverflow("unique counts"),
        )?;
    }
    context.cancellation.check()?;
    drop(entries);
    let unique_count = counts.len();
    let values = upload_bytes_with_context(
        backend,
        &[u64::try_from(unique_count)
            .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("unique values"))?],
        input.descriptor().dtype(),
        input,
        &unique_bytes,
        context,
    )?;
    drop(unique_bytes);
    let inverse_indices = if return_inverse {
        Some(upload_i64_with_context(
            backend,
            input.descriptor().shape(),
            input,
            &inverse,
            context,
        )?)
    } else {
        None
    };
    drop(inverse);
    let counts_tensor = if return_counts {
        Some(upload_i64_with_context(
            backend,
            &[u64::try_from(unique_count)
                .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("unique counts"))?],
            input,
            &counts,
            context,
        )?)
    } else {
        None
    };
    Ok(UniqueResult {
        values,
        inverse_indices,
        counts: counts_tensor,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivisionRoundingMode {
    Floor,
    Trunc,
}

pub fn div_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    rounding_mode: Option<DivisionRoundingMode>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    let divided = canonical_div_with_context(backend, input, other, context)?;
    match rounding_mode {
        None => Ok(divided),
        Some(DivisionRoundingMode::Floor) => {
            Ok(canonical_floor_with_context(backend, &divided, context)?)
        }
        Some(DivisionRoundingMode::Trunc) => {
            let count = element_count(divided.descriptor().shape())?;
            let mut values = backend.workspace_vec::<f32>(context, count)?;
            for linear_index in 0..count {
                check_periodically(linear_index, context.cancellation)?;
                let indices = unravel_index(linear_index, divided.descriptor().shape())?;
                values.try_push(read_f32(&divided, &indices)?.trunc())?;
            }
            upload_f32_with_context(backend, &divided, &values, context)
        }
    }
}

pub fn div_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    rounding_mode: Option<DivisionRoundingMode>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<
    crate::generated_elementwise_or_runtime_operation_05::DivideVjp,
    ElementwiseRuntimePartSixError,
> {
    context.cancellation.check()?;
    let mut gradients =
        canonical_div_vjp_with_context(backend, input, other, output_gradient, context)?;
    if rounding_mode.is_some() {
        canonical_zero_in_place_with_context(backend, &mut gradients.input, context)?;
        if let Some(other) = gradients.other.as_mut() {
            canonical_zero_in_place_with_context(backend, other, context)?;
        }
    }
    Ok(gradients)
}

pub fn div_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    rounding_mode: Option<DivisionRoundingMode>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    let mut tangent = canonical_div_jvp_with_context(
        backend,
        input,
        other,
        input_tangent,
        other_tangent,
        context,
    )?;
    if rounding_mode.is_some() {
        canonical_zero_in_place_with_context(backend, &mut tangent, context)?;
    }
    Ok(tangent)
}

pub fn inference_mode_exact_native(
    enabled: bool,
    current_mode: GradientMode,
    cancellation: &CancellationToken,
) -> Result<GradientMode, ElementwiseRuntimePartSixError> {
    cancellation.check()?;
    let mode = if enabled {
        GradientMode::Inference
    } else {
        current_mode
    };
    cancellation.check()?;
    Ok(mode)
}

pub fn jit_is_tracing_exact_native(
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartSixError> {
    cancellation.check()?;
    Ok(false)
}

pub fn sinc_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_f32_cpu(input, SINC_FUNCTION_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(UnaryOperation::Sinc, input, descriptor, context)?
        .0)
}

pub fn sinc_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    require_matching_f32(input, output_gradient, SINC_FUNCTION_OPERATION_ID)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = read_f32(input, &indices)?;
        let gradient = read_f32(output_gradient, &indices)?;
        let derivative = if value == 0.0 {
            0.0
        } else {
            let argument = std::f32::consts::PI * value;
            (argument * argument.cos() - argument.sin()) / (std::f32::consts::PI * value * value)
        };
        values.try_push(gradient * derivative)?;
    }
    upload_f32_with_context(backend, input, &values, context)
}

pub fn sinc_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    sinc_function_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn sqrt_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sqrt_with_context(backend, input, context)?)
}

pub fn sqrt_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sqrt_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn sqrt_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    context.cancellation.check()?;
    Ok(canonical_sqrt_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

#[derive(Debug)]
pub struct CheckpointExecution {
    outputs: Vec<Tensor>,
    record: CheckpointRecord,
    use_reentrant: bool,
    autocast: AutocastPolicy,
}

impl Drop for CheckpointExecution {
    fn drop(&mut self) {
        self.release();
    }
}

impl CheckpointExecution {
    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }

    pub fn use_reentrant(&self) -> bool {
        self.use_reentrant
    }

    pub fn autocast(&self) -> &AutocastPolicy {
        &self.autocast
    }

    pub fn forward_mode(&self) -> GradientMode {
        if self.use_reentrant {
            GradientMode::NoGrad
        } else {
            GradientMode::Enabled
        }
    }

    pub fn recompute_mode(&self) -> GradientMode {
        GradientMode::Enabled
    }

    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.record.needs_input_grad(index)
    }

    pub fn saved_input_count(&self) -> usize {
        self.record.saved_tensor_count()
    }

    pub fn shallow_recompute_inputs_exact_native(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Tensor>, ElementwiseRuntimePartSixError> {
        cancellation.check()?;
        let saved_inputs = self.record.saved_tensors()?;
        let mut shallow = Vec::new();
        shallow
            .try_reserve_exact(saved_inputs.len())
            .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("checkpoint shallow inputs"))?;
        for input in &saved_inputs {
            cancellation.check()?;
            shallow.push(input.view(input.descriptor().clone(), ViewAccess::ReadOnly)?);
        }
        cancellation.check()?;
        Ok(shallow)
    }

    pub fn release(&mut self) {
        self.record.release();
        self.outputs.clear();
    }

    pub fn recompute_exact_native<F>(
        &self,
        cancellation: &CancellationToken,
        function: F,
    ) -> Result<Vec<Tensor>, ElementwiseRuntimePartSixError>
    where
        F: Fn(
            &[Tensor],
            GradientMode,
            &CancellationToken,
        ) -> Result<Vec<Tensor>, ElementwiseRuntimePartSixError>,
    {
        cancellation.check()?;
        let saved_inputs = self.shallow_recompute_inputs_exact_native(cancellation)?;
        let outputs = function(&saved_inputs, self.recompute_mode(), cancellation)?;
        if outputs.len() != self.outputs.len() {
            return Err(ElementwiseRuntimePartSixError::Invalid(
                "checkpoint recomputation changed output arity",
            ));
        }
        cancellation.check()?;
        Ok(outputs)
    }
}

pub fn checkpoint_exact_native<F>(
    inputs: &[Tensor],
    use_reentrant: bool,
    cancellation: &CancellationToken,
    function: F,
) -> Result<CheckpointExecution, ElementwiseRuntimePartSixError>
where
    F: Fn(
        &[Tensor],
        GradientMode,
        &CancellationToken,
    ) -> Result<Vec<Tensor>, ElementwiseRuntimePartSixError>,
{
    cancellation.check()?;
    let forward_mode = if use_reentrant {
        GradientMode::NoGrad
    } else {
        GradientMode::Enabled
    };
    let outputs = function(inputs, forward_mode, cancellation)?;
    if outputs.is_empty() {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "checkpoint function must return at least one tensor",
        ));
    }
    cancellation.check()?;
    let record = CheckpointRecord::capture(inputs, vec![false; inputs.len()])?;
    Ok(CheckpointExecution {
        outputs,
        record,
        use_reentrant,
        autocast: AutocastPolicy::new(false, DType::F32, true)?,
    })
}

pub fn checkpoint_execution_from_outputs_exact_native(
    inputs: &[Tensor],
    outputs: Vec<Tensor>,
    needs_input_grad: Vec<bool>,
    use_reentrant: bool,
    autocast: AutocastPolicy,
    cancellation: &CancellationToken,
) -> Result<CheckpointExecution, ElementwiseRuntimePartSixError> {
    cancellation.check()?;
    if outputs.is_empty() {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "checkpoint function must return at least one tensor",
        ));
    }
    if needs_input_grad.len() != inputs.len() {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "checkpoint needs-input-gradient arity differs from its inputs",
        ));
    }
    let record = CheckpointRecord::capture(inputs, needs_input_grad)?;
    cancellation.check()?;
    Ok(CheckpointExecution {
        outputs,
        record,
        use_reentrant,
        autocast,
    })
}

pub fn xpu_get_device_properties_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeDeviceProperties, ElementwiseRuntimePartSixError> {
    cancellation.check()?;
    if device.kind() != DeviceKind::Xpu || capabilities.device() != device {
        return Err(ElementwiseRuntimePartSixError::UnsupportedDevice {
            operation: XPU_GET_DEVICE_PROPERTIES_OPERATION_ID,
            device,
        });
    }
    let properties = capabilities.device_properties().cloned().ok_or(
        ElementwiseRuntimePartSixError::UnsupportedDevice {
            operation: XPU_GET_DEVICE_PROPERTIES_OPERATION_ID,
            device,
        },
    )?;
    cancellation.check()?;
    Ok(properties)
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    template: &Tensor,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    let descriptor = TensorDescriptor::contiguous(
        template.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        template.descriptor().stream(),
    )?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_i64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    template: &Tensor,
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    let byte_count = values
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(ElementwiseRuntimePartSixError::ShapeOverflow("i64 tensor"))?;
    let mut bytes = backend.workspace_vec::<u8>(context, byte_count)?;
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        for byte in value.to_ne_bytes() {
            bytes.try_push(byte)?;
        }
    }
    upload_bytes_with_context(backend, shape, DType::I64, template, &bytes, context)
}

fn upload_bytes_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    template: &Tensor,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixError> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        dtype,
        DeviceId::CPU,
        template.descriptor().stream(),
    )?;
    let expected = usize::try_from(descriptor.byte_len()?)
        .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("tensor bytes"))?;
    if bytes.len() != expected {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "tensor byte length does not match its descriptor",
        ));
    }
    let (mut tensor, _) = backend.allocate(descriptor, context)?;
    tensor.write()?.bytes_mut()?.copy_from_slice(bytes);
    context.cancellation.check()?;
    Ok(tensor)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSixError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSixError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_matching_f32(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(other, operation)?;
    if input.descriptor().shape() != other.descriptor().shape() {
        return Err(ElementwiseRuntimePartSixError::Invalid(
            "gradient/tangent shape must match the input",
        ));
    }
    if input.descriptor().stream() != other.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: input.descriptor().stream(),
            actual: other.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartSixError> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
    usize::try_from(count.ok_or(ElementwiseRuntimePartSixError::ShapeOverflow(
        "element count",
    ))?)
    .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("element count"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartSixError> {
    let mut indices = vec![0_u64; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartSixError::Invalid(
                "cannot index an empty tensor",
            ));
        }
        *slot = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartSixError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartSixError> {
    let bytes: [u8; 4] = tensor
        .element_bytes(indices)?
        .try_into()
        .map_err(|_| ElementwiseRuntimePartSixError::Invalid("f32 element width"))?;
    Ok(f32::from_ne_bytes(bytes))
}

fn scalar_order(left: DecodedScalar, right: DecodedScalar) -> Ordering {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left.cmp(&right),
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left.cmp(&right),
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left.cmp(&right),
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => real_order(left, right),
        _ => Ordering::Equal,
    }
}

fn real_order(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

fn scalar_equal(left: DecodedScalar, right: DecodedScalar) -> bool {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left == right,
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left == right,
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left == right,
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => left == right,
        _ => false,
    }
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartSixError> {
    if index.is_multiple_of(1_024) {
        cancellation.check()?;
    }
    Ok(())
}
