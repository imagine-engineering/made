use crate::cpu_backend::CpuWorkspaceVec;
use crate::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, DType, DecodedScalar,
    DeviceId, ExecutionContext, NumericClass, Scalar, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
    cpu_backend::{binary_broadcast_shape, broadcast_indices as canonical_broadcast_indices},
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_02::ElementwiseRuntimePartTwoError,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseRuntimePartThreeError, TorchArchiveValue,
    },
    promote_types,
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const TENSOR_CONSTRUCTOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-6A39A98E4F68";
pub const BITWISE_XOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-616CB031A549";
pub const CLAMP_OPERATION_ID: &str = "COMFY-TENSOR-OP-67B5EDD39C41";
pub const FROMBUFFER_OPERATION_ID: &str = "COMFY-TENSOR-OP-6A4C6EDFC695";
pub const FULL_LIKE_OPERATION_ID: &str = "COMFY-TENSOR-OP-6664BEC3F5BD";
pub const LOAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-65EF512E0143";
pub const MUL_OPERATION_ID: &str = "COMFY-TENSOR-OP-615251B481B7";
pub const NPU_CURRENT_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-6311D94BE18A";
pub const ADAMW_CONSTRUCTOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-6238000D28B1";
pub const POW_OPERATION_ID: &str = "COMFY-TENSOR-OP-67F181B603A1";
pub const EXPIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-69E05601EAEA";
pub const NDTRI_OPERATION_ID: &str = "COMFY-TENSOR-OP-6520A75955CD";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartNineError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    PartThree(#[from] ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    Cast(#[from] OperatorIndirectionError),
    #[error("elementwise/runtime part-nine operation was cancelled")]
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
    #[error("elementwise/runtime part-nine input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("native torch archive is invalid: {0}")]
    InvalidArchive(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBitwiseOperation {
    And,
    Or,
    Xor,
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartNineError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn tensor_constructor_with_context_exact_native(
    backend: &CpuBackend,
    values: &[Scalar],
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    if values.len() != element_count(shape)? {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "tensor constructor value count does not match shape".to_owned(),
        ));
    }
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("dtype byte width"))?;
    let byte_count = values.len().checked_mul(byte_width).ok_or(
        ElementwiseRuntimePartNineError::ShapeOverflow("tensor constructor bytes"),
    )?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        temporary_extend(
            &mut bytes,
            &dtype.encode_scalar(*value, TENSOR_CONSTRUCTOR_OPERATION_ID, DeviceId::CPU)?,
        )?;
    }
    upload_bytes_with_context(backend, shape, dtype, stream, &bytes, context)
}

pub fn bitwise_xor_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    bitwise_binary_with_context_exact_native(
        backend,
        left,
        right,
        NativeBitwiseOperation::Xor,
        BITWISE_XOR_OPERATION_ID,
        context,
    )
}

pub fn bitwise_binary_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    bitwise_operation: NativeBitwiseOperation,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    require_cpu(left, operation)?;
    require_cpu(right, operation)?;
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "bitwise operands use different streams".to_owned(),
        ));
    }
    let dtype = promote_types(left.descriptor().dtype(), right.descriptor().dtype())?;
    if !matches!(
        dtype.class(),
        NumericClass::Boolean | NumericClass::SignedInteger | NumericClass::UnsignedInteger
    ) {
        return Err(ElementwiseRuntimePartNineError::UnsupportedDType { operation, dtype });
    }
    let left = cast_to_with_context_exact_native(
        backend,
        left,
        dtype,
        DeviceId::CPU,
        false,
        false,
        context,
    )?;
    let right = cast_to_with_context_exact_native(
        backend,
        right,
        dtype,
        DeviceId::CPU,
        false,
        false,
        context,
    )?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("bitwise dtype width"))?;
    let byte_count = element_count(&shape)?.checked_mul(byte_width).ok_or(
        ElementwiseRuntimePartNineError::ShapeOverflow("bitwise bytes"),
    )?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let left_indices = canonical_broadcast_indices(&indices, left.descriptor().shape())?;
        let right_indices = canonical_broadcast_indices(&indices, right.descriptor().shape())?;
        let left_value = dtype.decode_scalar(left.element_bytes(&left_indices)?)?;
        let right_value = dtype.decode_scalar(right.element_bytes(&right_indices)?)?;
        let value = match (left_value, right_value) {
            (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => {
                Scalar::Boolean(match bitwise_operation {
                    NativeBitwiseOperation::And => left & right,
                    NativeBitwiseOperation::Or => left | right,
                    NativeBitwiseOperation::Xor => left ^ right,
                })
            }
            (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => {
                Scalar::Signed(match bitwise_operation {
                    NativeBitwiseOperation::And => left & right,
                    NativeBitwiseOperation::Or => left | right,
                    NativeBitwiseOperation::Xor => left ^ right,
                })
            }
            (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => {
                Scalar::Unsigned(match bitwise_operation {
                    NativeBitwiseOperation::And => left & right,
                    NativeBitwiseOperation::Or => left | right,
                    NativeBitwiseOperation::Xor => left ^ right,
                })
            }
            _ => {
                return Err(ElementwiseRuntimePartNineError::Invalid(
                    "bitwise operands decoded to incompatible scalars".to_owned(),
                ));
            }
        };
        temporary_extend(
            &mut bytes,
            &dtype.encode_scalar(value, operation, DeviceId::CPU)?,
        )?;
    }
    upload_bytes_with_context(
        backend,
        &shape,
        dtype,
        left.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn clamp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<Scalar>,
    maximum: Option<Scalar>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    let mut output = input.clone();
    crate::generated_elementwise_or_runtime_operation_03::clamp_in_place_with_context_exact_native(
        backend,
        &mut output,
        minimum,
        maximum,
        context,
    )?;
    Ok(output)
}

pub fn clamp_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    if minimum.is_none() && maximum.is_none() {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "clamp requires a minimum, a maximum, or both".to_owned(),
        ));
    }
    binary_same_shape_f32_with_context(
        backend,
        input,
        output_gradient,
        CLAMP_OPERATION_ID,
        |value, gradient| {
            let above_minimum = minimum.is_none_or(|minimum| value >= minimum);
            let below_maximum = maximum.is_none_or(|maximum| value <= maximum);
            if above_minimum && below_maximum {
                gradient
            } else {
                0.0
            }
        },
        context,
    )
}

pub fn clamp_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    clamp_vjp_with_context_exact_native(backend, input, minimum, maximum, input_tangent, context)
}

pub fn frombuffer_with_context_exact_native(
    backend: &CpuBackend,
    source: &[u8],
    dtype: DType,
    count: Option<usize>,
    offset: usize,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    let remaining = source.get(offset..).ok_or_else(|| {
        ElementwiseRuntimePartNineError::Invalid(
            "frombuffer offset exceeds the source length".to_owned(),
        )
    })?;
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("frombuffer dtype width"))?;
    if !offset.is_multiple_of(width) {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "frombuffer offset must be a multiple of the dtype width".to_owned(),
        ));
    }
    let element_count = match count {
        Some(count) => count,
        None => {
            if !remaining.len().is_multiple_of(width) {
                return Err(ElementwiseRuntimePartNineError::Invalid(
                    "frombuffer remaining bytes are not divisible by dtype width".to_owned(),
                ));
            }
            remaining.len() / width
        }
    };
    let byte_count =
        element_count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartNineError::ShapeOverflow(
                "frombuffer byte count",
            ))?;
    let bytes = remaining.get(..byte_count).ok_or_else(|| {
        ElementwiseRuntimePartNineError::Invalid(
            "frombuffer count exceeds the source length".to_owned(),
        )
    })?;
    upload_bytes_with_context(
        backend,
        &[u64::try_from(element_count)
            .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("frombuffer shape"))?],
        dtype,
        stream,
        bytes,
        context,
    )
}

pub fn full_like_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    value: Scalar,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    require_cpu(input, FULL_LIKE_OPERATION_ID)?;
    let dtype = dtype.unwrap_or(input.descriptor().dtype());
    let scalar = dtype.encode_scalar(value, FULL_LIKE_OPERATION_ID, DeviceId::CPU)?;
    let count = element_count(input.descriptor().shape())?;
    let byte_count =
        count
            .checked_mul(scalar.len())
            .ok_or(ElementwiseRuntimePartNineError::ShapeOverflow(
                "full_like bytes",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        temporary_extend(&mut bytes, &scalar)?;
    }
    upload_bytes_with_context(
        backend,
        input.descriptor().shape(),
        dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TorchArchiveLoadError {
    #[error("torch archive loading was cancelled")]
    Cancelled,
    #[error("torch archive was rejected: {reason}")]
    Rejected { reason: String },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

pub trait TorchArchiveLoader {
    fn load_weights_cpu(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<TorchArchiveValue, TorchArchiveLoadError>;
}

pub fn torch_load_with_context_exact_native(
    loader: &dyn TorchArchiveLoader,
    backend: &CpuBackend,
    map_location: DeviceId,
    weights_only: bool,
    context: &ExecutionContext<'_>,
) -> Result<TorchArchiveValue, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    if !weights_only {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "native torch.load requires weights_only=true".to_owned(),
        ));
    }
    if map_location != DeviceId::CPU {
        return Err(ElementwiseRuntimePartNineError::UnsupportedDevice {
            operation: LOAD_OPERATION_ID,
            device: map_location,
        });
    }
    loader
        .load_weights_cpu(backend, context)
        .map_err(|error| match error {
            TorchArchiveLoadError::Cancelled => ElementwiseRuntimePartNineError::Cancelled,
            TorchArchiveLoadError::Rejected { reason } => {
                ElementwiseRuntimePartNineError::InvalidArchive(reason)
            }
            TorchArchiveLoadError::Tensor(error) => error.into(),
        })
}

pub fn mul_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_forward_with_context(
        backend,
        BinaryOperation::Multiply,
        left,
        right,
        MUL_OPERATION_ID,
        context,
    )
}

#[derive(Debug)]
pub struct BinaryGradients {
    pub left: Tensor,
    pub right: Tensor,
}

pub fn mul_vjp_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<BinaryGradients, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_vjp_with_context(
        backend,
        left,
        right,
        output_gradient,
        MUL_OPERATION_ID,
        |left, right, gradient| (gradient * right, gradient * left),
        context,
    )
}

pub fn mul_jvp_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    left_tangent: &Tensor,
    right_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_jvp_with_context(
        backend,
        left,
        right,
        left_tangent,
        right_tangent,
        MUL_OPERATION_ID,
        |left, right, left_tangent, right_tangent| left_tangent * right + left * right_tangent,
        context,
    )
}

pub fn npu_current_device_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<u32, ElementwiseRuntimePartNineError> {
    cancellation.check()?;
    let device = capabilities.device();
    if device.kind() != DeviceKind::Npu {
        return Err(ElementwiseRuntimePartNineError::UnsupportedDevice {
            operation: NPU_CURRENT_DEVICE_OPERATION_ID,
            device,
        });
    }
    Ok(device.ordinal())
}

#[derive(Clone, Debug)]
pub struct NativeAdamW {
    beta1: f32,
    beta2: f32,
    learning_rate: f32,
    weight_decay: f32,
    epsilon: f32,
    amsgrad: bool,
    maximize: bool,
    steps: Vec<u64>,
    exponential_averages: Vec<Tensor>,
    exponential_average_squares: Vec<Tensor>,
    maximum_exponential_average_squares: Vec<Tensor>,
}

impl NativeAdamW {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context_exact_native(
        backend: &CpuBackend,
        parameters: &[Tensor],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        amsgrad: bool,
        maximize: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ElementwiseRuntimePartNineError> {
        context.cancellation.check()?;
        if parameters.is_empty()
            || !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
            || !learning_rate.is_finite()
            || learning_rate < 0.0
            || !epsilon.is_finite()
            || epsilon < 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(ElementwiseRuntimePartNineError::Invalid(
                "invalid native AdamW configuration".to_owned(),
            ));
        }
        let mut averages = Vec::new();
        let mut squares = Vec::new();
        let mut maximums = Vec::new();
        averages
            .try_reserve_exact(parameters.len())
            .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("AdamW state"))?;
        squares
            .try_reserve_exact(parameters.len())
            .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("AdamW state"))?;
        if amsgrad {
            maximums
                .try_reserve_exact(parameters.len())
                .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("AdamW state"))?;
        }
        for (index, parameter) in parameters.iter().enumerate() {
            check_periodically(index, context.cancellation)?;
            require_f32_cpu(parameter, ADAMW_CONSTRUCTOR_OPERATION_ID)?;
            averages.push(zero_like_with_context(backend, parameter, context)?);
            squares.push(zero_like_with_context(backend, parameter, context)?);
            if amsgrad {
                maximums.push(zero_like_with_context(backend, parameter, context)?);
            }
        }
        Ok(Self {
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            epsilon,
            amsgrad,
            maximize,
            steps: vec![0; parameters.len()],
            exponential_averages: averages,
            exponential_average_squares: squares,
            maximum_exponential_average_squares: maximums,
        })
    }

    pub fn step_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<(), ElementwiseRuntimePartNineError> {
        context.cancellation.check()?;
        if parameters.len() != self.steps.len() || gradients.len() != parameters.len() {
            return Err(ElementwiseRuntimePartNineError::Invalid(
                "AdamW parameter, gradient, and state lengths must match".to_owned(),
            ));
        }
        let mut next_steps = backend.workspace_vec(context, self.steps.len())?;
        for step in &self.steps {
            next_steps.try_push(
                step.checked_add(1)
                    .ok_or(ElementwiseRuntimePartNineError::ShapeOverflow("AdamW step"))?,
            )?;
        }
        crate::generated_elementwise_or_runtime_operation_02::adamw_with_context_exact_native(
            backend,
            parameters,
            gradients,
            &mut self.exponential_averages,
            &mut self.exponential_average_squares,
            &mut self.maximum_exponential_average_squares,
            &next_steps,
            self.amsgrad,
            self.beta1,
            self.beta2,
            self.learning_rate,
            self.weight_decay,
            self.epsilon,
            self.maximize,
            context,
        )?;
        self.steps = next_steps.iter().copied().collect();
        Ok(())
    }

    pub fn steps(&self) -> &[u64] {
        &self.steps
    }
}

pub fn pow_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_forward_with_context(
        backend,
        BinaryOperation::Power,
        base,
        exponent,
        POW_OPERATION_ID,
        context,
    )
}

pub fn pow_vjp_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<BinaryGradients, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_vjp_with_context(
        backend,
        base,
        exponent,
        output_gradient,
        POW_OPERATION_ID,
        |base, exponent, gradient| {
            let output = base.powf(exponent);
            (
                gradient * exponent * base.powf(exponent - 1.0),
                gradient * output * base.ln(),
            )
        },
        context,
    )
}

pub fn pow_jvp_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    base_tangent: &Tensor,
    exponent_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_jvp_with_context(
        backend,
        base,
        exponent,
        base_tangent,
        exponent_tangent,
        POW_OPERATION_ID,
        |base, exponent, base_tangent, exponent_tangent| {
            let output = base.powf(exponent);
            base_tangent * exponent * base.powf(exponent - 1.0)
                + exponent_tangent * output * base.ln()
        },
        context,
    )
}
pub fn expit_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_03::sigmoid_with_context_exact_native(
            backend, input, context,
        )?,
    )
}

pub fn expit_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    Ok(crate::generated_elementwise_or_runtime_operation_03::sigmoid_vjp_with_context_exact_native(
        backend, input, gradient, context,
    )?)
}

pub fn expit_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    Ok(crate::generated_elementwise_or_runtime_operation_03::sigmoid_jvp_with_context_exact_native(
        backend, input, tangent, context,
    )?)
}

pub fn ndtri_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    unary_f32_with_context(
        backend,
        input,
        NDTRI_OPERATION_ID,
        inverse_normal_cdf,
        context,
    )
}

pub fn ndtri_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    binary_same_shape_f32_with_context(
        backend,
        input,
        gradient,
        NDTRI_OPERATION_ID,
        |probability, gradient| {
            let quantile = inverse_normal_cdf(probability);
            gradient * (std::f32::consts::TAU).sqrt() * (0.5 * quantile * quantile).exp()
        },
        context,
    )
}

pub fn ndtri_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    context.cancellation.check()?;
    ndtri_vjp_with_context_exact_native(backend, input, tangent, context)
}

fn binary_forward_with_context(
    backend: &CpuBackend,
    operation: BinaryOperation,
    left: &Tensor,
    right: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    require_compatible(left, right, operation_id)?;
    require_f32_cpu(left, operation_id)?;
    require_f32_cpu(right, operation_id)?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, left.descriptor().stream())?;
    Ok(backend
        .binary(operation, left, right, descriptor, context)?
        .0)
}

fn binary_vjp_with_context<F>(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    gradient: &Tensor,
    operation: &'static str,
    derivative: F,
    context: &ExecutionContext<'_>,
) -> Result<BinaryGradients, ElementwiseRuntimePartNineError>
where
    F: Fn(f32, f32, f32) -> (f32, f32),
{
    require_compatible(left, right, operation)?;
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    require_f32_cpu(gradient, operation)?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    if gradient.descriptor().shape() != shape
        || gradient.descriptor().stream() != left.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "binary output gradient descriptor mismatch".to_owned(),
        ));
    }
    let mut left_values = workspace_filled(
        backend,
        context,
        element_count(left.descriptor().shape())?,
        0.0_f32,
    )?;
    let mut right_values = workspace_filled(
        backend,
        context,
        element_count(right.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let left_indices = canonical_broadcast_indices(&indices, left.descriptor().shape())?;
        let right_indices = canonical_broadcast_indices(&indices, right.descriptor().shape())?;
        let (left_gradient, right_gradient) = derivative(
            read_f32(left, &left_indices)?,
            read_f32(right, &right_indices)?,
            read_f32(gradient, &indices)?,
        );
        left_values[linear_index(&left_indices, left.descriptor().shape())?] += left_gradient;
        right_values[linear_index(&right_indices, right.descriptor().shape())?] += right_gradient;
    }
    Ok(BinaryGradients {
        left: upload_f32_with_context(
            backend,
            left.descriptor().shape(),
            left.descriptor().stream(),
            &left_values,
            context,
        )?,
        right: upload_f32_with_context(
            backend,
            right.descriptor().shape(),
            right.descriptor().stream(),
            &right_values,
            context,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
fn binary_jvp_with_context<F>(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    left_tangent: &Tensor,
    right_tangent: &Tensor,
    operation: &'static str,
    derivative: F,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError>
where
    F: Fn(f32, f32, f32, f32) -> f32,
{
    require_compatible(left, right, operation)?;
    require_compatible(left, left_tangent, operation)?;
    require_compatible(right, right_tangent, operation)?;
    if left.descriptor().shape() != left_tangent.descriptor().shape()
        || right.descriptor().shape() != right_tangent.descriptor().shape()
    {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "binary tangent shape mismatch".to_owned(),
        ));
    }
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, element_count(&shape)?)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let left_indices = canonical_broadcast_indices(&indices, left.descriptor().shape())?;
        let right_indices = canonical_broadcast_indices(&indices, right.descriptor().shape())?;
        values.try_push(derivative(
            read_f32(left, &left_indices)?,
            read_f32(right, &right_indices)?,
            read_f32(left_tangent, &left_indices)?,
            read_f32(right_tangent, &right_indices)?,
        ))?;
    }
    upload_f32_with_context(
        backend,
        &shape,
        left.descriptor().stream(),
        &values,
        context,
    )
}

fn unary_f32_with_context<F>(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    map: F,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError>
where
    F: Fn(f32) -> f32,
{
    require_f32_cpu(input, operation)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        values.try_push(map(read_f32(
            input,
            &unravel_index(linear, input.descriptor().shape())?,
        )?))?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn binary_same_shape_f32_with_context<F>(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
    map: F,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError>
where
    F: Fn(f32, f32) -> f32,
{
    require_compatible(left, right, operation)?;
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "tensor shapes must match".to_owned(),
        ));
    }
    let count = element_count(left.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, left.descriptor().shape())?;
        values.try_push(map(read_f32(left, &indices)?, read_f32(right, &indices)?))?;
    }
    upload_f32_with_context(
        backend,
        left.descriptor().shape(),
        left.descriptor().stream(),
        &values,
        context,
    )
}

fn inverse_normal_cdf(probability: f32) -> f32 {
    inverse_normal_cdf_f64(f64::from(probability)) as f32
}

pub fn inverse_normal_cdf_f64(probability: f64) -> f64 {
    if probability == 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&probability) || probability.is_nan() {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if probability < 0.02425 {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability > 0.97575 {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

fn zero_like_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.fill(Scalar::Float(0.0), descriptor, context)?.0)
}
fn upload_bytes_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn temporary_extend(
    values: &mut CpuWorkspaceVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimePartNineError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}
fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartNineError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartNineError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}
fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartNineError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartNineError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}
fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartNineError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartNineError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}
fn require_compatible(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartNineError> {
    require_cpu(left, operation)?;
    require_cpu(right, operation)?;
    if left.descriptor().dtype() != right.descriptor().dtype() {
        return Err(ElementwiseRuntimePartNineError::Invalid(
            "tensor dtypes must match".to_owned(),
        ));
    }
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: left.descriptor().stream(),
            actual: right.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}
fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartNineError> {
    usize::try_from(
        shape
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .ok_or(ElementwiseRuntimePartNineError::ShapeOverflow(
                "element count",
            ))?,
    )
    .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("element count"))
}
fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartNineError> {
    let mut indices = vec![0; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartNineError::Invalid(
                "cannot index an empty tensor".to_owned(),
            ));
        }
        *slot = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}
fn linear_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartNineError> {
    usize::try_from(
        indices
            .iter()
            .zip(shape)
            .try_fold(0_u64, |linear, (index, dimension)| {
                linear
                    .checked_mul(*dimension)
                    .and_then(|value| value.checked_add(*index))
            })
            .ok_or(ElementwiseRuntimePartNineError::ShapeOverflow(
                "linear index",
            ))?,
    )
    .map_err(|_| ElementwiseRuntimePartNineError::ShapeOverflow("linear index"))
}
fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartNineError> {
    Ok(f32::from_ne_bytes(array(tensor.element_bytes(indices)?)?))
}
fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], ElementwiseRuntimePartNineError> {
    bytes.try_into().map_err(|_| {
        ElementwiseRuntimePartNineError::Invalid(
            "fixed-width tensor element has an invalid byte length".to_owned(),
        )
    })
}
fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartNineError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
