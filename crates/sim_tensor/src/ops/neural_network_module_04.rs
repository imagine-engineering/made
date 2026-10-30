use crate::{
    CpuBackend, DeviceId, ExecutionContext, Tensor,
    generated_activation_normalization_functional_01::{
        FunctionalError, elu_jvp_with_context_exact_native as canonical_elu_jvp,
        elu_vjp_with_context_exact_native as canonical_elu_vjp,
        elu_with_context_exact_native as canonical_elu,
    },
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError,
        sigmoid_with_context_exact_native as canonical_sigmoid,
    },
    generated_neural_network_module_01::{
        AveragePoolGeometry, LossReduction, NeuralNetworkModuleError,
    },
    rng::{RngError, RngTransaction},
};
use thiserror::Error;

pub const AVG_POOL_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-D60003AC2B14";
pub const DROPOUT_OPERATION_ID: &str = "COMFY-TENSOR-OP-EE80ED2D81B0";
pub const ELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-F88B1E793668";
pub const IDENTITY_OPERATION_ID: &str = "COMFY-TENSOR-OP-D8ED4EE63E27";
pub const MSE_LOSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-FBA7DB6F71EB";
pub const MODULE_OPERATION_ID: &str = "COMFY-TENSOR-OP-EAF9D9989484";
pub const SIGMOID_OPERATION_ID: &str = "COMFY-TENSOR-OP-E1CEBEBBCBFE";

#[derive(Debug, Error, PartialEq)]
pub enum NeuralNetworkModulePartFourError {
    #[error("neural-network module part-four operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Activation(FunctionalError),
    #[error(transparent)]
    Module(NeuralNetworkModuleError),
    #[error("canonical sigmoid operation failed: {0}")]
    Sigmoid(String),
    #[error(transparent)]
    Rng(RngError),
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("allocation failed while preparing {0}")]
    AllocationFailed(&'static str),
}

impl From<sim_types::CancellationError> for NeuralNetworkModulePartFourError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<FunctionalError> for NeuralNetworkModulePartFourError {
    fn from(error: FunctionalError) -> Self {
        match error {
            FunctionalError::Cancelled => Self::Cancelled,
            error => Self::Activation(error),
        }
    }
}

impl From<NeuralNetworkModuleError> for NeuralNetworkModulePartFourError {
    fn from(error: NeuralNetworkModuleError) -> Self {
        match error {
            NeuralNetworkModuleError::Cancelled => Self::Cancelled,
            error => Self::Module(error),
        }
    }
}

impl From<NeuralNetworkFunctionalError> for NeuralNetworkModulePartFourError {
    fn from(error: NeuralNetworkFunctionalError) -> Self {
        match error {
            NeuralNetworkFunctionalError::Cancelled => Self::Cancelled,
            error => Self::Sigmoid(error.to_string()),
        }
    }
}

impl From<RngError> for NeuralNetworkModulePartFourError {
    fn from(error: RngError) -> Self {
        match error {
            RngError::Cancelled => Self::Cancelled,
            error => Self::Rng(error),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AveragePool1dVjp {
    pub input: Vec<f32>,
}

pub fn average_pool_1d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: usize,
    stride: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<crate::generated_sim_operator_indirection_01::TensorValues, NeuralNetworkModulePartFourError>
{
    context.cancellation.check()?;
    require_cpu(device, AVG_POOL_1D_OPERATION_ID)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &[kernel_size],
        &[stride],
        AVG_POOL_1D_OPERATION_ID,
    )?;
    let mut output = zeroed(geometry.output_count()?, "average-pool-1d output")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        let input_value = input.get(input_index).copied().ok_or(
            NeuralNetworkModuleError::ShapeOverflow("average-pool-1d input index"),
        )?;
        let output_value = output.get_mut(output_index).ok_or(
            NeuralNetworkModuleError::ShapeOverflow("average-pool-1d output index"),
        )?;
        *output_value = input_value.mul_add(scale, *output_value);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(crate::generated_sim_operator_indirection_01::TensorValues {
        values: output,
        shape: geometry.output_shape().to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_1d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: usize,
    stride: usize,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool1dVjp, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, AVG_POOL_1D_OPERATION_ID)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &[kernel_size],
        &[stride],
        AVG_POOL_1D_OPERATION_ID,
    )?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        AVG_POOL_1D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = zeroed(input.len(), "average-pool-1d input gradient")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        let output_gradient = output_gradient.get(output_index).copied().ok_or(
            NeuralNetworkModuleError::ShapeOverflow("average-pool-1d output gradient index"),
        )?;
        let input_gradient = input_gradient.get_mut(input_index).ok_or(
            NeuralNetworkModuleError::ShapeOverflow("average-pool-1d input gradient index"),
        )?;
        *input_gradient = output_gradient.mul_add(scale, *input_gradient);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(AveragePool1dVjp {
        input: input_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_1d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: usize,
    stride: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<crate::generated_sim_operator_indirection_01::TensorValues, NeuralNetworkModulePartFourError>
{
    average_pool_1d_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        kernel_size,
        stride,
        device,
        context,
    )
}

pub struct DropoutForward {
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
    pub transaction: RngTransaction,
}

pub fn dropout_with_context_exact_native(
    input: &[f32],
    probability: f32,
    training: bool,
    mut transaction: RngTransaction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<DropoutForward, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, DROPOUT_OPERATION_ID)?;
    transaction.require_device(device)?;
    validate_dropout_probability(probability)?;
    let mut output = zeroed(input.len(), "dropout output")?;
    let mut mask = bools(input.len(), "dropout mask")?;
    if !training || probability == 0.0 {
        output.copy_from_slice(input);
        mask.fill(true);
    } else if probability == 1.0 {
        mask.fill(false);
    } else {
        let inverse_keep_probability = (1.0 - probability).recip();
        for (index, input_value) in input.iter().copied().enumerate() {
            check_periodically(index, context)?;
            let random = transaction.next_u32(context.cancellation)?;
            let uniform = f64::from(random) / 4_294_967_296.0;
            let keep = uniform >= f64::from(probability);
            let output_value = output.get_mut(index).ok_or(
                NeuralNetworkModulePartFourError::ShapeOverflow("dropout output index"),
            )?;
            let mask_value = mask.get_mut(index).ok_or(
                NeuralNetworkModulePartFourError::ShapeOverflow("dropout mask index"),
            )?;
            *mask_value = keep;
            *output_value = if keep {
                input_value * inverse_keep_probability
            } else {
                0.0
            };
        }
    }
    context.cancellation.check()?;
    Ok(DropoutForward {
        values: output,
        mask,
        transaction,
    })
}

pub fn dropout_vjp_with_context_exact_native(
    output_gradient: &[f32],
    mask: &[bool],
    probability: f32,
    training: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, DROPOUT_OPERATION_ID)?;
    validate_dropout_probability(probability)?;
    if output_gradient.len() != mask.len() {
        return invalid(DROPOUT_OPERATION_ID, "dropout mask length must match the gradient");
    }
    let scale = if training && probability < 1.0 {
        (1.0 - probability).recip()
    } else {
        1.0
    };
    let mut gradient = zeroed(output_gradient.len(), "dropout gradient")?;
    for (index, ((gradient, output_gradient), keep)) in gradient
        .iter_mut()
        .zip(output_gradient)
        .zip(mask)
        .enumerate()
    {
        check_periodically(index, context)?;
        *gradient = if !training {
            *output_gradient
        } else if *keep && probability < 1.0 {
            *output_gradient * scale
        } else {
            0.0
        };
    }
    context.cancellation.check()?;
    Ok(gradient)
}

pub fn dropout_jvp_with_context_exact_native(
    input_tangent: &[f32],
    mask: &[bool],
    probability: f32,
    training: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    dropout_vjp_with_context_exact_native(
        input_tangent,
        mask,
        probability,
        training,
        device,
        context,
    )
}

pub fn elu_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(canonical_elu(backend, input, alpha, device, context)?)
}

pub fn elu_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(canonical_elu_vjp(
        backend,
        input,
        output_gradient,
        alpha,
        device,
        context,
    )?)
}

pub fn elu_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(canonical_elu_jvp(
        backend,
        input,
        input_tangent,
        alpha,
        device,
        context,
    )?)
}

pub fn identity_with_context_exact_native(
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(input.clone())
}

#[derive(Clone, Debug, PartialEq)]
pub struct MseLossVjp {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

pub fn mse_loss_with_context_exact_native(
    input: &[f32],
    target: &[f32],
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, MSE_LOSS_OPERATION_ID)?;
    require_length(
        target.len(),
        input.len(),
        MSE_LOSS_OPERATION_ID,
        "target",
    )?;
    let mut losses = zeroed(input.len(), "MSE loss")?;
    for (index, ((loss, input), target)) in
        losses.iter_mut().zip(input).zip(target).enumerate()
    {
        check_periodically(index, context)?;
        let difference = *input - *target;
        *loss = difference * difference;
    }
    reduce_loss(losses, reduction, context)
}

#[allow(clippy::too_many_arguments)]
pub fn mse_loss_vjp_with_context_exact_native(
    input: &[f32],
    target: &[f32],
    reduction: LossReduction,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<MseLossVjp, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, MSE_LOSS_OPERATION_ID)?;
    require_length(
        target.len(),
        input.len(),
        MSE_LOSS_OPERATION_ID,
        "target",
    )?;
    let expected_gradient = if reduction == LossReduction::None {
        input.len()
    } else {
        1
    };
    require_length(
        output_gradient.len(),
        expected_gradient,
        MSE_LOSS_OPERATION_ID,
        "output gradient",
    )?;
    let mean_scale = if reduction == LossReduction::Mean {
        (input.len() as f32).recip()
    } else {
        1.0
    };
    let mut input_gradient = zeroed(input.len(), "MSE input gradient")?;
    let mut target_gradient = zeroed(input.len(), "MSE target gradient")?;
    for index in 0..input.len() {
        check_periodically(index, context)?;
        let input_value = input.get(index).copied().ok_or(
            NeuralNetworkModulePartFourError::ShapeOverflow("MSE input index"),
        )?;
        let target_value = target.get(index).copied().ok_or(
            NeuralNetworkModulePartFourError::ShapeOverflow("MSE target index"),
        )?;
        let upstream_index = if reduction == LossReduction::None {
            index
        } else {
            0
        };
        let upstream = output_gradient.get(upstream_index).copied().ok_or(
            NeuralNetworkModulePartFourError::ShapeOverflow("MSE output gradient index"),
        )?;
        let gradient = 2.0 * (input_value - target_value) * upstream * mean_scale;
        *input_gradient.get_mut(index).ok_or(
            NeuralNetworkModulePartFourError::ShapeOverflow("MSE input gradient index"),
        )? = gradient;
        *target_gradient.get_mut(index).ok_or(
            NeuralNetworkModulePartFourError::ShapeOverflow("MSE target gradient index"),
        )? = -gradient;
    }
    context.cancellation.check()?;
    Ok(MseLossVjp {
        input: input_gradient,
        target: target_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn mse_loss_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    target: &[f32],
    target_tangent: &[f32],
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    require_cpu(device, MSE_LOSS_OPERATION_ID)?;
    require_length(
        target.len(),
        input.len(),
        MSE_LOSS_OPERATION_ID,
        "target",
    )?;
    require_length(
        input_tangent.len(),
        input.len(),
        MSE_LOSS_OPERATION_ID,
        "input tangent",
    )?;
    require_length(
        target_tangent.len(),
        input.len(),
        MSE_LOSS_OPERATION_ID,
        "target tangent",
    )?;
    let mut tangent = zeroed(input.len(), "MSE tangent")?;
    for (index, ((((tangent, input), input_tangent), target), target_tangent)) in tangent
        .iter_mut()
        .zip(input)
        .zip(input_tangent)
        .zip(target)
        .zip(target_tangent)
        .enumerate()
    {
        check_periodically(index, context)?;
        let difference = *input - *target;
        let tangent_difference = *input_tangent - *target_tangent;
        *tangent = 2.0 * difference * tangent_difference;
    }
    reduce_loss(tangent, reduction, context)
}

pub fn sigmoid_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid(backend, input, context)?)
}

pub fn sigmoid_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(crate::generated_neural_network_functional_01::sigmoid_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn sigmoid_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartFourError> {
    context.cancellation.check()?;
    Ok(crate::generated_neural_network_functional_01::sigmoid_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

fn validate_dropout_probability(
    probability: f32,
) -> Result<(), NeuralNetworkModulePartFourError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return invalid(
            DROPOUT_OPERATION_ID,
            "dropout probability must be finite and in the inclusive range zero to one",
        );
    }
    Ok(())
}

fn reduce_loss(
    values: Vec<f32>,
    reduction: LossReduction,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    if reduction == LossReduction::None {
        return Ok(values);
    }
    let mut sum = 0.0_f32;
    for (index, value) in values.iter().copied().enumerate() {
        check_periodically(index, context)?;
        sum += value;
    }
    if reduction == LossReduction::Mean {
        sum /= values.len() as f32;
    }
    context.cancellation.check()?;
    Ok(vec![sum])
}

fn require_cpu(
    device: DeviceId,
    operation: &'static str,
) -> Result<(), NeuralNetworkModulePartFourError> {
    if device != DeviceId::CPU {
        return invalid(operation, "only the certified CPU backend is supported");
    }
    Ok(())
}

fn require_length(
    actual: usize,
    expected: usize,
    operation: &'static str,
    name: &'static str,
) -> Result<(), NeuralNetworkModulePartFourError> {
    if actual != expected {
        return invalid(
            operation,
            format!("{name} requires {expected} values, got {actual}"),
        );
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, NeuralNetworkModulePartFourError> {
    Err(NeuralNetworkModulePartFourError::Invalid {
        operation,
        reason: reason.into(),
    })
}

fn zeroed(
    length: usize,
    name: &'static str,
) -> Result<Vec<f32>, NeuralNetworkModulePartFourError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NeuralNetworkModulePartFourError::AllocationFailed(name))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn bools(
    length: usize,
    name: &'static str,
) -> Result<Vec<bool>, NeuralNetworkModulePartFourError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NeuralNetworkModulePartFourError::AllocationFailed(name))?;
    values.resize(length, false);
    Ok(values)
}

fn check_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), NeuralNetworkModulePartFourError> {
    if index & 1023 == 0 {
        context.cancellation.check()?;
    }
    Ok(())
}
