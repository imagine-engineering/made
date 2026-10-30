use crate::{
    CpuBackend, DeviceId, ExecutionContext, Tensor,
    autograd::{AutogradTape, LeafId},
    generated_activation_normalization_functional_01::{
        FunctionalError, GeluApproximation,
        gelu_jvp_with_context_exact_native as canonical_gelu_jvp,
        gelu_vjp_with_context_exact_native as canonical_gelu_vjp,
        gelu_with_context_exact_native as canonical_gelu,
        relu_jvp_with_context_exact_native as canonical_relu_jvp,
        relu_vjp_with_context_exact_native as canonical_relu_vjp,
        relu_with_context_exact_native as canonical_relu,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, ConvolutionVjp, OperatorIndirectionError,
        TensorValues, convolution_jvp_with_context_exact_native as canonical_convolution_jvp,
        convolution_vjp_with_context_exact_native as canonical_convolution_vjp,
        convolution_with_context_exact_native as canonical_convolution,
    },
    generated_elementwise_or_runtime_operation_17::{
        ElementwiseRuntimePartSeventeenError,
        requires_grad_method_exact_native as canonical_requires_grad,
    },
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError,
        pixel_shuffle_jvp_with_context_exact_native as canonical_pixel_shuffle_jvp,
        pixel_shuffle_vjp_with_context_exact_native as canonical_pixel_shuffle_vjp,
        pixel_shuffle_with_context_exact_native as canonical_pixel_shuffle,
        pixel_unshuffle_jvp_with_context_exact_native as canonical_pixel_unshuffle_jvp,
        pixel_unshuffle_vjp_with_context_exact_native as canonical_pixel_unshuffle_vjp,
        pixel_unshuffle_with_context_exact_native as canonical_pixel_unshuffle,
    },
    generated_neural_network_module_01::{
        AveragePoolGeometry, LossReduction, NeuralNetworkModuleError,
        smooth_l1_loss_jvp_with_context_exact_native as canonical_smooth_l1_jvp,
        smooth_l1_loss_vjp_with_context_exact_native as canonical_smooth_l1_vjp,
        smooth_l1_loss_with_context_exact_native as canonical_smooth_l1,
    },
    generated_neural_network_module_02::{NeuralNetworkModulePartTwoError, Pad2dGeometry},
};
use thiserror::Error;

pub const CONV_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-C76DE8CF2CF0";
pub const GELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-96DE512D2417";
pub const L1_LOSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-9FCDF546FE24";
pub const MAX_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-A2E5CD9C8E38";
pub const MODULE_DICT_OPERATION_ID: &str = "COMFY-TENSOR-OP-A666696563C8";
pub const MODULE_LIST_OPERATION_ID: &str = "COMFY-TENSOR-OP-D44F00F8A19B";
pub const PARAMETER_OPERATION_ID: &str = "COMFY-TENSOR-OP-B122B2B8E01C";
pub const PIXEL_SHUFFLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-9E325E7C79AB";
pub const PIXEL_UNSHUFFLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-92EB003A2648";
pub const RELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-904C1E14BAE4";
pub const RELU_6_OPERATION_ID: &str = "COMFY-TENSOR-OP-BF1F1FC66EDA";
pub const ZERO_PAD_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-98A798917C70";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NeuralNetworkModulePartThreeError {
    #[error("neural-network module part-three operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Functional(FunctionalError),
    #[error(transparent)]
    Module(NeuralNetworkModuleError),
    #[error(transparent)]
    Operator(OperatorIndirectionError),
    #[error("canonical neural-network functional operation failed: {0}")]
    NeuralFunctional(String),
    #[error("canonical neural-network module part-two operation failed: {0}")]
    ModulePartTwo(String),
    #[error("canonical autograd operation failed: {0}")]
    Autograd(String),
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for NeuralNetworkModulePartThreeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<FunctionalError> for NeuralNetworkModulePartThreeError {
    fn from(error: FunctionalError) -> Self {
        match error {
            FunctionalError::Cancelled => Self::Cancelled,
            error => Self::Functional(error),
        }
    }
}

impl From<NeuralNetworkModuleError> for NeuralNetworkModulePartThreeError {
    fn from(error: NeuralNetworkModuleError) -> Self {
        match error {
            NeuralNetworkModuleError::Cancelled => Self::Cancelled,
            error => Self::Module(error),
        }
    }
}

impl From<OperatorIndirectionError> for NeuralNetworkModulePartThreeError {
    fn from(error: OperatorIndirectionError) -> Self {
        match error {
            OperatorIndirectionError::Cancelled => Self::Cancelled,
            error => Self::Operator(error),
        }
    }
}

impl From<NeuralNetworkFunctionalError> for NeuralNetworkModulePartThreeError {
    fn from(error: NeuralNetworkFunctionalError) -> Self {
        match error {
            NeuralNetworkFunctionalError::Cancelled => Self::Cancelled,
            error => Self::NeuralFunctional(error.to_string()),
        }
    }
}

impl From<ElementwiseRuntimePartSeventeenError> for NeuralNetworkModulePartThreeError {
    fn from(error: ElementwiseRuntimePartSeventeenError) -> Self {
        match error {
            ElementwiseRuntimePartSeventeenError::Cancelled => Self::Cancelled,
            error => Self::Autograd(error.to_string()),
        }
    }
}

impl From<NeuralNetworkModulePartTwoError> for NeuralNetworkModulePartThreeError {
    fn from(error: NeuralNetworkModulePartTwoError) -> Self {
        match error {
            NeuralNetworkModulePartTwoError::Cancelled => Self::Cancelled,
            error => Self::ModulePartTwo(error.to_string()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv_3d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        3,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        false,
        vec![0; 3],
        ConvolutionPaddingMode::Zeros,
    )?;
    Ok(canonical_convolution(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_3d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        3,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        false,
        vec![0; 3],
        ConvolutionPaddingMode::Zeros,
    )?;
    Ok(canonical_convolution_vjp(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output_gradient,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_3d_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        3,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        false,
        vec![0; 3],
        ConvolutionPaddingMode::Zeros,
    )?;
    Ok(canonical_convolution_jvp(
        input,
        input_tangent,
        input_shape,
        weight,
        weight_tangent,
        weight_shape,
        bias,
        bias_tangent,
        &geometry,
        device,
        context,
    )?)
}

pub fn gelu_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_gelu(
        backend,
        input,
        approximation,
        device,
        context,
    )?)
}

pub fn gelu_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_gelu_vjp(
        backend,
        input,
        output_gradient,
        approximation,
        device,
        context,
    )?)
}

pub fn gelu_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_gelu_jvp(
        backend,
        input,
        input_tangent,
        approximation,
        device,
        context,
    )?)
}

pub fn l1_loss_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_smooth_l1(
        backend, input, target, 0.0, reduction, device, context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn l1_loss_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    reduction: LossReduction,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_smooth_l1_vjp(
        backend,
        input,
        target,
        0.0,
        reduction,
        output_gradient,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn l1_loss_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    target: &[f32],
    target_tangent: &[f32],
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_smooth_l1_jvp(
        backend,
        input,
        input_tangent,
        target,
        target_tangent,
        0.0,
        reduction,
        device,
        context,
    )?)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaxPool2dVjp {
    pub input: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_cpu(device, MAX_POOL_2D_OPERATION_ID)?;
    validate_max_pool_padding(kernel_size, padding)?;
    let geometry = AveragePoolGeometry::new_extended(
        input,
        input_shape,
        &kernel_size,
        &stride,
        &padding,
        &dilation,
        ceil_mode,
        MAX_POOL_2D_OPERATION_ID,
    )?;
    let (values, _) = max_pool_selection(input, &geometry, context)?;
    Ok(TensorValues {
        values,
        shape: geometry.output_shape().to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<MaxPool2dVjp, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_cpu(device, MAX_POOL_2D_OPERATION_ID)?;
    validate_max_pool_padding(kernel_size, padding)?;
    let geometry = AveragePoolGeometry::new_extended(
        input,
        input_shape,
        &kernel_size,
        &stride,
        &padding,
        &dilation,
        ceil_mode,
        MAX_POOL_2D_OPERATION_ID,
    )?;
    let (_, selected) = max_pool_selection(input, &geometry, context)?;
    require_length(
        output_gradient.len(),
        selected.len(),
        MAX_POOL_2D_OPERATION_ID,
        "output gradient",
    )?;
    let mut gradient = vec![0.0; input.len()];
    for (output_index, input_index) in selected.into_iter().enumerate() {
        let input_gradient = gradient.get_mut(input_index).ok_or(
            NeuralNetworkModulePartThreeError::ShapeOverflow("max-pool selected index"),
        )?;
        let output_gradient = output_gradient.get(output_index).ok_or(
            NeuralNetworkModulePartThreeError::ShapeOverflow("max-pool output gradient index"),
        )?;
        *input_gradient += output_gradient;
    }
    context.cancellation.check()?;
    Ok(MaxPool2dVjp { input: gradient })
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_cpu(device, MAX_POOL_2D_OPERATION_ID)?;
    validate_max_pool_padding(kernel_size, padding)?;
    require_length(
        input_tangent.len(),
        input.len(),
        MAX_POOL_2D_OPERATION_ID,
        "input tangent",
    )?;
    let geometry = AveragePoolGeometry::new_extended(
        input,
        input_shape,
        &kernel_size,
        &stride,
        &padding,
        &dilation,
        ceil_mode,
        MAX_POOL_2D_OPERATION_ID,
    )?;
    let (_, selected) = max_pool_selection(input, &geometry, context)?;
    let values = selected
        .into_iter()
        .map(|index| {
            input_tangent.get(index).copied().ok_or(
                NeuralNetworkModulePartThreeError::ShapeOverflow("max-pool tangent index"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorValues {
        values,
        shape: geometry.output_shape().to_vec(),
    })
}

fn max_pool_selection(
    input: &[f32],
    geometry: &AveragePoolGeometry,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<f32>, Vec<usize>), NeuralNetworkModulePartThreeError> {
    let count = geometry.output_count()?;
    let mut values = vec![f32::NEG_INFINITY; count];
    let mut selected = vec![usize::MAX; count];
    geometry.for_each_connection(context, |input_index, output_index, _| {
        let value = *input
            .get(input_index)
            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                "max-pool input index",
            ))?;
        let selected_index =
            selected
                .get_mut(output_index)
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "max-pool output index",
                ))?;
        let output_value =
            values
                .get_mut(output_index)
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "max-pool output index",
                ))?;
        if *selected_index == usize::MAX || value.is_nan() || value > *output_value {
            *output_value = value;
            *selected_index = input_index;
        }
        Ok(())
    })?;
    if selected.contains(&usize::MAX) {
        return invalid(
            MAX_POOL_2D_OPERATION_ID,
            "max-pool window has no input values",
        );
    }
    Ok((values, selected))
}

pub fn parameter_exact_native(
    tape: &mut AutogradTape,
    input: &Tensor,
    leaf: Option<LeafId>,
    requires_grad: bool,
    cancellation: &sim_types::CancellationToken,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    cancellation.check()?;
    Ok(canonical_requires_grad(
        tape,
        input,
        leaf,
        requires_grad,
        cancellation,
    )?)
}

pub fn pixel_shuffle_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_shuffle(backend, input, factor, context)?)
}
pub fn pixel_shuffle_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_shape: &[u64],
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_shuffle_vjp(
        backend,
        input,
        input_shape,
        factor,
        context,
    )?)
}
pub fn pixel_shuffle_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_shuffle_jvp(
        backend, input, factor, context,
    )?)
}
pub fn pixel_unshuffle_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_unshuffle(backend, input, factor, context)?)
}
pub fn pixel_unshuffle_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_shape: &[u64],
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_unshuffle_vjp(
        backend,
        input,
        input_shape,
        factor,
        context,
    )?)
}
pub fn pixel_unshuffle_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_pixel_unshuffle_jvp(
        backend, input, factor, context,
    )?)
}

pub fn relu_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_relu(backend, input, device, context)?)
}
pub fn relu_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_relu_vjp(
        backend,
        input,
        output_gradient,
        device,
        context,
    )?)
}
pub fn relu_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    Ok(canonical_relu_jvp(
        backend,
        input,
        input_tangent,
        device,
        context,
    )?)
}

pub fn relu_6_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    let mut output = relu_module_with_context_exact_native(backend, input, device, context)?;
    for value in &mut output {
        if *value > 6.0 {
            *value = 6.0;
        }
    }
    context.cancellation.check()?;
    Ok(output)
}
pub fn relu_6_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_length(
        output_gradient.len(),
        input.len(),
        RELU_6_OPERATION_ID,
        "output gradient",
    )?;
    let mut gradient = canonical_relu_vjp(backend, input, output_gradient, device, context)?;
    for (gradient, input) in gradient.iter_mut().zip(input) {
        if *input >= 6.0 {
            *gradient = 0.0;
        }
    }
    Ok(gradient)
}
pub fn relu_6_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    relu_6_vjp_with_context_exact_native(backend, input, input_tangent, device, context)
}

pub fn zero_pad_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_cpu(device, ZERO_PAD_2D_OPERATION_ID)?;
    let geometry = Pad2dGeometry::new(
        input,
        input_shape,
        padding,
        ZERO_PAD_2D_OPERATION_ID,
        ConvolutionPaddingMode::Zeros,
    )?;
    let mut output = vec![0.0; geometry.output_count()?];
    geometry.for_each_mapping(context, |input_index, output_index| {
        if let Some(input_index) = input_index {
            output[output_index] = input[input_index];
        }
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape().to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn zero_pad_2d_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartThreeError> {
    context.cancellation.check()?;
    require_cpu(device, ZERO_PAD_2D_OPERATION_ID)?;
    let geometry = Pad2dGeometry::new(
        input,
        input_shape,
        padding,
        ZERO_PAD_2D_OPERATION_ID,
        ConvolutionPaddingMode::Zeros,
    )?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        ZERO_PAD_2D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = vec![0.0; input.len()];
    geometry.for_each_mapping(context, |input_index, output_index| {
        if let Some(input_index) = input_index {
            input_gradient[input_index] += output_gradient[output_index];
        }
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(input_gradient)
}

pub fn zero_pad_2d_jvp_with_context_exact_native(
    input_tangent: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartThreeError> {
    zero_pad_2d_with_context_exact_native(input_tangent, input_shape, padding, device, context)
}

fn require_cpu(
    device: DeviceId,
    operation: &'static str,
) -> Result<(), NeuralNetworkModulePartThreeError> {
    if device != DeviceId::CPU {
        return invalid(operation, "only the certified CPU backend is supported");
    }
    Ok(())
}
fn validate_max_pool_padding(
    kernel_size: [usize; 2],
    padding: [usize; 2],
) -> Result<(), NeuralNetworkModulePartThreeError> {
    if padding
        .iter()
        .zip(kernel_size)
        .any(|(padding, kernel)| *padding > kernel / 2)
    {
        return invalid(
            MAX_POOL_2D_OPERATION_ID,
            "padding must not exceed half the kernel size",
        );
    }
    Ok(())
}
fn require_length(
    actual: usize,
    expected: usize,
    operation: &'static str,
    name: &str,
) -> Result<(), NeuralNetworkModulePartThreeError> {
    if actual != expected {
        return invalid(
            operation,
            format!("{name} length {actual} does not match {expected}"),
        );
    }
    Ok(())
}
fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, NeuralNetworkModulePartThreeError> {
    Err(NeuralNetworkModulePartThreeError::Invalid {
        operation,
        reason: reason.into(),
    })
}
