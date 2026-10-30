use crate::{
    AutogradError, AutogradTape, CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext,
    GradientMode, GradientReducer, GradientStore, Layout, LeafId, OutputSlot, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError, UnaryOperation, ViewAccess,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError,
        sin_jvp_with_context_exact_native as canonical_sin_jvp_with_context,
        sin_vjp_with_context_exact_native as canonical_sin_vjp_with_context,
        sin_with_context_exact_native as canonical_sin_with_context,
    },
    generated_elementwise_or_runtime_operation_09::{
        BinaryGradients, ElementwiseRuntimePartNineError, NativeBitwiseOperation,
        bitwise_binary_with_context_exact_native as canonical_bitwise_binary_with_context,
        mul_jvp_with_context_exact_native as canonical_mul_jvp_with_context,
        mul_vjp_with_context_exact_native as canonical_mul_vjp_with_context,
        mul_with_context_exact_native as canonical_mul_with_context,
    },
    generated_elementwise_or_runtime_operation_10::{
        ElementwiseRuntimePartTenError, NativeCumulativeOperation,
        cumulative_with_context_exact_native as canonical_cumulative_with_context,
    },
    generated_elementwise_or_runtime_operation_15::{
        ElementwiseRuntimePartFifteenError,
        ones_like_with_context_exact_native as canonical_ones_like_with_context,
    },
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError,
        add_method_jvp_with_context_exact_native as canonical_add_jvp_with_context,
        add_method_vjp_with_context_exact_native as canonical_add_vjp_with_context,
        add_method_with_context_exact_native as canonical_add_with_context,
    },
};
use thiserror::Error;

pub const BACKWARD_OPERATION_ID: &str = "COMFY-TENSOR-OP-ED2FCEFE4ECE";
pub const EXP_OPERATION_ID: &str = "COMFY-TENSOR-OP-E9EE26A3960C";
pub const NEG_OPERATION_ID: &str = "COMFY-TENSOR-OP-ECF812A5CF81";
pub const UNFOLD_OPERATION_ID: &str = "COMFY-TENSOR-OP-E78AD841C264";
pub const ADDCMUL_OPERATION_ID: &str = "COMFY-TENSOR-OP-EBFD0D7FDA6D";
pub const BITWISE_OR_OPERATION_ID: &str = "COMFY-TENSOR-OP-E8EA8CB65E2C";
pub const CUMPROD_OPERATION_ID: &str = "COMFY-TENSOR-OP-ECAAC1BA206A";
pub const GET_DEFAULT_DTYPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-EC849F37A5FD";
pub const IS_GRAD_ENABLED_OPERATION_ID: &str = "COMFY-TENSOR-OP-E8537C6996DA";
pub const KRON_OPERATION_ID: &str = "COMFY-TENSOR-OP-F122D7D4E807";
pub const MLU_MEMORY_STATS_OPERATION_ID: &str = "COMFY-TENSOR-OP-EA16F5C2EAC6";
pub const SIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-E851105B589B";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTwentyOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    PartFive(#[from] ElementwiseRuntimePartFiveError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error(transparent)]
    PartTen(#[from] ElementwiseRuntimePartTenError),
    #[error(transparent)]
    PartFifteen(#[from] ElementwiseRuntimePartFifteenError),
    #[error(transparent)]
    PartSixteen(#[from] ElementwiseRuntimePartSixteenError),
    #[error("elementwise/runtime part-twenty-one execution was cancelled")]
    Cancelled,
    #[error("operation {operation} requires CPU f32 input")]
    UnsupportedInput { operation: &'static str },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwentyOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Debug)]
pub struct AddCmulGradients {
    pub input: Tensor,
    pub tensor_one: Tensor,
    pub tensor_two: Tensor,
}

#[derive(Debug)]
pub struct KronGradients {
    pub input: Tensor,
    pub other: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn backward_method_with_context_exact_native(
    backend: &CpuBackend,
    tape: &mut AutogradTape,
    output_slot: OutputSlot,
    output: &Tensor,
    gradient: Option<Tensor>,
    inputs: Option<&[LeafId]>,
    reducer: &dyn GradientReducer,
    gradient_store: &mut GradientStore,
    retain_graph: bool,
    create_graph: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    context.check()?;
    let gradient = match gradient {
        Some(gradient) => gradient,
        None => {
            if element_count(output.descriptor().shape())? != 1 {
                return invalid(
                    BACKWARD_OPERATION_ID,
                    "an implicit gradient is valid only for scalar outputs",
                );
            }
            canonical_ones_like_with_context(backend, output, None, context)?
        }
    };
    require_same_tensor_geometry(output, &gradient, BACKWARD_OPERATION_ID)?;
    tape.reverse_and_publish_with_context(
        vec![(output_slot, gradient)],
        reducer,
        retain_graph,
        create_graph,
        inputs,
        gradient_store,
        backend,
        context,
    )?;
    context.check()?;
    Ok(())
}

pub fn exp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    unary_forward_with_context(
        backend,
        input,
        UnaryOperation::Exponential,
        EXP_OPERATION_ID,
        context,
    )
}

pub fn exp_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    let exponential = exp_with_context_exact_native(backend, input, context)?;
    Ok(canonical_mul_with_context(
        backend,
        &exponential,
        output_gradient,
        context,
    )?)
}

pub fn exp_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    exp_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn neg_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    unary_forward_with_context(
        backend,
        input,
        UnaryOperation::Negate,
        NEG_OPERATION_ID,
        context,
    )
}

pub fn neg_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    neg_with_context_exact_native(backend, output_gradient, context)
}

pub fn neg_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    neg_with_context_exact_native(backend, input_tangent, context)
}

pub fn unfold_exact_native(
    input: &Tensor,
    dimension: i64,
    size: u64,
    step: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    cancellation.check()?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), UNFOLD_OPERATION_ID)?;
    let dimension_length = input.descriptor().shape()[axis];
    if size == 0 || step == 0 || size > dimension_length {
        return invalid(
            UNFOLD_OPERATION_ID,
            "size and step must be positive and size must not exceed the selected dimension",
        );
    }
    let mut shape = input.descriptor().shape().to_vec();
    shape[axis] = (dimension_length - size) / step + 1;
    shape.push(size);
    let mut strides = input.descriptor().strides().to_vec();
    let source_stride = strides[axis];
    let step = i64::try_from(step)
        .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("unfold step"))?;
    strides[axis] = source_stride.checked_mul(step).ok_or(
        ElementwiseRuntimePartTwentyOneError::ShapeOverflow("unfold stride"),
    )?;
    strides.push(source_stride);
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        input.descriptor().offset_elements(),
        input.descriptor().dtype(),
        Layout::Strided,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

pub fn unfold_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    size: u64,
    step: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, UNFOLD_OPERATION_ID)?;
    require_f32_cpu(output_gradient, UNFOLD_OPERATION_ID)?;
    let expected = unfold_exact_native(input, dimension, size, step, context.cancellation)?;
    require_same_tensor_geometry(&expected, output_gradient, UNFOLD_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), UNFOLD_OPERATION_ID)?;
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(output_gradient.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, output_gradient.descriptor().shape())?;
        let mut input_indices = output_indices[..input.descriptor().rank()].to_vec();
        input_indices[axis] = output_indices[axis]
            .checked_mul(step)
            .and_then(|value| value.checked_add(output_indices[input.descriptor().rank()]))
            .ok_or(ElementwiseRuntimePartTwentyOneError::ShapeOverflow(
                "unfold VJP index",
            ))?;
        let input_linear = ravel_index(&input_indices, input.descriptor().shape())?;
        values[input_linear] += read_f32(output_gradient, &output_indices, UNFOLD_OPERATION_ID)?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn unfold_jvp_exact_native(
    input_tangent: &Tensor,
    dimension: i64,
    size: u64,
    step: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    cancellation.check()?;
    unfold_exact_native(input_tangent, dimension, size, step, cancellation)
}

pub fn addcmul_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor_one: &Tensor,
    tensor_two: &Tensor,
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    let product = canonical_mul_with_context(backend, tensor_one, tensor_two, context)?;
    Ok(canonical_add_with_context(
        backend,
        input,
        ElementwiseOperand::Tensor(&product),
        value,
        context,
    )?)
}

pub fn addcmul_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor_one: &Tensor,
    tensor_two: &Tensor,
    value: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddCmulGradients, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    let product = canonical_mul_with_context(backend, tensor_one, tensor_two, context)?;
    let addition = canonical_add_vjp_with_context(
        backend,
        input,
        ElementwiseOperand::Tensor(&product),
        value,
        output_gradient,
        context,
    )?;
    let product_gradient =
        addition
            .other
            .ok_or_else(|| ElementwiseRuntimePartTwentyOneError::Invalid {
                operation: ADDCMUL_OPERATION_ID,
                reason: "tensor addend did not produce a gradient".to_owned(),
            })?;
    let BinaryGradients { left, right } = canonical_mul_vjp_with_context(
        backend,
        tensor_one,
        tensor_two,
        &product_gradient,
        context,
    )?;
    Ok(AddCmulGradients {
        input: addition.input,
        tensor_one: left,
        tensor_two: right,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn addcmul_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor_one: &Tensor,
    tensor_two: &Tensor,
    input_tangent: &Tensor,
    tensor_one_tangent: &Tensor,
    tensor_two_tangent: &Tensor,
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_same_f32(input, input_tangent, ADDCMUL_OPERATION_ID)?;
    let product_tangent = canonical_mul_jvp_with_context(
        backend,
        tensor_one,
        tensor_two,
        tensor_one_tangent,
        tensor_two_tangent,
        context,
    )?;
    Ok(canonical_add_jvp_with_context(
        backend,
        input_tangent,
        Some(&product_tangent),
        value,
        context,
    )?)
}

pub fn bitwise_or_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    Ok(canonical_bitwise_binary_with_context(
        backend,
        left,
        right,
        NativeBitwiseOperation::Or,
        BITWISE_OR_OPERATION_ID,
        context,
    )?)
}

pub fn cumprod_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    Ok(canonical_cumulative_with_context(
        backend,
        input,
        dimension,
        dtype,
        NativeCumulativeOperation::Product,
        CUMPROD_OPERATION_ID,
        context,
    )?)
}

pub fn cumprod_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_same_f32(input, output_gradient, CUMPROD_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), CUMPROD_OPERATION_ID)?;
    let shape = input.descriptor().shape();
    let mut result = workspace_filled(backend, context, element_count(shape)?, 0.0_f32)?;
    for output_linear in 0..element_count(shape)? {
        check_periodically(output_linear, context.cancellation)?;
        let output_indices = unravel_index(output_linear, shape)?;
        let output_axis = output_indices[axis];
        let gradient = read_f32(output_gradient, &output_indices, CUMPROD_OPERATION_ID)?;
        for input_axis in 0..=output_axis {
            let mut product = 1.0_f32;
            for factor_axis in 0..=output_axis {
                if factor_axis != input_axis {
                    let mut factor_indices = output_indices.clone();
                    factor_indices[axis] = factor_axis;
                    product *= read_f32(input, &factor_indices, CUMPROD_OPERATION_ID)?;
                }
            }
            let mut input_indices = output_indices.clone();
            input_indices[axis] = input_axis;
            result[ravel_index(&input_indices, shape)?] += gradient * product;
        }
    }
    upload_f32_with_context(
        backend,
        shape,
        input.descriptor().stream(),
        &result,
        context,
    )
}

pub fn cumprod_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_same_f32(input, input_tangent, CUMPROD_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), CUMPROD_OPERATION_ID)?;
    let shape = input.descriptor().shape();
    let mut result = workspace_filled(backend, context, element_count(shape)?, 0.0_f32)?;
    for output_linear in 0..element_count(shape)? {
        check_periodically(output_linear, context.cancellation)?;
        let output_indices = unravel_index(output_linear, shape)?;
        let output_axis = output_indices[axis];
        let mut tangent = 0.0_f32;
        for tangent_axis in 0..=output_axis {
            let mut tangent_indices = output_indices.clone();
            tangent_indices[axis] = tangent_axis;
            let mut term = read_f32(input_tangent, &tangent_indices, CUMPROD_OPERATION_ID)?;
            for factor_axis in 0..=output_axis {
                if factor_axis != tangent_axis {
                    let mut factor_indices = output_indices.clone();
                    factor_indices[axis] = factor_axis;
                    term *= read_f32(input, &factor_indices, CUMPROD_OPERATION_ID)?;
                }
            }
            tangent += term;
        }
        result[output_linear] = tangent;
    }
    upload_f32_with_context(
        backend,
        shape,
        input.descriptor().stream(),
        &result,
        context,
    )
}

pub fn get_default_dtype_exact_native(
    cancellation: &CancellationToken,
) -> Result<DType, ElementwiseRuntimePartTwentyOneError> {
    cancellation.check()?;
    Ok(DType::F32)
}

pub fn is_grad_enabled_exact_native(
    mode: GradientMode,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwentyOneError> {
    cancellation.check()?;
    Ok(mode == GradientMode::Enabled)
}

pub fn kron_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_binary_f32(input, other, KRON_OPERATION_ID)?;
    let geometry = KronGeometry::new(input.descriptor().shape(), other.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, element_count(&geometry.output_shape)?)?;
    for linear in 0..element_count(&geometry.output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &geometry.output_shape)?;
        let (input_indices, other_indices) = geometry.source_indices(&output_indices)?;
        values.try_push(
            read_f32(input, &input_indices, KRON_OPERATION_ID)?
                * read_f32(other, &other_indices, KRON_OPERATION_ID)?,
        )?;
    }
    upload_f32_with_context(
        backend,
        &geometry.output_shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn kron_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<KronGradients, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_binary_f32(input, other, KRON_OPERATION_ID)?;
    require_f32_cpu(output_gradient, KRON_OPERATION_ID)?;
    let geometry = KronGeometry::new(input.descriptor().shape(), other.descriptor().shape())?;
    if output_gradient.descriptor().shape() != geometry.output_shape {
        return invalid(
            KRON_OPERATION_ID,
            "output gradient shape does not match kron output",
        );
    }
    let input_count = element_count(input.descriptor().shape())?;
    let other_count = element_count(other.descriptor().shape())?;
    let mut input_values = workspace_filled(backend, context, input_count, 0.0_f32)?;
    let mut other_values = workspace_filled(backend, context, other_count, 0.0_f32)?;
    for linear in 0..element_count(&geometry.output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &geometry.output_shape)?;
        let (input_indices, other_indices) = geometry.source_indices(&output_indices)?;
        let gradient = read_f32(output_gradient, &output_indices, KRON_OPERATION_ID)?;
        input_values[ravel_index(&input_indices, input.descriptor().shape())?] +=
            gradient * read_f32(other, &other_indices, KRON_OPERATION_ID)?;
        other_values[ravel_index(&other_indices, other.descriptor().shape())?] +=
            gradient * read_f32(input, &input_indices, KRON_OPERATION_ID)?;
    }
    let input_gradient = upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &input_values,
        context,
    )?;
    drop(input_values);
    let other_gradient = upload_f32_with_context(
        backend,
        other.descriptor().shape(),
        other.descriptor().stream(),
        &other_values,
        context,
    )?;
    Ok(KronGradients {
        input: input_gradient,
        other: other_gradient,
    })
}

pub fn kron_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    require_same_f32(input, input_tangent, KRON_OPERATION_ID)?;
    require_same_f32(other, other_tangent, KRON_OPERATION_ID)?;
    let left = kron_with_context_exact_native(backend, input_tangent, other, context)?;
    let right = kron_with_context_exact_native(backend, input, other_tangent, context)?;
    Ok(canonical_add_with_context(
        backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        1.0,
        context,
    )?)
}

pub fn sin_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    Ok(canonical_sin_with_context(backend, input, context)?)
}

pub fn sin_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    Ok(canonical_sin_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn sin_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.cancellation.check()?;
    Ok(canonical_sin_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

fn unary_forward_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    unary_operation: UnaryOperation,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    require_f32_cpu(input, operation)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(unary_operation, input, descriptor, context)?
        .0)
}

struct KronGeometry {
    input_shape: Vec<u64>,
    other_shape: Vec<u64>,
    output_shape: Vec<u64>,
    input_padding: usize,
    other_padding: usize,
}

impl KronGeometry {
    fn new(
        input_shape: &[u64],
        other_shape: &[u64],
    ) -> Result<Self, ElementwiseRuntimePartTwentyOneError> {
        let rank = input_shape.len().max(other_shape.len());
        let input_padding = rank - input_shape.len();
        let other_padding = rank - other_shape.len();
        let mut output_shape = Vec::new();
        output_shape
            .try_reserve_exact(rank)
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("kron rank"))?;
        for axis in 0..rank {
            let input_dimension = if axis < input_padding {
                1
            } else {
                input_shape[axis - input_padding]
            };
            let other_dimension = if axis < other_padding {
                1
            } else {
                other_shape[axis - other_padding]
            };
            output_shape.push(input_dimension.checked_mul(other_dimension).ok_or(
                ElementwiseRuntimePartTwentyOneError::ShapeOverflow("kron shape"),
            )?);
        }
        Ok(Self {
            input_shape: input_shape.to_vec(),
            other_shape: other_shape.to_vec(),
            output_shape,
            input_padding,
            other_padding,
        })
    }

    fn source_indices(
        &self,
        output_indices: &[u64],
    ) -> Result<(Vec<u64>, Vec<u64>), ElementwiseRuntimePartTwentyOneError> {
        let mut input_indices = Vec::new();
        let mut other_indices = Vec::new();
        input_indices
            .try_reserve_exact(self.input_shape.len())
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("kron indices"))?;
        other_indices
            .try_reserve_exact(self.other_shape.len())
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("kron indices"))?;
        for (axis, output_index) in output_indices.iter().copied().enumerate() {
            let other_dimension = if axis < self.other_padding {
                1
            } else {
                self.other_shape[axis - self.other_padding]
            };
            if axis >= self.input_padding {
                input_indices.push(output_index / other_dimension);
            }
            if axis >= self.other_padding {
                other_indices.push(output_index % other_dimension);
            }
        }
        Ok((input_indices, other_indices))
    }
}

fn require_binary_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    if left.descriptor().stream() != right.descriptor().stream() {
        return invalid(operation, "input streams differ");
    }
    Ok(())
}

fn require_same_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    require_binary_f32(left, right, operation)?;
    require_same_tensor_geometry(left, right, operation)
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    if input.descriptor().device() != DeviceId::CPU || input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartTwentyOneError::UnsupportedInput { operation });
    }
    Ok(())
}

fn require_same_tensor_geometry(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    if left.descriptor().shape() != right.descriptor().shape()
        || left.descriptor().dtype() != right.descriptor().dtype()
        || left.descriptor().device() != right.descriptor().device()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return invalid(operation, "tensor geometry differs");
    }
    Ok(())
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ElementwiseRuntimePartTwentyOneError> {
    if rank == 0 {
        return invalid(operation, "operation requires a non-scalar tensor");
    }
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("rank"))?;
    let dimension = if dimension < 0 {
        dimension + rank
    } else {
        dimension
    };
    if !(0..rank).contains(&dimension) {
        return invalid(operation, "dimension is out of range");
    }
    usize::try_from(dimension)
        .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTwentyOneError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("element count"))?;
        count
            .checked_mul(dimension)
            .ok_or(ElementwiseRuntimePartTwentyOneError::ShapeOverflow(
                "element count",
            ))
    })
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTwentyOneError> {
    let mut indices = vec![0_u64; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[axis])
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("index"))?;
        if dimension == 0 {
            return invalid(KRON_OPERATION_ID, "zero-sized dimensions are unsupported");
        }
        indices[axis] = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartTwentyOneError> {
    if indices.len() != shape.len() {
        return invalid(KRON_OPERATION_ID, "index rank differs from shape rank");
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("linear index"))?;
            let index = usize::try_from(*index)
                .map_err(|_| ElementwiseRuntimePartTwentyOneError::ShapeOverflow("linear index"))?;
            linear
                .checked_mul(dimension)
                .and_then(|value| value.checked_add(index))
                .ok_or(ElementwiseRuntimePartTwentyOneError::ShapeOverflow(
                    "linear index",
                ))
        })
}

fn read_f32(
    input: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartTwentyOneError> {
    require_f32_cpu(input, operation)?;
    let bytes: [u8; 4] = input.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartTwentyOneError::Invalid {
            operation,
            reason: "f32 element has an invalid width".to_owned(),
        }
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyOneError> {
    context.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<crate::CpuWorkspaceVec<T>, ElementwiseRuntimePartTwentyOneError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwentyOneError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ElementwiseRuntimePartTwentyOneError> {
    Err(ElementwiseRuntimePartTwentyOneError::Invalid {
        operation,
        reason: reason.into(),
    })
}
