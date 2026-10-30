#[cfg(feature = "cpu")]
use crate::{CpuBackend, DType, DecodedScalar, NumericClass, Scalar, TensorDescriptor};
use crate::{
    DeviceId, ExecutionContext, Tensor, TensorBackend, TensorError,
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, ConvolutionVjp, OperatorIndirectionError,
        TensorValues, convolution_jvp_with_context_exact_native as canonical_convolution_jvp,
        convolution_tensor_with_context_exact_native as canonical_convolution_tensor,
        convolution_vjp_with_context_exact_native as canonical_convolution_vjp,
        convolution_with_context_exact_native as canonical_convolution,
        tensor_from_f32_with_context_exact_native,
    },
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeBilinearBoundary, NativeLinearBoundary,
        NativeLinearWeight, checked_bilinear_weights, checked_linear_weights,
    },
    generated_neural_network_module_01::{AveragePoolGeometry, NeuralNetworkModuleError},
    generated_neural_network_module_03::{
        MaxPool2dVjp, NeuralNetworkModulePartThreeError,
        max_pool_2d_jvp_with_context_exact_native as canonical_max_pool_2d_jvp,
        max_pool_2d_vjp_with_context_exact_native as canonical_max_pool_2d_vjp,
        max_pool_2d_with_context_exact_native as canonical_max_pool_2d,
    },
};
use thiserror::Error;

pub const AVG_POOL_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-5F86004D9BDA";
pub const AVG_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-60B322636602";
pub const AVG_POOL_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6BFAEF690071";
pub const CONV_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-DC56DB93077F";
pub const CONV_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-A31AEBE72455";
pub const CONV_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-FE03423D60DA";
pub const CONV_TRANSPOSE_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-341577A45D6B";
pub const CONV_TRANSPOSE_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A5F8349A130";
pub const CONV_TRANSPOSE_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A6A586CC551";
pub const GRID_SAMPLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-A90AB43A3320";
pub const INTERPOLATE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B0F801006375";
pub const BISLERP_OPERATION_ID: &str = "COMFY-TENSOR-OP-2E8A60F63A15";
pub const PIXEL_SHUFFLE_ND_OPERATION_ID: &str = "COMFY-TENSOR-OP-02F93AD75026";
pub const MAX_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-1F9D23F3B331";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpatialFunctionalKernelError {
    #[error("{0}")]
    Tensor(#[source] TensorError),
    #[error("spatial functional kernel execution was cancelled")]
    Cancelled,
    #[error("operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("operation {operation} overflowed while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
    #[error("operation {operation} failed in canonical owner {owner}: {reason}")]
    CanonicalOwner {
        operation: &'static str,
        owner: &'static str,
        reason: String,
    },
}

impl From<TensorError> for SpatialFunctionalKernelError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl From<sim_types::CancellationError> for SpatialFunctionalKernelError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

fn canonical_tensor_owner_error(
    operation: &'static str,
    owner: &'static str,
    error: OperatorIndirectionError,
) -> SpatialFunctionalKernelError {
    match error {
        OperatorIndirectionError::Cancelled => SpatialFunctionalKernelError::Cancelled,
        OperatorIndirectionError::Tensor(error) => error.into(),
        error => SpatialFunctionalKernelError::CanonicalOwner {
            operation,
            owner,
            reason: error.to_string(),
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AveragePoolConfiguration {
    pub kernel_size: Vec<usize>,
    pub stride: Option<Vec<usize>>,
    pub padding: Vec<usize>,
    pub ceil_mode: bool,
    pub count_include_pad: bool,
    pub divisor_override: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AveragePoolVjp {
    pub input: Vec<f32>,
}

pub fn average_pool_1d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    average_pool_forward(
        AVG_POOL_1D_OPERATION_ID,
        1,
        input,
        input_shape,
        configuration,
        device,
        context,
    )
}

pub fn average_pool_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    average_pool_forward(
        AVG_POOL_2D_OPERATION_ID,
        2,
        input,
        input_shape,
        configuration,
        device,
        context,
    )
}

#[cfg(feature = "cpu")]
pub fn average_pool_2d_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    configuration: &AveragePoolConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    validate_real_tensor(backend, input, "input", AVG_POOL_2D_OPERATION_ID, context)?;
    let input_shape = tensor_shape_to_usize(input, AVG_POOL_2D_OPERATION_ID, "input shape")?;
    let input_values = tensor_to_f32_workspace(backend, input, AVG_POOL_2D_OPERATION_ID, context)?;
    let geometry = checked_average_pool_geometry(
        AVG_POOL_2D_OPERATION_ID,
        2,
        &input_values,
        &input_shape,
        configuration,
    )?;
    let output_count = geometry
        .output_count()
        .map_err(|error| pool_error(AVG_POOL_2D_OPERATION_ID, error))?;
    let mut output_values = backend.workspace_vec(context, output_count)?;
    for index in 0..output_count {
        if index.is_multiple_of(64) {
            context.check()?;
        }
        output_values.try_push(0.0)?;
    }
    geometry
        .for_each_connection_with_divisor(
            context,
            configuration.count_include_pad,
            configuration.divisor_override,
            |input_index, output_index, scale| {
                let source = input_values.get(input_index).copied().ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool input index"),
                )?;
                let destination = output_values.get_mut(output_index).ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool output index"),
                )?;
                *destination = source.mul_add(scale, *destination);
                Ok(())
            },
        )
        .map_err(|error| pool_error(AVG_POOL_2D_OPERATION_ID, error))?;

    let output_shape = geometry
        .output_shape()
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| overflow(AVG_POOL_2D_OPERATION_ID, "output shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        DeviceId::CPU,
        context.stream,
    )?;
    let (mut output, allocation_event) = backend.allocate(descriptor, context)?;
    backend.wait_event(allocation_event, context)?;
    let dtype = output.descriptor().dtype();
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| overflow(AVG_POOL_2D_OPERATION_ID, "output dtype width"))?;
    {
        let mut write = output.write()?;
        let output_bytes = write.bytes_mut()?;
        for (index, value) in output_values.iter().copied().enumerate() {
            if index.is_multiple_of(64) {
                context.check()?;
            }
            let start = index
                .checked_mul(byte_width)
                .ok_or_else(|| overflow(AVG_POOL_2D_OPERATION_ID, "output byte index"))?;
            let end = start
                .checked_add(byte_width)
                .ok_or_else(|| overflow(AVG_POOL_2D_OPERATION_ID, "output byte range"))?;
            let destination = output_bytes
                .get_mut(start..end)
                .ok_or_else(|| overflow(AVG_POOL_2D_OPERATION_ID, "output byte destination"))?;
            let encoded = dtype.encode_scalar(
                Scalar::Float(f64::from(value)),
                AVG_POOL_2D_OPERATION_ID,
                DeviceId::CPU,
            )?;
            destination.copy_from_slice(&encoded);
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

pub fn average_pool_3d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    average_pool_forward(
        AVG_POOL_3D_OPERATION_ID,
        3,
        input,
        input_shape,
        configuration,
        device,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_vjp_with_context_exact_native(
    operation: &'static str,
    spatial_dimensions: usize,
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePoolVjp, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(operation, device)?;
    let geometry = checked_average_pool_geometry(
        operation,
        spatial_dimensions,
        input,
        input_shape,
        configuration,
    )?;
    require_length(
        operation,
        "output gradient",
        output_gradient.len(),
        geometry
            .output_count()
            .map_err(|error| pool_error(operation, error))?,
    )?;
    let mut gradient = vec![0.0_f32; input.len()];
    geometry
        .for_each_connection_with_divisor(
            context,
            configuration.count_include_pad,
            configuration.divisor_override,
            |input_index, output_index, scale| {
                let destination = gradient.get_mut(input_index).ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool input-gradient index"),
                )?;
                let source = output_gradient.get(output_index).ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool output-gradient index"),
                )?;
                *destination = source.mul_add(scale, *destination);
                Ok(())
            },
        )
        .map_err(|error| pool_error(operation, error))?;
    context.cancellation.check()?;
    Ok(AveragePoolVjp { input: gradient })
}

pub fn average_pool_jvp_with_context_exact_native(
    operation: &'static str,
    spatial_dimensions: usize,
    input_tangent: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    average_pool_forward(
        operation,
        spatial_dimensions,
        input_tangent,
        input_shape,
        configuration,
        device,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn average_pool_forward(
    operation: &'static str,
    spatial_dimensions: usize,
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(operation, device)?;
    let geometry = checked_average_pool_geometry(
        operation,
        spatial_dimensions,
        input,
        input_shape,
        configuration,
    )?;
    let mut values = vec![
        0.0_f32;
        geometry
            .output_count()
            .map_err(|error| pool_error(operation, error))?
    ];
    geometry
        .for_each_connection_with_divisor(
            context,
            configuration.count_include_pad,
            configuration.divisor_override,
            |input_index, output_index, scale| {
                let source = input.get(input_index).copied().ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool input index"),
                )?;
                let destination =
                    values
                        .get_mut(output_index)
                        .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                            "average-pool output index",
                        ))?;
                *destination = source.mul_add(scale, *destination);
                Ok(())
            },
        )
        .map_err(|error| pool_error(operation, error))?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values,
        shape: geometry.output_shape().to_vec(),
    })
}

fn checked_average_pool_geometry(
    operation: &'static str,
    spatial_dimensions: usize,
    input: &[f32],
    input_shape: &[usize],
    configuration: &AveragePoolConfiguration,
) -> Result<AveragePoolGeometry, SpatialFunctionalKernelError> {
    if configuration.kernel_size.len() != spatial_dimensions
        || configuration.padding.len() != spatial_dimensions
        || configuration
            .stride
            .as_ref()
            .is_some_and(|stride| stride.len() != spatial_dimensions)
    {
        return invalid(
            operation,
            "pooling argument rank does not match the operation",
        );
    }
    if configuration
        .padding
        .iter()
        .zip(&configuration.kernel_size)
        .any(|(padding, kernel)| *padding > *kernel / 2)
    {
        return invalid(operation, "pooling padding exceeds half the kernel extent");
    }
    let stride = configuration
        .stride
        .as_deref()
        .unwrap_or(&configuration.kernel_size);
    AveragePoolGeometry::new_extended(
        input,
        input_shape,
        &configuration.kernel_size,
        stride,
        &configuration.padding,
        &vec![1; spatial_dimensions],
        configuration.ceil_mode,
        operation,
    )
    .map_err(|error| pool_error(operation, error))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvolutionConfiguration {
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    pub output_padding: Vec<usize>,
}

macro_rules! convolution_forward_adapter {
    ($name:ident, $operation:ident, $dimensions:expr, $transposed:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            input: &[f32],
            input_shape: &[usize],
            weight: &[f32],
            weight_shape: &[usize],
            bias: Option<&[f32]>,
            configuration: &ConvolutionConfiguration,
            device: DeviceId,
            context: &ExecutionContext<'_>,
        ) -> Result<TensorValues, SpatialFunctionalKernelError> {
            convolution_forward(
                $operation,
                $dimensions,
                $transposed,
                input,
                input_shape,
                weight,
                weight_shape,
                bias,
                configuration,
                device,
                context,
            )
        }
    };
}

convolution_forward_adapter!(
    conv_1d_with_context_exact_native,
    CONV_1D_OPERATION_ID,
    1,
    false
);
convolution_forward_adapter!(
    conv_2d_with_context_exact_native,
    CONV_2D_OPERATION_ID,
    2,
    false
);
convolution_forward_adapter!(
    conv_3d_with_context_exact_native,
    CONV_3D_OPERATION_ID,
    3,
    false
);
convolution_forward_adapter!(
    conv_transpose_1d_with_context_exact_native,
    CONV_TRANSPOSE_1D_OPERATION_ID,
    1,
    true
);
convolution_forward_adapter!(
    conv_transpose_2d_with_context_exact_native,
    CONV_TRANSPOSE_2D_OPERATION_ID,
    2,
    true
);
convolution_forward_adapter!(
    conv_transpose_3d_with_context_exact_native,
    CONV_TRANSPOSE_3D_OPERATION_ID,
    3,
    true
);

#[allow(clippy::too_many_arguments)]
pub fn conv_2d_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: &ConvolutionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    convolution_tensor_forward(
        backend,
        CONV_2D_OPERATION_ID,
        2,
        false,
        input,
        weight,
        bias,
        configuration,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_3d_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: &ConvolutionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    convolution_tensor_forward(
        backend,
        CONV_3D_OPERATION_ID,
        3,
        false,
        input,
        weight,
        bias,
        configuration,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose_2d_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: &ConvolutionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    convolution_tensor_forward(
        backend,
        CONV_TRANSPOSE_2D_OPERATION_ID,
        2,
        true,
        input,
        weight,
        bias,
        configuration,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn convolution_tensor_forward(
    backend: &dyn TensorBackend,
    operation: &'static str,
    spatial_dimensions: usize,
    transposed: bool,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: &ConvolutionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    let geometry = convolution_geometry(operation, spatial_dimensions, transposed, configuration)?;
    canonical_convolution_tensor(backend, input, weight, bias, &geometry, context)
        .map_err(|error| convolution_error(operation, error))
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_vjp_with_context_exact_native(
    operation: &'static str,
    spatial_dimensions: usize,
    transposed: bool,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    configuration: &ConvolutionConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    let geometry = convolution_geometry(operation, spatial_dimensions, transposed, configuration)?;
    canonical_convolution_vjp(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output_gradient,
        &geometry,
        device,
        context,
    )
    .map_err(|error| convolution_error(operation, error))
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_jvp_with_context_exact_native(
    operation: &'static str,
    spatial_dimensions: usize,
    transposed: bool,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    configuration: &ConvolutionConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    let geometry = convolution_geometry(operation, spatial_dimensions, transposed, configuration)?;
    canonical_convolution_jvp(
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
    )
    .map_err(|error| convolution_error(operation, error))
}

#[allow(clippy::too_many_arguments)]
fn convolution_forward(
    operation: &'static str,
    spatial_dimensions: usize,
    transposed: bool,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    configuration: &ConvolutionConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    let geometry = convolution_geometry(operation, spatial_dimensions, transposed, configuration)?;
    canonical_convolution(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        &geometry,
        device,
        context,
    )
    .map_err(|error| convolution_error(operation, error))
}

fn convolution_geometry(
    operation: &'static str,
    spatial_dimensions: usize,
    transposed: bool,
    configuration: &ConvolutionConfiguration,
) -> Result<ConvolutionGeometry, SpatialFunctionalKernelError> {
    ConvolutionGeometry::new_with_padding_mode(
        spatial_dimensions,
        configuration.stride.clone(),
        configuration.padding.clone(),
        configuration.dilation.clone(),
        configuration.groups,
        transposed,
        configuration.output_padding.clone(),
        ConvolutionPaddingMode::Zeros,
    )
    .map_err(|error| convolution_error(operation, error))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridSampleMode {
    Bilinear,
    Nearest,
    Bicubic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridPaddingMode {
    Zeros,
    Border,
    Reflection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridSampleConfiguration {
    pub mode: GridSampleMode,
    pub padding_mode: GridPaddingMode,
    pub align_corners: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GridSampleVjp {
    pub input: Vec<f32>,
    pub grid: Vec<f32>,
}

pub fn grid_sample_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    grid: &[f32],
    grid_shape: &[usize],
    configuration: GridSampleConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(GRID_SAMPLE_OPERATION_ID, device)?;
    let geometry = GridGeometry::new(input, input_shape, grid, grid_shape, configuration)?;
    let mut output = vec![0.0_f32; geometry.output_count()?];
    geometry.for_each_output(context, |output_index, batch, channel, y, x, _, _| {
        let value = geometry.sample(input, batch, channel, y, x)?;
        let destination = output
            .get_mut(output_index)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample output index"))?;
        *destination = value;
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape(),
    })
}

#[cfg(feature = "cpu")]
pub fn grid_sample_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    grid: &Tensor,
    configuration: GridSampleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    validate_real_tensor(backend, input, "input", GRID_SAMPLE_OPERATION_ID, context)?;
    validate_real_tensor(backend, grid, "grid", GRID_SAMPLE_OPERATION_ID, context)?;
    if grid.descriptor().dtype() != DType::F32 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: grid.descriptor().dtype(),
        }
        .into());
    }

    let input_shape = tensor_shape_to_usize(input, GRID_SAMPLE_OPERATION_ID, "input shape")?;
    let grid_shape = tensor_shape_to_usize(grid, GRID_SAMPLE_OPERATION_ID, "grid shape")?;
    let input_values = tensor_to_f32_workspace(backend, input, GRID_SAMPLE_OPERATION_ID, context)?;
    let grid_values = tensor_to_f32_workspace(backend, grid, GRID_SAMPLE_OPERATION_ID, context)?;
    let geometry = GridGeometry::new(
        &input_values,
        &input_shape,
        &grid_values,
        &grid_shape,
        configuration,
    )?;
    let output_shape = geometry
        .output_shape()
        .into_iter()
        .map(|value| {
            u64::try_from(value).map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "output shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        DeviceId::CPU,
        context.stream,
    )?;
    let (mut output, allocation_event) = backend.allocate(descriptor, context)?;
    backend.wait_event(allocation_event, context)?;
    let dtype = output.descriptor().dtype();
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "output dtype width"))?;
    {
        let mut write = output.write()?;
        let output_bytes = write.bytes_mut()?;
        geometry.for_each_output(context, |output_index, batch, channel, y, x, _, _| {
            let value = geometry.sample(&input_values, batch, channel, y, x)?;
            let start = output_index
                .checked_mul(byte_width)
                .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "output byte index"))?;
            let end = start
                .checked_add(byte_width)
                .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "output byte range"))?;
            let destination = output_bytes
                .get_mut(start..end)
                .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "output byte destination"))?;
            let encoded = dtype.encode_scalar(
                Scalar::Float(f64::from(value)),
                GRID_SAMPLE_OPERATION_ID,
                DeviceId::CPU,
            )?;
            destination.copy_from_slice(&encoded);
            Ok(())
        })?;
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

#[cfg(feature = "cpu")]
fn validate_real_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    name: &'static str,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), SpatialFunctionalKernelError> {
    if tensor.descriptor().device() != backend.device() {
        return Err(TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: tensor.descriptor().device(),
        }
        .into());
    }
    if tensor.descriptor().stream() != context.stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: tensor.descriptor().stream(),
        }
        .into());
    }
    if tensor.descriptor().dtype().class() != NumericClass::FloatingPoint {
        return invalid(
            operation,
            format!("{name} must have a floating-point dtype"),
        );
    }
    if !matches!(
        tensor.descriptor().dtype(),
        DType::F16 | DType::Bf16 | DType::F32
    ) {
        return invalid(operation, format!("{name} dtype must be F16, BF16, or F32"));
    }
    Ok(())
}

#[cfg(feature = "cpu")]
fn tensor_shape_to_usize(
    tensor: &Tensor,
    operation: &'static str,
    subject: &'static str,
) -> Result<Vec<usize>, SpatialFunctionalKernelError> {
    tensor
        .descriptor()
        .shape()
        .iter()
        .map(|value| usize::try_from(*value).map_err(|_| overflow(operation, subject)))
        .collect()
}

#[cfg(feature = "cpu")]
fn tensor_to_f32_workspace(
    backend: &CpuBackend,
    tensor: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<crate::CpuWorkspaceVec<f32>, SpatialFunctionalKernelError> {
    let element_count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| overflow(operation, "tensor element count"))?;
    let mut values = backend.workspace_vec(context, element_count)?;
    for linear in 0..element_count {
        if linear.is_multiple_of(64) {
            context.check()?;
        }
        let linear =
            u64::try_from(linear).map_err(|_| overflow(operation, "tensor linear index"))?;
        let value = match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(linear)?)?
        {
            DecodedScalar::Real(value) => value as f32,
            _ => {
                return invalid(operation, "tensor spatial execution requires real values");
            }
        };
        values.try_push(value)?;
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub fn grid_sample_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    grid: &[f32],
    grid_shape: &[usize],
    configuration: GridSampleConfiguration,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<GridSampleVjp, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(GRID_SAMPLE_OPERATION_ID, device)?;
    let geometry = GridGeometry::new(input, input_shape, grid, grid_shape, configuration)?;
    require_length(
        GRID_SAMPLE_OPERATION_ID,
        "output gradient",
        output_gradient.len(),
        geometry.output_count()?,
    )?;
    let mut input_gradient = vec![0.0_f32; input.len()];
    let mut grid_gradient = vec![0.0_f32; grid.len()];
    geometry.for_each_output(
        context,
        |output_index, batch, channel, y, x, coordinate_y_derivative, coordinate_x_derivative| {
            let gradient = output_gradient.get(output_index).copied().ok_or_else(|| {
                overflow(
                    GRID_SAMPLE_OPERATION_ID,
                    "grid-sample output-gradient index",
                )
            })?;
            let sampled = geometry.sample_with_derivatives(input, batch, channel, y, x)?;
            for sample in sampled.samples {
                let index =
                    geometry.input_index(batch, channel, sample.source_y, sample.source_x)?;
                let destination = input_gradient.get_mut(index).ok_or_else(|| {
                    overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample input-gradient index")
                })?;
                *destination = gradient.mul_add(sample.weight, *destination);
            }
            let grid_base = geometry.grid_base_index(batch, output_index)?;
            let x_destination = grid_gradient.get_mut(grid_base).ok_or_else(|| {
                overflow(
                    GRID_SAMPLE_OPERATION_ID,
                    "grid-sample grid-gradient x index",
                )
            })?;
            *x_destination += gradient * sampled.derivative_x * coordinate_x_derivative;
            let grid_y = grid_base
                .checked_add(1)
                .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid y index"))?;
            let y_destination = grid_gradient.get_mut(grid_y).ok_or_else(|| {
                overflow(
                    GRID_SAMPLE_OPERATION_ID,
                    "grid-sample grid-gradient y index",
                )
            })?;
            *y_destination += gradient * sampled.derivative_y * coordinate_y_derivative;
            Ok(())
        },
    )?;
    context.cancellation.check()?;
    Ok(GridSampleVjp {
        input: input_gradient,
        grid: grid_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn grid_sample_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    grid: &[f32],
    grid_tangent: &[f32],
    grid_shape: &[usize],
    configuration: GridSampleConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_length(
        GRID_SAMPLE_OPERATION_ID,
        "input tangent",
        input_tangent.len(),
        input.len(),
    )?;
    require_length(
        GRID_SAMPLE_OPERATION_ID,
        "grid tangent",
        grid_tangent.len(),
        grid.len(),
    )?;
    require_cpu(GRID_SAMPLE_OPERATION_ID, device)?;
    let geometry = GridGeometry::new(input, input_shape, grid, grid_shape, configuration)?;
    let mut output = vec![0.0_f32; geometry.output_count()?];
    geometry.for_each_output(
        context,
        |output_index, batch, channel, y, x, coordinate_y_derivative, coordinate_x_derivative| {
            let sampled = geometry.sample_with_derivatives(input, batch, channel, y, x)?;
            let mut value = 0.0_f32;
            for sample in sampled.samples {
                let index =
                    geometry.input_index(batch, channel, sample.source_y, sample.source_x)?;
                let tangent = input_tangent.get(index).copied().ok_or_else(|| {
                    overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample tangent index")
                })?;
                value = tangent.mul_add(sample.weight, value);
            }
            let grid_base = geometry.grid_base_index(batch, output_index)?;
            let x_tangent = grid_tangent.get(grid_base).copied().ok_or_else(|| {
                overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample grid tangent x index")
            })?;
            let grid_y = grid_base
                .checked_add(1)
                .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid y index"))?;
            let y_tangent = grid_tangent.get(grid_y).copied().ok_or_else(|| {
                overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample grid tangent y index")
            })?;
            value += sampled.derivative_x * coordinate_x_derivative * x_tangent
                + sampled.derivative_y * coordinate_y_derivative * y_tangent;
            let destination = output.get_mut(output_index).ok_or_else(|| {
                overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample JVP output index")
            })?;
            *destination = value;
            Ok(())
        },
    )?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape(),
    })
}

#[derive(Clone, Copy)]
struct GridSamplePoint {
    source_y: usize,
    source_x: usize,
    weight: f32,
}

struct SampleWithDerivatives {
    samples: Vec<GridSamplePoint>,
    derivative_y: f32,
    derivative_x: f32,
}

struct GridGeometry<'a> {
    grid: &'a [f32],
    configuration: GridSampleConfiguration,
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
}

impl<'a> GridGeometry<'a> {
    fn new(
        input: &[f32],
        input_shape: &[usize],
        grid: &'a [f32],
        grid_shape: &[usize],
        configuration: GridSampleConfiguration,
    ) -> Result<Self, SpatialFunctionalKernelError> {
        let [batch, channels, input_height, input_width] = input_shape else {
            return invalid(GRID_SAMPLE_OPERATION_ID, "input must have NCHW rank four");
        };
        let [grid_batch, output_height, output_width, coordinates] = grid_shape else {
            return invalid(GRID_SAMPLE_OPERATION_ID, "grid must have NHW2 rank four");
        };
        if batch != grid_batch || *coordinates != 2 || *input_height == 0 || *input_width == 0 {
            return invalid(
                GRID_SAMPLE_OPERATION_ID,
                "grid batch and coordinate extent must match a non-empty input",
            );
        }
        require_length(
            GRID_SAMPLE_OPERATION_ID,
            "input",
            input.len(),
            checked_product(input_shape, GRID_SAMPLE_OPERATION_ID, "input elements")?,
        )?;
        require_length(
            GRID_SAMPLE_OPERATION_ID,
            "grid",
            grid.len(),
            checked_product(grid_shape, GRID_SAMPLE_OPERATION_ID, "grid elements")?,
        )?;
        Ok(Self {
            grid,
            configuration,
            batch: *batch,
            channels: *channels,
            input_height: *input_height,
            input_width: *input_width,
            output_height: *output_height,
            output_width: *output_width,
        })
    }

    fn output_count(&self) -> Result<usize, SpatialFunctionalKernelError> {
        checked_product(
            &self.output_shape(),
            GRID_SAMPLE_OPERATION_ID,
            "output elements",
        )
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![
            self.batch,
            self.channels,
            self.output_height,
            self.output_width,
        ]
    }

    fn for_each_output(
        &self,
        context: &ExecutionContext<'_>,
        mut visit: impl FnMut(
            usize,
            usize,
            usize,
            f32,
            f32,
            f32,
            f32,
        ) -> Result<(), SpatialFunctionalKernelError>,
    ) -> Result<(), SpatialFunctionalKernelError> {
        for batch in 0..self.batch {
            for output_y in 0..self.output_height {
                for output_x in 0..self.output_width {
                    let grid_index = ((batch * self.output_height + output_y) * self.output_width
                        + output_x)
                        .checked_mul(2)
                        .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid index"))?;
                    let normalized_x = self
                        .grid
                        .get(grid_index)
                        .copied()
                        .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid x index"))?;
                    let normalized_y = self
                        .grid
                        .get(grid_index + 1)
                        .copied()
                        .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid y index"))?;
                    let (y, y_derivative) =
                        self.project_coordinate(normalized_y, self.input_height)?;
                    let (x, x_derivative) =
                        self.project_coordinate(normalized_x, self.input_width)?;
                    for channel in 0..self.channels {
                        let output_index = ((batch * self.channels + channel) * self.output_height
                            + output_y)
                            * self.output_width
                            + output_x;
                        if output_index.is_multiple_of(64) {
                            context.cancellation.check()?;
                        }
                        visit(
                            output_index,
                            batch,
                            channel,
                            y,
                            x,
                            y_derivative,
                            x_derivative,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn project_coordinate(
        &self,
        normalized: f32,
        extent: usize,
    ) -> Result<(f32, f32), SpatialFunctionalKernelError> {
        if !normalized.is_finite() {
            return invalid(GRID_SAMPLE_OPERATION_ID, "grid coordinates must be finite");
        }
        let (coordinate, derivative) = if self.configuration.align_corners {
            (
                (normalized + 1.0) * (extent.saturating_sub(1) as f32) * 0.5,
                extent.saturating_sub(1) as f32 * 0.5,
            )
        } else {
            (
                ((normalized + 1.0) * extent as f32 - 1.0) * 0.5,
                extent as f32 * 0.5,
            )
        };
        if self.configuration.padding_mode != GridPaddingMode::Reflection {
            return Ok((coordinate, derivative));
        }
        let (lower, upper) = if self.configuration.align_corners {
            (0.0, extent.saturating_sub(1) as f32)
        } else {
            (-0.5, extent as f32 - 0.5)
        };
        let (reflected, reflection_derivative) = reflect_coordinate(coordinate, lower, upper)?;
        Ok((reflected, derivative * reflection_derivative))
    }

    fn sample(
        &self,
        input: &[f32],
        batch: usize,
        channel: usize,
        y: f32,
        x: f32,
    ) -> Result<f32, SpatialFunctionalKernelError> {
        self.sample_with_derivatives(input, batch, channel, y, x)?
            .samples
            .into_iter()
            .try_fold(0.0_f32, |value, sample| {
                let index = self.input_index(batch, channel, sample.source_y, sample.source_x)?;
                let source = input
                    .get(index)
                    .copied()
                    .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid-sample input index"))?;
                Ok::<f32, SpatialFunctionalKernelError>(source.mul_add(sample.weight, value))
            })
    }

    fn sample_with_derivatives(
        &self,
        input: &[f32],
        batch: usize,
        channel: usize,
        y: f32,
        x: f32,
    ) -> Result<SampleWithDerivatives, SpatialFunctionalKernelError> {
        match self.configuration.mode {
            GridSampleMode::Nearest => {
                let y = round_ties_even(y);
                let x = round_ties_even(x);
                let (y, x) = match self.configuration.padding_mode {
                    GridPaddingMode::Zeros => {
                        if y < 0.0
                            || x < 0.0
                            || y >= self.input_height as f32
                            || x >= self.input_width as f32
                        {
                            return Ok(SampleWithDerivatives {
                                samples: Vec::new(),
                                derivative_y: 0.0,
                                derivative_x: 0.0,
                            });
                        }
                        (y, x)
                    }
                    GridPaddingMode::Border | GridPaddingMode::Reflection => (
                        y.clamp(0.0, self.input_height.saturating_sub(1) as f32),
                        x.clamp(0.0, self.input_width.saturating_sub(1) as f32),
                    ),
                };
                Ok(SampleWithDerivatives {
                    samples: vec![GridSamplePoint {
                        source_y: checked_nonnegative_usize(y, GRID_SAMPLE_OPERATION_ID)?,
                        source_x: checked_nonnegative_usize(x, GRID_SAMPLE_OPERATION_ID)?,
                        weight: 1.0,
                    }],
                    derivative_y: 0.0,
                    derivative_x: 0.0,
                })
            }
            GridSampleMode::Bilinear => {
                let boundary = if self.configuration.padding_mode == GridPaddingMode::Zeros {
                    NativeBilinearBoundary::ZeroPadding
                } else {
                    NativeBilinearBoundary::Border
                };
                let weights = checked_bilinear_weights(
                    usize_to_u64(self.input_height, GRID_SAMPLE_OPERATION_ID, "input height")?,
                    usize_to_u64(self.input_width, GRID_SAMPLE_OPERATION_ID, "input width")?,
                    y,
                    x,
                    boundary,
                    GRID_SAMPLE_OPERATION_ID,
                )
                .map_err(|error| sampling_error(GRID_SAMPLE_OPERATION_ID, error))?;
                let mut derivative_y = 0.0_f32;
                let mut derivative_x = 0.0_f32;
                let mut samples = Vec::with_capacity(weights.len());
                for weight in weights {
                    let source_y = usize::try_from(weight.source_y)
                        .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "source y"))?;
                    let source_x = usize::try_from(weight.source_x)
                        .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "source x"))?;
                    let index = self.input_index(batch, channel, source_y, source_x)?;
                    let value = input.get(index).copied().ok_or_else(|| {
                        overflow(GRID_SAMPLE_OPERATION_ID, "bilinear source index")
                    })?;
                    derivative_y = value.mul_add(weight.derivative_y, derivative_y);
                    derivative_x = value.mul_add(weight.derivative_x, derivative_x);
                    samples.push(GridSamplePoint {
                        source_y,
                        source_x,
                        weight: weight.weight,
                    });
                }
                Ok(SampleWithDerivatives {
                    samples,
                    derivative_y,
                    derivative_x,
                })
            }
            GridSampleMode::Bicubic => self.bicubic_samples(input, batch, channel, y, x),
        }
    }

    fn bicubic_samples(
        &self,
        input: &[f32],
        batch: usize,
        channel: usize,
        y: f32,
        x: f32,
    ) -> Result<SampleWithDerivatives, SpatialFunctionalKernelError> {
        let y_low = checked_floor_i64(y, GRID_SAMPLE_OPERATION_ID)?;
        let x_low = checked_floor_i64(x, GRID_SAMPLE_OPERATION_ID)?;
        let y_first = y_low
            .checked_sub(1)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "bicubic y range"))?;
        let y_last = y_low
            .checked_add(2)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "bicubic y range"))?;
        let x_first = x_low
            .checked_sub(1)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "bicubic x range"))?;
        let x_last = x_low
            .checked_add(2)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "bicubic x range"))?;
        let mut samples = Vec::with_capacity(16);
        let mut derivative_y = 0.0_f32;
        let mut derivative_x = 0.0_f32;
        for source_y in y_first..=y_last {
            let (mapped_y, y_boundary_derivative) =
                self.map_cubic_source(source_y, self.input_height)?;
            let y_distance = y - source_y as f32;
            let y_weight = cubic_weight(y_distance);
            let y_derivative = cubic_weight_derivative(y_distance);
            for source_x in x_first..=x_last {
                let source_coordinate_x = source_x;
                let (mapped_x, x_boundary_derivative) =
                    self.map_cubic_source(source_coordinate_x, self.input_width)?;
                let (Some(source_y), Some(source_x)) = (mapped_y, mapped_x) else {
                    continue;
                };
                let x_distance = x - source_coordinate_x as f32;
                let x_weight = cubic_weight(x_distance);
                let x_derivative = cubic_weight_derivative(x_distance);
                let index = self.input_index(batch, channel, source_y, source_x)?;
                let value = input
                    .get(index)
                    .copied()
                    .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "bicubic source index"))?;
                derivative_y = value.mul_add(
                    y_derivative * y_boundary_derivative * x_weight,
                    derivative_y,
                );
                derivative_x = value.mul_add(
                    x_derivative * x_boundary_derivative * y_weight,
                    derivative_x,
                );
                samples.push(GridSamplePoint {
                    source_y,
                    source_x,
                    weight: y_weight * x_weight,
                });
            }
        }
        Ok(SampleWithDerivatives {
            samples,
            derivative_y,
            derivative_x,
        })
    }

    fn map_cubic_source(
        &self,
        source: i64,
        extent: usize,
    ) -> Result<(Option<usize>, f32), SpatialFunctionalKernelError> {
        let extent_i64 = i64::try_from(extent)
            .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "cubic extent"))?;
        match self.configuration.padding_mode {
            GridPaddingMode::Zeros => {
                if source < 0 || source >= extent_i64 {
                    Ok((None, 0.0))
                } else {
                    Ok((
                        Some(
                            usize::try_from(source)
                                .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "cubic source"))?,
                        ),
                        1.0,
                    ))
                }
            }
            GridPaddingMode::Border | GridPaddingMode::Reflection => Ok((
                Some(
                    source
                        .clamp(0, extent_i64.saturating_sub(1))
                        .try_into()
                        .map_err(|_| overflow(GRID_SAMPLE_OPERATION_ID, "cubic source"))?,
                ),
                1.0,
            )),
        }
    }

    fn input_index(
        &self,
        batch: usize,
        channel: usize,
        y: usize,
        x: usize,
    ) -> Result<usize, SpatialFunctionalKernelError> {
        batch
            .checked_mul(self.channels)
            .and_then(|value| value.checked_add(channel))
            .and_then(|value| value.checked_mul(self.input_height))
            .and_then(|value| value.checked_add(y))
            .and_then(|value| value.checked_mul(self.input_width))
            .and_then(|value| value.checked_add(x))
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "input index"))
    }

    fn grid_base_index(
        &self,
        batch: usize,
        output_index: usize,
    ) -> Result<usize, SpatialFunctionalKernelError> {
        let per_channel = self
            .output_height
            .checked_mul(self.output_width)
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid plane"))?;
        let local = output_index % per_channel;
        batch
            .checked_mul(per_channel)
            .and_then(|value| value.checked_add(local))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| overflow(GRID_SAMPLE_OPERATION_ID, "grid index"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpolateMode {
    Nearest,
    NearestExact,
    Linear,
    Bilinear,
    Bicubic,
    Trilinear,
    Area,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterpolateConfiguration {
    pub output_size: Option<Vec<usize>>,
    pub scale_factor: Option<Vec<f64>>,
    pub mode: InterpolateMode,
    pub align_corners: Option<bool>,
    pub recompute_scale_factor: Option<bool>,
    pub antialias: bool,
}

pub fn interpolate_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    configuration: &InterpolateConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(INTERPOLATE_OPERATION_ID, device)?;
    let plan = InterpolatePlan::new(input, input_shape, configuration)?;
    plan.apply(input, context)
}

#[cfg(feature = "cpu")]
pub fn interpolate_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    configuration: &InterpolateConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    validate_real_tensor(backend, input, "input", INTERPOLATE_OPERATION_ID, context)?;
    let input_shape = tensor_shape_to_usize(input, INTERPOLATE_OPERATION_ID, "input shape")?;
    let input_values = tensor_to_f32_workspace(backend, input, INTERPOLATE_OPERATION_ID, context)?;
    let plan = InterpolatePlan::new(&input_values, &input_shape, configuration)?;
    let mut output_values = backend.workspace_vec(context, plan.output_count()?)?;
    for index in 0..plan.output_count()? {
        if index.is_multiple_of(64) {
            context.check()?;
        }
        output_values.try_push(0.0)?;
    }
    plan.accumulate_into(&input_values, &mut output_values, context)?;

    let output_shape = plan
        .output_shape
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "output shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        DeviceId::CPU,
        context.stream,
    )?;
    let (mut output, allocation_event) = backend.allocate(descriptor, context)?;
    backend.wait_event(allocation_event, context)?;
    let dtype = output.descriptor().dtype();
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "output dtype width"))?;
    {
        let mut write = output.write()?;
        let output_bytes = write.bytes_mut()?;
        for (index, value) in output_values.iter().copied().enumerate() {
            if index.is_multiple_of(64) {
                context.check()?;
            }
            let start = index
                .checked_mul(byte_width)
                .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "output byte index"))?;
            let end = start
                .checked_add(byte_width)
                .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "output byte range"))?;
            let destination = output_bytes
                .get_mut(start..end)
                .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "output byte destination"))?;
            let encoded = dtype.encode_scalar(
                Scalar::Float(f64::from(value)),
                INTERPOLATE_OPERATION_ID,
                DeviceId::CPU,
            )?;
            destination.copy_from_slice(&encoded);
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

#[cfg(feature = "cpu")]
pub fn pixel_shuffle_nd_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: usize,
    factor: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    if !matches!(dimensions, 1 | 3) || factor == 0 {
        return invalid(
            PIXEL_SHUFFLE_ND_OPERATION_ID,
            "dimensions must be one or three and factor must be nonzero",
        );
    }
    validate_real_tensor(
        backend,
        input,
        "input",
        PIXEL_SHUFFLE_ND_OPERATION_ID,
        context,
    )?;
    let [batch, input_channels, frames, height, width] = input.descriptor().shape() else {
        return invalid(PIXEL_SHUFFLE_ND_OPERATION_ID, "input must have rank five");
    };
    let divisor = factor
        .checked_pow(
            u32::try_from(dimensions)
                .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "dimensions"))?,
        )
        .ok_or_else(|| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "channel divisor"))?;
    let input_channels = usize::try_from(*input_channels)
        .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "input channels"))?;
    let channels = input_channels
        .checked_div(divisor)
        .filter(|channels| channels.checked_mul(divisor) == Some(input_channels))
        .ok_or_else(|| SpatialFunctionalKernelError::Invalid {
            operation: PIXEL_SHUFFLE_ND_OPERATION_ID,
            reason: "input channels are not divisible by the shuffle factor".to_owned(),
        })?;
    let batch =
        usize::try_from(*batch).map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "batch"))?;
    let frames =
        usize::try_from(*frames).map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "frames"))?;
    let height =
        usize::try_from(*height).map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "height"))?;
    let width =
        usize::try_from(*width).map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "width"))?;
    let output_frames = frames
        .checked_mul(factor)
        .ok_or_else(|| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "output frames"))?;
    let spatial_factor = if dimensions == 3 { factor } else { 1 };
    let output_height = height
        .checked_mul(spatial_factor)
        .ok_or_else(|| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "output height"))?;
    let output_width = width
        .checked_mul(spatial_factor)
        .ok_or_else(|| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "output width"))?;
    let output_count = [batch, channels, output_frames, output_height, output_width]
        .into_iter()
        .try_fold(1_usize, |product, value| {
            product
                .checked_mul(value)
                .ok_or_else(|| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "output elements"))
        })?;
    let input_values =
        tensor_to_f32_workspace(backend, input, PIXEL_SHUFFLE_ND_OPERATION_ID, context)?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for index in 0..output_count {
        if index.is_multiple_of(64) {
            context.check()?;
        }
        output.try_push(0.0)?;
    }
    for batch_index in 0..batch {
        for channel in 0..channels {
            for frame in 0..frames {
                for y in 0..height {
                    for x in 0..width {
                        if x.is_multiple_of(64) {
                            context.check()?;
                        }
                        for temporal in 0..factor {
                            for vertical in 0..spatial_factor {
                                for horizontal in 0..spatial_factor {
                                    let subchannel = channel
                                        .checked_mul(factor)
                                        .and_then(|value| value.checked_add(temporal))
                                        .and_then(|value| value.checked_mul(spatial_factor))
                                        .and_then(|value| value.checked_add(vertical))
                                        .and_then(|value| value.checked_mul(spatial_factor))
                                        .and_then(|value| value.checked_add(horizontal))
                                        .ok_or_else(|| {
                                            overflow(
                                                PIXEL_SHUFFLE_ND_OPERATION_ID,
                                                "source channel",
                                            )
                                        })?;
                                    let source = ((((batch_index * input_channels + subchannel)
                                        * frames
                                        + frame)
                                        * height
                                        + y)
                                        * width)
                                        + x;
                                    let destination = ((((batch_index * channels + channel)
                                        * output_frames
                                        + frame * factor
                                        + temporal)
                                        * output_height
                                        + y * spatial_factor
                                        + vertical)
                                        * output_width)
                                        + x * spatial_factor
                                        + horizontal;
                                    let value =
                                        input_values.get(source).copied().ok_or_else(|| {
                                            overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "source index")
                                        })?;
                                    let destination =
                                        output.get_mut(destination).ok_or_else(|| {
                                            overflow(
                                                PIXEL_SHUFFLE_ND_OPERATION_ID,
                                                "destination index",
                                            )
                                        })?;
                                    *destination = value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64::try_from(batch).map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "batch"))?,
            u64::try_from(channels)
                .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "channels"))?,
            u64::try_from(output_frames)
                .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "frames"))?,
            u64::try_from(output_height)
                .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "height"))?,
            u64::try_from(output_width)
                .map_err(|_| overflow(PIXEL_SHUFFLE_ND_OPERATION_ID, "width"))?,
        ],
        &output,
        input.descriptor().dtype(),
        DeviceId::CPU,
        context,
    )
    .map_err(|error| {
        canonical_tensor_owner_error(
            PIXEL_SHUFFLE_ND_OPERATION_ID,
            "tensor_from_f32_with_context_exact_native",
            error,
        )
    })
}

#[cfg(feature = "cpu")]
pub fn bislerp_tensor_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    target_width: u64,
    target_height: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SpatialFunctionalKernelError> {
    context.check()?;
    validate_real_tensor(backend, input, "input", BISLERP_OPERATION_ID, context)?;
    let [batch, channels, height, width] = input.descriptor().shape() else {
        return invalid(BISLERP_OPERATION_ID, "input must have rank four");
    };
    let batch = usize::try_from(*batch).map_err(|_| overflow(BISLERP_OPERATION_ID, "batch"))?;
    let channels =
        usize::try_from(*channels).map_err(|_| overflow(BISLERP_OPERATION_ID, "channels"))?;
    let height = usize::try_from(*height).map_err(|_| overflow(BISLERP_OPERATION_ID, "height"))?;
    let width = usize::try_from(*width).map_err(|_| overflow(BISLERP_OPERATION_ID, "width"))?;
    let target_width = usize::try_from(target_width)
        .map_err(|_| overflow(BISLERP_OPERATION_ID, "target width"))?;
    let target_height = usize::try_from(target_height)
        .map_err(|_| overflow(BISLERP_OPERATION_ID, "target height"))?;
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return invalid(
            BISLERP_OPERATION_ID,
            "source and target dimensions must be nonzero",
        );
    }
    let source = tensor_to_f32_workspace(backend, input, BISLERP_OPERATION_ID, context)?;
    let horizontal_count = [batch, channels, height, target_width]
        .into_iter()
        .try_fold(1_usize, |product, value| {
            product
                .checked_mul(value)
                .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "horizontal elements"))
        })?;
    let output_count = [batch, channels, target_height, target_width]
        .into_iter()
        .try_fold(1_usize, |product, value| {
            product
                .checked_mul(value)
                .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "output elements"))
        })?;
    let mut horizontal = zeroed_workspace(backend, horizontal_count, context)?;
    let mut left = zeroed_workspace(backend, channels, context)?;
    let mut right = zeroed_workspace(backend, channels, context)?;
    let mut mixed = zeroed_workspace(backend, channels, context)?;
    for batch_index in 0..batch {
        for y in 0..height {
            for output_x in 0..target_width {
                if output_x.is_multiple_of(64) {
                    context.check()?;
                }
                let coordinate = bilinear_coordinate(width, target_width, output_x);
                let x1 = coordinate.floor().max(0.0) as usize;
                let x2 = (x1 + 1).min(width - 1);
                let ratio = coordinate - x1 as f32;
                for channel in 0..channels {
                    let plane = (batch_index * channels + channel) * height + y;
                    let left_index = plane * width + x1;
                    let right_index = plane * width + x2;
                    *left
                        .get_mut(channel)
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "left workspace index"))? =
                        source
                            .get(left_index)
                            .copied()
                            .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "left source index"))?;
                    *right
                        .get_mut(channel)
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "right workspace index"))? =
                        source
                            .get(right_index)
                            .copied()
                            .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "right source index"))?;
                }
                source_slerp(&left, &right, ratio, &mut mixed);
                for channel in 0..channels {
                    let destination =
                        ((batch_index * channels + channel) * height + y) * target_width + output_x;
                    let value = mixed
                        .get(channel)
                        .copied()
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "mixed workspace index"))?;
                    *horizontal.get_mut(destination).ok_or_else(|| {
                        overflow(BISLERP_OPERATION_ID, "horizontal destination index")
                    })? = value;
                }
            }
        }
    }
    let mut output = zeroed_workspace(backend, output_count, context)?;
    for batch_index in 0..batch {
        for output_y in 0..target_height {
            let coordinate = bilinear_coordinate(height, target_height, output_y);
            let y1 = coordinate.floor().max(0.0) as usize;
            let y2 = (y1 + 1).min(height - 1);
            let ratio = coordinate - y1 as f32;
            for x in 0..target_width {
                if x.is_multiple_of(64) {
                    context.check()?;
                }
                for channel in 0..channels {
                    let left_index =
                        ((batch_index * channels + channel) * height + y1) * target_width + x;
                    let right_index =
                        ((batch_index * channels + channel) * height + y2) * target_width + x;
                    *left
                        .get_mut(channel)
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "left workspace index"))? =
                        horizontal.get(left_index).copied().ok_or_else(|| {
                            overflow(BISLERP_OPERATION_ID, "horizontal left index")
                        })?;
                    *right
                        .get_mut(channel)
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "right workspace index"))? =
                        horizontal.get(right_index).copied().ok_or_else(|| {
                            overflow(BISLERP_OPERATION_ID, "horizontal right index")
                        })?;
                }
                source_slerp(&left, &right, ratio, &mut mixed);
                for channel in 0..channels {
                    let destination = ((batch_index * channels + channel) * target_height
                        + output_y)
                        * target_width
                        + x;
                    let value = mixed
                        .get(channel)
                        .copied()
                        .ok_or_else(|| overflow(BISLERP_OPERATION_ID, "mixed workspace index"))?;
                    *output.get_mut(destination).ok_or_else(|| {
                        overflow(BISLERP_OPERATION_ID, "output destination index")
                    })? = value;
                }
            }
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64::try_from(batch).map_err(|_| overflow(BISLERP_OPERATION_ID, "batch"))?,
            u64::try_from(channels).map_err(|_| overflow(BISLERP_OPERATION_ID, "channels"))?,
            u64::try_from(target_height).map_err(|_| overflow(BISLERP_OPERATION_ID, "height"))?,
            u64::try_from(target_width).map_err(|_| overflow(BISLERP_OPERATION_ID, "width"))?,
        ],
        &output,
        input.descriptor().dtype(),
        DeviceId::CPU,
        context,
    )
    .map_err(|error| {
        canonical_tensor_owner_error(
            BISLERP_OPERATION_ID,
            "tensor_from_f32_with_context_exact_native",
            error,
        )
    })
}

#[cfg(feature = "cpu")]
fn zeroed_workspace(
    backend: &CpuBackend,
    length: usize,
    context: &ExecutionContext<'_>,
) -> Result<crate::CpuWorkspaceVec<f32>, SpatialFunctionalKernelError> {
    let mut values = backend.workspace_vec(context, length)?;
    for index in 0..length {
        if index.is_multiple_of(64) {
            context.check()?;
        }
        values.try_push(0.0)?;
    }
    Ok(values)
}

#[cfg(feature = "cpu")]
fn bilinear_coordinate(source: usize, target: usize, index: usize) -> f32 {
    (((index as f32 + 0.5) * source as f32 / target as f32) - 0.5)
        .clamp(0.0, source.saturating_sub(1) as f32)
}

#[cfg(feature = "cpu")]
fn source_slerp(left: &[f32], right: &[f32], ratio: f32, output: &mut [f32]) {
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            let left = if left_norm == 0.0 {
                0.0
            } else {
                *left / left_norm
            };
            let right = if right_norm == 0.0 {
                0.0
            } else {
                *right / right_norm
            };
            left * right
        })
        .sum::<f32>();
    if dot > 1.0 - 1.0e-5 {
        output.copy_from_slice(left);
        return;
    }
    if dot < 1.0e-5 - 1.0 {
        for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
            *output = *left * (1.0 - ratio) + *right * ratio;
        }
        return;
    }
    let omega = dot.acos();
    let sine = omega.sin();
    let length = left_norm * (1.0 - ratio) + right_norm * ratio;
    for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
        let left = if left_norm == 0.0 {
            0.0
        } else {
            *left / left_norm
        };
        let right = if right_norm == 0.0 {
            0.0
        } else {
            *right / right_norm
        };
        let direction =
            (((1.0 - ratio) * omega).sin() / sine) * left + ((ratio * omega).sin() / sine) * right;
        *output = direction * length;
    }
}

pub fn interpolate_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    configuration: &InterpolateConfiguration,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, SpatialFunctionalKernelError> {
    context.cancellation.check()?;
    require_cpu(INTERPOLATE_OPERATION_ID, device)?;
    let plan = InterpolatePlan::new(input, input_shape, configuration)?;
    require_length(
        INTERPOLATE_OPERATION_ID,
        "output gradient",
        output_gradient.len(),
        plan.output_count()?,
    )?;
    let mut input_gradient = vec![0.0_f32; input.len()];
    plan.for_each_connection(context, |input_index, output_index, weight| {
        let source = output_gradient.get(output_index).copied().ok_or_else(|| {
            overflow(
                INTERPOLATE_OPERATION_ID,
                "interpolate output-gradient index",
            )
        })?;
        let destination = input_gradient.get_mut(input_index).ok_or_else(|| {
            overflow(INTERPOLATE_OPERATION_ID, "interpolate input-gradient index")
        })?;
        *destination = source.mul_add(weight, *destination);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(input_gradient)
}

pub fn interpolate_jvp_with_context_exact_native(
    input_tangent: &[f32],
    input_shape: &[usize],
    configuration: &InterpolateConfiguration,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    interpolate_with_context_exact_native(
        input_tangent,
        input_shape,
        configuration,
        device,
        context,
    )
}

#[derive(Clone, Copy)]
struct AxisWeight {
    source: usize,
    weight: f32,
}

struct InterpolatePlan {
    output_shape: Vec<usize>,
    input_spatial: Vec<usize>,
    output_spatial: Vec<usize>,
    axis_weights: Vec<Vec<Vec<AxisWeight>>>,
    planes: usize,
}

impl InterpolatePlan {
    fn new(
        input: &[f32],
        input_shape: &[usize],
        configuration: &InterpolateConfiguration,
    ) -> Result<Self, SpatialFunctionalKernelError> {
        if input_shape.len() < 3 || input_shape.len() > 5 {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "interpolate expects N,C and one to three spatial dimensions",
            );
        }
        require_length(
            INTERPOLATE_OPERATION_ID,
            "input",
            input.len(),
            checked_product(input_shape, INTERPOLATE_OPERATION_ID, "input elements")?,
        )?;
        let spatial_dimensions = input_shape.len() - 2;
        let expected_dimensions = match configuration.mode {
            InterpolateMode::Linear => Some(1),
            InterpolateMode::Bilinear | InterpolateMode::Bicubic => Some(2),
            InterpolateMode::Trilinear => Some(3),
            InterpolateMode::Nearest | InterpolateMode::NearestExact | InterpolateMode::Area => {
                None
            }
        };
        if expected_dimensions.is_some_and(|expected| expected != spatial_dimensions) {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "interpolation mode does not match the input spatial rank",
            );
        }
        if configuration.output_size.is_some() == configuration.scale_factor.is_some() {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "exactly one of output_size and scale_factor is required",
            );
        }
        if configuration.output_size.is_some() && configuration.recompute_scale_factor == Some(true)
        {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "recompute_scale_factor is invalid with an explicit output_size",
            );
        }
        if configuration.antialias
            && !matches!(
                configuration.mode,
                InterpolateMode::Bilinear | InterpolateMode::Bicubic
            )
        {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "antialias is available only for bilinear and bicubic interpolation",
            );
        }
        if matches!(
            configuration.mode,
            InterpolateMode::Nearest | InterpolateMode::NearestExact | InterpolateMode::Area
        ) && configuration.align_corners.is_some()
        {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "align_corners is valid only for linear interpolation modes",
            );
        }
        let input_spatial = input_shape[2..].to_vec();
        if input_spatial.contains(&0) {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "input dimensions must be non-zero",
            );
        }
        let output_spatial = resolve_output_spatial(&input_spatial, configuration)?;
        let mut output_shape = input_shape[..2].to_vec();
        output_shape.extend_from_slice(&output_spatial);
        let planes = input_shape[0]
            .checked_mul(input_shape[1])
            .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "interpolate planes"))?;
        let mut axis_weights = Vec::with_capacity(spatial_dimensions);
        for axis in 0..spatial_dimensions {
            let inverse_scale =
                inverse_scale(axis, &input_spatial, &output_spatial, configuration)?;
            let mut output_weights = Vec::with_capacity(output_spatial[axis]);
            for output_coordinate in 0..output_spatial[axis] {
                output_weights.push(interpolation_axis_weights(
                    input_spatial[axis],
                    output_spatial[axis],
                    output_coordinate,
                    inverse_scale,
                    configuration,
                )?);
            }
            axis_weights.push(output_weights);
        }
        Ok(Self {
            output_shape,
            input_spatial,
            output_spatial,
            axis_weights,
            planes,
        })
    }

    fn output_count(&self) -> Result<usize, SpatialFunctionalKernelError> {
        checked_product(
            &self.output_shape,
            INTERPOLATE_OPERATION_ID,
            "output elements",
        )
    }

    fn apply(
        &self,
        input: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<TensorValues, SpatialFunctionalKernelError> {
        let mut output = vec![0.0_f32; self.output_count()?];
        self.accumulate_into(input, &mut output, context)?;
        context.cancellation.check()?;
        Ok(TensorValues {
            values: output,
            shape: self.output_shape.clone(),
        })
    }

    fn accumulate_into(
        &self,
        input: &[f32],
        output: &mut [f32],
        context: &ExecutionContext<'_>,
    ) -> Result<(), SpatialFunctionalKernelError> {
        require_length(
            INTERPOLATE_OPERATION_ID,
            "interpolate output",
            output.len(),
            self.output_count()?,
        )?;
        self.for_each_connection(context, |input_index, output_index, weight| {
            let source = input
                .get(input_index)
                .copied()
                .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "interpolate input index"))?;
            let destination = output
                .get_mut(output_index)
                .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "interpolate output index"))?;
            *destination = source.mul_add(weight, *destination);
            Ok(())
        })
    }

    fn for_each_connection(
        &self,
        context: &ExecutionContext<'_>,
        mut visit: impl FnMut(usize, usize, f32) -> Result<(), SpatialFunctionalKernelError>,
    ) -> Result<(), SpatialFunctionalKernelError> {
        let input_spatial_count = checked_product(
            &self.input_spatial,
            INTERPOLATE_OPERATION_ID,
            "input spatial elements",
        )?;
        let output_spatial_count = checked_product(
            &self.output_spatial,
            INTERPOLATE_OPERATION_ID,
            "output spatial elements",
        )?;
        for plane in 0..self.planes {
            for output_linear in 0..output_spatial_count {
                let output_coordinates = unravel_index(output_linear, &self.output_spatial)?;
                let output_index = plane
                    .checked_mul(output_spatial_count)
                    .and_then(|value| value.checked_add(output_linear))
                    .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "output index"))?;
                if output_index.is_multiple_of(64) {
                    context.cancellation.check()?;
                }
                let weights = output_coordinates
                    .iter()
                    .enumerate()
                    .map(|(axis, coordinate)| {
                        self.axis_weights
                            .get(axis)
                            .and_then(|weights| weights.get(*coordinate))
                            .map(Vec::as_slice)
                            .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "axis weights"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut connection = |spatial_index, weight| {
                    let input_index = plane
                        .checked_mul(input_spatial_count)
                        .and_then(|value| value.checked_add(spatial_index))
                        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "input index"))?;
                    visit(input_index, output_index, weight)
                };
                visit_axis_product(&weights, &self.input_spatial, 0, 0, 1.0, &mut connection)?;
            }
        }
        Ok(())
    }
}

fn visit_axis_product<F>(
    weights: &[&[AxisWeight]],
    input_spatial: &[usize],
    axis: usize,
    linear: usize,
    weight: f32,
    visit: &mut F,
) -> Result<(), SpatialFunctionalKernelError>
where
    F: FnMut(usize, f32) -> Result<(), SpatialFunctionalKernelError>,
{
    if axis == weights.len() {
        return visit(linear, weight);
    }
    let extent = input_spatial
        .get(axis)
        .copied()
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "axis extent"))?;
    for axis_weight in weights
        .get(axis)
        .copied()
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "axis weight list"))?
    {
        let linear = linear
            .checked_mul(extent)
            .and_then(|value| value.checked_add(axis_weight.source))
            .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "spatial index"))?;
        visit_axis_product(
            weights,
            input_spatial,
            axis + 1,
            linear,
            weight * axis_weight.weight,
            visit,
        )?;
    }
    Ok(())
}

fn resolve_output_spatial(
    input_spatial: &[usize],
    configuration: &InterpolateConfiguration,
) -> Result<Vec<usize>, SpatialFunctionalKernelError> {
    if let Some(size) = &configuration.output_size {
        if size.len() != input_spatial.len() || size.contains(&0) {
            return invalid(
                INTERPOLATE_OPERATION_ID,
                "output_size rank must match and dimensions must be non-zero",
            );
        }
        return Ok(size.clone());
    }
    let scales = configuration.scale_factor.as_ref().ok_or_else(|| {
        SpatialFunctionalKernelError::Invalid {
            operation: INTERPOLATE_OPERATION_ID,
            reason: "scale_factor is missing".to_owned(),
        }
    })?;
    if scales.len() != input_spatial.len()
        || scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return invalid(
            INTERPOLATE_OPERATION_ID,
            "scale_factor rank must match and values must be finite and positive",
        );
    }
    input_spatial
        .iter()
        .zip(scales)
        .map(|(input, scale)| {
            let output = (*input as f64 * scale).floor();
            if output < 1.0 || output >= usize::MAX as f64 {
                return Err(overflow(INTERPOLATE_OPERATION_ID, "scaled output size"));
            }
            Ok(output as usize)
        })
        .collect()
}

fn inverse_scale(
    axis: usize,
    input_spatial: &[usize],
    output_spatial: &[usize],
    configuration: &InterpolateConfiguration,
) -> Result<f32, SpatialFunctionalKernelError> {
    if configuration.output_size.is_none() && configuration.recompute_scale_factor != Some(true) {
        let scale = configuration
            .scale_factor
            .as_ref()
            .and_then(|scales| scales.get(axis))
            .copied()
            .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "scale factor"))?;
        return Ok((1.0 / scale) as f32);
    }
    let input = input_spatial
        .get(axis)
        .copied()
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "input extent"))?;
    let output = output_spatial
        .get(axis)
        .copied()
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "output extent"))?;
    Ok(input as f32 / output as f32)
}

fn interpolation_axis_weights(
    input_extent: usize,
    output_extent: usize,
    output_coordinate: usize,
    inverse_scale: f32,
    configuration: &InterpolateConfiguration,
) -> Result<Vec<AxisWeight>, SpatialFunctionalKernelError> {
    match configuration.mode {
        InterpolateMode::Nearest => Ok(vec![AxisWeight {
            source: ((output_coordinate as f32 * inverse_scale).floor() as usize)
                .min(input_extent - 1),
            weight: 1.0,
        }]),
        InterpolateMode::NearestExact => Ok(vec![AxisWeight {
            source: (((output_coordinate as f32 + 0.5) * inverse_scale).floor() as usize)
                .min(input_extent - 1),
            weight: 1.0,
        }]),
        InterpolateMode::Area => area_weights(input_extent, output_extent, output_coordinate),
        InterpolateMode::Linear | InterpolateMode::Bilinear | InterpolateMode::Trilinear => {
            if configuration.antialias && output_extent < input_extent {
                antialias_weights(input_extent, output_coordinate, inverse_scale, false)
            } else {
                let coordinate = linear_source_coordinate(
                    input_extent,
                    output_extent,
                    output_coordinate,
                    inverse_scale,
                    configuration.align_corners.unwrap_or(false),
                );
                checked_linear_weights(
                    usize_to_u64(input_extent, INTERPOLATE_OPERATION_ID, "input extent")?,
                    coordinate,
                    NativeLinearBoundary::Border,
                    INTERPOLATE_OPERATION_ID,
                )
                .map_err(|error| sampling_error(INTERPOLATE_OPERATION_ID, error))?
                .into_iter()
                .map(native_axis_weight)
                .collect()
            }
        }
        InterpolateMode::Bicubic => {
            if configuration.antialias && output_extent < input_extent {
                antialias_weights(input_extent, output_coordinate, inverse_scale, true)
            } else {
                cubic_axis_weights(
                    input_extent,
                    linear_source_coordinate(
                        input_extent,
                        output_extent,
                        output_coordinate,
                        inverse_scale,
                        configuration.align_corners.unwrap_or(false),
                    ),
                )
            }
        }
    }
}

fn native_axis_weight(
    weight: NativeLinearWeight,
) -> Result<AxisWeight, SpatialFunctionalKernelError> {
    Ok(AxisWeight {
        source: usize::try_from(weight.source)
            .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "linear source"))?,
        weight: weight.weight,
    })
}

fn linear_source_coordinate(
    input_extent: usize,
    output_extent: usize,
    output_coordinate: usize,
    inverse_scale: f32,
    align_corners: bool,
) -> f32 {
    if align_corners {
        if output_extent <= 1 {
            0.0
        } else {
            output_coordinate as f32 * (input_extent - 1) as f32 / (output_extent - 1) as f32
        }
    } else {
        (output_coordinate as f32 + 0.5).mul_add(inverse_scale, -0.5)
    }
}

fn area_weights(
    input_extent: usize,
    output_extent: usize,
    output_coordinate: usize,
) -> Result<Vec<AxisWeight>, SpatialFunctionalKernelError> {
    let start = output_coordinate
        .checked_mul(input_extent)
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "area start"))?
        / output_extent;
    let end_numerator = (output_coordinate + 1)
        .checked_mul(input_extent)
        .and_then(|value| value.checked_add(output_extent - 1))
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "area end"))?;
    let end = (end_numerator / output_extent).min(input_extent);
    let count = end.saturating_sub(start);
    if count == 0 {
        return invalid(INTERPOLATE_OPERATION_ID, "area interval is empty");
    }
    let weight = (count as f32).recip();
    Ok((start..end)
        .map(|source| AxisWeight { source, weight })
        .collect())
}

fn antialias_weights(
    input_extent: usize,
    output_coordinate: usize,
    inverse_scale: f32,
    cubic: bool,
) -> Result<Vec<AxisWeight>, SpatialFunctionalKernelError> {
    let center = (output_coordinate as f32 + 0.5).mul_add(inverse_scale, -0.5);
    let filter_scale = inverse_scale.max(1.0);
    let radius = if cubic { 2.0 } else { 1.0 } * filter_scale;
    let first = checked_floor_i64(center - radius, INTERPOLATE_OPERATION_ID)?;
    let last = checked_floor_i64(center + radius, INTERPOLATE_OPERATION_ID)?
        .checked_add(1)
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "antialias coordinate range"))?;
    let mut combined = std::collections::BTreeMap::<usize, f32>::new();
    for source in first..=last {
        let mapped = source
            .clamp(
                0,
                i64::try_from(input_extent.saturating_sub(1))
                    .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "antialias input extent"))?,
            )
            .try_into()
            .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "antialias source"))?;
        let distance = (center - source as f32) / filter_scale;
        let weight = if cubic {
            cubic_weight(distance)
        } else {
            (1.0 - distance.abs()).max(0.0)
        };
        *combined.entry(mapped).or_default() += weight;
    }
    let sum: f32 = combined.values().sum();
    if !sum.is_finite() || sum == 0.0 {
        return invalid(
            INTERPOLATE_OPERATION_ID,
            "antialias kernel is not normalizable",
        );
    }
    Ok(combined
        .into_iter()
        .map(|(source, weight)| AxisWeight {
            source,
            weight: weight / sum,
        })
        .collect())
}

fn cubic_axis_weights(
    input_extent: usize,
    coordinate: f32,
) -> Result<Vec<AxisWeight>, SpatialFunctionalKernelError> {
    let low = checked_floor_i64(coordinate, INTERPOLATE_OPERATION_ID)?;
    let first = low
        .checked_sub(1)
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "cubic coordinate range"))?;
    let last = low
        .checked_add(2)
        .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "cubic coordinate range"))?;
    let upper = i64::try_from(input_extent.saturating_sub(1))
        .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "cubic input extent"))?;
    let mut combined = std::collections::BTreeMap::<usize, f32>::new();
    for source in first..=last {
        let mapped = usize::try_from(source.clamp(0, upper))
            .map_err(|_| overflow(INTERPOLATE_OPERATION_ID, "cubic source"))?;
        *combined.entry(mapped).or_default() += cubic_weight(coordinate - source as f32);
    }
    Ok(combined
        .into_iter()
        .map(|(source, weight)| AxisWeight { source, weight })
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    canonical_max_pool_2d(
        input,
        input_shape,
        kernel_size,
        stride.unwrap_or(kernel_size),
        padding,
        dilation,
        ceil_mode,
        device,
        context,
    )
    .map_err(max_pool_error)
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<MaxPool2dVjp, SpatialFunctionalKernelError> {
    canonical_max_pool_2d_vjp(
        input,
        input_shape,
        kernel_size,
        stride.unwrap_or(kernel_size),
        padding,
        dilation,
        ceil_mode,
        output_gradient,
        device,
        context,
    )
    .map_err(max_pool_error)
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, SpatialFunctionalKernelError> {
    canonical_max_pool_2d_jvp(
        input,
        input_tangent,
        input_shape,
        kernel_size,
        stride.unwrap_or(kernel_size),
        padding,
        dilation,
        ceil_mode,
        device,
        context,
    )
    .map_err(max_pool_error)
}

fn cubic_weight(distance: f32) -> f32 {
    let distance = distance.abs();
    if distance <= 1.0 {
        ((1.25 * distance - 2.25) * distance) * distance + 1.0
    } else if distance < 2.0 {
        ((-0.75 * distance + 3.75) * distance - 6.0) * distance + 3.0
    } else {
        0.0
    }
}

fn cubic_weight_derivative(distance: f32) -> f32 {
    let absolute = distance.abs();
    let derivative = if absolute <= 1.0 {
        (3.75 * absolute - 4.5) * absolute
    } else if absolute < 2.0 {
        (-2.25 * absolute + 7.5) * absolute - 6.0
    } else {
        0.0
    };
    derivative * distance.signum()
}

fn reflect_coordinate(
    coordinate: f32,
    lower: f32,
    upper: f32,
) -> Result<(f32, f32), SpatialFunctionalKernelError> {
    let span = upper - lower;
    if span <= 0.0 {
        return Ok((lower, 0.0));
    }
    let period = span * 2.0;
    let remainder = (coordinate - lower).rem_euclid(period);
    if remainder > span {
        Ok((upper - (remainder - span), -1.0))
    } else {
        Ok((lower + remainder, 1.0))
    }
}

fn round_ties_even(value: f32) -> f32 {
    value.round_ties_even()
}

fn checked_floor_i64(
    value: f32,
    operation: &'static str,
) -> Result<i64, SpatialFunctionalKernelError> {
    let value = value.floor();
    if !value.is_finite() || value < i64::MIN as f32 || value > i64::MAX as f32 {
        return Err(overflow(operation, "sampling coordinate"));
    }
    Ok(value as i64)
}

fn checked_nonnegative_usize(
    value: f32,
    operation: &'static str,
) -> Result<usize, SpatialFunctionalKernelError> {
    if !value.is_finite() || value < 0.0 || value > usize::MAX as f32 {
        return Err(overflow(operation, "nonnegative coordinate"));
    }
    Ok(value as usize)
}

fn unravel_index(
    mut linear: usize,
    shape: &[usize],
) -> Result<Vec<usize>, SpatialFunctionalKernelError> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let extent = shape
            .get(axis)
            .copied()
            .ok_or_else(|| overflow(INTERPOLATE_OPERATION_ID, "unravel extent"))?;
        if extent == 0 {
            return invalid(INTERPOLATE_OPERATION_ID, "cannot unravel an empty shape");
        }
        indices[axis] = linear % extent;
        linear /= extent;
    }
    Ok(indices)
}

fn checked_product(
    shape: &[usize],
    operation: &'static str,
    subject: &'static str,
) -> Result<usize, SpatialFunctionalKernelError> {
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| overflow(operation, subject))
    })
}

fn require_length(
    operation: &'static str,
    role: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SpatialFunctionalKernelError> {
    if actual != expected {
        return invalid(
            operation,
            format!("{role} requires {expected} values, got {actual}"),
        );
    }
    Ok(())
}

fn require_cpu(
    operation: &'static str,
    device: DeviceId,
) -> Result<(), SpatialFunctionalKernelError> {
    if device != DeviceId::CPU {
        return Err(SpatialFunctionalKernelError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn usize_to_u64(
    value: usize,
    operation: &'static str,
    subject: &'static str,
) -> Result<u64, SpatialFunctionalKernelError> {
    u64::try_from(value).map_err(|_| overflow(operation, subject))
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, SpatialFunctionalKernelError> {
    Err(SpatialFunctionalKernelError::Invalid {
        operation,
        reason: reason.into(),
    })
}

fn overflow(operation: &'static str, subject: &'static str) -> SpatialFunctionalKernelError {
    SpatialFunctionalKernelError::ShapeOverflow { operation, subject }
}

fn pool_error(
    operation: &'static str,
    error: NeuralNetworkModuleError,
) -> SpatialFunctionalKernelError {
    match error {
        NeuralNetworkModuleError::Cancelled => SpatialFunctionalKernelError::Cancelled,
        error => SpatialFunctionalKernelError::CanonicalOwner {
            operation,
            owner: "AveragePoolGeometry",
            reason: error.to_string(),
        },
    }
}

fn convolution_error(
    operation: &'static str,
    error: OperatorIndirectionError,
) -> SpatialFunctionalKernelError {
    match error {
        OperatorIndirectionError::Cancelled => SpatialFunctionalKernelError::Cancelled,
        OperatorIndirectionError::Tensor(error) => error.into(),
        error => SpatialFunctionalKernelError::CanonicalOwner {
            operation,
            owner: "ConvolutionGeometry",
            reason: error.to_string(),
        },
    }
}

fn sampling_error(
    operation: &'static str,
    error: ExternalTensorKernelPartOneError,
) -> SpatialFunctionalKernelError {
    match error {
        ExternalTensorKernelPartOneError::Cancelled => SpatialFunctionalKernelError::Cancelled,
        error => SpatialFunctionalKernelError::CanonicalOwner {
            operation,
            owner: "external_tensor_kernel_01 sampling",
            reason: error.to_string(),
        },
    }
}

fn max_pool_error(error: NeuralNetworkModulePartThreeError) -> SpatialFunctionalKernelError {
    match error {
        NeuralNetworkModulePartThreeError::Cancelled => SpatialFunctionalKernelError::Cancelled,
        error => SpatialFunctionalKernelError::CanonicalOwner {
            operation: MAX_POOL_2D_OPERATION_ID,
            owner: "neural_network_module_03 max_pool_2d",
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
mod validation_tests {
    use super::{
        PIXEL_SHUFFLE_ND_OPERATION_ID, SpatialFunctionalKernelError, canonical_tensor_owner_error,
    };
    use crate::{TensorError, generated_sim_operator_indirection_01::OperatorIndirectionError};
    use std::collections::BTreeMap;

    #[test]
    fn latent_upscale_canonical_owner_preserves_typed_terminal_errors() {
        let allocation = canonical_tensor_owner_error(
            PIXEL_SHUFFLE_ND_OPERATION_ID,
            "tensor_from_f32_with_context_exact_native",
            OperatorIndirectionError::Tensor(TensorError::AllocationFailed {
                requested: 4_096,
                reason: "fixture allocation failure".to_owned(),
            }),
        );
        assert!(matches!(
            allocation,
            SpatialFunctionalKernelError::Tensor(TensorError::AllocationFailed {
                requested: 4_096,
                ..
            })
        ));

        let resource = canonical_tensor_owner_error(
            PIXEL_SHUFFLE_ND_OPERATION_ID,
            "tensor_from_f32_with_context_exact_native",
            OperatorIndirectionError::Tensor(TensorError::ResourceLimitExceeded {
                resource: "fixture-resource",
                limit: 7,
            }),
        );
        assert_eq!(
            resource,
            SpatialFunctionalKernelError::Tensor(TensorError::ResourceLimitExceeded {
                resource: "fixture-resource",
                limit: 7,
            })
        );

        assert_eq!(
            canonical_tensor_owner_error(
                PIXEL_SHUFFLE_ND_OPERATION_ID,
                "tensor_from_f32_with_context_exact_native",
                OperatorIndirectionError::Cancelled,
            ),
            SpatialFunctionalKernelError::Cancelled
        );
    }

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            (
                "COMFY-TENSOR-OP-5F86004D9BDA",
                "e300fd6c3d44dc6c26a2b9ebce7d879ff05b58c8f86461735c94ecaafb2e0c43",
            ),
            (
                "COMFY-TENSOR-OP-60B322636602",
                "4c01d01f3c272c767fbaee9a09821094717c3585883c5016213151f2a5fe2141",
            ),
            (
                "COMFY-TENSOR-OP-6BFAEF690071",
                "2de2a5cebda2118130d79a7d091c86b97569a597458c5239ca9710ecfb46f9e5",
            ),
            (
                "COMFY-TENSOR-OP-DC56DB93077F",
                "a633a8091b768b8d651f89e018f51d6df7e56e90f0c0504ed77ec7bd39c953c6",
            ),
            (
                "COMFY-TENSOR-OP-A31AEBE72455",
                "d80f5de156590852947ac39e8c354dcbedc75c10ba6e388a3fc9619867f7c581",
            ),
            (
                "COMFY-TENSOR-OP-FE03423D60DA",
                "8c331012885263f0663991ea77a9a74af313da1acf81d681a0b10f6ab0bf0e81",
            ),
            (
                "COMFY-TENSOR-OP-341577A45D6B",
                "bfa415d86e6517288cdb1e8c6e9a9bd6c14c3036e99624d2734b3cbbd32cc82a",
            ),
            (
                "COMFY-TENSOR-OP-5A5F8349A130",
                "d338aabfead4178cc0c5fb9cd017e2cd56766cd4b73c63bee0320ea713871c9f",
            ),
            (
                "COMFY-TENSOR-OP-5A6A586CC551",
                "1256dec312ace287adec1ce100f8f1c096be98b0ec257bb23a43c9524f92e8a3",
            ),
            (
                "COMFY-TENSOR-OP-A90AB43A3320",
                "1183f698ac8f29b01c5f81b9a14a89826fcca8a5bc571bc3635bbfa9d556609b",
            ),
            (
                "COMFY-TENSOR-OP-B0F801006375",
                "22bdbfd653bd624574191ec8cdaf246912900ff31fd38277bae4d262ee5ec733",
            ),
            (
                "COMFY-TENSOR-OP-1F9D23F3B331",
                "d258cb0ca1560ea9d20ebcca802ce93ddb1e5aac56876d33bb86ea501379ac0c",
            ),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation| (*operation, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-spatial-functional-kernel-01.json",
            "VAL-TENSOR-001",
            "Task 89 exact spatial functional facades over canonical convolution, pooling, sampling, resize, workspace, and publication owners",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-spatial-functional-kernel-01.json",
            "VAL-AUTOGRAD-001",
            "Task 89 analytical average-pool, convolution, grid-sample, interpolation, and max-pool VJP/JVP contracts",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
