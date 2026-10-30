use crate::{
    CancellationToken, ConvolutionSpec, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_accelerated_attention_kernel_01::{
        AttentionKernelError, AttentionKernelRequest, AttentionMask, AttentionVjp,
        CheckedAttentionInvocation,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const CAST_BIAS_WEIGHT_OPERATION_ID: &str = "COMFY-TENSOR-OP-97205767AA40";
pub const CAST_MODULES_WITH_VBAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-23DA7F686728";
pub const CAST_TO_OPERATION_ID: &str = "COMFY-TENSOR-OP-56B106D5BEE7";
pub const CONV1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-A553C4928CA6";
pub const CONV2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-4B62764DCD01";
pub const CONV3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6CF91D19480B";
pub const CONV_TRANSPOSE1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-227F5D04687A";
pub const CONV_TRANSPOSE2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6F126397E86F";
pub const LINEAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-5D8E418C8374";
pub const MANUAL_CAST_LAYER_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-7ADDDB2261D6";
pub const MIXED_PRECISION_OPS_OPERATION_ID: &str = "COMFY-TENSOR-OP-4C30712EC2F7";
pub const SCALED_DOT_PRODUCT_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-5EAFEF13DE9D";
pub const UNCAST_BIAS_WEIGHT_OPERATION_ID: &str = "COMFY-TENSOR-OP-86BEA4A2DC25";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OperatorIndirectionError {
    #[error(transparent)]
    Tensor(TensorError),
    #[error(transparent)]
    Attention(AttentionKernelError),
    #[error("operator-indirection execution was cancelled")]
    Cancelled,
    #[error("operator-indirection shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("operator-indirection value count for {name} must be {expected}, got {actual}")]
    ValueCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("operator-indirection parameter is invalid: {0}")]
    Invalid(&'static str),
    #[error("operator-indirection device {device:?} is unsupported for {operation}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
}

impl From<sim_types::CancellationError> for OperatorIndirectionError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<TensorError> for OperatorIndirectionError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl From<AttentionKernelError> for OperatorIndirectionError {
    fn from(error: AttentionKernelError) -> Self {
        match error {
            AttentionKernelError::Cancelled => Self::Cancelled,
            AttentionKernelError::Tensor(error) => error.into(),
            AttentionKernelError::UnsupportedDevice { device, .. } => Self::UnsupportedDevice {
                operation: SCALED_DOT_PRODUCT_ATTENTION_OPERATION_ID,
                device,
            },
            error => Self::Attention(error),
        }
    }
}

pub fn cast_to_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: DType,
    device: DeviceId,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    cast_to_impl(backend, input, dtype, device, non_blocking, copy, context)
}

pub fn cast_to_with_backend_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    dtype: DType,
    device: DeviceId,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    cast_to_impl(backend, input, dtype, device, non_blocking, copy, context)
}

#[allow(clippy::too_many_arguments)]
fn cast_to_impl(
    backend: &dyn TensorBackend,
    input: &Tensor,
    dtype: DType,
    device: DeviceId,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    context.cancellation.check()?;
    if backend.device() != device {
        return Err(OperatorIndirectionError::UnsupportedDevice {
            operation: CAST_TO_OPERATION_ID,
            device,
        });
    }
    require_cpu(CAST_TO_OPERATION_ID, device)?;
    require_cpu(CAST_TO_OPERATION_ID, input.descriptor().device())?;
    if input.descriptor().dtype() == dtype && input.descriptor().device() == device && !copy {
        return Ok(input.clone());
    }
    let stream = input.descriptor().stream();
    if input.descriptor().dtype() == dtype && input.descriptor().device() == device {
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            dtype,
            device,
            stream,
        )?;
        let (output, event) = backend.copy(input, descriptor, context)?;
        if non_blocking {
            backend.wait_event(event, context)?;
        }
        context.cancellation.check()?;
        return Ok(output);
    }

    let element_count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast element count"))?;
    let target_width = usize::try_from(dtype.byte_width())
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast target width"))?;
    let capacity = element_count
        .checked_mul(target_width)
        .ok_or(OperatorIndirectionError::ShapeOverflow("cast output bytes"))?;
    let requested = u64::try_from(capacity)
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast output bytes"))?;
    let _workspace = backend.reserve_workspace(context, requested)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast output bytes"))?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let decoded = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        output.extend(encode_decoded(dtype, decoded, device)?);
    }
    let descriptor =
        TensorDescriptor::contiguous(input.descriptor().shape().to_vec(), dtype, device, stream)?;
    let (mut output_tensor, allocation_event) = backend.allocate(descriptor, context)?;
    backend.wait_event(allocation_event, context)?;
    output_tensor.write()?.storage_bytes_mut()?.copy_from_slice(&output);
    let event = backend.record_event(context)?;
    if non_blocking {
        backend.wait_event(event, context)?;
    }
    context.cancellation.check()?;
    Ok(output_tensor)
}

pub fn tensor_to_f32_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, OperatorIndirectionError> {
    tensor_to_f32_impl(backend, input, context)
}

pub fn tensor_to_f32_with_backend_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, OperatorIndirectionError> {
    tensor_to_f32_impl(backend, input, context)
}

fn tensor_to_f32_impl(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, OperatorIndirectionError> {
    let copy = !input.descriptor().is_contiguous()?;
    let converted = cast_to_impl(
        backend,
        input,
        DType::F32,
        input.descriptor().device(),
        false,
        copy,
        context,
    )?;
    let bytes = converted.contiguous_bytes()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / std::mem::size_of::<f32>())
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("decoded f32 values"))?;
    for (index, encoded) in bytes.chunks_exact(std::mem::size_of::<f32>()).enumerate() {
        check_periodically(index, context.cancellation)?;
        let encoded: [u8; 4] = encoded
            .try_into()
            .map_err(|_| OperatorIndirectionError::Invalid("unaligned f32 tensor bytes"))?;
        values.push(f32::from_ne_bytes(encoded));
    }
    Ok(values)
}

pub fn tensor_from_f32_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    tensor_from_f32_impl(backend, shape, values, dtype, device, context)
}

pub fn tensor_from_f32_with_backend_exact_native(
    backend: &dyn TensorBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    tensor_from_f32_impl(backend, shape, values, dtype, device, context)
}

#[allow(clippy::too_many_arguments)]
fn tensor_from_f32_impl(
    backend: &dyn TensorBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    require_cpu(CAST_TO_OPERATION_ID, device)?;
    if backend.device() != device {
        return Err(OperatorIndirectionError::UnsupportedDevice {
            operation: CAST_TO_OPERATION_ID,
            device,
        });
    }
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), DType::F32, device, context.stream)?;
    let expected = usize::try_from(descriptor.element_count()?)
        .map_err(|_| OperatorIndirectionError::ShapeOverflow("f32 tensor element count"))?;
    if values.len() != expected {
        return Err(OperatorIndirectionError::ValueCount {
            name: "f32 tensor values",
            expected,
            actual: values.len(),
        });
    }
    let (mut tensor, allocation_event) = backend.allocate(descriptor, context)?;
    backend.wait_event(allocation_event, context)?;
    {
        let mut write = tensor.write()?;
        for (destination, value) in write
            .bytes_mut()?
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(values)
        {
            context.check()?;
            destination.copy_from_slice(&value.to_ne_bytes());
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    cast_to_impl(
        backend,
        &tensor,
        dtype,
        device,
        false,
        false,
        context,
    )
}

fn encode_decoded(
    dtype: DType,
    decoded: DecodedScalar,
    device: DeviceId,
) -> Result<Vec<u8>, TensorError> {
    dtype.encode_decoded_scalar(decoded, CAST_TO_OPERATION_ID, device)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConvolutionPaddingMode {
    #[default]
    Zeros,
    Reflect,
    Replicate,
    Circular,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvolutionGeometry {
    spatial_dimensions: usize,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    transposed: bool,
    output_padding: Vec<usize>,
    padding_mode: ConvolutionPaddingMode,
}

impl ConvolutionGeometry {
    pub fn new(
        spatial_dimensions: usize,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        transposed: bool,
        output_padding: Vec<usize>,
    ) -> Result<Self, OperatorIndirectionError> {
        Self::new_with_padding_mode(
            spatial_dimensions,
            stride,
            padding,
            dilation,
            groups,
            transposed,
            output_padding,
            ConvolutionPaddingMode::Zeros,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_padding_mode(
        spatial_dimensions: usize,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        transposed: bool,
        output_padding: Vec<usize>,
        padding_mode: ConvolutionPaddingMode,
    ) -> Result<Self, OperatorIndirectionError> {
        if !(1..=3).contains(&spatial_dimensions) {
            return Err(OperatorIndirectionError::Invalid(
                "convolution supports one, two, or three spatial dimensions",
            ));
        }
        for (name, values) in [
            ("stride", stride.as_slice()),
            ("padding", padding.as_slice()),
            ("dilation", dilation.as_slice()),
            ("output padding", output_padding.as_slice()),
        ] {
            if values.len() != spatial_dimensions {
                return Err(OperatorIndirectionError::Invalid(match name {
                    "stride" => "convolution stride rank mismatch",
                    "padding" => "convolution padding rank mismatch",
                    "dilation" => "convolution dilation rank mismatch",
                    _ => "convolution output-padding rank mismatch",
                }));
            }
        }
        if groups == 0 || stride.contains(&0) || dilation.contains(&0) {
            return Err(OperatorIndirectionError::Invalid(
                "convolution groups, stride, and dilation must be nonzero",
            ));
        }
        if !transposed && output_padding.iter().any(|value| *value != 0) {
            return Err(OperatorIndirectionError::Invalid(
                "ordinary convolution does not accept output padding",
            ));
        }
        if transposed && padding_mode != ConvolutionPaddingMode::Zeros {
            return Err(OperatorIndirectionError::Invalid(
                "transposed convolution accepts only zero padding",
            ));
        }
        if transposed
            && output_padding
                .iter()
                .zip(&stride)
                .any(|(output_padding, stride)| output_padding >= stride)
        {
            return Err(OperatorIndirectionError::Invalid(
                "transposed-convolution output padding must be smaller than stride",
            ));
        }
        Ok(Self {
            spatial_dimensions,
            stride,
            padding,
            dilation,
            groups,
            transposed,
            output_padding,
            padding_mode,
        })
    }

    pub const fn spatial_dimensions(&self) -> usize {
        self.spatial_dimensions
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn padding(&self) -> &[usize] {
        &self.padding
    }

    pub fn dilation(&self) -> &[usize] {
        &self.dilation
    }

    pub const fn groups(&self) -> usize {
        self.groups
    }

    pub const fn transposed(&self) -> bool {
        self.transposed
    }

    pub fn output_padding(&self) -> &[usize] {
        &self.output_padding
    }

    pub const fn padding_mode(&self) -> ConvolutionPaddingMode {
        self.padding_mode
    }

    pub fn checked_output_shape(
        &self,
        input_shape: &[u64],
        weight_shape: &[u64],
        bias_shape: Option<&[u64]>,
    ) -> Result<Vec<u64>, OperatorIndirectionError> {
        let input_shape = checked_shape_to_usize(input_shape, "convolution input shape")?;
        let weight_shape = checked_shape_to_usize(weight_shape, "convolution weight shape")?;
        let bias_channels = match bias_shape {
            Some([channels]) => Some(
                usize::try_from(*channels).map_err(|_| {
                    OperatorIndirectionError::ShapeOverflow("convolution bias shape")
                })?,
            ),
            Some(_) => {
                return Err(OperatorIndirectionError::Invalid(
                    "convolution bias must have rank one",
                ));
            }
            None => None,
        };
        CheckedConvolution::new(&input_shape, &weight_shape, bias_channels, self)?
            .output_shape
            .into_iter()
            .map(|extent| {
                u64::try_from(extent).map_err(|_| {
                    OperatorIndirectionError::ShapeOverflow("convolution output shape")
                })
            })
            .collect()
    }

    fn operation_id(&self) -> &'static str {
        match (self.transposed, self.spatial_dimensions) {
            (false, 1) => CONV1D_OPERATION_ID,
            (false, 3) => CONV3D_OPERATION_ID,
            (true, 1) => CONV_TRANSPOSE1D_OPERATION_ID,
            (true, 2) => CONV_TRANSPOSE2D_OPERATION_ID,
            _ => CONV2D_OPERATION_ID,
        }
    }
}

pub fn convolution_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    geometry: &ConvolutionGeometry,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    context.check()?;
    if geometry.padding_mode() != ConvolutionPaddingMode::Zeros {
        return Err(OperatorIndirectionError::Invalid(
            "tensor convolution requires zero padding",
        ));
    }
    let output_shape = geometry.checked_output_shape(
        input.descriptor().shape(),
        weight.descriptor().shape(),
        bias.map(|tensor| tensor.descriptor().shape()),
    )?;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let specification = ConvolutionSpec {
        stride: convolution_dimensions_to_u64(geometry.stride())?,
        padding: convolution_dimensions_to_u64(geometry.padding())?,
        dilation: convolution_dimensions_to_u64(geometry.dilation())?,
        groups: u64::try_from(geometry.groups())
            .map_err(|_| OperatorIndirectionError::ShapeOverflow("convolution groups"))?,
        transposed: geometry.transposed(),
        output_padding: convolution_dimensions_to_u64(geometry.output_padding())?,
    };
    let (output, event) = if let Some(bias) = bias {
        backend.convolution(
            &specification,
            &[input.clone(), weight.clone(), bias.clone()],
            descriptor,
            context,
        )?
    } else {
        backend.convolution(
            &specification,
            &[input.clone(), weight.clone()],
            descriptor,
            context,
        )?
    };
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

fn convolution_dimensions_to_u64(
    dimensions: &[usize],
) -> Result<Vec<u64>, OperatorIndirectionError> {
    dimensions
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension)
                .map_err(|_| OperatorIndirectionError::ShapeOverflow("convolution dimensions"))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorValues {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionVjp {
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

pub fn convolution_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    geometry: &ConvolutionGeometry,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(geometry.operation_id(), device)?;
    let checked = CheckedConvolution::new(input_shape, weight_shape, bias.map(<[f32]>::len), geometry)?;
    checked.require_values(input, weight)?;
    let mut output = vec![0.0; checked.output_count()?];
    convolution_into_checked(&checked, input, weight, bias, &mut output, cancellation)?;
    cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: checked.output_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_into_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    geometry: &ConvolutionGeometry,
    device: DeviceId,
    output: &mut [f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<usize>, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(geometry.operation_id(), device)?;
    let checked = CheckedConvolution::new(input_shape, weight_shape, bias.map(<[f32]>::len), geometry)?;
    checked.require_values(input, weight)?;
    require_len("convolution output", checked.output_count()?, output.len())?;
    convolution_into_checked(&checked, input, weight, bias, output, cancellation)?;
    cancellation.check()?;
    Ok(checked.output_shape)
}

fn convolution_into_checked(
    checked: &CheckedConvolution,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    cancellation: &CancellationToken,
) -> Result<(), OperatorIndirectionError> {
    output.fill(0.0);
    if let Some(bias) = bias {
        for (index, output_value) in output.iter_mut().enumerate() {
            let coordinates = unravel_index_usize(index, &checked.output_shape)?;
            *output_value = bias[coordinates[1]];
        }
    }
    checked.for_each_connection(
        cancellation,
        |input_index, weight_index, output_index, _| {
            output[output_index] =
                input[input_index].mul_add(weight[weight_index], output[output_index]);
            Ok(())
        },
    )?;
    Ok(())
}

pub fn convolution_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    geometry: &ConvolutionGeometry,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(geometry.operation_id(), device)?;
    let checked = CheckedConvolution::new(input_shape, weight_shape, bias.map(<[f32]>::len), geometry)?;
    checked.require_values(input, weight)?;
    require_len(
        "convolution output gradient",
        checked.output_count()?,
        output_gradient.len(),
    )?;
    let mut input_gradient = vec![0.0; input.len()];
    let mut weight_gradient = vec![0.0; weight.len()];
    checked.for_each_connection(
        cancellation,
        |input_index, weight_index, output_index, _| {
            let gradient = output_gradient[output_index];
            input_gradient[input_index] =
                weight[weight_index].mul_add(gradient, input_gradient[input_index]);
            weight_gradient[weight_index] =
                input[input_index].mul_add(gradient, weight_gradient[weight_index]);
            Ok(())
        },
    )?;
    let bias_gradient = if bias.is_some() {
        let mut gradient = vec![0.0; checked.output_channels];
        for (index, value) in output_gradient.iter().enumerate() {
            let coordinates = unravel_index_usize(index, &checked.output_shape)?;
            gradient[coordinates[1]] += value;
        }
        Some(gradient)
    } else {
        None
    };
    cancellation.check()?;
    Ok(ConvolutionVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    geometry: &ConvolutionGeometry,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(geometry.operation_id(), device)?;
    let checked = CheckedConvolution::new(input_shape, weight_shape, bias.map(<[f32]>::len), geometry)?;
    checked.require_values(input, weight)?;
    require_len(
        "convolution input tangent",
        input.len(),
        input_tangent.len(),
    )?;
    require_len(
        "convolution weight tangent",
        weight.len(),
        weight_tangent.len(),
    )?;
    if let Some(tangent) = bias_tangent {
        if bias.is_none() {
            return Err(OperatorIndirectionError::Invalid(
                "convolution bias tangent requires a bias parameter",
            ));
        }
        require_len(
            "convolution bias tangent",
            checked.output_channels,
            tangent.len(),
        )?;
    }
    let mut output = vec![0.0; checked.output_count()?];
    if let Some(bias_tangent) = bias_tangent {
        for (index, output_value) in output.iter_mut().enumerate() {
            let coordinates = unravel_index_usize(index, &checked.output_shape)?;
            *output_value = bias_tangent[coordinates[1]];
        }
    }
    checked.for_each_connection(
        cancellation,
        |input_index, weight_index, output_index, _| {
            output[output_index] += input_tangent[input_index] * weight[weight_index]
                + input[input_index] * weight_tangent[weight_index];
            Ok(())
        },
    )?;
    cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: checked.output_shape,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearVjp {
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

pub fn linear_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(LINEAR_OPERATION_ID, device)?;
    let checked = CheckedLinear::new(input_shape, weight_shape, bias)?;
    checked.require_values(input, weight)?;
    let mut output = vec![0.0; checked.output_count()?];
    for row in 0..checked.rows {
        for output_channel in 0..checked.output_width {
            check_periodically(row * checked.output_width + output_channel, cancellation)?;
            let mut sum = bias.map_or(0.0, |values| values[output_channel]);
            for input_channel in 0..checked.input_width {
                sum = input[row * checked.input_width + input_channel].mul_add(
                    weight[output_channel * checked.input_width + input_channel],
                    sum,
                );
            }
            output[row * checked.output_width + output_channel] = sum;
        }
    }
    cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: checked.output_shape,
    })
}

pub fn linear_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<LinearVjp, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(LINEAR_OPERATION_ID, device)?;
    let checked = CheckedLinear::new(input_shape, weight_shape, bias)?;
    checked.require_values(input, weight)?;
    require_len(
        "linear output gradient",
        checked.output_count()?,
        output_gradient.len(),
    )?;
    let mut input_gradient = vec![0.0; input.len()];
    let mut weight_gradient = vec![0.0; weight.len()];
    let mut bias_gradient = bias.map(|_| vec![0.0; checked.output_width]);
    for row in 0..checked.rows {
        for output_channel in 0..checked.output_width {
            check_periodically(row * checked.output_width + output_channel, cancellation)?;
            let gradient = output_gradient[row * checked.output_width + output_channel];
            if let Some(bias_gradient) = bias_gradient.as_mut() {
                bias_gradient[output_channel] += gradient;
            }
            for input_channel in 0..checked.input_width {
                let input_index = row * checked.input_width + input_channel;
                let weight_index = output_channel * checked.input_width + input_channel;
                input_gradient[input_index] =
                    weight[weight_index].mul_add(gradient, input_gradient[input_index]);
                weight_gradient[weight_index] =
                    input[input_index].mul_add(gradient, weight_gradient[weight_index]);
            }
        }
    }
    cancellation.check()?;
    Ok(LinearVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, OperatorIndirectionError> {
    context.check()?;
    let cancellation = context.cancellation;
    require_cpu(LINEAR_OPERATION_ID, device)?;
    let checked = CheckedLinear::new(input_shape, weight_shape, bias)?;
    checked.require_values(input, weight)?;
    require_len("linear input tangent", input.len(), input_tangent.len())?;
    require_len("linear weight tangent", weight.len(), weight_tangent.len())?;
    if let Some(bias_tangent) = bias_tangent {
        if bias.is_none() {
            return Err(OperatorIndirectionError::Invalid(
                "linear bias tangent requires a bias parameter",
            ));
        }
        require_len(
            "linear bias tangent",
            checked.output_width,
            bias_tangent.len(),
        )?;
    }
    let mut output = vec![0.0; checked.output_count()?];
    for row in 0..checked.rows {
        for output_channel in 0..checked.output_width {
            check_periodically(row * checked.output_width + output_channel, cancellation)?;
            let mut sum = bias_tangent.map_or(0.0, |values| values[output_channel]);
            for input_channel in 0..checked.input_width {
                let input_index = row * checked.input_width + input_channel;
                let weight_index = output_channel * checked.input_width + input_channel;
                sum += input_tangent[input_index] * weight[weight_index]
                    + input[input_index] * weight_tangent[weight_index];
            }
            output[row * checked.output_width + output_channel] = sum;
        }
    }
    cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: checked.output_shape,
    })
}

pub fn scaled_dot_product_attention_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, OperatorIndirectionError> {
    Ok(
        CheckedAttentionInvocation::new(request, query, key, value, mask)?
            .execute_with_context(backend, 1, context)?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_vjp_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    output_gradient: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<AttentionVjp, OperatorIndirectionError> {
    Ok(
        CheckedAttentionInvocation::new(request, query, key, value, mask)?.vjp_with_context(
            backend,
            output_gradient,
            context,
        )?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_jvp_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    query_tangent: &[f32],
    key_tangent: &[f32],
    value_tangent: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, OperatorIndirectionError> {
    Ok(
        CheckedAttentionInvocation::new(request, query, key, value, mask)?.jvp_with_context(
            backend,
            query_tangent,
            key_tangent,
            value_tangent,
            context,
        )?,
    )
}

struct CheckedLinear {
    rows: usize,
    input_width: usize,
    output_width: usize,
    output_shape: Vec<usize>,
}

impl CheckedLinear {
    fn new(
        input_shape: &[usize],
        weight_shape: &[usize],
        bias: Option<&[f32]>,
    ) -> Result<Self, OperatorIndirectionError> {
        let input_width = *input_shape.last().ok_or(OperatorIndirectionError::Invalid(
            "linear input rank must be nonzero",
        ))?;
        if weight_shape.len() != 2 || weight_shape[1] != input_width {
            return Err(OperatorIndirectionError::Invalid(
                "linear weight must have shape [output, input]",
            ));
        }
        let output_width = weight_shape[0];
        if bias.is_some_and(|bias| bias.len() != output_width) {
            return Err(OperatorIndirectionError::Invalid(
                "linear bias must match output width",
            ));
        }
        let input_count = checked_product(input_shape, "linear input shape")?;
        let rows =
            input_count
                .checked_div(input_width)
                .ok_or(OperatorIndirectionError::Invalid(
                    "linear input width must be nonzero",
                ))?;
        let mut output_shape = input_shape.to_vec();
        let last = output_shape
            .last_mut()
            .ok_or(OperatorIndirectionError::Invalid("linear output rank"))?;
        *last = output_width;
        Ok(Self {
            rows,
            input_width,
            output_width,
            output_shape,
        })
    }

    fn require_values(
        &self,
        input: &[f32],
        weight: &[f32],
    ) -> Result<(), OperatorIndirectionError> {
        require_len(
            "linear input",
            checked_product(&[self.rows, self.input_width], "linear input values")?,
            input.len(),
        )?;
        require_len(
            "linear weight",
            checked_product(
                &[self.output_width, self.input_width],
                "linear weight values",
            )?,
            weight.len(),
        )
    }

    fn output_count(&self) -> Result<usize, OperatorIndirectionError> {
        checked_product(&self.output_shape, "linear output shape")
    }
}

struct CheckedConvolution<'a> {
    geometry: &'a ConvolutionGeometry,
    input_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    output_shape: Vec<usize>,
    output_channels: usize,
    input_channels_per_group: usize,
    output_channels_per_group: usize,
    kernel_shape: Vec<usize>,
}

impl<'a> CheckedConvolution<'a> {
    fn new(
        input_shape: &[usize],
        weight_shape: &[usize],
        bias_channels: Option<usize>,
        geometry: &'a ConvolutionGeometry,
    ) -> Result<Self, OperatorIndirectionError> {
        let rank = geometry.spatial_dimensions + 2;
        if input_shape.len() != rank || weight_shape.len() != rank {
            return Err(OperatorIndirectionError::Invalid(
                "convolution input and weight rank must match spatial dimensions",
            ));
        }
        if input_shape[1..].contains(&0) || weight_shape.contains(&0) {
            return Err(OperatorIndirectionError::Invalid(
                "convolution dimensions must be nonzero",
            ));
        }
        let input_channels = input_shape[1];
        if !input_channels.is_multiple_of(geometry.groups) {
            return Err(OperatorIndirectionError::Invalid(
                "convolution input channels must be divisible by groups",
            ));
        }
        let (output_channels, input_channels_per_group, output_channels_per_group) = if geometry
            .transposed
        {
            if weight_shape[0] != input_channels {
                return Err(OperatorIndirectionError::Invalid(
                    "transposed-convolution weight input channels mismatch",
                ));
            }
            let output_channels = weight_shape[1].checked_mul(geometry.groups).ok_or(
                OperatorIndirectionError::ShapeOverflow("transposed-convolution output channels"),
            )?;
            (
                output_channels,
                input_channels / geometry.groups,
                weight_shape[1],
            )
        } else {
            if weight_shape[1] != input_channels / geometry.groups
                || !weight_shape[0].is_multiple_of(geometry.groups)
            {
                return Err(OperatorIndirectionError::Invalid(
                    "convolution weight channels are incompatible with groups",
                ));
            }
            (
                weight_shape[0],
                weight_shape[1],
                weight_shape[0] / geometry.groups,
            )
        };
        if bias_channels.is_some_and(|channels| channels != output_channels) {
            return Err(OperatorIndirectionError::Invalid(
                "convolution bias must match output channels",
            ));
        }
        let kernel_shape = weight_shape[2..].to_vec();
        let mut output_shape = vec![input_shape[0], output_channels];
        for dimension in 0..geometry.spatial_dimensions {
            let input = input_shape[dimension + 2];
            if geometry.padding_mode == ConvolutionPaddingMode::Reflect
                && geometry.padding[dimension] >= input
            {
                return Err(OperatorIndirectionError::Invalid(
                    "reflection padding must be smaller than the input dimension",
                ));
            }
            if geometry.padding_mode == ConvolutionPaddingMode::Circular
                && geometry.padding[dimension] > input
            {
                return Err(OperatorIndirectionError::Invalid(
                    "circular padding cannot wrap an input dimension more than once",
                ));
            }
            let kernel_extent = geometry.dilation[dimension]
                .checked_mul(kernel_shape[dimension].saturating_sub(1))
                .and_then(|value| value.checked_add(1))
                .ok_or(OperatorIndirectionError::ShapeOverflow(
                    "convolution kernel extent",
                ))?;
            let output = if geometry.transposed {
                let doubled_padding = geometry.padding[dimension].checked_mul(2).ok_or(
                    OperatorIndirectionError::ShapeOverflow("transposed-convolution padding"),
                )?;
                input
                    .saturating_sub(1)
                    .checked_mul(geometry.stride[dimension])
                    .and_then(|value| value.checked_add(kernel_extent))
                    .and_then(|value| value.checked_add(geometry.output_padding[dimension]))
                    .and_then(|value| value.checked_sub(doubled_padding))
                    .ok_or(OperatorIndirectionError::Invalid(
                        "transposed-convolution output dimension is non-positive",
                    ))?
            } else {
                let doubled_padding = geometry.padding[dimension].checked_mul(2).ok_or(
                    OperatorIndirectionError::ShapeOverflow("convolution padding"),
                )?;
                let padded = input.checked_add(doubled_padding).ok_or(
                    OperatorIndirectionError::ShapeOverflow("convolution padded input"),
                )?;
                if padded < kernel_extent {
                    return Err(OperatorIndirectionError::Invalid(
                        "convolution kernel exceeds padded input",
                    ));
                }
                (padded - kernel_extent) / geometry.stride[dimension] + 1
            };
            if output == 0 {
                return Err(OperatorIndirectionError::Invalid(
                    "convolution output dimensions must be nonzero",
                ));
            }
            output_shape.push(output);
        }
        Ok(Self {
            geometry,
            input_shape: input_shape.to_vec(),
            weight_shape: weight_shape.to_vec(),
            output_shape,
            output_channels,
            input_channels_per_group,
            output_channels_per_group,
            kernel_shape,
        })
    }

    fn require_values(
        &self,
        input: &[f32],
        weight: &[f32],
    ) -> Result<(), OperatorIndirectionError> {
        require_len(
            "convolution input",
            checked_product(&self.input_shape, "convolution input")?,
            input.len(),
        )?;
        require_len(
            "convolution weight",
            checked_product(&self.weight_shape, "convolution weight")?,
            weight.len(),
        )
    }

    fn output_count(&self) -> Result<usize, OperatorIndirectionError> {
        checked_product(&self.output_shape, "convolution output")
    }

    fn for_each_connection(
        &self,
        cancellation: &CancellationToken,
        mut visitor: impl FnMut(usize, usize, usize, usize) -> Result<(), OperatorIndirectionError>,
    ) -> Result<(), OperatorIndirectionError> {
        if self.geometry.transposed {
            self.for_each_transposed_connection(cancellation, visitor)
        } else {
            let output_count = self.output_count()?;
            for output_index in 0..output_count {
                check_periodically(output_index, cancellation)?;
                let output_coordinates = unravel_index_usize(output_index, &self.output_shape)?;
                let batch = output_coordinates[0];
                let output_channel = output_coordinates[1];
                let group = output_channel / self.output_channels_per_group;
                for input_in_group in 0..self.input_channels_per_group {
                    let input_channel = group * self.input_channels_per_group + input_in_group;
                    let kernel_count = checked_product(&self.kernel_shape, "convolution kernel")?;
                    for kernel_index in 0..kernel_count {
                        let kernel_coordinates =
                            unravel_index_usize(kernel_index, &self.kernel_shape)?;
                        let mut input_coordinates = vec![batch, input_channel];
                        let mut inside = true;
                        for dimension in 0..self.geometry.spatial_dimensions {
                            let source = output_coordinates[dimension + 2]
                                .checked_mul(self.geometry.stride[dimension])
                                .and_then(|value| {
                                    value.checked_add(
                                        kernel_coordinates[dimension]
                                            .checked_mul(self.geometry.dilation[dimension])?,
                                    )
                                })
                                .ok_or(OperatorIndirectionError::ShapeOverflow(
                                    "convolution input coordinate",
                                ))?;
                            let Some(source) = map_padded_coordinate(
                                source,
                                self.geometry.padding[dimension],
                                self.input_shape[dimension + 2],
                                self.geometry.padding_mode,
                            )?
                            else {
                                inside = false;
                                break;
                            };
                            input_coordinates.push(source);
                        }
                        if !inside {
                            continue;
                        }
                        let mut weight_coordinates = vec![output_channel, input_in_group];
                        weight_coordinates.extend_from_slice(&kernel_coordinates);
                        visitor(
                            ravel_index(&input_coordinates, &self.input_shape)?,
                            ravel_index(&weight_coordinates, &self.weight_shape)?,
                            output_index,
                            output_channel,
                        )?;
                    }
                }
            }
            Ok(())
        }
    }

    fn for_each_transposed_connection(
        &self,
        cancellation: &CancellationToken,
        mut visitor: impl FnMut(usize, usize, usize, usize) -> Result<(), OperatorIndirectionError>,
    ) -> Result<(), OperatorIndirectionError> {
        let input_count = checked_product(&self.input_shape, "transposed-convolution input")?;
        let kernel_count = checked_product(&self.kernel_shape, "transposed-convolution kernel")?;
        for input_index in 0..input_count {
            check_periodically(input_index, cancellation)?;
            let input_coordinates = unravel_index_usize(input_index, &self.input_shape)?;
            let input_channel = input_coordinates[1];
            let group = input_channel / self.input_channels_per_group;
            for output_in_group in 0..self.output_channels_per_group {
                let output_channel = group * self.output_channels_per_group + output_in_group;
                for kernel_index in 0..kernel_count {
                    let kernel_coordinates = unravel_index_usize(kernel_index, &self.kernel_shape)?;
                    let mut output_coordinates = vec![input_coordinates[0], output_channel];
                    let mut inside = true;
                    for dimension in 0..self.geometry.spatial_dimensions {
                        let destination = input_coordinates[dimension + 2]
                            .checked_mul(self.geometry.stride[dimension])
                            .and_then(|value| {
                                value.checked_add(
                                    kernel_coordinates[dimension]
                                        .checked_mul(self.geometry.dilation[dimension])?,
                                )
                            })
                            .ok_or(OperatorIndirectionError::ShapeOverflow(
                                "transposed-convolution output coordinate",
                            ))?;
                        let Some(destination) =
                            destination.checked_sub(self.geometry.padding[dimension])
                        else {
                            inside = false;
                            break;
                        };
                        if destination >= self.output_shape[dimension + 2] {
                            inside = false;
                            break;
                        }
                        output_coordinates.push(destination);
                    }
                    if !inside {
                        continue;
                    }
                    let mut weight_coordinates = vec![input_channel, output_in_group];
                    weight_coordinates.extend_from_slice(&kernel_coordinates);
                    visitor(
                        input_index,
                        ravel_index(&weight_coordinates, &self.weight_shape)?,
                        ravel_index(&output_coordinates, &self.output_shape)?,
                        output_channel,
                    )?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn map_padded_coordinate(
    coordinate_before_padding: usize,
    padding: usize,
    extent: usize,
    mode: ConvolutionPaddingMode,
) -> Result<Option<usize>, OperatorIndirectionError> {
    if coordinate_before_padding >= padding {
        let coordinate = coordinate_before_padding - padding;
        if coordinate < extent {
            return Ok(Some(coordinate));
        }
        return match mode {
            ConvolutionPaddingMode::Zeros => Ok(None),
            ConvolutionPaddingMode::Replicate => Ok(Some(extent - 1)),
            ConvolutionPaddingMode::Circular => Ok(Some(coordinate % extent)),
            ConvolutionPaddingMode::Reflect => Ok(Some(reflect_coordinate(coordinate, extent)?)),
        };
    }

    let distance = padding - coordinate_before_padding;
    match mode {
        ConvolutionPaddingMode::Zeros => Ok(None),
        ConvolutionPaddingMode::Replicate => Ok(Some(0)),
        ConvolutionPaddingMode::Circular => {
            let remainder = distance % extent;
            Ok(Some(if remainder == 0 {
                0
            } else {
                extent - remainder
            }))
        }
        ConvolutionPaddingMode::Reflect => {
            let period = reflection_period(extent)?;
            let remainder = distance % period;
            let coordinate = if remainder == 0 {
                0
            } else {
                period - remainder
            };
            Ok(Some(reflect_coordinate(coordinate, extent)?))
        }
    }
}

fn reflect_coordinate(coordinate: usize, extent: usize) -> Result<usize, OperatorIndirectionError> {
    let period = reflection_period(extent)?;
    let coordinate = coordinate % period;
    Ok(if coordinate < extent {
        coordinate
    } else {
        period - coordinate
    })
}

fn reflection_period(extent: usize) -> Result<usize, OperatorIndirectionError> {
    extent
        .checked_sub(1)
        .and_then(|value| value.checked_mul(2))
        .filter(|period| *period != 0)
        .ok_or(OperatorIndirectionError::Invalid(
            "reflection padding requires an input dimension larger than one",
        ))
}

fn require_cpu(operation: &'static str, device: DeviceId) -> Result<(), OperatorIndirectionError> {
    if device.kind() == DeviceKind::Cpu && device.ordinal() == 0 {
        Ok(())
    } else {
        Err(OperatorIndirectionError::UnsupportedDevice { operation, device })
    }
}

fn require_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), OperatorIndirectionError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OperatorIndirectionError::ValueCount {
            name,
            expected,
            actual,
        })
    }
}

fn checked_product(
    values: &[usize],
    name: &'static str,
) -> Result<usize, OperatorIndirectionError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(OperatorIndirectionError::ShapeOverflow(name))
    })
}

fn checked_shape_to_usize(
    shape: &[u64],
    name: &'static str,
) -> Result<Vec<usize>, OperatorIndirectionError> {
    shape
        .iter()
        .map(|extent| {
            usize::try_from(*extent).map_err(|_| OperatorIndirectionError::ShapeOverflow(name))
        })
        .collect()
}

fn unravel_index(mut linear: usize, shape: &[u64]) -> Result<Vec<u64>, OperatorIndirectionError> {
    let mut coordinates = vec![0; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let extent = usize::try_from(shape[dimension])
            .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast coordinates"))?;
        if extent == 0 {
            return Ok(coordinates);
        }
        coordinates[dimension] = u64::try_from(linear % extent)
            .map_err(|_| OperatorIndirectionError::ShapeOverflow("cast coordinate"))?;
        linear /= extent;
    }
    Ok(coordinates)
}

fn unravel_index_usize(
    mut linear: usize,
    shape: &[usize],
) -> Result<Vec<usize>, OperatorIndirectionError> {
    let mut coordinates = vec![0; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let extent = shape[dimension];
        if extent == 0 {
            return Err(OperatorIndirectionError::Invalid(
                "zero-sized dimensions are unsupported for this operator",
            ));
        }
        coordinates[dimension] = linear % extent;
        linear /= extent;
    }
    Ok(coordinates)
}

fn ravel_index(coordinates: &[usize], shape: &[usize]) -> Result<usize, OperatorIndirectionError> {
    if coordinates.len() != shape.len() {
        return Err(OperatorIndirectionError::Invalid(
            "coordinate rank mismatch",
        ));
    }
    coordinates
        .iter()
        .zip(shape)
        .try_fold(0_usize, |index, (coordinate, extent)| {
            if coordinate >= extent {
                return Err(OperatorIndirectionError::Invalid(
                    "coordinate is outside tensor shape",
                ));
            }
            index
                .checked_mul(*extent)
                .and_then(|index| index.checked_add(*coordinate))
                .ok_or(OperatorIndirectionError::ShapeOverflow("flat tensor index"))
        })
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), OperatorIndirectionError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
