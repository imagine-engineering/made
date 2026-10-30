use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, GradScalerConfig, GradScalerError, NativeGradScaler, NumericClass,
    RngCheckpoint, RngError, RngTransaction, Scalar, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
    generated_activation_normalization_functional_01::{
        relu_jvp_with_context_exact_native as relu_values_jvp_exact_native,
        relu_vjp_with_context_exact_native as relu_values_vjp_exact_native,
        relu_with_context_exact_native as relu_values_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, tensor_to_f32_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError, equal_scalar_with_context_exact_native,
    },
};
use thiserror::Error;

pub const TENSOR_EQ_OPERATION_ID: &str = "COMFY-TENSOR-OP-30F43E74E34C";
pub const GRAD_SCALER_OPERATION_ID: &str = "COMFY-TENSOR-OP-2C340B4A7331";
pub const COMPILE_OPERATION_ID: &str = "COMFY-TENSOR-OP-2881ABE3D797";
pub const CUDA_DEVICE_COUNT_OPERATION_ID: &str = "COMFY-TENSOR-OP-28BA5C917CFB";
pub const CUDA_IS_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-310F887878BC";
pub const DEG2RAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-2ED7F479B4BD";
pub const ISFINITE_OPERATION_ID: &str = "COMFY-TENSOR-OP-2C5A78E85B7F";
pub const MPS_EMPTY_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-2E9C9B320055";
pub const NAN_TO_NUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-290B200830F8";
pub const KAIMING_UNIFORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-3118FDEB2829";
pub const REMOVE_PARAMETRIZATIONS_OPERATION_ID: &str = "COMFY-TENSOR-OP-2D7E75B69A4E";
pub const RELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-289B94EF73DE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KaimingMode {
    FanIn,
    FanOut,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KaimingNonlinearity {
    Linear,
    Convolution,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Selu,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ElementwiseRuntimePartFourError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    GradScaler(#[from] GradScalerError),
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error("elementwise/runtime part-four operation was cancelled")]
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
    #[error("elementwise/runtime part-four input is invalid: {0}")]
    Invalid(&'static str),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartFourError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn tensor_eq_scalar_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    Ok(equal_scalar_with_context_exact_native(
        backend, input, other, context,
    )?)
}

pub fn grad_scaler_exact_native(
    config: GradScalerConfig,
    cancellation: &CancellationToken,
) -> Result<NativeGradScaler, ElementwiseRuntimePartFourError> {
    cancellation.check()?;
    let scaler = NativeGradScaler::new(config)?;
    cancellation.check()?;
    Ok(scaler)
}

pub fn deg2rad_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    map_numeric_tensor(
        backend,
        input,
        input.descriptor().dtype(),
        DEG2RAD_OPERATION_ID,
        context,
        |value, dtype| match value {
            DecodedScalar::Real(value) => Ok(DecodedScalar::Real(if dtype == DType::F32 {
                f64::from(value as f32 * (std::f32::consts::PI / 180.0))
            } else {
                value.to_radians()
            })),
            DecodedScalar::Complex { real, imaginary } => Ok(DecodedScalar::Complex {
                real: real.to_radians(),
                imaginary: imaginary.to_radians(),
            }),
            _ => Err(ElementwiseRuntimePartFourError::UnsupportedDType {
                operation: DEG2RAD_OPERATION_ID,
                dtype,
            }),
        },
    )
}

pub fn deg2rad_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    scale_real_tensor(
        backend,
        output_gradient,
        if output_gradient.descriptor().dtype() == DType::F32 {
            f64::from(std::f32::consts::PI / 180.0)
        } else {
            std::f64::consts::PI / 180.0
        },
        DEG2RAD_OPERATION_ID,
        context,
    )
}

pub fn deg2rad_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    deg2rad_vjp_with_context_exact_native(backend, input_tangent, context)
}

pub fn isfinite_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    map_numeric_tensor(
        backend,
        input,
        DType::Bool,
        ISFINITE_OPERATION_ID,
        context,
        |value, _| {
            Ok(DecodedScalar::Boolean(match value {
                DecodedScalar::Real(value) => value.is_finite(),
                DecodedScalar::Complex { real, imaginary } => {
                    real.is_finite() && imaginary.is_finite()
                }
                DecodedScalar::Boolean(_)
                | DecodedScalar::Signed(_)
                | DecodedScalar::Unsigned(_) => true,
            }))
        },
    )
}

pub fn nan_to_num_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    nan: Option<f64>,
    positive_infinity: Option<f64>,
    negative_infinity: Option<f64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    let dtype = input.descriptor().dtype();
    let (default_positive, default_negative) = if dtype.class() == NumericClass::FloatingPoint {
        let info = dtype.floating_point_info()?;
        (info.maximum(), info.minimum())
    } else if dtype.class() == NumericClass::Complex {
        let component = if dtype == DType::Complex64 {
            DType::F32
        } else {
            DType::F64
        };
        let info = component.floating_point_info()?;
        (info.maximum(), info.minimum())
    } else {
        (0.0, 0.0)
    };
    let nan = nan.unwrap_or(0.0);
    let positive_infinity = positive_infinity.unwrap_or(default_positive);
    let negative_infinity = negative_infinity.unwrap_or(default_negative);
    map_numeric_tensor(
        backend,
        input,
        dtype,
        NAN_TO_NUM_OPERATION_ID,
        context,
        |value, _| {
            Ok(match value {
                DecodedScalar::Real(value) => DecodedScalar::Real(replace_non_finite(
                    value,
                    nan,
                    positive_infinity,
                    negative_infinity,
                )),
                DecodedScalar::Complex { real, imaginary } => DecodedScalar::Complex {
                    real: replace_non_finite(real, nan, positive_infinity, negative_infinity),
                    imaginary: replace_non_finite(
                        imaginary,
                        nan,
                        positive_infinity,
                        negative_infinity,
                    ),
                },
                other => other,
            })
        },
    )
}

pub fn nan_to_num_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    nan_to_num_gradient_exact_native(backend, input, output_gradient, context)
}

pub fn nan_to_num_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    nan_to_num_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn relu_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    if input.descriptor().dtype().class() != NumericClass::FloatingPoint {
        return Err(ElementwiseRuntimePartFourError::UnsupportedDType {
            operation: RELU_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let values = relu_values_exact_native(backend, &values, input.descriptor().device(), context)
        .map_err(|_| {
        ElementwiseRuntimePartFourError::Invalid("canonical ReLU dispatch failed")
    })?;
    upload_real_values(
        backend,
        input.descriptor().shape(),
        &values,
        input.descriptor().dtype(),
        input.descriptor().stream(),
        RELU_OPERATION_ID,
        context,
    )
}

pub fn relu_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    relu_gradient_exact_native(backend, input, output_gradient, false, context)
}

pub fn relu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    context.cancellation.check()?;
    relu_gradient_exact_native(backend, input, input_tangent, true, context)
}

pub fn kaiming_uniform_in_place_exact_native(
    input: &mut Tensor,
    mut rng: RngTransaction,
    negative_slope: f64,
    mode: KaimingMode,
    nonlinearity: KaimingNonlinearity,
    cancellation: &CancellationToken,
) -> Result<RngCheckpoint, ElementwiseRuntimePartFourError> {
    cancellation.check()?;
    require_cpu(input, KAIMING_UNIFORM_OPERATION_ID)?;
    rng.require_device(input.descriptor().device())?;
    if input.descriptor().dtype().class() != NumericClass::FloatingPoint {
        return Err(ElementwiseRuntimePartFourError::UnsupportedDType {
            operation: KAIMING_UNIFORM_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let shape = input.descriptor().shape();
    if shape.len() < 2 {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "Kaiming initialization requires at least two dimensions",
        ));
    }
    let element_count = checked_element_count(shape, "Kaiming output")?;
    if element_count == 0 {
        cancellation.check()?;
        return Ok(rng.commit());
    }
    if !negative_slope.is_finite() {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "Kaiming negative slope must be finite",
        ));
    }
    let receptive =
        shape
            .get(2..)
            .ok_or(ElementwiseRuntimePartFourError::ShapeOverflow(
                "Kaiming receptive field",
            ))?
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product.checked_mul(*dimension).ok_or(
                    ElementwiseRuntimePartFourError::ShapeOverflow("Kaiming receptive field"),
                )
            })?;
    let fan = match mode {
        KaimingMode::FanIn => shape[1].checked_mul(receptive),
        KaimingMode::FanOut => shape[0].checked_mul(receptive),
    }
    .ok_or(ElementwiseRuntimePartFourError::ShapeOverflow(
        "Kaiming fan",
    ))?;
    if fan == 0 {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "Kaiming fan must be nonzero",
        ));
    }
    let gain = kaiming_gain(nonlinearity, negative_slope);
    let bound = 3.0_f64.sqrt() * gain / (fan as f64).sqrt();
    let dtype = input.descriptor().dtype();
    let mut candidate = input.clone();
    let mut write = candidate.write()?;
    for linear_index in 0..element_count {
        if linear_index % 1_024 == 0 {
            cancellation.check()?;
        }
        let unit = uniform_unit(dtype, &mut rng, cancellation)?;
        let value = (2.0 * unit - 1.0) * bound;
        let encoded = dtype.encode_scalar(
            Scalar::Float(value),
            KAIMING_UNIFORM_OPERATION_ID,
            DeviceId::CPU,
        )?;
        let indices = unravel_index(linear_index, shape)?;
        write.element_bytes_mut(&indices)?.copy_from_slice(&encoded);
    }
    drop(write);
    cancellation.check()?;
    let checkpoint = rng.commit();
    input.commit_in_place(candidate)?;
    Ok(checkpoint)
}

fn relu_gradient_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    jvp: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    if input.descriptor().shape() != gradient.descriptor().shape() {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "ReLU gradient shape must match input",
        ));
    }
    if input.descriptor().dtype().class() != NumericClass::FloatingPoint
        || input.descriptor().dtype() != gradient.descriptor().dtype()
    {
        return Err(ElementwiseRuntimePartFourError::UnsupportedDType {
            operation: RELU_OPERATION_ID,
            dtype: gradient.descriptor().dtype(),
        });
    }
    if input.descriptor().device() != gradient.descriptor().device()
        || input.descriptor().stream() != gradient.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "ReLU gradient device and stream must match input",
        ));
    }
    let input_values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let gradient_values = tensor_to_f32_with_context_exact_native(backend, gradient, context)?;
    let values = if jvp {
        relu_values_jvp_exact_native(
            backend,
            &input_values,
            &gradient_values,
            input.descriptor().device(),
            context,
        )
    } else {
        relu_values_vjp_exact_native(
            backend,
            &input_values,
            &gradient_values,
            input.descriptor().device(),
            context,
        )
    }
    .map_err(|_| ElementwiseRuntimePartFourError::Invalid("canonical ReLU gradient failed"))?;
    upload_real_values(
        backend,
        input.descriptor().shape(),
        &values,
        gradient.descriptor().dtype(),
        gradient.descriptor().stream(),
        RELU_OPERATION_ID,
        context,
    )
}

fn replace_non_finite(value: f64, nan: f64, positive: f64, negative: f64) -> f64 {
    if value.is_nan() {
        nan
    } else if value == f64::INFINITY {
        positive
    } else if value == f64::NEG_INFINITY {
        negative
    } else {
        value
    }
}

fn nan_to_num_gradient_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    if input.descriptor().shape() != gradient.descriptor().shape() {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "gradient shape must match input",
        ));
    }
    if input.descriptor().dtype() != gradient.descriptor().dtype() {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "nan_to_num gradient dtype must match input",
        ));
    }
    if input.descriptor().device() != gradient.descriptor().device()
        || input.descriptor().stream() != gradient.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "nan_to_num gradient device and stream must match input",
        ));
    }
    let dtype = gradient.descriptor().dtype();
    map_binary_same_shape_tensor(
        backend,
        input,
        gradient,
        NAN_TO_NUM_OPERATION_ID,
        context,
        |input_value, gradient_value| match (input_value, gradient_value) {
            (DecodedScalar::Real(input), DecodedScalar::Real(gradient)) => {
                Ok(DecodedScalar::Real(if input.is_finite() {
                    gradient
                } else {
                    0.0
                }))
            }
            (
                DecodedScalar::Complex {
                    real: input_real,
                    imaginary: input_imaginary,
                },
                DecodedScalar::Complex {
                    real: gradient_real,
                    imaginary: gradient_imaginary,
                },
            ) => Ok(DecodedScalar::Complex {
                real: if input_real.is_finite() {
                    gradient_real
                } else {
                    0.0
                },
                imaginary: if input_imaginary.is_finite() {
                    gradient_imaginary
                } else {
                    0.0
                },
            }),
            (DecodedScalar::Boolean(_), gradient)
            | (DecodedScalar::Signed(_), gradient)
            | (DecodedScalar::Unsigned(_), gradient) => Ok(gradient),
            _ => Err(ElementwiseRuntimePartFourError::UnsupportedDType {
                operation: NAN_TO_NUM_OPERATION_ID,
                dtype,
            }),
        },
    )
}

fn scale_real_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    factor: f64,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    map_numeric_tensor(
        backend,
        input,
        input.descriptor().dtype(),
        operation,
        context,
        |value, dtype| match value {
            DecodedScalar::Real(value) => Ok(DecodedScalar::Real(value * factor)),
            DecodedScalar::Complex { real, imaginary } => Ok(DecodedScalar::Complex {
                real: real * factor,
                imaginary: imaginary * factor,
            }),
            _ => Err(ElementwiseRuntimePartFourError::UnsupportedDType { operation, dtype }),
        },
    )
}

fn map_numeric_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    output_dtype: DType,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    function: impl Fn(DecodedScalar, DType) -> Result<DecodedScalar, ElementwiseRuntimePartFourError>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    require_cpu(input, operation)?;
    let shape = input.descriptor().shape();
    let element_count = checked_element_count(shape, "elementwise output")?;
    let byte_width = usize::try_from(output_dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("elementwise bytes"))?;
    let byte_count = element_count.checked_mul(byte_width).ok_or(
        ElementwiseRuntimePartFourError::ShapeOverflow("elementwise bytes"),
    )?;
    let mut bytes = temporary_vec(backend, context, byte_count, "elementwise bytes")?;
    for linear_index in 0..element_count {
        if linear_index % 1_024 == 0 {
            context.check()?;
        }
        let indices = unravel_index(linear_index, shape)?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        let value = function(value, input.descriptor().dtype())?;
        temporary_extend(&mut bytes, &encode_decoded(output_dtype, value, operation)?)?;
    }
    upload_bytes(
        backend,
        shape,
        output_dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

fn map_binary_same_shape_tensor(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    function: impl Fn(
        DecodedScalar,
        DecodedScalar,
    ) -> Result<DecodedScalar, ElementwiseRuntimePartFourError>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    require_cpu(left, operation)?;
    require_cpu(right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "binary elementwise shapes must match",
        ));
    }
    let output_dtype = right.descriptor().dtype();
    let shape = left.descriptor().shape();
    let element_count = checked_element_count(shape, "binary elementwise output")?;
    let byte_width = usize::try_from(output_dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("binary elementwise bytes"))?;
    let byte_count = element_count.checked_mul(byte_width).ok_or(
        ElementwiseRuntimePartFourError::ShapeOverflow("binary elementwise bytes"),
    )?;
    let mut bytes = temporary_vec(backend, context, byte_count, "binary elementwise bytes")?;
    for linear_index in 0..element_count {
        if linear_index % 1_024 == 0 {
            context.check()?;
        }
        let indices = unravel_index(linear_index, shape)?;
        let left_value = left
            .descriptor()
            .dtype()
            .decode_scalar(left.element_bytes(&indices)?)?;
        let right_value = right
            .descriptor()
            .dtype()
            .decode_scalar(right.element_bytes(&indices)?)?;
        temporary_extend(
            &mut bytes,
            &encode_decoded(output_dtype, function(left_value, right_value)?, operation)?,
        )?;
    }
    upload_bytes(
        backend,
        shape,
        output_dtype,
        right.descriptor().stream(),
        &bytes,
        context,
    )
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: crate::StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    output.write()?.bytes_mut()?.copy_from_slice(bytes);
    context.check()?;
    Ok(output)
}

fn upload_real_values(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    stream: crate::StreamId,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourError> {
    let expected = checked_element_count(shape, "real-value output")?;
    if values.len() != expected {
        return Err(ElementwiseRuntimePartFourError::Invalid(
            "real-value output length does not match shape",
        ));
    }
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("real-value bytes"))?;
    let byte_count =
        expected
            .checked_mul(byte_width)
            .ok_or(ElementwiseRuntimePartFourError::ShapeOverflow(
                "real-value bytes",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_count, "real-value bytes")?;
    for (index, value) in values.iter().enumerate() {
        if index % 1_024 == 0 {
            context.check()?;
        }
        temporary_extend(
            &mut bytes,
            &dtype.encode_scalar(Scalar::Float(f64::from(*value)), operation, DeviceId::CPU)?,
        )?;
    }
    upload_bytes(backend, shape, dtype, stream, &bytes, context)
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _allocation: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartFourError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_extend(
    values: &mut TemporaryVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimePartFourError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}

fn encode_decoded(
    dtype: DType,
    value: DecodedScalar,
    operation: &'static str,
) -> Result<Vec<u8>, ElementwiseRuntimePartFourError> {
    Ok(dtype.encode_decoded_scalar(value, operation, DeviceId::CPU)?)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFourError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartFourError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn checked_element_count(
    shape: &[u64],
    name: &'static str,
) -> Result<usize, ElementwiseRuntimePartFourError> {
    let count = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(ElementwiseRuntimePartFourError::ShapeOverflow(name))
    })?;
    usize::try_from(count).map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow(name))
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartFourError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.len())
        .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("tensor indices"))?;
    indices.resize(shape.len(), 0);
    for dimension_index in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[dimension_index])
            .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("tensor dimension"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartFourError::ShapeOverflow(
                "zero-sized tensor index",
            ));
        }
        indices[dimension_index] = u64::try_from(linear_index % dimension)
            .map_err(|_| ElementwiseRuntimePartFourError::ShapeOverflow("tensor index"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn kaiming_gain(nonlinearity: KaimingNonlinearity, negative_slope: f64) -> f64 {
    match nonlinearity {
        KaimingNonlinearity::Linear
        | KaimingNonlinearity::Convolution
        | KaimingNonlinearity::Sigmoid => 1.0,
        KaimingNonlinearity::Tanh => 5.0 / 3.0,
        KaimingNonlinearity::Relu => 2.0_f64.sqrt(),
        KaimingNonlinearity::LeakyRelu => (2.0 / (1.0 + negative_slope.powi(2))).sqrt(),
        KaimingNonlinearity::Selu => 0.75,
    }
}

fn uniform_unit(
    dtype: DType,
    rng: &mut RngTransaction,
    cancellation: &CancellationToken,
) -> Result<f64, ElementwiseRuntimePartFourError> {
    if dtype == DType::F64 {
        let high = u64::from(rng.next_u32(cancellation)? >> 5);
        let low = u64::from(rng.next_u32(cancellation)? >> 6);
        Ok(((high << 26) | low) as f64 / ((1_u64 << 53) as f64))
    } else {
        let word = rng.next_u32(cancellation)? & 0x00ff_ffff;
        Ok(f64::from(word) / f64::from(1_u32 << 24))
    }
}
