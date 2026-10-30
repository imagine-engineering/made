use crate::{
    CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, Rgb8ImageTensor, Tensor, TensorDescriptor, TensorError,
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeBiquadCoefficients, NativeMelScaleConfiguration,
        NativeMorphologyOperation, biquad_with_context_exact_native,
        mel_scale_project_vjp_with_context_exact_native,
        mel_scale_project_with_context_exact_native, native_morphology_with_context_exact,
        normalize_jvp_with_context_exact_native,
        normalize_vjp_with_context_exact_native, normalize_with_context_exact_native,
        to_tensor_with_context_exact_native as image_bytes_to_tensor_with_context_exact_native,
        validate_audio_parameters,
    },
    generated_external_tensor_kernel_02::{
        ExternalTensorKernelPartTwoError, apply_matrix, apply_transpose, map_color,
        map_color_pair,
    },
};
use thiserror::Error;

pub const LAB_TO_RGB_OPERATION_ID: &str = "COMFY-TENSOR-OP-F37B4E403ACF";
pub const BOTTOM_HAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-C5A306EB73FD";
pub const BASS_BIQUAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-F73C7107B450";
pub const MEL_SCALE_OPERATION_ID: &str = "COMFY-TENSOR-OP-EBA0D3470A35";
pub const BOX_CONVERT_OPERATION_ID: &str = "COMFY-TENSOR-OP-E937CE70AC37";
pub const COMPOSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-FBC26239461B";
pub const TO_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-D2AF4145E6CE";

#[derive(Debug, Error)]
pub enum ExternalTensorKernelPartThreeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartOne(#[from] ExternalTensorKernelPartOneError),
    #[error(transparent)]
    PartTwo(#[from] ExternalTensorKernelPartTwoError),
    #[error("external tensor-kernel part-three execution was cancelled")]
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
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed for operation {operation} while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
}

impl From<sim_types::CancellationError> for ExternalTensorKernelPartThreeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBoxFormat {
    Xyxy,
    CenterXyWidthHeight,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeTensorTransform {
    Normalize {
        mean: Vec<f32>,
        standard_deviation: Vec<f32>,
    },
}

fn invalid(
    operation: &'static str,
    reason: impl Into<String>,
) -> ExternalTensorKernelPartThreeError {
    ExternalTensorKernelPartThreeError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn overflow(operation: &'static str, subject: &'static str) -> ExternalTensorKernelPartThreeError {
    ExternalTensorKernelPartThreeError::ShapeOverflow { operation, subject }
}


fn lab_inverse_transfer(value: f32) -> f32 {
    let cubed = value * value * value;
    if cubed > 0.008_856 {
        cubed
    } else {
        (value - 16.0 / 116.0) / 7.787
    }
}

fn lab_inverse_transfer_derivative(value: f32) -> f32 {
    if value * value * value > 0.008_856 {
        3.0 * value * value
    } else {
        1.0 / 7.787
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value > 0.003_130_8 {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * value
    }
}

fn linear_to_srgb_derivative(value: f32) -> f32 {
    if value > 0.003_130_8 {
        (1.055 / 2.4) * value.powf(1.0 / 2.4 - 1.0)
    } else {
        12.92
    }
}

fn lab_to_rgb_unclamped(lab: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let fy = (lab[0] + 16.0) / 116.0;
    let f = [lab[1] / 500.0 + fy, fy, fy - lab[2] / 200.0];
    let xyz = [
        0.950_47 * lab_inverse_transfer(f[0]),
        lab_inverse_transfer(f[1]),
        1.088_83 * lab_inverse_transfer(f[2]),
    ];
    let linear = [
        3.240_481_4 * xyz[0] - 1.537_151_5 * xyz[1] - 0.498_536_32 * xyz[2],
        -0.969_254_9 * xyz[0] + 1.875_99 * xyz[1] + 0.041_555_93 * xyz[2],
        0.055_646_64 * xyz[0] - 0.204_041_34 * xyz[1] + 1.057_311_1 * xyz[2],
    ];
    (f, linear, linear.map(linear_to_srgb))
}

fn lab_to_rgb_value(lab: [f32; 3]) -> [f32; 3] {
    lab_to_rgb_unclamped(lab)
        .2
        .map(|value| value.clamp(0.0, 1.0))
}

fn lab_to_rgb_jacobian(lab: [f32; 3]) -> [[f32; 3]; 3] {
    let (f, linear, srgb) = lab_to_rgb_unclamped(lab);
    let lab_to_f = [
        [1.0 / 116.0, 1.0 / 500.0, 0.0],
        [1.0 / 116.0, 0.0, 0.0],
        [1.0 / 116.0, 0.0, -1.0 / 200.0],
    ];
    let white = [0.950_47, 1.0, 1.088_83];
    let xyz_to_linear = [
        [3.240_481_4, -1.537_151_5, -0.498_536_32],
        [-0.969_254_9, 1.875_99, 0.041_555_93],
        [0.055_646_64, -0.204_041_34, 1.057_311_1],
    ];
    let mut jacobian = [[0.0_f32; 3]; 3];
    for output in 0..3 {
        if !(0.0..=1.0).contains(&srgb[output]) {
            continue;
        }
        let output_derivative = linear_to_srgb_derivative(linear[output]);
        for input in 0..3 {
            for intermediate in 0..3 {
                jacobian[output][input] += output_derivative
                    * xyz_to_linear[output][intermediate]
                    * white[intermediate]
                    * lab_inverse_transfer_derivative(f[intermediate])
                    * lab_to_f[intermediate][input];
            }
        }
    }
    jacobian
}

pub fn lab_to_rgb_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(map_color(
        backend,
        input,
        LAB_TO_RGB_OPERATION_ID,
        context,
        lab_to_rgb_value,
    )?)
}

pub fn lab_to_rgb_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(map_color_pair(
        backend,
        input,
        input_tangent,
        LAB_TO_RGB_OPERATION_ID,
        context,
        |value, tangent| apply_matrix(lab_to_rgb_jacobian(value), tangent),
    )?)
}

pub fn lab_to_rgb_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(map_color_pair(
        backend,
        input,
        output_gradient,
        LAB_TO_RGB_OPERATION_ID,
        context,
        |value, gradient| apply_transpose(lab_to_rgb_jacobian(value), gradient),
    )?)
}

pub fn bottom_hat_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    kernel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(native_morphology_with_context_exact(
        backend,
        input,
        kernel,
        NativeMorphologyOperation::BottomHat,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn bass_biquad_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    gain: f64,
    central_frequency: f64,
    quality: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    let coefficients = bass_coefficients(sample_rate, gain, central_frequency, quality)?;
    Ok(biquad_with_context_exact_native(
        backend,
        waveform,
        coefficients,
        true,
        BASS_BIQUAD_OPERATION_ID,
        context,
    )?)
}

fn bass_coefficients(
    sample_rate: u32,
    gain: f64,
    central_frequency: f64,
    quality: f64,
) -> Result<NativeBiquadCoefficients, ExternalTensorKernelPartThreeError> {
    validate_audio_parameters(
        sample_rate,
        central_frequency,
        quality,
        BASS_BIQUAD_OPERATION_ID,
    )?;
    if !gain.is_finite() {
        return Err(invalid(BASS_BIQUAD_OPERATION_ID, "gain must be finite"));
    }
    let angular_frequency = 2.0 * std::f64::consts::PI * central_frequency / f64::from(sample_rate);
    let amplitude = (gain / 40.0 * std::f64::consts::LN_10).exp();
    let alpha = angular_frequency.sin() / 2.0
        * ((amplitude + 1.0 / amplitude) * (1.0 / quality - 1.0) + 2.0).sqrt();
    let beta = 2.0 * amplitude.sqrt() * alpha;
    let cosine = angular_frequency.cos();
    Ok(NativeBiquadCoefficients {
        b0: amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine + beta),
        b1: 2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
        b2: amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine - beta),
        a0: (amplitude + 1.0) + (amplitude - 1.0) * cosine + beta,
        a1: -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
        a2: (amplitude + 1.0) + (amplitude - 1.0) * cosine - beta,
    })
}

pub fn mel_scale_with_context_exact_native(
    backend: &CpuBackend,
    spectrogram: &Tensor,
    configuration: NativeMelScaleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(mel_scale_project_with_context_exact_native(
        backend,
        spectrogram,
        configuration,
        MEL_SCALE_OPERATION_ID,
        context,
    )?)
}

pub fn mel_scale_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    configuration: NativeMelScaleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    mel_scale_with_context_exact_native(backend, input_tangent, configuration, context)
}

pub fn mel_scale_vjp_with_context_exact_native(
    backend: &CpuBackend,
    spectrogram: &Tensor,
    output_gradient: &Tensor,
    configuration: NativeMelScaleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    Ok(mel_scale_project_vjp_with_context_exact_native(
        backend,
        spectrogram,
        output_gradient,
        configuration,
        MEL_SCALE_OPERATION_ID,
        context,
    )?)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<CpuWorkspaceVec<T>, ExternalTensorKernelPartThreeError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn require_box_tensor(input: &Tensor) -> Result<(), ExternalTensorKernelPartThreeError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ExternalTensorKernelPartThreeError::UnsupportedDevice {
            operation: BOX_CONVERT_OPERATION_ID,
            device: input.descriptor().device(),
        });
    }
    if input.descriptor().dtype() != DType::F32 {
        return Err(ExternalTensorKernelPartThreeError::UnsupportedDType {
            operation: BOX_CONVERT_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    match input.descriptor().shape().last() {
        Some(4) => Ok(()),
        _ => Err(invalid(
            BOX_CONVERT_OPERATION_ID,
            "box tensor must have a final dimension of four",
        )),
    }
}

fn box_value(
    input: [f32; 4],
    input_format: NativeBoxFormat,
    output_format: NativeBoxFormat,
) -> [f32; 4] {
    match (input_format, output_format) {
        (left, right) if left == right => input,
        (NativeBoxFormat::CenterXyWidthHeight, NativeBoxFormat::Xyxy) => [
            input[0] - input[2] / 2.0,
            input[1] - input[3] / 2.0,
            input[0] + input[2] / 2.0,
            input[1] + input[3] / 2.0,
        ],
        (NativeBoxFormat::Xyxy, NativeBoxFormat::CenterXyWidthHeight) => [
            (input[0] + input[2]) / 2.0,
            (input[1] + input[3]) / 2.0,
            input[2] - input[0],
            input[3] - input[1],
        ],
        _ => input,
    }
}

fn box_transpose_value(
    gradient: [f32; 4],
    input_format: NativeBoxFormat,
    output_format: NativeBoxFormat,
) -> [f32; 4] {
    match (input_format, output_format) {
        (left, right) if left == right => gradient,
        (NativeBoxFormat::CenterXyWidthHeight, NativeBoxFormat::Xyxy) => [
            gradient[0] + gradient[2],
            gradient[1] + gradient[3],
            (gradient[2] - gradient[0]) / 2.0,
            (gradient[3] - gradient[1]) / 2.0,
        ],
        (NativeBoxFormat::Xyxy, NativeBoxFormat::CenterXyWidthHeight) => [
            gradient[0] / 2.0 - gradient[2],
            gradient[1] / 2.0 - gradient[3],
            gradient[0] / 2.0 + gradient[2],
            gradient[1] / 2.0 + gradient[3],
        ],
        _ => gradient,
    }
}

fn box_map(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    mut transform: impl FnMut([f32; 4]) -> [f32; 4],
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    require_box_tensor(input)?;
    let shape = input.descriptor().shape();
    let box_count = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| overflow(BOX_CONVERT_OPERATION_ID, "box count"))?;
    let prefix_shape = &shape[..shape.len() - 1];
    let output_count = box_count
        .checked_mul(4)
        .ok_or_else(|| overflow(BOX_CONVERT_OPERATION_ID, "box output elements"))?;
    let mut output = temporary_vec(backend, context, output_count)?;
    for linear in 0..box_count {
        if linear & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        let mut remainder = linear;
        let mut indices = vec![0_u64; shape.len()];
        for axis in (0..prefix_shape.len()).rev() {
            let dimension = usize::try_from(prefix_shape[axis])
                .map_err(|_| overflow(BOX_CONVERT_OPERATION_ID, "box axis"))?;
            if dimension == 0 {
                return Err(invalid(
                    BOX_CONVERT_OPERATION_ID,
                    "cannot index an empty box prefix",
                ));
            }
            indices[axis] = u64::try_from(remainder % dimension)
                .map_err(|_| overflow(BOX_CONVERT_OPERATION_ID, "box coordinate"))?;
            remainder /= dimension;
        }
        let mut value = [0.0_f32; 4];
        for coordinate in 0..4 {
            indices[shape.len() - 1] = coordinate as u64;
            value[coordinate] = match DType::F32.decode_scalar(input.element_bytes(&indices)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => {
                    return Err(invalid(
                        BOX_CONVERT_OPERATION_ID,
                        "canonical scalar decoder returned a non-real box coordinate",
                    ));
                }
            };
        }
        for coordinate in transform(value) {
            output.try_push(coordinate)?;
        }
    }
    context.cancellation.check()?;
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.upload_f32(descriptor, &output, context)?.0)
}


pub fn box_convert_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_format: NativeBoxFormat,
    output_format: NativeBoxFormat,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    box_map(backend, input, context, |value| {
        box_value(value, input_format, output_format)
    })
}

pub fn box_convert_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    input_format: NativeBoxFormat,
    output_format: NativeBoxFormat,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    box_convert_with_context_exact_native(
        backend,
        input_tangent,
        input_format,
        output_format,
        context,
    )
}


pub fn box_convert_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_format: NativeBoxFormat,
    output_format: NativeBoxFormat,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    box_map(
        backend,
        output_gradient,
        context,
        |gradient| box_transpose_value(gradient, input_format, output_format),
    )
}

pub fn compose_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    transforms: &[NativeTensorTransform],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    let mut output = input.clone();
    for transform in transforms {
        context.cancellation.check()?;
        output = match transform {
            NativeTensorTransform::Normalize {
                mean,
                standard_deviation,
            } => normalize_with_context_exact_native(
                backend,
                &output,
                mean,
                standard_deviation,
                context,
            )?,
        };
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn compose_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    transforms: &[NativeTensorTransform],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    let mut output = input_tangent.clone();
    for transform in transforms {
        context.cancellation.check()?;
        output = match transform {
            NativeTensorTransform::Normalize {
                standard_deviation, ..
            } => normalize_jvp_with_context_exact_native(
                backend,
                &output,
                standard_deviation,
                context,
            )?,
        };
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn compose_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    transforms: &[NativeTensorTransform],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    let mut gradient = output_gradient.clone();
    for transform in transforms.iter().rev() {
        context.cancellation.check()?;
        gradient = match transform {
            NativeTensorTransform::Normalize {
                standard_deviation, ..
            } => normalize_vjp_with_context_exact_native(
                backend,
                &gradient,
                standard_deviation,
                context,
            )?,
        };
    }
    context.cancellation.check()?;
    Ok(gradient)
}

pub fn to_tensor_with_context_exact_native(
    backend: &CpuBackend,
    image: &Rgb8ImageTensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartThreeError> {
    context.cancellation.check()?;
    let (height, width) = image.dimensions()?;
    Ok(image_bytes_to_tensor_with_context_exact_native(
        backend,
        image.as_u8_slice()?,
        height,
        width,
        3,
        image.tensor().descriptor().stream(),
        context,
    )?)
}
