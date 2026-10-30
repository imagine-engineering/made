use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, Layout, ResizeCrop, ResizeMode, ResizeSpec, StreamId,
    Tensor, TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    generated_elementwise_or_runtime_operation_10::hann_window_with_context_exact_native,
    generated_elementwise_or_runtime_operation_12::stft_with_context_exact_native,
};
use thiserror::Error;

pub const REARRANGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-41E0A53BDA36";
pub const MORPHOLOGY_CLOSING_OPERATION_ID: &str = "COMFY-TENSOR-OP-338E50E9975A";
pub const MORPHOLOGY_GRADIENT_OPERATION_ID: &str = "COMFY-TENSOR-OP-165C20AC8DD8";
pub const MORPHOLOGY_OPENING_OPERATION_ID: &str = "COMFY-TENSOR-OP-363BE404A764";
pub const MORPHOLOGY_DILATION_OPERATION_ID: &str = "COMFY-TENSOR-OP-AF5C2820E4C3";
pub const MORPHOLOGY_EROSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-9236C1C08976";
pub const MORPHOLOGY_TOP_HAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-AC69F309A190";
pub const MORPHOLOGY_BOTTOM_HAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-C5A306EB73FD";
pub const EQUALIZER_BIQUAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-0607DAA06439";
pub const RESAMPLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0A14AB1C4005";
pub const TREBLE_BIQUAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-49A168D86220";
pub const MEL_SPECTROGRAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-367BF5D133D8";
pub const ROI_ALIGN_OPERATION_ID: &str = "COMFY-TENSOR-OP-0ABA532316FA";
pub const NORMALIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0AB66ED3B4C2";
pub const RESIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0882F83B464A";
pub const TO_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-2799F344E971";

#[derive(Debug, Error)]
pub enum ExternalTensorKernelPartOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("external tensor kernel execution was cancelled")]
    Cancelled,
    #[error("operation {operation} does not support device {device:?}")]
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
    #[error("operation {operation} overflowed while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
}

impl From<sim_types::CancellationError> for ExternalTensorKernelPartOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRearrangePlan {
    input_shape: Vec<u64>,
    output_shape: Vec<u64>,
    mapping: NativeRearrangeMapping,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NativeRearrangeMapping {
    Explicit(Vec<u64>),
    AtomicAxes {
        lengths: Vec<u64>,
        input_groups: Vec<Vec<usize>>,
        output_groups: Vec<Vec<usize>>,
    },
}

impl NativeRearrangePlan {
    pub fn checked(
        input_shape: Vec<u64>,
        output_shape: Vec<u64>,
        source_linear_indices: Vec<u64>,
    ) -> Result<Self, ExternalTensorKernelPartOneError> {
        Self::checked_for_operation(
            REARRANGE_OPERATION_ID,
            input_shape,
            output_shape,
            source_linear_indices,
        )
    }

    pub fn checked_for_operation(
        operation: &'static str,
        input_shape: Vec<u64>,
        output_shape: Vec<u64>,
        source_linear_indices: Vec<u64>,
    ) -> Result<Self, ExternalTensorKernelPartOneError> {
        let input_count = element_count(&input_shape, operation, "input elements")?;
        let output_count = element_count(&output_shape, operation, "output elements")?;
        if input_count != output_count || source_linear_indices.len() != output_count {
            return Err(invalid(
                operation,
                "rearrangement must preserve element count and provide one source per output",
            ));
        }
        let mut seen = vec![false; input_count];
        for &source in &source_linear_indices {
            let source = usize::try_from(source)
                .map_err(|_| shape_overflow(operation, "rearrangement source index"))?;
            let slot = seen.get_mut(source).ok_or_else(|| {
                invalid(operation, "rearrangement source index is outside the input")
            })?;
            if *slot {
                return Err(invalid(
                    operation,
                    "rearrangement source mapping is not one-to-one",
                ));
            }
            *slot = true;
        }
        Ok(Self {
            input_shape,
            output_shape,
            mapping: NativeRearrangeMapping::Explicit(source_linear_indices),
        })
    }

    pub fn from_atomic_axes(
        operation: &'static str,
        input_shape: Vec<u64>,
        lengths: Vec<u64>,
        input_groups: Vec<Vec<usize>>,
        output_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ExternalTensorKernelPartOneError> {
        if input_groups.len() != input_shape.len() {
            return Err(invalid(
                operation,
                "atomic rearrangement requires one axis composition per input dimension",
            ));
        }
        validate_atomic_axis_partition(operation, &lengths, &input_groups, "input")?;
        validate_atomic_axis_partition(operation, &lengths, &output_groups, "output")?;
        for (dimension, group) in input_shape.iter().zip(&input_groups) {
            if atomic_group_length(operation, &lengths, group)? != *dimension {
                return Err(invalid(
                    operation,
                    "atomic input composition does not match the declared input shape",
                ));
            }
        }
        let output_shape = output_groups
            .iter()
            .map(|group| atomic_group_length(operation, &lengths, group))
            .collect::<Result<Vec<_>, _>>()?;
        if element_count(&input_shape, operation, "input elements")?
            != element_count(&output_shape, operation, "output elements")?
        {
            return Err(invalid(
                operation,
                "atomic rearrangement must preserve element count",
            ));
        }
        Ok(Self {
            input_shape,
            output_shape,
            mapping: NativeRearrangeMapping::AtomicAxes {
                lengths,
                input_groups,
                output_groups,
            },
        })
    }

    pub fn patch_embedding(
        input_shape: &[u64],
        temporal_patch: u64,
        height_patch: u64,
        width_patch: u64,
    ) -> Result<Self, ExternalTensorKernelPartOneError> {
        let [batch, channels, temporal, height, width] = input_shape else {
            return Err(invalid(
                REARRANGE_OPERATION_ID,
                "patch rearrangement expects BCTHW rank five input",
            ));
        };
        if temporal_patch == 0 || height_patch == 0 || width_patch == 0 {
            return Err(invalid(
                REARRANGE_OPERATION_ID,
                "patch dimensions must be non-zero",
            ));
        }
        if temporal % temporal_patch != 0 || height % height_patch != 0 || width % width_patch != 0
        {
            return Err(invalid(
                REARRANGE_OPERATION_ID,
                "input spatial and temporal dimensions must be divisible by patch dimensions",
            ));
        }
        let temporal_blocks = temporal / temporal_patch;
        let height_blocks = height / height_patch;
        let width_blocks = width / width_patch;
        Self::from_atomic_axes(
            REARRANGE_OPERATION_ID,
            input_shape.to_vec(),
            vec![
                *batch,
                *channels,
                temporal_blocks,
                temporal_patch,
                height_blocks,
                height_patch,
                width_blocks,
                width_patch,
            ],
            vec![vec![0], vec![1], vec![2, 3], vec![4, 5], vec![6, 7]],
            vec![vec![0], vec![2], vec![4], vec![6], vec![1, 3, 5, 7]],
        )
    }

    pub fn input_shape(&self) -> &[u64] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[u64] {
        &self.output_shape
    }

    pub fn is_symbolic(&self) -> bool {
        matches!(self.mapping, NativeRearrangeMapping::AtomicAxes { .. })
    }

    fn source_linear_index(
        &self,
        output_linear: usize,
        operation: &'static str,
    ) -> Result<usize, ExternalTensorKernelPartOneError> {
        match &self.mapping {
            NativeRearrangeMapping::Explicit(sources) => {
                let source = sources.get(output_linear).ok_or_else(|| {
                    invalid(operation, "rearrangement output index is outside plan")
                })?;
                usize::try_from(*source)
                    .map_err(|_| shape_overflow(operation, "rearrangement source index"))
            }
            NativeRearrangeMapping::AtomicAxes {
                lengths,
                input_groups,
                output_groups,
            } => {
                let output_indices = unravel_index(output_linear, &self.output_shape, operation)?;
                let mut atomic_indices = vec![0_u64; lengths.len()];
                for (&index, group) in output_indices.iter().zip(output_groups) {
                    decode_atomic_group(operation, index, lengths, group, &mut atomic_indices)?;
                }
                let input_indices = input_groups
                    .iter()
                    .map(|group| compose_atomic_group(operation, lengths, group, &atomic_indices))
                    .collect::<Result<Vec<_>, _>>()?;
                ravel_index(&input_indices, &self.input_shape, operation)
            }
        }
    }

    fn view_descriptor(
        &self,
        input: &Tensor,
        operation: &'static str,
    ) -> Result<Option<TensorDescriptor>, ExternalTensorKernelPartOneError> {
        let NativeRearrangeMapping::AtomicAxes {
            lengths,
            input_groups,
            output_groups,
        } = &self.mapping
        else {
            return Ok(None);
        };
        let mut atomic_strides = vec![None; lengths.len()];
        for (group, &input_stride) in input_groups.iter().zip(input.descriptor().strides()) {
            let mut stride = i128::from(input_stride);
            for &axis in group.iter().rev() {
                let slot = atomic_strides
                    .get_mut(axis)
                    .ok_or_else(|| invalid(operation, "atomic input axis is outside plan"))?;
                *slot = Some(stride);
                stride = stride
                    .checked_mul(i128::from(lengths[axis]))
                    .ok_or_else(|| shape_overflow(operation, "atomic input stride"))?;
            }
        }
        let mut output_strides = Vec::with_capacity(output_groups.len());
        for group in output_groups {
            let non_unit = group
                .iter()
                .copied()
                .filter(|&axis| lengths[axis] > 1)
                .collect::<Vec<_>>();
            let output_stride = if let Some(&last) = non_unit.last() {
                for axes in non_unit.windows(2) {
                    let left = atomic_strides[axes[0]]
                        .ok_or_else(|| invalid(operation, "atomic input stride is missing"))?;
                    let right = atomic_strides[axes[1]]
                        .ok_or_else(|| invalid(operation, "atomic input stride is missing"))?;
                    let expected = right
                        .checked_mul(i128::from(lengths[axes[1]]))
                        .ok_or_else(|| shape_overflow(operation, "merged output stride"))?;
                    if left != expected {
                        return Ok(None);
                    }
                }
                atomic_strides[last]
                    .ok_or_else(|| invalid(operation, "atomic output stride is missing"))?
            } else {
                0
            };
            output_strides.push(
                i64::try_from(output_stride)
                    .map_err(|_| shape_overflow(operation, "output view stride"))?,
            );
        }
        Ok(Some(TensorDescriptor::new_strided(
            self.output_shape.clone(),
            output_strides,
            input.descriptor().offset_elements(),
            input.descriptor().dtype(),
            Layout::Strided,
            input.descriptor().device(),
            input.descriptor().stream(),
        )?))
    }

    fn inverse_symbolic(
        &self,
        operation: &'static str,
    ) -> Result<Option<Self>, ExternalTensorKernelPartOneError> {
        let NativeRearrangeMapping::AtomicAxes {
            lengths,
            input_groups,
            output_groups,
        } = &self.mapping
        else {
            return Ok(None);
        };
        Self::from_atomic_axes(
            operation,
            self.output_shape.clone(),
            lengths.clone(),
            output_groups.clone(),
            input_groups.clone(),
        )
        .map(Some)
    }
}

pub fn rearrange_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    plan: &NativeRearrangePlan,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, REARRANGE_OPERATION_ID)?;
    if input.descriptor().shape() != plan.input_shape {
        return Err(invalid(
            REARRANGE_OPERATION_ID,
            "input shape does not match rearrangement plan",
        ));
    }
    rearrange_fresh_exact_native(backend, input, plan, REARRANGE_OPERATION_ID, context)
}

pub fn rearrange_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    if input.descriptor().shape() != plan.input_shape {
        return Err(invalid(
            operation,
            "input shape does not match rearrangement plan",
        ));
    }
    context.cancellation.check()?;
    if let Some(descriptor) = plan.view_descriptor(input, operation)? {
        let output = input.view(descriptor, ViewAccess::ReadOnly)?;
        context.cancellation.check()?;
        return Ok(output);
    }
    rearrange_fresh_exact_native(backend, input, plan, operation, context)
}

pub fn rearrange_tensor_with_context_exact_native_for_operation(
    backend: &dyn TensorBackend,
    input: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    if input.descriptor().device() != backend.device() {
        return Err(TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: input.descriptor().device(),
        }
        .into());
    }
    if input.descriptor().stream() != context.stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: input.descriptor().stream(),
        }
        .into());
    }
    if input.descriptor().shape() != plan.input_shape {
        return Err(invalid(
            operation,
            "input shape does not match rearrangement plan",
        ));
    }
    if let Some(descriptor) = plan.view_descriptor(input, operation)? {
        let output = input.view(descriptor, ViewAccess::ReadOnly)?;
        context.cancellation.check()?;
        return Ok(output);
    }
    let descriptor = TensorDescriptor::contiguous(
        plan.output_shape.clone(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    let output_count = element_count(&plan.output_shape, operation, "output elements")?;
    {
        let mut write = output.write()?;
        for output_linear in 0..output_count {
            check_periodically(output_linear, context.cancellation)?;
            let source_linear = plan.source_linear_index(output_linear, operation)?;
            let source_indices = unravel_index(source_linear, input.descriptor().shape(), operation)?;
            let output_indices = unravel_index(output_linear, &plan.output_shape, operation)?;
            write
                .element_bytes_mut(&output_indices)?
                .copy_from_slice(input.element_bytes(&source_indices)?);
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.cancellation.check()?;
    Ok(output)
}

pub fn rearrange_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    plan: &NativeRearrangePlan,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    rearrange_with_context_exact_native(backend, input_tangent, plan, context)
}

pub fn rearrange_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    plan: &NativeRearrangePlan,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    rearrange_vjp_fresh_exact_native(
        backend,
        output_gradient,
        plan,
        REARRANGE_OPERATION_ID,
        context,
    )
}

pub fn rearrange_jvp_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    rearrange_with_context_exact_native_for_operation(
        backend,
        input_tangent,
        plan,
        operation,
        context,
    )
}

pub fn rearrange_vjp_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    if let Some(inverse) = plan.inverse_symbolic(operation)? {
        return rearrange_with_context_exact_native_for_operation(
            backend,
            output_gradient,
            &inverse,
            operation,
            context,
        );
    }
    rearrange_vjp_fresh_exact_native(backend, output_gradient, plan, operation, context)
}

fn rearrange_vjp_fresh_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    require_cpu(output_gradient, operation)?;
    if output_gradient.descriptor().shape() != plan.output_shape {
        return Err(invalid(
            operation,
            "gradient shape does not match rearrangement output",
        ));
    }
    let width = usize::try_from(output_gradient.descriptor().dtype().byte_width())
        .map_err(|_| shape_overflow(operation, "dtype width"))?;
    let input_count = element_count(&plan.input_shape, operation, "input gradient")?;
    let byte_count = input_count
        .checked_mul(width)
        .ok_or_else(|| shape_overflow(operation, "input gradient bytes"))?;
    let mut bytes = workspace_filled(
        backend,
        context,
        byte_count,
        0_u8,
    )?;
    let output_count = element_count(&plan.output_shape, operation, "output gradient")?;
    for output_linear in 0..output_count {
        check_periodically(output_linear, context.cancellation)?;
        let output_indices = unravel_index(output_linear, &plan.output_shape, operation)?;
        let source_linear = plan.source_linear_index(output_linear, operation)?;
        let start = source_linear
            .checked_mul(width)
            .ok_or_else(|| shape_overflow(operation, "gradient destination"))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| shape_overflow(operation, "gradient destination"))?;
        let destination = bytes
            .get_mut(start..end)
            .ok_or_else(|| invalid(operation, "gradient destination is outside storage"))?;
        destination.copy_from_slice(output_gradient.element_bytes(&output_indices)?);
    }
    upload_bytes_with_context(
        backend,
        &plan.input_shape,
        output_gradient.descriptor().dtype(),
        &bytes,
        context,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMorphologyOperation {
    Dilation,
    Erosion,
    Opening,
    Closing,
    Gradient,
    TopHat,
    BottomHat,
}

pub fn native_morphology_with_context_exact(
    backend: &CpuBackend,
    input: &Tensor,
    kernel: &Tensor,
    operation: NativeMorphologyOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, morphology_operation_id(operation))?;
    require_f32_cpu(kernel, morphology_operation_id(operation))?;
    require_same_stream(input, kernel)?;
    let [batch, channels, height, width] = input.descriptor().shape() else {
        return Err(invalid(
            morphology_operation_id(operation),
            "morphology expects BCHW rank four input",
        ));
    };
    let [kernel_height, kernel_width] = kernel.descriptor().shape() else {
        return Err(invalid(
            morphology_operation_id(operation),
            "morphology kernel must be rank two",
        ));
    };
    if *kernel_height == 0 || *kernel_width == 0 {
        return Err(invalid(
            morphology_operation_id(operation),
            "morphology kernel must be non-empty",
        ));
    }
    let kernel_count = element_count(
        &[*kernel_height, *kernel_width],
        morphology_operation_id(operation),
        "morphology kernel",
    )?;
    let mut kernel_mask = temporary_vec(
        backend,
        context,
        kernel_count,
    )?;
    for kernel_y in 0..*kernel_height {
        for kernel_x in 0..*kernel_width {
            check_periodically(kernel_mask.len(), context.cancellation)?;
            kernel_mask.try_push(read_f32(kernel, &[kernel_y, kernel_x])? != 0.0)?;
        }
    }
    let shape = [*batch, *channels, *height, *width];
    let source =
        tensor_f32_with_context(backend, input, context, morphology_operation_id(operation))?;
    let primitive = |values: &[f32], dilate: bool| {
        morphology_primitive(
            values,
            shape,
            &kernel_mask,
            (*kernel_height, *kernel_width),
            (*kernel_height / 2, *kernel_width / 2),
            dilate,
            backend,
            context,
            morphology_operation_id(operation),
        )
    };
    let values = match operation {
        NativeMorphologyOperation::Dilation => primitive(&source, true)?,
        NativeMorphologyOperation::Erosion => primitive(&source, false)?,
        NativeMorphologyOperation::Opening => primitive(&primitive(&source, false)?, true)?,
        NativeMorphologyOperation::Closing => primitive(&primitive(&source, true)?, false)?,
        NativeMorphologyOperation::Gradient => subtract_vectors(
            backend,
            context,
            &primitive(&source, true)?,
            &primitive(&source, false)?,
            morphology_operation_id(operation),
        )?,
        NativeMorphologyOperation::TopHat => subtract_vectors(
            backend,
            context,
            &source,
            &primitive(&primitive(&source, false)?, true)?,
            morphology_operation_id(operation),
        )?,
        NativeMorphologyOperation::BottomHat => subtract_vectors(
            backend,
            context,
            &primitive(&primitive(&source, true)?, false)?,
            &source,
            morphology_operation_id(operation),
        )?,
    };
    upload_f32_with_context(backend, &shape, &values, context)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeBiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
}

pub fn biquad_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    coefficients: NativeBiquadCoefficients,
    clamp: bool,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(waveform, operation)?;
    if waveform.descriptor().rank() == 0 || !coefficients_are_valid(coefficients) {
        return Err(invalid(
            operation,
            "invalid waveform or biquad coefficients",
        ));
    }
    let shape = waveform.descriptor().shape();
    let time = usize::try_from(
        *shape
            .last()
            .ok_or_else(|| invalid(operation, "waveform requires a time axis"))?,
    )
    .map_err(|_| shape_overflow(operation, "time length"))?;
    let count = element_count(shape, operation, "waveform elements")?;
    let channels = count.checked_div(time).unwrap_or(0);
    let input = tensor_f32_with_context(backend, waveform, context, operation)?;
    let mut output = workspace_filled(
        backend,
        context,
        count,
        0.0_f32,
    )?;
    let a0 = coefficients.a0;
    for channel in 0..channels {
        let start = channel
            .checked_mul(time)
            .ok_or_else(|| shape_overflow(operation, "channel offset"))?;
        let mut x1 = 0.0_f64;
        let mut x2 = 0.0_f64;
        let mut y1 = 0.0_f64;
        let mut y2 = 0.0_f64;
        for sample in 0..time {
            check_periodically(sample, context.cancellation)?;
            let index = start + sample;
            let x0 = f64::from(input[index]);
            let raw = (coefficients.b0 * x0 + coefficients.b1 * x1 + coefficients.b2 * x2
                - coefficients.a1 * y1
                - coefficients.a2 * y2)
                / a0;
            let y0 = if clamp { raw.clamp(-1.0, 1.0) } else { raw };
            output[index] = y0 as f32;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = raw;
        }
    }
    upload_f32_with_context(backend, shape, &output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn equalizer_biquad_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    center_frequency: f64,
    gain: f64,
    quality: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    validate_audio_parameters(
        sample_rate,
        center_frequency,
        quality,
        EQUALIZER_BIQUAD_OPERATION_ID,
    )?;
    let w0 = 2.0 * std::f64::consts::PI * center_frequency / f64::from(sample_rate);
    let alpha = w0.sin() / (2.0 * quality);
    let amplitude = (gain / 40.0 * std::f64::consts::LN_10).exp();
    let coefficients = NativeBiquadCoefficients {
        b0: 1.0 + alpha * amplitude,
        b1: -2.0 * w0.cos(),
        b2: 1.0 - alpha * amplitude,
        a0: 1.0 + alpha / amplitude,
        a1: -2.0 * w0.cos(),
        a2: 1.0 - alpha / amplitude,
    };
    biquad_with_context_exact_native(
        backend,
        waveform,
        coefficients,
        true,
        EQUALIZER_BIQUAD_OPERATION_ID,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn treble_biquad_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    gain: f64,
    central_frequency: f64,
    quality: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    validate_audio_parameters(
        sample_rate,
        central_frequency,
        quality,
        TREBLE_BIQUAD_OPERATION_ID,
    )?;
    let w0 = 2.0 * std::f64::consts::PI * central_frequency / f64::from(sample_rate);
    let amplitude = (gain / 40.0 * std::f64::consts::LN_10).exp();
    let alpha =
        w0.sin() / 2.0 * ((amplitude + 1.0 / amplitude) * (1.0 / quality - 1.0) + 2.0).sqrt();
    let beta = 2.0 * amplitude.sqrt() * alpha;
    let cosine = w0.cos();
    let coefficients = NativeBiquadCoefficients {
        b0: amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta),
        b1: -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
        b2: amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta),
        a0: (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta,
        a1: 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
        a2: (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta,
    };
    biquad_with_context_exact_native(
        backend,
        waveform,
        coefficients,
        true,
        TREBLE_BIQUAD_OPERATION_ID,
        context,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeResampleConfiguration {
    pub original_frequency: u32,
    pub new_frequency: u32,
    pub lowpass_filter_width: u32,
    pub rolloff: f64,
}

impl NativeResampleConfiguration {
    pub fn torchaudio_default(original_frequency: u32, new_frequency: u32) -> Self {
        Self {
            original_frequency,
            new_frequency,
            lowpass_filter_width: 6,
            rolloff: 0.99,
        }
    }

    fn validate(self) -> Result<Self, ExternalTensorKernelPartOneError> {
        if self.original_frequency == 0
            || self.new_frequency == 0
            || self.lowpass_filter_width == 0
            || !self.rolloff.is_finite()
            || !(0.0..=1.0).contains(&self.rolloff)
            || self.rolloff == 0.0
        {
            return Err(invalid(
                RESAMPLE_OPERATION_ID,
                "invalid native resample configuration",
            ));
        }
        Ok(self)
    }
}

pub fn resample_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    configuration: NativeResampleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let configuration = configuration.validate()?;
    require_f32_cpu(waveform, RESAMPLE_OPERATION_ID)?;
    if waveform.descriptor().rank() == 0 {
        return Err(invalid(
            RESAMPLE_OPERATION_ID,
            "resample expects at least one time dimension",
        ));
    }
    let input_length = usize::try_from(
        *waveform
            .descriptor()
            .shape()
            .last()
            .ok_or_else(|| invalid(RESAMPLE_OPERATION_ID, "missing time axis"))?,
    )
    .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "input length"))?;
    if configuration.original_frequency == configuration.new_frequency {
        return copy_tensor_with_context(backend, waveform, context, RESAMPLE_OPERATION_ID);
    }
    let output_length = resampled_length(input_length, configuration)?;
    let input = tensor_f32_with_context(backend, waveform, context, RESAMPLE_OPERATION_ID)?;
    let channels = input.len().checked_div(input_length).unwrap_or(0);
    let output_count = channels
        .checked_mul(output_length)
        .ok_or_else(|| shape_overflow(RESAMPLE_OPERATION_ID, "output elements"))?;
    let mut output = workspace_filled(
        backend,
        context,
        output_count,
        0.0_f32,
    )?;
    for channel in 0..channels {
        for output_index in 0..output_length {
            check_periodically(output_index, context.cancellation)?;
            let mut value = 0.0_f64;
            let weights =
                resample_weights(backend, context, input_length, output_index, configuration)?;
            for &(input_index, weight) in weights.iter() {
                value += f64::from(input[channel * input_length + input_index]) * weight;
            }
            output[channel * output_length + output_index] = value as f32;
        }
    }
    let mut shape = waveform.descriptor().shape().to_vec();
    let last = shape
        .last_mut()
        .ok_or_else(|| invalid(RESAMPLE_OPERATION_ID, "missing time axis"))?;
    *last = u64::try_from(output_length)
        .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "output length"))?;
    upload_f32_with_context(backend, &shape, &output, context)
}

pub fn resample_jvp_with_context_exact_native(
    backend: &CpuBackend,
    waveform_tangent: &Tensor,
    configuration: NativeResampleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    resample_with_context_exact_native(backend, waveform_tangent, configuration, context)
}

pub fn resample_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input_shape: &[u64],
    output_gradient: &Tensor,
    configuration: NativeResampleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let configuration = configuration.validate()?;
    require_f32_cpu(output_gradient, RESAMPLE_OPERATION_ID)?;
    if input_shape.is_empty() {
        return Err(invalid(
            RESAMPLE_OPERATION_ID,
            "resample VJP requires an input time axis",
        ));
    }
    let input_length = usize::try_from(
        *input_shape
            .last()
            .ok_or_else(|| invalid(RESAMPLE_OPERATION_ID, "missing input time axis"))?,
    )
    .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "input length"))?;
    let output_length = resampled_length(input_length, configuration)?;
    let mut expected = input_shape.to_vec();
    *expected
        .last_mut()
        .ok_or_else(|| invalid(RESAMPLE_OPERATION_ID, "missing output time axis"))? =
        u64::try_from(output_length)
            .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "output length"))?;
    if output_gradient.descriptor().shape() != expected {
        return Err(invalid(
            RESAMPLE_OPERATION_ID,
            "output gradient shape does not match resample output",
        ));
    }
    let output_values =
        tensor_f32_with_context(backend, output_gradient, context, RESAMPLE_OPERATION_ID)?;
    let channels = output_values.len().checked_div(output_length).unwrap_or(0);
    let gradient_count = channels
        .checked_mul(input_length)
        .ok_or_else(|| shape_overflow(RESAMPLE_OPERATION_ID, "input gradient elements"))?;
    let mut gradient = workspace_filled(
        backend,
        context,
        gradient_count,
        0.0_f32,
    )?;
    for channel in 0..channels {
        for output_index in 0..output_length {
            check_periodically(output_index, context.cancellation)?;
            let upstream = output_values[channel * output_length + output_index];
            let weights =
                resample_weights(backend, context, input_length, output_index, configuration)?;
            for &(input_index, weight) in weights.iter() {
                gradient[channel * input_length + input_index] += upstream * weight as f32;
            }
        }
    }
    upload_f32_with_context(backend, input_shape, &gradient, context)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMelScale {
    Htk,
    Slaney,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMelNormalization {
    None,
    Slaney,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeMelSpectrogramConfiguration {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: Option<usize>,
    pub hop_length: Option<usize>,
    pub f_min: f64,
    pub f_max: Option<f64>,
    pub n_mels: usize,
    pub power: f32,
    pub center: bool,
    pub normalized: bool,
    pub mel_scale: NativeMelScale,
    pub mel_normalization: NativeMelNormalization,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeMelScaleConfiguration {
    pub n_mels: usize,
    pub sample_rate: u32,
    pub f_min: f64,
    pub f_max: Option<f64>,
    pub n_stft: usize,
    pub mel_scale: NativeMelScale,
    pub mel_normalization: NativeMelNormalization,
}

pub fn mel_scale_project_with_context_exact_native(
    backend: &CpuBackend,
    spectrogram: &Tensor,
    configuration: NativeMelScaleConfiguration,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    validate_mel_scale_configuration(configuration, operation)?;
    require_f32_cpu(spectrogram, operation)?;
    let shape = spectrogram.descriptor().shape();
    if shape.len() < 2 {
        return Err(invalid(
            operation,
            "mel scale requires [..., frequency, frame] input",
        ));
    }
    let frequency_axis = shape.len() - 2;
    let frequency_count = usize::try_from(shape[frequency_axis])
        .map_err(|_| shape_overflow(operation, "frequency bins"))?;
    if frequency_count != configuration.n_stft {
        return Err(invalid(
            operation,
            "spectrogram frequency axis does not match n_stft",
        ));
    }
    let frame_count =
        usize::try_from(shape[shape.len() - 1]).map_err(|_| shape_overflow(operation, "frames"))?;
    let leading_count = shape[..frequency_axis]
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| shape_overflow(operation, "leading spectrogram batch"))?;
    let filters = mel_filter_bank(backend, context, frequency_count, configuration, operation)?;
    let input = tensor_f32_with_context(backend, spectrogram, context, operation)?;
    let output_count = leading_count
        .checked_mul(configuration.n_mels)
        .and_then(|count| count.checked_mul(frame_count))
        .ok_or_else(|| shape_overflow(operation, "mel output elements"))?;
    let mut output = workspace_filled(
        backend,
        context,
        output_count,
        0.0_f32,
    )?;
    for leading in 0..leading_count {
        for mel in 0..configuration.n_mels {
            for frame in 0..frame_count {
                check_periodically(frame, context.cancellation)?;
                let mut value = 0.0_f32;
                for frequency in 0..frequency_count {
                    let source = (leading * frequency_count + frequency) * frame_count + frame;
                    value += filters[mel * frequency_count + frequency] * input[source];
                }
                output[(leading * configuration.n_mels + mel) * frame_count + frame] = value;
            }
        }
    }
    let mut output_shape = shape.to_vec();
    output_shape[frequency_axis] =
        u64::try_from(configuration.n_mels).map_err(|_| shape_overflow(operation, "mel bins"))?;
    upload_f32_with_context(backend, &output_shape, &output, context)
}

pub fn mel_scale_project_vjp_with_context_exact_native(
    backend: &CpuBackend,
    spectrogram: &Tensor,
    output_gradient: &Tensor,
    configuration: NativeMelScaleConfiguration,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    validate_mel_scale_configuration(configuration, operation)?;
    require_f32_cpu(spectrogram, operation)?;
    require_f32_cpu(output_gradient, operation)?;
    let input_shape = spectrogram.descriptor().shape();
    if input_shape.len() < 2 {
        return Err(invalid(
            operation,
            "mel scale requires [..., frequency, frame] input",
        ));
    }
    let frequency_axis = input_shape.len() - 2;
    let frequency_count = usize::try_from(input_shape[frequency_axis])
        .map_err(|_| shape_overflow(operation, "frequency bins"))?;
    if frequency_count != configuration.n_stft {
        return Err(invalid(
            operation,
            "spectrogram frequency axis does not match n_stft",
        ));
    }
    let mut expected_output_shape = input_shape.to_vec();
    expected_output_shape[frequency_axis] =
        u64::try_from(configuration.n_mels).map_err(|_| shape_overflow(operation, "mel bins"))?;
    if output_gradient.descriptor().shape() != expected_output_shape
        || output_gradient.descriptor().stream() != spectrogram.descriptor().stream()
    {
        return Err(invalid(
            operation,
            "output gradient shape or stream does not match mel-scale output",
        ));
    }
    let frame_count = usize::try_from(input_shape[input_shape.len() - 1])
        .map_err(|_| shape_overflow(operation, "frames"))?;
    let leading_count = input_shape[..frequency_axis]
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| shape_overflow(operation, "leading spectrogram batch"))?;
    let filters = mel_filter_bank(backend, context, frequency_count, configuration, operation)?;
    let upstream = tensor_f32_with_context(backend, output_gradient, context, operation)?;
    let input_count = element_count(input_shape, operation, "mel input gradient")?;
    let mut gradient = workspace_filled(
        backend,
        context,
        input_count,
        0.0_f32,
    )?;
    for leading in 0..leading_count {
        for mel in 0..configuration.n_mels {
            for frame in 0..frame_count {
                check_periodically(frame, context.cancellation)?;
                let value = upstream[(leading * configuration.n_mels + mel) * frame_count + frame];
                for frequency in 0..frequency_count {
                    let destination = (leading * frequency_count + frequency) * frame_count + frame;
                    gradient[destination] += filters[mel * frequency_count + frequency] * value;
                }
            }
        }
    }
    upload_f32_with_context(backend, input_shape, &gradient, context)
}

pub fn mel_spectrogram_with_context_exact_native(
    backend: &CpuBackend,
    waveform: &Tensor,
    configuration: NativeMelSpectrogramConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    validate_mel_configuration(configuration)?;
    require_f32_cpu(waveform, MEL_SPECTROGRAM_OPERATION_ID)?;
    if waveform.descriptor().rank() == 0 {
        return Err(invalid(
            MEL_SPECTROGRAM_OPERATION_ID,
            "mel spectrogram requires a time axis",
        ));
    }
    let waveform_shape = waveform.descriptor().shape();
    let leading_shape = waveform_shape[..waveform_shape.len() - 1].to_vec();
    let time = *waveform_shape
        .last()
        .ok_or_else(|| invalid(MEL_SPECTROGRAM_OPERATION_ID, "missing time axis"))?;
    let flattened_batch = leading_shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "flattened audio batch"))?;
    let stft_input = if waveform.descriptor().rank() <= 2 {
        waveform.clone()
    } else {
        copy_tensor_with_context(backend, waveform, context, MEL_SPECTROGRAM_OPERATION_ID)?.view(
            TensorDescriptor::contiguous(
                vec![flattened_batch, time],
                DType::F32,
                DeviceId::CPU,
                waveform.descriptor().stream(),
            )?,
            crate::ViewAccess::ReadOnly,
        )?
    };
    let win_length = configuration.win_length.unwrap_or(configuration.n_fft);
    let window = hann_window_with_context_exact_native(
        backend,
        win_length,
        true,
        DType::F32,
        waveform.descriptor().stream(),
        context,
    )
    .map_err(|error| invalid(MEL_SPECTROGRAM_OPERATION_ID, error.to_string()))?;
    let spectrum = stft_with_context_exact_native(
        backend,
        &stft_input,
        configuration.n_fft,
        configuration.hop_length,
        Some(win_length),
        Some(&window),
        configuration.center,
        configuration.normalized,
        true,
        context,
    )
    .map_err(|error| invalid(MEL_SPECTROGRAM_OPERATION_ID, error.to_string()))?;
    let spectrum_shape = spectrum.descriptor().shape();
    let (batch, frequencies, frames) = match spectrum_shape {
        [frequencies, frames] => (1, *frequencies, *frames),
        [batch, frequencies, frames] => (*batch, *frequencies, *frames),
        _ => {
            return Err(invalid(
                MEL_SPECTROGRAM_OPERATION_ID,
                "native STFT returned an unexpected shape",
            ));
        }
    };
    let frequency_count = usize::try_from(frequencies)
        .map_err(|_| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "frequency bins"))?;
    let filter_bank = mel_filter_bank(
        backend,
        context,
        frequency_count,
        NativeMelScaleConfiguration {
            n_mels: configuration.n_mels,
            sample_rate: configuration.sample_rate,
            f_min: configuration.f_min,
            f_max: configuration.f_max,
            n_stft: frequency_count,
            mel_scale: configuration.mel_scale,
            mel_normalization: configuration.mel_normalization,
        },
        MEL_SPECTROGRAM_OPERATION_ID,
    )?;
    let output_count = usize::try_from(batch)
        .ok()
        .and_then(|value| value.checked_mul(configuration.n_mels))
        .and_then(|value| value.checked_mul(usize::try_from(frames).ok()?))
        .ok_or_else(|| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "mel output elements"))?;
    let mut output = workspace_filled(
        backend,
        context,
        output_count,
        0.0_f32,
    )?;
    let frame_count = usize::try_from(frames)
        .map_err(|_| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "frames"))?;
    for batch_index in 0..usize::try_from(batch)
        .map_err(|_| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "batch"))?
    {
        for mel in 0..configuration.n_mels {
            for frame in 0..frame_count {
                check_periodically(frame, context.cancellation)?;
                let mut value = 0.0_f32;
                for frequency in 0..frequency_count {
                    let spectrum_indices = if spectrum_shape.len() == 2 {
                        vec![frequency as u64, frame as u64]
                    } else {
                        vec![batch_index as u64, frequency as u64, frame as u64]
                    };
                    let (real, imaginary) = read_complex64(&spectrum, &spectrum_indices)?;
                    let magnitude = real.hypot(imaginary).powf(configuration.power);
                    value += filter_bank[mel * frequency_count + frequency] * magnitude;
                }
                output[(batch_index * configuration.n_mels + mel) * frame_count + frame] = value;
            }
        }
    }
    let mut output_shape = leading_shape;
    output_shape.push(
        u64::try_from(configuration.n_mels)
            .map_err(|_| shape_overflow(MEL_SPECTROGRAM_OPERATION_ID, "mel bins"))?,
    );
    output_shape.push(frames);
    upload_f32_with_context(backend, &output_shape, &output, context)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoiAlignConfiguration {
    pub output_height: u64,
    pub output_width: u64,
    pub spatial_scale_numerator: u32,
    pub spatial_scale_denominator: u32,
    pub sampling_ratio: i32,
    pub aligned: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBilinearBoundary {
    RoiAlign,
    ZeroPadding,
    Border,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeLinearBoundary {
    ZeroPadding,
    Border,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeLinearWeight {
    pub source: u64,
    pub weight: f32,
    pub derivative: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeBilinearWeight {
    pub source_y: u64,
    pub source_x: u64,
    pub weight: f32,
    pub derivative_y: f32,
    pub derivative_x: f32,
}

pub fn checked_bilinear_weights(
    height: u64,
    width: u64,
    y: f32,
    x: f32,
    boundary: NativeBilinearBoundary,
    operation: &'static str,
) -> Result<Vec<NativeBilinearWeight>, ExternalTensorKernelPartOneError> {
    if height == 0 || width == 0 {
        return Err(invalid(
            operation,
            "bilinear sampling requires non-zero dimensions",
        ));
    }
    if !y.is_finite() || !x.is_finite() {
        return Err(invalid(operation, "bilinear coordinates must be finite"));
    }
    match boundary {
        NativeBilinearBoundary::RoiAlign => {
            roi_align_bilinear_weights(height, width, y, x, operation)
        }
        NativeBilinearBoundary::ZeroPadding => {
            zero_padded_bilinear_weights(height, width, y, x, operation)
        }
        NativeBilinearBoundary::Border => border_bilinear_weights(height, width, y, x, operation),
    }
}

pub fn checked_linear_weights(
    extent: u64,
    coordinate: f32,
    boundary: NativeLinearBoundary,
    operation: &'static str,
) -> Result<Vec<NativeLinearWeight>, ExternalTensorKernelPartOneError> {
    if extent == 0 {
        return Err(invalid(
            operation,
            "linear sampling requires a non-zero dimension",
        ));
    }
    if !coordinate.is_finite() {
        return Err(invalid(operation, "linear coordinate must be finite"));
    }
    let (coordinate, coordinate_derivative) = match boundary {
        NativeLinearBoundary::ZeroPadding => (coordinate, 1.0),
        NativeLinearBoundary::Border => {
            let upper = (extent - 1) as f32;
            (
                coordinate.clamp(0.0, upper),
                if coordinate > 0.0 && coordinate < upper {
                    1.0
                } else {
                    0.0
                },
            )
        }
    };
    let low = checked_floor_i64(coordinate, operation)?;
    let high = low
        .checked_add(1)
        .ok_or_else(|| shape_overflow(operation, "linear sampling coordinate"))?;
    let fraction = coordinate - low as f32;
    let extent = i64::try_from(extent)
        .map_err(|_| shape_overflow(operation, "linear sampling extent"))?;
    [(low, 1.0 - fraction, -coordinate_derivative),
        (high, fraction, coordinate_derivative)]
        .into_iter()
        .filter(|(source, ..)| *source >= 0 && *source < extent)
        .map(|(source, weight, derivative)| {
            Ok(NativeLinearWeight {
                source: u64::try_from(source)
                    .map_err(|_| shape_overflow(operation, "linear source index"))?,
                weight,
                derivative,
            })
        })
        .collect()
}

impl RoiAlignConfiguration {
    fn spatial_scale(self) -> Result<f32, ExternalTensorKernelPartOneError> {
        if self.output_height == 0
            || self.output_width == 0
            || self.spatial_scale_denominator == 0
            || self.sampling_ratio < -1
        {
            return Err(invalid(
                ROI_ALIGN_OPERATION_ID,
                "invalid ROI Align configuration",
            ));
        }
        Ok(self.spatial_scale_numerator as f32 / self.spatial_scale_denominator as f32)
    }
}

pub fn roi_align_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    boxes_by_batch: &[Tensor],
    configuration: RoiAlignConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let plan = RoiAlignPlan::new(backend, input, boxes_by_batch, configuration, context)?;
    let mut output = workspace_filled(
        backend,
        context,
        plan.output_element_count()?,
        0.0_f32,
    )?;
    for roi in 0..plan.rois.len() {
        for channel in 0..plan.channels {
            for output_y in 0..plan.output_height {
                for output_x in 0..plan.output_width {
                    check_periodically_u64(output_x, context.cancellation)?;
                    let samples = plan.samples(backend, context, roi, output_y, output_x)?;
                    let mut value = 0.0_f32;
                    for &(y, x) in samples.iter() {
                        value += roi_bilinear_value(input, plan.rois[roi].batch, channel, y, x)?;
                    }
                    let denominator = samples.len().max(1) as f32;
                    output[plan.output_linear(roi, channel, output_y, output_x)?] =
                        value / denominator;
                }
            }
        }
    }
    upload_f32_with_context(backend, &plan.output_shape()?, &output, context)
}

pub fn roi_align_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    boxes_by_batch: &[Tensor],
    configuration: RoiAlignConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    roi_align_with_context_exact_native(
        backend,
        input_tangent,
        boxes_by_batch,
        configuration,
        context,
    )
}

pub fn roi_align_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    boxes_by_batch: &[Tensor],
    output_gradient: &Tensor,
    configuration: RoiAlignConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let plan = RoiAlignPlan::new(backend, input, boxes_by_batch, configuration, context)?;
    require_f32_cpu(output_gradient, ROI_ALIGN_OPERATION_ID)?;
    require_same_stream(input, output_gradient)?;
    if output_gradient.descriptor().shape() != plan.output_shape()? {
        return Err(invalid(
            ROI_ALIGN_OPERATION_ID,
            "ROI Align gradient shape does not match output",
        ));
    }
    let gradient_count = element_count(
        input.descriptor().shape(),
        ROI_ALIGN_OPERATION_ID,
        "input gradient",
    )?;
    let mut gradient = workspace_filled(
        backend,
        context,
        gradient_count,
        0.0_f32,
    )?;
    for roi in 0..plan.rois.len() {
        for channel in 0..plan.channels {
            for output_y in 0..plan.output_height {
                for output_x in 0..plan.output_width {
                    check_periodically_u64(output_x, context.cancellation)?;
                    let output_indices = [roi as u64, channel, output_y, output_x];
                    let upstream = read_f32(output_gradient, &output_indices)?;
                    let samples = plan.samples(backend, context, roi, output_y, output_x)?;
                    let scale = upstream / samples.len().max(1) as f32;
                    for &(y, x) in samples.iter() {
                        for sample in checked_bilinear_weights(
                            plan.height,
                            plan.width,
                            y,
                            x,
                            NativeBilinearBoundary::RoiAlign,
                            ROI_ALIGN_OPERATION_ID,
                        )? {
                            let source = ravel_index(
                                &[
                                    plan.rois[roi].batch,
                                    channel,
                                    sample.source_y,
                                    sample.source_x,
                                ],
                                input.descriptor().shape(),
                                ROI_ALIGN_OPERATION_ID,
                            )?;
                            gradient[source] += scale * sample.weight;
                        }
                    }
                }
            }
        }
    }
    upload_f32_with_context(backend, input.descriptor().shape(), &gradient, context)
}

pub fn normalize_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mean: &[f32],
    standard_deviation: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, NORMALIZE_OPERATION_ID)?;
    if input.descriptor().rank() < 3 {
        return Err(invalid(
            NORMALIZE_OPERATION_ID,
            "normalize expects [..., C, H, W] input",
        ));
    }
    let channel_axis = input.descriptor().rank() - 3;
    let channels = usize::try_from(input.descriptor().shape()[channel_axis])
        .map_err(|_| shape_overflow(NORMALIZE_OPERATION_ID, "channels"))?;
    if !matches!(mean.len(), 1) && mean.len() != channels {
        return Err(invalid(
            NORMALIZE_OPERATION_ID,
            "mean must have one value or one value per channel",
        ));
    }
    if !matches!(standard_deviation.len(), 1) && standard_deviation.len() != channels {
        return Err(invalid(
            NORMALIZE_OPERATION_ID,
            "standard deviation must have one value or one value per channel",
        ));
    }
    if standard_deviation
        .iter()
        .any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err(invalid(
            NORMALIZE_OPERATION_ID,
            "standard deviation must contain finite non-zero values",
        ));
    }
    let count = element_count(
        input.descriptor().shape(),
        NORMALIZE_OPERATION_ID,
        "normalized elements",
    )?;
    let mut output = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape(), NORMALIZE_OPERATION_ID)?;
        let channel = usize::try_from(indices[channel_axis])
            .map_err(|_| shape_overflow(NORMALIZE_OPERATION_ID, "channel index"))?;
        let mean = mean[if mean.len() == 1 { 0 } else { channel }];
        let standard_deviation = standard_deviation[if standard_deviation.len() == 1 {
            0
        } else {
            channel
        }];
        output.try_push((read_f32(input, &indices)? - mean) / standard_deviation)?;
    }
    upload_f32_with_context(backend, input.descriptor().shape(), &output, context)
}

pub fn normalize_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    standard_deviation: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let zero_mean = workspace_filled(
        backend,
        context,
        standard_deviation.len(),
        0.0_f32,
    )?;
    normalize_with_context_exact_native(
        backend,
        input_tangent,
        &zero_mean,
        standard_deviation,
        context,
    )
}

pub fn normalize_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    standard_deviation: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    normalize_jvp_with_context_exact_native(backend, output_gradient, standard_deviation, context)
}

#[allow(clippy::too_many_arguments)]
pub fn resize_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_height: u64,
    output_width: u64,
    mode: ResizeMode,
    antialias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    resize_with_coordinate_transform_with_context_exact_native(
        backend,
        input,
        output_height,
        output_width,
        mode,
        antialias,
        false,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resize_with_coordinate_transform_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_height: u64,
    output_width: u64,
    mode: ResizeMode,
    antialias: bool,
    align_corners: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, RESIZE_OPERATION_ID)?;
    if input.descriptor().rank() < 3 || output_height == 0 || output_width == 0 {
        return Err(invalid(
            RESIZE_OPERATION_ID,
            "resize expects [..., C, H, W] and non-zero output dimensions",
        ));
    }
    if !matches!(input.descriptor().dtype(), DType::F32 | DType::U8) {
        return Err(ExternalTensorKernelPartOneError::UnsupportedDType {
            operation: RESIZE_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    if antialias && !matches!(mode, ResizeMode::Bilinear | ResizeMode::Bicubic) {
        return Err(invalid(
            RESIZE_OPERATION_ID,
            "antialias is supported only for bilinear and bicubic resize",
        ));
    }
    let rank = input.descriptor().rank();
    let channels = input.descriptor().shape()[rank - 3];
    let input_height = input.descriptor().shape()[rank - 2];
    let input_width = input.descriptor().shape()[rank - 1];
    let batch = input.descriptor().shape()[..rank - 3]
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| shape_overflow(RESIZE_OPERATION_ID, "flattened batch"))?;
    let f32_input = cast_image_input_to_f32(backend, input, context)?;
    let input_descriptor = TensorDescriptor::contiguous(
        vec![batch, channels, input_height, input_width],
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let reshaped = f32_input.view(input_descriptor, crate::ViewAccess::ReadOnly)?;
    let output_descriptor = TensorDescriptor::contiguous(
        vec![batch, channels, output_height, output_width],
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let (resized, _) = backend.resize(
        ResizeSpec {
            width: output_width,
            height: output_height,
            mode,
            crop: ResizeCrop::Disabled,
            antialias,
            align_corners,
        },
        &reshaped,
        output_descriptor,
        context,
    )?;
    let mut output_shape = input.descriptor().shape().to_vec();
    output_shape[rank - 2] = output_height;
    output_shape[rank - 1] = output_width;
    if input.descriptor().dtype() == DType::F32 {
        return Ok(resized.view(
            TensorDescriptor::contiguous(
                output_shape,
                DType::F32,
                DeviceId::CPU,
                input.descriptor().stream(),
            )?,
            crate::ViewAccess::Writable,
        )?);
    }
    let values = tensor_f32_with_context(backend, &resized, context, RESIZE_OPERATION_ID)?;
    let mut bytes = temporary_vec(
        backend,
        context,
        values.len(),
    )?;
    for value in values.iter() {
        bytes.try_push(value.round().clamp(0.0, 255.0) as u8)?;
    }
    upload_bytes_with_context(backend, &output_shape, DType::U8, &bytes, context)
}

#[allow(clippy::too_many_arguments)]
pub fn to_tensor_with_context_exact_native(
    backend: &CpuBackend,
    image_hwc_u8: &[u8],
    height: u64,
    width: u64,
    channels: u64,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    to_tensor_with_staging(
        backend,
        image_hwc_u8,
        height,
        width,
        channels,
        stream,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn to_tensor_with_staging(
    backend: &CpuBackend,
    image_hwc_u8: &[u8],
    height: u64,
    width: u64,
    channels: u64,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    if context.stream != stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: stream,
        }
        .into());
    }
    if height == 0 || width == 0 || !matches!(channels, 1 | 3 | 4) {
        return Err(invalid(
            TO_TENSOR_OPERATION_ID,
            "native image must have non-zero dimensions and 1, 3, or 4 channels",
        ));
    }
    let expected = usize::try_from(height)
        .ok()
        .and_then(|value| value.checked_mul(usize::try_from(width).ok()?))
        .and_then(|value| value.checked_mul(usize::try_from(channels).ok()?))
        .ok_or_else(|| shape_overflow(TO_TENSOR_OPERATION_ID, "image bytes"))?;
    if image_hwc_u8.len() != expected {
        return Err(invalid(
            TO_TENSOR_OPERATION_ID,
            "native image byte length does not match HWC dimensions",
        ));
    }
    let count = expected;
    let mut output = workspace_filled(backend, context, count, 0.0_f32)?;
    let height_usize =
        usize::try_from(height).map_err(|_| shape_overflow(TO_TENSOR_OPERATION_ID, "height"))?;
    let width_usize =
        usize::try_from(width).map_err(|_| shape_overflow(TO_TENSOR_OPERATION_ID, "width"))?;
    let channels_usize = usize::try_from(channels)
        .map_err(|_| shape_overflow(TO_TENSOR_OPERATION_ID, "channels"))?;
    for channel in 0..channels_usize {
        for y in 0..height_usize {
            for x in 0..width_usize {
                check_periodically(x, context.cancellation)?;
                let source = (y * width_usize + x) * channels_usize + channel;
                let destination = (channel * height_usize + y) * width_usize + x;
                output[destination] = f32::from(image_hwc_u8[source]) / 255.0;
            }
        }
    }
    upload_f32_with_context(backend, &[channels, height, width], &output, context)
}

#[derive(Clone, Copy)]
struct NativeRoi {
    batch: u64,
    start_y: f32,
    start_x: f32,
    height: f32,
    width: f32,
}

struct RoiAlignPlan {
    rois: CpuWorkspaceVec<NativeRoi>,
    channels: u64,
    height: u64,
    width: u64,
    output_height: u64,
    output_width: u64,
    sampling_ratio: i32,
}

impl RoiAlignPlan {
    fn new(
        backend: &CpuBackend,
        input: &Tensor,
        boxes_by_batch: &[Tensor],
        configuration: RoiAlignConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ExternalTensorKernelPartOneError> {
        context.cancellation.check()?;
        require_f32_cpu(input, ROI_ALIGN_OPERATION_ID)?;
        let [batch, channels, height, width] = input.descriptor().shape() else {
            return Err(invalid(
                ROI_ALIGN_OPERATION_ID,
                "ROI Align input must be NCHW rank four",
            ));
        };
        if boxes_by_batch.len()
            != usize::try_from(*batch)
                .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "batch"))?
        {
            return Err(invalid(
                ROI_ALIGN_OPERATION_ID,
                "ROI box list must contain one tensor per batch",
            ));
        }
        let scale = configuration.spatial_scale()?;
        let offset = if configuration.aligned { 0.5 } else { 0.0 };
        let roi_capacity = boxes_by_batch.iter().try_fold(0_usize, |total, boxes| {
            let count = boxes.descriptor().shape().first().copied().unwrap_or(0);
            let count = usize::try_from(count)
                .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "ROI count"))?;
            total
                .checked_add(count)
                .ok_or_else(|| shape_overflow(ROI_ALIGN_OPERATION_ID, "ROI count"))
        })?;
        let mut rois = temporary_vec(
            backend,
            context,
            roi_capacity,
        )?;
        for (batch_index, boxes) in boxes_by_batch.iter().enumerate() {
            require_f32_cpu(boxes, ROI_ALIGN_OPERATION_ID)?;
            require_same_stream(input, boxes)?;
            let [box_count, coordinates] = boxes.descriptor().shape() else {
                return Err(invalid(
                    ROI_ALIGN_OPERATION_ID,
                    "ROI boxes must have shape [K, 4]",
                ));
            };
            if *coordinates != 4 {
                return Err(invalid(
                    ROI_ALIGN_OPERATION_ID,
                    "ROI boxes must have four coordinates",
                ));
            }
            for box_index in 0..*box_count {
                check_periodically_u64(box_index, context.cancellation)?;
                let start_x = read_f32(boxes, &[box_index, 0])? * scale - offset;
                let start_y = read_f32(boxes, &[box_index, 1])? * scale - offset;
                let end_x = read_f32(boxes, &[box_index, 2])? * scale - offset;
                let end_y = read_f32(boxes, &[box_index, 3])? * scale - offset;
                if ![start_x, start_y, end_x, end_y]
                    .iter()
                    .all(|value| value.is_finite())
                {
                    return Err(invalid(
                        ROI_ALIGN_OPERATION_ID,
                        "ROI coordinates must be finite",
                    ));
                }
                let mut roi_width = end_x - start_x;
                let mut roi_height = end_y - start_y;
                if !configuration.aligned {
                    roi_width = roi_width.max(1.0);
                    roi_height = roi_height.max(1.0);
                }
                rois.try_push(NativeRoi {
                    batch: u64::try_from(batch_index)
                        .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "batch index"))?,
                    start_y,
                    start_x,
                    height: roi_height,
                    width: roi_width,
                })?;
            }
        }
        Ok(Self {
            rois,
            channels: *channels,
            height: *height,
            width: *width,
            output_height: configuration.output_height,
            output_width: configuration.output_width,
            sampling_ratio: configuration.sampling_ratio,
        })
    }

    fn output_shape(&self) -> Result<Vec<u64>, ExternalTensorKernelPartOneError> {
        Ok(vec![
            u64::try_from(self.rois.len())
                .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "ROI count"))?,
            self.channels,
            self.output_height,
            self.output_width,
        ])
    }

    fn output_element_count(&self) -> Result<usize, ExternalTensorKernelPartOneError> {
        element_count(&self.output_shape()?, ROI_ALIGN_OPERATION_ID, "ROI output")
    }

    fn output_linear(
        &self,
        roi: usize,
        channel: u64,
        output_y: u64,
        output_x: u64,
    ) -> Result<usize, ExternalTensorKernelPartOneError> {
        ravel_index(
            &[roi as u64, channel, output_y, output_x],
            &self.output_shape()?,
            ROI_ALIGN_OPERATION_ID,
        )
    }

    fn samples(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        roi_index: usize,
        output_y: u64,
        output_x: u64,
    ) -> Result<CpuWorkspaceVec<(f32, f32)>, ExternalTensorKernelPartOneError> {
        let roi = self
            .rois
            .get(roi_index)
            .ok_or_else(|| invalid(ROI_ALIGN_OPERATION_ID, "ROI index is outside plan"))?;
        let grid_height = if self.sampling_ratio > 0 {
            u64::try_from(self.sampling_ratio)
                .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "sampling ratio"))?
        } else {
            (roi.height / self.output_height as f32).ceil().max(1.0) as u64
        };
        let grid_width = if self.sampling_ratio > 0 {
            u64::try_from(self.sampling_ratio)
                .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "sampling ratio"))?
        } else {
            (roi.width / self.output_width as f32).ceil().max(1.0) as u64
        };
        let sample_count = usize::try_from(
            grid_height
                .checked_mul(grid_width)
                .ok_or_else(|| shape_overflow(ROI_ALIGN_OPERATION_ID, "sampling grid"))?,
        )
        .map_err(|_| shape_overflow(ROI_ALIGN_OPERATION_ID, "sampling grid"))?;
        let mut samples = temporary_vec(
            backend,
            context,
            sample_count,
        )?;
        let bin_height = roi.height / self.output_height as f32;
        let bin_width = roi.width / self.output_width as f32;
        for sample_y in 0..grid_height {
            for sample_x in 0..grid_width {
                let y = roi.start_y
                    + (output_y as f32 + (sample_y as f32 + 0.5) / grid_height as f32) * bin_height;
                let x = roi.start_x
                    + (output_x as f32 + (sample_x as f32 + 0.5) / grid_width as f32) * bin_width;
                samples.try_push((y, x))?;
            }
        }
        Ok(samples)
    }
}

fn roi_bilinear_value(
    input: &Tensor,
    batch: u64,
    channel: u64,
    y: f32,
    x: f32,
) -> Result<f32, ExternalTensorKernelPartOneError> {
    let [_, _, height, width] = input.descriptor().shape() else {
        return Err(invalid(ROI_ALIGN_OPERATION_ID, "ROI input rank changed"));
    };
    let mut value = 0.0_f32;
    for sample in checked_bilinear_weights(
        *height,
        *width,
        y,
        x,
        NativeBilinearBoundary::RoiAlign,
        ROI_ALIGN_OPERATION_ID,
    )? {
        value +=
            read_f32(input, &[batch, channel, sample.source_y, sample.source_x])? * sample.weight;
    }
    Ok(value)
}

fn roi_align_bilinear_weights(
    height: u64,
    width: u64,
    y: f32,
    x: f32,
    operation: &'static str,
) -> Result<Vec<NativeBilinearWeight>, ExternalTensorKernelPartOneError> {
    if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
        return Ok(Vec::new());
    }
    let y_clamped_low = y < 0.0;
    let x_clamped_low = x < 0.0;
    let y = y.max(0.0);
    let x = x.max(0.0);
    let mut y_low = checked_floor_u64(y, operation)?;
    let mut x_low = checked_floor_u64(x, operation)?;
    let (y_high, y_fraction, y_derivative) = if y_low >= height - 1 {
        y_low = height - 1;
        (y_low, 0.0, 0.0)
    } else {
        (
            y_low + 1,
            y - y_low as f32,
            if y_clamped_low { 0.0 } else { 1.0 },
        )
    };
    let (x_high, x_fraction, x_derivative) = if x_low >= width - 1 {
        x_low = width - 1;
        (x_low, 0.0, 0.0)
    } else {
        (
            x_low + 1,
            x - x_low as f32,
            if x_clamped_low { 0.0 } else { 1.0 },
        )
    };
    Ok(bilinear_weight_quad(
        y_low,
        y_high,
        x_low,
        x_high,
        y_fraction,
        x_fraction,
        y_derivative,
        x_derivative,
    ))
}

fn zero_padded_bilinear_weights(
    height: u64,
    width: u64,
    y: f32,
    x: f32,
    operation: &'static str,
) -> Result<Vec<NativeBilinearWeight>, ExternalTensorKernelPartOneError> {
    if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
        return Ok(Vec::new());
    }
    let y_low = checked_floor_i64(y, operation)?;
    let x_low = checked_floor_i64(x, operation)?;
    let y_high = y_low
        .checked_add(1)
        .ok_or_else(|| shape_overflow(operation, "bilinear y coordinate"))?;
    let x_high = x_low
        .checked_add(1)
        .ok_or_else(|| shape_overflow(operation, "bilinear x coordinate"))?;
    let y_fraction = y - y_low as f32;
    let x_fraction = x - x_low as f32;
    let candidates = [
        (y_low, x_low, 1.0 - y_fraction, 1.0 - x_fraction, -1.0, -1.0),
        (y_low, x_high, 1.0 - y_fraction, x_fraction, -1.0, 1.0),
        (y_high, x_low, y_fraction, 1.0 - x_fraction, 1.0, -1.0),
        (y_high, x_high, y_fraction, x_fraction, 1.0, 1.0),
    ];
    let height =
        i64::try_from(height).map_err(|_| shape_overflow(operation, "bilinear image height"))?;
    let width =
        i64::try_from(width).map_err(|_| shape_overflow(operation, "bilinear image width"))?;
    candidates
        .into_iter()
        .filter(|(source_y, source_x, ..)| {
            *source_y >= 0 && *source_y < height && *source_x >= 0 && *source_x < width
        })
        .map(|(source_y, source_x, y_weight, x_weight, y_sign, x_sign)| {
            Ok(NativeBilinearWeight {
                source_y: u64::try_from(source_y)
                    .map_err(|_| shape_overflow(operation, "bilinear source y"))?,
                source_x: u64::try_from(source_x)
                    .map_err(|_| shape_overflow(operation, "bilinear source x"))?,
                weight: y_weight * x_weight,
                derivative_y: y_sign * x_weight,
                derivative_x: x_sign * y_weight,
            })
        })
        .collect()
}

fn border_bilinear_weights(
    height: u64,
    width: u64,
    y: f32,
    x: f32,
    operation: &'static str,
) -> Result<Vec<NativeBilinearWeight>, ExternalTensorKernelPartOneError> {
    let y_weights = checked_linear_weights(
        height,
        y,
        NativeLinearBoundary::Border,
        operation,
    )?;
    let x_weights = checked_linear_weights(
        width,
        x,
        NativeLinearBoundary::Border,
        operation,
    )?;
    let mut weights = Vec::with_capacity(y_weights.len().saturating_mul(x_weights.len()));
    for y_weight in y_weights {
        for x_weight in &x_weights {
            weights.push(NativeBilinearWeight {
                source_y: y_weight.source,
                source_x: x_weight.source,
                weight: y_weight.weight * x_weight.weight,
                derivative_y: y_weight.derivative * x_weight.weight,
                derivative_x: x_weight.derivative * y_weight.weight,
            });
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
fn bilinear_weight_quad(
    y_low: u64,
    y_high: u64,
    x_low: u64,
    x_high: u64,
    y_fraction: f32,
    x_fraction: f32,
    y_derivative: f32,
    x_derivative: f32,
) -> Vec<NativeBilinearWeight> {
    let low_y = 1.0 - y_fraction;
    let low_x = 1.0 - x_fraction;
    vec![
        NativeBilinearWeight {
            source_y: y_low,
            source_x: x_low,
            weight: low_y * low_x,
            derivative_y: -y_derivative * low_x,
            derivative_x: -x_derivative * low_y,
        },
        NativeBilinearWeight {
            source_y: y_low,
            source_x: x_high,
            weight: low_y * x_fraction,
            derivative_y: -y_derivative * x_fraction,
            derivative_x: x_derivative * low_y,
        },
        NativeBilinearWeight {
            source_y: y_high,
            source_x: x_low,
            weight: y_fraction * low_x,
            derivative_y: y_derivative * low_x,
            derivative_x: -x_derivative * y_fraction,
        },
        NativeBilinearWeight {
            source_y: y_high,
            source_x: x_high,
            weight: y_fraction * x_fraction,
            derivative_y: y_derivative * x_fraction,
            derivative_x: x_derivative * y_fraction,
        },
    ]
}

fn checked_floor_u64(
    value: f32,
    operation: &'static str,
) -> Result<u64, ExternalTensorKernelPartOneError> {
    let floor = value.floor();
    if floor < 0.0 || floor > u64::MAX as f32 {
        return Err(shape_overflow(operation, "bilinear coordinate"));
    }
    Ok(floor as u64)
}

fn checked_floor_i64(
    value: f32,
    operation: &'static str,
) -> Result<i64, ExternalTensorKernelPartOneError> {
    let floor = value.floor();
    if floor < i64::MIN as f32 || floor > i64::MAX as f32 {
        return Err(shape_overflow(operation, "bilinear coordinate"));
    }
    Ok(floor as i64)
}

fn morphology_primitive(
    input: &[f32],
    shape: [u64; 4],
    kernel_mask: &[bool],
    kernel_shape: (u64, u64),
    anchor: (u64, u64),
    dilate: bool,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    operation: &'static str,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartOneError> {
    let count = element_count(&shape, operation, "morphology elements")?;
    let mut output = workspace_filled(
        backend,
        context,
        count,
        0.0_f32,
    )?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape, operation)?;
        let y = i64::try_from(indices[2]).map_err(|_| shape_overflow(operation, "morphology y"))?;
        let x = i64::try_from(indices[3]).map_err(|_| shape_overflow(operation, "morphology x"))?;
        let mut selected: Option<f32> = None;
        for kernel_y in 0..kernel_shape.0 {
            for kernel_x in 0..kernel_shape.1 {
                let source_y = y + i64::try_from(kernel_y)
                    .map_err(|_| shape_overflow(operation, "kernel y"))?
                    - i64::try_from(anchor.0)
                        .map_err(|_| shape_overflow(operation, "kernel anchor y"))?;
                let source_x = x + i64::try_from(kernel_x)
                    .map_err(|_| shape_overflow(operation, "kernel x"))?
                    - i64::try_from(anchor.1)
                        .map_err(|_| shape_overflow(operation, "kernel anchor x"))?;
                let source_value = if source_y < 0
                    || source_x < 0
                    || source_y >= shape[2] as i64
                    || source_x >= shape[3] as i64
                {
                    if dilate { -10_000.0 } else { 10_000.0 }
                } else {
                    let source = ravel_index(
                        &[indices[0], indices[1], source_y as u64, source_x as u64],
                        &shape,
                        operation,
                    )?;
                    input[source]
                };
                let mask_y = if dilate {
                    kernel_shape.0 - kernel_y - 1
                } else {
                    kernel_y
                };
                let mask_x = if dilate {
                    kernel_shape.1 - kernel_x - 1
                } else {
                    kernel_x
                };
                let mask_index = usize::try_from(mask_y)
                    .ok()
                    .and_then(|mask_y| {
                        usize::try_from(kernel_shape.1)
                            .ok()
                            .and_then(|width| mask_y.checked_mul(width))
                    })
                    .and_then(|index| {
                        usize::try_from(mask_x)
                            .ok()
                            .and_then(|mask_x| index.checked_add(mask_x))
                    })
                    .ok_or_else(|| shape_overflow(operation, "morphology kernel index"))?;
                let active = *kernel_mask
                    .get(mask_index)
                    .ok_or_else(|| invalid(operation, "morphology kernel mask is incomplete"))?;
                let neighborhood = if active { 0.0 } else { -10_000.0 };
                let value = if dilate {
                    source_value + neighborhood
                } else {
                    source_value - neighborhood
                };
                selected = Some(match selected {
                    Some(current) if current.is_nan() || value.is_nan() => f32::NAN,
                    Some(current) if dilate => current.max(value),
                    Some(current) => current.min(value),
                    None => value,
                });
            }
        }
        output[linear] =
            selected.ok_or_else(|| invalid(operation, "morphology kernel is empty"))?;
    }
    Ok(output)
}

fn morphology_operation_id(operation: NativeMorphologyOperation) -> &'static str {
    match operation {
        NativeMorphologyOperation::Dilation => MORPHOLOGY_DILATION_OPERATION_ID,
        NativeMorphologyOperation::Erosion => MORPHOLOGY_EROSION_OPERATION_ID,
        NativeMorphologyOperation::Opening => MORPHOLOGY_OPENING_OPERATION_ID,
        NativeMorphologyOperation::Closing => MORPHOLOGY_CLOSING_OPERATION_ID,
        NativeMorphologyOperation::Gradient => MORPHOLOGY_GRADIENT_OPERATION_ID,
        NativeMorphologyOperation::TopHat => MORPHOLOGY_TOP_HAT_OPERATION_ID,
        NativeMorphologyOperation::BottomHat => MORPHOLOGY_BOTTOM_HAT_OPERATION_ID,
    }
}

fn subtract_vectors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    left: &[f32],
    right: &[f32],
    operation: &'static str,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartOneError> {
    if left.len() != right.len() {
        return Err(invalid(
            operation,
            "morphology operands have different sizes",
        ));
    }
    let mut output = temporary_vec(backend, context, left.len())?;
    for (left, right) in left.iter().zip(right) {
        output.try_push(left - right)?;
    }
    Ok(output)
}

fn resampled_length(
    input_length: usize,
    configuration: NativeResampleConfiguration,
) -> Result<usize, ExternalTensorKernelPartOneError> {
    let numerator = u128::try_from(input_length)
        .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "input length"))?
        .checked_mul(u128::from(configuration.new_frequency))
        .ok_or_else(|| shape_overflow(RESAMPLE_OPERATION_ID, "resampled length"))?;
    let denominator = u128::from(configuration.original_frequency);
    usize::try_from(numerator.div_ceil(denominator))
        .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "resampled length"))
}

fn resample_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    input_length: usize,
    output_index: usize,
    configuration: NativeResampleConfiguration,
) -> Result<CpuWorkspaceVec<(usize, f64)>, ExternalTensorKernelPartOneError> {
    let divisor = greatest_common_divisor(
        configuration.original_frequency,
        configuration.new_frequency,
    );
    let original = configuration.original_frequency / divisor;
    let new = configuration.new_frequency / divisor;
    let base_frequency = f64::from(original.min(new)) * configuration.rolloff;
    let width = (f64::from(configuration.lowpass_filter_width) * f64::from(original)
        / base_frequency)
        .ceil() as i64;
    let phase = u32::try_from(
        output_index
            % usize::try_from(new)
                .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "resample phase"))?,
    )
    .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "resample phase"))?;
    let step = output_index
        / usize::try_from(new)
            .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "resample step"))?;
    let kernel_length = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(i64::from(original)))
        .ok_or_else(|| shape_overflow(RESAMPLE_OPERATION_ID, "resample kernel length"))?;
    let kernel_capacity = usize::try_from(kernel_length)
        .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "resample kernel length"))?;
    let mut weights = temporary_vec(
        backend,
        context,
        kernel_capacity,
    )?;
    let input_length_i64 = i64::try_from(input_length)
        .map_err(|_| shape_overflow(RESAMPLE_OPERATION_ID, "input length"))?;
    for kernel_index in 0..kernel_length {
        let input_index = i64::try_from(step)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(original)))
            .and_then(|value| value.checked_add(kernel_index))
            .and_then(|value| value.checked_sub(width))
            .ok_or_else(|| shape_overflow(RESAMPLE_OPERATION_ID, "resample source index"))?;
        if input_index < 0 || input_index >= input_length_i64 {
            continue;
        }
        let index = (kernel_index - width) as f64 / f64::from(original);
        let mut t = (-f64::from(phase) / f64::from(new) + index) * base_frequency;
        t = t.clamp(
            -f64::from(configuration.lowpass_filter_width),
            f64::from(configuration.lowpass_filter_width),
        );
        let window =
            (t * std::f64::consts::PI / f64::from(configuration.lowpass_filter_width) / 2.0)
                .cos()
                .powi(2);
        let argument = t * std::f64::consts::PI;
        let sinc = if argument == 0.0 {
            1.0
        } else {
            argument.sin() / argument
        };
        let weight = sinc * window * base_frequency / f64::from(original);
        weights.try_push((input_index as usize, weight))?;
    }
    Ok(weights)
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn validate_mel_configuration(
    configuration: NativeMelSpectrogramConfiguration,
) -> Result<(), ExternalTensorKernelPartOneError> {
    let maximum = configuration
        .f_max
        .unwrap_or(f64::from(configuration.sample_rate) / 2.0);
    if configuration.sample_rate == 0
        || configuration.n_fft == 0
        || configuration.n_mels == 0
        || configuration.power <= 0.0
        || !configuration.power.is_finite()
        || !configuration.f_min.is_finite()
        || !maximum.is_finite()
        || configuration.f_min < 0.0
        || maximum <= configuration.f_min
        || maximum > f64::from(configuration.sample_rate) / 2.0
    {
        return Err(invalid(
            MEL_SPECTROGRAM_OPERATION_ID,
            "invalid mel spectrogram configuration",
        ));
    }
    Ok(())
}

fn validate_mel_scale_configuration(
    configuration: NativeMelScaleConfiguration,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartOneError> {
    let maximum = configuration
        .f_max
        .unwrap_or(f64::from(configuration.sample_rate) / 2.0);
    if configuration.sample_rate == 0
        || configuration.n_mels == 0
        || configuration.n_stft < 2
        || !configuration.f_min.is_finite()
        || !maximum.is_finite()
        || configuration.f_min < 0.0
        || maximum <= configuration.f_min
        || maximum > f64::from(configuration.sample_rate) / 2.0
    {
        return Err(invalid(operation, "invalid mel-scale configuration"));
    }
    Ok(())
}

fn mel_filter_bank(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    frequency_count: usize,
    configuration: NativeMelScaleConfiguration,
    operation: &'static str,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartOneError> {
    validate_mel_scale_configuration(configuration, operation)?;
    if frequency_count != configuration.n_stft {
        return Err(invalid(
            operation,
            "mel filter-bank frequency count does not match n_stft",
        ));
    }
    let maximum = configuration
        .f_max
        .unwrap_or(f64::from(configuration.sample_rate) / 2.0);
    let minimum_mel = frequency_to_mel(configuration.f_min, configuration.mel_scale);
    let maximum_mel = frequency_to_mel(maximum, configuration.mel_scale);
    let point_count = configuration
        .n_mels
        .checked_add(2)
        .ok_or_else(|| shape_overflow(operation, "mel points"))?;
    let mut frequencies = temporary_vec(
        backend,
        context,
        point_count,
    )?;
    for index in 0..point_count {
        let mel =
            minimum_mel + (maximum_mel - minimum_mel) * index as f64 / (point_count - 1) as f64;
        frequencies.try_push(mel_to_frequency(mel, configuration.mel_scale))?;
    }
    let filter_count = configuration
        .n_mels
        .checked_mul(frequency_count)
        .ok_or_else(|| shape_overflow(operation, "mel filter bank"))?;
    let mut filters = workspace_filled(
        backend,
        context,
        filter_count,
        0.0_f32,
    )?;
    for mel in 0..configuration.n_mels {
        let left = frequencies[mel];
        let center = frequencies[mel + 1];
        let right = frequencies[mel + 2];
        let normalization = if configuration.mel_normalization == NativeMelNormalization::Slaney {
            2.0 / (right - left)
        } else {
            1.0
        };
        for frequency in 0..frequency_count {
            let hertz = frequency as f64 * (f64::from(configuration.sample_rate) / 2.0)
                / (frequency_count - 1) as f64;
            let weight = ((hertz - left) / (center - left))
                .min((right - hertz) / (right - center))
                .max(0.0);
            filters[mel * frequency_count + frequency] = (weight * normalization) as f32;
        }
    }
    Ok(filters)
}

fn frequency_to_mel(frequency: f64, scale: NativeMelScale) -> f64 {
    match scale {
        NativeMelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        NativeMelScale::Slaney if frequency < 1000.0 => frequency / (200.0 / 3.0),
        NativeMelScale::Slaney => 15.0 + (frequency / 1000.0).ln() / (6.4_f64.ln() / 27.0),
    }
}

fn mel_to_frequency(mel: f64, scale: NativeMelScale) -> f64 {
    match scale {
        NativeMelScale::Htk => 700.0 * (10_f64.powf(mel / 2595.0) - 1.0),
        NativeMelScale::Slaney if mel < 15.0 => mel * (200.0 / 3.0),
        NativeMelScale::Slaney => 1000.0 * (mel - 15.0).mul_add(6.4_f64.ln() / 27.0, 0.0).exp(),
    }
}

fn cast_image_input_to_f32(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    if input.descriptor().dtype() == DType::F32 {
        return copy_tensor_with_context(backend, input, context, RESIZE_OPERATION_ID);
    }
    let count = element_count(
        input.descriptor().shape(),
        RESIZE_OPERATION_ID,
        "resize input",
    )?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape(), RESIZE_OPERATION_ID)?;
        values.try_push(read_real(input, &indices)? as f32)?;
    }
    upload_f32_with_context(backend, input.descriptor().shape(), &values, context)
}

fn rearrange_fresh_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    plan: &NativeRearrangePlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| shape_overflow(operation, "dtype width"))?;
    let output_count = element_count(&plan.output_shape, operation, "output elements")?;
    let byte_count = output_count
        .checked_mul(width)
        .ok_or_else(|| shape_overflow(operation, "output bytes"))?;
    let mut bytes = temporary_vec(backend, context, byte_count)?;
    for output_linear in 0..output_count {
        check_periodically(output_linear, context.cancellation)?;
        let source_linear = plan.source_linear_index(output_linear, operation)?;
        let indices = unravel_index(source_linear, input.descriptor().shape(), operation)?;
        for byte in input.element_bytes(&indices)? {
            bytes.try_push(*byte)?;
        }
    }
    upload_bytes_with_context(
        backend,
        &plan.output_shape,
        input.descriptor().dtype(),
        &bytes,
        context,
    )
}

fn validate_atomic_axis_partition(
    operation: &'static str,
    lengths: &[u64],
    groups: &[Vec<usize>],
    side: &'static str,
) -> Result<(), ExternalTensorKernelPartOneError> {
    let mut seen = vec![false; lengths.len()];
    for &axis in groups.iter().flatten() {
        let slot = seen
            .get_mut(axis)
            .ok_or_else(|| invalid(operation, format!("atomic {side} axis is outside plan")))?;
        if *slot {
            return Err(invalid(
                operation,
                format!("atomic {side} axis is repeated"),
            ));
        }
        *slot = true;
    }
    if seen.contains(&false) {
        return Err(invalid(
            operation,
            format!("atomic {side} composition omits an axis"),
        ));
    }
    Ok(())
}

fn atomic_group_length(
    operation: &'static str,
    lengths: &[u64],
    group: &[usize],
) -> Result<u64, ExternalTensorKernelPartOneError> {
    group.iter().try_fold(1_u64, |product, &axis| {
        product
            .checked_mul(
                *lengths
                    .get(axis)
                    .ok_or_else(|| invalid(operation, "atomic axis is outside plan"))?,
            )
            .ok_or_else(|| shape_overflow(operation, "atomic axis composition"))
    })
}

fn decode_atomic_group(
    operation: &'static str,
    mut index: u64,
    lengths: &[u64],
    group: &[usize],
    atomic_indices: &mut [u64],
) -> Result<(), ExternalTensorKernelPartOneError> {
    for &axis in group.iter().rev() {
        let length = *lengths
            .get(axis)
            .ok_or_else(|| invalid(operation, "atomic output axis is outside plan"))?;
        let slot = atomic_indices
            .get_mut(axis)
            .ok_or_else(|| invalid(operation, "atomic output index is outside plan"))?;
        *slot = index % length;
        index /= length;
    }
    if index != 0 {
        return Err(invalid(
            operation,
            "output coordinate exceeds atomic composition",
        ));
    }
    Ok(())
}

fn compose_atomic_group(
    operation: &'static str,
    lengths: &[u64],
    group: &[usize],
    atomic_indices: &[u64],
) -> Result<u64, ExternalTensorKernelPartOneError> {
    group.iter().try_fold(0_u64, |index, &axis| {
        let length = *lengths
            .get(axis)
            .ok_or_else(|| invalid(operation, "atomic input axis is outside plan"))?;
        let coordinate = *atomic_indices
            .get(axis)
            .ok_or_else(|| invalid(operation, "atomic input index is outside plan"))?;
        index
            .checked_mul(length)
            .and_then(|index| index.checked_add(coordinate))
            .ok_or_else(|| shape_overflow(operation, "atomic input coordinate"))
    })
}

fn coefficients_are_valid(coefficients: NativeBiquadCoefficients) -> bool {
    [
        coefficients.b0,
        coefficients.b1,
        coefficients.b2,
        coefficients.a0,
        coefficients.a1,
        coefficients.a2,
    ]
    .iter()
    .all(|value| value.is_finite())
        && coefficients.a0 != 0.0
}

pub(crate) fn validate_audio_parameters(
    sample_rate: u32,
    frequency: f64,
    quality: f64,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartOneError> {
    if sample_rate == 0
        || !frequency.is_finite()
        || frequency <= 0.0
        || frequency > f64::from(sample_rate) / 2.0
        || !quality.is_finite()
        || quality <= 0.0
    {
        return Err(invalid(
            operation,
            "invalid sample rate, frequency, or quality",
        ));
    }
    Ok(())
}

fn copy_tensor_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    operation: &'static str,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    let (output, _) = backend.copy(
        input,
        TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            input.descriptor().dtype(),
            DeviceId::CPU,
            input.descriptor().stream(),
        )?,
        context,
    )?;
    Ok(output)
}

fn tensor_f32_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
    operation: &'static str,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartOneError> {
    require_f32_cpu(tensor, operation)?;
    let count = element_count(tensor.descriptor().shape(), operation, "tensor elements")?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, tensor.descriptor().shape(), operation)?;
        values.try_push(read_f32(tensor, &indices)?)?;
    }
    Ok(values)
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ExternalTensorKernelPartOneError> {
    let bytes: [u8; 4] = tensor
        .element_bytes(indices)?
        .try_into()
        .map_err(|_| invalid(REARRANGE_OPERATION_ID, "F32 element has invalid byte width"))?;
    Ok(f32::from_ne_bytes(bytes))
}

fn read_complex64(
    tensor: &Tensor,
    indices: &[u64],
) -> Result<(f32, f32), ExternalTensorKernelPartOneError> {
    let bytes: [u8; 8] = tensor.element_bytes(indices)?.try_into().map_err(|_| {
        invalid(
            MEL_SPECTROGRAM_OPERATION_ID,
            "Complex64 element has invalid byte width",
        )
    })?;
    Ok((
        f32::from_ne_bytes(
            bytes[..4]
                .try_into()
                .map_err(|_| invalid(MEL_SPECTROGRAM_OPERATION_ID, "invalid complex real bytes"))?,
        ),
        f32::from_ne_bytes(bytes[4..].try_into().map_err(|_| {
            invalid(
                MEL_SPECTROGRAM_OPERATION_ID,
                "invalid complex imaginary bytes",
            )
        })?),
    ))
}

fn read_real(tensor: &Tensor, indices: &[u64]) -> Result<f64, ExternalTensorKernelPartOneError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value),
        DecodedScalar::Unsigned(value) => Ok(value as f64),
        DecodedScalar::Signed(value) => Ok(value as f64),
        DecodedScalar::Boolean(value) => Ok(f64::from(u8::from(value))),
        DecodedScalar::Complex { .. } => Err(invalid(
            RESIZE_OPERATION_ID,
            "complex image dtype is unsupported",
        )),
    }
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartOneError> {
    context.cancellation.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<CpuWorkspaceVec<T>, ExternalTensorKernelPartOneError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ExternalTensorKernelPartOneError> {
    let mut values = temporary_vec(backend, context, count)?;
    for _ in 0..count {
        values.try_push(value)?;
    }
    Ok(values)
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartOneError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ExternalTensorKernelPartOneError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartOneError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ExternalTensorKernelPartOneError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_same_stream(
    left: &Tensor,
    right: &Tensor,
) -> Result<(), ExternalTensorKernelPartOneError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: left.descriptor().stream(),
            actual: right.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn element_count(
    shape: &[u64],
    operation: &'static str,
    subject: &'static str,
) -> Result<usize, ExternalTensorKernelPartOneError> {
    let count = shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| shape_overflow(operation, subject))?;
    usize::try_from(count).map_err(|_| shape_overflow(operation, subject))
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
    operation: &'static str,
) -> Result<usize, ExternalTensorKernelPartOneError> {
    if indices.len() != shape.len() {
        return Err(invalid(operation, "index rank does not match shape"));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(invalid(operation, "index is outside shape"));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or_else(|| shape_overflow(operation, "linear index"))?;
    }
    usize::try_from(linear).map_err(|_| shape_overflow(operation, "linear index"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
    operation: &'static str,
) -> Result<Vec<u64>, ExternalTensorKernelPartOneError> {
    let count = element_count(shape, operation, "index shape")?;
    if linear >= count {
        return Err(invalid(operation, "linear index is outside shape"));
    }
    let mut indices = vec![0_u64; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let size = usize::try_from(shape[dimension])
            .map_err(|_| shape_overflow(operation, "dimension"))?;
        if size == 0 {
            return Err(invalid(
                operation,
                "cannot unravel an index through an empty dimension",
            ));
        }
        indices[dimension] =
            u64::try_from(linear % size).map_err(|_| shape_overflow(operation, "index"))?;
        linear /= size;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ExternalTensorKernelPartOneError> {
    if index.is_multiple_of(1024) {
        cancellation.check()?;
    }
    Ok(())
}

fn check_periodically_u64(
    index: u64,
    cancellation: &CancellationToken,
) -> Result<(), ExternalTensorKernelPartOneError> {
    if index.is_multiple_of(1024) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> ExternalTensorKernelPartOneError {
    ExternalTensorKernelPartOneError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn shape_overflow(
    operation: &'static str,
    subject: &'static str,
) -> ExternalTensorKernelPartOneError {
    ExternalTensorKernelPartOneError::ShapeOverflow { operation, subject }
}
