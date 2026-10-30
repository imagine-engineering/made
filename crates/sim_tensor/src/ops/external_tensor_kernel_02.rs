use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, Rgb8ImageTensor, StreamId, Tensor, TensorDescriptor,
    TensorError,
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, OperatorIndirectionError,
        convolution_into_with_context_exact_native,
    },
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeBilinearBoundary, NativeMorphologyOperation,
        NativeRearrangePlan, checked_bilinear_weights, native_morphology_with_context_exact,
        rearrange_jvp_with_context_exact_native_for_operation,
        rearrange_vjp_with_context_exact_native_for_operation,
        rearrange_with_context_exact_native_for_operation,
    },
};
use thiserror::Error;

pub const EINOPS_REARRANGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-A56F89536902";
pub const RGB_TO_LAB_OPERATION_ID: &str = "COMFY-TENSOR-OP-4F9C05E204D4";
pub const RGB_TO_YCBCR_OPERATION_ID: &str = "COMFY-TENSOR-OP-A555F803F554";
pub const YCBCR_TO_RGB_OPERATION_ID: &str = "COMFY-TENSOR-OP-9EF1D9EB674A";
pub const CANNY_OPERATION_ID: &str = "COMFY-TENSOR-OP-A551C36699B7";
pub const DILATION_OPERATION_ID: &str = "COMFY-TENSOR-OP-AF5C2820E4C3";
pub const EROSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-9236C1C08976";
pub const TOP_HAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-AC69F309A190";
pub const EFFICIENTNET_V2_S_OPERATION_ID: &str = "COMFY-TENSOR-OP-638DE6179D46";
pub const RAFT_LARGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-852D8E9DBC9C";
pub const DEFORM_CONV2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-9E730487CA71";
pub const TO_PIL_IMAGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B7926028DA57";

#[derive(Debug, Error)]
pub enum ExternalTensorKernelPartTwoError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartOne(#[from] ExternalTensorKernelPartOneError),
    #[error(transparent)]
    Convolution(#[from] OperatorIndirectionError),
    #[error("external tensor-kernel part-two execution was cancelled")]
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

impl From<sim_types::CancellationError> for ExternalTensorKernelPartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> ExternalTensorKernelPartTwoError {
    ExternalTensorKernelPartTwoError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn overflow(operation: &'static str, subject: &'static str) -> ExternalTensorKernelPartTwoError {
    ExternalTensorKernelPartTwoError::ShapeOverflow { operation, subject }
}

fn remap_operation(
    error: ExternalTensorKernelPartTwoError,
    operation: &'static str,
) -> ExternalTensorKernelPartTwoError {
    match error {
        ExternalTensorKernelPartTwoError::UnsupportedDevice { device, .. } => {
            ExternalTensorKernelPartTwoError::UnsupportedDevice { operation, device }
        }
        ExternalTensorKernelPartTwoError::UnsupportedDType { dtype, .. } => {
            ExternalTensorKernelPartTwoError::UnsupportedDType { operation, dtype }
        }
        ExternalTensorKernelPartTwoError::Invalid { reason, .. } => {
            ExternalTensorKernelPartTwoError::Invalid { operation, reason }
        }
        ExternalTensorKernelPartTwoError::ShapeOverflow { subject, .. } => {
            ExternalTensorKernelPartTwoError::ShapeOverflow { operation, subject }
        }
        error => error,
    }
}


fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartTwoError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ExternalTensorKernelPartTwoError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ExternalTensorKernelPartTwoError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_same_shape_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartTwoError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(invalid(operation, "tensor shapes do not match"));
    }
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(invalid(operation, "tensor streams do not match"));
    }
    Ok(())
}

fn element_count(
    shape: &[u64],
    operation: &'static str,
    subject: &'static str,
) -> Result<usize, ExternalTensorKernelPartTwoError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| overflow(operation, subject))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
    operation: &'static str,
) -> Result<Vec<u64>, ExternalTensorKernelPartTwoError> {
    let mut indices = vec![0_u64; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[axis]).map_err(|_| overflow(operation, "axis"))?;
        if dimension == 0 {
            return Err(invalid(operation, "cannot unravel an empty tensor"));
        }
        indices[axis] =
            u64::try_from(linear % dimension).map_err(|_| overflow(operation, "coordinate"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
    operation: &'static str,
) -> Result<usize, ExternalTensorKernelPartTwoError> {
    if indices.len() != shape.len() {
        return Err(invalid(operation, "index rank does not match tensor rank"));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(invalid(operation, "index is outside tensor shape"));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or_else(|| overflow(operation, "linear index"))?;
    }
    usize::try_from(linear).map_err(|_| overflow(operation, "linear index"))
}

fn read_f32(
    tensor: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ExternalTensorKernelPartTwoError> {
    match DType::F32.decode_scalar(tensor.element_bytes(indices)?)? {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(invalid(
            operation,
            "canonical F32 decoder returned a non-real scalar",
        )),
    }
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn tensor_f32_values(
    backend: &CpuBackend,
    tensor: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartTwoError> {
    require_f32_cpu(tensor, operation)?;
    let count = element_count(tensor.descriptor().shape(), operation, "tensor elements")?;
    let mut values = temporary_vec(backend, context, count, operation)?;
    for linear in 0..count {
        if linear & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        values.try_push(read_f32(
            tensor,
            &unravel_index(linear, tensor.descriptor().shape(), operation)?,
            operation,
        )?)?;
    }
    context.cancellation.check()?;
    Ok(values)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _operation: &'static str,
) -> Result<CpuWorkspaceVec<T>, ExternalTensorKernelPartTwoError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
    operation: &'static str,
) -> Result<CpuWorkspaceVec<T>, ExternalTensorKernelPartTwoError> {
    let mut values = temporary_vec(backend, context, count, operation)?;
    for _ in 0..count {
        values.try_push(value)?;
    }
    Ok(values)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RearrangeAxis {
    Named(String),
    Ellipsis(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParsedRearrangeAxis {
    Named(String),
    Unit,
    Ellipsis,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParsedRearrangeTerm {
    Axis(ParsedRearrangeAxis),
    Group(Vec<ParsedRearrangeAxis>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeEinopsRecipe {
    input: Vec<ParsedRearrangeTerm>,
    output: Vec<ParsedRearrangeTerm>,
}

impl NativeEinopsRecipe {
    pub fn parse(pattern: &str) -> Result<Self, ExternalTensorKernelPartTwoError> {
        let mut parts = pattern.split("->");
        let input = parts.next().ok_or_else(|| {
            invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "rearrange pattern has no input side",
            )
        })?;
        let output = parts.next().ok_or_else(|| {
            invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "rearrange pattern requires one arrow",
            )
        })?;
        if parts.next().is_some() {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "rearrange pattern contains more than one arrow",
            ));
        }
        let input = parse_rearrange_side(input)?;
        let output = parse_rearrange_side(output)?;
        validate_rearrange_axis_sets(&input, &output)?;
        Ok(Self { input, output })
    }

    pub fn parse_repeat(pattern: &str) -> Result<Self, ExternalTensorKernelPartTwoError> {
        let mut parts = pattern.split("->");
        let input = parse_rearrange_side(parts.next().ok_or_else(|| {
            invalid(EINOPS_REARRANGE_OPERATION_ID, "repeat pattern has no input side")
        })?)?;
        let output = parse_rearrange_side(parts.next().ok_or_else(|| {
            invalid(EINOPS_REARRANGE_OPERATION_ID, "repeat pattern requires one arrow")
        })?)?;
        if parts.next().is_some() {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "repeat pattern contains more than one arrow",
            ));
        }
        validate_repeat_axis_sets(&input, &output)?;
        Ok(Self { input, output })
    }

    pub fn compile(
        &self,
        input_shape: &[u64],
        axis_lengths: &BTreeMap<String, u64>,
        cancellation: &CancellationToken,
    ) -> Result<NativeRearrangePlan, ExternalTensorKernelPartTwoError> {
        cancellation.check()?;
        let named_axes = named_rearrange_axes(&self.input)?;
        for axis in axis_lengths.keys() {
            if !named_axes.contains(axis) {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    format!("axis length was supplied for unknown axis {axis}"),
                ));
            }
        }

        let fixed_input_terms = self
            .input
            .iter()
            .filter(|term| {
                !matches!(
                    term,
                    ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis)
                )
            })
            .count();
        let has_input_ellipsis = self.input.iter().any(|term| {
            matches!(
                term,
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis)
            )
        });
        let ellipsis_dimensions = if has_input_ellipsis {
            input_shape
                .len()
                .checked_sub(fixed_input_terms)
                .ok_or_else(|| {
                    invalid(
                        EINOPS_REARRANGE_OPERATION_ID,
                        "input rank is smaller than the fixed pattern rank",
                    )
                })?
        } else {
            if input_shape.len() != fixed_input_terms {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "input rank does not match the rearrange pattern",
                ));
            }
            0
        };

        let mut lengths = BTreeMap::new();
        for (axis, length) in axis_lengths {
            lengths.insert(RearrangeAxis::Named(axis.clone()), *length);
        }
        let mut input_groups = Vec::new();
        let mut input_dimension = 0_usize;
        for term in &self.input {
            match term {
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis) => {
                    for ellipsis in 0..ellipsis_dimensions {
                        let length = input_shape[input_dimension];
                        let axis = RearrangeAxis::Ellipsis(ellipsis);
                        lengths.insert(axis.clone(), length);
                        input_groups.push(vec![axis]);
                        input_dimension += 1;
                    }
                }
                ParsedRearrangeTerm::Axis(axis) => {
                    let dimension = input_shape[input_dimension];
                    let factors = bind_input_rearrange_group(
                        std::slice::from_ref(axis),
                        dimension,
                        &mut lengths,
                    )?;
                    input_groups.push(factors);
                    input_dimension += 1;
                }
                ParsedRearrangeTerm::Group(axes) => {
                    if axes
                        .iter()
                        .any(|axis| matches!(axis, ParsedRearrangeAxis::Ellipsis))
                    {
                        return Err(invalid(
                            EINOPS_REARRANGE_OPERATION_ID,
                            "input ellipsis cannot be parenthesized",
                        ));
                    }
                    let dimension = input_shape[input_dimension];
                    input_groups.push(bind_input_rearrange_group(axes, dimension, &mut lengths)?);
                    input_dimension += 1;
                }
            }
        }

        let ellipsis_axes = (0..ellipsis_dimensions)
            .map(RearrangeAxis::Ellipsis)
            .collect::<Vec<_>>();
        let mut output_groups = Vec::new();
        for term in &self.output {
            match term {
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis) => {
                    output_groups.extend(ellipsis_axes.iter().cloned().map(|axis| vec![axis]));
                }
                ParsedRearrangeTerm::Axis(axis) => {
                    output_groups.push(output_rearrange_group(
                        std::slice::from_ref(axis),
                        &ellipsis_axes,
                        &lengths,
                    )?);
                }
                ParsedRearrangeTerm::Group(axes) => {
                    output_groups.push(output_rearrange_group(axes, &ellipsis_axes, &lengths)?);
                }
            }
        }
        cancellation.check()?;
        let ordered_axes = lengths.keys().cloned().collect::<Vec<_>>();
        let axis_indices = ordered_axes
            .iter()
            .enumerate()
            .map(|(index, axis)| (axis.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let atomic_lengths = ordered_axes
            .iter()
            .map(|axis| lengths[axis])
            .collect::<Vec<_>>();
        let input_groups = rearrange_group_indices(&input_groups, &axis_indices)?;
        let output_groups = rearrange_group_indices(&output_groups, &axis_indices)?;
        Ok(NativeRearrangePlan::from_atomic_axes(
            EINOPS_REARRANGE_OPERATION_ID,
            input_shape.to_vec(),
            atomic_lengths,
            input_groups,
            output_groups,
        )?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeEinopsRepeatPlan {
    input_shape: Vec<u64>,
    output_shape: Vec<u64>,
    source_linear_indices: Vec<u64>,
}

impl NativeEinopsRepeatPlan {
    pub fn compile(
        recipe: &NativeEinopsRecipe,
        input_shape: &[u64],
        axis_lengths: &BTreeMap<String, u64>,
        cancellation: &CancellationToken,
    ) -> Result<Self, ExternalTensorKernelPartTwoError> {
        cancellation.check()?;
        let input_named = named_rearrange_axes(&recipe.input)?;
        let output_named = named_rearrange_axes(&recipe.output)?;
        for axis in axis_lengths.keys() {
            if !output_named.contains(axis) {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    format!("axis length was supplied for unknown repeat axis {axis}"),
                ));
            }
        }
        let fixed_input_terms = recipe
            .input
            .iter()
            .filter(|term| {
                !matches!(
                    term,
                    ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis)
                )
            })
            .count();
        let has_ellipsis = recipe.input.iter().any(|term| {
            matches!(
                term,
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis)
            )
        });
        let ellipsis_dimensions = if has_ellipsis {
            input_shape.len().checked_sub(fixed_input_terms).ok_or_else(|| {
                invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "input rank is smaller than the fixed repeat pattern rank",
                )
            })?
        } else {
            if input_shape.len() != fixed_input_terms {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "input rank does not match the repeat pattern",
                ));
            }
            0
        };

        let mut lengths = BTreeMap::new();
        for (axis, length) in axis_lengths {
            lengths.insert(RearrangeAxis::Named(axis.clone()), *length);
        }
        let mut input_groups = Vec::new();
        let mut input_dimension = 0_usize;
        for term in &recipe.input {
            match term {
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis) => {
                    for ellipsis in 0..ellipsis_dimensions {
                        let axis = RearrangeAxis::Ellipsis(ellipsis);
                        lengths.insert(axis.clone(), input_shape[input_dimension]);
                        input_groups.push(vec![axis]);
                        input_dimension += 1;
                    }
                }
                ParsedRearrangeTerm::Axis(axis) => {
                    input_groups.push(bind_input_rearrange_group(
                        std::slice::from_ref(axis),
                        input_shape[input_dimension],
                        &mut lengths,
                    )?);
                    input_dimension += 1;
                }
                ParsedRearrangeTerm::Group(axes) => {
                    if axes
                        .iter()
                        .any(|axis| matches!(axis, ParsedRearrangeAxis::Ellipsis))
                    {
                        return Err(invalid(
                            EINOPS_REARRANGE_OPERATION_ID,
                            "input ellipsis cannot be parenthesized",
                        ));
                    }
                    input_groups.push(bind_input_rearrange_group(
                        axes,
                        input_shape[input_dimension],
                        &mut lengths,
                    )?);
                    input_dimension += 1;
                }
            }
        }
        for axis in output_named.difference(&input_named) {
            let key = RearrangeAxis::Named(axis.clone());
            if !lengths.contains_key(&key) {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    format!("new repeat axis {axis} requires an explicit length"),
                ));
            }
        }

        let ellipsis_axes = (0..ellipsis_dimensions)
            .map(RearrangeAxis::Ellipsis)
            .collect::<Vec<_>>();
        let mut output_groups = Vec::new();
        for term in &recipe.output {
            match term {
                ParsedRearrangeTerm::Axis(ParsedRearrangeAxis::Ellipsis) => {
                    output_groups.extend(ellipsis_axes.iter().cloned().map(|axis| vec![axis]));
                }
                ParsedRearrangeTerm::Axis(axis) => output_groups.push(output_rearrange_group(
                    std::slice::from_ref(axis),
                    &ellipsis_axes,
                    &lengths,
                )?),
                ParsedRearrangeTerm::Group(axes) => output_groups.push(output_rearrange_group(
                    axes,
                    &ellipsis_axes,
                    &lengths,
                )?),
            }
        }
        let output_shape = output_groups
            .iter()
            .map(|group| repeat_group_length(group, &lengths))
            .collect::<Result<Vec<_>, _>>()?;
        let output_count = element_count(
            &output_shape,
            EINOPS_REARRANGE_OPERATION_ID,
            "repeat output elements",
        )?;
        let mut source_linear_indices = Vec::new();
        source_linear_indices.try_reserve_exact(output_count).map_err(|_| {
            overflow(
                EINOPS_REARRANGE_OPERATION_ID,
                "repeat source-index allocation",
            )
        })?;
        for linear in 0..output_count {
            if linear & 0x3ff == 0 {
                cancellation.check()?;
            }
            let output_indices = unravel_index(
                linear,
                &output_shape,
                EINOPS_REARRANGE_OPERATION_ID,
            )?;
            let mut coordinates = BTreeMap::new();
            for (&index, group) in output_indices.iter().zip(&output_groups) {
                decode_repeat_group(index, group, &lengths, &mut coordinates)?;
            }
            let input_indices = input_groups
                .iter()
                .map(|group| encode_repeat_group(group, &lengths, &coordinates))
                .collect::<Result<Vec<_>, _>>()?;
            source_linear_indices.push(u64::try_from(ravel_index(
                &input_indices,
                input_shape,
                EINOPS_REARRANGE_OPERATION_ID,
            )?)
            .map_err(|_| overflow(EINOPS_REARRANGE_OPERATION_ID, "repeat source index"))?);
        }
        cancellation.check()?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_shape,
            source_linear_indices,
        })
    }
}

pub fn einops_repeat_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ExternalTensorKernelPartTwoError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    let recipe = NativeEinopsRecipe::parse_repeat(pattern)
        .map_err(|error| remap_operation(error, operation))?;
    let plan = NativeEinopsRepeatPlan::compile(
        &recipe,
        input.descriptor().shape(),
        axis_lengths,
        context.cancellation,
    )
    .map_err(|error| remap_operation(error, operation))?;
    repeat_plan_forward(backend, input, &plan, operation, context)
}

pub fn einops_repeat_jvp_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    einops_repeat_with_context_exact_native_for_operation(
        backend,
        input_tangent,
        pattern,
        axis_lengths,
        operation,
        context,
    )
}

pub fn einops_repeat_vjp_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    require_f32_cpu(output_gradient, operation)?;
    let recipe = NativeEinopsRecipe::parse_repeat(pattern)
        .map_err(|error| remap_operation(error, operation))?;
    let plan = NativeEinopsRepeatPlan::compile(
        &recipe,
        input_shape,
        axis_lengths,
        context.cancellation,
    )
    .map_err(|error| remap_operation(error, operation))?;
    if output_gradient.descriptor().shape() != plan.output_shape {
        return Err(invalid(operation, "repeat gradient shape does not match output"));
    }
    let input_count = element_count(input_shape, operation, "repeat input gradient")?;
    let mut values = temporary_filled(backend, context, input_count, 0.0_f32, operation)?;
    for (output_linear, source_linear) in plan.source_linear_indices.iter().enumerate() {
        if output_linear & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        let source = usize::try_from(*source_linear)
            .map_err(|_| overflow(operation, "repeat gradient source"))?;
        let destination = values
            .get_mut(source)
            .ok_or_else(|| invalid(operation, "repeat gradient source is outside input"))?;
        *destination += read_f32(
            output_gradient,
            &unravel_index(output_linear, &plan.output_shape, operation)?,
            operation,
        )?;
    }
    context.cancellation.check()?;
    upload_f32(
        backend,
        input_shape,
        output_gradient.descriptor().stream(),
        &values,
        context,
    )
}

fn repeat_plan_forward(
    backend: &CpuBackend,
    input: &Tensor,
    plan: &NativeEinopsRepeatPlan,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    if input.descriptor().shape() != plan.input_shape {
        return Err(invalid(operation, "repeat input shape changed after planning"));
    }
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| overflow(operation, "repeat dtype width"))?;
    let capacity = plan
        .source_linear_indices
        .len()
        .checked_mul(width)
        .ok_or_else(|| overflow(operation, "repeat output bytes"))?;
    let mut bytes = temporary_vec(backend, context, capacity, operation)?;
    for (output_linear, source_linear) in plan.source_linear_indices.iter().enumerate() {
        if output_linear & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        for byte in input.linear_element_bytes(*source_linear)?.iter().copied() {
            bytes.try_push(byte)?;
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        plan.output_shape.clone(),
        input.descriptor().dtype(),
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}


pub fn einops_rearrange_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let recipe = NativeEinopsRecipe::parse(pattern)?;
    let plan = recipe.compile(
        input.descriptor().shape(),
        axis_lengths,
        context.cancellation,
    )?;
    Ok(rearrange_with_context_exact_native_for_operation(
        backend,
        input,
        &plan,
        EINOPS_REARRANGE_OPERATION_ID,
        context,
    )?)
}


pub fn einops_rearrange_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let recipe = NativeEinopsRecipe::parse(pattern)?;
    let plan = recipe.compile(
        input_tangent.descriptor().shape(),
        axis_lengths,
        context.cancellation,
    )?;
    Ok(rearrange_jvp_with_context_exact_native_for_operation(
        backend,
        input_tangent,
        &plan,
        EINOPS_REARRANGE_OPERATION_ID,
        context,
    )?)
}


pub fn einops_rearrange_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let recipe = NativeEinopsRecipe::parse(pattern)?;
    let plan = recipe.compile(input_shape, axis_lengths, context.cancellation)?;
    Ok(rearrange_vjp_with_context_exact_native_for_operation(
        backend,
        output_gradient,
        &plan,
        EINOPS_REARRANGE_OPERATION_ID,
        context,
    )?)
}

fn parse_rearrange_side(
    source: &str,
) -> Result<Vec<ParsedRearrangeTerm>, ExternalTensorKernelPartTwoError> {
    let characters = source.as_bytes();
    let mut index = 0_usize;
    let mut terms = Vec::new();
    while index < characters.len() {
        if characters[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if characters[index] == b'(' {
            index += 1;
            let mut axes = Vec::new();
            loop {
                while index < characters.len() && characters[index].is_ascii_whitespace() {
                    index += 1;
                }
                if index >= characters.len() {
                    return Err(invalid(
                        EINOPS_REARRANGE_OPERATION_ID,
                        "rearrange group is not closed",
                    ));
                }
                if characters[index] == b')' {
                    index += 1;
                    break;
                }
                if characters[index] == b'(' {
                    return Err(invalid(
                        EINOPS_REARRANGE_OPERATION_ID,
                        "nested rearrange groups are unsupported",
                    ));
                }
                axes.push(parse_rearrange_axis(characters, &mut index)?);
            }
            terms.push(ParsedRearrangeTerm::Group(axes));
        } else if characters[index] == b')' {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "rearrange pattern contains an unmatched closing parenthesis",
            ));
        } else {
            terms.push(ParsedRearrangeTerm::Axis(parse_rearrange_axis(
                characters, &mut index,
            )?));
        }
    }
    validate_rearrange_side(&terms)?;
    Ok(terms)
}

fn parse_rearrange_axis(
    source: &[u8],
    index: &mut usize,
) -> Result<ParsedRearrangeAxis, ExternalTensorKernelPartTwoError> {
    if source.get(*index..index.saturating_add(3)) == Some(b"...") {
        *index += 3;
        return Ok(ParsedRearrangeAxis::Ellipsis);
    }
    let start = *index;
    let first = *source
        .get(*index)
        .ok_or_else(|| invalid(EINOPS_REARRANGE_OPERATION_ID, "rearrange axis is missing"))?;
    if first == b'1' {
        *index += 1;
        if source
            .get(*index)
            .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
        {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "only the anonymous unit axis literal 1 is supported",
            ));
        }
        return Ok(ParsedRearrangeAxis::Unit);
    }
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "rearrange axes must be identifiers, 1, or ellipsis",
        ));
    }
    *index += 1;
    while source
        .get(*index)
        .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
    {
        *index += 1;
    }
    let name = std::str::from_utf8(&source[start..*index])
        .map_err(|_| invalid(EINOPS_REARRANGE_OPERATION_ID, "axis name is not UTF-8"))?;
    Ok(ParsedRearrangeAxis::Named(name.to_owned()))
}

fn validate_rearrange_side(
    terms: &[ParsedRearrangeTerm],
) -> Result<(), ExternalTensorKernelPartTwoError> {
    let mut named = BTreeSet::new();
    let mut ellipsis = false;
    for axis in terms.iter().flat_map(|term| match term {
        ParsedRearrangeTerm::Axis(axis) => std::slice::from_ref(axis),
        ParsedRearrangeTerm::Group(axes) => axes.as_slice(),
    }) {
        match axis {
            ParsedRearrangeAxis::Named(name) if !named.insert(name.clone()) => {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    format!("rearrange axis {name} appears more than once on one side"),
                ));
            }
            ParsedRearrangeAxis::Ellipsis if ellipsis => {
                return Err(invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "rearrange side contains more than one ellipsis",
                ));
            }
            ParsedRearrangeAxis::Ellipsis => ellipsis = true,
            _ => {}
        }
    }
    Ok(())
}

fn named_rearrange_axes(
    terms: &[ParsedRearrangeTerm],
) -> Result<BTreeSet<String>, ExternalTensorKernelPartTwoError> {
    let mut axes = BTreeSet::new();
    for axis in terms.iter().flat_map(|term| match term {
        ParsedRearrangeTerm::Axis(axis) => std::slice::from_ref(axis),
        ParsedRearrangeTerm::Group(axes) => axes.as_slice(),
    }) {
        if let ParsedRearrangeAxis::Named(name) = axis
            && !axes.insert(name.clone())
        {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                format!("rearrange axis {name} is duplicated"),
            ));
        }
    }
    Ok(axes)
}

fn validate_rearrange_axis_sets(
    input: &[ParsedRearrangeTerm],
    output: &[ParsedRearrangeTerm],
) -> Result<(), ExternalTensorKernelPartTwoError> {
    if named_rearrange_axes(input)? != named_rearrange_axes(output)? {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "input and output rearrange axes do not match",
        ));
    }
    let input_ellipsis = input.iter().any(|term| match term {
        ParsedRearrangeTerm::Axis(axis) => matches!(axis, ParsedRearrangeAxis::Ellipsis),
        ParsedRearrangeTerm::Group(axes) => axes
            .iter()
            .any(|axis| matches!(axis, ParsedRearrangeAxis::Ellipsis)),
    });
    let output_ellipsis = output.iter().any(|term| match term {
        ParsedRearrangeTerm::Axis(axis) => matches!(axis, ParsedRearrangeAxis::Ellipsis),
        ParsedRearrangeTerm::Group(axes) => axes
            .iter()
            .any(|axis| matches!(axis, ParsedRearrangeAxis::Ellipsis)),
    });
    if input_ellipsis != output_ellipsis {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "ellipsis must appear on both sides of a rearrange pattern",
        ));
    }
    Ok(())
}

fn validate_repeat_axis_sets(
    input: &[ParsedRearrangeTerm],
    output: &[ParsedRearrangeTerm],
) -> Result<(), ExternalTensorKernelPartTwoError> {
    let input_axes = named_rearrange_axes(input)?;
    let output_axes = named_rearrange_axes(output)?;
    if !input_axes.is_subset(&output_axes) {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "every input repeat axis must appear on the output side",
        ));
    }
    let input_ellipsis = side_has_ellipsis(input);
    let output_ellipsis = side_has_ellipsis(output);
    if input_ellipsis != output_ellipsis {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "ellipsis must appear on both sides of a repeat pattern",
        ));
    }
    Ok(())
}

fn side_has_ellipsis(terms: &[ParsedRearrangeTerm]) -> bool {
    terms.iter().any(|term| match term {
        ParsedRearrangeTerm::Axis(axis) => matches!(axis, ParsedRearrangeAxis::Ellipsis),
        ParsedRearrangeTerm::Group(axes) => axes
            .iter()
            .any(|axis| matches!(axis, ParsedRearrangeAxis::Ellipsis)),
    })
}

fn repeat_group_length(
    group: &[RearrangeAxis],
    lengths: &BTreeMap<RearrangeAxis, u64>,
) -> Result<u64, ExternalTensorKernelPartTwoError> {
    group.iter().try_fold(1_u64, |product, axis| {
        product
            .checked_mul(*lengths.get(axis).ok_or_else(|| {
                invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "repeat axis has no resolved length",
                )
            })?)
            .ok_or_else(|| overflow(EINOPS_REARRANGE_OPERATION_ID, "repeat axis product"))
    })
}

fn decode_repeat_group(
    mut index: u64,
    group: &[RearrangeAxis],
    lengths: &BTreeMap<RearrangeAxis, u64>,
    coordinates: &mut BTreeMap<RearrangeAxis, u64>,
) -> Result<(), ExternalTensorKernelPartTwoError> {
    for axis in group.iter().rev() {
        let length = *lengths.get(axis).ok_or_else(|| {
            invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "repeat output axis has no resolved length",
            )
        })?;
        if length == 0 {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "empty repeat output has no element coordinates",
            ));
        }
        coordinates.insert(axis.clone(), index % length);
        index /= length;
    }
    if index != 0 {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "repeat output coordinate is outside its group",
        ));
    }
    Ok(())
}

fn encode_repeat_group(
    group: &[RearrangeAxis],
    lengths: &BTreeMap<RearrangeAxis, u64>,
    coordinates: &BTreeMap<RearrangeAxis, u64>,
) -> Result<u64, ExternalTensorKernelPartTwoError> {
    group.iter().try_fold(0_u64, |index, axis| {
        index
            .checked_mul(*lengths.get(axis).ok_or_else(|| {
                invalid(
                    EINOPS_REARRANGE_OPERATION_ID,
                    "repeat input axis has no resolved length",
                )
            })?)
            .and_then(|value| value.checked_add(coordinates.get(axis).copied().unwrap_or(0)))
            .ok_or_else(|| overflow(EINOPS_REARRANGE_OPERATION_ID, "repeat input coordinate"))
    })
}

fn bind_input_rearrange_group(
    axes: &[ParsedRearrangeAxis],
    dimension: u64,
    lengths: &mut BTreeMap<RearrangeAxis, u64>,
) -> Result<Vec<RearrangeAxis>, ExternalTensorKernelPartTwoError> {
    let factors = axes
        .iter()
        .filter_map(|axis| match axis {
            ParsedRearrangeAxis::Named(name) => Some(RearrangeAxis::Named(name.clone())),
            ParsedRearrangeAxis::Unit => None,
            ParsedRearrangeAxis::Ellipsis => None,
        })
        .collect::<Vec<_>>();
    if factors.is_empty() {
        if dimension != 1 {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "unit-only rearrange group requires an input dimension of one",
            ));
        }
        return Ok(factors);
    }
    let unknown = factors
        .iter()
        .filter(|axis| !lengths.contains_key(*axis))
        .cloned()
        .collect::<Vec<_>>();
    if unknown.len() > 1 {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "at most one axis length may be inferred per input dimension",
        ));
    }
    let known_product = factors
        .iter()
        .filter_map(|axis| lengths.get(axis).copied())
        .try_fold(1_u64, |product, length| product.checked_mul(length))
        .ok_or_else(|| overflow(EINOPS_REARRANGE_OPERATION_ID, "axis product"))?;
    if let Some(axis) = unknown.first() {
        if known_product == 0 || !dimension.is_multiple_of(known_product) {
            return Err(invalid(
                EINOPS_REARRANGE_OPERATION_ID,
                "input dimension is not divisible by known axis lengths",
            ));
        }
        lengths.insert(axis.clone(), dimension / known_product);
    } else if known_product != dimension {
        return Err(invalid(
            EINOPS_REARRANGE_OPERATION_ID,
            "input dimension does not match supplied axis lengths",
        ));
    }
    Ok(factors)
}

fn output_rearrange_group(
    axes: &[ParsedRearrangeAxis],
    ellipsis_axes: &[RearrangeAxis],
    lengths: &BTreeMap<RearrangeAxis, u64>,
) -> Result<Vec<RearrangeAxis>, ExternalTensorKernelPartTwoError> {
    let mut output = Vec::new();
    for axis in axes {
        match axis {
            ParsedRearrangeAxis::Named(name) => {
                let axis = RearrangeAxis::Named(name.clone());
                if !lengths.contains_key(&axis) {
                    return Err(invalid(
                        EINOPS_REARRANGE_OPERATION_ID,
                        format!("output axis {name} has no inferred length"),
                    ));
                }
                output.push(axis);
            }
            ParsedRearrangeAxis::Unit => {}
            ParsedRearrangeAxis::Ellipsis => output.extend_from_slice(ellipsis_axes),
        }
    }
    Ok(output)
}

fn rearrange_group_indices(
    groups: &[Vec<RearrangeAxis>],
    axis_indices: &BTreeMap<RearrangeAxis, usize>,
) -> Result<Vec<Vec<usize>>, ExternalTensorKernelPartTwoError> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|axis| {
                    axis_indices.get(axis).copied().ok_or_else(|| {
                        invalid(
                            EINOPS_REARRANGE_OPERATION_ID,
                            "rearrange atomic axis is not interned",
                        )
                    })
                })
                .collect()
        })
        .collect()
}

fn color_geometry(
    input: &Tensor,
    operation: &'static str,
) -> Result<(usize, usize), ExternalTensorKernelPartTwoError> {
    require_f32_cpu(input, operation)?;
    let shape = input.descriptor().shape();
    if shape.len() < 3 || shape[shape.len() - 3] != 3 {
        return Err(invalid(
            operation,
            "color input must have shape [...,3,H,W]",
        ));
    }
    let pixels = element_count(shape, operation, "color elements")? / 3;
    Ok((shape.len() - 3, pixels))
}

fn color_triplet(
    input: &Tensor,
    channel_axis: usize,
    pixel_linear: usize,
    operation: &'static str,
) -> Result<[f32; 3], ExternalTensorKernelPartTwoError> {
    let shape = input.descriptor().shape();
    let mut pixel_shape = shape.to_vec();
    pixel_shape.remove(channel_axis);
    let pixel_indices = unravel_index(pixel_linear, &pixel_shape, operation)?;
    let mut indices = vec![0_u64; shape.len()];
    let mut source_axis = 0;
    for (axis, index) in indices.iter_mut().enumerate() {
        if axis != channel_axis {
            *index = pixel_indices[source_axis];
            source_axis += 1;
        }
    }
    let mut output = [0.0; 3];
    for channel in 0..3 {
        indices[channel_axis] = channel as u64;
        output[channel] = read_f32(input, &indices, operation)?;
    }
    Ok(output)
}

fn write_color_triplet(
    output: &mut [f32],
    shape: &[u64],
    channel_axis: usize,
    pixel_linear: usize,
    values: [f32; 3],
    operation: &'static str,
) -> Result<(), ExternalTensorKernelPartTwoError> {
    let mut pixel_shape = shape.to_vec();
    pixel_shape.remove(channel_axis);
    let pixel_indices = unravel_index(pixel_linear, &pixel_shape, operation)?;
    let mut indices = vec![0_u64; shape.len()];
    for channel in 0..3 {
        let mut source_axis = 0;
        for (axis, index) in indices.iter_mut().enumerate() {
            if axis == channel_axis {
                *index = channel as u64;
            } else {
                *index = pixel_indices[source_axis];
                source_axis += 1;
            }
        }
        let destination = ravel_index(&indices, shape, operation)?;
        output[destination] = values[channel];
    }
    Ok(())
}

fn map_color_inputs<const INPUTS: usize>(
    backend: &CpuBackend,
    inputs: [&Tensor; INPUTS],
    operation: &'static str,
    context: &ExecutionContext<'_>,
    mut transform: impl FnMut([[f32; 3]; INPUTS]) -> [f32; 3],
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let input = inputs
        .first()
        .copied()
        .ok_or_else(|| invalid(operation, "color traversal requires at least one input"))?;
    let (channel_axis, pixels) = color_geometry(input, operation)?;
    for other in inputs.iter().skip(1) {
        require_same_shape_stream(input, other, operation)?;
        require_f32_cpu(other, operation)?;
    }
    let mut output = temporary_filled(
        backend,
        context,
        element_count(input.descriptor().shape(), operation, "color output")?,
        0.0,
        operation,
    )?;
    for pixel in 0..pixels {
        if pixel & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        let mut triplets = [[0.0; 3]; INPUTS];
        for (triplet, input) in triplets.iter_mut().zip(inputs) {
            *triplet = color_triplet(input, channel_axis, pixel, operation)?;
        }
        write_color_triplet(
            &mut output,
            input.descriptor().shape(),
            channel_axis,
            pixel,
            transform(triplets),
            operation,
        )?;
    }
    context.cancellation.check()?;
    upload_f32(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub(crate) fn map_color(
    backend: &CpuBackend,
    input: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    mut transform: impl FnMut([f32; 3]) -> [f32; 3],
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_inputs(
        backend,
        [input],
        operation,
        context,
        |[value]| transform(value),
    )
}

pub(crate) fn map_color_pair(
    backend: &CpuBackend,
    primal: &Tensor,
    other: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    mut transform: impl FnMut([f32; 3], [f32; 3]) -> [f32; 3],
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_inputs(
        backend,
        [primal, other],
        operation,
        context,
        |[primal, other]| transform(primal, other),
    )
}

fn srgb_to_linear(value: f32) -> f32 {
    if value > 0.04045 {
        ((value + 0.055) / 1.055).powf(2.4)
    } else {
        value / 12.92
    }
}

fn srgb_to_linear_derivative(value: f32) -> f32 {
    if value > 0.04045 {
        (2.4 / 1.055) * ((value + 0.055) / 1.055).powf(1.4)
    } else {
        1.0 / 12.92
    }
}

fn lab_transfer(value: f32) -> f32 {
    if value > 0.008856 {
        value.powf(1.0 / 3.0)
    } else {
        7.787 * value + 16.0 / 116.0
    }
}

fn lab_transfer_derivative(value: f32) -> f32 {
    if value > 0.008856 {
        1.0 / (3.0 * value.powf(2.0 / 3.0))
    } else {
        7.787
    }
}

fn rgb_to_lab_value(rgb: [f32; 3]) -> [f32; 3] {
    let linear = rgb.map(srgb_to_linear);
    let xyz = [
        0.412_453 * linear[0] + 0.357_580 * linear[1] + 0.180_423 * linear[2],
        0.212_671 * linear[0] + 0.715_160 * linear[1] + 0.072_169 * linear[2],
        0.019_334 * linear[0] + 0.119_193 * linear[1] + 0.950_227 * linear[2],
    ];
    let f = [
        lab_transfer(xyz[0] / 0.95047),
        lab_transfer(xyz[1]),
        lab_transfer(xyz[2] / 1.08883),
    ];
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        200.0 * (f[1] - f[2]),
    ]
}

fn rgb_to_lab_jacobian(rgb: [f32; 3]) -> [[f32; 3]; 3] {
    let linear = rgb.map(srgb_to_linear);
    let linear_derivative = rgb.map(srgb_to_linear_derivative);
    let xyz = [
        0.412_453 * linear[0] + 0.357_580 * linear[1] + 0.180_423 * linear[2],
        0.212_671 * linear[0] + 0.715_160 * linear[1] + 0.072_169 * linear[2],
        0.019_334 * linear[0] + 0.119_193 * linear[1] + 0.950_227 * linear[2],
    ];
    let xyz_matrix = [
        [
            0.412_453 / 0.95047,
            0.357_580 / 0.95047,
            0.180_423 / 0.95047,
        ],
        [0.212_671, 0.715_160, 0.072_169],
        [
            0.019_334 / 1.08883,
            0.119_193 / 1.08883,
            0.950_227 / 1.08883,
        ],
    ];
    let transfer = [
        lab_transfer_derivative(xyz[0] / 0.95047),
        lab_transfer_derivative(xyz[1]),
        lab_transfer_derivative(xyz[2] / 1.08883),
    ];
    let lab_matrix = [
        [0.0, 116.0, 0.0],
        [500.0, -500.0, 0.0],
        [0.0, 200.0, -200.0],
    ];
    let mut jacobian = [[0.0; 3]; 3];
    for output in 0..3 {
        for input in 0..3 {
            for intermediate in 0..3 {
                jacobian[output][input] += lab_matrix[output][intermediate]
                    * transfer[intermediate]
                    * xyz_matrix[intermediate][input]
                    * linear_derivative[input];
            }
        }
    }
    jacobian
}

pub(crate) fn apply_matrix(matrix: [[f32; 3]; 3], value: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * value[0] + row[1] * value[1] + row[2] * value[2])
}

pub(crate) fn apply_transpose(matrix: [[f32; 3]; 3], value: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * value[0] + matrix[1][0] * value[1] + matrix[2][0] * value[2],
        matrix[0][1] * value[0] + matrix[1][1] * value[1] + matrix[2][1] * value[2],
        matrix[0][2] * value[0] + matrix[1][2] * value[1] + matrix[2][2] * value[2],
    ]
}

pub fn rgb_to_lab_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color(
        backend,
        input,
        RGB_TO_LAB_OPERATION_ID,
        context,
        rgb_to_lab_value,
    )
}

pub fn rgb_to_lab_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_pair(
        backend,
        input,
        input_tangent,
        RGB_TO_LAB_OPERATION_ID,
        context,
        |value, tangent| apply_matrix(rgb_to_lab_jacobian(value), tangent),
    )
}

pub fn rgb_to_lab_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_pair(
        backend,
        input,
        output_gradient,
        RGB_TO_LAB_OPERATION_ID,
        context,
        |value, gradient| apply_transpose(rgb_to_lab_jacobian(value), gradient),
    )
}

fn rgb_to_ycbcr_value(rgb: [f32; 3]) -> [f32; 3] {
    let y = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
    let cb = (rgb[2] - y) * 0.564 + 0.5;
    let cr = (rgb[0] - y) * 0.713 + 0.5;
    [y, cb, cr]
}

fn rgb_to_ycbcr_tangent(value: [f32; 3]) -> [f32; 3] {
    let y = value[0] * 0.299 + value[1] * 0.587 + value[2] * 0.114;
    [y, (value[2] - y) * 0.564, (value[0] - y) * 0.713]
}

fn rgb_to_ycbcr_transpose(value: [f32; 3]) -> [f32; 3] {
    let y_gradient = value[0] - value[1] * 0.564 - value[2] * 0.713;
    [
        y_gradient * 0.299 + value[2] * 0.713,
        y_gradient * 0.587,
        y_gradient * 0.114 + value[1] * 0.564,
    ]
}

fn ycbcr_to_rgb_unclamped(value: [f32; 3]) -> [f32; 3] {
    let cb = value[1] - 0.5;
    let cr = value[2] - 0.5;
    [
        value[0] + 1.403 * cr,
        value[0] - 0.714 * cr - 0.344 * cb,
        value[0] + 1.773 * cb,
    ]
}

fn ycbcr_to_rgb_tangent(value: [f32; 3]) -> [f32; 3] {
    [
        value[0] + 1.403 * value[2],
        value[0] - 0.714 * value[2] - 0.344 * value[1],
        value[0] + 1.773 * value[1],
    ]
}

fn ycbcr_to_rgb_transpose(value: [f32; 3]) -> [f32; 3] {
    [
        value[0] + value[1] + value[2],
        -0.344 * value[1] + 1.773 * value[2],
        1.403 * value[0] - 0.714 * value[1],
    ]
}

pub fn rgb_to_ycbcr_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color(
        backend,
        input,
        RGB_TO_YCBCR_OPERATION_ID,
        context,
        rgb_to_ycbcr_value,
    )
}

pub fn rgb_to_ycbcr_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color(
        backend,
        input_tangent,
        RGB_TO_YCBCR_OPERATION_ID,
        context,
        rgb_to_ycbcr_tangent,
    )
}

pub fn rgb_to_ycbcr_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color(
        backend,
        output_gradient,
        RGB_TO_YCBCR_OPERATION_ID,
        context,
        rgb_to_ycbcr_transpose,
    )
}

pub fn ycbcr_to_rgb_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color(
        backend,
        input,
        YCBCR_TO_RGB_OPERATION_ID,
        context,
        |value| ycbcr_to_rgb_unclamped(value).map(|channel| channel.clamp(0.0, 1.0)),
    )
}

pub fn ycbcr_to_rgb_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_pair(
        backend,
        input,
        input_tangent,
        YCBCR_TO_RGB_OPERATION_ID,
        context,
        |primal, tangent| {
            let raw = ycbcr_to_rgb_unclamped(primal);
            let mut result = ycbcr_to_rgb_tangent(tangent);
            for channel in 0..3 {
                if !(0.0..=1.0).contains(&raw[channel]) {
                    result[channel] = 0.0;
                }
            }
            result
        },
    )
}

pub fn ycbcr_to_rgb_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    map_color_pair(
        backend,
        input,
        output_gradient,
        YCBCR_TO_RGB_OPERATION_ID,
        context,
        |primal, mut gradient| {
            let raw = ycbcr_to_rgb_unclamped(primal);
            for channel in 0..3 {
                if !(0.0..=1.0).contains(&raw[channel]) {
                    gradient[channel] = 0.0;
                }
            }
            ycbcr_to_rgb_transpose(gradient)
        },
    )
}

#[derive(Clone, Debug)]
pub struct NativeCannyOutput {
    magnitude: Tensor,
    edges: Tensor,
}

impl NativeCannyOutput {
    pub fn magnitude(&self) -> &Tensor {
        &self.magnitude
    }

    pub fn edges(&self) -> &Tensor {
        &self.edges
    }
}


pub fn canny_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    low_threshold: f32,
    high_threshold: f32,
    context: &ExecutionContext<'_>,
) -> Result<NativeCannyOutput, ExternalTensorKernelPartTwoError> {
    canny_impl(
        backend,
        input,
        low_threshold,
        high_threshold,
        context,
    )
}

fn canny_impl(
    backend: &CpuBackend,
    input: &Tensor,
    low_threshold: f32,
    high_threshold: f32,
    context: &ExecutionContext<'_>,
) -> Result<NativeCannyOutput, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    require_f32_cpu(input, CANNY_OPERATION_ID)?;
    if !low_threshold.is_finite()
        || !high_threshold.is_finite()
        || low_threshold <= 0.0
        || low_threshold > high_threshold
        || high_threshold >= 1.0
    {
        return Err(invalid(
            CANNY_OPERATION_ID,
            "Canny thresholds must satisfy 0 < low <= high < 1",
        ));
    }
    let [batch, channels, height, width] = input.descriptor().shape() else {
        return Err(invalid(
            CANNY_OPERATION_ID,
            "Canny expects BCHW rank four input",
        ));
    };
    if !matches!(channels, 1 | 3) {
        return Err(invalid(
            CANNY_OPERATION_ID,
            "Canny accepts one-channel or three-channel input",
        ));
    }
    if *height < 3 || *width < 3 {
        return Err(invalid(
            CANNY_OPERATION_ID,
            "Canny reflect filtering requires height and width of at least three",
        ));
    }
    let batch = usize::try_from(*batch).map_err(|_| overflow(CANNY_OPERATION_ID, "batch"))?;
    let channels =
        usize::try_from(*channels).map_err(|_| overflow(CANNY_OPERATION_ID, "channels"))?;
    let height = usize::try_from(*height).map_err(|_| overflow(CANNY_OPERATION_ID, "height"))?;
    let width = usize::try_from(*width).map_err(|_| overflow(CANNY_OPERATION_ID, "width"))?;
    let input_values = tensor_f32_values(backend, input, CANNY_OPERATION_ID, context)?;
    let pixel_count = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| overflow(CANNY_OPERATION_ID, "grayscale elements"))?;
    let mut grayscale = temporary_filled(
        backend,
        context,
        pixel_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    for batch_index in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let pixel = (batch_index * height + y) * width + x;
                if pixel & 0x3ff == 0 {
                    context.cancellation.check()?;
                }
                grayscale[pixel] = if channels == 1 {
                    input_values[pixel]
                } else {
                    let plane = height * width;
                    let base = batch_index * channels * plane + y * width + x;
                    input_values[base].mul_add(
                        0.299,
                        input_values[base + plane]
                            .mul_add(0.587, input_values[base + 2 * plane] * 0.114),
                    )
                };
            }
        }
    }

    const GAUSSIAN: [f32; 5] = [
        0.054_488_685,
        0.244_201_35,
        0.402_619_96,
        0.244_201_35,
        0.054_488_685,
    ];
    let horizontal_geometry = ConvolutionGeometry::new_with_padding_mode(
        2,
        vec![1, 1],
        vec![0, 2],
        vec![1, 1],
        1,
        false,
        vec![0, 0],
        ConvolutionPaddingMode::Reflect,
    )?;
    let mut horizontal = temporary_filled(
        backend,
        context,
        pixel_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    let horizontal_shape = convolution_into_with_context_exact_native(
        &grayscale,
        &[batch, 1, height, width],
        &GAUSSIAN,
        &[1, 1, 1, 5],
        None,
        &horizontal_geometry,
        DeviceId::CPU,
        &mut horizontal,
        context,
    )?;
    drop(input_values);
    drop(grayscale);
    let vertical_geometry = ConvolutionGeometry::new_with_padding_mode(
        2,
        vec![1, 1],
        vec![2, 0],
        vec![1, 1],
        1,
        false,
        vec![0, 0],
        ConvolutionPaddingMode::Reflect,
    )?;
    let mut blurred = temporary_filled(
        backend,
        context,
        pixel_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    let blurred_shape = convolution_into_with_context_exact_native(
        &horizontal,
        &horizontal_shape,
        &GAUSSIAN,
        &[1, 1, 5, 1],
        None,
        &vertical_geometry,
        DeviceId::CPU,
        &mut blurred,
        context,
    )?;
    drop(horizontal);
    const SOBEL: [f32; 18] = [
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0,
        1.0,
    ];
    let sobel_geometry = ConvolutionGeometry::new_with_padding_mode(
        2,
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
        1,
        false,
        vec![0, 0],
        ConvolutionPaddingMode::Replicate,
    )?;
    let gradient_count = pixel_count
        .checked_mul(2)
        .ok_or_else(|| overflow(CANNY_OPERATION_ID, "gradient elements"))?;
    let mut gradients = temporary_filled(
        backend,
        context,
        gradient_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    let gradient_shape = convolution_into_with_context_exact_native(
        &blurred,
        &blurred_shape,
        &SOBEL,
        &[2, 1, 3, 3],
        None,
        &sobel_geometry,
        DeviceId::CPU,
        &mut gradients,
        context,
    )?;
    if gradient_shape != [batch, 2, height, width] {
        return Err(invalid(
            CANNY_OPERATION_ID,
            "canonical Sobel shape mismatch",
        ));
    }
    drop(blurred);
    let mut raw_magnitude = temporary_filled(
        backend,
        context,
        pixel_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    let mut direction = temporary_filled(
        backend,
        context,
        pixel_count,
        0_i8,
        CANNY_OPERATION_ID,
    )?;
    for pixel in 0..pixel_count {
        if pixel & 0x3ff == 0 {
            context.cancellation.check()?;
        }
        let batch_index = pixel / (height * width);
        let spatial = pixel % (height * width);
        let gradient_base = batch_index * 2 * height * width + spatial;
        let gradient_x = gradients[gradient_base];
        let gradient_y = gradients[gradient_base + height * width];
        raw_magnitude[pixel] = gradient_x
            .mul_add(gradient_x, gradient_y.mul_add(gradient_y, 1e-6))
            .sqrt();
        direction[pixel] = ((gradient_y.atan2(gradient_x).to_degrees() / 45.0).round_ties_even()
            as i32)
            .rem_euclid(4) as i8;
    }
    drop(gradients);
    let mut magnitude = temporary_filled(
        backend,
        context,
        pixel_count,
        0.0_f32,
        CANNY_OPERATION_ID,
    )?;
    const NEIGHBORS: [((isize, isize), (isize, isize)); 4] = [
        ((0, -1), (0, 1)),
        ((-1, -1), (1, 1)),
        ((-1, 0), (1, 0)),
        ((-1, 1), (1, -1)),
    ];
    for batch_index in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let pixel = (batch_index * height + y) * width + x;
                if pixel & 0x3ff == 0 {
                    context.cancellation.check()?;
                }
                let pair = NEIGHBORS[usize::try_from(direction[pixel])
                    .map_err(|_| invalid(CANNY_OPERATION_ID, "invalid Canny direction"))?];
                let first =
                    canny_neighbor(&raw_magnitude, batch_index, height, width, y, x, pair.0);
                let second =
                    canny_neighbor(&raw_magnitude, batch_index, height, width, y, x, pair.1);
                if raw_magnitude[pixel] > first && raw_magnitude[pixel] > second {
                    magnitude[pixel] = raw_magnitude[pixel];
                }
            }
        }
    }
    drop(raw_magnitude);
    drop(direction);
    let mut weak = temporary_vec(backend, context, pixel_count, CANNY_OPERATION_ID)?;
    let mut strong = temporary_vec(backend, context, pixel_count, CANNY_OPERATION_ID)?;
    for value in magnitude.iter() {
        weak.try_push(*value > low_threshold)?;
        strong.try_push(*value > high_threshold)?;
    }
    loop {
        let mut changed = false;
        let mut previous = temporary_vec(backend, context, pixel_count, CANNY_OPERATION_ID)?;
        for value in strong.iter() {
            previous.try_push(*value)?;
        }
        for batch_index in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let pixel = (batch_index * height + y) * width + x;
                    if pixel & 0x3ff == 0 {
                        context.cancellation.check()?;
                    }
                    if weak[pixel]
                        && !previous[pixel]
                        && canny_has_strong_neighbor(&previous, batch_index, height, width, y, x)
                    {
                        strong[pixel] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut edges = temporary_vec(backend, context, pixel_count, CANNY_OPERATION_ID)?;
    for value in strong.iter() {
        edges.try_push(if *value { 1.0 } else { 0.0 })?;
    }
    context.cancellation.check()?;
    let output_shape = [batch as u64, 1, height as u64, width as u64];
    let magnitude = upload_f32(
        backend,
        &output_shape,
        input.descriptor().stream(),
        &magnitude,
        context,
    )?;
    let edges = upload_f32(
        backend,
        &output_shape,
        input.descriptor().stream(),
        &edges,
        context,
    )?;
    Ok(NativeCannyOutput { magnitude, edges })
}

fn canny_neighbor(
    values: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    offset: (isize, isize),
) -> f32 {
    let Some(y) = y.checked_add_signed(offset.0) else {
        return 0.0;
    };
    let Some(x) = x.checked_add_signed(offset.1) else {
        return 0.0;
    };
    if y >= height || x >= width {
        0.0
    } else {
        values[(batch * height + y) * width + x]
    }
}

fn canny_has_strong_neighbor(
    values: &[bool],
    batch: usize,
    height: usize,
    width: usize,
    y: usize,
    x: usize,
) -> bool {
    (-1_isize..=1).any(|offset_y| {
        (-1_isize..=1).any(|offset_x| {
            (offset_y != 0 || offset_x != 0)
                && y.checked_add_signed(offset_y)
                    .zip(x.checked_add_signed(offset_x))
                    .is_some_and(|(source_y, source_x)| {
                        source_y < height
                            && source_x < width
                            && values[(batch * height + source_y) * width + source_x]
                    })
        })
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDeformConv2dConfiguration {
    pub stride: [u64; 2],
    pub padding: [u64; 2],
    pub dilation: [u64; 2],
}

impl Default for NativeDeformConv2dConfiguration {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        }
    }
}

#[derive(Clone, Debug)]
struct NativeDeformConv2dPlan {
    batch: usize,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
    channels_per_weight_group: usize,
    output_channels_per_weight_group: usize,
    offset_groups: usize,
    channels_per_offset_group: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    stream: StreamId,
}

impl NativeDeformConv2dPlan {
    fn checked(
        input: &Tensor,
        offset: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        mask: Option<&Tensor>,
        configuration: NativeDeformConv2dConfiguration,
    ) -> Result<Self, ExternalTensorKernelPartTwoError> {
        for tensor in [Some(input), Some(offset), Some(weight), bias, mask]
            .into_iter()
            .flatten()
        {
            require_f32_cpu(tensor, DEFORM_CONV2D_OPERATION_ID)?;
            if tensor.descriptor().stream() != input.descriptor().stream() {
                return Err(invalid(
                    DEFORM_CONV2D_OPERATION_ID,
                    "deform-convolution tensors must use the same stream",
                ));
            }
        }
        let input_shape = tensor_shape_4(input, "input")?;
        let offset_shape = tensor_shape_4(offset, "offset")?;
        let weight_shape = tensor_shape_4(weight, "weight")?;
        if input_shape[1] == 0
            || input_shape[2] == 0
            || input_shape[3] == 0
            || weight_shape.contains(&0)
        {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "deform-convolution channel, spatial, and kernel dimensions must be non-zero",
            ));
        }
        let stride = configuration
            .stride
            .map(|value| usize::try_from(value))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| overflow(DEFORM_CONV2D_OPERATION_ID, "stride"))?;
        let padding = configuration
            .padding
            .map(|value| usize::try_from(value))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| overflow(DEFORM_CONV2D_OPERATION_ID, "padding"))?;
        let dilation = configuration
            .dilation
            .map(|value| usize::try_from(value))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| overflow(DEFORM_CONV2D_OPERATION_ID, "dilation"))?;
        let stride = [stride[0], stride[1]];
        let padding = [padding[0], padding[1]];
        let dilation = [dilation[0], dilation[1]];
        if stride.contains(&0) || dilation.contains(&0) {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "deform-convolution stride and dilation must be non-zero",
            ));
        }
        if !input_shape[1].is_multiple_of(weight_shape[1]) {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "input channels must be divisible by weight channels per group",
            ));
        }
        let weight_groups = input_shape[1] / weight_shape[1];
        if weight_groups == 0 || !weight_shape[0].is_multiple_of(weight_groups) {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "output channels must be divisible by weight groups",
            ));
        }
        let output_height = deform_conv_output_dimension(
            input_shape[2],
            padding[0],
            weight_shape[2],
            dilation[0],
            stride[0],
        )?;
        let output_width = deform_conv_output_dimension(
            input_shape[3],
            padding[1],
            weight_shape[3],
            dilation[1],
            stride[1],
        )?;
        let kernel_area = weight_shape[2]
            .checked_mul(weight_shape[3])
            .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "kernel area"))?;
        let offset_group_width = kernel_area
            .checked_mul(2)
            .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "offset channels"))?;
        if offset_shape[0] != input_shape[0]
            || offset_shape[2] != output_height
            || offset_shape[3] != output_width
            || offset_group_width == 0
            || !offset_shape[1].is_multiple_of(offset_group_width)
        {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "offset shape does not match deform-convolution geometry",
            ));
        }
        let offset_groups = offset_shape[1] / offset_group_width;
        if offset_groups == 0 || !input_shape[1].is_multiple_of(offset_groups) {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "input channels must be divisible by offset groups",
            ));
        }
        if let Some(mask) = mask {
            let mask_shape = tensor_shape_4(mask, "mask")?;
            let expected_channels = offset_groups
                .checked_mul(kernel_area)
                .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "mask channels"))?;
            if mask_shape
                != [
                    input_shape[0],
                    expected_channels,
                    output_height,
                    output_width,
                ]
            {
                return Err(invalid(
                    DEFORM_CONV2D_OPERATION_ID,
                    "mask shape does not match deform-convolution geometry",
                ));
            }
        }
        if let Some(bias) = bias
            && tensor_shape_1(bias, "bias")? != weight_shape[0]
        {
            return Err(invalid(
                DEFORM_CONV2D_OPERATION_ID,
                "bias length must match output channels",
            ));
        }
        Ok(Self {
            batch: input_shape[0],
            input_channels: input_shape[1],
            input_height: input_shape[2],
            input_width: input_shape[3],
            output_channels: weight_shape[0],
            output_height,
            output_width,
            channels_per_weight_group: weight_shape[1],
            output_channels_per_weight_group: weight_shape[0] / weight_groups,
            offset_groups,
            channels_per_offset_group: input_shape[1] / offset_groups,
            kernel_height: weight_shape[2],
            kernel_width: weight_shape[3],
            stride,
            padding,
            dilation,
            stream: input.descriptor().stream(),
        })
    }

    fn output_shape(&self) -> [u64; 4] {
        [
            self.batch as u64,
            self.output_channels as u64,
            self.output_height as u64,
            self.output_width as u64,
        ]
    }

    fn output_count(&self) -> Result<usize, ExternalTensorKernelPartTwoError> {
        self.batch
            .checked_mul(self.output_channels)
            .and_then(|value| value.checked_mul(self.output_height))
            .and_then(|value| value.checked_mul(self.output_width))
            .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "output elements"))
    }

    fn input_index(&self, batch: usize, channel: usize, y: u64, x: u64) -> usize {
        ((batch * self.input_channels + channel) * self.input_height + y as usize)
            * self.input_width
            + x as usize
    }

    fn weight_index(
        &self,
        output_channel: usize,
        input_channel_in_group: usize,
        kernel_y: usize,
        kernel_x: usize,
    ) -> usize {
        ((output_channel * self.channels_per_weight_group + input_channel_in_group)
            * self.kernel_height
            + kernel_y)
            * self.kernel_width
            + kernel_x
    }

    fn offset_index(
        &self,
        batch: usize,
        channel: usize,
        output_y: usize,
        output_x: usize,
    ) -> usize {
        ((batch * self.offset_groups * 2 * self.kernel_height * self.kernel_width + channel)
            * self.output_height
            + output_y)
            * self.output_width
            + output_x
    }

    fn mask_index(&self, batch: usize, channel: usize, output_y: usize, output_x: usize) -> usize {
        ((batch * self.offset_groups * self.kernel_height * self.kernel_width + channel)
            * self.output_height
            + output_y)
            * self.output_width
            + output_x
    }

    fn output_index(
        &self,
        batch: usize,
        output_channel: usize,
        output_y: usize,
        output_x: usize,
    ) -> usize {
        ((batch * self.output_channels + output_channel) * self.output_height + output_y)
            * self.output_width
            + output_x
    }
}


#[allow(clippy::too_many_arguments)]
pub fn deform_conv2d_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    deform_conv2d_impl(
        backend,
        input,
        offset,
        weight,
        bias,
        configuration,
        mask,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn deform_conv2d_impl(
    backend: &CpuBackend,
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let plan = NativeDeformConv2dPlan::checked(input, offset, weight, bias, mask, configuration)?;
    let input_values =
        tensor_f32_values(backend, input, DEFORM_CONV2D_OPERATION_ID, context)?;
    let offset_values =
        tensor_f32_values(backend, offset, DEFORM_CONV2D_OPERATION_ID, context)?;
    let weight_values =
        tensor_f32_values(backend, weight, DEFORM_CONV2D_OPERATION_ID, context)?;
    let bias_values = bias
        .map(|tensor| tensor_f32_values(backend, tensor, DEFORM_CONV2D_OPERATION_ID, context))
        .transpose()?;
    let mask_values = mask
        .map(|tensor| tensor_f32_values(backend, tensor, DEFORM_CONV2D_OPERATION_ID, context))
        .transpose()?;
    let output = deform_conv2d_forward_values(
        backend,
        &plan,
        &input_values,
        &offset_values,
        &weight_values,
        bias_values.as_deref(),
        mask_values.as_deref(),
        context,
    )?;
    upload_f32(backend, &plan.output_shape(), plan.stream, &output, context)
}

fn deform_conv2d_forward_values(
    backend: &CpuBackend,
    plan: &NativeDeformConv2dPlan,
    input: &[f32],
    offset: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    mask: Option<&[f32]>,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ExternalTensorKernelPartTwoError> {
    let mut output = temporary_filled(
        backend,
        context,
        plan.output_count()?,
        0.0_f32,
        DEFORM_CONV2D_OPERATION_ID,
    )?;
    for batch in 0..plan.batch {
        for output_channel in 0..plan.output_channels {
            let weight_group = output_channel / plan.output_channels_per_weight_group;
            let input_channel_start = weight_group * plan.channels_per_weight_group;
            for output_y in 0..plan.output_height {
                for output_x in 0..plan.output_width {
                    let output_index = plan.output_index(batch, output_channel, output_y, output_x);
                    if output_index & 0x3ff == 0 {
                        context.cancellation.check()?;
                    }
                    let mut value = bias.map_or(0.0, |values| values[output_channel]);
                    for input_channel_in_group in 0..plan.channels_per_weight_group {
                        let input_channel = input_channel_start + input_channel_in_group;
                        let offset_group = input_channel / plan.channels_per_offset_group;
                        for kernel_y in 0..plan.kernel_height {
                            for kernel_x in 0..plan.kernel_width {
                                let kernel = kernel_y * plan.kernel_width + kernel_x;
                                let offset_channel =
                                    offset_group * 2 * plan.kernel_height * plan.kernel_width
                                        + 2 * kernel;
                                let offset_y = offset
                                    [plan.offset_index(batch, offset_channel, output_y, output_x)];
                                let offset_x = offset[plan.offset_index(
                                    batch,
                                    offset_channel + 1,
                                    output_y,
                                    output_x,
                                )];
                                let sample_y = output_y as f32 * plan.stride[0] as f32
                                    - plan.padding[0] as f32
                                    + kernel_y as f32 * plan.dilation[0] as f32
                                    + offset_y;
                                let sample_x = output_x as f32 * plan.stride[1] as f32
                                    - plan.padding[1] as f32
                                    + kernel_x as f32 * plan.dilation[1] as f32
                                    + offset_x;
                                let mut sample = 0.0_f32;
                                for bilinear in checked_bilinear_weights(
                                    plan.input_height as u64,
                                    plan.input_width as u64,
                                    sample_y,
                                    sample_x,
                                    NativeBilinearBoundary::ZeroPadding,
                                    DEFORM_CONV2D_OPERATION_ID,
                                )? {
                                    sample = input[plan.input_index(
                                        batch,
                                        input_channel,
                                        bilinear.source_y,
                                        bilinear.source_x,
                                    )]
                                    .mul_add(bilinear.weight, sample);
                                }
                                let modulation = mask.map_or(1.0, |values| {
                                    values[plan.mask_index(
                                        batch,
                                        offset_group * plan.kernel_height * plan.kernel_width
                                            + kernel,
                                        output_y,
                                        output_x,
                                    )]
                                });
                                let weight_index = plan.weight_index(
                                    output_channel,
                                    input_channel_in_group,
                                    kernel_y,
                                    kernel_x,
                                );
                                value = (sample * modulation).mul_add(weight[weight_index], value);
                            }
                        }
                    }
                    output[output_index] = value;
                }
            }
        }
    }
    context.cancellation.check()?;
    Ok(output)
}

#[derive(Clone, Debug)]
pub struct NativeDeformConv2dVjp {
    pub input: Tensor,
    pub offset: Tensor,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub mask: Option<Tensor>,
}

#[allow(clippy::too_many_arguments)]
pub fn deform_conv2d_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<NativeDeformConv2dVjp, ExternalTensorKernelPartTwoError> {
    deform_conv2d_vjp_impl(
        backend,
        input,
        offset,
        weight,
        bias,
        configuration,
        mask,
        output_gradient,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn deform_conv2d_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<NativeDeformConv2dVjp, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let plan = NativeDeformConv2dPlan::checked(input, offset, weight, bias, mask, configuration)?;
    require_f32_cpu(output_gradient, DEFORM_CONV2D_OPERATION_ID)?;
    if output_gradient.descriptor().shape() != plan.output_shape()
        || output_gradient.descriptor().stream() != plan.stream
    {
        return Err(invalid(
            DEFORM_CONV2D_OPERATION_ID,
            "output gradient does not match deform-convolution output",
        ));
    }
    let input_values =
        tensor_f32_values(backend, input, DEFORM_CONV2D_OPERATION_ID, context)?;
    let offset_values =
        tensor_f32_values(backend, offset, DEFORM_CONV2D_OPERATION_ID, context)?;
    let weight_values =
        tensor_f32_values(backend, weight, DEFORM_CONV2D_OPERATION_ID, context)?;
    let mask_values = mask
        .map(|tensor| tensor_f32_values(backend, tensor, DEFORM_CONV2D_OPERATION_ID, context))
        .transpose()?;
    let output_gradient_values = tensor_f32_values(
        backend,
        output_gradient,
        DEFORM_CONV2D_OPERATION_ID,
        context,
    )?;
    let mut input_gradient = temporary_filled(
        backend,
        context,
        input_values.len(),
        0.0_f32,
        DEFORM_CONV2D_OPERATION_ID,
    )?;
    let mut offset_gradient = temporary_filled(
        backend,
        context,
        offset_values.len(),
        0.0_f32,
        DEFORM_CONV2D_OPERATION_ID,
    )?;
    let mut weight_gradient = temporary_filled(
        backend,
        context,
        weight_values.len(),
        0.0_f32,
        DEFORM_CONV2D_OPERATION_ID,
    )?;
    let mut mask_gradient = match mask_values.as_ref() {
        Some(values) => Some(temporary_filled(
            backend,
            context,
            values.len(),
            0.0_f32,
            DEFORM_CONV2D_OPERATION_ID,
        )?),
        None => None,
    };
    let mut bias_gradient = match bias {
        Some(_) => Some(temporary_filled(
            backend,
            context,
            plan.output_channels,
            0.0_f32,
            DEFORM_CONV2D_OPERATION_ID,
        )?),
        None => None,
    };
    for batch in 0..plan.batch {
        for output_channel in 0..plan.output_channels {
            let weight_group = output_channel / plan.output_channels_per_weight_group;
            let input_channel_start = weight_group * plan.channels_per_weight_group;
            for output_y in 0..plan.output_height {
                for output_x in 0..plan.output_width {
                    let output_index = plan.output_index(batch, output_channel, output_y, output_x);
                    if output_index & 0x3ff == 0 {
                        context.cancellation.check()?;
                    }
                    let output_derivative = output_gradient_values[output_index];
                    if let Some(gradient) = bias_gradient.as_mut() {
                        gradient[output_channel] += output_derivative;
                    }
                    for input_channel_in_group in 0..plan.channels_per_weight_group {
                        let input_channel = input_channel_start + input_channel_in_group;
                        let offset_group = input_channel / plan.channels_per_offset_group;
                        for kernel_y in 0..plan.kernel_height {
                            for kernel_x in 0..plan.kernel_width {
                                let kernel = kernel_y * plan.kernel_width + kernel_x;
                                let offset_channel =
                                    offset_group * 2 * plan.kernel_height * plan.kernel_width
                                        + 2 * kernel;
                                let offset_y_index =
                                    plan.offset_index(batch, offset_channel, output_y, output_x);
                                let offset_x_index = plan.offset_index(
                                    batch,
                                    offset_channel + 1,
                                    output_y,
                                    output_x,
                                );
                                let sample_y = output_y as f32 * plan.stride[0] as f32
                                    - plan.padding[0] as f32
                                    + kernel_y as f32 * plan.dilation[0] as f32
                                    + offset_values[offset_y_index];
                                let sample_x = output_x as f32 * plan.stride[1] as f32
                                    - plan.padding[1] as f32
                                    + kernel_x as f32 * plan.dilation[1] as f32
                                    + offset_values[offset_x_index];
                                let bilinear_weights = checked_bilinear_weights(
                                    plan.input_height as u64,
                                    plan.input_width as u64,
                                    sample_y,
                                    sample_x,
                                    NativeBilinearBoundary::ZeroPadding,
                                    DEFORM_CONV2D_OPERATION_ID,
                                )?;
                                let mut sample = 0.0_f32;
                                let mut sample_derivative_y = 0.0_f32;
                                let mut sample_derivative_x = 0.0_f32;
                                for bilinear in &bilinear_weights {
                                    let input_index = plan.input_index(
                                        batch,
                                        input_channel,
                                        bilinear.source_y,
                                        bilinear.source_x,
                                    );
                                    let input_value = input_values[input_index];
                                    sample = input_value.mul_add(bilinear.weight, sample);
                                    sample_derivative_y = input_value
                                        .mul_add(bilinear.derivative_y, sample_derivative_y);
                                    sample_derivative_x = input_value
                                        .mul_add(bilinear.derivative_x, sample_derivative_x);
                                }
                                let mask_index = plan.mask_index(
                                    batch,
                                    offset_group * plan.kernel_height * plan.kernel_width + kernel,
                                    output_y,
                                    output_x,
                                );
                                let modulation = mask_values
                                    .as_ref()
                                    .map_or(1.0, |values| values[mask_index]);
                                let weight_index = plan.weight_index(
                                    output_channel,
                                    input_channel_in_group,
                                    kernel_y,
                                    kernel_x,
                                );
                                let weight_value = weight_values[weight_index];
                                let common = output_derivative * weight_value * modulation;
                                for bilinear in &bilinear_weights {
                                    let input_index = plan.input_index(
                                        batch,
                                        input_channel,
                                        bilinear.source_y,
                                        bilinear.source_x,
                                    );
                                    input_gradient[input_index] = common
                                        .mul_add(bilinear.weight, input_gradient[input_index]);
                                }
                                weight_gradient[weight_index] =
                                    (output_derivative * sample * modulation)
                                        .mul_add(1.0, weight_gradient[weight_index]);
                                offset_gradient[offset_y_index] = (common * sample_derivative_y)
                                    .mul_add(1.0, offset_gradient[offset_y_index]);
                                offset_gradient[offset_x_index] = (common * sample_derivative_x)
                                    .mul_add(1.0, offset_gradient[offset_x_index]);
                                if let Some(gradient) = mask_gradient.as_mut() {
                                    gradient[mask_index] =
                                        (output_derivative * weight_value * sample)
                                            .mul_add(1.0, gradient[mask_index]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    context.cancellation.check()?;
    let input_gradient = upload_f32(
        backend,
        input.descriptor().shape(),
        plan.stream,
        &input_gradient,
        context,
    )?;
    let offset_gradient = upload_f32(
        backend,
        offset.descriptor().shape(),
        plan.stream,
        &offset_gradient,
        context,
    )?;
    let weight_gradient = upload_f32(
        backend,
        weight.descriptor().shape(),
        plan.stream,
        &weight_gradient,
        context,
    )?;
    let bias_gradient = match (bias, bias_gradient) {
        (Some(tensor), Some(values)) => Some(upload_f32(
            backend,
            tensor.descriptor().shape(),
            plan.stream,
            &values,
            context,
        )?),
        _ => None,
    };
    let mask_gradient = match (mask, mask_gradient) {
        (Some(tensor), Some(values)) => Some(upload_f32(
            backend,
            tensor.descriptor().shape(),
            plan.stream,
            &values,
            context,
        )?),
        _ => None,
    };
    Ok(NativeDeformConv2dVjp {
        input: input_gradient,
        offset: offset_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
        mask: mask_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn deform_conv2d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: Option<&Tensor>,
    offset: &Tensor,
    offset_tangent: Option<&Tensor>,
    weight: &Tensor,
    weight_tangent: Option<&Tensor>,
    bias: Option<&Tensor>,
    bias_tangent: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    mask_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    deform_conv2d_jvp_impl(
        backend,
        input,
        input_tangent,
        offset,
        offset_tangent,
        weight,
        weight_tangent,
        bias,
        bias_tangent,
        configuration,
        mask,
        mask_tangent,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn deform_conv2d_jvp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: Option<&Tensor>,
    offset: &Tensor,
    offset_tangent: Option<&Tensor>,
    weight: &Tensor,
    weight_tangent: Option<&Tensor>,
    bias: Option<&Tensor>,
    bias_tangent: Option<&Tensor>,
    configuration: NativeDeformConv2dConfiguration,
    mask: Option<&Tensor>,
    mask_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    let plan = NativeDeformConv2dPlan::checked(input, offset, weight, bias, mask, configuration)?;
    for (primal, tangent, name) in [
        (Some(input), input_tangent, "input"),
        (Some(offset), offset_tangent, "offset"),
        (Some(weight), weight_tangent, "weight"),
        (bias, bias_tangent, "bias"),
        (mask, mask_tangent, "mask"),
    ] {
        match (primal, tangent) {
            (Some(primal), Some(tangent)) => {
                require_matching_tensor(primal, tangent, name)?;
            }
            (None, Some(_)) => {
                return Err(invalid(
                    DEFORM_CONV2D_OPERATION_ID,
                    format!("{name} tangent requires a matching primal"),
                ));
            }
            _ => {}
        }
    }
    let input_values =
        tensor_f32_values(backend, input, DEFORM_CONV2D_OPERATION_ID, context)?;
    let offset_values =
        tensor_f32_values(backend, offset, DEFORM_CONV2D_OPERATION_ID, context)?;
    let weight_values =
        tensor_f32_values(backend, weight, DEFORM_CONV2D_OPERATION_ID, context)?;
    let mask_values = mask
        .map(|tensor| tensor_f32_values(backend, tensor, DEFORM_CONV2D_OPERATION_ID, context))
        .transpose()?;
    let input_tangent_values = optional_tensor_values(backend, input_tangent, context)?;
    let offset_tangent_values = optional_tensor_values(backend, offset_tangent, context)?;
    let weight_tangent_values = optional_tensor_values(backend, weight_tangent, context)?;
    let bias_tangent_values = optional_tensor_values(backend, bias_tangent, context)?;
    let mask_tangent_values = optional_tensor_values(backend, mask_tangent, context)?;
    let mut output = temporary_filled(
        backend,
        context,
        plan.output_count()?,
        0.0_f32,
        DEFORM_CONV2D_OPERATION_ID,
    )?;
    for batch in 0..plan.batch {
        for output_channel in 0..plan.output_channels {
            let weight_group = output_channel / plan.output_channels_per_weight_group;
            let input_channel_start = weight_group * plan.channels_per_weight_group;
            for output_y in 0..plan.output_height {
                for output_x in 0..plan.output_width {
                    let output_index = plan.output_index(batch, output_channel, output_y, output_x);
                    if output_index & 0x3ff == 0 {
                        context.cancellation.check()?;
                    }
                    let mut value = bias_tangent_values
                        .as_ref()
                        .map_or(0.0, |values| values[output_channel]);
                    for input_channel_in_group in 0..plan.channels_per_weight_group {
                        let input_channel = input_channel_start + input_channel_in_group;
                        let offset_group = input_channel / plan.channels_per_offset_group;
                        for kernel_y in 0..plan.kernel_height {
                            for kernel_x in 0..plan.kernel_width {
                                let kernel = kernel_y * plan.kernel_width + kernel_x;
                                let offset_channel =
                                    offset_group * 2 * plan.kernel_height * plan.kernel_width
                                        + 2 * kernel;
                                let offset_y_index =
                                    plan.offset_index(batch, offset_channel, output_y, output_x);
                                let offset_x_index = plan.offset_index(
                                    batch,
                                    offset_channel + 1,
                                    output_y,
                                    output_x,
                                );
                                let sample_y = output_y as f32 * plan.stride[0] as f32
                                    - plan.padding[0] as f32
                                    + kernel_y as f32 * plan.dilation[0] as f32
                                    + offset_values[offset_y_index];
                                let sample_x = output_x as f32 * plan.stride[1] as f32
                                    - plan.padding[1] as f32
                                    + kernel_x as f32 * plan.dilation[1] as f32
                                    + offset_values[offset_x_index];
                                let bilinear_weights = checked_bilinear_weights(
                                    plan.input_height as u64,
                                    plan.input_width as u64,
                                    sample_y,
                                    sample_x,
                                    NativeBilinearBoundary::ZeroPadding,
                                    DEFORM_CONV2D_OPERATION_ID,
                                )?;
                                let mut sample = 0.0_f32;
                                let mut sample_tangent = 0.0_f32;
                                for bilinear in &bilinear_weights {
                                    let input_index = plan.input_index(
                                        batch,
                                        input_channel,
                                        bilinear.source_y,
                                        bilinear.source_x,
                                    );
                                    let input_value = input_values[input_index];
                                    sample = input_value.mul_add(bilinear.weight, sample);
                                    sample_tangent = input_tangent_values
                                        .as_ref()
                                        .map_or(0.0, |values| values[input_index])
                                        .mul_add(bilinear.weight, sample_tangent);
                                    if let Some(values) = offset_tangent_values.as_ref() {
                                        sample_tangent = (input_value
                                            * (bilinear.derivative_y * values[offset_y_index]
                                                + bilinear.derivative_x * values[offset_x_index]))
                                            .mul_add(1.0, sample_tangent);
                                    }
                                }
                                let mask_index = plan.mask_index(
                                    batch,
                                    offset_group * plan.kernel_height * plan.kernel_width + kernel,
                                    output_y,
                                    output_x,
                                );
                                let modulation = mask_values
                                    .as_ref()
                                    .map_or(1.0, |values| values[mask_index]);
                                let modulation_tangent = mask_tangent_values
                                    .as_ref()
                                    .map_or(0.0, |values| values[mask_index]);
                                let weight_index = plan.weight_index(
                                    output_channel,
                                    input_channel_in_group,
                                    kernel_y,
                                    kernel_x,
                                );
                                let weight_value = weight_values[weight_index];
                                let weight_derivative = weight_tangent_values
                                    .as_ref()
                                    .map_or(0.0, |values| values[weight_index]);
                                value += weight_derivative * sample * modulation
                                    + weight_value * sample_tangent * modulation
                                    + weight_value * sample * modulation_tangent;
                            }
                        }
                    }
                    output[output_index] = value;
                }
            }
        }
    }
    upload_f32(backend, &plan.output_shape(), plan.stream, &output, context)
}

fn require_matching_tensor(
    primal: &Tensor,
    tangent: &Tensor,
    name: &str,
) -> Result<(), ExternalTensorKernelPartTwoError> {
    require_f32_cpu(tangent, DEFORM_CONV2D_OPERATION_ID)?;
    if primal.descriptor().shape() != tangent.descriptor().shape()
        || primal.descriptor().stream() != tangent.descriptor().stream()
    {
        return Err(invalid(
            DEFORM_CONV2D_OPERATION_ID,
            format!("{name} tangent does not match its primal"),
        ));
    }
    Ok(())
}

fn optional_tensor_values(
    backend: &CpuBackend,
    tensor: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Option<CpuWorkspaceVec<f32>>, ExternalTensorKernelPartTwoError> {
    tensor
        .map(|tensor| tensor_f32_values(backend, tensor, DEFORM_CONV2D_OPERATION_ID, context))
        .transpose()
}

fn tensor_shape_4(
    tensor: &Tensor,
    name: &'static str,
) -> Result<[usize; 4], ExternalTensorKernelPartTwoError> {
    let [first, second, third, fourth] = tensor.descriptor().shape() else {
        return Err(invalid(
            DEFORM_CONV2D_OPERATION_ID,
            format!("deform-convolution {name} must be rank four"),
        ));
    };
    [first, second, third, fourth]
        .map(|value| usize::try_from(*value))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map(|shape| [shape[0], shape[1], shape[2], shape[3]])
        .map_err(|_| overflow(DEFORM_CONV2D_OPERATION_ID, "tensor shape"))
}

fn tensor_shape_1(
    tensor: &Tensor,
    name: &'static str,
) -> Result<usize, ExternalTensorKernelPartTwoError> {
    let [length] = tensor.descriptor().shape() else {
        return Err(invalid(
            DEFORM_CONV2D_OPERATION_ID,
            format!("deform-convolution {name} must be rank one"),
        ));
    };
    usize::try_from(*length).map_err(|_| overflow(DEFORM_CONV2D_OPERATION_ID, "tensor shape"))
}

fn deform_conv_output_dimension(
    input: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
) -> Result<usize, ExternalTensorKernelPartTwoError> {
    let padded = input
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "padding"))?,
        )
        .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "padded input"))?;
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| overflow(DEFORM_CONV2D_OPERATION_ID, "effective kernel"))?;
    if padded < effective_kernel {
        return Err(invalid(
            DEFORM_CONV2D_OPERATION_ID,
            "effective kernel is larger than the padded input",
        ));
    }
    Ok((padded - effective_kernel) / stride + 1)
}


pub fn dilation_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    kernel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    Ok(native_morphology_with_context_exact(
        backend,
        input,
        kernel,
        NativeMorphologyOperation::Dilation,
        context,
    )?)
}


pub fn erosion_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    kernel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    Ok(native_morphology_with_context_exact(
        backend,
        input,
        kernel,
        NativeMorphologyOperation::Erosion,
        context,
    )?)
}


pub fn top_hat_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    kernel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    Ok(native_morphology_with_context_exact(
        backend,
        input,
        kernel,
        NativeMorphologyOperation::TopHat,
        context,
    )?)
}


pub fn to_pil_image_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Rgb8ImageTensor, ExternalTensorKernelPartTwoError> {
    context.cancellation.check()?;
    Ok(Rgb8ImageTensor::from_logical_chw(backend, context, input)?)
}
