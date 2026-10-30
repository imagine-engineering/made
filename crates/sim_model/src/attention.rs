use serde::{Deserialize, Serialize};
use sim_tensor::{
    CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, Scalar, Tensor, TensorDescriptor,
    TensorError,
    generated_accelerated_attention_kernel_01::{
        AttentionKernelError, AttentionKernelKind, AttentionKernelRequest,
        AttentionLayout as TensorAttentionLayout, AttentionMask as TensorAttentionMask,
        AttentionMaskShape as TensorAttentionMaskShape, AttentionShape, CheckedAttentionInvocation,
    },
};
use sim_types::CancellationToken;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionBackend {
    PytorchSdp,
    SageOrFlash,
    SplitOrSubQuadratic,
    Xformers,
}

impl AttentionBackend {
    pub const fn feature_id(self) -> &'static str {
        match self {
            Self::PytorchSdp => "COMFY-MODEL-0001",
            Self::SageOrFlash => "COMFY-MODEL-0002",
            Self::SplitOrSubQuadratic => "COMFY-MODEL-0003",
            Self::Xformers => "COMFY-MODEL-0004",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionFallbackPolicy {
    AllowExactNative,
    Forbid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MathSdpSelection {
    Enabled(AttentionKernelKind),
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MathSdpReductionPolicy {
    allow_fp16_bf16: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdpaBackend {
    CudnnAttention,
    FlashAttention,
    EfficientAttention,
    Math,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdpaKernelSelection {
    backends: Vec<SdpaBackend>,
    set_priority: bool,
}

impl SdpaKernelSelection {
    pub fn backends(&self) -> &[SdpaBackend] {
        &self.backends
    }

    pub const fn set_priority(&self) -> bool {
        self.set_priority
    }
}

impl MathSdpReductionPolicy {
    pub const fn allow_fp16_bf16(self) -> bool {
        self.allow_fp16_bf16
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionMaskShape {
    KeyTokens,
    QueryByKey,
    BatchQueryByKey,
    BatchHeadQueryByKey,
}

#[derive(Clone, Copy, Debug)]
pub enum AttentionMask<'a> {
    Boolean {
        values: &'a [bool],
        shape: AttentionMaskShape,
    },
    Additive {
        values: &'a [f32],
        shape: AttentionMaskShape,
    },
    OrderedAdditive {
        first_values: &'a [f32],
        second_values: &'a [f32],
        shape: AttentionMaskShape,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionRequest {
    pub backend: AttentionBackend,
    pub fallback: AttentionFallbackPolicy,
    pub batch: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub heads: usize,
    pub head_dimension: usize,
    pub value_dimension: usize,
    pub scale: Option<f32>,
    pub workspace_limit_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryPairLayout {
    SplitHalf,
    SplitHeadHalfPrefix { rotated_pairs: usize },
    Adjacent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryFrequencyLayout {
    Global,
    ResetPerAxis,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotaryScaling {
    None,
    Linear {
        factor: f32,
    },
    Yarn {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum RotaryPositionSequence<'a> {
    Unsigned(&'a [usize]),
    Signed(&'a [i64]),
    Float(&'a [f32]),
}

impl RotaryPositionSequence<'_> {
    fn len(self) -> usize {
        match self {
            Self::Unsigned(values) => values.len(),
            Self::Signed(values) => values.len(),
            Self::Float(values) => values.len(),
        }
    }

    fn position(self, index: usize) -> Option<f64> {
        match self {
            Self::Unsigned(values) => values.get(index).map(|value| f64::from(*value as f32)),
            Self::Signed(values) => values.get(index).map(|value| f64::from(*value as f32)),
            Self::Float(values) => values.get(index).map(|value| f64::from(*value)),
        }
    }

    fn all_finite(self) -> bool {
        match self {
            Self::Unsigned(_) | Self::Signed(_) => true,
            Self::Float(values) => values.iter().all(|value| value.is_finite()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RotaryPositions<'a> {
    Scalar(RotaryPositionSequence<'a>),
    Multiaxis(&'a [RotaryPositionSequence<'a>]),
}

#[derive(Clone, Copy, Debug)]
pub struct RotaryTableRequest<'a> {
    pub positions: RotaryPositions<'a>,
    pub axis_dimensions: &'a [usize],
    pub rotary_dimension: usize,
    pub theta: f32,
    pub scaling: RotaryScaling,
    pub frequency_layout: RotaryFrequencyLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RotaryTable {
    entries: RotaryTableEntries,
    tokens: usize,
    rotary_dimension: usize,
}

impl RotaryTable {
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    pub const fn rotary_dimension(&self) -> usize {
        self.rotary_dimension
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn legacy_f32_entries(&self) -> Option<&[[f32; 2]]> {
        match &self.entries {
            RotaryTableEntries::GlobalF32(entries) => Some(entries),
            RotaryTableEntries::ResetPerAxisF64ToF32(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum RotaryTableEntries {
    GlobalF32(Vec<[f32; 2]>),
    ResetPerAxisF64ToF32(Vec<[f32; 2]>),
}

impl RotaryTableEntries {
    fn with_capacity(
        frequency_layout: RotaryFrequencyLayout,
        entry_count: usize,
    ) -> Result<Self, AttentionError> {
        match frequency_layout {
            RotaryFrequencyLayout::Global => {
                let mut entries = Vec::new();
                entries.try_reserve_exact(entry_count).map_err(|_| {
                    AttentionError::AllocationFailed {
                        name: "rotary table",
                    }
                })?;
                Ok(Self::GlobalF32(entries))
            }
            RotaryFrequencyLayout::ResetPerAxis => {
                let mut entries = Vec::new();
                entries.try_reserve_exact(entry_count).map_err(|_| {
                    AttentionError::AllocationFailed {
                        name: "rotary table",
                    }
                })?;
                Ok(Self::ResetPerAxisF64ToF32(entries))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::GlobalF32(entries) | Self::ResetPerAxisF64ToF32(entries) => entries.len(),
        }
    }

    fn push(
        &mut self,
        position: f64,
        pair: usize,
        pairs: usize,
        theta: f32,
        scaling: RotaryScaling,
    ) {
        match self {
            Self::GlobalF32(entries) => entries.push(rotary_entry_f32(
                position as f32,
                pair,
                pairs,
                theta,
                scaling,
            )),
            Self::ResetPerAxisF64ToF32(entries) => entries.push(rotary_entry_f64_to_f32(
                position,
                pair,
                pairs,
                f64::from(theta),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionOutcome {
    pub values: Vec<f32>,
    pub requested_backend: AttentionBackend,
    pub effective_backend: AttentionBackend,
    pub fallback_reason: Option<String>,
    pub query_chunk_size: usize,
    pub peak_workspace_bytes: usize,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum AttentionError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("attention dimensions or workspace size overflowed")]
    ShapeOverflow,
    #[error("attention allocation for {name} failed")]
    AllocationFailed { name: &'static str },
    #[error("attention dimension {name} must be non-zero")]
    EmptyDimension { name: &'static str },
    #[error("attention {name} expected {expected} values, got {actual}")]
    ValueCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("attention mask expected {expected} values, got {actual}")]
    MaskValueCount { expected: usize, actual: usize },
    #[error("attention ordered additive {term} mask value at index {index} is not finite")]
    NonFiniteOrderedMask { term: &'static str, index: usize },
    #[error("attention scale must be finite and greater than zero")]
    InvalidScale,
    #[error("attention tensor {name} expected shape {expected:?}, got {actual:?}")]
    TensorShape {
        name: &'static str,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("attention tensor dtype {dtype:?} is unsupported; expected F32, F16, or BF16")]
    UnsupportedTensorDType { dtype: DType },
    #[error("attention workspace requires at least {required} bytes, got {available}")]
    WorkspaceTooSmall { required: usize, available: usize },
    #[error("attention backend {backend:?} is unavailable: {reason}")]
    UnsupportedBackend {
        backend: AttentionBackend,
        reason: String,
    },
    #[error("attention was cancelled")]
    Cancelled,
    #[error("SDPA kernel selection must contain between one and four unique backends")]
    InvalidSdpaSelection,
    #[error("rotary configuration is invalid: {reason}")]
    InvalidRotaryConfiguration { reason: &'static str },
    #[error("rotary input is invalid: {reason}")]
    InvalidRotaryInput { reason: &'static str },
}

pub fn precompute_rotary_table(
    request: RotaryTableRequest<'_>,
    cancellation: &CancellationToken,
) -> Result<RotaryTable, AttentionError> {
    cancellation.check()?;
    validate_rotary_configuration(request)?;
    let (tokens, axes) = match request.positions {
        RotaryPositions::Scalar(positions) => (positions.len(), None),
        RotaryPositions::Multiaxis(axes) => (
            axes.first()
                .map(|axis| axis.len())
                .ok_or(AttentionError::InvalidRotaryInput {
                    reason: "multiaxis positions require at least one axis",
                })?,
            Some(axes),
        ),
    };
    if tokens == 0 {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary positions must not be empty",
        });
    }
    if axes.is_some_and(|axes| axes.iter().any(|axis| axis.len() != tokens)) {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "multiaxis positions must have equal token lengths",
        });
    }
    let finite = match request.positions {
        RotaryPositions::Scalar(positions) => positions.all_finite(),
        RotaryPositions::Multiaxis(axes) => axes.iter().all(|axis| axis.all_finite()),
    };
    if !finite {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary positions must be finite",
        });
    }
    let pairs = request.rotary_dimension / 2;
    let entry_count = tokens
        .checked_mul(pairs)
        .ok_or(AttentionError::ShapeOverflow)?;
    let mut entries = RotaryTableEntries::with_capacity(request.frequency_layout, entry_count)?;

    for token in 0..tokens {
        match request.positions {
            RotaryPositions::Scalar(positions) => {
                let position =
                    positions
                        .position(token)
                        .ok_or(AttentionError::InvalidRotaryInput {
                            reason: "scalar rotary position is missing",
                        })?;
                for pair in 0..pairs {
                    if (token * pairs + pair).is_multiple_of(256) {
                        cancellation.check()?;
                    }
                    entries.push(position, pair, pairs, request.theta, request.scaling);
                }
            }
            RotaryPositions::Multiaxis(axes) => {
                let mut global_pair = 0_usize;
                for (axis_index, axis_dimension) in
                    request.axis_dimensions.iter().copied().enumerate()
                {
                    let axis_pairs = axis_dimension / 2;
                    let position = axes
                        .get(axis_index)
                        .and_then(|axis| axis.position(token))
                        .ok_or(AttentionError::InvalidRotaryInput {
                            reason: "multiaxis rotary position is missing",
                        })?;
                    for local_pair in 0..axis_pairs {
                        if (token * pairs + global_pair).is_multiple_of(256) {
                            cancellation.check()?;
                        }
                        let (frequency_pair, frequency_pairs) = match request.frequency_layout {
                            RotaryFrequencyLayout::Global => (global_pair, pairs),
                            RotaryFrequencyLayout::ResetPerAxis => (local_pair, axis_pairs),
                        };
                        entries.push(
                            position,
                            frequency_pair,
                            frequency_pairs,
                            request.theta,
                            request.scaling,
                        );
                        global_pair = global_pair
                            .checked_add(1)
                            .ok_or(AttentionError::ShapeOverflow)?;
                    }
                }
            }
        }
    }
    cancellation.check()?;
    Ok(RotaryTable {
        entries,
        tokens,
        rotary_dimension: request.rotary_dimension,
    })
}

fn validate_rotary_configuration(request: RotaryTableRequest<'_>) -> Result<(), AttentionError> {
    if request.rotary_dimension == 0 || !request.rotary_dimension.is_multiple_of(2) {
        return Err(AttentionError::InvalidRotaryConfiguration {
            reason: "rotary dimension must be nonzero and even",
        });
    }
    if !request.theta.is_finite() || request.theta <= 0.0 {
        return Err(AttentionError::InvalidRotaryConfiguration {
            reason: "theta must be finite and positive",
        });
    }
    match request.scaling {
        RotaryScaling::None => {}
        RotaryScaling::Linear { factor } if factor.is_finite() && factor >= 1.0 => {}
        RotaryScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
        } if factor.is_finite()
            && factor >= 1.0
            && beta_fast.is_finite()
            && beta_slow.is_finite()
            && beta_fast > beta_slow
            && beta_slow >= 0.0 => {}
        _ => {
            return Err(AttentionError::InvalidRotaryConfiguration {
                reason: "rotary scaling parameters are invalid",
            });
        }
    }
    match request.positions {
        RotaryPositions::Scalar(_) => {
            if !request.axis_dimensions.is_empty()
                || request.frequency_layout != RotaryFrequencyLayout::Global
            {
                return Err(AttentionError::InvalidRotaryConfiguration {
                    reason: "scalar rotary positions cannot declare frequency axes",
                });
            }
        }
        RotaryPositions::Multiaxis(axes) => {
            if axes.is_empty() || axes.len() != request.axis_dimensions.len() {
                return Err(AttentionError::InvalidRotaryConfiguration {
                    reason: "multiaxis positions and dimensions must have equal nonzero counts",
                });
            }
            let total = request
                .axis_dimensions
                .iter()
                .try_fold(0_usize, |sum, dimension| {
                    if !dimension.is_multiple_of(2)
                        || (request.frequency_layout == RotaryFrequencyLayout::ResetPerAxis
                            && *dimension == 0)
                    {
                        return None;
                    }
                    sum.checked_add(*dimension)
                });
            if total != Some(request.rotary_dimension) {
                return Err(AttentionError::InvalidRotaryConfiguration {
                    reason: "multiaxis dimensions must be even and cover the rotary width",
                });
            }
        }
    }
    if request.frequency_layout == RotaryFrequencyLayout::ResetPerAxis
        && request.scaling != RotaryScaling::None
    {
        return Err(AttentionError::InvalidRotaryConfiguration {
            reason: "per-axis source frequencies do not support decoder scaling",
        });
    }
    Ok(())
}

fn rotary_entry_f32(
    position: f32,
    pair: usize,
    pairs: usize,
    theta: f32,
    scaling: RotaryScaling,
) -> [f32; 2] {
    let exponent = pair as f32 / pairs as f32;
    let mut frequency = theta.powf(-exponent);
    let mut scaled_position = position;
    match scaling {
        RotaryScaling::None => {}
        RotaryScaling::Linear { factor } => scaled_position /= factor,
        RotaryScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
        } => {
            let progress = if pairs <= 1 {
                0.0
            } else {
                pair as f32 / (pairs - 1) as f32
            };
            let ramp =
                ((progress * beta_fast - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0);
            frequency *= (1.0 - ramp) / factor + ramp;
        }
    }
    let angle = scaled_position * frequency;
    [angle.cos(), angle.sin()]
}

fn rotary_entry_f64_to_f32(position: f64, pair: usize, pairs: usize, theta: f64) -> [f32; 2] {
    let exponent = pair as f64 / pairs as f64;
    let angle = position * theta.powf(-exponent);
    [angle.cos() as f32, angle.sin() as f32]
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rotary_table(
    values: &[f32],
    batch: usize,
    tokens: usize,
    heads: usize,
    head_dimension: usize,
    table: &RotaryTable,
    layout: RotaryPairLayout,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, AttentionError> {
    cancellation.check()?;
    validate_rotary_application(
        values.len(),
        batch,
        tokens,
        heads,
        head_dimension,
        table,
        layout,
    )?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| AttentionError::AllocationFailed {
            name: "rotary output",
        })?;
    output.extend_from_slice(values);
    apply_rotary_values_in_place(
        values,
        &mut output,
        batch,
        tokens,
        heads,
        head_dimension,
        table,
        layout,
        cancellation,
    )?;
    cancellation.check()?;
    Ok(output)
}

pub fn apply_rotary_table_tensor_with_context(
    backend: &CpuBackend,
    values: &Tensor,
    table: &RotaryTable,
    layout: RotaryPairLayout,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AttentionError> {
    context.cancellation.check()?;
    let (batch, tokens, heads, head_dimension, dtype) =
        validate_rotary_tensor_descriptor(values.descriptor(), context)?;
    let shape = values.descriptor().shape();
    let input = tensor_to_f32_workspace(backend, values, context)?;
    validate_rotary_application(
        input.len(),
        batch,
        tokens,
        heads,
        head_dimension,
        table,
        layout,
    )?;
    let mut output = backend.workspace_vec::<f32>(context, input.len())?;
    for value in input.iter().copied() {
        context.cancellation.check()?;
        output.try_push(value)?;
    }
    apply_rotary_values_in_place(
        &input,
        &mut output,
        batch,
        tokens,
        heads,
        head_dimension,
        table,
        layout,
        context.cancellation,
    )?;
    let encoded_length = input
        .len()
        .checked_mul(
            usize::try_from(dtype.byte_width()).map_err(|_| AttentionError::ShapeOverflow)?,
        )
        .ok_or(AttentionError::ShapeOverflow)?;
    let mut encoded = backend.workspace_vec::<u8>(context, encoded_length)?;
    for value in output.iter().copied() {
        context.cancellation.check()?;
        for byte in dtype.encode_scalar(
            Scalar::Float(f64::from(value)),
            "zed.sim-model.rotary-attention",
            DeviceId::CPU,
        )? {
            encoded.try_push(byte)?;
        }
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, context.stream)?;
    let tensor = backend.upload_bytes(descriptor, &encoded, context)?.0;
    drop(encoded);
    drop(output);
    drop(input);
    context.cancellation.check()?;
    Ok(tensor)
}

fn validate_rotary_tensor_descriptor(
    descriptor: &TensorDescriptor,
    context: &ExecutionContext<'_>,
) -> Result<(usize, usize, usize, usize, DType), AttentionError> {
    let shape = descriptor.shape();
    if shape.len() != 4 {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary tensor must have batch, token, head, and channel axes",
        });
    }
    let batch = usize::try_from(shape[0]).map_err(|_| AttentionError::ShapeOverflow)?;
    let tokens = usize::try_from(shape[1]).map_err(|_| AttentionError::ShapeOverflow)?;
    let heads = usize::try_from(shape[2]).map_err(|_| AttentionError::ShapeOverflow)?;
    let head_dimension = usize::try_from(shape[3]).map_err(|_| AttentionError::ShapeOverflow)?;
    let dtype = descriptor.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(AttentionError::UnsupportedTensorDType { dtype });
    }
    if descriptor.device() != DeviceId::CPU {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::CPU,
            actual: descriptor.device(),
        }
        .into());
    }
    if descriptor.stream() != context.stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: descriptor.stream(),
        }
        .into());
    }
    Ok((batch, tokens, heads, head_dimension, dtype))
}

fn validate_rotary_application(
    values: usize,
    batch: usize,
    tokens: usize,
    heads: usize,
    head_dimension: usize,
    table: &RotaryTable,
    layout: RotaryPairLayout,
) -> Result<(), AttentionError> {
    for (name, value) in [
        ("batch", batch),
        ("tokens", tokens),
        ("heads", heads),
        ("head_dimension", head_dimension),
    ] {
        if value == 0 {
            return Err(AttentionError::EmptyDimension { name });
        }
    }
    if table.rotary_dimension > head_dimension {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary width exceeds the attention head width",
        });
    }
    match layout {
        RotaryPairLayout::SplitHalf | RotaryPairLayout::Adjacent => {}
        RotaryPairLayout::SplitHeadHalfPrefix { rotated_pairs }
            if head_dimension.is_multiple_of(2)
                && rotated_pairs > 0
                && rotated_pairs <= table.rotary_dimension / 2 => {}
        RotaryPairLayout::SplitHeadHalfPrefix { .. } => {
            return Err(AttentionError::InvalidRotaryInput {
                reason: "split-head rotary prefix must cover a nonzero bounded pair count in an even head width",
            });
        }
    }
    if table.tokens != tokens {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary table token count does not match the value tensor",
        });
    }
    let expected = checked_product(&[batch, tokens, heads, head_dimension])?;
    if values != expected {
        return Err(AttentionError::ValueCount {
            name: "rotary values",
            expected,
            actual: values,
        });
    }
    let table_entries = tokens
        .checked_mul(table.rotary_dimension / 2)
        .ok_or(AttentionError::ShapeOverflow)?;
    if table.entries.len() != table_entries {
        return Err(AttentionError::InvalidRotaryInput {
            reason: "rotary table storage is incomplete",
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_rotary_values_in_place(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    tokens: usize,
    heads: usize,
    head_dimension: usize,
    table: &RotaryTable,
    layout: RotaryPairLayout,
    cancellation: &CancellationToken,
) -> Result<(), AttentionError> {
    let table_pairs = table.rotary_dimension / 2;
    let pairs = match layout {
        RotaryPairLayout::SplitHalf | RotaryPairLayout::Adjacent => table_pairs,
        RotaryPairLayout::SplitHeadHalfPrefix { rotated_pairs } => rotated_pairs,
    };
    for batch_index in 0..batch {
        for token in 0..tokens {
            for head in 0..heads {
                for pair in 0..pairs {
                    let work = (((batch_index * tokens + token) * heads + head) * pairs) + pair;
                    if work.is_multiple_of(256) {
                        cancellation.check()?;
                    }
                    let base = ((batch_index * tokens + token) * heads + head)
                        .checked_mul(head_dimension)
                        .ok_or(AttentionError::ShapeOverflow)?;
                    let (left_offset, right_offset) = match layout {
                        RotaryPairLayout::SplitHalf => (pair, pair + table_pairs),
                        RotaryPairLayout::SplitHeadHalfPrefix { .. } => {
                            (pair, pair + head_dimension / 2)
                        }
                        RotaryPairLayout::Adjacent => {
                            let left = pair.checked_mul(2).ok_or(AttentionError::ShapeOverflow)?;
                            (left, left + 1)
                        }
                    };
                    let left_index = base
                        .checked_add(left_offset)
                        .ok_or(AttentionError::ShapeOverflow)?;
                    let right_index = base
                        .checked_add(right_offset)
                        .ok_or(AttentionError::ShapeOverflow)?;
                    let left =
                        *input
                            .get(left_index)
                            .ok_or(AttentionError::InvalidRotaryInput {
                                reason: "rotary left component is missing",
                            })?;
                    let right =
                        *input
                            .get(right_index)
                            .ok_or(AttentionError::InvalidRotaryInput {
                                reason: "rotary right component is missing",
                            })?;
                    let entries = match &table.entries {
                        RotaryTableEntries::GlobalF32(entries)
                        | RotaryTableEntries::ResetPerAxisF64ToF32(entries) => entries,
                    };
                    let [cosine, sine] = *entries.get(token * table_pairs + pair).ok_or(
                        AttentionError::InvalidRotaryInput {
                            reason: "rotary table entry is missing",
                        },
                    )?;
                    let rotated_left = left * cosine - right * sine;
                    let rotated_right = right * cosine + left * sine;
                    *output
                        .get_mut(left_index)
                        .ok_or(AttentionError::InvalidRotaryInput {
                            reason: "rotary output left component is missing",
                        })? = rotated_left;
                    *output
                        .get_mut(right_index)
                        .ok_or(AttentionError::InvalidRotaryInput {
                            reason: "rotary output right component is missing",
                        })? = rotated_right;
                }
            }
        }
    }
    Ok(())
}

pub fn enable_math_sdp_exact_native(
    enabled: bool,
    cancellation: &CancellationToken,
) -> Result<MathSdpSelection, AttentionError> {
    cancellation.check()?;
    Ok(if enabled {
        MathSdpSelection::Enabled(AttentionKernelKind::ReferenceSdp)
    } else {
        MathSdpSelection::Disabled
    })
}

pub fn enable_flash_sdp_exact_native(
    enabled: bool,
    cancellation: &CancellationToken,
) -> Result<MathSdpSelection, AttentionError> {
    cancellation.check()?;
    Ok(if enabled {
        MathSdpSelection::Enabled(AttentionKernelKind::FlashAttention)
    } else {
        MathSdpSelection::Disabled
    })
}

pub fn enable_mem_efficient_sdp_exact_native(
    enabled: bool,
    cancellation: &CancellationToken,
) -> Result<MathSdpSelection, AttentionError> {
    cancellation.check()?;
    Ok(if enabled {
        MathSdpSelection::Enabled(AttentionKernelKind::ReferenceSdp)
    } else {
        MathSdpSelection::Disabled
    })
}

pub fn allow_fp16_bf16_reduction_math_sdp_exact_native(
    enabled: bool,
    cancellation: &CancellationToken,
) -> Result<MathSdpReductionPolicy, AttentionError> {
    cancellation.check()?;
    Ok(MathSdpReductionPolicy {
        allow_fp16_bf16: enabled,
    })
}

pub fn sdpa_kernel_exact_native(
    backends: &[SdpaBackend],
    set_priority: bool,
    cancellation: &CancellationToken,
) -> Result<SdpaKernelSelection, AttentionError> {
    cancellation.check()?;
    if backends.is_empty() || backends.len() > 4 {
        return Err(AttentionError::InvalidSdpaSelection);
    }
    for (index, backend) in backends.iter().enumerate() {
        cancellation.check()?;
        if backends
            .get(..index)
            .is_some_and(|prior| prior.contains(backend))
        {
            return Err(AttentionError::InvalidSdpaSelection);
        }
    }
    cancellation.check()?;
    Ok(SdpaKernelSelection {
        backends: backends.to_vec(),
        set_priority,
    })
}

impl From<sim_types::CancellationError> for AttentionError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn scaled_dot_product_attention_with_context(
    backend: &CpuBackend,
    request: AttentionRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    context: &ExecutionContext<'_>,
) -> Result<AttentionOutcome, AttentionError> {
    let prepared = prepare_attention(request, query, key, value, mask, context.cancellation)?;
    let output = prepared
        .invocation
        .execute_with_context(backend, prepared.query_chunk_size, context)
        .map_err(|error| map_kernel_error(error, request.backend))?;
    Ok(prepared.outcome(output))
}

pub fn scaled_dot_product_attention_tensor_with_context(
    backend: &CpuBackend,
    request: AttentionRequest,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<AttentionMask<'_>>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, AttentionError> {
    context.cancellation.check()?;
    validate_tensor_inputs(request, query, key, value, context)?;

    let query_values = tensor_to_f32_workspace(backend, query, context)?;
    let key_values = tensor_to_f32_workspace(backend, key, context)?;
    let value_values = tensor_to_f32_workspace(backend, value, context)?;
    let prepared = prepare_attention(
        request,
        &query_values,
        &key_values,
        &value_values,
        mask,
        context.cancellation,
    )?;
    let output_length = checked_product(&[
        request.batch,
        request.query_tokens,
        request.heads,
        request.value_dimension,
    ])?;
    let output_temporary_bytes = checked_bytes::<f32>(output_length)?;
    let output_workspace = backend.reserve_workspace(context, output_temporary_bytes)?;
    let output = prepared
        .invocation
        .execute_with_context(backend, prepared.query_chunk_size, context)
        .map_err(|error| map_kernel_error(error, request.backend))?;
    drop(query_values);
    drop(key_values);
    drop(value_values);

    let dtype = query.descriptor().dtype();
    let encoded_length = output_length
        .checked_mul(
            usize::try_from(dtype.byte_width()).map_err(|_| AttentionError::ShapeOverflow)?,
        )
        .ok_or(AttentionError::ShapeOverflow)?;
    let mut encoded = backend.workspace_vec::<u8>(context, encoded_length)?;
    for value in output {
        context.cancellation.check()?;
        for byte in dtype.encode_scalar(
            Scalar::Float(f64::from(value)),
            "zed.sim-model.scaled-dot-product-attention",
            query.descriptor().device(),
        )? {
            encoded.try_push(byte)?;
        }
    }
    context.cancellation.check()?;
    let output_shape = expected_tensor_shape(
        request.batch,
        request.query_tokens,
        request.heads,
        request.value_dimension,
    )?;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        dtype,
        query.descriptor().device(),
        query.descriptor().stream(),
    )?;
    let tensor = backend.upload_bytes(descriptor, &encoded, context)?.0;
    drop(encoded);
    drop(output_workspace);
    context.cancellation.check()?;
    Ok(tensor)
}

fn validate_tensor_inputs(
    request: AttentionRequest,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), AttentionError> {
    let expected_query = expected_tensor_shape(
        request.batch,
        request.query_tokens,
        request.heads,
        request.head_dimension,
    )?;
    let expected_key = expected_tensor_shape(
        request.batch,
        request.key_tokens,
        request.heads,
        request.head_dimension,
    )?;
    let expected_value = expected_tensor_shape(
        request.batch,
        request.key_tokens,
        request.heads,
        request.value_dimension,
    )?;
    for (name, tensor, expected) in [
        ("query", query, expected_query),
        ("key", key, expected_key),
        ("value", value, expected_value),
    ] {
        if tensor.descriptor().shape() != expected {
            return Err(AttentionError::TensorShape {
                name,
                expected,
                actual: tensor.descriptor().shape().to_vec(),
            });
        }
        if tensor.descriptor().device() != DeviceId::CPU {
            return Err(TensorError::DeviceMismatch {
                expected: DeviceId::CPU,
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
    }
    let dtype = query.descriptor().dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(AttentionError::UnsupportedTensorDType { dtype });
    }
    for tensor in [key, value] {
        if tensor.descriptor().dtype() != dtype {
            return Err(TensorError::DTypeMismatch {
                expected: dtype,
                actual: tensor.descriptor().dtype(),
            }
            .into());
        }
        if tensor.descriptor().device() != query.descriptor().device() {
            return Err(TensorError::DeviceMismatch {
                expected: query.descriptor().device(),
                actual: tensor.descriptor().device(),
            }
            .into());
        }
        if tensor.descriptor().stream() != query.descriptor().stream() {
            return Err(TensorError::StreamMismatch {
                expected: query.descriptor().stream(),
                actual: tensor.descriptor().stream(),
            }
            .into());
        }
    }
    Ok(())
}

fn expected_tensor_shape(
    batch: usize,
    tokens: usize,
    heads: usize,
    dimension: usize,
) -> Result<Vec<u64>, AttentionError> {
    [batch, tokens, heads, dimension]
        .into_iter()
        .map(|value| u64::try_from(value).map_err(|_| AttentionError::ShapeOverflow))
        .collect()
}

fn tensor_to_f32_workspace(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, AttentionError> {
    let length = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| AttentionError::ShapeOverflow)?;
    let mut values = backend.workspace_vec(context, length)?;
    for index in 0..length {
        context.cancellation.check()?;
        let index = u64::try_from(index).map_err(|_| AttentionError::ShapeOverflow)?;
        let value = match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?
        {
            DecodedScalar::Real(value) => value as f32,
            _ => {
                return Err(AttentionError::UnsupportedTensorDType {
                    dtype: tensor.descriptor().dtype(),
                });
            }
        };
        values.try_push(value)?;
    }
    Ok(values)
}

fn checked_product(values: &[usize]) -> Result<usize, AttentionError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(AttentionError::ShapeOverflow)
    })
}

fn checked_bytes<T>(elements: usize) -> Result<u64, AttentionError> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| {
            u64::try_from(std::mem::size_of::<T>())
                .ok()
                .and_then(|width| elements.checked_mul(width))
        })
        .ok_or(AttentionError::ShapeOverflow)
}

struct PreparedAttention<'a> {
    invocation: CheckedAttentionInvocation<'a>,
    requested_backend: AttentionBackend,
    effective_backend: AttentionBackend,
    fallback_reason: Option<String>,
    query_chunk_size: usize,
    peak_workspace_bytes: usize,
}

impl PreparedAttention<'_> {
    fn outcome(self, values: Vec<f32>) -> AttentionOutcome {
        AttentionOutcome {
            values,
            requested_backend: self.requested_backend,
            effective_backend: self.effective_backend,
            fallback_reason: self.fallback_reason,
            query_chunk_size: self.query_chunk_size,
            peak_workspace_bytes: self.peak_workspace_bytes,
        }
    }
}

fn prepare_attention<'a>(
    request: AttentionRequest,
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    mask: Option<AttentionMask<'a>>,
    cancellation: &CancellationToken,
) -> Result<PreparedAttention<'a>, AttentionError> {
    let tensor_mask = mask.map(map_mask);
    let invocation = CheckedAttentionInvocation::new(
        AttentionKernelRequest {
            kind: AttentionKernelKind::ReferenceSdp,
            device: DeviceId::CPU,
            layout: TensorAttentionLayout::Nhd,
            shape: AttentionShape {
                batch: request.batch,
                query_tokens: request.query_tokens,
                key_tokens: request.key_tokens,
                heads: request.heads,
                head_dimension: request.head_dimension,
                value_dimension: request.value_dimension,
            },
            scale: request.scale,
            causal: false,
            dropout_probability: 0.0,
        },
        query,
        key,
        value,
        tensor_mask,
    )
    .map_err(|error| map_kernel_error(error, request.backend))?;
    cancellation.check()?;
    let (effective_backend, fallback_reason) = select_backend(request.backend, request.fallback)?;
    let score_row_bytes = invocation
        .score_row_bytes()
        .map_err(|error| map_kernel_error(error, request.backend))?;
    if request.workspace_limit_bytes < score_row_bytes {
        return Err(AttentionError::WorkspaceTooSmall {
            required: score_row_bytes,
            available: request.workspace_limit_bytes,
        });
    }
    let maximum_rows = request.workspace_limit_bytes / score_row_bytes;
    let query_chunk_size = if effective_backend == AttentionBackend::SplitOrSubQuadratic {
        maximum_rows.max(1).min(request.query_tokens)
    } else {
        1
    };
    Ok(PreparedAttention {
        invocation,
        requested_backend: request.backend,
        effective_backend,
        fallback_reason,
        query_chunk_size,
        peak_workspace_bytes: score_row_bytes,
    })
}

fn select_backend(
    backend: AttentionBackend,
    fallback: AttentionFallbackPolicy,
) -> Result<(AttentionBackend, Option<String>), AttentionError> {
    match backend {
        AttentionBackend::PytorchSdp | AttentionBackend::SplitOrSubQuadratic => Ok((backend, None)),
        AttentionBackend::SageOrFlash | AttentionBackend::Xformers
            if fallback == AttentionFallbackPolicy::AllowExactNative =>
        {
            Ok((
                AttentionBackend::PytorchSdp,
                Some(format!(
                    "{} has no certified native kernel on this device; used exact native SDP",
                    backend.feature_id()
                )),
            ))
        }
        AttentionBackend::SageOrFlash | AttentionBackend::Xformers => {
            Err(AttentionError::UnsupportedBackend {
                backend,
                reason: "no certified native kernel is registered for this device".to_owned(),
            })
        }
    }
}

fn map_mask(mask: AttentionMask<'_>) -> TensorAttentionMask<'_> {
    match mask {
        AttentionMask::Boolean { values, shape } => TensorAttentionMask::Boolean {
            values,
            shape: map_mask_shape(shape),
        },
        AttentionMask::Additive { values, shape } => TensorAttentionMask::Additive {
            values,
            shape: map_mask_shape(shape),
        },
        AttentionMask::OrderedAdditive {
            first_values,
            second_values,
            shape,
        } => TensorAttentionMask::OrderedAdditive {
            first_values,
            second_values,
            shape: map_mask_shape(shape),
        },
    }
}

const fn map_mask_shape(shape: AttentionMaskShape) -> TensorAttentionMaskShape {
    match shape {
        AttentionMaskShape::KeyTokens => TensorAttentionMaskShape::KeyTokens,
        AttentionMaskShape::QueryByKey => TensorAttentionMaskShape::QueryByKey,
        AttentionMaskShape::BatchQueryByKey => TensorAttentionMaskShape::BatchQueryByKey,
        AttentionMaskShape::BatchHeadQueryByKey => TensorAttentionMaskShape::BatchHeadQueryByKey,
    }
}

fn map_kernel_error(error: AttentionKernelError, backend: AttentionBackend) -> AttentionError {
    match error {
        AttentionKernelError::Tensor(error) => AttentionError::Tensor(error),
        AttentionKernelError::ShapeOverflow => AttentionError::ShapeOverflow,
        AttentionKernelError::AllocationFailed { name } => {
            AttentionError::AllocationFailed { name }
        }
        AttentionKernelError::EmptyDimension { name } => AttentionError::EmptyDimension { name },
        AttentionKernelError::ValueCount {
            name,
            expected,
            actual,
        } => AttentionError::ValueCount {
            name,
            expected,
            actual,
        },
        AttentionKernelError::MaskValueCount { expected, actual } => {
            AttentionError::MaskValueCount { expected, actual }
        }
        AttentionKernelError::NonFiniteOrderedMask { term, index } => {
            AttentionError::NonFiniteOrderedMask { term, index }
        }
        AttentionKernelError::InvalidScale => AttentionError::InvalidScale,
        AttentionKernelError::Cancelled => AttentionError::Cancelled,
        AttentionKernelError::UnsupportedDropout
        | AttentionKernelError::UnsupportedLayout { .. }
        | AttentionKernelError::UnsupportedMask { .. }
        | AttentionKernelError::UnsupportedDevice { .. }
        | AttentionKernelError::GradientValueCount { .. } => AttentionError::UnsupportedBackend {
            backend,
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(
        backend: &CpuBackend,
        dtype: DType,
        shape: Vec<u64>,
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&dtype.encode_scalar(
                Scalar::Float(f64::from(*value)),
                "zed.sim-model.attention-test",
                DeviceId::CPU,
            )?);
        }
        let descriptor = TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, context.stream)?;
        Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
    }

    fn real_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let length = tensor.descriptor().element_count()?;
        let mut values = Vec::new();
        for index in 0..length {
            match tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.linear_element_bytes(index)?)?
            {
                DecodedScalar::Real(value) => values.push(value as f32),
                value => return Err(format!("expected real tensor value, got {value:?}").into()),
            }
        }
        Ok(values)
    }

    fn request(backend: AttentionBackend) -> AttentionRequest {
        AttentionRequest {
            backend,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: 1,
            query_tokens: 2,
            key_tokens: 2,
            heads: 1,
            head_dimension: 2,
            value_dimension: 2,
            scale: Some(1.0),
            workspace_limit_bytes: 8,
        }
    }

    #[test]
    fn ordered_additive_mask_is_mirrored_without_precombining()
    -> Result<(), Box<dyn std::error::Error>> {
        let request = AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: 1,
            query_tokens: 1,
            key_tokens: 2,
            heads: 1,
            head_dimension: 1,
            value_dimension: 1,
            scale: Some(1.0),
            workspace_limit_bytes: 8,
        };
        let query = [1.0e10_f32];
        let key = [1.0e10_f32, 0.0];
        let value = [0.0_f32, 2.0];
        let first = [-1.0e20_f32, 0.0];
        let second = [-100.0_f32, 0.0];
        let token = CancellationToken::default();
        let (backend, workspace_authority) = sim_tensor::CpuWorkspaceAuthority::create_backend(64)?;
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(8)?,
            &token,
        );
        let output = scaled_dot_product_attention_with_context(
            &backend,
            request,
            &query,
            &key,
            &value,
            Some(AttentionMask::OrderedAdditive {
                first_values: &first,
                second_values: &second,
                shape: AttentionMaskShape::KeyTokens,
            }),
            &context,
        )?;
        assert_eq!(output.values, vec![2.0]);

        let non_finite = [f32::NAN, 0.0];
        assert!(matches!(
            scaled_dot_product_attention_with_context(
                &backend,
                request,
                &query,
                &key,
                &value,
                Some(AttentionMask::OrderedAdditive {
                    first_values: &non_finite,
                    second_values: &second,
                    shape: AttentionMaskShape::KeyTokens,
                }),
                &context,
            ),
            Err(AttentionError::NonFiniteOrderedMask {
                term: "first",
                index: 0,
            })
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn exact_backends_preserve_masks_and_explicit_fallback()
    -> Result<(), Box<dyn std::error::Error>> {
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = query;
        let value = [2.0, 0.0, 0.0, 4.0];
        let mask_values = [true, false, true, true];
        let mask = Some(AttentionMask::Boolean {
            values: &mask_values,
            shape: AttentionMaskShape::QueryByKey,
        });
        let token = CancellationToken::default();
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(8)?,
            &token,
        );
        let exact = scaled_dot_product_attention_with_context(
            &backend,
            request(AttentionBackend::PytorchSdp),
            &query,
            &key,
            &value,
            mask,
            &context,
        )
        .expect("native SDP succeeds");
        assert_eq!(&exact.values[..2], &[2.0, 0.0]);
        let split = scaled_dot_product_attention_with_context(
            &backend,
            request(AttentionBackend::SplitOrSubQuadratic),
            &query,
            &key,
            &value,
            mask,
            &context,
        )
        .expect("split attention succeeds");
        assert_eq!(split.values, exact.values);
        let fallback = scaled_dot_product_attention_with_context(
            &backend,
            request(AttentionBackend::Xformers),
            &query,
            &key,
            &value,
            mask,
            &context,
        )
        .expect("explicit exact fallback succeeds");
        assert_eq!(fallback.effective_backend, AttentionBackend::PytorchSdp);
        assert!(fallback.fallback_reason.is_some());
        Ok(())
    }

    #[test]
    fn model_attention_adapter_preserves_exact_workspace_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = query;
        let value = [2.0, 0.0, 0.0, 4.0];
        let required = 2 * u64::try_from(std::mem::size_of::<f32>())?;
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(required)?,
            &cancellation,
        );
        let outcome = scaled_dot_product_attention_with_context(
            &backend,
            request(AttentionBackend::PytorchSdp),
            &query,
            &key,
            &value,
            None,
            &context,
        )?;
        assert_eq!(outcome.values.len(), 4);
        assert_eq!(context.scratch.peak_bytes(), required);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let insufficient = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(required - 1)?,
            &cancellation,
        );
        assert!(
            scaled_dot_product_attention_with_context(
                &backend,
                request(AttentionBackend::PytorchSdp),
                &query,
                &key,
                &value,
                None,
                &insufficient,
            )
            .is_err()
        );
        assert_eq!(insufficient.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(required)?,
            &cancelled,
        );
        assert!(
            scaled_dot_product_attention_with_context(
                &backend,
                request(AttentionBackend::PytorchSdp),
                &query,
                &key,
                &value,
                None,
                &cancelled_context,
            )
            .is_err()
        );
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn tensor_attention_supports_self_and_cross_attention_with_explicit_scale()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(256)?,
            &cancellation,
        );
        let query = tensor(
            &backend,
            DType::F32,
            vec![1, 2, 1, 2],
            &[1.0, 0.0, 0.0, 1.0],
            &context,
        )?;
        let key = tensor(
            &backend,
            DType::F32,
            vec![1, 3, 1, 2],
            &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            &context,
        )?;
        let value = tensor(
            &backend,
            DType::F32,
            vec![1, 3, 1, 2],
            &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            &context,
        )?;
        let input_storage = [query.storage_id(), key.storage_id(), value.storage_id()];
        let cross = scaled_dot_product_attention_tensor_with_context(
            &backend,
            AttentionRequest {
                backend: AttentionBackend::SplitOrSubQuadratic,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch: 1,
                query_tokens: 2,
                key_tokens: 3,
                heads: 1,
                head_dimension: 2,
                value_dimension: 2,
                scale: Some(0.5),
                workspace_limit_bytes: 12,
            },
            &query,
            &key,
            &value,
            None,
            &context,
        )?;
        assert_eq!(cross.descriptor().shape(), &[1, 2, 1, 2]);
        assert_eq!(cross.descriptor().dtype(), DType::F32);
        assert_eq!(cross.descriptor().device(), DeviceId::CPU);
        assert_eq!(cross.descriptor().stream(), context.stream);
        assert!(!input_storage.contains(&cross.storage_id()));
        assert_eq!(
            [query.storage_id(), key.storage_id(), value.storage_id()],
            input_storage
        );
        let cross_values = real_values(&cross)?;
        let first_denominator = 0.5_f32.exp() + 1.0 + (-0.5_f32).exp();
        let expected_first = (0.5_f32.exp() + 3.0 * (-0.5_f32).exp()) / first_denominator;
        assert!(
            cross_values
                .first()
                .is_some_and(|value| (*value - expected_first).abs() < 1.0e-6)
        );
        assert!(
            cross_values
                .get(1)
                .is_some_and(|value| (*value - (2.0 / first_denominator)).abs() < 1.0e-6)
        );

        let self_attention = scaled_dot_product_attention_tensor_with_context(
            &backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch: 1,
                query_tokens: 2,
                key_tokens: 2,
                heads: 1,
                head_dimension: 2,
                value_dimension: 2,
                scale: Some(1.0),
                workspace_limit_bytes: 8,
            },
            &query,
            &query,
            &query,
            None,
            &context,
        )?;
        assert_eq!(self_attention.descriptor().shape(), &[1, 2, 1, 2]);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn tensor_attention_preserves_supported_float_dtypes_and_accounts_temporaries()
    -> Result<(), Box<dyn std::error::Error>> {
        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            let (backend, workspace_authority) =
                sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
            let cancellation = CancellationToken::default();
            let context = backend.execution_context(
                sim_tensor::StreamId::DEFAULT,
                workspace_authority.authorize_workspace(92)?,
                &cancellation,
            );
            let query = tensor(
                &backend,
                dtype,
                vec![1, 2, 1, 2],
                &[1.0, 0.0, 0.0, 1.0],
                &context,
            )?;
            let key = tensor(
                &backend,
                dtype,
                vec![1, 3, 1, 2],
                &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                &context,
            )?;
            let value = tensor(
                &backend,
                dtype,
                vec![1, 3, 1, 2],
                &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
                &context,
            )?;
            let output = scaled_dot_product_attention_tensor_with_context(
                &backend,
                AttentionRequest {
                    backend: AttentionBackend::SplitOrSubQuadratic,
                    fallback: AttentionFallbackPolicy::AllowExactNative,
                    batch: 1,
                    query_tokens: 2,
                    key_tokens: 3,
                    heads: 1,
                    head_dimension: 2,
                    value_dimension: 2,
                    scale: Some(1.0),
                    workspace_limit_bytes: 12,
                },
                &query,
                &key,
                &value,
                None,
                &context,
            )?;
            assert_eq!(output.descriptor().dtype(), dtype);
            assert_eq!(context.scratch.peak_bytes(), 92);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }

    #[test]
    fn tensor_attention_releases_workspace_on_exhaustion_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(91)?,
            &cancellation,
        );
        let query = tensor(
            &backend,
            DType::F32,
            vec![1, 2, 1, 2],
            &[1.0, 0.0, 0.0, 1.0],
            &context,
        )?;
        let key = tensor(
            &backend,
            DType::F32,
            vec![1, 3, 1, 2],
            &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            &context,
        )?;
        let value = tensor(
            &backend,
            DType::F32,
            vec![1, 3, 1, 2],
            &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            &context,
        )?;
        let request = AttentionRequest {
            backend: AttentionBackend::SplitOrSubQuadratic,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: 1,
            query_tokens: 2,
            key_tokens: 3,
            heads: 1,
            head_dimension: 2,
            value_dimension: 2,
            scale: Some(1.0),
            workspace_limit_bytes: 12,
        };
        assert!(matches!(
            scaled_dot_product_attention_tensor_with_context(
                &backend, request, &query, &key, &value, None, &context,
            ),
            Err(AttentionError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(92)?,
            &cancelled,
        );
        assert!(matches!(
            scaled_dot_product_attention_tensor_with_context(
                &backend,
                request,
                &query,
                &key,
                &value,
                None,
                &cancelled_context,
            ),
            Err(AttentionError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    fn scalar_rotary_table(
        positions: &[usize],
        rotary_dimension: usize,
        cancellation: &CancellationToken,
    ) -> Result<RotaryTable, AttentionError> {
        precompute_rotary_table(
            RotaryTableRequest {
                positions: RotaryPositions::Scalar(RotaryPositionSequence::Unsigned(positions)),
                axis_dimensions: &[],
                rotary_dimension,
                theta: 10_000.0,
                scaling: RotaryScaling::None,
                frequency_layout: RotaryFrequencyLayout::Global,
            },
            cancellation,
        )
    }

    #[test]
    fn scalar_split_half_preserves_decoder_bits_and_nonrotary_tail()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let table = scalar_rotary_table(&[0, 1], 4, &cancellation)?;
        let values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let output = apply_rotary_table(
            &values,
            1,
            2,
            1,
            6,
            &table,
            RotaryPairLayout::SplitHalf,
            &cancellation,
        )?;
        assert_eq!(&output[..6], &values[..6]);
        assert_eq!(output[6].to_bits(), 0xc072_a1c0);
        assert_eq!(output[7].to_bits(), 0x40fc_c989);
        assert_eq!(output[8].to_bits(), 0x412c_0c5c);
        assert_eq!(output[9].to_bits(), 0x4121_45a1);
        assert_eq!(&output[10..], &values[10..]);
        Ok(())
    }

    #[test]
    fn qwen_multiaxis_adjacent_oracle_resets_f64_frequencies_per_axis()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let temporal = [-3.0_f32];
        let row = [5.0_f32];
        let column = [-7.0_f32];
        let axes = [
            RotaryPositionSequence::Float(&temporal),
            RotaryPositionSequence::Float(&row),
            RotaryPositionSequence::Float(&column),
        ];
        let table = precompute_rotary_table(
            RotaryTableRequest {
                positions: RotaryPositions::Multiaxis(&axes),
                axis_dimensions: &[16, 56, 56],
                rotary_dimension: 128,
                theta: 10_000.0,
                scaling: RotaryScaling::None,
                frequency_layout: RotaryFrequencyLayout::ResetPerAxis,
            },
            &cancellation,
        )?;
        let entries = match &table.entries {
            RotaryTableEntries::ResetPerAxisF64ToF32(entries) => entries,
            RotaryTableEntries::GlobalF32(_) => return Err("expected per-axis table".into()),
        };
        for (pair, expected) in [
            (3, [0x3f7e_d94f, 0xbdc1_ffc1]),
            (21, [0x3f7f_61e7, 0x3d8e_2b7f]),
            (53, [0x3f7f_e9b1, 0xbcd5_bb1b]),
        ] {
            let entry = entries.get(pair).ok_or("missing Qwen oracle entry")?;
            assert_eq!([entry[0].to_bits(), entry[1].to_bits()], expected);
        }

        let values = (0..128)
            .map(|index| (index as f32 - 31.0) * 0.125)
            .collect::<Vec<_>>();
        let output = apply_rotary_table(
            &values,
            1,
            1,
            1,
            128,
            &table,
            RotaryPairLayout::Adjacent,
            &cancellation,
        )?;
        assert_eq!(output[6].to_bits(), 0xc059_49c0);
        assert_eq!(output[7].to_bits(), 0xc02c_3101);
        assert_eq!(output[42].to_bits(), 0x3fa2_3f3b);
        assert_eq!(output[43].to_bits(), 0x3fcb_c12a);
        assert_eq!(output[106].to_bits(), 0x4119_ea27);
        assert_eq!(output[107].to_bits(), 0x4114_08e4);
        let f64_multiply_then_cast = (f64::from(values[106]) * f64::from(entries[53][0])
            - f64::from(values[107]) * f64::from(entries[53][1]))
            as f32;
        assert_ne!(output[106].to_bits(), f64_multiply_then_cast.to_bits());
        Ok(())
    }

    #[test]
    fn rotary_validation_rejects_invalid_geometry_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        assert!(matches!(
            scalar_rotary_table(&[0], 3, &cancellation),
            Err(AttentionError::InvalidRotaryConfiguration { .. })
        ));

        let first = [0.0_f32, 1.0];
        let second = [0.0_f32];
        let axes = [
            RotaryPositionSequence::Float(&first),
            RotaryPositionSequence::Float(&second),
        ];
        assert!(matches!(
            precompute_rotary_table(
                RotaryTableRequest {
                    positions: RotaryPositions::Multiaxis(&axes),
                    axis_dimensions: &[2, 2],
                    rotary_dimension: 4,
                    theta: 10_000.0,
                    scaling: RotaryScaling::None,
                    frequency_layout: RotaryFrequencyLayout::ResetPerAxis,
                },
                &cancellation,
            ),
            Err(AttentionError::InvalidRotaryInput { .. })
        ));
        assert!(matches!(
            precompute_rotary_table(
                RotaryTableRequest {
                    positions: RotaryPositions::Multiaxis(&axes),
                    axis_dimensions: &[2, 4],
                    rotary_dimension: 4,
                    theta: 10_000.0,
                    scaling: RotaryScaling::None,
                    frequency_layout: RotaryFrequencyLayout::ResetPerAxis,
                },
                &cancellation,
            ),
            Err(AttentionError::InvalidRotaryConfiguration { .. })
        ));
        let table = scalar_rotary_table(&[0], 4, &cancellation)?;
        assert!(matches!(
            apply_rotary_table(
                &[],
                usize::MAX,
                1,
                2,
                4,
                &table,
                RotaryPairLayout::SplitHalf,
                &cancellation,
            ),
            Err(AttentionError::ShapeOverflow)
        ));
        assert!(matches!(
            RotaryTableEntries::with_capacity(RotaryFrequencyLayout::Global, usize::MAX),
            Err(AttentionError::AllocationFailed {
                name: "rotary table"
            })
        ));
        assert!(matches!(
            apply_rotary_table(
                &[],
                usize::MAX,
                1,
                0,
                4,
                &table,
                RotaryPairLayout::SplitHalf,
                &cancellation,
            ),
            Err(AttentionError::EmptyDimension { name: "heads" })
        ));
        assert!(matches!(
            apply_rotary_table(
                &[0.0; 4],
                1,
                2,
                1,
                4,
                &table,
                RotaryPairLayout::SplitHalf,
                &cancellation,
            ),
            Err(AttentionError::InvalidRotaryInput { .. })
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            scalar_rotary_table(&[0], 4, &cancelled),
            Err(AttentionError::Cancelled)
        ));
        assert!(matches!(
            apply_rotary_table(
                &[1.0; 4],
                1,
                1,
                1,
                4,
                &table,
                RotaryPairLayout::SplitHalf,
                &cancelled,
            ),
            Err(AttentionError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn rotary_tensor_surface_checks_target_stream_and_workspace_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let setup_context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024)?,
            &cancellation,
        );
        let table = scalar_rotary_table(&[1], 4, &cancellation)?;
        let input = tensor(
            &backend,
            DType::F32,
            vec![1, 1, 1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &setup_context,
        )?;
        let input_values = real_values(&input)?;
        let input_storage = input.storage_id();

        let exact_context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(48)?,
            &cancellation,
        );
        let output = apply_rotary_table_tensor_with_context(
            &backend,
            &input,
            &table,
            RotaryPairLayout::SplitHalf,
            &exact_context,
        )?;
        assert_eq!(output.descriptor(), input.descriptor());
        assert_ne!(output.storage_id(), input_storage);
        assert_eq!(exact_context.scratch.peak_bytes(), 48);
        assert_eq!(exact_context.scratch.in_use_bytes(), 0);

        let insufficient = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(31)?,
            &cancellation,
        );
        assert!(matches!(
            apply_rotary_table_tensor_with_context(
                &backend,
                &input,
                &table,
                RotaryPairLayout::SplitHalf,
                &insufficient,
            ),
            Err(AttentionError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient.scratch.in_use_bytes(), 0);
        assert_eq!(input.storage_id(), input_storage);
        assert_eq!(real_values(&input)?, input_values);

        let invalid_dtype = tensor(
            &backend,
            DType::I64,
            vec![1, 1, 1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &setup_context,
        )?;
        assert!(matches!(
            apply_rotary_table_tensor_with_context(
                &backend,
                &invalid_dtype,
                &table,
                RotaryPairLayout::SplitHalf,
                &setup_context,
            ),
            Err(AttentionError::UnsupportedTensorDType { dtype: DType::I64 })
        ));

        let metal = TensorDescriptor::contiguous(
            vec![1, 1, 1, 4],
            DType::F32,
            DeviceId::from_source_device("metal")?,
            sim_tensor::StreamId::DEFAULT,
        )?;
        assert!(matches!(
            validate_rotary_tensor_descriptor(&metal, &setup_context),
            Err(AttentionError::Tensor(TensorError::DeviceMismatch { .. }))
        ));
        let wrong_stream = TensorDescriptor::contiguous(
            vec![1, 1, 1, 4],
            DType::F32,
            DeviceId::CPU,
            sim_tensor::StreamId::new(7),
        )?;
        assert!(matches!(
            validate_rotary_tensor_descriptor(&wrong_stream, &setup_context),
            Err(AttentionError::Tensor(TensorError::StreamMismatch { .. }))
        ));
        Ok(())
    }
}
