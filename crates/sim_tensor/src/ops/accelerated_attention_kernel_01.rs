use crate::{CpuBackend, CpuWorkspaceVec, DeviceId, ExecutionContext, TensorError};
use thiserror::Error;

pub const FLASH_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-B8796E1BECDE";
pub const SAGE_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-1354AC34A777";
pub const SAGE_ATTENTION_3_OPERATION_ID: &str = "COMFY-TENSOR-OP-5E24E0493F83";
pub const XFORMERS_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-9973AD8D6DCC";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionLayout {
    Nhd,
    Hnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKernelKind {
    ReferenceSdp,
    FlashAttention,
    SageAttention,
    SageAttention3Blackwell,
    XformersMemoryEfficient,
}

impl AttentionKernelKind {
    pub const fn operation_id(self) -> Option<&'static str> {
        match self {
            Self::ReferenceSdp => None,
            Self::FlashAttention => Some(FLASH_ATTENTION_OPERATION_ID),
            Self::SageAttention => Some(SAGE_ATTENTION_OPERATION_ID),
            Self::SageAttention3Blackwell => Some(SAGE_ATTENTION_3_OPERATION_ID),
            Self::XformersMemoryEfficient => Some(XFORMERS_ATTENTION_OPERATION_ID),
        }
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
pub struct AttentionShape {
    pub batch: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub heads: usize,
    pub head_dimension: usize,
    pub value_dimension: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionKernelRequest {
    pub kind: AttentionKernelKind,
    pub device: DeviceId,
    pub layout: AttentionLayout,
    pub shape: AttentionShape,
    pub scale: Option<f32>,
    pub causal: bool,
    pub dropout_probability: f32,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum AttentionKernelError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("attention dimensions overflowed")]
    ShapeOverflow,
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
    #[error(
        "attention dropout probability must be finite and zero for deterministic native execution"
    )]
    UnsupportedDropout,
    #[error("attention kernel {kind:?} requires layout {required:?}, got {actual:?}")]
    UnsupportedLayout {
        kind: AttentionKernelKind,
        required: AttentionLayout,
        actual: AttentionLayout,
    },
    #[error("attention kernel {kind:?} does not accept an attention mask")]
    UnsupportedMask { kind: AttentionKernelKind },
    #[error("attention kernel {kind:?} has no certified adapter for device {device:?}")]
    UnsupportedDevice {
        kind: AttentionKernelKind,
        device: DeviceId,
    },
    #[error("attention gradient expected {expected} values, got {actual}")]
    GradientValueCount { expected: usize, actual: usize },
    #[error("attention allocation for {name} failed")]
    AllocationFailed { name: &'static str },
    #[error("attention was cancelled")]
    Cancelled,
}

impl From<sim_types::CancellationError> for AttentionKernelError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionVjp {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct CheckedAttentionInvocation<'a> {
    request: AttentionKernelRequest,
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    mask: Option<AttentionMask<'a>>,
    scale: f32,
}

impl<'a> CheckedAttentionInvocation<'a> {
    pub fn new(
        request: AttentionKernelRequest,
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        mask: Option<AttentionMask<'a>>,
    ) -> Result<Self, AttentionKernelError> {
        validate_request(request, query, key, value, mask)?;
        let scale = request
            .scale
            .unwrap_or_else(|| 1.0 / (request.shape.head_dimension as f32).sqrt());
        Ok(Self {
            request,
            query,
            key,
            value,
            mask,
            scale,
        })
    }

    pub const fn query_tokens(self) -> usize {
        self.request.shape.query_tokens
    }

    pub fn score_row_bytes(self) -> Result<usize, AttentionKernelError> {
        self.request
            .shape
            .key_tokens
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(AttentionKernelError::ShapeOverflow)
    }

    fn validate_ordered_mask_finite(
        self,
        context: &ExecutionContext<'_>,
    ) -> Result<(), AttentionKernelError> {
        let Some(AttentionMask::OrderedAdditive {
            first_values,
            second_values,
            ..
        }) = self.mask
        else {
            return Ok(());
        };
        for (term, values) in [("first", first_values), ("second", second_values)] {
            for (index, value) in values.iter().enumerate() {
                if index.is_multiple_of(1_024) {
                    context.cancellation.check()?;
                }
                if !value.is_finite() {
                    return Err(AttentionKernelError::NonFiniteOrderedMask { term, index });
                }
            }
        }
        context.cancellation.check()?;
        Ok(())
    }

    pub fn execute_with_context(
        self,
        backend: &CpuBackend,
        query_chunk_size: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, AttentionKernelError> {
        self.execute_impl(backend, query_chunk_size, context)
    }

    fn execute_impl(
        self,
        backend: &CpuBackend,
        query_chunk_size: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, AttentionKernelError> {
        if query_chunk_size == 0 {
            return Err(AttentionKernelError::EmptyDimension {
                name: "query_chunk_size",
            });
        }
        context.cancellation.check()?;
        self.validate_ordered_mask_finite(context)?;
        let shape = self.request.shape;
        let output_length = checked_product(&[
            shape.batch,
            shape.query_tokens,
            shape.heads,
            shape.value_dimension,
        ])?;
        let mut output = zeroed(output_length, "attention output")?;
        let mut probabilities = temporary_zeroed(backend, context, shape.key_tokens)?;
        for batch in 0..shape.batch {
            for head in 0..shape.heads {
                for query_start in (0..shape.query_tokens).step_by(query_chunk_size) {
                    let query_end = query_start
                        .saturating_add(query_chunk_size)
                        .min(shape.query_tokens);
                    for query_token in query_start..query_end {
                        context.cancellation.check()?;
                        self.probabilities_for_row(batch, head, query_token, &mut probabilities)?;
                        for value_component in 0..shape.value_dimension {
                            let mut result = 0.0_f32;
                            for (key_token, probability) in probabilities.iter().enumerate() {
                                let value_index = tensor_index(
                                    self.request.layout,
                                    shape.batch,
                                    shape.key_tokens,
                                    shape.heads,
                                    shape.value_dimension,
                                    batch,
                                    key_token,
                                    head,
                                    value_component,
                                );
                                result +=
                                    probability * read_value(self.value, value_index, "value")?;
                            }
                            let output_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.query_tokens,
                                shape.heads,
                                shape.value_dimension,
                                batch,
                                query_token,
                                head,
                                value_component,
                            );
                            write_value(&mut output, output_index, result, "output")?;
                        }
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn vjp_with_context(
        self,
        backend: &CpuBackend,
        output_gradient: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<AttentionVjp, AttentionKernelError> {
        self.vjp_impl(backend, output_gradient, context)
    }

    fn vjp_impl(
        self,
        backend: &CpuBackend,
        output_gradient: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<AttentionVjp, AttentionKernelError> {
        let shape = self.request.shape;
        let output_length = checked_product(&[
            shape.batch,
            shape.query_tokens,
            shape.heads,
            shape.value_dimension,
        ])?;
        if output_gradient.len() != output_length {
            return Err(AttentionKernelError::GradientValueCount {
                expected: output_length,
                actual: output_gradient.len(),
            });
        }
        context.cancellation.check()?;
        self.validate_ordered_mask_finite(context)?;
        let mut query_gradient = zeroed(self.query.len(), "query gradient")?;
        let mut key_gradient = zeroed(self.key.len(), "key gradient")?;
        let mut value_gradient = zeroed(self.value.len(), "value gradient")?;
        let mut probabilities = temporary_zeroed(backend, context, shape.key_tokens)?;
        let mut probability_gradient = temporary_zeroed(backend, context, shape.key_tokens)?;
        for batch in 0..shape.batch {
            for head in 0..shape.heads {
                for query_token in 0..shape.query_tokens {
                    context.cancellation.check()?;
                    self.probabilities_for_row(batch, head, query_token, &mut probabilities)?;
                    for key_token in 0..shape.key_tokens {
                        let mut gradient = 0.0_f32;
                        for component in 0..shape.value_dimension {
                            let output_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.query_tokens,
                                shape.heads,
                                shape.value_dimension,
                                batch,
                                query_token,
                                head,
                                component,
                            );
                            let value_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.key_tokens,
                                shape.heads,
                                shape.value_dimension,
                                batch,
                                key_token,
                                head,
                                component,
                            );
                            let output_gradient =
                                read_value(output_gradient, output_index, "output gradient")?;
                            gradient +=
                                output_gradient * read_value(self.value, value_index, "value")?;
                            add_value(
                                &mut value_gradient,
                                value_index,
                                read_value(&probabilities, key_token, "probability")?
                                    * output_gradient,
                                "value gradient",
                            )?;
                        }
                        write_value(
                            &mut probability_gradient,
                            key_token,
                            gradient,
                            "probability gradient",
                        )?;
                    }
                    let centered = probabilities
                        .iter()
                        .zip(probability_gradient.iter())
                        .map(|(probability, gradient)| probability * gradient)
                        .sum::<f32>();
                    for key_token in 0..shape.key_tokens {
                        let score_gradient = read_value(&probabilities, key_token, "probability")?
                            * (read_value(
                                &probability_gradient,
                                key_token,
                                "probability gradient",
                            )? - centered);
                        for component in 0..shape.head_dimension {
                            let query_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.query_tokens,
                                shape.heads,
                                shape.head_dimension,
                                batch,
                                query_token,
                                head,
                                component,
                            );
                            let key_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.key_tokens,
                                shape.heads,
                                shape.head_dimension,
                                batch,
                                key_token,
                                head,
                                component,
                            );
                            add_value(
                                &mut query_gradient,
                                query_index,
                                self.scale
                                    * score_gradient
                                    * read_value(self.key, key_index, "key")?,
                                "query gradient",
                            )?;
                            add_value(
                                &mut key_gradient,
                                key_index,
                                self.scale
                                    * score_gradient
                                    * read_value(self.query, query_index, "query")?,
                                "key gradient",
                            )?;
                        }
                    }
                }
            }
        }
        Ok(AttentionVjp {
            query: query_gradient,
            key: key_gradient,
            value: value_gradient,
        })
    }

    pub fn jvp_with_context(
        self,
        backend: &CpuBackend,
        query_tangent: &[f32],
        key_tangent: &[f32],
        value_tangent: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, AttentionKernelError> {
        self.jvp_impl(backend, query_tangent, key_tangent, value_tangent, context)
    }

    fn jvp_impl(
        self,
        backend: &CpuBackend,
        query_tangent: &[f32],
        key_tangent: &[f32],
        value_tangent: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, AttentionKernelError> {
        for (expected, actual) in [
            (self.query.len(), query_tangent.len()),
            (self.key.len(), key_tangent.len()),
            (self.value.len(), value_tangent.len()),
        ] {
            if expected != actual {
                return Err(AttentionKernelError::GradientValueCount { expected, actual });
            }
        }
        context.cancellation.check()?;
        self.validate_ordered_mask_finite(context)?;
        let shape = self.request.shape;
        let output_length = checked_product(&[
            shape.batch,
            shape.query_tokens,
            shape.heads,
            shape.value_dimension,
        ])?;
        let mut output_tangent = zeroed(output_length, "output tangent")?;
        let mut probabilities = temporary_zeroed(backend, context, shape.key_tokens)?;
        let mut score_tangent = temporary_zeroed(backend, context, shape.key_tokens)?;
        for batch in 0..shape.batch {
            for head in 0..shape.heads {
                for query_token in 0..shape.query_tokens {
                    context.cancellation.check()?;
                    self.probabilities_for_row(batch, head, query_token, &mut probabilities)?;
                    for key_token in 0..shape.key_tokens {
                        let mut tangent = 0.0_f32;
                        for component in 0..shape.head_dimension {
                            let query_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.query_tokens,
                                shape.heads,
                                shape.head_dimension,
                                batch,
                                query_token,
                                head,
                                component,
                            );
                            let key_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.key_tokens,
                                shape.heads,
                                shape.head_dimension,
                                batch,
                                key_token,
                                head,
                                component,
                            );
                            tangent += read_value(query_tangent, query_index, "query tangent")?
                                * read_value(self.key, key_index, "key")?
                                + read_value(self.query, query_index, "query")?
                                    * read_value(key_tangent, key_index, "key tangent")?;
                        }
                        write_value(
                            &mut score_tangent,
                            key_token,
                            tangent * self.scale,
                            "score tangent",
                        )?;
                    }
                    let centered = probabilities
                        .iter()
                        .zip(score_tangent.iter())
                        .map(|(probability, tangent)| probability * tangent)
                        .sum::<f32>();
                    for component in 0..shape.value_dimension {
                        let mut tangent = 0.0_f32;
                        for key_token in 0..shape.key_tokens {
                            let value_index = tensor_index(
                                self.request.layout,
                                shape.batch,
                                shape.key_tokens,
                                shape.heads,
                                shape.value_dimension,
                                batch,
                                key_token,
                                head,
                                component,
                            );
                            let probability = read_value(&probabilities, key_token, "probability")?;
                            let probability_tangent = probability
                                * (read_value(&score_tangent, key_token, "score tangent")?
                                    - centered);
                            tangent += probability_tangent
                                * read_value(self.value, value_index, "value")?
                                + probability
                                    * read_value(value_tangent, value_index, "value tangent")?;
                        }
                        let output_index = tensor_index(
                            self.request.layout,
                            shape.batch,
                            shape.query_tokens,
                            shape.heads,
                            shape.value_dimension,
                            batch,
                            query_token,
                            head,
                            component,
                        );
                        write_value(&mut output_tangent, output_index, tangent, "output tangent")?;
                    }
                }
            }
        }
        Ok(output_tangent)
    }

    fn probabilities_for_row(
        self,
        batch: usize,
        head: usize,
        query_token: usize,
        probabilities: &mut [f32],
    ) -> Result<(), AttentionKernelError> {
        let shape = self.request.shape;
        let mut maximum = f32::NEG_INFINITY;
        for key_token in 0..shape.key_tokens {
            let mut score = 0.0_f32;
            for component in 0..shape.head_dimension {
                let query_index = tensor_index(
                    self.request.layout,
                    shape.batch,
                    shape.query_tokens,
                    shape.heads,
                    shape.head_dimension,
                    batch,
                    query_token,
                    head,
                    component,
                );
                let key_index = tensor_index(
                    self.request.layout,
                    shape.batch,
                    shape.key_tokens,
                    shape.heads,
                    shape.head_dimension,
                    batch,
                    key_token,
                    head,
                    component,
                );
                score += read_value(self.query, query_index, "query")?
                    * read_value(self.key, key_index, "key")?;
            }
            score *= self.scale;
            if self.request.causal
                && (key_token as u128 + shape.query_tokens as u128
                    > query_token as u128 + shape.key_tokens as u128)
            {
                score = f32::NEG_INFINITY;
            }
            score = apply_mask(score, self.mask, shape, batch, head, query_token, key_token)?;
            write_value(probabilities, key_token, score, "score row")?;
            maximum = if score.is_nan() {
                f32::NAN
            } else {
                maximum.max(score)
            };
        }
        if maximum == f32::NEG_INFINITY {
            probabilities.fill(0.0);
            return Ok(());
        }
        let mut denominator = 0.0_f32;
        for score in probabilities.iter_mut() {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        if denominator == 0.0 {
            probabilities.fill(0.0);
        } else {
            for probability in probabilities {
                *probability /= denominator;
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_func_with_context_exact_native(
    backend: &CpuBackend,
    shape: AttentionShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    dropout_probability: f32,
    causal: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AttentionKernelError> {
    exact_native_with_context(
        backend,
        AttentionKernelRequest {
            kind: AttentionKernelKind::FlashAttention,
            device,
            layout: AttentionLayout::Nhd,
            shape,
            scale: None,
            causal,
            dropout_probability,
        },
        query,
        key,
        value,
        None,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sageattn_with_context_exact_native(
    backend: &CpuBackend,
    shape: AttentionShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    causal: bool,
    layout: AttentionLayout,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AttentionKernelError> {
    exact_native_with_context(
        backend,
        AttentionKernelRequest {
            kind: AttentionKernelKind::SageAttention,
            device,
            layout,
            shape,
            scale: None,
            causal,
            dropout_probability: 0.0,
        },
        query,
        key,
        value,
        mask,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sageattn3_blackwell_with_context_exact_native(
    backend: &CpuBackend,
    shape: AttentionShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    causal: bool,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AttentionKernelError> {
    exact_native_with_context(
        backend,
        AttentionKernelRequest {
            kind: AttentionKernelKind::SageAttention3Blackwell,
            device,
            layout: AttentionLayout::Hnd,
            shape,
            scale: None,
            causal,
            dropout_probability: 0.0,
        },
        query,
        key,
        value,
        None,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn memory_efficient_attention_with_context_exact_native(
    backend: &CpuBackend,
    shape: AttentionShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    attention_bias: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AttentionKernelError> {
    let mask = attention_bias.map(|values| AttentionMask::Additive {
        values,
        shape: AttentionMaskShape::BatchHeadQueryByKey,
    });
    exact_native_with_context(
        backend,
        AttentionKernelRequest {
            kind: AttentionKernelKind::XformersMemoryEfficient,
            device,
            layout: AttentionLayout::Nhd,
            shape,
            scale: None,
            causal: false,
            dropout_probability: 0.0,
        },
        query,
        key,
        value,
        mask,
        context,
    )
}

fn exact_native_with_context(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, AttentionKernelError> {
    CheckedAttentionInvocation::new(request, query, key, value, mask)?
        .execute_with_context(backend, 1, context)
}

fn validate_request(
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
) -> Result<(), AttentionKernelError> {
    let shape = request.shape;
    for (name, dimension) in [
        ("batch", shape.batch),
        ("query_tokens", shape.query_tokens),
        ("key_tokens", shape.key_tokens),
        ("heads", shape.heads),
        ("head_dimension", shape.head_dimension),
        ("value_dimension", shape.value_dimension),
    ] {
        if dimension == 0 {
            return Err(AttentionKernelError::EmptyDimension { name });
        }
    }
    if request.device != DeviceId::CPU {
        return Err(AttentionKernelError::UnsupportedDevice {
            kind: request.kind,
            device: request.device,
        });
    }
    if request
        .scale
        .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        return Err(AttentionKernelError::InvalidScale);
    }
    if !request.dropout_probability.is_finite() || request.dropout_probability != 0.0 {
        return Err(AttentionKernelError::UnsupportedDropout);
    }
    match request.kind {
        AttentionKernelKind::FlashAttention | AttentionKernelKind::XformersMemoryEfficient
            if request.layout != AttentionLayout::Nhd =>
        {
            return Err(AttentionKernelError::UnsupportedLayout {
                kind: request.kind,
                required: AttentionLayout::Nhd,
                actual: request.layout,
            });
        }
        AttentionKernelKind::SageAttention3Blackwell if request.layout != AttentionLayout::Hnd => {
            return Err(AttentionKernelError::UnsupportedLayout {
                kind: request.kind,
                required: AttentionLayout::Hnd,
                actual: request.layout,
            });
        }
        _ => {}
    }
    if mask.is_some()
        && matches!(
            request.kind,
            AttentionKernelKind::FlashAttention | AttentionKernelKind::SageAttention3Blackwell
        )
    {
        return Err(AttentionKernelError::UnsupportedMask { kind: request.kind });
    }
    let query_expected = checked_product(&[
        shape.batch,
        shape.query_tokens,
        shape.heads,
        shape.head_dimension,
    ])?;
    let key_expected = checked_product(&[
        shape.batch,
        shape.key_tokens,
        shape.heads,
        shape.head_dimension,
    ])?;
    let value_expected = checked_product(&[
        shape.batch,
        shape.key_tokens,
        shape.heads,
        shape.value_dimension,
    ])?;
    for (name, expected, actual) in [
        ("query", query_expected, query.len()),
        ("key", key_expected, key.len()),
        ("value", value_expected, value.len()),
    ] {
        if expected != actual {
            return Err(AttentionKernelError::ValueCount {
                name,
                expected,
                actual,
            });
        }
    }
    if let Some(mask) = mask {
        let (actual, mask_shape) = match mask {
            AttentionMask::Boolean { values, shape } => (values.len(), shape),
            AttentionMask::Additive { values, shape } => (values.len(), shape),
            AttentionMask::OrderedAdditive {
                first_values,
                second_values,
                shape: mask_shape,
            } => {
                let expected = mask_length(shape, mask_shape)?;
                for values in [first_values, second_values] {
                    if values.len() != expected {
                        return Err(AttentionKernelError::MaskValueCount {
                            expected,
                            actual: values.len(),
                        });
                    }
                }
                (expected, mask_shape)
            }
        };
        let expected = mask_length(shape, mask_shape)?;
        if actual != expected {
            return Err(AttentionKernelError::MaskValueCount { expected, actual });
        }
    }
    Ok(())
}

fn apply_mask(
    score: f32,
    mask: Option<AttentionMask<'_>>,
    shape: AttentionShape,
    batch: usize,
    head: usize,
    query_token: usize,
    key_token: usize,
) -> Result<f32, AttentionKernelError> {
    let Some(mask) = mask else {
        return Ok(score);
    };
    let index = mask_index(shape, mask.shape(), batch, head, query_token, key_token);
    match mask {
        AttentionMask::Boolean { values, .. } => {
            if values
                .get(index)
                .copied()
                .ok_or(AttentionKernelError::MaskValueCount {
                    expected: index.saturating_add(1),
                    actual: values.len(),
                })?
            {
                Ok(score)
            } else {
                Ok(f32::NEG_INFINITY)
            }
        }
        AttentionMask::Additive { values, .. } => Ok(score
            + values
                .get(index)
                .copied()
                .ok_or(AttentionKernelError::MaskValueCount {
                    expected: index.saturating_add(1),
                    actual: values.len(),
                })?),
        AttentionMask::OrderedAdditive {
            first_values,
            second_values,
            ..
        } => {
            let first =
                first_values
                    .get(index)
                    .copied()
                    .ok_or(AttentionKernelError::MaskValueCount {
                        expected: index.saturating_add(1),
                        actual: first_values.len(),
                    })?;
            let second =
                second_values
                    .get(index)
                    .copied()
                    .ok_or(AttentionKernelError::MaskValueCount {
                        expected: index.saturating_add(1),
                        actual: second_values.len(),
                    })?;
            Ok((score + first) + second)
        }
    }
}

impl AttentionMask<'_> {
    const fn shape(self) -> AttentionMaskShape {
        match self {
            Self::Boolean { shape, .. }
            | Self::Additive { shape, .. }
            | Self::OrderedAdditive { shape, .. } => shape,
        }
    }
}

fn mask_length(
    shape: AttentionShape,
    mask_shape: AttentionMaskShape,
) -> Result<usize, AttentionKernelError> {
    match mask_shape {
        AttentionMaskShape::KeyTokens => Ok(shape.key_tokens),
        AttentionMaskShape::QueryByKey => checked_product(&[shape.query_tokens, shape.key_tokens]),
        AttentionMaskShape::BatchQueryByKey => {
            checked_product(&[shape.batch, shape.query_tokens, shape.key_tokens])
        }
        AttentionMaskShape::BatchHeadQueryByKey => checked_product(&[
            shape.batch,
            shape.heads,
            shape.query_tokens,
            shape.key_tokens,
        ]),
    }
}

fn mask_index(
    shape: AttentionShape,
    mask_shape: AttentionMaskShape,
    batch: usize,
    head: usize,
    query_token: usize,
    key_token: usize,
) -> usize {
    match mask_shape {
        AttentionMaskShape::KeyTokens => key_token,
        AttentionMaskShape::QueryByKey => query_token * shape.key_tokens + key_token,
        AttentionMaskShape::BatchQueryByKey => {
            (batch * shape.query_tokens + query_token) * shape.key_tokens + key_token
        }
        AttentionMaskShape::BatchHeadQueryByKey => {
            (((batch * shape.heads + head) * shape.query_tokens + query_token) * shape.key_tokens)
                + key_token
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn tensor_index(
    layout: AttentionLayout,
    _batch_count: usize,
    token_count: usize,
    head_count: usize,
    component_count: usize,
    batch: usize,
    token: usize,
    head: usize,
    component: usize,
) -> usize {
    match layout {
        AttentionLayout::Nhd => {
            (((batch * token_count + token) * head_count + head) * component_count) + component
        }
        AttentionLayout::Hnd => {
            (((batch * head_count + head) * token_count + token) * component_count) + component
        }
    }
}

fn read_value(
    values: &[f32],
    index: usize,
    name: &'static str,
) -> Result<f32, AttentionKernelError> {
    values
        .get(index)
        .copied()
        .ok_or(AttentionKernelError::ValueCount {
            name,
            expected: index.saturating_add(1),
            actual: values.len(),
        })
}

fn write_value(
    values: &mut [f32],
    index: usize,
    value: f32,
    name: &'static str,
) -> Result<(), AttentionKernelError> {
    let actual = values.len();
    let destination = values
        .get_mut(index)
        .ok_or(AttentionKernelError::ValueCount {
            name,
            expected: index.saturating_add(1),
            actual,
        })?;
    *destination = value;
    Ok(())
}

fn add_value(
    values: &mut [f32],
    index: usize,
    value: f32,
    name: &'static str,
) -> Result<(), AttentionKernelError> {
    let actual = values.len();
    let destination = values
        .get_mut(index)
        .ok_or(AttentionKernelError::ValueCount {
            name,
            expected: index.saturating_add(1),
            actual,
        })?;
    *destination += value;
    Ok(())
}

fn zeroed(length: usize, name: &'static str) -> Result<Vec<f32>, AttentionKernelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| AttentionKernelError::AllocationFailed { name })?;
    values.resize(length, 0.0);
    Ok(values)
}

fn temporary_zeroed(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    length: usize,
) -> Result<CpuWorkspaceVec<f32>, AttentionKernelError> {
    let mut values = backend.workspace_vec(context, length)?;
    for _ in 0..length {
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn checked_product(values: &[usize]) -> Result<usize, AttentionKernelError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(AttentionKernelError::ShapeOverflow)
    })
}
