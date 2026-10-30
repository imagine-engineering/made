use crate::{
    QuantizationError, QuantizedLinearMatrix, QuantizedMaterialization, QuantizedMatrix,
    quantization::{QuantLinearLayout, QuantLinearScale, quantize_linear_matrix},
};
use sim_tensor::{
    DType, DeviceId, ExecutionContext, Tensor, TensorBackend, TensorError,
    autograd::breadth::{AutogradBreadthError, FunctionContext},
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_backend_exact_native,
        linear_vjp_with_context_exact_native, linear_with_context_exact_native,
        tensor_from_f32_with_backend_exact_native, tensor_to_f32_with_backend_exact_native,
    },
};
use thiserror::Error;

const INPUT_GRADIENT_INDEX: usize = 0;
const WEIGHT_GRADIENT_INDEX: usize = 1;
const BIAS_GRADIENT_INDEX: usize = 2;
const QUANT_LINEAR_INPUT_ARITY: usize = 6;

#[derive(Clone, Debug)]
pub enum QuantLinearWeight {
    Dense(Tensor),
    Quantized(QuantizedLinearMatrix),
    CatalogQuantized(QuantizedMatrix),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantLinearOptions {
    pub layout: Option<QuantLinearLayout>,
    pub input_scale: QuantLinearScale,
    pub compute_dtype: DType,
    pub weight_requires_grad: bool,
    pub fp8_backward: bool,
}

impl QuantLinearOptions {
    pub fn from_source_layout(
        source_layout: Option<&str>,
        input_scale: QuantLinearScale,
        compute_dtype: DType,
        weight_requires_grad: bool,
        fp8_backward: bool,
    ) -> Result<Self, QuantLinearError> {
        let layout = source_layout
            .map(|source_name| {
                QuantLinearLayout::from_source_name(source_name).ok_or_else(|| {
                    QuantLinearError::UnsupportedLayout {
                        source_name: source_name.to_owned(),
                    }
                })
            })
            .transpose()?;
        Ok(Self {
            layout,
            input_scale,
            compute_dtype,
            weight_requires_grad,
            fp8_backward,
        })
    }
}

#[derive(Debug)]
pub struct QuantLinearExecution {
    output: Tensor,
    context: FunctionContext,
    output_shape: Vec<u64>,
    input_shape: Vec<u64>,
    has_bias: bool,
    weight_requires_grad: bool,
    compute_dtype: DType,
    fp8_backward: bool,
    weight_layout: Option<QuantLinearLayout>,
    cached_input: Option<QuantizedLinearMatrix>,
    cached_weight: Option<QuantLinearWeightCache>,
}

#[derive(Clone, Debug)]
enum QuantLinearWeightCache {
    Layout(QuantizedLinearMatrix),
    Catalog(QuantizedMatrix),
}

impl QuantLinearWeightCache {
    fn materialize(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantizedMaterialization, QuantizationError> {
        match self {
            Self::Layout(matrix) => matrix.materialize(backend, context),
            Self::Catalog(matrix) => matrix.materialize(backend, context),
        }
    }
}

enum MaterializedValues {
    Dense(Vec<f32>),
    Quantized(QuantizedMaterialization),
}

impl MaterializedValues {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Dense(values) => values,
            Self::Quantized(materialization) => materialization.values(),
        }
    }
}

impl QuantLinearExecution {
    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn backward(
        &mut self,
        backend: &dyn TensorBackend,
        output_gradient: &Tensor,
        create_graph: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantLinearGradients, QuantLinearError> {
        let result = self.backward_impl(backend, output_gradient, create_graph, context);
        self.context.release();
        self.cached_input = None;
        self.cached_weight = None;
        result
    }

    fn backward_impl(
        &self,
        backend: &dyn TensorBackend,
        output_gradient: &Tensor,
        create_graph: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantLinearGradients, QuantLinearError> {
        check_cancelled(context)?;
        if create_graph {
            return Err(QuantLinearError::OnceDifferentiable);
        }
        validate_tensor(
            output_gradient,
            backend.device(),
            Some(&self.output_shape),
            "output gradient",
        )?;
        let saved = self.context.saved_tensors()?;
        let (input_values, input_shape, backward_weight) = if self.fp8_backward {
            let backward_weight = saved.first().ok_or(QuantLinearError::ReleasedState)?;
            let cached_input = self
                .cached_input
                .as_ref()
                .ok_or(QuantLinearError::ReleasedState)?;
            (
                MaterializedValues::Quantized(cached_input.materialize(backend, context)?),
                vec![cached_input.rows(), cached_input.columns()],
                backward_weight,
            )
        } else {
            let backward_input = saved.first().ok_or(QuantLinearError::ReleasedState)?;
            let backward_weight = saved.get(1).ok_or(QuantLinearError::ReleasedState)?;
            (
                MaterializedValues::Dense(compute_values(
                    backend,
                    backward_input,
                    self.compute_dtype,
                    context,
                )?),
                shape_to_usize(backward_input.descriptor().shape())?,
                backward_weight,
            )
        };
        let gradient_values =
            compute_values(backend, output_gradient, self.compute_dtype, context)?;
        let weight_values = match &self.cached_weight {
            Some(weight) => MaterializedValues::Quantized(weight.materialize(backend, context)?),
            None => MaterializedValues::Dense(compute_values(
                backend,
                backward_weight,
                self.compute_dtype,
                context,
            )?),
        };
        let weight_shape = shape_to_usize(backward_weight.descriptor().shape())?;
        let quantized_gradient = if self.fp8_backward {
            let output_shape = shape_to_usize(output_gradient.descriptor().shape())?;
            let output_columns = *output_shape
                .last()
                .ok_or(QuantLinearError::InvalidShape("output gradient rank"))?;
            Some(quantized_values(
                QuantLinearLayout::TensorCoreFp8E5M2,
                &gradient_values,
                flattened_rows(&output_shape)?,
                output_columns,
                QuantLinearScale::Default,
                self.compute_dtype,
                backend,
                context,
            )?)
        } else {
            None
        };
        let quantized_weight =
            if self.fp8_backward && !self.weight_layout.is_some_and(QuantLinearLayout::is_fp8) {
                Some(quantized_values(
                    QuantLinearLayout::TensorCoreFp8E4M3,
                    weight_values.as_slice(),
                    *weight_shape
                        .first()
                        .ok_or(QuantLinearError::InvalidShape("weight rank"))?,
                    *weight_shape
                        .get(1)
                        .ok_or(QuantLinearError::InvalidShape("weight rank"))?,
                    QuantLinearScale::Default,
                    self.compute_dtype,
                    backend,
                    context,
                )?)
            } else {
                None
            };
        let matmul_gradient = quantized_gradient
            .as_ref()
            .map_or(gradient_values.as_slice(), QuantizedMaterialization::values);
        let matmul_weight = quantized_weight
            .as_ref()
            .map_or(weight_values.as_slice(), QuantizedMaterialization::values);
        let weight_rows = *weight_shape
            .first()
            .ok_or(QuantLinearError::InvalidShape("weight rank"))?;
        let bias_marker = self.has_bias.then(|| vec![0.0; weight_rows]);
        let mut gradients = linear_vjp_with_context_exact_native(
            input_values.as_slice(),
            &input_shape,
            matmul_weight,
            &weight_shape,
            bias_marker.as_deref(),
            matmul_gradient,
            backend.device(),
            context,
        )?;
        if self.fp8_backward && self.has_bias {
            gradients.bias = linear_vjp_with_context_exact_native(
                input_values.as_slice(),
                &input_shape,
                weight_values.as_slice(),
                &weight_shape,
                bias_marker.as_deref(),
                &gradient_values,
                backend.device(),
                context,
            )?
            .bias;
        }
        check_cancelled(context)?;

        let input_gradient = tensor_from_f32_with_backend_exact_native(
            backend,
            &self.input_shape,
            &gradients.input,
            self.compute_dtype,
            backend.device(),
            context,
        )?;
        let weight_gradient = if self.weight_requires_grad {
            Some(tensor_from_f32_with_backend_exact_native(
                backend,
                backward_weight.descriptor().shape(),
                &gradients.weight,
                self.compute_dtype,
                backend.device(),
                context,
            )?)
        } else {
            None
        };
        let bias_gradient = gradients
            .bias
            .as_deref()
            .map(|values| {
                tensor_from_f32_with_backend_exact_native(
                    backend,
                    &[u64::try_from(values.len()).map_err(|_| {
                        QuantLinearError::InvalidShape("bias gradient length overflow")
                    })?],
                    values,
                    self.compute_dtype,
                    backend.device(),
                    context,
                )
                .map_err(QuantLinearError::from)
            })
            .transpose()?;
        check_cancelled(context)?;
        let mut result = std::array::from_fn(|_| None);
        result[INPUT_GRADIENT_INDEX] = Some(input_gradient);
        result[WEIGHT_GRADIENT_INDEX] = weight_gradient;
        result[BIAS_GRADIENT_INDEX] = bias_gradient;
        Ok(QuantLinearGradients { values: result })
    }
}

#[derive(Debug)]
pub struct QuantLinearGradients {
    values: [Option<Tensor>; QUANT_LINEAR_INPUT_ARITY],
}

impl QuantLinearGradients {
    pub const fn input_arity(&self) -> usize {
        QUANT_LINEAR_INPUT_ARITY
    }

    pub fn input(&self) -> Option<&Tensor> {
        self.values[INPUT_GRADIENT_INDEX].as_ref()
    }

    pub fn weight(&self) -> Option<&Tensor> {
        self.values[WEIGHT_GRADIENT_INDEX].as_ref()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.values[BIAS_GRADIENT_INDEX].as_ref()
    }

    pub fn as_slice(&self) -> &[Option<Tensor>] {
        &self.values
    }
}

#[derive(Debug, Error)]
pub enum QuantLinearError {
    #[error("quantized linear requires a rank-two or higher input")]
    InvalidInputRank,
    #[error("quantized linear {0} shape is invalid")]
    InvalidShape(&'static str),
    #[error("quantized linear backend does not support device {device:?}")]
    UnsupportedDevice { device: DeviceId },
    #[error("quantized linear does not support dtype {dtype:?}")]
    UnsupportedDType { dtype: DType },
    #[error("quantized linear does not support source layout {source_name}")]
    UnsupportedLayout { source_name: String },
    #[error("quantized linear tensor shapes do not match")]
    ShapeMismatch,
    #[error("quantized linear backward is once differentiable")]
    OnceDifferentiable,
    #[error("quantized linear checkpoint state has been released")]
    ReleasedState,
    #[error("quantized linear execution was cancelled")]
    Cancelled,
    #[error(transparent)]
    Quantization(#[from] QuantizationError),
    #[error(transparent)]
    NativeOps(#[from] crate::NativeOpsError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Linear(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Autograd(#[from] AutogradBreadthError),
}

pub fn quant_linear_forward_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: QuantLinearWeight,
    bias: Option<&Tensor>,
    options: QuantLinearOptions,
    context: &ExecutionContext<'_>,
) -> Result<QuantLinearExecution, QuantLinearError> {
    check_cancelled(context)?;
    validate_compute_dtype(options.compute_dtype)?;
    validate_tensor(input, backend.device(), None, "input")?;
    if input.descriptor().shape().len() < 2 {
        return Err(QuantLinearError::InvalidInputRank);
    }
    let input_shape = shape_to_usize(input.descriptor().shape())?;
    let input_columns = *input_shape
        .last()
        .ok_or(QuantLinearError::InvalidInputRank)?;
    let input_rows = flattened_rows(&input_shape)?;
    let input_values = compute_values(backend, input, options.compute_dtype, context)?;
    let (weight_values, weight_shape, saved_weight, weight_layout, cached_weight) = match weight {
        QuantLinearWeight::Dense(weight) => {
            validate_tensor(&weight, backend.device(), None, "weight")?;
            let shape = shape_to_usize(weight.descriptor().shape())?;
            let values = compute_values(backend, &weight, options.compute_dtype, context)?;
            (MaterializedValues::Dense(values), shape, weight, None, None)
        }
        QuantLinearWeight::Quantized(matrix) => {
            let shape = vec![matrix.rows(), matrix.columns()];
            let materialization = matrix.materialize(backend, context)?;
            let identity = tensor_from_f32_with_backend_exact_native(
                backend,
                &shape_to_u64(&shape)?,
                materialization.values(),
                options.compute_dtype,
                backend.device(),
                context,
            )?;
            (
                MaterializedValues::Quantized(materialization),
                shape,
                identity,
                Some(matrix.layout()),
                Some(QuantLinearWeightCache::Layout(matrix)),
            )
        }
        QuantLinearWeight::CatalogQuantized(matrix) => {
            let shape = vec![matrix.rows(), matrix.columns()];
            let materialization = matrix.materialize(backend, context)?;
            let identity = tensor_from_f32_with_backend_exact_native(
                backend,
                &shape_to_u64(&shape)?,
                materialization.values(),
                options.compute_dtype,
                backend.device(),
                context,
            )?;
            (
                MaterializedValues::Quantized(materialization),
                shape,
                identity,
                None,
                Some(QuantLinearWeightCache::Catalog(matrix)),
            )
        }
    };
    if weight_shape.len() != 2 || weight_shape[1] != input_columns {
        return Err(QuantLinearError::ShapeMismatch);
    }
    let bias_values = bias
        .map(|bias| {
            let weight_rows = *weight_shape
                .first()
                .ok_or(QuantLinearError::InvalidShape("weight rank"))?;
            let bias_width = u64::try_from(weight_rows)
                .map_err(|_| QuantLinearError::InvalidShape("bias width overflow"))?;
            validate_tensor(bias, backend.device(), Some(&[bias_width]), "bias")?;
            compute_values(backend, bias, options.compute_dtype, context)
        })
        .transpose()?;
    let forward_quantized = options
        .layout
        .map(|layout| {
            quantize_linear_matrix(
                layout,
                options.compute_dtype,
                &input_values,
                input_rows,
                input_columns,
                options.input_scale,
                context.cancellation,
            )
        })
        .transpose()?;
    let forward_input = forward_quantized
        .as_ref()
        .map(|matrix| matrix.materialize(backend, context))
        .transpose()?
        .map(MaterializedValues::Quantized)
        .unwrap_or_else(|| MaterializedValues::Dense(input_values.clone()));
    let output_values = linear_with_context_exact_native(
        forward_input.as_slice(),
        &[input_rows, input_columns],
        weight_values.as_slice(),
        &weight_shape,
        bias_values.as_deref(),
        backend.device(),
        context,
    )?;
    let mut output_shape = input.descriptor().shape().to_vec();
    let output_width = u64::try_from(
        *weight_shape
            .first()
            .ok_or(QuantLinearError::InvalidShape("weight rank"))?,
    )
    .map_err(|_| QuantLinearError::InvalidShape("output width overflow"))?;
    let Some(last) = output_shape.last_mut() else {
        return Err(QuantLinearError::InvalidInputRank);
    };
    *last = output_width;
    let output = tensor_from_f32_with_backend_exact_native(
        backend,
        &output_shape,
        &output_values.values,
        options.compute_dtype,
        backend.device(),
        context,
    )?;

    let cached_input = if options.fp8_backward {
        Some(if options.layout.is_some_and(QuantLinearLayout::is_fp8) {
            forward_quantized.ok_or(QuantLinearError::ReleasedState)?
        } else {
            quantize_linear_matrix(
                QuantLinearLayout::TensorCoreFp8E4M3,
                options.compute_dtype,
                &input_values,
                input_rows,
                input_columns,
                QuantLinearScale::Default,
                context.cancellation,
            )?
        })
    } else {
        None
    };
    check_cancelled(context)?;
    let mut function_context = FunctionContext::new(vec![
        true,
        options.weight_requires_grad,
        bias.is_some(),
        false,
        false,
        false,
    ]);
    if options.fp8_backward {
        function_context.save_for_backward(&[&saved_weight])?;
    } else {
        function_context.save_for_backward(&[input, &saved_weight])?;
    }
    Ok(QuantLinearExecution {
        output,
        context: function_context,
        output_shape,
        input_shape: input.descriptor().shape().to_vec(),
        has_bias: bias.is_some(),
        weight_requires_grad: options.weight_requires_grad,
        compute_dtype: options.compute_dtype,
        fp8_backward: options.fp8_backward,
        weight_layout,
        cached_input,
        cached_weight,
    })
}

fn quantized_values(
    layout: QuantLinearLayout,
    values: &[f32],
    rows: usize,
    columns: usize,
    scale: QuantLinearScale,
    dtype: DType,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<QuantizedMaterialization, QuantLinearError> {
    let quantized = quantize_linear_matrix(
        layout,
        dtype,
        values,
        rows,
        columns,
        scale,
        context.cancellation,
    )?;
    Ok(quantized.materialize(backend, context)?)
}

fn compute_values(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, QuantLinearError> {
    let cast = cast_to_with_backend_exact_native(
        backend,
        tensor,
        dtype,
        backend.device(),
        false,
        false,
        context,
    )?;
    Ok(tensor_to_f32_with_backend_exact_native(
        backend, &cast, context,
    )?)
}

fn validate_compute_dtype(dtype: DType) -> Result<(), QuantLinearError> {
    if matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
        Ok(())
    } else {
        Err(QuantLinearError::UnsupportedDType { dtype })
    }
}

fn validate_tensor(
    tensor: &Tensor,
    device: DeviceId,
    expected_shape: Option<&[u64]>,
    _subject: &'static str,
) -> Result<(), QuantLinearError> {
    if tensor.descriptor().device() != device {
        return Err(QuantLinearError::UnsupportedDevice {
            device: tensor.descriptor().device(),
        });
    }
    validate_compute_dtype(tensor.descriptor().dtype())?;
    if expected_shape.is_some_and(|shape| tensor.descriptor().shape() != shape) {
        return Err(QuantLinearError::ShapeMismatch);
    }
    Ok(())
}

fn flattened_rows(shape: &[usize]) -> Result<usize, QuantLinearError> {
    shape
        .get(..shape.len().saturating_sub(1))
        .ok_or(QuantLinearError::InvalidInputRank)?
        .iter()
        .try_fold(1_usize, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or(QuantLinearError::InvalidShape("flattened row overflow"))
        })
}

fn shape_to_usize(shape: &[u64]) -> Result<Vec<usize>, QuantLinearError> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .map_err(|_| QuantLinearError::InvalidShape("dimension overflow"))
        })
        .collect()
}

fn shape_to_u64(shape: &[usize]) -> Result<Vec<u64>, QuantLinearError> {
    shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension)
                .map_err(|_| QuantLinearError::InvalidShape("dimension overflow"))
        })
        .collect()
}

fn check_cancelled(context: &ExecutionContext<'_>) -> Result<(), QuantLinearError> {
    if context.cancellation.is_cancelled() {
        Err(QuantLinearError::Cancelled)
    } else {
        Ok(())
    }
}
