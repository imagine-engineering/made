use super::{
    AutogradError, AutogradTape, BackwardRule, GradientMode, HigherOrderContext, OutputSlot,
    SavedTensor,
};
use crate::{
    AutocastPolicy, CpuBackend, DType, DeviceId, ExecutionContext, MemoryFormatReference, Tensor,
    TensorBackend, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_06::{
        CheckpointExecution, ElementwiseRuntimePartSixError,
        checkpoint_execution_from_outputs_exact_native,
    },
    generated_elementwise_or_runtime_operation_08::{
        index_select_vjp_with_context_exact_native, index_select_with_context_exact_native,
        square_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::{
        BinaryGradients, mul_vjp_with_context_exact_native, mul_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_13::addmm_with_context_exact_native,
    generated_elementwise_or_runtime_operation_16::add_method_with_context_exact_native,
    generated_linear_algebra_01::{
        EinsumGradients, MatmulGradients, einsum_jvp_with_context_exact_native,
        einsum_vjp_with_context_exact_native, einsum_with_context_exact_native,
        mm_vjp_with_context_exact_native, mm_with_context_exact_native,
        transpose_last_two_with_context_exact_native,
    },
    generated_reduction_01::tensor_min_with_context_exact_native,
    generated_reduction_02::tensor_sum_with_context_exact_native,
    generated_storage_dtype_device_01::clone_with_context_exact_native,
};
use std::sync::Arc;
use thiserror::Error;

pub use super::GradientStore;
pub use super::{
    GradScalerConfig, GradScalerOptimizerDecision, NativeGradScaler, NativeTrainingComputeDType,
    NativeTrainingError, NativeTrainingExample, NativeTrainingLoss, NativeTrainingOptimizer,
    NativeTrainingPlan, NativeTrainingResult, train_lora_parameters_exact_native,
};
pub use crate::generated_elementwise_or_runtime_operation_07::NativeSgd;
pub use crate::generated_elementwise_or_runtime_operation_09::NativeAdamW;
pub use crate::generated_elementwise_or_runtime_operation_11::NativeAdam;
pub use crate::generated_elementwise_or_runtime_operation_14::detach_exact_native;
pub use crate::generated_elementwise_or_runtime_operation_23::NativeRmsprop;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutogradConstructOwner {
    Tape,
    FunctionContext,
    GradientStore,
    Checkpoint,
    CustomFunction,
    GradScaler,
    Sgd,
    Adam,
    AdamW,
    Rmsprop,
    TensorAlias,
    AutocastPolicy,
    QuantizedModelAdapter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutogradConstructContract {
    pub id: &'static str,
    pub construct: &'static str,
    pub symbol: &'static str,
    pub owner: AutogradConstructOwner,
}

pub const AUTOGRAD_CONSTRUCTS: [AutogradConstructContract; 36] = [
    contract(
        "COMFY-AUTOGRAD-0164A83D79F9",
        "gradient-state",
        "torch.Tensor.requires_grad",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-08DA3A226CB4",
        "custom-autograd-function",
        "vector_quantize",
        AutogradConstructOwner::CustomFunction,
    ),
    contract(
        "COMFY-AUTOGRAD-0BDFE52B87F3",
        "optimizer-or-gradient-scaler",
        "torch.training.step",
        AutogradConstructOwner::GradScaler,
    ),
    contract(
        "COMFY-AUTOGRAD-0C5FA58D517B",
        "optimizer-or-gradient-scaler",
        "torch.optim.Adam",
        AutogradConstructOwner::Adam,
    ),
    contract(
        "COMFY-AUTOGRAD-104D91298DF9",
        "custom-autograd-function",
        "CheckpointFunction",
        AutogradConstructOwner::Checkpoint,
    ),
    contract(
        "COMFY-AUTOGRAD-1691472B873D",
        "custom-function-context",
        "torch.autograd.FunctionCtx.needs_input_grad",
        AutogradConstructOwner::FunctionContext,
    ),
    contract(
        "COMFY-AUTOGRAD-2682346109CE",
        "gradient-mode",
        "torch.enable_grad",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-285F07173F3E",
        "mixed-precision-autograd",
        "torch.amp.GradScaler",
        AutogradConstructOwner::GradScaler,
    ),
    contract(
        "COMFY-AUTOGRAD-30043B9C2264",
        "custom-autograd-function",
        "QuantLinearFunc",
        AutogradConstructOwner::QuantizedModelAdapter,
    ),
    contract(
        "COMFY-AUTOGRAD-304CC342AC2A",
        "custom-autograd-function",
        "HadaWeightTucker",
        AutogradConstructOwner::CustomFunction,
    ),
    contract(
        "COMFY-AUTOGRAD-35DAFB8F8753",
        "reverse-mode-execution",
        "torch.Tensor.backward",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-3CBCCC7F6931",
        "reverse-mode-execution",
        "torch.autograd.grad",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-4CF4D676FFBB",
        "optimizer-or-gradient-scaler",
        "torch.optim.RMSprop",
        AutogradConstructOwner::Rmsprop,
    ),
    contract(
        "COMFY-AUTOGRAD-58A5B3D9CFE8",
        "gradient-mode",
        "torch.inference_mode",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-617621E1EEBE",
        "gradient-state",
        "torch.Tensor.requires_grad_",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-619FFDF53F34",
        "graph-detachment-or-storage-alias",
        "torch.Tensor.detach",
        AutogradConstructOwner::TensorAlias,
    ),
    contract(
        "COMFY-AUTOGRAD-640C4BF17167",
        "custom-autograd-function",
        "AddAuxLoss",
        AutogradConstructOwner::CustomFunction,
    ),
    contract(
        "COMFY-AUTOGRAD-75400A23E6BE",
        "optimizer-or-gradient-scaler",
        "torch.training.scale",
        AutogradConstructOwner::GradScaler,
    ),
    contract(
        "COMFY-AUTOGRAD-77E715FA8F5B",
        "mixed-precision-autograd",
        "torch.cuda.amp.autocast",
        AutogradConstructOwner::AutocastPolicy,
    ),
    contract(
        "COMFY-AUTOGRAD-885F94147CD4",
        "reverse-mode-execution",
        "torch.Tensor.grad",
        AutogradConstructOwner::GradientStore,
    ),
    contract(
        "COMFY-AUTOGRAD-97F154ABF757",
        "optimizer-or-gradient-scaler",
        "torch.optim._functional.adamw",
        AutogradConstructOwner::AdamW,
    ),
    contract(
        "COMFY-AUTOGRAD-9A036C261AF5",
        "custom-function-context",
        "torch.autograd.FunctionCtx.save_for_backward",
        AutogradConstructOwner::FunctionContext,
    ),
    contract(
        "COMFY-AUTOGRAD-A1ACCD3F23F9",
        "custom-autograd-function",
        "OffloadCheckpointFunction",
        AutogradConstructOwner::Checkpoint,
    ),
    contract(
        "COMFY-AUTOGRAD-A1FE605E0A41",
        "optimizer-or-gradient-scaler",
        "torch.optim.SGD",
        AutogradConstructOwner::Sgd,
    ),
    contract(
        "COMFY-AUTOGRAD-A50883A5EA1D",
        "custom-autograd-function",
        "HadaWeight",
        AutogradConstructOwner::CustomFunction,
    ),
    contract(
        "COMFY-AUTOGRAD-ABC8AAD8B0B5",
        "optimizer-or-gradient-scaler",
        "torch.training.zero_grad",
        AutogradConstructOwner::GradientStore,
    ),
    contract(
        "COMFY-AUTOGRAD-B16B6C3AAC27",
        "optimizer-or-gradient-scaler",
        "torch.training.update",
        AutogradConstructOwner::GradScaler,
    ),
    contract(
        "COMFY-AUTOGRAD-B575430CB29A",
        "graph-detachment-or-storage-alias",
        "torch.Tensor.data",
        AutogradConstructOwner::TensorAlias,
    ),
    contract(
        "COMFY-AUTOGRAD-B6C63329EB83",
        "mixed-precision-autograd",
        "torch.autocast",
        AutogradConstructOwner::AutocastPolicy,
    ),
    contract(
        "COMFY-AUTOGRAD-B93A2676328D",
        "custom-function-context",
        "torch.autograd.FunctionCtx.mark_non_differentiable",
        AutogradConstructOwner::FunctionContext,
    ),
    contract(
        "COMFY-AUTOGRAD-BC03B0A46C6A",
        "optimizer-or-gradient-scaler",
        "torch.training.unscale_",
        AutogradConstructOwner::GradScaler,
    ),
    contract(
        "COMFY-AUTOGRAD-C235B5282FB7",
        "custom-function-context",
        "torch.autograd.FunctionCtx.saved_tensors",
        AutogradConstructOwner::FunctionContext,
    ),
    contract(
        "COMFY-AUTOGRAD-CBC045CBB408",
        "optimizer-or-gradient-scaler",
        "torch.optim.AdamW",
        AutogradConstructOwner::AdamW,
    ),
    contract(
        "COMFY-AUTOGRAD-E31FB2A11AFF",
        "gradient-mode",
        "torch.no_grad",
        AutogradConstructOwner::Tape,
    ),
    contract(
        "COMFY-AUTOGRAD-E50C96639633",
        "activation-checkpointing",
        "torch.utils.checkpoint.checkpoint",
        AutogradConstructOwner::Checkpoint,
    ),
    contract(
        "COMFY-AUTOGRAD-F5EB56FAE2E4",
        "gradient-state",
        "torch.Tensor.requires_grad keyword",
        AutogradConstructOwner::Tape,
    ),
];

const fn contract(
    id: &'static str,
    construct: &'static str,
    symbol: &'static str,
    owner: AutogradConstructOwner,
) -> AutogradConstructContract {
    AutogradConstructContract {
        id,
        construct,
        symbol,
        owner,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HigherOrderPolicy {
    Analytical,
    FirstOrderOnly,
    OnceDifferentiable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CustomFunctionContract {
    pub id: &'static str,
    pub symbol: &'static str,
    pub forward_arity: usize,
    pub variadic_inputs: bool,
    pub forward_outputs: usize,
    pub backward_inputs: usize,
    pub backward_outputs: usize,
    pub higher_order: HigherOrderPolicy,
    pub fixture: &'static str,
}

impl CustomFunctionContract {
    pub fn validate_higher_order_request(
        self,
        create_graph: bool,
    ) -> Result<(), AutogradBreadthError> {
        if !create_graph || self.higher_order == HigherOrderPolicy::Analytical {
            Ok(())
        } else {
            Err(AutogradBreadthError::HigherOrderUnavailable {
                symbol: self.symbol,
                policy: self.higher_order,
            })
        }
    }
}

pub const CUSTOM_FUNCTIONS: [CustomFunctionContract; 7] = [
    custom_function(
        "COMFY-AUTOGRAD-08DA3A226CB4",
        "vector_quantize",
        2,
        false,
        2,
        2,
        2,
        HigherOrderPolicy::Analytical,
        "breadth-v1.json#vector_quantize",
    ),
    custom_function(
        "COMFY-AUTOGRAD-104D91298DF9",
        "CheckpointFunction",
        3,
        true,
        1,
        1,
        3,
        HigherOrderPolicy::FirstOrderOnly,
        "breadth-v1.json#checkpoint_function",
    ),
    custom_function(
        "COMFY-AUTOGRAD-30043B9C2264",
        "QuantLinearFunc",
        6,
        false,
        1,
        1,
        6,
        HigherOrderPolicy::OnceDifferentiable,
        "crates/sim_test_support/parity/fixtures/quant-linear-source-oracle.json#callable",
    ),
    custom_function(
        "COMFY-AUTOGRAD-304CC342AC2A",
        "HadaWeightTucker",
        7,
        false,
        1,
        1,
        7,
        HigherOrderPolicy::Analytical,
        "breadth-v1.json#hada_weight_tucker",
    ),
    custom_function(
        "COMFY-AUTOGRAD-640C4BF17167",
        "AddAuxLoss",
        2,
        false,
        1,
        1,
        2,
        HigherOrderPolicy::Analytical,
        "breadth-v1.json#add_aux_loss",
    ),
    custom_function(
        "COMFY-AUTOGRAD-A1ACCD3F23F9",
        "OffloadCheckpointFunction",
        2,
        false,
        1,
        1,
        2,
        HigherOrderPolicy::FirstOrderOnly,
        "breadth-v1.json#offload_checkpoint",
    ),
    custom_function(
        "COMFY-AUTOGRAD-A50883A5EA1D",
        "HadaWeight",
        5,
        false,
        1,
        1,
        5,
        HigherOrderPolicy::Analytical,
        "breadth-v1.json#hada_weight",
    ),
];

const fn custom_function(
    id: &'static str,
    symbol: &'static str,
    forward_arity: usize,
    variadic_inputs: bool,
    forward_outputs: usize,
    backward_inputs: usize,
    backward_outputs: usize,
    higher_order: HigherOrderPolicy,
    fixture: &'static str,
) -> CustomFunctionContract {
    CustomFunctionContract {
        id,
        symbol,
        forward_arity,
        variadic_inputs,
        forward_outputs,
        backward_inputs,
        backward_outputs,
        higher_order,
        fixture,
    }
}

#[derive(Debug, Error)]
pub enum AutogradBreadthError {
    #[error("native autograd breadth operation was cancelled")]
    Cancelled,
    #[error("native autograd breadth operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("native autograd breadth operation {operation} does not support dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("canonical tensor operation {operation} failed: {reason}")]
    CanonicalOperation {
        operation: &'static str,
        reason: String,
    },
    #[error("invalid native autograd function input: {0}")]
    InvalidInput(String),
    #[error("a custom autograd context has already released its saved tensors")]
    ReleasedContext,
    #[error("custom autograd gradient arity is invalid: expected {expected}, received {actual}")]
    GradientArity { expected: usize, actual: usize },
    #[error("custom autograd function {symbol} rejects create_graph under {policy:?} policy")]
    HigherOrderUnavailable {
        symbol: &'static str,
        policy: HigherOrderPolicy,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
}

#[derive(Clone, Debug)]
pub struct FunctionContext {
    needs_input_grad: Vec<bool>,
    saved: Vec<SavedTensor>,
    non_differentiable_outputs: Vec<usize>,
    released: bool,
}

impl FunctionContext {
    pub fn new(needs_input_grad: Vec<bool>) -> Self {
        Self {
            needs_input_grad,
            saved: Vec::new(),
            non_differentiable_outputs: Vec::new(),
            released: false,
        }
    }

    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.needs_input_grad.get(index).copied().unwrap_or(false)
    }

    pub fn save_for_backward(&mut self, tensors: &[&Tensor]) -> Result<(), AutogradBreadthError> {
        if self.released {
            return Err(AutogradBreadthError::ReleasedContext);
        }
        self.saved
            .extend(tensors.iter().map(|tensor| SavedTensor::capture(tensor)));
        Ok(())
    }

    pub fn saved_tensors(&self) -> Result<Vec<Tensor>, AutogradBreadthError> {
        if self.released {
            return Err(AutogradBreadthError::ReleasedContext);
        }
        self.saved
            .iter()
            .map(|saved| {
                saved.validate()?;
                Ok(saved.tensor().clone())
            })
            .collect()
    }

    pub fn mark_non_differentiable(&mut self, output: usize) -> Result<(), AutogradBreadthError> {
        if self.released {
            return Err(AutogradBreadthError::ReleasedContext);
        }
        if !self.non_differentiable_outputs.contains(&output) {
            self.non_differentiable_outputs.push(output);
        }
        Ok(())
    }

    pub fn is_non_differentiable(&self, output: usize) -> bool {
        self.non_differentiable_outputs.contains(&output)
    }

    pub fn retained_tensor_count(&self) -> usize {
        self.saved.len()
    }

    pub fn release(&mut self) {
        self.saved.clear();
        self.released = true;
    }
}

pub struct VectorQuantizeFunction {
    context: FunctionContext,
    output_shape: Vec<u64>,
}

impl VectorQuantizeFunction {
    pub fn forward(
        backend: &CpuBackend,
        input: &Tensor,
        codebook: &Tensor,
        needs_input_grad: [bool; 2],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor, Tensor), AutogradBreadthError> {
        require_f32_execution(input, execution)?;
        require_f32_execution(codebook, execution)?;
        let input_shape = require_rank(input, 2)?;
        let codebook_shape = require_rank(codebook, 2)?;
        if input_shape[1] != codebook_shape[1] || codebook_shape[0] == 0 {
            return Err(AutogradBreadthError::InvalidInput(
                "vector quantization dimensions are incompatible".to_owned(),
            ));
        }
        let codebook_squared = square_with_context_exact_native(backend, codebook, execution)
            .map_err(|error| {
                canonical_error("vector_quantize.square_codebook", error, execution)
            })?;
        let codebook_squares = tensor_sum_with_context_exact_native(
            backend,
            &codebook_squared,
            Some(&[1]),
            false,
            None,
            execution,
        )
        .map_err(|error| canonical_error("vector_quantize.sum_codebook", error, execution))?;
        let input_squared = square_with_context_exact_native(backend, input, execution)
            .map_err(|error| canonical_error("vector_quantize.square_input", error, execution))?;
        let input_squares = tensor_sum_with_context_exact_native(
            backend,
            &input_squared,
            Some(&[1]),
            true,
            None,
            execution,
        )
        .map_err(|error| canonical_error("vector_quantize.sum_input", error, execution))?;
        let distance_base = add_method_with_context_exact_native(
            backend,
            &input_squares,
            ElementwiseOperand::Tensor(&codebook_squares),
            1.0,
            execution,
        )
        .map_err(|error| canonical_error("vector_quantize.distance_base", error, execution))?;
        let codebook_transposed = transpose_last_two_with_context_exact_native(codebook, execution)
            .map_err(|error| canonical_error("vector_quantize.transpose", error, execution))?;
        let distances = addmm_with_context_exact_native(
            backend,
            &distance_base,
            input,
            &codebook_transposed,
            1.0,
            -2.0,
            execution,
        )
        .map_err(|error| canonical_error("vector_quantize.addmm", error, execution))?;
        let minimum =
            tensor_min_with_context_exact_native(backend, &distances, Some(1), false, execution)
                .map_err(|error| canonical_error("vector_quantize.min", error, execution))?;
        let indices = minimum.indices.ok_or_else(|| {
            AutogradBreadthError::InvalidInput(
                "vector quantization minimum did not return indices".to_owned(),
            )
        })?;
        let output =
            index_select_with_context_exact_native(backend, codebook, 0, &indices, execution)
                .map_err(|error| {
                    canonical_error("vector_quantize.index_select", error, execution)
                })?;
        let mut context = FunctionContext::new(needs_input_grad.to_vec());
        context.save_for_backward(&[&indices, codebook])?;
        context.mark_non_differentiable(1)?;
        Ok((
            Self {
                context,
                output_shape: input_shape,
            },
            output,
            indices,
        ))
    }

    pub fn forward_recorded(
        backend: &CpuBackend,
        tape: &mut AutogradTape,
        input: &Tensor,
        codebook: &Tensor,
        needs_input_grad: [bool; 2],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor, Tensor, Option<OutputSlot>), AutogradBreadthError> {
        let (function, output, indices) =
            Self::forward(backend, input, codebook, needs_input_grad, execution)?;
        let slots = tape.record_operation(
            &[input, codebook],
            &[&output, &indices],
            &[true, false],
            function.context.saved.clone(),
            Arc::new(VectorQuantizeBackwardRule {
                needs_input_grad,
                output_shape: function.output_shape.clone(),
            }),
        )?;
        let slot = slots.and_then(|slots| slots.first().copied());
        Ok((function, output, indices, slot))
    }

    pub fn backward(
        mut self,
        backend: &CpuBackend,
        output_gradients: [Option<&Tensor>; 2],
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 2], AutogradBreadthError> {
        let Some(grad_output) = output_gradients[0] else {
            self.context.release();
            return Ok([None, None]);
        };
        require_gradient_shape(grad_output, &self.output_shape, execution)?;
        let saved = self.context.saved_tensors()?;
        let indices = saved.first().ok_or_else(|| {
            AutogradBreadthError::InvalidInput("missing vector-quantize indices".to_owned())
        })?;
        let codebook = saved.get(1).ok_or_else(|| {
            AutogradBreadthError::InvalidInput("missing vector-quantize codebook".to_owned())
        })?;
        let grad_input = if self.context.needs_input_grad(0) {
            Some(
                clone_with_context_exact_native(
                    backend,
                    grad_output,
                    MemoryFormatReference::PreserveFormat,
                    execution,
                )
                .map_err(|error| {
                    canonical_error("vector_quantize.clone_gradient", error, execution)
                })?,
            )
        } else {
            None
        };
        let grad_codebook = if self.context.needs_input_grad(1) {
            Some(
                index_select_vjp_with_context_exact_native(
                    backend,
                    codebook,
                    0,
                    indices,
                    grad_output,
                    execution,
                )
                .map_err(|error| {
                    canonical_error("vector_quantize.index_select_vjp", error, execution)
                })?,
            )
        } else {
            None
        };
        self.context.release();
        Ok([grad_input, grad_codebook])
    }
}

struct VectorQuantizeBackwardRule {
    needs_input_grad: [bool; 2],
    output_shape: Vec<u64>,
}

impl BackwardRule for VectorQuantizeBackwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("vector_quantize"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "vector_quantize received an invalid output gradient shape".to_owned(),
            });
        }
        let input_gradient = if self.needs_input_grad[0] {
            Some(
                clone_with_context_exact_native(
                    backend,
                    &gradient,
                    MemoryFormatReference::PreserveFormat,
                    execution,
                )
                .map_err(|error| rule_operation_error("vector_quantize_clone", error))?,
            )
        } else {
            None
        };
        let codebook_gradient = if self.needs_input_grad[1] {
            Some(
                index_select_vjp_with_context_exact_native(
                    backend,
                    saved_tensor(saved, 1, "vector_quantize")?,
                    0,
                    saved_tensor(saved, 0, "vector_quantize")?,
                    &gradient,
                    execution,
                )
                .map_err(|error| rule_operation_error("vector_quantize_index_select_vjp", error))?,
            )
        } else {
            None
        };
        Ok(vec![input_gradient, codebook_gradient])
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "vector_quantize"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "vector_quantize received an invalid output gradient shape".to_owned(),
            });
        }
        let input_gradient = if self.needs_input_grad[0] {
            Some(recorded_clone(&gradient, context)?)
        } else {
            None
        };
        let codebook_gradient = if self.needs_input_grad[1] {
            Some(recorded_index_select_vjp(
                saved_tensor(saved, 1, "vector_quantize")?,
                0,
                saved_tensor(saved, 0, "vector_quantize")?,
                &gradient,
                context,
            )?)
        } else {
            None
        };
        Ok(vec![input_gradient, codebook_gradient])
    }
}

pub struct AddAuxLossFunction {
    requires_aux_loss: bool,
    loss_dtype: DType,
    output_shape: Vec<u64>,
    output_dtype: DType,
}

impl AddAuxLossFunction {
    pub fn forward(
        input: &Tensor,
        loss: &Tensor,
        loss_requires_grad: bool,
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor), AutogradBreadthError> {
        require_cpu_execution(input, execution)?;
        require_cpu_execution(loss, execution)?;
        Ok((
            Self {
                requires_aux_loss: loss_requires_grad,
                loss_dtype: loss.descriptor().dtype(),
                output_shape: input.descriptor().shape().to_vec(),
                output_dtype: input.descriptor().dtype(),
            },
            input.clone(),
        ))
    }

    pub fn forward_recorded(
        backend: &CpuBackend,
        tape: &mut AutogradTape,
        input: &Tensor,
        loss: &Tensor,
        loss_requires_grad: bool,
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor, Option<OutputSlot>), AutogradBreadthError> {
        let (function, output) = Self::forward(input, loss, loss_requires_grad, execution)?;
        let auxiliary = if loss_requires_grad {
            let descriptor = TensorDescriptor::contiguous(
                vec![1],
                function.loss_dtype,
                output.descriptor().device(),
                output.descriptor().stream(),
            )?;
            Some(
                backend
                    .fill(crate::Scalar::Float(1.0), descriptor, execution)?
                    .0,
            )
        } else {
            None
        };
        let slots = tape.record_operation(
            &[input, loss],
            &[&output],
            &[true],
            Vec::new(),
            Arc::new(AddAuxLossBackwardRule {
                auxiliary,
                output_shape: function.output_shape.clone(),
                output_dtype: function.output_dtype,
            }),
        )?;
        let slot = slots.and_then(|slots| slots.first().copied());
        Ok((function, output, slot))
    }

    pub fn backward(
        self,
        backend: &CpuBackend,
        grad_output: Option<&Tensor>,
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 2], AutogradBreadthError> {
        let Some(grad_output) = grad_output else {
            return Ok([None, None]);
        };
        require_cpu_execution(grad_output, execution)?;
        if grad_output.descriptor().shape() != self.output_shape
            || grad_output.descriptor().dtype() != self.output_dtype
        {
            return Err(AutogradBreadthError::InvalidInput(
                "AddAuxLoss output gradient descriptor differs from the forward output".to_owned(),
            ));
        }
        let auxiliary = if self.requires_aux_loss {
            let descriptor = TensorDescriptor::contiguous(
                vec![1],
                self.loss_dtype,
                grad_output.descriptor().device(),
                grad_output.descriptor().stream(),
            )?;
            Some(
                backend
                    .fill(crate::Scalar::Float(1.0), descriptor, execution)?
                    .0,
            )
        } else {
            None
        };
        Ok([Some(grad_output.clone()), auxiliary])
    }
}

struct AddAuxLossBackwardRule {
    auxiliary: Option<Tensor>,
    output_shape: Vec<u64>,
    output_dtype: DType,
}

impl BackwardRule for AddAuxLossBackwardRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        cancellation.check().map_err(|_| AutogradError::Cancelled)?;
        let gradient = output_gradients.first().cloned().flatten();
        if let Some(gradient) = &gradient
            && (gradient.descriptor().shape() != self.output_shape
                || gradient.descriptor().dtype() != self.output_dtype)
        {
            return Err(AutogradError::InvalidGraph {
                reason: "AddAuxLoss output gradient descriptor differs from the forward output"
                    .to_owned(),
            });
        }
        Ok(vec![gradient, self.auxiliary.clone()])
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "AddAuxLoss"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        if gradient.descriptor().shape() != self.output_shape
            || gradient.descriptor().dtype() != self.output_dtype
        {
            return Err(AutogradError::InvalidGraph {
                reason: "AddAuxLoss output gradient descriptor differs from the forward output"
                    .to_owned(),
            });
        }
        context.record_operation(
            &[&gradient],
            &[&gradient],
            &[true],
            Vec::new(),
            Arc::new(AnalyticalIdentityRule),
        )?;
        Ok(vec![Some(gradient), self.auxiliary.clone()])
    }
}

struct AnalyticalIdentityRule;

impl BackwardRule for AnalyticalIdentityRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        cancellation.check().map_err(|_| AutogradError::Cancelled)?;
        Ok(vec![output_gradients.first().cloned().flatten()])
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "analytical_identity"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None]);
        };
        context.record_operation(
            &[&gradient],
            &[&gradient],
            &[true],
            Vec::new(),
            Arc::new(Self),
        )?;
        Ok(vec![Some(gradient)])
    }
}

fn contextual_rule_only(symbol: &'static str) -> AutogradError {
    AutogradError::InvalidGraph {
        reason: format!("autograd rule {symbol} requires caller execution context"),
    }
}

fn rule_operation_error(operation: &'static str, error: impl std::fmt::Display) -> AutogradError {
    AutogradError::InvalidGraph {
        reason: format!("canonical higher-order operation {operation} failed: {error}"),
    }
}

fn saved_tensor<'a>(
    saved: &'a [SavedTensor],
    index: usize,
    symbol: &'static str,
) -> Result<&'a Tensor, AutogradError> {
    saved
        .get(index)
        .map(SavedTensor::tensor)
        .ok_or_else(|| AutogradError::InvalidGraph {
            reason: format!("autograd rule {symbol} is missing saved tensor {index}"),
        })
}

fn add_rule_gradients(
    backend: &CpuBackend,
    left: Option<Tensor>,
    right: Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<Option<Tensor>, AutogradError> {
    match left {
        Some(left) => Ok(Some(
            add_method_with_context_exact_native(
                backend,
                &left,
                ElementwiseOperand::Tensor(&right),
                1.0,
                execution,
            )
            .map_err(|error| rule_operation_error("gradient_add", error))?,
        )),
        None => Ok(Some(right)),
    }
}

struct CloneForwardRule;

impl BackwardRule for CloneForwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("clone"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(
            clone_with_context_exact_native(
                backend,
                &gradient,
                MemoryFormatReference::PreserveFormat,
                execution,
            )
            .map_err(|error| rule_operation_error("clone_vjp", error))?,
        )])
    }
}

struct MulForwardRule;

impl BackwardRule for MulForwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("mul"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        let gradients = mul_vjp_with_context_exact_native(
            backend,
            saved_tensor(saved, 0, "mul")?,
            saved_tensor(saved, 1, "mul")?,
            &gradient,
            execution,
        )
        .map_err(|error| rule_operation_error("mul_vjp", error))?;
        Ok(vec![Some(gradients.left), Some(gradients.right)])
    }
}

struct MmForwardRule;

impl BackwardRule for MmForwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("mm"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        let gradients = mm_vjp_with_context_exact_native(
            backend,
            saved_tensor(saved, 0, "mm")?,
            saved_tensor(saved, 1, "mm")?,
            &gradient,
            execution,
        )
        .map_err(|error| rule_operation_error("mm_vjp", error))?;
        Ok(vec![Some(gradients.input), Some(gradients.other)])
    }
}

struct EinsumForwardRule {
    equation: &'static str,
    operand_count: usize,
}

impl BackwardRule for EinsumForwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("einsum"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; self.operand_count]);
        };
        let operands = saved
            .iter()
            .take(self.operand_count)
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        if operands.len() != self.operand_count {
            return Err(AutogradError::GradientArity {
                expected: self.operand_count,
                actual: operands.len(),
            });
        }
        let gradients = einsum_vjp_with_context_exact_native(
            backend,
            self.equation,
            &operands,
            &gradient,
            execution,
        )
        .map_err(|error| rule_operation_error("einsum_vjp", error))?;
        Ok(gradients.operands.into_iter().map(Some).collect())
    }
}

struct MulVjpHigherRule;

impl BackwardRule for MulVjpHigherRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("mul_vjp"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let left = saved_tensor(saved, 0, "mul_vjp")?;
        let right = saved_tensor(saved, 1, "mul_vjp")?;
        let gradient = saved_tensor(saved, 2, "mul_vjp")?;
        let left_seed = output_gradients.first().cloned().flatten();
        let right_seed = output_gradients.get(1).cloned().flatten();
        let mut input_gradient = None;
        let mut right_gradient = None;
        let mut output_gradient = None;
        if let Some(seed) = &right_seed {
            input_gradient = Some(
                mul_vjp_with_context_exact_native(backend, left, seed, gradient, execution)
                    .map_err(|error| rule_operation_error("mul_vjp_hessian_left", error))?
                    .left,
            );
            output_gradient = add_rule_gradients(
                backend,
                output_gradient,
                mul_with_context_exact_native(backend, left, seed, execution)
                    .map_err(|error| rule_operation_error("mul_vjp_hessian_output", error))?,
                execution,
            )?;
        }
        if let Some(seed) = &left_seed {
            right_gradient = Some(
                mul_vjp_with_context_exact_native(backend, seed, right, gradient, execution)
                    .map_err(|error| rule_operation_error("mul_vjp_hessian_right", error))?
                    .right,
            );
            output_gradient = add_rule_gradients(
                backend,
                output_gradient,
                mul_with_context_exact_native(backend, seed, right, execution)
                    .map_err(|error| rule_operation_error("mul_vjp_hessian_output", error))?,
                execution,
            )?;
        }
        Ok(vec![input_gradient, right_gradient, output_gradient])
    }
}

struct MmVjpHigherRule;

impl BackwardRule for MmVjpHigherRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("mm_vjp"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let input = saved_tensor(saved, 0, "mm_vjp")?;
        let other = saved_tensor(saved, 1, "mm_vjp")?;
        let gradient = saved_tensor(saved, 2, "mm_vjp")?;
        let input_seed = output_gradients.first().cloned().flatten();
        let other_seed = output_gradients.get(1).cloned().flatten();
        let mut input_gradient = None;
        let mut other_gradient = None;
        let mut output_gradient = None;
        if let Some(seed) = &other_seed {
            input_gradient = Some(
                mm_vjp_with_context_exact_native(backend, input, seed, gradient, execution)
                    .map_err(|error| rule_operation_error("mm_vjp_hessian_input", error))?
                    .input,
            );
            output_gradient = add_rule_gradients(
                backend,
                output_gradient,
                mm_with_context_exact_native(backend, input, seed, execution)
                    .map_err(|error| rule_operation_error("mm_vjp_hessian_output", error))?,
                execution,
            )?;
        }
        if let Some(seed) = &input_seed {
            other_gradient = Some(
                mm_vjp_with_context_exact_native(backend, seed, other, gradient, execution)
                    .map_err(|error| rule_operation_error("mm_vjp_hessian_other", error))?
                    .other,
            );
            output_gradient = add_rule_gradients(
                backend,
                output_gradient,
                mm_with_context_exact_native(backend, seed, other, execution)
                    .map_err(|error| rule_operation_error("mm_vjp_hessian_output", error))?,
                execution,
            )?;
        }
        Ok(vec![input_gradient, other_gradient, output_gradient])
    }
}

struct EinsumVjpHigherRule {
    equation: &'static str,
    operand_count: usize,
}

impl BackwardRule for EinsumVjpHigherRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("einsum_vjp"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let operands = saved
            .iter()
            .take(self.operand_count)
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        if operands.len() != self.operand_count {
            return Err(AutogradError::GradientArity {
                expected: self.operand_count,
                actual: operands.len(),
            });
        }
        let gradient = saved_tensor(saved, self.operand_count, "einsum_vjp")?;
        let mut input_gradients = vec![None; self.operand_count];
        let mut tangents = Vec::new();
        tangents
            .try_reserve_exact(self.operand_count)
            .map_err(|_| AutogradError::AllocationFailed)?;
        for (index, operand) in operands.iter().enumerate() {
            if let Some(seed) = output_gradients.get(index).cloned().flatten() {
                tangents.push(seed);
            } else {
                tangents.push(
                    backend
                        .fill(
                            crate::Scalar::Float(0.0),
                            operand.descriptor().clone(),
                            execution,
                        )
                        .map_err(|error| rule_operation_error("einsum_zero_tangent", error))?
                        .0,
                );
            }
        }
        for (seed_index, seed) in output_gradients.iter().enumerate() {
            let Some(seed) = seed else {
                continue;
            };
            let mut perturbed = operands.clone();
            perturbed[seed_index] = seed.clone();
            let contribution = einsum_vjp_with_context_exact_native(
                backend,
                self.equation,
                &perturbed,
                gradient,
                execution,
            )
            .map_err(|error| rule_operation_error("einsum_vjp_hessian", error))?;
            for (index, candidate) in contribution.operands.into_iter().enumerate() {
                if index != seed_index {
                    input_gradients[index] = add_rule_gradients(
                        backend,
                        input_gradients[index].take(),
                        candidate,
                        execution,
                    )?;
                }
            }
        }
        let output_gradient = if output_gradients.iter().any(Option::is_some) {
            Some(
                einsum_jvp_with_context_exact_native(
                    backend,
                    self.equation,
                    &operands,
                    &tangents,
                    execution,
                )
                .map_err(|error| rule_operation_error("einsum_vjp_hessian_output", error))?,
            )
        } else {
            None
        };
        input_gradients.push(output_gradient);
        Ok(input_gradients)
    }
}

struct IndexSelectVjpHigherRule {
    dimension: i64,
}

impl BackwardRule for IndexSelectVjpHigherRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("index_select_vjp"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None, None]);
        };
        let selected = index_select_with_context_exact_native(
            backend,
            &gradient,
            self.dimension,
            saved_tensor(saved, 1, "index_select_vjp")?,
            execution,
        )
        .map_err(|error| rule_operation_error("index_select_vjp_hessian", error))?;
        Ok(vec![None, None, Some(selected)])
    }
}

fn recorded_clone(
    input: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let output = clone_with_context_exact_native(
        context.backend(),
        input,
        MemoryFormatReference::PreserveFormat,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("clone", error))?;
    context.record_operation(
        &[input],
        &[&output],
        &[true],
        Vec::new(),
        Arc::new(CloneForwardRule),
    )?;
    Ok(output)
}

fn recorded_mul(
    left: &Tensor,
    right: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let output = mul_with_context_exact_native(context.backend(), left, right, context.execution())
        .map_err(|error| rule_operation_error("mul", error))?;
    context.record_operation(
        &[left, right],
        &[&output],
        &[true],
        vec![SavedTensor::capture(left), SavedTensor::capture(right)],
        Arc::new(MulForwardRule),
    )?;
    Ok(output)
}

fn recorded_mul_vjp(
    left: &Tensor,
    right: &Tensor,
    gradient: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<BinaryGradients, AutogradError> {
    let gradients = mul_vjp_with_context_exact_native(
        context.backend(),
        left,
        right,
        gradient,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("mul_vjp", error))?;
    context.record_operation(
        &[left, right, gradient],
        &[&gradients.left, &gradients.right],
        &[true, true],
        vec![
            SavedTensor::capture(left),
            SavedTensor::capture(right),
            SavedTensor::capture(gradient),
        ],
        Arc::new(MulVjpHigherRule),
    )?;
    Ok(gradients)
}

fn recorded_mm(
    input: &Tensor,
    other: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let output = mm_with_context_exact_native(context.backend(), input, other, context.execution())
        .map_err(|error| rule_operation_error("mm", error))?;
    context.record_operation(
        &[input, other],
        &[&output],
        &[true],
        vec![SavedTensor::capture(input), SavedTensor::capture(other)],
        Arc::new(MmForwardRule),
    )?;
    Ok(output)
}

fn recorded_mm_vjp(
    input: &Tensor,
    other: &Tensor,
    gradient: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<MatmulGradients, AutogradError> {
    let gradients = mm_vjp_with_context_exact_native(
        context.backend(),
        input,
        other,
        gradient,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("mm_vjp", error))?;
    context.record_operation(
        &[input, other, gradient],
        &[&gradients.input, &gradients.other],
        &[true, true],
        vec![
            SavedTensor::capture(input),
            SavedTensor::capture(other),
            SavedTensor::capture(gradient),
        ],
        Arc::new(MmVjpHigherRule),
    )?;
    Ok(gradients)
}

fn recorded_einsum(
    equation: &'static str,
    operands: &[Tensor],
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let output = einsum_with_context_exact_native(
        context.backend(),
        equation,
        operands,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("einsum", error))?;
    let input_refs = operands.iter().collect::<Vec<_>>();
    context.record_operation(
        &input_refs,
        &[&output],
        &[true],
        operands.iter().map(SavedTensor::capture).collect(),
        Arc::new(EinsumForwardRule {
            equation,
            operand_count: operands.len(),
        }),
    )?;
    Ok(output)
}

fn recorded_einsum_vjp(
    equation: &'static str,
    operands: &[Tensor],
    gradient: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<EinsumGradients, AutogradError> {
    let gradients = einsum_vjp_with_context_exact_native(
        context.backend(),
        equation,
        operands,
        gradient,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("einsum_vjp", error))?;
    let mut inputs = operands.iter().collect::<Vec<_>>();
    inputs.push(gradient);
    let outputs = gradients.operands.iter().collect::<Vec<_>>();
    let mut saved = operands
        .iter()
        .map(SavedTensor::capture)
        .collect::<Vec<_>>();
    saved.push(SavedTensor::capture(gradient));
    context.record_operation(
        &inputs,
        &outputs,
        &vec![true; outputs.len()],
        saved,
        Arc::new(EinsumVjpHigherRule {
            equation,
            operand_count: operands.len(),
        }),
    )?;
    Ok(gradients)
}

fn recorded_index_select_vjp(
    input: &Tensor,
    dimension: i64,
    indices: &Tensor,
    gradient: &Tensor,
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let output = index_select_vjp_with_context_exact_native(
        context.backend(),
        input,
        dimension,
        indices,
        gradient,
        context.execution(),
    )
    .map_err(|error| rule_operation_error("index_select_vjp", error))?;
    context.record_operation(
        &[input, indices, gradient],
        &[&output],
        &[true],
        vec![
            SavedTensor::capture(input),
            SavedTensor::capture(indices),
            SavedTensor::capture(gradient),
        ],
        Arc::new(IndexSelectVjpHigherRule { dimension }),
    )?;
    Ok(output)
}

pub struct HadaWeightFunction {
    context: FunctionContext,
    output_shape: Vec<u64>,
}

impl HadaWeightFunction {
    pub fn forward(
        backend: &CpuBackend,
        factors: [&Tensor; 4],
        scale: &Tensor,
        needs_input_grad: [bool; 5],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor), AutogradBreadthError> {
        for factor in factors {
            require_f32_execution(factor, execution)?;
        }
        require_f32_execution(scale, execution)?;
        if scale.descriptor().element_count()? != 1 {
            return Err(AutogradBreadthError::InvalidInput(
                "HadaWeight scale must contain exactly one value".to_owned(),
            ));
        }
        let first = mm_with_context_exact_native(backend, factors[0], factors[1], execution)
            .map_err(|error| canonical_error("HadaWeight.first_mm", error, execution))?;
        let second = mm_with_context_exact_native(backend, factors[2], factors[3], execution)
            .map_err(|error| canonical_error("HadaWeight.second_mm", error, execution))?;
        let product = mul_with_context_exact_native(backend, &first, &second, execution)
            .map_err(|error| canonical_error("HadaWeight.product", error, execution))?;
        let output = mul_with_context_exact_native(backend, &product, scale, execution)
            .map_err(|error| canonical_error("HadaWeight.scale", error, execution))?;
        let output_shape = output.descriptor().shape().to_vec();
        let mut context = FunctionContext::new(needs_input_grad.to_vec());
        context.save_for_backward(&[factors[0], factors[1], factors[2], factors[3], scale])?;
        Ok((
            Self {
                context,
                output_shape,
            },
            output,
        ))
    }

    pub fn forward_recorded(
        backend: &CpuBackend,
        tape: &mut AutogradTape,
        factors: [&Tensor; 4],
        scale: &Tensor,
        needs_input_grad: [bool; 5],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor, Option<OutputSlot>), AutogradBreadthError> {
        let (function, output) =
            Self::forward(backend, factors, scale, needs_input_grad, execution)?;
        let slots = tape.record_operation(
            &[factors[0], factors[1], factors[2], factors[3], scale],
            &[&output],
            &[true],
            function.context.saved.clone(),
            Arc::new(HadaWeightBackwardRule {
                needs_input_grad,
                output_shape: function.output_shape.clone(),
            }),
        )?;
        Ok((
            function,
            output,
            slots.and_then(|slots| slots.first().copied()),
        ))
    }

    pub fn backward(
        mut self,
        backend: &CpuBackend,
        grad_output: Option<&Tensor>,
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 5], AutogradBreadthError> {
        let Some(grad_output) = grad_output else {
            self.context.release();
            return Ok([None, None, None, None, None]);
        };
        require_gradient_shape(grad_output, &self.output_shape, execution)?;
        let saved = self.context.saved_tensors()?;
        if saved.len() != 5 {
            return Err(AutogradBreadthError::GradientArity {
                expected: 5,
                actual: saved.len(),
            });
        }
        let candidates = hada_weight_vjp(backend, &saved, grad_output, execution)?;
        let mut result: [Option<Tensor>; 5] = [None, None, None, None, None];
        for (index, candidate) in candidates.into_iter().enumerate() {
            if self.context.needs_input_grad(index) {
                result[index] = Some(candidate);
            }
        }
        self.context.release();
        Ok(result)
    }
}

struct HadaWeightBackwardRule {
    needs_input_grad: [bool; 5],
    output_shape: Vec<u64>,
}

impl BackwardRule for HadaWeightBackwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("HadaWeight"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; 5]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "HadaWeight received an invalid output gradient shape".to_owned(),
            });
        }
        let tensors = saved
            .iter()
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        let candidates = hada_weight_vjp(backend, &tensors, &gradient, execution)?;
        Ok(candidates
            .into_iter()
            .enumerate()
            .map(|(index, gradient)| self.needs_input_grad[index].then_some(gradient))
            .collect())
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "HadaWeight"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; 5]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "HadaWeight received an invalid output gradient shape".to_owned(),
            });
        }
        let tensors = saved
            .iter()
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        if tensors.len() != 5 {
            return Err(AutogradError::GradientArity {
                expected: 5,
                actual: tensors.len(),
            });
        }
        let first = recorded_mm(&tensors[0], &tensors[1], context)?;
        let second = recorded_mm(&tensors[2], &tensors[3], context)?;
        let product = recorded_mul(&first, &second, context)?;
        let scaled_vjp = recorded_mul_vjp(&product, &tensors[4], &gradient, context)?;
        let product_vjp = recorded_mul_vjp(&first, &second, &scaled_vjp.left, context)?;
        let first_vjp = recorded_mm_vjp(&tensors[0], &tensors[1], &product_vjp.left, context)?;
        let second_vjp = recorded_mm_vjp(&tensors[2], &tensors[3], &product_vjp.right, context)?;
        let candidates = [
            first_vjp.input,
            first_vjp.other,
            second_vjp.input,
            second_vjp.other,
            scaled_vjp.right,
        ];
        Ok(candidates
            .into_iter()
            .enumerate()
            .map(|(index, gradient)| self.needs_input_grad[index].then_some(gradient))
            .collect())
    }
}

fn hada_weight_vjp(
    backend: &CpuBackend,
    saved: &[Tensor],
    gradient: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<[Tensor; 5], AutogradError> {
    if saved.len() != 5 {
        return Err(AutogradError::GradientArity {
            expected: 5,
            actual: saved.len(),
        });
    }
    let first = mm_with_context_exact_native(backend, &saved[0], &saved[1], execution)
        .map_err(|error| rule_operation_error("HadaWeight.first_mm", error))?;
    let second = mm_with_context_exact_native(backend, &saved[2], &saved[3], execution)
        .map_err(|error| rule_operation_error("HadaWeight.second_mm", error))?;
    let product = mul_with_context_exact_native(backend, &first, &second, execution)
        .map_err(|error| rule_operation_error("HadaWeight.product", error))?;
    let scaled_vjp =
        mul_vjp_with_context_exact_native(backend, &product, &saved[4], gradient, execution)
            .map_err(|error| rule_operation_error("HadaWeight.scale_vjp", error))?;
    let product_vjp =
        mul_vjp_with_context_exact_native(backend, &first, &second, &scaled_vjp.left, execution)
            .map_err(|error| rule_operation_error("HadaWeight.product_vjp", error))?;
    let first_vjp = mm_vjp_with_context_exact_native(
        backend,
        &saved[0],
        &saved[1],
        &product_vjp.left,
        execution,
    )
    .map_err(|error| rule_operation_error("HadaWeight.first_mm_vjp", error))?;
    let second_vjp = mm_vjp_with_context_exact_native(
        backend,
        &saved[2],
        &saved[3],
        &product_vjp.right,
        execution,
    )
    .map_err(|error| rule_operation_error("HadaWeight.second_mm_vjp", error))?;
    Ok([
        first_vjp.input,
        first_vjp.other,
        second_vjp.input,
        second_vjp.other,
        scaled_vjp.right,
    ])
}

pub struct HadaWeightTuckerFunction {
    context: FunctionContext,
    output_shape: Vec<u64>,
}

impl HadaWeightTuckerFunction {
    pub fn forward(
        backend: &CpuBackend,
        tensors: [&Tensor; 6],
        scale: &Tensor,
        needs_input_grad: [bool; 7],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor), AutogradBreadthError> {
        for tensor in tensors {
            require_f32_execution(tensor, execution)?;
        }
        require_f32_execution(scale, execution)?;
        if scale.descriptor().element_count()? != 1 {
            return Err(AutogradBreadthError::InvalidInput(
                "HadaWeightTucker scale must contain exactly one value".to_owned(),
            ));
        }
        let first = einsum_with_context_exact_native(
            backend,
            "ij...,jr,ip->pr...",
            &[tensors[0].clone(), tensors[2].clone(), tensors[1].clone()],
            execution,
        )
        .map_err(|error| canonical_error("HadaWeightTucker.first_einsum", error, execution))?;
        let second = einsum_with_context_exact_native(
            backend,
            "ij...,jr,ip->pr...",
            &[tensors[3].clone(), tensors[5].clone(), tensors[4].clone()],
            execution,
        )
        .map_err(|error| canonical_error("HadaWeightTucker.second_einsum", error, execution))?;
        let product = mul_with_context_exact_native(backend, &first, &second, execution)
            .map_err(|error| canonical_error("HadaWeightTucker.product", error, execution))?;
        let output = mul_with_context_exact_native(backend, &product, scale, execution)
            .map_err(|error| canonical_error("HadaWeightTucker.scale", error, execution))?;
        let output_shape = output.descriptor().shape().to_vec();
        let mut context = FunctionContext::new(needs_input_grad.to_vec());
        context.save_for_backward(&[
            tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], scale,
        ])?;
        Ok((
            Self {
                context,
                output_shape,
            },
            output,
        ))
    }

    pub fn forward_recorded(
        backend: &CpuBackend,
        tape: &mut AutogradTape,
        tensors: [&Tensor; 6],
        scale: &Tensor,
        needs_input_grad: [bool; 7],
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor, Option<OutputSlot>), AutogradBreadthError> {
        let (function, output) =
            Self::forward(backend, tensors, scale, needs_input_grad, execution)?;
        let slots = tape.record_operation(
            &[
                tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], scale,
            ],
            &[&output],
            &[true],
            function.context.saved.clone(),
            Arc::new(HadaWeightTuckerBackwardRule {
                needs_input_grad,
                output_shape: function.output_shape.clone(),
            }),
        )?;
        Ok((
            function,
            output,
            slots.and_then(|slots| slots.first().copied()),
        ))
    }

    pub fn backward(
        mut self,
        backend: &CpuBackend,
        grad_output: Option<&Tensor>,
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 7], AutogradBreadthError> {
        let Some(grad_output) = grad_output else {
            self.context.release();
            return Ok([None, None, None, None, None, None, None]);
        };
        require_gradient_shape(grad_output, &self.output_shape, execution)?;
        let saved = self.context.saved_tensors()?;
        if saved.len() != 7 {
            return Err(AutogradBreadthError::GradientArity {
                expected: 7,
                actual: saved.len(),
            });
        }
        let candidates = hada_weight_tucker_vjp(backend, &saved, grad_output, execution)?;
        let mut result: [Option<Tensor>; 7] = [None, None, None, None, None, None, None];
        for (index, candidate) in candidates.into_iter().enumerate() {
            if self.context.needs_input_grad(index) {
                result[index] = Some(candidate);
            }
        }
        self.context.release();
        Ok(result)
    }
}

const HADA_TUCKER_EQUATION: &str = "ij...,jr,ip->pr...";

struct HadaWeightTuckerBackwardRule {
    needs_input_grad: [bool; 7],
    output_shape: Vec<u64>,
}

impl BackwardRule for HadaWeightTuckerBackwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &crate::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(contextual_rule_only("HadaWeightTucker"))
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; 7]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "HadaWeightTucker received an invalid output gradient shape".to_owned(),
            });
        }
        let tensors = saved
            .iter()
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        let candidates = hada_weight_tucker_vjp(backend, &tensors, &gradient, execution)?;
        Ok(candidates
            .into_iter()
            .enumerate()
            .map(|(index, gradient)| self.needs_input_grad[index].then_some(gradient))
            .collect())
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "HadaWeightTucker"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; 7]);
        };
        if gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "HadaWeightTucker received an invalid output gradient shape".to_owned(),
            });
        }
        let tensors = saved
            .iter()
            .map(|saved| saved.tensor().clone())
            .collect::<Vec<_>>();
        if tensors.len() != 7 {
            return Err(AutogradError::GradientArity {
                expected: 7,
                actual: tensors.len(),
            });
        }
        let first_operands = [tensors[0].clone(), tensors[2].clone(), tensors[1].clone()];
        let second_operands = [tensors[3].clone(), tensors[5].clone(), tensors[4].clone()];
        let first = recorded_einsum(HADA_TUCKER_EQUATION, &first_operands, context)?;
        let second = recorded_einsum(HADA_TUCKER_EQUATION, &second_operands, context)?;
        let product = recorded_mul(&first, &second, context)?;
        let scaled_vjp = recorded_mul_vjp(&product, &tensors[6], &gradient, context)?;
        let product_vjp = recorded_mul_vjp(&first, &second, &scaled_vjp.left, context)?;
        let first_gradients = recorded_einsum_vjp(
            HADA_TUCKER_EQUATION,
            &first_operands,
            &product_vjp.left,
            context,
        )?;
        let second_gradients = recorded_einsum_vjp(
            HADA_TUCKER_EQUATION,
            &second_operands,
            &product_vjp.right,
            context,
        )?;
        if first_gradients.operands.len() != 3 || second_gradients.operands.len() != 3 {
            return Err(AutogradError::GradientArity {
                expected: 6,
                actual: first_gradients.operands.len() + second_gradients.operands.len(),
            });
        }
        let candidates = [
            first_gradients.operands[0].clone(),
            first_gradients.operands[2].clone(),
            first_gradients.operands[1].clone(),
            second_gradients.operands[0].clone(),
            second_gradients.operands[2].clone(),
            second_gradients.operands[1].clone(),
            scaled_vjp.right,
        ];
        Ok(candidates
            .into_iter()
            .enumerate()
            .map(|(index, gradient)| self.needs_input_grad[index].then_some(gradient))
            .collect())
    }
}

fn hada_weight_tucker_vjp(
    backend: &CpuBackend,
    saved: &[Tensor],
    gradient: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<[Tensor; 7], AutogradError> {
    if saved.len() != 7 {
        return Err(AutogradError::GradientArity {
            expected: 7,
            actual: saved.len(),
        });
    }
    let first_operands = [saved[0].clone(), saved[2].clone(), saved[1].clone()];
    let second_operands = [saved[3].clone(), saved[5].clone(), saved[4].clone()];
    let first =
        einsum_with_context_exact_native(backend, HADA_TUCKER_EQUATION, &first_operands, execution)
            .map_err(|error| rule_operation_error("HadaWeightTucker.first_einsum", error))?;
    let second = einsum_with_context_exact_native(
        backend,
        HADA_TUCKER_EQUATION,
        &second_operands,
        execution,
    )
    .map_err(|error| rule_operation_error("HadaWeightTucker.second_einsum", error))?;
    let product = mul_with_context_exact_native(backend, &first, &second, execution)
        .map_err(|error| rule_operation_error("HadaWeightTucker.product", error))?;
    let scaled_vjp =
        mul_vjp_with_context_exact_native(backend, &product, &saved[6], gradient, execution)
            .map_err(|error| rule_operation_error("HadaWeightTucker.scale_vjp", error))?;
    let product_vjp =
        mul_vjp_with_context_exact_native(backend, &first, &second, &scaled_vjp.left, execution)
            .map_err(|error| rule_operation_error("HadaWeightTucker.product_vjp", error))?;
    let first_gradients = einsum_vjp_with_context_exact_native(
        backend,
        HADA_TUCKER_EQUATION,
        &first_operands,
        &product_vjp.left,
        execution,
    )
    .map_err(|error| rule_operation_error("HadaWeightTucker.first_einsum_vjp", error))?;
    let second_gradients = einsum_vjp_with_context_exact_native(
        backend,
        HADA_TUCKER_EQUATION,
        &second_operands,
        &product_vjp.right,
        execution,
    )
    .map_err(|error| rule_operation_error("HadaWeightTucker.second_einsum_vjp", error))?;
    if first_gradients.operands.len() != 3 || second_gradients.operands.len() != 3 {
        return Err(AutogradError::GradientArity {
            expected: 6,
            actual: first_gradients.operands.len() + second_gradients.operands.len(),
        });
    }
    Ok([
        first_gradients.operands[0].clone(),
        first_gradients.operands[2].clone(),
        first_gradients.operands[1].clone(),
        second_gradients.operands[0].clone(),
        second_gradients.operands[2].clone(),
        second_gradients.operands[1].clone(),
        scaled_vjp.right,
    ])
}

pub trait CheckpointCallable: Send + Sync {
    fn forward(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError>;
    fn recompute_vjp(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        parameters: &[Tensor],
        output_gradients: &[Option<Tensor>],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError>;
}

pub struct CheckpointFunction {
    callable: Option<Arc<dyn CheckpointCallable>>,
    execution: CheckpointExecution,
    input_count: usize,
}

impl CheckpointFunction {
    pub fn forward(
        backend: &CpuBackend,
        callable: Arc<dyn CheckpointCallable>,
        inputs: &[Tensor],
        parameters: &[Tensor],
        needs_input_grad: Vec<bool>,
        autocast: AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Vec<Tensor>), AutogradBreadthError> {
        check_execution(execution)?;
        if needs_input_grad.len() != inputs.len() + parameters.len() {
            return Err(AutogradBreadthError::GradientArity {
                expected: inputs.len() + parameters.len(),
                actual: needs_input_grad.len(),
            });
        }
        for tensor in inputs.iter().chain(parameters) {
            require_cpu_execution(tensor, execution)?;
        }
        let outputs =
            callable.forward(backend, inputs, GradientMode::NoGrad, &autocast, execution)?;
        if outputs.is_empty() {
            return Err(AutogradBreadthError::InvalidInput(
                "checkpoint callable returned no outputs".to_owned(),
            ));
        }
        for output in &outputs {
            require_cpu_execution(output, execution)?;
        }
        let saved = inputs.iter().chain(parameters).cloned().collect::<Vec<_>>();
        let checkpoint_execution = checkpoint_execution_from_outputs_exact_native(
            &saved,
            outputs.clone(),
            needs_input_grad,
            true,
            autocast,
            execution.cancellation,
        )
        .map_err(|error| checkpoint_error("CheckpointFunction.capture", error, execution))?;
        Ok((
            Self {
                callable: Some(callable),
                execution: checkpoint_execution,
                input_count: inputs.len(),
            },
            outputs,
        ))
    }

    pub fn backward(
        self,
        backend: &CpuBackend,
        output_gradients: &[Option<Tensor>],
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        self.backward_with_options(backend, output_gradients, false, execution)
    }

    pub fn backward_with_options(
        mut self,
        backend: &CpuBackend,
        output_gradients: &[Option<Tensor>],
        create_graph: bool,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        if create_graph {
            return Err(AutogradBreadthError::HigherOrderUnavailable {
                symbol: "CheckpointFunction",
                policy: HigherOrderPolicy::FirstOrderOnly,
            });
        }
        if output_gradients.len() != self.execution.outputs().len() {
            return Err(AutogradBreadthError::GradientArity {
                expected: self.execution.outputs().len(),
                actual: output_gradients.len(),
            });
        }
        for (gradient, descriptor) in output_gradients
            .iter()
            .zip(self.execution.outputs().iter().map(Tensor::descriptor))
        {
            if let Some(gradient) = gradient {
                require_gradient_descriptor(gradient, descriptor, execution)?;
            }
        }
        let saved = self
            .execution
            .shallow_recompute_inputs_exact_native(execution.cancellation)
            .map_err(|error| {
                checkpoint_error("CheckpointFunction.shallow_inputs", error, execution)
            })?;
        let callable = self.callable.take().ok_or_else(|| {
            AutogradBreadthError::InvalidInput(
                "checkpoint callable has already been released".to_owned(),
            )
        })?;
        let recompute_inputs = saved.get(..self.input_count).ok_or_else(|| {
            AutogradBreadthError::InvalidInput(
                "checkpoint input partition exceeds saved tensor arity".to_owned(),
            )
        })?;
        let recompute_parameters = saved.get(self.input_count..).ok_or_else(|| {
            AutogradBreadthError::InvalidInput(
                "checkpoint parameter partition exceeds saved tensor arity".to_owned(),
            )
        })?;
        let mut gradients = callable.recompute_vjp(
            backend,
            recompute_inputs,
            recompute_parameters,
            output_gradients,
            self.execution.recompute_mode(),
            self.execution.autocast(),
            execution,
        )?;
        drop(callable);
        if gradients.len() != saved.len() {
            return Err(AutogradBreadthError::GradientArity {
                expected: saved.len(),
                actual: gradients.len(),
            });
        }
        for (index, gradient) in gradients.iter_mut().enumerate() {
            if !self.execution.needs_input_grad(index) {
                *gradient = None;
            }
        }
        check_execution(execution)?;
        self.execution.release();
        Ok(gradients)
    }

    pub fn backward_source_arity(
        self,
        backend: &CpuBackend,
        output_gradients: &[Option<Tensor>],
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        let gradients = self.backward(backend, output_gradients, execution)?;
        let mut source_gradients = Vec::new();
        source_gradients
            .try_reserve_exact(gradients.len() + 2)
            .map_err(|_| {
                AutogradBreadthError::InvalidInput(
                    "checkpoint source gradient arity overflowed".to_owned(),
                )
            })?;
        source_gradients.push(None);
        source_gradients.push(None);
        source_gradients.extend(gradients);
        Ok(source_gradients)
    }
}

pub struct OffloadCheckpointFunction {
    inner: CheckpointFunction,
}

impl OffloadCheckpointFunction {
    pub fn forward(
        backend: &CpuBackend,
        callable: Arc<dyn CheckpointCallable>,
        input: &Tensor,
        needs_input_grad: bool,
        autocast: AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<(Self, Tensor), AutogradBreadthError> {
        let (inner, outputs) = CheckpointFunction::forward(
            backend,
            callable,
            std::slice::from_ref(input),
            &[],
            vec![needs_input_grad],
            autocast,
            execution,
        )?;
        let output = outputs.into_iter().next().ok_or_else(|| {
            AutogradBreadthError::InvalidInput(
                "offload checkpoint callable returned no output".to_owned(),
            )
        })?;
        Ok((Self { inner }, output))
    }

    pub fn backward(
        self,
        backend: &CpuBackend,
        gradient: Option<Tensor>,
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 2], AutogradBreadthError> {
        self.backward_with_options(backend, gradient, false, execution)
    }

    pub fn backward_with_options(
        self,
        backend: &CpuBackend,
        gradient: Option<Tensor>,
        create_graph: bool,
        execution: &ExecutionContext<'_>,
    ) -> Result<[Option<Tensor>; 2], AutogradBreadthError> {
        if create_graph {
            return Err(AutogradBreadthError::HigherOrderUnavailable {
                symbol: "OffloadCheckpointFunction",
                policy: HigherOrderPolicy::FirstOrderOnly,
            });
        }
        let gradients = self.inner.backward(backend, &[gradient], execution)?;
        Ok([gradients.into_iter().next().flatten(), None])
    }
}

fn require_rank(tensor: &Tensor, rank: usize) -> Result<Vec<u64>, AutogradBreadthError> {
    require_f32_cpu(tensor)?;
    if tensor.descriptor().shape().len() != rank {
        return Err(AutogradBreadthError::InvalidInput(format!(
            "expected rank {rank}, received {}",
            tensor.descriptor().shape().len()
        )));
    }
    Ok(tensor.descriptor().shape().to_vec())
}

fn require_f32_cpu(tensor: &Tensor) -> Result<(), AutogradBreadthError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(AutogradBreadthError::UnsupportedDevice {
            operation: "custom autograd function",
            device: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(AutogradBreadthError::UnsupportedDType {
            operation: "custom autograd function",
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_cpu_execution(
    tensor: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<(), AutogradBreadthError> {
    check_execution(execution)?;
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(AutogradBreadthError::UnsupportedDevice {
            operation: "custom autograd function",
            device: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().stream() != execution.stream {
        return Err(TensorError::StreamMismatch {
            expected: execution.stream,
            actual: tensor.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_f32_execution(
    tensor: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<(), AutogradBreadthError> {
    require_cpu_execution(tensor, execution)?;
    require_f32_cpu(tensor)
}

fn check_execution(execution: &ExecutionContext<'_>) -> Result<(), AutogradBreadthError> {
    match execution.check() {
        Ok(()) => Ok(()),
        Err(TensorError::Cancelled) => Err(AutogradBreadthError::Cancelled),
        Err(error) => Err(AutogradBreadthError::Tensor(error)),
    }
}

fn require_gradient_shape(
    gradient: &Tensor,
    expected_shape: &[u64],
    execution: &ExecutionContext<'_>,
) -> Result<(), AutogradBreadthError> {
    require_f32_execution(gradient, execution)?;
    if gradient.descriptor().shape() != expected_shape {
        return Err(AutogradBreadthError::InvalidInput(format!(
            "gradient shape must be {expected_shape:?}, received {:?}",
            gradient.descriptor().shape()
        )));
    }
    Ok(())
}

fn require_gradient_descriptor(
    gradient: &Tensor,
    expected: &TensorDescriptor,
    execution: &ExecutionContext<'_>,
) -> Result<(), AutogradBreadthError> {
    require_cpu_execution(gradient, execution)?;
    if gradient.descriptor().shape() != expected.shape()
        || gradient.descriptor().dtype() != expected.dtype()
        || gradient.descriptor().device() != expected.device()
    {
        return Err(AutogradBreadthError::InvalidInput(
            "checkpoint output gradient descriptor differs from the forward output".to_owned(),
        ));
    }
    Ok(())
}

fn canonical_error(
    operation: &'static str,
    error: impl std::fmt::Display,
    execution: &ExecutionContext<'_>,
) -> AutogradBreadthError {
    if execution.cancellation.is_cancelled() {
        AutogradBreadthError::Cancelled
    } else {
        AutogradBreadthError::CanonicalOperation {
            operation,
            reason: error.to_string(),
        }
    }
}

fn checkpoint_error(
    operation: &'static str,
    error: ElementwiseRuntimePartSixError,
    execution: &ExecutionContext<'_>,
) -> AutogradBreadthError {
    if execution.cancellation.is_cancelled() {
        return AutogradBreadthError::Cancelled;
    }
    match error {
        ElementwiseRuntimePartSixError::Cancelled
        | ElementwiseRuntimePartSixError::Tensor(TensorError::Cancelled) => {
            AutogradBreadthError::Cancelled
        }
        ElementwiseRuntimePartSixError::Tensor(error) => AutogradBreadthError::Tensor(error),
        ElementwiseRuntimePartSixError::Autograd(error) => AutogradBreadthError::Autograd(error),
        ElementwiseRuntimePartSixError::UnsupportedDevice { device, .. } => {
            AutogradBreadthError::UnsupportedDevice { operation, device }
        }
        ElementwiseRuntimePartSixError::UnsupportedDType { dtype, .. } => {
            AutogradBreadthError::UnsupportedDType { operation, dtype }
        }
        error => AutogradBreadthError::CanonicalOperation {
            operation,
            reason: error.to_string(),
        },
    }
}
