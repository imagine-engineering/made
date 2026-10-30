use crate::{
    MappedModelWeights, PatchComputeBoundary, PatchGraph, PatchGraphError, PatchPayload,
    PatchTensor, PatchValueTransform, QuantizationError, QuantizedLinearMatrix, QuantizedMatrix,
    SemanticPatchOperation,
};
use sim_tensor::{
    AutogradError, AutogradTape, BackwardRule, BinaryOperation, ConvolutionSpec, CpuBackend, DType,
    DecodedScalar, DeviceId, ExecutionContext, HigherOrderContext, IndexSpec, Layout,
    LinearAlgebraOperation, OutputSlot, ReductionOperation, ReductionSpec, RngError,
    RngTransaction, SavedTensor, Scalar, ScalarSide, Tensor, TensorBackend, TensorDescriptor,
    TensorError, UnaryOperation, ViewAccess,
    autograd::breadth::{
        AutogradBreadthError, HadaWeightFunction, HadaWeightTuckerFunction, HigherOrderPolicy,
    },
    generated_elementwise_or_runtime_operation_09::mul_vjp_with_context_exact_native,
    generated_elementwise_or_runtime_operation_21::kron_vjp_with_context_exact_native,
    generated_linear_algebra_01::{
        einsum_vjp_with_context_exact_native, einsum_with_context_exact_native,
        inverse_vjp_with_context_exact_native, mm_vjp_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, tensor_from_f32_with_backend_exact_native,
        tensor_to_f32_with_backend_exact_native,
    },
};
use sim_types::DeviceKind;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use thiserror::Error;

pub const WEIGHT_ADAPTER_ORDER: [AdapterFamily; 6] = [
    AdapterFamily::Lora,
    AdapterFamily::Loha,
    AdapterFamily::Lokr,
    AdapterFamily::Glora,
    AdapterFamily::Oft,
    AdapterFamily::Boft,
];

pub const ADAPTER_MAP_ORDER: [(&str, AdapterFamily); 4] = [
    ("LoRA", AdapterFamily::Lora),
    ("LoHa", AdapterFamily::Loha),
    ("LoKr", AdapterFamily::Lokr),
    ("OFT", AdapterFamily::Oft),
];

const MAX_MODULE_KEY_BYTES: usize = 64 * 1024;
const MAX_ADAPTER_BINDINGS: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdapterFamily {
    Lora,
    Loha,
    Lokr,
    Glora,
    Oft,
    Boft,
}

impl AdapterFamily {
    pub const fn source_class(self) -> &'static str {
        match self {
            Self::Lora => "LoRAAdapter",
            Self::Loha => "LoHaAdapter",
            Self::Lokr => "LoKrAdapter",
            Self::Glora => "GLoRAAdapter",
            Self::Oft => "OFTAdapter",
            Self::Boft => "BOFTAdapter",
        }
    }

    pub const fn source_name(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Loha => "loha",
            Self::Lokr => "lokr",
            Self::Glora => "glora",
            Self::Oft => "oft",
            Self::Boft => "boft",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainableAdapterKind {
    LoraDiff,
    LohaDiff,
    LokrDiff,
    OftDiff,
}

#[derive(Clone, Debug)]
pub struct TrainableWeightOutput {
    output: Tensor,
    output_slot: Option<OutputSlot>,
}

impl TrainableWeightOutput {
    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn output_slot(&self) -> Option<OutputSlot> {
        self.output_slot
    }

    pub fn into_output(self) -> Tensor {
        self.output
    }
}

#[derive(Clone, Debug)]
enum TrainableBackwardPlan {
    Lora {
        alpha: f32,
        rank: u64,
        has_mid: bool,
        difference_shape: Vec<u64>,
    },
    Loha {
        alpha: f32,
        rank: u64,
        tucker: bool,
        difference_shape: Vec<u64>,
    },
    Lokr {
        first: LokrFactorPlan,
        second: LokrFactorPlan,
        difference_shape: Vec<u64>,
    },
    Oft {
        constraint: f32,
        has_rescale: bool,
        raw_norm: f32,
        constrained: bool,
        difference_shape: Vec<u64>,
    },
}

#[derive(Clone, Debug)]
enum LokrFactorPlan {
    Direct {
        input: usize,
    },
    Matrix {
        up: usize,
        down: usize,
        alpha: f32,
        rank: u64,
    },
    Tucker {
        tucker: usize,
        up: usize,
        down: usize,
        alpha: f32,
        rank: u64,
    },
}

struct TrainableWeightBackwardRule {
    plan: TrainableBackwardPlan,
    input_count: usize,
    output_shape: Vec<u64>,
}

struct TrainableAddBackwardRule {
    base_shape: Vec<u64>,
    difference_shape: Vec<u64>,
}

struct TrainableReshapeBackwardRule {
    input_shape: Vec<u64>,
}

impl TrainableReshapeBackwardRule {
    fn gradient(
        &self,
        output_gradients: &[Option<Tensor>],
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(reshape_autograd(&gradient, &self.input_shape)?)])
    }
}

impl BackwardRule for TrainableReshapeBackwardRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &sim_types::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        self.gradient(output_gradients)
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "WeightAdapterReshape"
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
        Ok(vec![Some(record_trainable_reshape(
            &gradient,
            &self.input_shape,
            context,
        )?)])
    }
}

impl TrainableAddBackwardRule {
    fn gradients(
        &self,
        output_gradients: &[Option<Tensor>],
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        Ok(vec![
            Some(reshape_autograd(&gradient, &self.base_shape)?),
            Some(reshape_autograd(&gradient, &self.difference_shape)?),
        ])
    }
}

impl BackwardRule for TrainableAddBackwardRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &sim_types::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        self.gradients(output_gradients)
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "WeightAdapterTrainAdd"
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
        Ok(vec![
            Some(record_trainable_reshape(
                &gradient,
                &self.base_shape,
                context,
            )?),
            Some(record_trainable_reshape(
                &gradient,
                &self.difference_shape,
                context,
            )?),
        ])
    }
}

fn record_trainable_reshape(
    input: &Tensor,
    shape: &[u64],
    context: &mut HigherOrderContext<'_, '_>,
) -> Result<Tensor, AutogradError> {
    let input_shape = input.descriptor().shape().to_vec();
    let output = reshape_autograd(input, shape)?;
    context.record_operation(
        &[input],
        &[&output],
        &[true],
        Vec::new(),
        Arc::new(TrainableReshapeBackwardRule { input_shape }),
    )?;
    Ok(output)
}

impl TrainableBackwardPlan {
    fn with_difference_shape(self, difference_shape: Vec<u64>) -> Self {
        match self {
            Self::Lora {
                alpha,
                rank,
                has_mid,
                ..
            } => Self::Lora {
                alpha,
                rank,
                has_mid,
                difference_shape,
            },
            Self::Loha {
                alpha,
                rank,
                tucker,
                ..
            } => Self::Loha {
                alpha,
                rank,
                tucker,
                difference_shape,
            },
            Self::Lokr { first, second, .. } => Self::Lokr {
                first,
                second,
                difference_shape,
            },
            Self::Oft {
                constraint,
                has_rescale,
                raw_norm,
                constrained,
                ..
            } => Self::Oft {
                constraint,
                has_rescale,
                raw_norm,
                constrained,
                difference_shape,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum AdapterTensor {
    Dense(Tensor),
    Quantized(QuantizedMatrix),
    QuantizedLinear(QuantizedLinearMatrix),
}

impl AdapterTensor {
    pub fn shape(&self) -> Result<Vec<u64>, WeightAdapterError> {
        match self {
            Self::Dense(tensor) => Ok(tensor.descriptor().shape().to_vec()),
            Self::Quantized(matrix) => Ok(vec![
                checked_u64(matrix.rows(), "quantized rows")?,
                checked_u64(matrix.columns(), "quantized columns")?,
            ]),
            Self::QuantizedLinear(matrix) => Ok(vec![
                checked_u64(matrix.rows(), "quantized-linear rows")?,
                checked_u64(matrix.columns(), "quantized-linear columns")?,
            ]),
        }
    }

    pub fn storage_bytes(&self) -> Result<u64, WeightAdapterError> {
        match self {
            Self::Dense(tensor) => Ok(tensor.storage_byte_len()),
            Self::Quantized(matrix) => checked_u64(matrix.storage_bytes(), "quantized storage"),
            Self::QuantizedLinear(matrix) => {
                checked_u64(matrix.storage_bytes(), "quantized-linear storage")
            }
        }
    }

    pub fn scalar_f32(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<f32, WeightAdapterError> {
        let values = self.materialize_values(backend, context)?;
        match values.as_slice() {
            [value] if value.is_finite() => Ok(*value),
            [_] => Err(WeightAdapterError::NonFinite("adapter scalar")),
            _ => Err(WeightAdapterError::InvalidShape(
                "adapter scalar must contain exactly one value".into(),
            )),
        }
    }

    fn materialize_values(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, WeightAdapterError> {
        context.check()?;
        let values = match self {
            Self::Dense(tensor) => {
                require_tensor_boundary(tensor, backend, context, "adapter tensor")?;
                tensor_to_f32_with_backend_exact_native(backend, tensor, context)?
            }
            Self::Quantized(matrix) => matrix.materialize(backend, context)?.values().to_vec(),
            Self::QuantizedLinear(matrix) => {
                matrix.materialize(backend, context)?.values().to_vec()
            }
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err(WeightAdapterError::NonFinite("adapter tensor"));
        }
        context.check()?;
        Ok(values)
    }

    fn materialize_tensor(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, WeightAdapterError> {
        match self {
            Self::Dense(tensor) => {
                require_tensor_boundary(tensor, backend, context, "adapter tensor")?;
                require_f32(tensor, "adapter tensor")?;
                Ok(tensor.clone())
            }
            Self::Quantized(_) | Self::QuantizedLinear(_) => {
                let shape = self.shape()?;
                let values = self.materialize_values(backend, context)?;
                Ok(tensor_from_f32_with_backend_exact_native(
                    backend,
                    &shape,
                    &values,
                    DType::F32,
                    backend.device(),
                    context,
                )?)
            }
        }
    }

    pub fn to_patch_tensor(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<PatchTensor, WeightAdapterError> {
        Ok(PatchTensor::checked(
            self.shape()?,
            self.materialize_values(backend, context)?,
        )?)
    }
}

#[derive(Clone, Debug)]
pub enum NativeWeightAdapter {
    Lora {
        up: AdapterTensor,
        down: AdapterTensor,
        alpha: Option<f32>,
        mid: Option<AdapterTensor>,
        dora_scale: Option<AdapterTensor>,
        reshape: Option<Vec<u64>>,
    },
    Loha {
        first_up: AdapterTensor,
        first_down: AdapterTensor,
        second_up: AdapterTensor,
        second_down: AdapterTensor,
        first_tucker: Option<AdapterTensor>,
        second_tucker: Option<AdapterTensor>,
        alpha: Option<f32>,
        dora_scale: Option<AdapterTensor>,
    },
    Lokr {
        first: Option<AdapterTensor>,
        second: Option<AdapterTensor>,
        first_up: Option<AdapterTensor>,
        first_down: Option<AdapterTensor>,
        second_up: Option<AdapterTensor>,
        second_down: Option<AdapterTensor>,
        second_tucker: Option<AdapterTensor>,
        alpha: Option<f32>,
        dora_scale: Option<AdapterTensor>,
    },
    Glora {
        first_a: AdapterTensor,
        second_a: AdapterTensor,
        first_b: AdapterTensor,
        second_b: AdapterTensor,
        alpha: Option<f32>,
        dora_scale: Option<AdapterTensor>,
    },
    Oft {
        blocks: AdapterTensor,
        rescale: Option<AdapterTensor>,
        constraint: Option<f32>,
        dora_scale: Option<AdapterTensor>,
    },
    Boft {
        blocks: AdapterTensor,
        rescale: Option<AdapterTensor>,
        constraint: Option<f32>,
        dora_scale: Option<AdapterTensor>,
    },
}

impl NativeWeightAdapter {
    pub const fn family(&self) -> AdapterFamily {
        match self {
            Self::Lora { .. } => AdapterFamily::Lora,
            Self::Loha { .. } => AdapterFamily::Loha,
            Self::Lokr { .. } => AdapterFamily::Lokr,
            Self::Glora { .. } => AdapterFamily::Glora,
            Self::Oft { .. } => AdapterFamily::Oft,
            Self::Boft { .. } => AdapterFamily::Boft,
        }
    }

    pub const fn trainable_kind(&self) -> Result<TrainableAdapterKind, WeightAdapterError> {
        match self {
            Self::Lora { .. } => Ok(TrainableAdapterKind::LoraDiff),
            Self::Loha { .. } => Ok(TrainableAdapterKind::LohaDiff),
            Self::Lokr { .. } => Ok(TrainableAdapterKind::LokrDiff),
            Self::Oft { .. } => Ok(TrainableAdapterKind::OftDiff),
            Self::Glora { .. } | Self::Boft { .. } => Err(
                WeightAdapterError::UnsupportedTrainableFamily(self.family()),
            ),
        }
    }

    pub fn create_trainable(
        kind: TrainableAdapterKind,
        base_weight: &Tensor,
        rank: u64,
        alpha: f32,
        backend: &dyn TensorBackend,
        transaction: &mut RngTransaction,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, WeightAdapterError> {
        context.check()?;
        require_tensor_boundary(base_weight, backend, context, "trainable base weight")?;
        require_f32(base_weight, "trainable base weight")?;
        validate_finite("trainable alpha", alpha)?;
        if rank == 0 {
            return Err(WeightAdapterError::InvalidShape(
                "trainable adapter rank must be positive".into(),
            ));
        }
        let shape = base_weight.descriptor().shape();
        if shape.len() < 2 || shape.contains(&0) {
            return Err(WeightAdapterError::InvalidShape(
                "trainable base weight must have rank at least two and no empty dimensions".into(),
            ));
        }
        let cpu_backend = backend
            .cpu_backend()
            .ok_or(WeightAdapterError::UnsupportedDevice {
                operation: "trainable adapter initialization",
                device: backend.device(),
            })?;
        transaction.require_device(backend.device())?;
        let mut staged = transaction.clone();
        let output_dimension = shape[0];
        let flattened_input = checked_product(&shape[1..])?;
        let adapter = match kind {
            TrainableAdapterKind::LoraDiff => {
                let up = random_uniform_tensor(
                    cpu_backend,
                    &[output_dimension, rank],
                    1.0 / (rank as f32).sqrt(),
                    &mut staged,
                    context,
                )?;
                let down = zero_tensor(cpu_backend, &[rank, flattened_input], context)?;
                Self::Lora {
                    up: AdapterTensor::Dense(up),
                    down: AdapterTensor::Dense(down),
                    alpha: Some(alpha),
                    mid: None,
                    dora_scale: None,
                    reshape: None,
                }
            }
            TrainableAdapterKind::LohaDiff => Self::Loha {
                first_up: AdapterTensor::Dense(random_normal_tensor(
                    cpu_backend,
                    &[output_dimension, rank],
                    0.1,
                    &mut staged,
                    context,
                )?),
                first_down: AdapterTensor::Dense(zero_tensor(
                    cpu_backend,
                    &[rank, flattened_input],
                    context,
                )?),
                second_up: AdapterTensor::Dense(random_normal_tensor(
                    cpu_backend,
                    &[output_dimension, rank],
                    0.1,
                    &mut staged,
                    context,
                )?),
                second_down: AdapterTensor::Dense(random_normal_tensor(
                    cpu_backend,
                    &[rank, flattened_input],
                    0.01,
                    &mut staged,
                    context,
                )?),
                first_tucker: None,
                second_tucker: None,
                alpha: Some(alpha),
                dora_scale: None,
            },
            TrainableAdapterKind::LokrDiff => {
                let (output_left, output_right) =
                    crate::factorize_patch_dimension(output_dimension, Some(rank))?;
                let input_dimension = shape[1];
                let (input_left, input_right) =
                    crate::factorize_patch_dimension(input_dimension, Some(rank))?;
                let first = zero_tensor(cpu_backend, &[output_left, input_left], context)?;
                let mut second_shape = vec![output_right, input_right];
                second_shape.extend_from_slice(&shape[2..]);
                let fan_in = checked_product(&second_shape[1..])?;
                let second = random_uniform_tensor(
                    cpu_backend,
                    &second_shape,
                    1.0 / (fan_in as f32).sqrt(),
                    &mut staged,
                    context,
                )?;
                Self::Lokr {
                    first: Some(AdapterTensor::Dense(first)),
                    second: Some(AdapterTensor::Dense(second)),
                    first_up: None,
                    first_down: None,
                    second_up: None,
                    second_down: None,
                    second_tucker: None,
                    alpha: Some(alpha),
                    dora_scale: None,
                }
            }
            TrainableAdapterKind::OftDiff => {
                let (block_size, block_count) =
                    crate::factorize_patch_dimension(output_dimension, Some(rank))?;
                Self::Oft {
                    blocks: AdapterTensor::Dense(zero_tensor(
                        cpu_backend,
                        &[block_count, block_size, block_size],
                        context,
                    )?),
                    rescale: None,
                    constraint: Some(alpha),
                    dora_scale: None,
                }
            }
        };
        adapter.validate()?;
        context.check()?;
        *transaction = staged;
        Ok(adapter)
    }

    pub fn forward_trainable_recorded(
        &self,
        base_weight: &Tensor,
        backend: &dyn TensorBackend,
        tape: &mut AutogradTape,
        context: &ExecutionContext<'_>,
    ) -> Result<TrainableWeightOutput, WeightAdapterError> {
        self.validate()?;
        require_tensor_boundary(base_weight, backend, context, "trainable base weight")?;
        require_f32(base_weight, "trainable base weight")?;
        let cpu_backend = backend
            .cpu_backend()
            .ok_or(WeightAdapterError::UnsupportedDevice {
                operation: "trainable autograd",
                device: backend.device(),
            })?;
        if matches!(self, Self::Loha { .. }) {
            return self.forward_trainable_loha_recorded(base_weight, cpu_backend, tape, context);
        }
        let (inputs, output, plan) = match self {
            Self::Oft {
                blocks,
                rescale,
                constraint,
                dora_scale: _,
            } => trainable_oft(
                blocks,
                rescale.as_ref(),
                constraint.ok_or_else(|| {
                    WeightAdapterError::InvalidPlan(
                        "OFT trainable execution requires an explicit constraint".into(),
                    )
                })?,
                base_weight,
                cpu_backend,
                context,
            )?,
            _ => {
                let (inputs, difference, plan) = self.trainable_difference(cpu_backend, context)?;
                if difference.descriptor().element_count()?
                    != base_weight.descriptor().element_count()?
                {
                    return Err(WeightAdapterError::InvalidShape(
                        "trainable adapter difference does not match the base weight element count"
                            .into(),
                    ));
                }
                let difference_shape = difference.descriptor().shape().to_vec();
                let difference =
                    reshape_read_only(&difference, base_weight.descriptor().shape().to_vec())?;
                let output = binary(
                    cpu_backend,
                    base_weight,
                    &difference,
                    BinaryOperation::Add,
                    context,
                )?;
                (inputs, output, plan.with_difference_shape(difference_shape))
            }
        };
        let mut recorded_inputs = Vec::new();
        recorded_inputs
            .try_reserve_exact(inputs.len() + 1)
            .map_err(|_| WeightAdapterError::ShapeOverflow)?;
        recorded_inputs.push(base_weight.clone());
        recorded_inputs.extend(inputs);
        let base_saved = matches!(plan, TrainableBackwardPlan::Oft { .. });
        let needs_grad = recorded_inputs
            .iter()
            .any(|input| tape.requires_grad(input) || tape.output_slot(input).is_some());
        if !needs_grad {
            return Ok(TrainableWeightOutput {
                output,
                output_slot: None,
            });
        }
        let saved_tensors = recorded_inputs
            .iter()
            .skip(usize::from(!base_saved))
            .map(SavedTensor::capture)
            .collect::<Vec<_>>();
        let input_references = recorded_inputs.iter().collect::<Vec<_>>();
        let input_count = recorded_inputs.len();
        let output_shape = output.descriptor().shape().to_vec();
        let slots = tape.record_operation(
            &input_references,
            &[&output],
            &[true],
            saved_tensors,
            Arc::new(TrainableWeightBackwardRule {
                plan,
                input_count,
                output_shape,
            }),
        )?;
        Ok(TrainableWeightOutput {
            output,
            output_slot: slots.and_then(|slots| slots.first().copied()),
        })
    }

    fn forward_trainable_loha_recorded(
        &self,
        base_weight: &Tensor,
        backend: &CpuBackend,
        tape: &mut AutogradTape,
        context: &ExecutionContext<'_>,
    ) -> Result<TrainableWeightOutput, WeightAdapterError> {
        let Self::Loha {
            first_up,
            first_down,
            second_up,
            second_down,
            first_tucker,
            second_tucker,
            alpha,
            dora_scale: _,
        } = self
        else {
            return Err(WeightAdapterError::InvalidPlan(
                "LoHa recorded execution received another adapter family".into(),
            ));
        };
        let first_up = trainable_dense(first_up, backend, context, "LoHa first up")?;
        let first_down = trainable_dense(first_down, backend, context, "LoHa first down")?;
        let second_up = trainable_dense(second_up, backend, context, "LoHa second up")?;
        let second_down = trainable_dense(second_down, backend, context, "LoHa second down")?;
        let rank = first_extent(&first_down, "LoHa trainable rank")?;
        let scale = trainable_scale(*alpha, rank, "LoHa")?;
        let scale = tensor_from_f32_with_backend_exact_native(
            backend,
            &[1],
            &[scale],
            DType::F32,
            backend.device(),
            context,
        )?;
        let needs =
            |tensor: &Tensor| tape.requires_grad(tensor) || tape.output_slot(tensor).is_some();
        let difference = match (first_tucker, second_tucker) {
            (None, None) => {
                let factors = [&first_up, &first_down, &second_up, &second_down];
                let needs_input_grad = [
                    needs(&first_up),
                    needs(&first_down),
                    needs(&second_up),
                    needs(&second_down),
                    false,
                ];
                HadaWeightFunction::forward_recorded(
                    backend,
                    tape,
                    factors,
                    &scale,
                    needs_input_grad,
                    context,
                )?
                .1
            }
            (Some(first_tucker), Some(second_tucker)) => {
                let first_tucker =
                    trainable_dense(first_tucker, backend, context, "LoHa first Tucker")?;
                let second_tucker =
                    trainable_dense(second_tucker, backend, context, "LoHa second Tucker")?;
                let tensors = [
                    &first_tucker,
                    &first_up,
                    &first_down,
                    &second_tucker,
                    &second_up,
                    &second_down,
                ];
                let needs_input_grad = [
                    needs(&first_tucker),
                    needs(&first_up),
                    needs(&first_down),
                    needs(&second_tucker),
                    needs(&second_up),
                    needs(&second_down),
                    false,
                ];
                HadaWeightTuckerFunction::forward_recorded(
                    backend,
                    tape,
                    tensors,
                    &scale,
                    needs_input_grad,
                    context,
                )?
                .1
            }
            _ => {
                return Err(WeightAdapterError::InvalidShape(
                    "LoHa trainable Tucker tensors must be supplied as a pair".into(),
                ));
            }
        };
        if difference.descriptor().element_count()? != base_weight.descriptor().element_count()? {
            return Err(WeightAdapterError::InvalidShape(
                "LoHa trainable difference does not match the base weight element count".into(),
            ));
        }
        let difference_shape = difference.descriptor().shape().to_vec();
        let base_shape = base_weight.descriptor().shape().to_vec();
        let reshaped = reshape_read_only(&difference, base_shape.clone())?;
        let output = binary(
            backend,
            base_weight,
            &reshaped,
            BinaryOperation::Add,
            context,
        )?;
        let slots = tape.record_operation(
            &[base_weight, &difference],
            &[&output],
            &[true],
            Vec::new(),
            Arc::new(TrainableAddBackwardRule {
                base_shape,
                difference_shape,
            }),
        )?;
        Ok(TrainableWeightOutput {
            output,
            output_slot: slots.and_then(|slots| slots.first().copied()),
        })
    }

    fn trainable_difference(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, Tensor, TrainableBackwardPlan), WeightAdapterError> {
        match self {
            Self::Lora {
                up,
                down,
                alpha,
                mid,
                dora_scale: _,
                reshape: _,
            } => trainable_lora(up, down, *alpha, mid.as_ref(), backend, context),
            Self::Loha {
                first_up,
                first_down,
                second_up,
                second_down,
                first_tucker,
                second_tucker,
                alpha,
                dora_scale: _,
            } => trainable_loha(
                first_up,
                first_down,
                second_up,
                second_down,
                first_tucker.as_ref(),
                second_tucker.as_ref(),
                *alpha,
                backend,
                context,
            ),
            Self::Lokr {
                first,
                second,
                first_up,
                first_down,
                second_up,
                second_down,
                second_tucker,
                alpha,
                dora_scale: _,
            } => trainable_lokr(
                first.as_ref(),
                second.as_ref(),
                first_up.as_ref(),
                first_down.as_ref(),
                second_up.as_ref(),
                second_down.as_ref(),
                second_tucker.as_ref(),
                *alpha,
                backend,
                context,
            ),
            Self::Oft { .. } => Err(WeightAdapterError::InvalidPlan(
                "OFT trainable execution requires the base-weight transform path".into(),
            )),
            Self::Glora { .. } | Self::Boft { .. } => Err(
                WeightAdapterError::UnsupportedTrainableFamily(self.family()),
            ),
        }
    }

    pub fn calculate_static_patch_graph(
        &self,
        base_artifact_digest: impl Into<String>,
        identifier: impl Into<String>,
        target_key: impl Into<String>,
        expected_shape: Vec<u64>,
        strength: f32,
        strength_model: f32,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<PatchGraph, WeightAdapterError> {
        validate_finite("strength", strength)?;
        validate_finite("strength_model", strength_model)?;
        let payload = self.to_patch_payload(backend, context)?;
        Ok(PatchGraph::checked_semantic(
            base_artifact_digest,
            vec![SemanticPatchOperation {
                identifier: identifier.into(),
                target_key: target_key.into(),
                expected_shape,
                strength,
                strength_model,
                slices: Vec::new(),
                transform: PatchValueTransform::default(),
                payload,
            }],
        )?)
    }

    pub fn apply_static(
        &self,
        source: &MappedModelWeights,
        identifier: impl Into<String>,
        target_key: impl Into<String>,
        expected_shape: Vec<u64>,
        strength: f32,
        strength_model: f32,
        compute_boundary: PatchComputeBoundary,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<MappedModelWeights, WeightAdapterError> {
        let graph = self.calculate_static_patch_graph(
            source.base_artifact_digest(),
            identifier,
            target_key,
            expected_shape,
            strength,
            strength_model,
            backend,
            context,
        )?;
        Ok(graph.apply_with_compute_boundary(backend, source, compute_boundary, context)?)
    }

    pub fn to_patch_payload(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<PatchPayload, WeightAdapterError> {
        Ok(match self {
            Self::Lora {
                up,
                down,
                alpha,
                mid,
                dora_scale,
                reshape,
            } => PatchPayload::Lora {
                up: up.to_patch_tensor(backend, context)?,
                down: down.to_patch_tensor(backend, context)?,
                mid: patch_optional(mid, backend, context)?,
                alpha: *alpha,
                dora_scale: patch_optional(dora_scale, backend, context)?,
                reshape: reshape.clone(),
            },
            Self::Loha {
                first_up,
                first_down,
                second_up,
                second_down,
                first_tucker,
                second_tucker,
                alpha,
                dora_scale,
            } => PatchPayload::Loha {
                first_up: first_up.to_patch_tensor(backend, context)?,
                first_down: first_down.to_patch_tensor(backend, context)?,
                second_up: second_up.to_patch_tensor(backend, context)?,
                second_down: second_down.to_patch_tensor(backend, context)?,
                first_tucker: patch_optional(first_tucker, backend, context)?,
                second_tucker: patch_optional(second_tucker, backend, context)?,
                alpha: *alpha,
                dora_scale: patch_optional(dora_scale, backend, context)?,
            },
            Self::Lokr {
                first,
                second,
                first_up,
                first_down,
                second_up,
                second_down,
                second_tucker,
                alpha,
                dora_scale,
            } => PatchPayload::Lokr {
                first: patch_optional(first, backend, context)?,
                second: patch_optional(second, backend, context)?,
                first_up: patch_optional(first_up, backend, context)?,
                first_down: patch_optional(first_down, backend, context)?,
                second_up: patch_optional(second_up, backend, context)?,
                second_down: patch_optional(second_down, backend, context)?,
                second_tucker: patch_optional(second_tucker, backend, context)?,
                alpha: *alpha,
                dora_scale: patch_optional(dora_scale, backend, context)?,
            },
            Self::Glora {
                first_a,
                second_a,
                first_b,
                second_b,
                alpha,
                dora_scale,
            } => PatchPayload::Glora {
                first_a: first_a.to_patch_tensor(backend, context)?,
                second_a: second_a.to_patch_tensor(backend, context)?,
                first_b: first_b.to_patch_tensor(backend, context)?,
                second_b: second_b.to_patch_tensor(backend, context)?,
                alpha: *alpha,
                dora_scale: patch_optional(dora_scale, backend, context)?,
            },
            Self::Oft {
                blocks,
                rescale,
                constraint,
                dora_scale,
            } => PatchPayload::Oft {
                blocks: blocks.to_patch_tensor(backend, context)?,
                rescale: patch_optional(rescale, backend, context)?,
                constraint: *constraint,
                dora_scale: patch_optional(dora_scale, backend, context)?,
            },
            Self::Boft {
                blocks,
                rescale,
                constraint,
                dora_scale,
            } => PatchPayload::Boft {
                blocks: blocks.to_patch_tensor(backend, context)?,
                rescale: patch_optional(rescale, backend, context)?,
                constraint: *constraint,
                dora_scale: patch_optional(dora_scale, backend, context)?,
            },
        })
    }

    fn validate(&self) -> Result<(), WeightAdapterError> {
        match self {
            Self::Lora {
                up,
                down,
                alpha,
                mid,
                reshape,
                ..
            } => {
                validate_matrix_pair(up, down, "LoRA")?;
                validate_optional_finite("LoRA alpha", *alpha)?;
                if let Some(mid) = mid {
                    validate_convolution_tucker(mid, up, down, "LoRA mid")?;
                }
                if let Some(shape) = reshape
                    && (shape.is_empty() || shape.contains(&0))
                {
                    return Err(WeightAdapterError::InvalidShape(
                        "LoRA reshape dimensions must be positive".into(),
                    ));
                }
            }
            Self::Loha {
                first_up,
                first_down,
                second_up,
                second_down,
                first_tucker,
                second_tucker,
                alpha,
                ..
            } => {
                if first_tucker.is_some() != second_tucker.is_some() {
                    return Err(WeightAdapterError::InvalidShape(
                        "LoHa Tucker tensors must be supplied as a pair".into(),
                    ));
                }
                match (first_tucker, second_tucker) {
                    (Some(first_tucker), Some(second_tucker)) => {
                        let first_shape = validate_tucker_factor(
                            first_tucker,
                            first_up,
                            first_down,
                            "LoHa first factor",
                        )?;
                        let second_shape = validate_tucker_factor(
                            second_tucker,
                            second_up,
                            second_down,
                            "LoHa second factor",
                        )?;
                        if first_shape != second_shape {
                            return Err(WeightAdapterError::InvalidShape(
                                "LoHa Tucker factors produce different shapes".into(),
                            ));
                        }
                    }
                    (None, None) => {
                        validate_matrix_pair(first_up, first_down, "LoHa first factor")?;
                        validate_matrix_pair(second_up, second_down, "LoHa second factor")?;
                    }
                    _ => {
                        return Err(WeightAdapterError::InvalidShape(
                            "LoHa Tucker tensors must be supplied as a pair".into(),
                        ));
                    }
                }
                validate_optional_finite("LoHa alpha", *alpha)?;
            }
            Self::Lokr {
                first,
                second,
                first_up,
                first_down,
                second_up,
                second_down,
                second_tucker,
                alpha,
                ..
            } => {
                validate_direct_or_pair("LoKr first factor", first, first_up, first_down)?;
                match second_tucker {
                    Some(tucker) => match (second, second_up, second_down) {
                        (None, Some(up), Some(down)) => {
                            validate_tucker_factor(tucker, up, down, "LoKr second factor")?;
                        }
                        _ => {
                            return Err(WeightAdapterError::InvalidShape(
                                "LoKr Tucker tensor requires decomposed second factors".into(),
                            ));
                        }
                    },
                    None => validate_direct_or_pair(
                        "LoKr second factor",
                        second,
                        second_up,
                        second_down,
                    )?,
                }
                validate_optional_finite("LoKr alpha", *alpha)?;
            }
            Self::Glora {
                first_a,
                second_a,
                first_b,
                second_b,
                alpha,
                ..
            } => {
                for (name, tensor) in [
                    ("GLoRA first A", first_a),
                    ("GLoRA second A", second_a),
                    ("GLoRA first B", first_b),
                    ("GLoRA second B", second_b),
                ] {
                    require_rank_two(tensor, name)?;
                }
                validate_glora_layout(first_a, second_a, first_b, second_b)?;
                validate_optional_finite("GLoRA alpha", *alpha)?;
            }
            Self::Oft {
                blocks, constraint, ..
            } => {
                require_square_blocks(blocks, 3, "OFT blocks")?;
                validate_optional_nonnegative("OFT constraint", *constraint)?;
            }
            Self::Boft {
                blocks, constraint, ..
            } => {
                require_square_blocks(blocks, 4, "BOFT blocks")?;
                let shape = blocks.shape()?;
                if shape[2] % 2 != 0 {
                    return Err(WeightAdapterError::InvalidShape(
                        "BOFT block size must be even".into(),
                    ));
                }
                let channels = shape[1]
                    .checked_mul(shape[2])
                    .ok_or(WeightAdapterError::ShapeOverflow)?;
                for stage in 0..shape[0] {
                    let grouping = 2_u64
                        .checked_pow(
                            u32::try_from(stage).map_err(|_| WeightAdapterError::ShapeOverflow)?,
                        )
                        .and_then(|value| value.checked_mul(shape[2] / 2))
                        .ok_or(WeightAdapterError::ShapeOverflow)?;
                    let stage_width = grouping
                        .checked_mul(2)
                        .ok_or(WeightAdapterError::ShapeOverflow)?;
                    if stage_width == 0 || channels % stage_width != 0 {
                        return Err(WeightAdapterError::InvalidShape(
                            "BOFT stage geometry does not divide the output channels".into(),
                        ));
                    }
                }
                validate_optional_nonnegative("BOFT constraint", *constraint)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct WeightAdapterLoadRequest {
    pub prefix: String,
    pub tensors: BTreeMap<String, AdapterTensor>,
    pub alpha: Option<f32>,
    pub dora_scale: Option<AdapterTensor>,
}

#[derive(Clone, Debug)]
pub struct LoadedWeightAdapter {
    adapter: NativeWeightAdapter,
    loaded_keys: BTreeSet<String>,
}

impl LoadedWeightAdapter {
    pub fn adapter(&self) -> &NativeWeightAdapter {
        &self.adapter
    }

    pub fn loaded_keys(&self) -> &BTreeSet<String> {
        &self.loaded_keys
    }

    pub fn into_adapter(self) -> NativeWeightAdapter {
        self.adapter
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeightAdapterRegistry;

impl WeightAdapterRegistry {
    pub const fn ordered_families(self) -> &'static [AdapterFamily] {
        &WEIGHT_ADAPTER_ORDER
    }

    pub fn named_family(self, name: &str) -> Option<AdapterFamily> {
        ADAPTER_MAP_ORDER
            .iter()
            .find_map(|(candidate, family)| (*candidate == name).then_some(*family))
    }

    pub fn load_first(
        self,
        request: &WeightAdapterLoadRequest,
    ) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
        validate_module_key(&request.prefix)?;
        validate_optional_finite("adapter alpha", request.alpha)?;
        for family in WEIGHT_ADAPTER_ORDER {
            if let Some(loaded) = load_family(family, request)? {
                loaded.adapter.validate()?;
                return Ok(Some(loaded));
            }
        }
        Ok(None)
    }

    pub fn load_unique(
        self,
        request: &WeightAdapterLoadRequest,
    ) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
        validate_module_key(&request.prefix)?;
        validate_optional_finite("adapter alpha", request.alpha)?;
        let mut loaded: Option<LoadedWeightAdapter> = None;
        for family in WEIGHT_ADAPTER_ORDER {
            let Some(candidate) = load_family(family, request)? else {
                continue;
            };
            candidate.adapter.validate()?;
            if let Some(previous) = &loaded {
                return Err(WeightAdapterError::AmbiguousFamilies {
                    first: previous.adapter.family(),
                    second: candidate.adapter.family(),
                });
            }
            loaded = Some(candidate);
        }
        Ok(loaded)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Linear,
    Convolution { dimensions: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleTypeInfo {
    pub kind: LayerKind,
    pub stride: Vec<u64>,
    pub padding: Vec<u64>,
    pub dilation: Vec<u64>,
    pub groups: u64,
    pub kernel_size: Vec<u64>,
    pub input_channels: Option<u64>,
    pub output_channels: Option<u64>,
    pub has_weight: bool,
}

impl ModuleTypeInfo {
    pub fn linear(input_features: u64, output_features: u64) -> Result<Self, WeightAdapterError> {
        if input_features == 0 || output_features == 0 {
            return Err(WeightAdapterError::InvalidModule(
                "linear feature counts must be positive".into(),
            ));
        }
        Ok(Self {
            kind: LayerKind::Linear,
            stride: vec![1],
            padding: vec![0],
            dilation: vec![1],
            groups: 1,
            kernel_size: vec![1],
            input_channels: Some(input_features),
            output_channels: Some(output_features),
            has_weight: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn convolution(
        dimensions: u8,
        stride: Vec<u64>,
        padding: Vec<u64>,
        dilation: Vec<u64>,
        groups: u64,
        kernel_size: Vec<u64>,
        input_channels: Option<u64>,
        output_channels: Option<u64>,
    ) -> Result<Self, WeightAdapterError> {
        if !(1..=3).contains(&dimensions) {
            return Err(WeightAdapterError::InvalidModule(
                "convolution dimension must be one, two, or three".into(),
            ));
        }
        let rank = usize::from(dimensions);
        if stride.len() != rank
            || padding.len() != rank
            || dilation.len() != rank
            || kernel_size.len() != rank
            || stride.contains(&0)
            || dilation.contains(&0)
            || kernel_size.contains(&0)
            || groups == 0
        {
            return Err(WeightAdapterError::InvalidModule(
                "convolution geometry is malformed".into(),
            ));
        }
        Ok(Self {
            kind: LayerKind::Convolution { dimensions },
            stride,
            padding,
            dilation,
            groups,
            kernel_size,
            input_channels,
            output_channels,
            has_weight: true,
        })
    }

    pub fn infer_custom(class_name: &str) -> Result<Self, WeightAdapterError> {
        let normalized = class_name.to_ascii_lowercase();
        if normalized.contains("conv3d") {
            Self::convolution(
                3,
                vec![1; 3],
                vec![0; 3],
                vec![1; 3],
                1,
                vec![1; 3],
                None,
                None,
            )
        } else if normalized.contains("conv2d") {
            Self::convolution(
                2,
                vec![1; 2],
                vec![0; 2],
                vec![1; 2],
                1,
                vec![1; 2],
                None,
                None,
            )
        } else if normalized.contains("conv1d") {
            Self::convolution(1, vec![1], vec![0], vec![1], 1, vec![1], None, None)
        } else if normalized.contains("conv") {
            Self::convolution(
                2,
                vec![1; 2],
                vec![0; 2],
                vec![1; 2],
                1,
                vec![1; 2],
                None,
                None,
            )
        } else {
            Ok(Self {
                kind: LayerKind::Linear,
                stride: vec![1],
                padding: vec![0],
                dilation: vec![1],
                groups: 1,
                kernel_size: vec![1],
                input_channels: None,
                output_channels: None,
                has_weight: true,
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct BypassBinding {
    module_key: String,
    adapter: NativeWeightAdapter,
    strength: f32,
}

impl BypassBinding {
    pub fn checked(
        module_key: impl Into<String>,
        adapter: NativeWeightAdapter,
        strength: f32,
    ) -> Result<Self, WeightAdapterError> {
        let module_key = normalize_module_key(module_key.into())?;
        validate_finite("adapter strength", strength)?;
        adapter.validate()?;
        Ok(Self {
            module_key,
            adapter,
            strength,
        })
    }

    pub fn module_key(&self) -> &str {
        &self.module_key
    }

    pub fn adapter(&self) -> &NativeWeightAdapter {
        &self.adapter
    }

    pub const fn strength(&self) -> f32 {
        self.strength
    }
}

#[derive(Clone, Debug)]
pub enum BypassPatch {
    Adapter {
        patch_strength: f32,
        adapter: NativeWeightAdapter,
    },
    StaticPatch,
}

#[derive(Clone, Debug, Default)]
pub struct BypassInjectionManager {
    bindings: BTreeMap<String, BypassBinding>,
}

impl BypassInjectionManager {
    pub fn add_adapter(
        &mut self,
        module_key: impl Into<String>,
        adapter: NativeWeightAdapter,
        strength: f32,
    ) -> Result<(), WeightAdapterError> {
        if self.bindings.len() >= MAX_ADAPTER_BINDINGS {
            return Err(WeightAdapterError::TooManyBindings(MAX_ADAPTER_BINDINGS));
        }
        let binding = BypassBinding::checked(module_key, adapter, strength)?;
        self.bindings
            .insert(binding.module_key().to_owned(), binding);
        Ok(())
    }

    pub fn clear_adapters(&mut self) {
        self.bindings.clear();
    }

    pub fn create_injections(
        &self,
        modules: &BTreeMap<String, ModuleTypeInfo>,
    ) -> Result<BypassRuntimePlan, WeightAdapterError> {
        let mut hooks = BTreeMap::new();
        for (module_key, binding) in &self.bindings {
            let Some(module) = modules.get(module_key) else {
                continue;
            };
            if !module.has_weight {
                continue;
            }
            hooks.insert(
                module_key.clone(),
                BypassForwardHook::checked(binding.clone(), module.clone())?,
            );
        }
        Ok(BypassRuntimePlan { hooks })
    }

    pub fn from_patches(
        patches: BTreeMap<String, Vec<BypassPatch>>,
        strength: f32,
    ) -> Result<Self, WeightAdapterError> {
        validate_finite("global adapter strength", strength)?;
        let mut manager = Self::default();
        for (key, patch_list) in patches {
            for patch in patch_list {
                if let BypassPatch::Adapter {
                    patch_strength,
                    adapter,
                } = patch
                {
                    validate_finite("patch adapter strength", patch_strength)?;
                    manager.add_adapter(key.clone(), adapter, strength * patch_strength)?;
                }
            }
        }
        Ok(manager)
    }
}

#[derive(Clone, Debug)]
pub struct BypassForwardHook {
    binding: BypassBinding,
    module: ModuleTypeInfo,
    injected: bool,
}

impl BypassForwardHook {
    pub fn checked(
        binding: BypassBinding,
        module: ModuleTypeInfo,
    ) -> Result<Self, WeightAdapterError> {
        validate_layer_compatibility(&binding.adapter, &module)?;
        Ok(Self {
            binding,
            module,
            injected: false,
        })
    }

    pub fn inject(&mut self) -> bool {
        let changed = !self.injected;
        self.injected = true;
        changed
    }

    pub fn eject(&mut self) -> bool {
        let changed = self.injected;
        self.injected = false;
        changed
    }

    pub const fn is_injected(&self) -> bool {
        self.injected
    }

    pub fn execute<BaseForward>(
        &self,
        backend: &dyn TensorBackend,
        input: &Tensor,
        base_forward: BaseForward,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, WeightAdapterError>
    where
        BaseForward: FnOnce(
            &dyn TensorBackend,
            &Tensor,
            &ExecutionContext<'_>,
        ) -> Result<Tensor, WeightAdapterError>,
    {
        if !self.injected {
            return base_forward(backend, input, context);
        }
        execute_bypass(
            &self.binding.adapter,
            self.binding.strength,
            &self.module,
            backend,
            input,
            base_forward,
            context,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct BypassRuntimePlan {
    hooks: BTreeMap<String, BypassForwardHook>,
}

impl BypassRuntimePlan {
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    pub fn inject_all(&mut self) -> usize {
        self.hooks
            .values_mut()
            .fold(0, |count, hook| count + usize::from(hook.inject()))
    }

    pub fn eject_all(&mut self) -> usize {
        self.hooks
            .values_mut()
            .fold(0, |count, hook| count + usize::from(hook.eject()))
    }

    pub fn hook(&self, module_key: &str) -> Option<&BypassForwardHook> {
        self.hooks.get(module_key)
    }
}

fn execute_bypass<BaseForward>(
    adapter: &NativeWeightAdapter,
    multiplier: f32,
    module: &ModuleTypeInfo,
    backend: &dyn TensorBackend,
    input: &Tensor,
    base_forward: BaseForward,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError>
where
    BaseForward: FnOnce(
        &dyn TensorBackend,
        &Tensor,
        &ExecutionContext<'_>,
    ) -> Result<Tensor, WeightAdapterError>,
{
    context.check()?;
    require_execution_input(input, backend, context)?;
    validate_finite("adapter multiplier", multiplier)?;
    if let NativeWeightAdapter::Glora { .. } = adapter {
        let (first_path, second_path) =
            glora_paths(adapter, multiplier, module, backend, input, context)?;
        let modified_input = binary(backend, input, &first_path, BinaryOperation::Add, context)?;
        let base = base_forward(backend, &modified_input, context)?;
        validate_base_output(&base, backend, context)?;
        let output = binary(backend, &base, &second_path, BinaryOperation::Add, context)?;
        context.check()?;
        return Ok(output);
    }

    let base = base_forward(backend, input, context)?;
    validate_base_output(&base, backend, context)?;
    let combined = match adapter {
        NativeWeightAdapter::Oft { .. } | NativeWeightAdapter::Boft { .. } => base,
        _ => {
            let additive = additive_path(adapter, multiplier, module, backend, input, context)?;
            binary(backend, &base, &additive, BinaryOperation::Add, context)?
        }
    };
    let output = match adapter {
        NativeWeightAdapter::Oft { .. } => {
            apply_oft_transform(adapter, multiplier, module, backend, &combined, context)?
        }
        NativeWeightAdapter::Boft { .. } => {
            apply_boft_transform(adapter, multiplier, module, backend, &combined, context)?
        }
        _ => combined,
    };
    context.check()?;
    Ok(output)
}

fn additive_path(
    adapter: &NativeWeightAdapter,
    multiplier: f32,
    module: &ModuleTypeInfo,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    match adapter {
        NativeWeightAdapter::Lora {
            up,
            down,
            alpha,
            mid,
            ..
        } => {
            let down = down.materialize_tensor(backend, context)?;
            let up = up.materialize_tensor(backend, context)?;
            let rank = first_extent(&down, "LoRA down rank")?;
            let scale = alpha.unwrap_or(rank as f32) / rank as f32 * multiplier;
            let hidden = match mid {
                Some(mid) => {
                    let hidden = apply_layer(backend, input, &down, module, true, context)?;
                    let mid = mid.materialize_tensor(backend, context)?;
                    apply_layer(backend, &hidden, &mid, module, false, context)?
                }
                None => apply_layer(backend, input, &down, module, true, context)?,
            };
            let output = apply_pointwise_layer(backend, &hidden, &up, module, context)?;
            scalar(backend, &output, scale, BinaryOperation::Multiply, context)
        }
        NativeWeightAdapter::Loha {
            first_up,
            first_down,
            second_up,
            second_down,
            first_tucker,
            second_tucker,
            alpha,
            ..
        } => {
            let first_up = first_up.materialize_tensor(backend, context)?;
            let first_down = first_down.materialize_tensor(backend, context)?;
            let second_up = second_up.materialize_tensor(backend, context)?;
            let second_down = second_down.materialize_tensor(backend, context)?;
            let rank = first_extent(&first_down, "LoHa rank")?;
            let first = match first_tucker {
                Some(tucker) => tucker_rebuild(
                    backend,
                    &tucker.materialize_tensor(backend, context)?,
                    &first_up,
                    &first_down,
                    context,
                )?,
                None => matmul_two_dimensional(backend, &first_up, &first_down, context)?,
            };
            let second = match second_tucker {
                Some(tucker) => tucker_rebuild(
                    backend,
                    &tucker.materialize_tensor(backend, context)?,
                    &second_up,
                    &second_down,
                    context,
                )?,
                None => matmul_two_dimensional(backend, &second_up, &second_down, context)?,
            };
            let difference = binary(backend, &first, &second, BinaryOperation::Multiply, context)?;
            let scale = alpha.unwrap_or(rank as f32) / rank as f32 * multiplier;
            let difference = scalar(
                backend,
                &difference,
                scale,
                BinaryOperation::Multiply,
                context,
            )?;
            apply_layer(backend, input, &difference, module, true, context)
        }
        NativeWeightAdapter::Lokr {
            first,
            second,
            first_up,
            first_down,
            second_up,
            second_down,
            second_tucker,
            alpha,
            ..
        } => {
            let (first, first_rank) =
                rebuild_lokr_factor(backend, first, first_up, first_down, None, context)?;
            let (second, second_rank) = rebuild_lokr_factor(
                backend,
                second,
                second_up,
                second_down,
                second_tucker.as_ref(),
                context,
            )?;
            let rank = first_rank.or(second_rank);
            let scale = match (*alpha, rank) {
                (Some(alpha), Some(rank)) => alpha / rank as f32,
                _ => 1.0,
            } * multiplier;
            let (difference, event) = backend.kronecker_product(&first, &second, context)?;
            backend.wait_event(event, context)?;
            let difference = scalar(
                backend,
                &difference,
                scale,
                BinaryOperation::Multiply,
                context,
            )?;
            apply_layer(backend, input, &difference, module, true, context)
        }
        NativeWeightAdapter::Glora { .. }
        | NativeWeightAdapter::Oft { .. }
        | NativeWeightAdapter::Boft { .. } => Err(WeightAdapterError::InvalidPlan(
            "adapter has no ordinary additive path".into(),
        )),
    }
}

fn glora_paths(
    adapter: &NativeWeightAdapter,
    multiplier: f32,
    module: &ModuleTypeInfo,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), WeightAdapterError> {
    let NativeWeightAdapter::Glora {
        first_a,
        second_a,
        first_b,
        second_b,
        alpha,
        ..
    } = adapter
    else {
        return Err(WeightAdapterError::InvalidPlan(
            "GLoRA path requested for another adapter".into(),
        ));
    };
    let first_a = first_a.materialize_tensor(backend, context)?;
    let second_a = second_a.materialize_tensor(backend, context)?;
    let first_b = first_b.materialize_tensor(backend, context)?;
    let second_b = second_b.materialize_tensor(backend, context)?;
    let input_width = input
        .descriptor()
        .shape()
        .last()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("GLoRA input is scalar".into()))?;
    let old = glora_old_layout(&first_a, &second_a, &first_b, &second_b)?;
    let new = glora_new_layout(&first_a, &second_a, &first_b, &second_b)?;
    let old = old && !(new && second_a.descriptor().shape()[0] != input_width);
    let rank = if old {
        first_a.descriptor().shape()[0]
    } else {
        second_a.descriptor().shape()[0]
    };
    let scale = alpha.unwrap_or(rank as f32) / rank as f32 * multiplier;
    let (first_path, second_path) = if old {
        let first_path = apply_pointwise_layer(backend, input, &second_a, module, context)?;
        let first_path = apply_pointwise_layer(backend, &first_path, &first_a, module, context)?;
        let second_path = apply_layer(backend, input, &first_b, module, true, context)?;
        let second_path = apply_pointwise_layer(backend, &second_path, &second_b, module, context)?;
        (first_path, second_path)
    } else {
        let first_path = apply_pointwise_layer(backend, input, &first_a, module, context)?;
        let first_path = apply_pointwise_layer(backend, &first_path, &second_a, module, context)?;
        let second_path = apply_layer(backend, input, &second_b, module, true, context)?;
        let second_path = apply_pointwise_layer(backend, &second_path, &first_b, module, context)?;
        (first_path, second_path)
    };
    Ok((
        scalar(
            backend,
            &first_path,
            scale,
            BinaryOperation::Multiply,
            context,
        )?,
        scalar(
            backend,
            &second_path,
            scale,
            BinaryOperation::Multiply,
            context,
        )?,
    ))
}

fn trainable_dense(
    value: &AdapterTensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    name: &'static str,
) -> Result<Tensor, WeightAdapterError> {
    let AdapterTensor::Dense(tensor) = value else {
        return Err(WeightAdapterError::QuantizedTrainableTensor { name });
    };
    require_tensor_boundary(tensor, backend, context, name)?;
    require_f32(tensor, name)?;
    Ok(tensor.clone())
}

fn zero_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    let (tensor, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(tensor)
}

fn random_uniform_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    bound: f32,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    validate_finite("trainable uniform bound", bound)?;
    let count = checked_usize(checked_product(shape)?, "trainable tensor element count")?;
    let mut values = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        let unit = transaction.next_unit_f32(context.cancellation)?;
        values.try_push((unit * 2.0 - 1.0) * bound)?;
    }
    Ok(tensor_from_f32_with_backend_exact_native(
        backend,
        shape,
        &values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn random_normal_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    standard_deviation: f32,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    validate_finite("trainable normal standard deviation", standard_deviation)?;
    let count = checked_usize(checked_product(shape)?, "trainable tensor element count")?;
    let mut values = backend.workspace_vec(context, count)?;
    while values.len() < count {
        let pair = transaction.next_standard_normal_pair(context.cancellation)?;
        for value in pair {
            if values.len() == count {
                break;
            }
            values.try_push((value as f32) * standard_deviation)?;
        }
    }
    Ok(tensor_from_f32_with_backend_exact_native(
        backend,
        shape,
        &values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn trainable_scale(
    alpha: Option<f32>,
    rank: u64,
    family: &'static str,
) -> Result<f32, WeightAdapterError> {
    let alpha = alpha.ok_or_else(|| {
        WeightAdapterError::InvalidPlan(format!(
            "{family} trainable execution requires an explicit alpha"
        ))
    })?;
    validate_finite("trainable alpha", alpha)?;
    if rank == 0 {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{family} trainable rank must be positive"
        )));
    }
    let scale = alpha / rank as f32;
    validate_finite("trainable scale", scale)?;
    Ok(scale)
}

fn trainable_lora(
    up: &AdapterTensor,
    down: &AdapterTensor,
    alpha: Option<f32>,
    mid: Option<&AdapterTensor>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Tensor>, Tensor, TrainableBackwardPlan), WeightAdapterError> {
    let up = trainable_dense(up, backend, context, "LoRA trainable up")?;
    let down = trainable_dense(down, backend, context, "LoRA trainable down")?;
    let rank = first_extent(&down, "LoRA trainable rank")?;
    let scale = trainable_scale(alpha, rank, "LoRA")?;
    let mut inputs = vec![up.clone(), down.clone()];
    let difference = match mid {
        Some(mid) => {
            let mid = trainable_dense(mid, backend, context, "LoRA trainable mid")?;
            let difference = tucker_rebuild_from_conv(backend, &mid, &up, &down, context)?;
            inputs.push(mid);
            difference
        }
        None => matmul_two_dimensional(backend, &up, &down, context)?,
    };
    let difference = scalar(
        backend,
        &difference,
        scale,
        BinaryOperation::Multiply,
        context,
    )?;
    Ok((
        inputs,
        difference,
        TrainableBackwardPlan::Lora {
            alpha: alpha.unwrap_or(0.0),
            rank,
            has_mid: mid.is_some(),
            difference_shape: Vec::new(),
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn trainable_loha(
    first_up: &AdapterTensor,
    first_down: &AdapterTensor,
    second_up: &AdapterTensor,
    second_down: &AdapterTensor,
    first_tucker: Option<&AdapterTensor>,
    second_tucker: Option<&AdapterTensor>,
    alpha: Option<f32>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Tensor>, Tensor, TrainableBackwardPlan), WeightAdapterError> {
    let first_up = trainable_dense(first_up, backend, context, "LoHa first up")?;
    let first_down = trainable_dense(first_down, backend, context, "LoHa first down")?;
    let second_up = trainable_dense(second_up, backend, context, "LoHa second up")?;
    let second_down = trainable_dense(second_down, backend, context, "LoHa second down")?;
    let rank = first_extent(&first_down, "LoHa trainable rank")?;
    let scale = trainable_scale(alpha, rank, "LoHa")?;
    let mut inputs = vec![
        first_up.clone(),
        first_down.clone(),
        second_up.clone(),
        second_down.clone(),
    ];
    let (first, second, tucker) = match (first_tucker, second_tucker) {
        (None, None) => (
            matmul_two_dimensional(backend, &first_up, &first_down, context)?,
            matmul_two_dimensional(backend, &second_up, &second_down, context)?,
            false,
        ),
        (Some(first_tucker), Some(second_tucker)) => {
            let first_tucker =
                trainable_dense(first_tucker, backend, context, "LoHa first Tucker")?;
            let second_tucker =
                trainable_dense(second_tucker, backend, context, "LoHa second Tucker")?;
            let first = tucker_rebuild(backend, &first_tucker, &first_up, &first_down, context)?;
            let second =
                tucker_rebuild(backend, &second_tucker, &second_up, &second_down, context)?;
            inputs.extend([first_tucker, second_tucker]);
            (first, second, true)
        }
        _ => {
            return Err(WeightAdapterError::InvalidShape(
                "LoHa trainable Tucker tensors must be supplied as a pair".into(),
            ));
        }
    };
    let difference = binary(backend, &first, &second, BinaryOperation::Multiply, context)?;
    let difference = scalar(
        backend,
        &difference,
        scale,
        BinaryOperation::Multiply,
        context,
    )?;
    Ok((
        inputs,
        difference,
        TrainableBackwardPlan::Loha {
            alpha: alpha.unwrap_or(0.0),
            rank,
            tucker,
            difference_shape: Vec::new(),
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn trainable_lokr(
    first: Option<&AdapterTensor>,
    second: Option<&AdapterTensor>,
    first_up: Option<&AdapterTensor>,
    first_down: Option<&AdapterTensor>,
    second_up: Option<&AdapterTensor>,
    second_down: Option<&AdapterTensor>,
    second_tucker: Option<&AdapterTensor>,
    alpha: Option<f32>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Tensor>, Tensor, TrainableBackwardPlan), WeightAdapterError> {
    let alpha = alpha.ok_or_else(|| {
        WeightAdapterError::InvalidPlan(
            "LoKr trainable execution requires an explicit alpha".into(),
        )
    })?;
    validate_finite("LoKr trainable alpha", alpha)?;
    let mut inputs = Vec::new();
    let (first_value, first_plan) = trainable_lokr_factor(
        "LoKr first",
        first,
        first_up,
        first_down,
        None,
        alpha,
        &mut inputs,
        backend,
        context,
    )?;
    let (second_value, second_plan) = trainable_lokr_factor(
        "LoKr second",
        second,
        second_up,
        second_down,
        second_tucker,
        alpha,
        &mut inputs,
        backend,
        context,
    )?;
    let first_shape = first_value.descriptor().shape().to_vec();
    let mut expanded_shape = first_shape.clone();
    expanded_shape.extend(std::iter::repeat_n(
        1,
        second_value
            .descriptor()
            .rank()
            .saturating_sub(first_shape.len()),
    ));
    let first_expanded = reshape_read_only(&first_value, expanded_shape)?;
    let (difference, event) = backend.kronecker_product(&first_expanded, &second_value, context)?;
    backend.wait_event(event, context)?;
    Ok((
        inputs,
        difference,
        TrainableBackwardPlan::Lokr {
            first: first_plan,
            second: second_plan,
            difference_shape: Vec::new(),
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn trainable_lokr_factor(
    name: &'static str,
    direct: Option<&AdapterTensor>,
    up: Option<&AdapterTensor>,
    down: Option<&AdapterTensor>,
    tucker: Option<&AdapterTensor>,
    alpha: f32,
    inputs: &mut Vec<Tensor>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, LokrFactorPlan), WeightAdapterError> {
    if let Some(direct) = direct {
        let value = trainable_dense(direct, backend, context, name)?;
        let input = inputs.len() + 1;
        inputs.push(value.clone());
        return Ok((value, LokrFactorPlan::Direct { input }));
    }
    let up = trainable_dense(
        up.ok_or_else(|| WeightAdapterError::InvalidShape(format!("{name} up is missing")))?,
        backend,
        context,
        name,
    )?;
    let down = trainable_dense(
        down.ok_or_else(|| WeightAdapterError::InvalidShape(format!("{name} down is missing")))?,
        backend,
        context,
        name,
    )?;
    let up_index = inputs.len() + 1;
    inputs.push(up.clone());
    let down_index = inputs.len() + 1;
    inputs.push(down.clone());
    let rank = first_extent(&down, "LoKr trainable rank")?;
    let scale = trainable_scale(Some(alpha), rank, "LoKr")?;
    let (value, plan) = match tucker {
        Some(tucker) => {
            let tucker = trainable_dense(tucker, backend, context, name)?;
            let tucker_index = inputs.len() + 1;
            inputs.push(tucker.clone());
            (
                tucker_rebuild(backend, &tucker, &up, &down, context)?,
                LokrFactorPlan::Tucker {
                    tucker: tucker_index,
                    up: up_index,
                    down: down_index,
                    alpha,
                    rank,
                },
            )
        }
        None => (
            matmul_two_dimensional(backend, &up, &down, context)?,
            LokrFactorPlan::Matrix {
                up: up_index,
                down: down_index,
                alpha,
                rank,
            },
        ),
    };
    Ok((
        scalar(backend, &value, scale, BinaryOperation::Multiply, context)?,
        plan,
    ))
}

fn trainable_oft(
    blocks: &AdapterTensor,
    rescale: Option<&AdapterTensor>,
    constraint: f32,
    base_weight: &Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Tensor>, Tensor, TrainableBackwardPlan), WeightAdapterError> {
    let blocks = trainable_dense(blocks, backend, context, "OFT trainable blocks")?;
    let shape = blocks.descriptor().shape().to_vec();
    let [block_count, block_size, columns] = shape.as_slice() else {
        return Err(WeightAdapterError::InvalidShape(
            "OFT trainable blocks must have rank three".into(),
        ));
    };
    if block_size != columns {
        return Err(WeightAdapterError::InvalidShape(
            "OFT trainable blocks must be square".into(),
        ));
    }
    let transposed = permute_last_two_view(&blocks)?;
    let skew = binary(
        backend,
        &blocks,
        &transposed,
        BinaryOperation::Subtract,
        context,
    )?;
    let raw_norm = frobenius_norm(backend, &skew, context)?;
    let adjusted_norm = raw_norm + 1.0e-8;
    let constrained = constraint > 0.0 && adjusted_norm > constraint;
    let skew = if constrained {
        scalar(
            backend,
            &skew,
            constraint / adjusted_norm,
            BinaryOperation::Multiply,
            context,
        )?
    } else {
        skew
    };
    let identity = identity_batch(backend, *block_count, *block_size, context)?;
    let plus = binary(backend, &identity, &skew, BinaryOperation::Add, context)?;
    let minus = binary(
        backend,
        &identity,
        &skew,
        BinaryOperation::Subtract,
        context,
    )?;
    let (inverse, event) = backend.matrix_inverse(&minus, context)?;
    backend.wait_event(event, context)?;
    let rotation = batch_matrix_multiply(backend, &plus, &inverse, context)?;
    let base_shape = base_weight.descriptor().shape();
    let output_channels = base_shape.first().copied().ok_or_else(|| {
        WeightAdapterError::InvalidShape("OFT trainable base weight is scalar".into())
    })?;
    if output_channels
        != block_count
            .checked_mul(*block_size)
            .ok_or(WeightAdapterError::ShapeOverflow)?
    {
        return Err(WeightAdapterError::InvalidShape(
            "OFT trainable block geometry does not match base output channels".into(),
        ));
    }
    let mut blocked_shape = vec![*block_count, *block_size];
    blocked_shape.extend_from_slice(&base_shape[1..]);
    let blocked_weight = reshape_read_only(base_weight, blocked_shape.clone())?;
    let rotated = einsum_with_context_exact_native(
        backend,
        "knm,kn...->km...",
        &[rotation, blocked_weight],
        context,
    )
    .map_err(|error| WeightAdapterError::CanonicalOperation {
        operation: "OFT trainable rotation",
        reason: error.to_string(),
    })?;
    let rotated = reshape_read_only(&rotated, base_shape.to_vec())?;
    let mut inputs = vec![blocks];
    let output = match rescale {
        Some(rescale) => {
            let rescale = trainable_dense(rescale, backend, context, "OFT trainable rescale")?;
            inputs.push(rescale.clone());
            binary(
                backend,
                &rotated,
                &rescale,
                BinaryOperation::Multiply,
                context,
            )?
        }
        None => rotated,
    };
    Ok((
        inputs,
        output,
        TrainableBackwardPlan::Oft {
            constraint,
            has_rescale: rescale.is_some(),
            raw_norm,
            constrained,
            difference_shape: blocked_shape,
        },
    ))
}

impl BackwardRule for TrainableWeightBackwardRule {
    fn vjp(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &sim_types::CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Err(AutogradError::InvalidGraph {
            reason: "weight-adapter trainable reverse requires the caller's CPU execution context"
                .into(),
        })
    }

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved_tensors: &[SavedTensor],
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        execution
            .cancellation
            .check()
            .map_err(|_| AutogradError::Cancelled)?;
        let Some(output_gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None; self.input_count]);
        };
        if output_gradient.descriptor().shape() != self.output_shape {
            return Err(AutogradError::InvalidGraph {
                reason: "weight-adapter output gradient has the wrong shape".into(),
            });
        }
        let gradients = match &self.plan {
            TrainableBackwardPlan::Lora {
                alpha,
                rank,
                has_mid,
                difference_shape,
            } => trainable_lora_vjp(
                saved_tensors,
                &output_gradient,
                *alpha,
                *rank,
                *has_mid,
                difference_shape,
                backend,
                execution,
            )?,
            TrainableBackwardPlan::Loha {
                alpha,
                rank,
                tucker,
                difference_shape,
            } => trainable_loha_vjp(
                saved_tensors,
                &output_gradient,
                *alpha,
                *rank,
                *tucker,
                difference_shape,
                backend,
                execution,
            )?,
            TrainableBackwardPlan::Lokr {
                first,
                second,
                difference_shape,
            } => trainable_lokr_vjp(
                saved_tensors,
                &output_gradient,
                first,
                second,
                difference_shape,
                self.input_count,
                backend,
                execution,
            )?,
            TrainableBackwardPlan::Oft {
                constraint,
                has_rescale,
                raw_norm,
                constrained,
                difference_shape,
            } => trainable_oft_vjp(
                saved_tensors,
                &output_gradient,
                *constraint,
                *has_rescale,
                *raw_norm,
                *constrained,
                difference_shape,
                backend,
                execution,
            )?,
        };
        if gradients.len() != self.input_count {
            return Err(AutogradError::GradientArity {
                expected: self.input_count,
                actual: gradients.len(),
            });
        }
        execution
            .cancellation
            .check()
            .map_err(|_| AutogradError::Cancelled)?;
        Ok(gradients.into_iter().map(Some).collect())
    }

    fn symbol(&self) -> &'static str {
        "WeightAdapterTrainBase"
    }
}

fn trainable_lora_vjp(
    saved: &[SavedTensor],
    output_gradient: &Tensor,
    alpha: f32,
    rank: u64,
    has_mid: bool,
    difference_shape: &[u64],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, AutogradError> {
    let up = saved_tensor_at(saved, 0, "LoRA up")?;
    let down = saved_tensor_at(saved, 1, "LoRA down")?;
    let gradient = reshape_autograd(output_gradient, difference_shape)?;
    let gradient = scale_autograd(backend, &gradient, alpha / rank as f32, execution)?;
    let mut gradients = vec![output_gradient.clone()];
    if has_mid {
        let mid = saved_tensor_at(saved, 2, "LoRA mid")?;
        let operands = [mid.clone(), up.clone(), down.clone()];
        let candidates = einsum_vjp_with_context_exact_native(
            backend,
            "mn...,im,nj->ij...",
            &operands,
            &gradient,
            execution,
        )
        .map_err(|error| autograd_operation_error("LoRA Tucker VJP", error))?;
        if candidates.operands.len() != 3 {
            return Err(AutogradError::GradientArity {
                expected: 3,
                actual: candidates.operands.len(),
            });
        }
        gradients.extend([
            candidates.operands[1].clone(),
            candidates.operands[2].clone(),
            candidates.operands[0].clone(),
        ]);
    } else {
        let candidates = mm_vjp_with_context_exact_native(backend, up, down, &gradient, execution)
            .map_err(|error| autograd_operation_error("LoRA matrix VJP", error))?;
        gradients.extend([candidates.input, candidates.other]);
    }
    Ok(gradients)
}

#[allow(clippy::too_many_arguments)]
fn trainable_loha_vjp(
    saved: &[SavedTensor],
    output_gradient: &Tensor,
    alpha: f32,
    rank: u64,
    tucker: bool,
    difference_shape: &[u64],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, AutogradError> {
    let first_up = saved_tensor_at(saved, 0, "LoHa first up")?;
    let first_down = saved_tensor_at(saved, 1, "LoHa first down")?;
    let second_up = saved_tensor_at(saved, 2, "LoHa second up")?;
    let second_down = saved_tensor_at(saved, 3, "LoHa second down")?;
    let (first, second) = if tucker {
        let first_tucker = saved_tensor_at(saved, 4, "LoHa first Tucker")?;
        let second_tucker = saved_tensor_at(saved, 5, "LoHa second Tucker")?;
        (
            einsum_with_context_exact_native(
                backend,
                "ij...,jr,ip->pr...",
                &[first_tucker.clone(), first_down.clone(), first_up.clone()],
                execution,
            )
            .map_err(|error| autograd_operation_error("LoHa first Tucker", error))?,
            einsum_with_context_exact_native(
                backend,
                "ij...,jr,ip->pr...",
                &[
                    second_tucker.clone(),
                    second_down.clone(),
                    second_up.clone(),
                ],
                execution,
            )
            .map_err(|error| autograd_operation_error("LoHa second Tucker", error))?,
        )
    } else {
        (
            matmul_two_dimensional(backend, first_up, first_down, execution)
                .map_err(|error| autograd_operation_error("LoHa first matrix", error))?,
            matmul_two_dimensional(backend, second_up, second_down, execution)
                .map_err(|error| autograd_operation_error("LoHa second matrix", error))?,
        )
    };
    let gradient = reshape_autograd(output_gradient, difference_shape)?;
    let gradient = scale_autograd(backend, &gradient, alpha / rank as f32, execution)?;
    let products =
        mul_vjp_with_context_exact_native(backend, &first, &second, &gradient, execution)
            .map_err(|error| autograd_operation_error("LoHa product VJP", error))?;
    let mut gradients = vec![output_gradient.clone()];
    if tucker {
        let first_tucker = saved_tensor_at(saved, 4, "LoHa first Tucker")?;
        let second_tucker = saved_tensor_at(saved, 5, "LoHa second Tucker")?;
        let first_candidates = einsum_vjp_with_context_exact_native(
            backend,
            "ij...,jr,ip->pr...",
            &[first_tucker.clone(), first_down.clone(), first_up.clone()],
            &products.left,
            execution,
        )
        .map_err(|error| autograd_operation_error("LoHa first Tucker VJP", error))?;
        let second_candidates = einsum_vjp_with_context_exact_native(
            backend,
            "ij...,jr,ip->pr...",
            &[
                second_tucker.clone(),
                second_down.clone(),
                second_up.clone(),
            ],
            &products.right,
            execution,
        )
        .map_err(|error| autograd_operation_error("LoHa second Tucker VJP", error))?;
        if first_candidates.operands.len() != 3 || second_candidates.operands.len() != 3 {
            return Err(AutogradError::GradientArity {
                expected: 6,
                actual: first_candidates.operands.len() + second_candidates.operands.len(),
            });
        }
        gradients.extend([
            first_candidates.operands[2].clone(),
            first_candidates.operands[1].clone(),
            second_candidates.operands[2].clone(),
            second_candidates.operands[1].clone(),
            first_candidates.operands[0].clone(),
            second_candidates.operands[0].clone(),
        ]);
    } else {
        let first_candidates = mm_vjp_with_context_exact_native(
            backend,
            first_up,
            first_down,
            &products.left,
            execution,
        )
        .map_err(|error| autograd_operation_error("LoHa first matrix VJP", error))?;
        let second_candidates = mm_vjp_with_context_exact_native(
            backend,
            second_up,
            second_down,
            &products.right,
            execution,
        )
        .map_err(|error| autograd_operation_error("LoHa second matrix VJP", error))?;
        gradients.extend([
            first_candidates.input,
            first_candidates.other,
            second_candidates.input,
            second_candidates.other,
        ]);
    }
    Ok(gradients)
}

#[allow(clippy::too_many_arguments)]
fn trainable_lokr_vjp(
    saved: &[SavedTensor],
    output_gradient: &Tensor,
    first_plan: &LokrFactorPlan,
    second_plan: &LokrFactorPlan,
    difference_shape: &[u64],
    input_count: usize,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, AutogradError> {
    let first = rebuild_lokr_trainable_factor(saved, first_plan, backend, execution)?;
    let second = rebuild_lokr_trainable_factor(saved, second_plan, backend, execution)?;
    let first_shape = first.descriptor().shape().to_vec();
    let mut expanded_shape = first_shape.clone();
    expanded_shape.extend(std::iter::repeat_n(
        1,
        second.descriptor().rank().saturating_sub(first_shape.len()),
    ));
    let first_expanded = reshape_autograd(&first, &expanded_shape)?;
    let gradient = reshape_autograd(output_gradient, difference_shape)?;
    let candidates =
        kron_vjp_with_context_exact_native(backend, &first_expanded, &second, &gradient, execution)
            .map_err(|error| autograd_operation_error("LoKr Kronecker VJP", error))?;
    let first_gradient = reshape_autograd(&candidates.input, &first_shape)?;
    let mut gradients = (0..input_count)
        .map(|_| None)
        .collect::<Vec<Option<Tensor>>>();
    gradients[0] = Some(output_gradient.clone());
    apply_lokr_factor_vjp(
        saved,
        first_plan,
        &first_gradient,
        &mut gradients,
        backend,
        execution,
    )?;
    apply_lokr_factor_vjp(
        saved,
        second_plan,
        &candidates.other,
        &mut gradients,
        backend,
        execution,
    )?;
    gradients
        .into_iter()
        .enumerate()
        .map(|(index, gradient)| {
            gradient.ok_or_else(|| AutogradError::InvalidGraph {
                reason: format!("LoKr input gradient {index} was not produced"),
            })
        })
        .collect()
}

fn rebuild_lokr_trainable_factor(
    saved: &[SavedTensor],
    plan: &LokrFactorPlan,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, AutogradError> {
    match plan {
        LokrFactorPlan::Direct { input } => Ok(saved_input(saved, *input, false)?.clone()),
        LokrFactorPlan::Matrix {
            up,
            down,
            alpha,
            rank,
        } => {
            let value = matmul_two_dimensional(
                backend,
                saved_input(saved, *up, false)?,
                saved_input(saved, *down, false)?,
                execution,
            )
            .map_err(|error| autograd_operation_error("LoKr matrix rebuild", error))?;
            scale_autograd(backend, &value, *alpha / *rank as f32, execution)
        }
        LokrFactorPlan::Tucker {
            tucker,
            up,
            down,
            alpha,
            rank,
        } => {
            let value = einsum_with_context_exact_native(
                backend,
                "ij...,jr,ip->pr...",
                &[
                    saved_input(saved, *tucker, false)?.clone(),
                    saved_input(saved, *down, false)?.clone(),
                    saved_input(saved, *up, false)?.clone(),
                ],
                execution,
            )
            .map_err(|error| autograd_operation_error("LoKr Tucker rebuild", error))?;
            scale_autograd(backend, &value, *alpha / *rank as f32, execution)
        }
    }
}

fn apply_lokr_factor_vjp(
    saved: &[SavedTensor],
    plan: &LokrFactorPlan,
    gradient: &Tensor,
    gradients: &mut [Option<Tensor>],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<(), AutogradError> {
    match plan {
        LokrFactorPlan::Direct { input } => set_input_gradient(gradients, *input, gradient.clone()),
        LokrFactorPlan::Matrix {
            up,
            down,
            alpha,
            rank,
        } => {
            let gradient = scale_autograd(backend, gradient, *alpha / *rank as f32, execution)?;
            let candidates = mm_vjp_with_context_exact_native(
                backend,
                saved_input(saved, *up, false)?,
                saved_input(saved, *down, false)?,
                &gradient,
                execution,
            )
            .map_err(|error| autograd_operation_error("LoKr matrix VJP", error))?;
            set_input_gradient(gradients, *up, candidates.input)?;
            set_input_gradient(gradients, *down, candidates.other)
        }
        LokrFactorPlan::Tucker {
            tucker,
            up,
            down,
            alpha,
            rank,
        } => {
            let gradient = scale_autograd(backend, gradient, *alpha / *rank as f32, execution)?;
            let candidates = einsum_vjp_with_context_exact_native(
                backend,
                "ij...,jr,ip->pr...",
                &[
                    saved_input(saved, *tucker, false)?.clone(),
                    saved_input(saved, *down, false)?.clone(),
                    saved_input(saved, *up, false)?.clone(),
                ],
                &gradient,
                execution,
            )
            .map_err(|error| autograd_operation_error("LoKr Tucker VJP", error))?;
            if candidates.operands.len() != 3 {
                return Err(AutogradError::GradientArity {
                    expected: 3,
                    actual: candidates.operands.len(),
                });
            }
            set_input_gradient(gradients, *tucker, candidates.operands[0].clone())?;
            set_input_gradient(gradients, *down, candidates.operands[1].clone())?;
            set_input_gradient(gradients, *up, candidates.operands[2].clone())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn trainable_oft_vjp(
    saved: &[SavedTensor],
    output_gradient: &Tensor,
    constraint: f32,
    has_rescale: bool,
    raw_norm: f32,
    constrained: bool,
    blocked_shape: &[u64],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, AutogradError> {
    let base = saved_input(saved, 0, true)?;
    let blocks = saved_input(saved, 1, true)?;
    let transposed = permute_last_two_view(blocks)
        .map_err(|error| autograd_operation_error("OFT transpose", error))?;
    let raw_skew = binary(
        backend,
        blocks,
        &transposed,
        BinaryOperation::Subtract,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT skew", error))?;
    let adjusted_norm = raw_norm + 1.0e-8;
    let skew = if constrained {
        scale_autograd(backend, &raw_skew, constraint / adjusted_norm, execution)?
    } else {
        raw_skew.clone()
    };
    let shape = blocks.descriptor().shape();
    let [block_count, block_size, _] = shape else {
        return Err(AutogradError::InvalidGraph {
            reason: "OFT saved blocks are not rank three".into(),
        });
    };
    let identity = identity_batch(backend, *block_count, *block_size, execution)
        .map_err(|error| autograd_operation_error("OFT identity", error))?;
    let plus = binary(backend, &identity, &skew, BinaryOperation::Add, execution)
        .map_err(|error| autograd_operation_error("OFT plus", error))?;
    let minus = binary(
        backend,
        &identity,
        &skew,
        BinaryOperation::Subtract,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT minus", error))?;
    let (inverse, event) = backend
        .matrix_inverse(&minus, execution)
        .map_err(AutogradError::Tensor)?;
    backend
        .wait_event(event, execution)
        .map_err(AutogradError::Tensor)?;
    let rotation = batch_matrix_multiply(backend, &plus, &inverse, execution)
        .map_err(|error| autograd_operation_error("OFT rotation", error))?;
    let blocked_base = reshape_autograd(base, blocked_shape)?;
    let rotated_blocked = einsum_with_context_exact_native(
        backend,
        "knm,kn...->km...",
        &[rotation.clone(), blocked_base.clone()],
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT weight rotation", error))?;
    let rotated = reshape_autograd(&rotated_blocked, base.descriptor().shape())?;
    let (rotated_gradient, rescale_gradient) = if has_rescale {
        let rescale = saved_input(saved, 2, true)?;
        let candidates = mul_vjp_with_context_exact_native(
            backend,
            &rotated,
            rescale,
            output_gradient,
            execution,
        )
        .map_err(|error| autograd_operation_error("OFT rescale VJP", error))?;
        (candidates.left, Some(candidates.right))
    } else {
        (output_gradient.clone(), None)
    };
    let rotated_gradient =
        reshape_autograd(&rotated_gradient, rotated_blocked.descriptor().shape())?;
    let rotation_candidates = einsum_vjp_with_context_exact_native(
        backend,
        "knm,kn...->km...",
        &[rotation, blocked_base],
        &rotated_gradient,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT rotation VJP", error))?;
    if rotation_candidates.operands.len() != 2 {
        return Err(AutogradError::GradientArity {
            expected: 2,
            actual: rotation_candidates.operands.len(),
        });
    }
    let base_gradient =
        reshape_autograd(&rotation_candidates.operands[1], base.descriptor().shape())?;
    let cayley_candidates = einsum_vjp_with_context_exact_native(
        backend,
        "bij,bjk->bik",
        &[plus, inverse],
        &rotation_candidates.operands[0],
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT Cayley product VJP", error))?;
    if cayley_candidates.operands.len() != 2 {
        return Err(AutogradError::GradientArity {
            expected: 2,
            actual: cayley_candidates.operands.len(),
        });
    }
    let minus_gradient = inverse_vjp_with_context_exact_native(
        backend,
        &minus,
        &cayley_candidates.operands[1],
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT inverse VJP", error))?;
    let negative_minus = scale_autograd(backend, &minus_gradient, -1.0, execution)?;
    let skew_gradient = binary(
        backend,
        &cayley_candidates.operands[0],
        &negative_minus,
        BinaryOperation::Add,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT skew gradient", error))?;
    let raw_skew_gradient = if constrained && raw_norm > 0.0 {
        constrained_skew_vjp(
            backend,
            &raw_skew,
            &skew_gradient,
            constraint,
            raw_norm,
            execution,
        )?
    } else if constrained {
        scale_autograd(
            backend,
            &skew_gradient,
            constraint / adjusted_norm,
            execution,
        )?
    } else {
        skew_gradient
    };
    let transposed_gradient = permute_last_two_view(&raw_skew_gradient)
        .map_err(|error| autograd_operation_error("OFT skew-gradient transpose", error))?;
    let blocks_gradient = binary(
        backend,
        &raw_skew_gradient,
        &transposed_gradient,
        BinaryOperation::Subtract,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT blocks gradient", error))?;
    let mut gradients = vec![base_gradient, blocks_gradient];
    if let Some(rescale_gradient) = rescale_gradient {
        gradients.push(rescale_gradient);
    }
    Ok(gradients)
}

fn constrained_skew_vjp(
    backend: &CpuBackend,
    skew: &Tensor,
    gradient: &Tensor,
    constraint: f32,
    raw_norm: f32,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, AutogradError> {
    let adjusted_norm = raw_norm + 1.0e-8;
    let direct = scale_autograd(backend, gradient, constraint / adjusted_norm, execution)?;
    let product = binary(
        backend,
        skew,
        gradient,
        BinaryOperation::Multiply,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT constraint dot product", error))?;
    let dot = sum_all_tensor(backend, &product, execution)
        .map_err(|error| autograd_operation_error("OFT constraint reduction", error))?;
    let projected = binary(backend, skew, &dot, BinaryOperation::Multiply, execution)
        .map_err(|error| autograd_operation_error("OFT constraint projection", error))?;
    let coefficient = -constraint / (adjusted_norm * adjusted_norm * raw_norm);
    let projected = scale_autograd(backend, &projected, coefficient, execution)?;
    binary(
        backend,
        &direct,
        &projected,
        BinaryOperation::Add,
        execution,
    )
    .map_err(|error| autograd_operation_error("OFT constraint VJP", error))
}

fn sum_all_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let dimensions = (0..input.descriptor().rank())
        .map(|dimension| checked_u64(dimension, "autograd reduction dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let specification = ReductionSpec {
        operation: ReductionOperation::Sum,
        dimensions,
        keep_dimensions: false,
        accumulation_dtype: Some(DType::F32),
        correction: 0,
    };
    let descriptor = output_descriptor(input, Vec::new(), execution)?;
    let (output, event) = backend.reduction(&specification, input, descriptor, execution)?;
    backend.wait_event(event, execution)?;
    Ok(output)
}

fn saved_tensor_at<'a>(
    saved: &'a [SavedTensor],
    index: usize,
    name: &'static str,
) -> Result<&'a Tensor, AutogradError> {
    saved
        .get(index)
        .map(SavedTensor::tensor)
        .ok_or_else(|| AutogradError::InvalidGraph {
            reason: format!("weight-adapter saved tensor {name} is missing"),
        })
}

fn saved_input(
    saved: &[SavedTensor],
    input: usize,
    base_saved: bool,
) -> Result<&Tensor, AutogradError> {
    let saved_index = if base_saved {
        input
    } else {
        input
            .checked_sub(1)
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "weight-adapter base input was not saved".into(),
            })?
    };
    saved_tensor_at(saved, saved_index, "input")
}

fn set_input_gradient(
    gradients: &mut [Option<Tensor>],
    input: usize,
    gradient: Tensor,
) -> Result<(), AutogradError> {
    let slot = gradients
        .get_mut(input)
        .ok_or_else(|| AutogradError::InvalidGraph {
            reason: "weight-adapter gradient input index is out of bounds".into(),
        })?;
    if slot.is_some() {
        return Err(AutogradError::InvalidGraph {
            reason: "weight-adapter produced a duplicate input gradient".into(),
        });
    }
    *slot = Some(gradient);
    Ok(())
}

fn reshape_autograd(input: &Tensor, shape: &[u64]) -> Result<Tensor, AutogradError> {
    reshape_read_only(input, shape.to_vec())
        .map_err(|error| autograd_operation_error("weight-adapter reshape", error))
}

fn scale_autograd(
    backend: &CpuBackend,
    input: &Tensor,
    scale: f32,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, AutogradError> {
    scalar(backend, input, scale, BinaryOperation::Multiply, execution)
        .map_err(|error| autograd_operation_error("weight-adapter scale", error))
}

fn autograd_operation_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> AutogradError {
    AutogradError::InvalidGraph {
        reason: format!("{operation} failed: {error}"),
    }
}

fn apply_oft_transform(
    adapter: &NativeWeightAdapter,
    multiplier: f32,
    module: &ModuleTypeInfo,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let NativeWeightAdapter::Oft {
        blocks,
        rescale,
        constraint,
        ..
    } = adapter
    else {
        return Err(WeightAdapterError::InvalidPlan(
            "OFT transform requested for another adapter".into(),
        ));
    };
    let blocks = blocks.materialize_tensor(backend, context)?;
    let rotation = cayley_rotations(backend, &blocks, constraint.unwrap_or(0.0), context)?;
    let rotation = interpolate_rotation(backend, &rotation, multiplier, context)?;
    let shape = blocks.descriptor().shape();
    let output = apply_block_rotation(
        backend, input, &rotation, shape[0], shape[1], module, context,
    )?;
    apply_rescale(backend, &output, rescale.as_ref(), context)
}

fn apply_boft_transform(
    adapter: &NativeWeightAdapter,
    multiplier: f32,
    module: &ModuleTypeInfo,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let NativeWeightAdapter::Boft {
        blocks,
        rescale,
        constraint,
        ..
    } = adapter
    else {
        return Err(WeightAdapterError::InvalidPlan(
            "BOFT transform requested for another adapter".into(),
        ));
    };
    let blocks = blocks.materialize_tensor(backend, context)?;
    let rotation = cayley_rotations(backend, &blocks, constraint.unwrap_or(0.0), context)?;
    let rotation = interpolate_rotation(backend, &rotation, multiplier, context)?;
    let shape = blocks.descriptor().shape();
    let stages = checked_usize(shape[0], "BOFT stage count")?;
    let block_count = shape[1];
    let block_size = shape[2];
    let mut output = channels_last_contiguous(backend, input, module, context)?;
    let channels = output
        .descriptor()
        .shape()
        .last()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("BOFT output is scalar".into()))?;
    if channels
        != block_count
            .checked_mul(block_size)
            .ok_or(WeightAdapterError::ShapeOverflow)?
    {
        return Err(WeightAdapterError::InvalidShape(
            "BOFT output channels do not match the block geometry".into(),
        ));
    }
    let half_block = block_size / 2;
    for stage in 0..stages {
        context.check()?;
        let grouping = 2_u64
            .checked_pow(u32::try_from(stage).map_err(|_| WeightAdapterError::ShapeOverflow)?)
            .and_then(|value| value.checked_mul(half_block))
            .ok_or(WeightAdapterError::ShapeOverflow)?;
        let outer = channels
            .checked_div(
                2_u64
                    .checked_mul(grouping)
                    .ok_or(WeightAdapterError::ShapeOverflow)?,
            )
            .ok_or(WeightAdapterError::ShapeOverflow)?;
        let prefix = prefix_shape(output.descriptor().shape())?;
        let mut arranged_shape = prefix.clone();
        arranged_shape.extend([outer, 2, grouping]);
        output = reshape_read_only(&output, arranged_shape)?;
        output = permute_last_two(backend, &output, context)?;
        let mut flattened = prefix.clone();
        flattened.extend([block_count, block_size]);
        output = reshape_read_only(&output, flattened)?;
        let stage_rotation_descriptor = output_descriptor(
            &rotation,
            vec![block_count, block_size, block_size],
            context,
        )?;
        let stage_start = checked_u64(stage, "BOFT stage")?
            .checked_mul(block_count)
            .ok_or(WeightAdapterError::ShapeOverflow)?;
        let (stage_rotation, event) = backend.indexing(
            &IndexSpec::Narrow {
                dimension: 0,
                start: i64::try_from(stage_start).map_err(|_| WeightAdapterError::ShapeOverflow)?,
                length: block_count,
            },
            std::slice::from_ref(&rotation),
            stage_rotation_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        output = apply_batched_row_rotation(backend, &output, &stage_rotation, context)?;
        let mut unflattened = prefix.clone();
        unflattened.extend([outer, grouping, 2]);
        output = reshape_read_only(&output, unflattened)?;
        output = permute_last_two(backend, &output, context)?;
        let mut restored = prefix.clone();
        restored.push(channels);
        output = reshape_read_only(&output, restored)?;
    }
    output = apply_rescale(backend, &output, rescale.as_ref(), context)?;
    restore_channels(backend, &output, module, context)
}

fn cayley_rotations(
    backend: &dyn TensorBackend,
    blocks: &Tensor,
    constraint: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    validate_optional_nonnegative("orthogonal constraint", Some(constraint))?;
    let shape = blocks.descriptor().shape();
    if shape.len() < 3 || shape[shape.len() - 1] != shape[shape.len() - 2] {
        return Err(WeightAdapterError::InvalidShape(
            "orthogonal blocks must be a square matrix batch".into(),
        ));
    }
    let size = shape[shape.len() - 1];
    let batch = checked_product(&shape[..shape.len() - 2])?;
    let blocks = reshape_read_only(blocks, vec![batch, size, size])?;
    let transposed = permute_last_two_view(&blocks)?;
    let mut skew = binary(
        backend,
        &blocks,
        &transposed,
        BinaryOperation::Subtract,
        context,
    )?;
    if constraint > 0.0 {
        let norm = frobenius_norm(backend, &skew, context)? + 1.0e-8;
        if norm > constraint {
            skew = scalar(
                backend,
                &skew,
                constraint / norm,
                BinaryOperation::Multiply,
                context,
            )?;
        }
    }
    let identity = identity_batch(backend, batch, size, context)?;
    let plus = binary(backend, &identity, &skew, BinaryOperation::Add, context)?;
    let minus = binary(
        backend,
        &identity,
        &skew,
        BinaryOperation::Subtract,
        context,
    )?;
    let (inverse, event) = backend.matrix_inverse(&minus, context)?;
    backend.wait_event(event, context)?;
    batch_matrix_multiply(backend, &plus, &inverse, context)
}

fn interpolate_rotation(
    backend: &dyn TensorBackend,
    rotation: &Tensor,
    multiplier: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if multiplier == 1.0 {
        return Ok(rotation.clone());
    }
    let shape = rotation.descriptor().shape();
    let [batch, rows, columns] = shape else {
        return Err(WeightAdapterError::InvalidShape(
            "rotation must have rank three".into(),
        ));
    };
    if rows != columns {
        return Err(WeightAdapterError::InvalidShape(
            "rotation must be square".into(),
        ));
    }
    let identity = identity_batch(backend, *batch, *rows, context)?;
    let rotation = scalar(
        backend,
        rotation,
        multiplier,
        BinaryOperation::Multiply,
        context,
    )?;
    let identity = scalar(
        backend,
        &identity,
        1.0 - multiplier,
        BinaryOperation::Multiply,
        context,
    )?;
    binary(backend, &rotation, &identity, BinaryOperation::Add, context)
}

fn apply_block_rotation(
    backend: &dyn TensorBackend,
    input: &Tensor,
    rotation: &Tensor,
    block_count: u64,
    block_size: u64,
    module: &ModuleTypeInfo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let input = channels_last_contiguous(backend, input, module, context)?;
    let channels = input
        .descriptor()
        .shape()
        .last()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("OFT output is scalar".into()))?;
    if channels
        != block_count
            .checked_mul(block_size)
            .ok_or(WeightAdapterError::ShapeOverflow)?
    {
        return Err(WeightAdapterError::InvalidShape(
            "OFT output channels do not match the block geometry".into(),
        ));
    }
    let mut blocked_shape = prefix_shape(input.descriptor().shape())?;
    blocked_shape.extend([block_count, block_size]);
    let blocked = reshape_read_only(&input, blocked_shape)?;
    let output = apply_batched_row_rotation(backend, &blocked, rotation, context)?;
    let mut restored = prefix_shape(output.descriptor().shape())?;
    restored.pop();
    restored.push(channels);
    let output = reshape_read_only(&output, restored)?;
    restore_channels(backend, &output, module, context)
}

fn apply_batched_row_rotation(
    backend: &dyn TensorBackend,
    input: &Tensor,
    rotation: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let input_shape = input.descriptor().shape();
    if input_shape.len() < 2 {
        return Err(WeightAdapterError::InvalidShape(
            "block rotation input must have rank at least two".into(),
        ));
    }
    let block_count = input_shape[input_shape.len() - 2];
    let block_size = input_shape[input_shape.len() - 1];
    if rotation.descriptor().shape() != [block_count, block_size, block_size] {
        return Err(WeightAdapterError::InvalidShape(
            "block rotation matrix shape is incompatible".into(),
        ));
    }
    let rows = checked_product(&input_shape[..input_shape.len() - 2])?;
    let row_blocks = rows
        .checked_mul(block_count)
        .ok_or(WeightAdapterError::ShapeOverflow)?;
    let input = reshape_read_only(input, vec![row_blocks, 1, block_size])?;
    let rotation = repeat_matrix_batch(backend, rotation, rows, context)?;
    let output = batch_matrix_multiply(backend, &input, &rotation, context)?;
    let mut shape = input_shape.to_vec();
    shape.truncate(shape.len() - 2);
    shape.extend([block_count, block_size]);
    reshape_read_only(&output, shape)
}

fn apply_rescale(
    backend: &dyn TensorBackend,
    input: &Tensor,
    rescale: Option<&AdapterTensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let Some(rescale) = rescale else {
        return Ok(input.clone());
    };
    let rescale = rescale.materialize_tensor(backend, context)?;
    let channels = input
        .descriptor()
        .shape()
        .last()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("rescale input is scalar".into()))?;
    if rescale.descriptor().element_count()? != channels {
        return Err(WeightAdapterError::InvalidShape(
            "orthogonal adapter rescale width is incompatible".into(),
        ));
    }
    let rescale = reshape_read_only(&rescale, vec![channels])?;
    binary(backend, input, &rescale, BinaryOperation::Multiply, context)
}

fn apply_layer(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    module: &ModuleTypeInfo,
    use_module_geometry: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    match module.kind {
        LayerKind::Linear => linear(backend, input, weight, context),
        LayerKind::Convolution { dimensions } => {
            let weight = convolution_weight(weight, dimensions, module)?;
            convolution(
                backend,
                input,
                &weight,
                module,
                use_module_geometry,
                context,
            )
        }
    }
}

fn apply_pointwise_layer(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    module: &ModuleTypeInfo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    match module.kind {
        LayerKind::Linear => linear(backend, input, weight, context),
        LayerKind::Convolution { dimensions } => {
            let weight = pointwise_convolution_weight(weight, dimensions)?;
            convolution(backend, input, &weight, module, false, context)
        }
    }
}

fn linear(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    require_f32(input, "linear input")?;
    require_f32(weight, "linear weight")?;
    require_same_device_stream(input, weight, backend, context)?;
    let [output_width, input_width] = weight.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "linear weight must have rank two".into(),
        ));
    };
    let input_shape = input.descriptor().shape();
    let actual_width = input_shape
        .last()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("linear input is scalar".into()))?;
    if actual_width != *input_width {
        return Err(WeightAdapterError::InvalidShape(
            "linear input width does not match weight".into(),
        ));
    }
    let rows = checked_product(&input_shape[..input_shape.len() - 1])?;
    let input = contiguous_copy(backend, input, context)?;
    let input = reshape_read_only(&input, vec![1, rows, *input_width])?;
    let weight = permute_last_two_view(weight)?;
    let weight = reshape_read_only(&weight, vec![1, *input_width, *output_width])?;
    let output = batch_matrix_multiply(backend, &input, &weight, context)?;
    let mut output_shape = input_shape.to_vec();
    let width = output_shape
        .last_mut()
        .ok_or_else(|| WeightAdapterError::InvalidShape("linear input is scalar".into()))?;
    *width = *output_width;
    reshape_read_only(&output, output_shape)
}

fn convolution(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: &Tensor,
    module: &ModuleTypeInfo,
    use_module_geometry: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let LayerKind::Convolution { dimensions } = module.kind else {
        return Err(WeightAdapterError::InvalidModule(
            "convolution requested for a linear module".into(),
        ));
    };
    require_f32(input, "convolution input")?;
    require_f32(weight, "convolution weight")?;
    require_same_device_stream(input, weight, backend, context)?;
    let spatial = usize::from(dimensions);
    if input.descriptor().rank() != spatial + 2 || weight.descriptor().rank() != spatial + 2 {
        return Err(WeightAdapterError::InvalidShape(
            "convolution rank does not match module type".into(),
        ));
    }
    let stride = if use_module_geometry {
        module.stride.clone()
    } else {
        vec![1; spatial]
    };
    let padding = if use_module_geometry {
        module.padding.clone()
    } else {
        vec![0; spatial]
    };
    let dilation = if use_module_geometry {
        module.dilation.clone()
    } else {
        vec![1; spatial]
    };
    let groups = if use_module_geometry {
        module.groups
    } else {
        1
    };
    let output_shape = convolution_output_shape(
        input.descriptor().shape(),
        weight.descriptor().shape(),
        &stride,
        &padding,
        &dilation,
        groups,
    )?;
    let descriptor = output_descriptor(input, output_shape, context)?;
    let specification = ConvolutionSpec {
        stride,
        padding,
        dilation,
        groups,
        transposed: false,
        output_padding: vec![0; spatial],
    };
    let (output, event) = backend.convolution(
        &specification,
        &[input.clone(), weight.clone()],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn matmul_two_dimensional(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let [rows, contracted] = left.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "matrix left operand must have rank two".into(),
        ));
    };
    let [right_contracted, columns] = right.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "matrix right operand must have rank two".into(),
        ));
    };
    if contracted != right_contracted {
        return Err(WeightAdapterError::InvalidShape(
            "matrix contraction dimensions are incompatible".into(),
        ));
    }
    let left = reshape_read_only(left, vec![1, *rows, *contracted])?;
    let right = reshape_read_only(right, vec![1, *contracted, *columns])?;
    let output = batch_matrix_multiply(backend, &left, &right, context)?;
    reshape_read_only(&output, vec![*rows, *columns])
}

fn batch_matrix_multiply(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let [batch, rows, contracted] = left.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "batch-matrix left operand must have rank three".into(),
        ));
    };
    let [right_batch, right_contracted, columns] = right.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "batch-matrix right operand must have rank three".into(),
        ));
    };
    if batch != right_batch || contracted != right_contracted {
        return Err(WeightAdapterError::InvalidShape(
            "batch-matrix contraction dimensions are incompatible".into(),
        ));
    }
    require_same_device_stream(left, right, backend, context)?;
    let descriptor = output_descriptor(left, vec![*batch, *rows, *columns], context)?;
    let (output, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[left.clone(), right.clone()],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn tucker_rebuild(
    backend: &dyn TensorBackend,
    tucker: &Tensor,
    up: &Tensor,
    down: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let tucker_shape = tucker.descriptor().shape();
    if tucker_shape.len() < 3 {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker tensor must have at least one spatial dimension".into(),
        ));
    }
    let [up_rows, _] = up.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker up factor must have rank two".into(),
        ));
    };
    let [down_rows, _] = down.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker down factor must have rank two".into(),
        ));
    };
    if *up_rows != tucker_shape[0] {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker up factor does not match the first core axis".into(),
        ));
    }
    if *down_rows != tucker_shape[1] {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker down factor does not match the second core axis".into(),
        ));
    }
    let up = permute_copy(backend, up, &[1, 0], context)?;
    tucker_rebuild_conventional(backend, tucker, &up, down, context)
}

fn tucker_rebuild_from_conv(
    backend: &dyn TensorBackend,
    tucker: &Tensor,
    up: &Tensor,
    down: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    tucker_rebuild_conventional(backend, tucker, up, down, context)
}

fn tucker_rebuild_conventional(
    backend: &dyn TensorBackend,
    tucker: &Tensor,
    up: &Tensor,
    down: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let tucker_shape = tucker.descriptor().shape();
    if tucker_shape.len() < 2 {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker tensor must have rank at least two".into(),
        ));
    }
    let [output_channels, first_rank] = up.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker up factor must have rank two".into(),
        ));
    };
    let [second_rank, input_channels] = down.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker down factor must have rank two".into(),
        ));
    };
    if tucker_shape[0] != *first_rank || tucker_shape[1] != *second_rank {
        return Err(WeightAdapterError::InvalidShape(
            "Tucker ranks do not match factor matrices".into(),
        ));
    }
    let spatial_shape = &tucker_shape[2..];
    let spatial_count = checked_product(spatial_shape)?;
    let mut permutation = (2..tucker_shape.len()).collect::<Vec<_>>();
    permutation.extend([0, 1]);
    let tucker = permute_copy(backend, tucker, &permutation, context)?;
    let tucker = reshape_read_only(&tucker, vec![spatial_count, *first_rank, *second_rank])?;
    let down = reshape_read_only(down, vec![1, *second_rank, *input_channels])?;
    let down = repeat_matrix_batch(backend, &down, spatial_count, context)?;
    let middle = batch_matrix_multiply(backend, &tucker, &down, context)?;
    let up = reshape_read_only(up, vec![1, *output_channels, *first_rank])?;
    let up = repeat_matrix_batch(backend, &up, spatial_count, context)?;
    let rebuilt = batch_matrix_multiply(backend, &up, &middle, context)?;
    let rebuilt = permute_copy(backend, &rebuilt, &[1, 2, 0], context)?;
    let mut shape = vec![*output_channels, *input_channels];
    shape.extend_from_slice(spatial_shape);
    reshape_read_only(&rebuilt, shape)
}

fn rebuild_lokr_factor(
    backend: &dyn TensorBackend,
    direct: &Option<AdapterTensor>,
    up: &Option<AdapterTensor>,
    down: &Option<AdapterTensor>,
    tucker: Option<&AdapterTensor>,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<u64>), WeightAdapterError> {
    if let Some(direct) = direct {
        return Ok((direct.materialize_tensor(backend, context)?, None));
    }
    let up = up
        .as_ref()
        .ok_or_else(|| WeightAdapterError::InvalidShape("LoKr up factor is missing".into()))?
        .materialize_tensor(backend, context)?;
    let down = down
        .as_ref()
        .ok_or_else(|| WeightAdapterError::InvalidShape("LoKr down factor is missing".into()))?
        .materialize_tensor(backend, context)?;
    let rank = first_extent(&down, "LoKr rank")?;
    let rebuilt = match tucker {
        Some(tucker) => tucker_rebuild(
            backend,
            &tucker.materialize_tensor(backend, context)?,
            &up,
            &down,
            context,
        )?,
        None => matmul_two_dimensional(backend, &up, &down, context)?,
    };
    Ok((rebuilt, Some(rank)))
}

fn binary(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    operation: BinaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    require_same_device_stream(left, right, backend, context)?;
    require_f32(left, "binary left operand")?;
    require_f32(right, "binary right operand")?;
    let shape = broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let descriptor = output_descriptor(left, shape, context)?;
    let (output, event) = backend.binary(operation, left, right, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn scalar(
    backend: &dyn TensorBackend,
    input: &Tensor,
    value: f32,
    operation: BinaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    validate_finite("scalar", value)?;
    require_execution_input(input, backend, context)?;
    let descriptor = output_descriptor(input, input.descriptor().shape().to_vec(), context)?;
    let (output, event) = backend.binary_scalar(
        operation,
        input,
        Scalar::Float(f64::from(value)),
        ScalarSide::Right,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn frobenius_norm(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<f32, WeightAdapterError> {
    let squared = binary(backend, input, input, BinaryOperation::Multiply, context)?;
    let dimensions = (0..squared.descriptor().rank())
        .map(|dimension| checked_u64(dimension, "reduction dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let specification = ReductionSpec {
        operation: ReductionOperation::Sum,
        dimensions,
        keep_dimensions: false,
        accumulation_dtype: Some(DType::F32),
        correction: 0,
    };
    let descriptor = output_descriptor(&squared, Vec::new(), context)?;
    let (sum, event) = backend.reduction(&specification, &squared, descriptor, context)?;
    backend.wait_event(event, context)?;
    let descriptor = output_descriptor(&sum, Vec::new(), context)?;
    let (norm, event) = backend.unary(UnaryOperation::SquareRoot, &sum, descriptor, context)?;
    backend.wait_event(event, context)?;
    let values = tensor_to_f32_with_backend_exact_native(backend, &norm, context)?;
    values
        .first()
        .copied()
        .ok_or_else(|| WeightAdapterError::InvalidShape("norm result is empty".into()))
}

fn identity_batch(
    backend: &dyn TensorBackend,
    batch: u64,
    size: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if backend.device().kind() != DeviceKind::Cpu {
        return Err(WeightAdapterError::UnsupportedDevice {
            operation: "identity batch upload",
            device: backend.device(),
        });
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![batch, size, size],
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (mut identity, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = identity.write()?;
        for batch_index in 0..batch {
            for diagonal in 0..size {
                context.check()?;
                let linear = batch_index
                    .checked_mul(size)
                    .and_then(|value| value.checked_add(diagonal))
                    .and_then(|value| value.checked_mul(size))
                    .and_then(|value| value.checked_add(diagonal))
                    .ok_or(WeightAdapterError::ShapeOverflow)?;
                let start = checked_usize(
                    linear
                        .checked_mul(4)
                        .ok_or(WeightAdapterError::ShapeOverflow)?,
                    "identity byte offset",
                )?;
                let end = start
                    .checked_add(4)
                    .ok_or(WeightAdapterError::ShapeOverflow)?;
                let bytes = write.bytes_mut()?;
                let destination = bytes
                    .get_mut(start..end)
                    .ok_or(WeightAdapterError::ShapeOverflow)?;
                destination.copy_from_slice(&1.0_f32.to_ne_bytes());
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(identity)
}

fn repeat_matrix_batch(
    backend: &dyn TensorBackend,
    input: &Tensor,
    repetitions: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if repetitions == 1 {
        return Ok(input.clone());
    }
    if backend.device().kind() != DeviceKind::Cpu {
        return Err(WeightAdapterError::UnsupportedDevice {
            operation: "matrix batch repeat",
            device: backend.device(),
        });
    }
    require_f32(input, "matrix batch repeat")?;
    let [batch, rows, columns] = input.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(
            "matrix batch repeat requires rank three".into(),
        ));
    };
    let source = tensor_to_f32_with_backend_exact_native(backend, input, context)?;
    let capacity = checked_usize(
        checked_u64(source.len(), "matrix batch source length")?
            .checked_mul(repetitions)
            .ok_or(WeightAdapterError::ShapeOverflow)?,
        "matrix batch repeated length",
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| WeightAdapterError::ShapeOverflow)?;
    for _ in 0..repetitions {
        context.check()?;
        values.extend_from_slice(&source);
    }
    Ok(tensor_from_f32_with_backend_exact_native(
        backend,
        &[
            batch
                .checked_mul(repetitions)
                .ok_or(WeightAdapterError::ShapeOverflow)?,
            *rows,
            *columns,
        ],
        &values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn channels_last_contiguous(
    backend: &dyn TensorBackend,
    input: &Tensor,
    module: &ModuleTypeInfo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if !matches!(module.kind, LayerKind::Convolution { .. }) {
        return contiguous_copy(backend, input, context);
    }
    let rank = input.descriptor().rank();
    if rank < 3 {
        return Err(WeightAdapterError::InvalidShape(
            "convolution output rank is too small".into(),
        ));
    }
    let mut permutation = vec![0];
    permutation.extend(2..rank);
    permutation.push(1);
    permute_copy(backend, input, &permutation, context)
}

fn restore_channels(
    backend: &dyn TensorBackend,
    input: &Tensor,
    module: &ModuleTypeInfo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if !matches!(module.kind, LayerKind::Convolution { .. }) {
        return Ok(input.clone());
    }
    let rank = input.descriptor().rank();
    let mut permutation = vec![0, rank - 1];
    permutation.extend(1..rank - 1);
    permute_copy(backend, input, &permutation, context)
}

fn permute_last_two(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let rank = input.descriptor().rank();
    if rank < 2 {
        return Err(WeightAdapterError::InvalidShape(
            "transpose requires rank at least two".into(),
        ));
    }
    let mut permutation = (0..rank).collect::<Vec<_>>();
    permutation.swap(rank - 2, rank - 1);
    permute_copy(backend, input, &permutation, context)
}

fn permute_last_two_view(input: &Tensor) -> Result<Tensor, WeightAdapterError> {
    let rank = input.descriptor().rank();
    if rank < 2 {
        return Err(WeightAdapterError::InvalidShape(
            "transpose requires rank at least two".into(),
        ));
    }
    let mut permutation = (0..rank).collect::<Vec<_>>();
    permutation.swap(rank - 2, rank - 1);
    Ok(input.view(
        input.descriptor().permuted_view(&permutation)?,
        ViewAccess::ReadOnly,
    )?)
}

fn permute_copy(
    backend: &dyn TensorBackend,
    input: &Tensor,
    permutation: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    let view = input.view(
        input.descriptor().permuted_view(permutation)?,
        ViewAccess::ReadOnly,
    )?;
    contiguous_copy(backend, &view, context)
}

fn contiguous_copy(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, WeightAdapterError> {
    if input.descriptor().is_contiguous()? {
        return Ok(input.clone());
    }
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.copy(input, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn reshape_read_only(input: &Tensor, shape: Vec<u64>) -> Result<Tensor, WeightAdapterError> {
    Ok(input.view(
        input.descriptor().reshaped_view(shape)?,
        ViewAccess::ReadOnly,
    )?)
}

fn output_descriptor(
    like: &Tensor,
    shape: Vec<u64>,
    context: &ExecutionContext<'_>,
) -> Result<TensorDescriptor, WeightAdapterError> {
    Ok(TensorDescriptor::contiguous(
        shape,
        like.descriptor().dtype(),
        like.descriptor().device(),
        context.stream,
    )?)
}

fn convolution_weight(
    weight: &Tensor,
    dimensions: u8,
    module: &ModuleTypeInfo,
) -> Result<Tensor, WeightAdapterError> {
    if weight.descriptor().rank() == usize::from(dimensions) + 2 {
        return Ok(weight.clone());
    }
    if weight.descriptor().rank() != 2 {
        return Err(WeightAdapterError::InvalidShape(
            "convolution adapter weight has an invalid rank".into(),
        ));
    }
    let shape = weight.descriptor().shape();
    let input_channels = module.input_channels.ok_or_else(|| {
        WeightAdapterError::InvalidModule(
            "convolution input channels are required to reshape adapter weights".into(),
        )
    })?;
    let expected = input_channels
        .checked_mul(checked_product(&module.kernel_size)?)
        .ok_or(WeightAdapterError::ShapeOverflow)?;
    if shape[1] != expected {
        return Err(WeightAdapterError::InvalidShape(
            "flattened convolution adapter width does not match module geometry".into(),
        ));
    }
    let mut target = vec![shape[0], input_channels];
    target.extend_from_slice(&module.kernel_size);
    reshape_read_only(weight, target)
}

fn pointwise_convolution_weight(
    weight: &Tensor,
    dimensions: u8,
) -> Result<Tensor, WeightAdapterError> {
    if weight.descriptor().rank() == usize::from(dimensions) + 2 {
        return Ok(weight.clone());
    }
    if weight.descriptor().rank() != 2 {
        return Err(WeightAdapterError::InvalidShape(
            "pointwise adapter weight has an invalid rank".into(),
        ));
    }
    let mut target = weight.descriptor().shape().to_vec();
    target.extend(vec![1; usize::from(dimensions)]);
    reshape_read_only(weight, target)
}

fn convolution_output_shape(
    input: &[u64],
    weight: &[u64],
    stride: &[u64],
    padding: &[u64],
    dilation: &[u64],
    groups: u64,
) -> Result<Vec<u64>, WeightAdapterError> {
    if groups == 0 || input.len() != weight.len() || input.len() != stride.len() + 2 {
        return Err(WeightAdapterError::InvalidShape(
            "convolution shape geometry is inconsistent".into(),
        ));
    }
    if input[1]
        != weight[1]
            .checked_mul(groups)
            .ok_or(WeightAdapterError::ShapeOverflow)?
    {
        return Err(WeightAdapterError::InvalidShape(
            "convolution input channels do not match weight groups".into(),
        ));
    }
    let mut output = vec![input[0], weight[0]];
    for dimension in 0..stride.len() {
        let padded = input[dimension + 2]
            .checked_add(
                padding[dimension]
                    .checked_mul(2)
                    .ok_or(WeightAdapterError::ShapeOverflow)?,
            )
            .ok_or(WeightAdapterError::ShapeOverflow)?;
        let kernel = dilation[dimension]
            .checked_mul(weight[dimension + 2].checked_sub(1).ok_or_else(|| {
                WeightAdapterError::InvalidShape("zero convolution kernel".into())
            })?)
            .and_then(|value| value.checked_add(1))
            .ok_or(WeightAdapterError::ShapeOverflow)?;
        if padded < kernel {
            return Err(WeightAdapterError::InvalidShape(
                "convolution kernel exceeds padded input".into(),
            ));
        }
        output.push((padded - kernel) / stride[dimension] + 1);
    }
    Ok(output)
}

fn load_family(
    family: AdapterFamily,
    request: &WeightAdapterLoadRequest,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    match family {
        AdapterFamily::Lora => load_lora(request),
        AdapterFamily::Loha => load_loha(request),
        AdapterFamily::Lokr => load_lokr(request),
        AdapterFamily::Glora => load_glora(request),
        AdapterFamily::Oft => load_oft(request, false),
        AdapterFamily::Boft => load_oft(request, true),
    }
}

fn load_lora(
    request: &WeightAdapterLoadRequest,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    let candidates = [
        (
            ".lora_up.weight",
            ".lora_down.weight",
            Some(".lora_mid.weight"),
        ),
        ("_lora.up.weight", "_lora.down.weight", None),
        (".lora_B.weight", ".lora_A.weight", None),
        (".lora.up.weight", ".lora.down.weight", None),
        (".lora_B", ".lora_A", None),
        (
            ".lora_linear_layer.up.weight",
            ".lora_linear_layer.down.weight",
            None,
        ),
        (".lora_B.default.weight", ".lora_A.default.weight", None),
    ];
    let mut matched = None;
    for (up_suffix, down_suffix, mid_suffix) in candidates {
        let up_key = format!("{}{}", request.prefix, up_suffix);
        let Some(up) = request.tensors.get(&up_key) else {
            continue;
        };
        let down_key = format!("{}{}", request.prefix, down_suffix);
        let down = required_tensor(request, &down_key, AdapterFamily::Lora)?;
        if matched.is_some() {
            return Err(WeightAdapterError::AmbiguousVariants(AdapterFamily::Lora));
        }
        let mid_key = mid_suffix.map(|suffix| format!("{}{}", request.prefix, suffix));
        let mid = mid_key
            .as_ref()
            .and_then(|key| request.tensors.get(key))
            .cloned();
        let reshape_key = format!("{}.reshape_weight", request.prefix);
        let reshape = match request.tensors.get(&reshape_key) {
            Some(tensor) => Some(decode_integral_shape(tensor)?),
            None => None,
        };
        let mut loaded_keys = BTreeSet::from([up_key, down_key]);
        if mid.is_some() {
            if let Some(mid_key) = mid_key {
                loaded_keys.insert(mid_key);
            }
        }
        if reshape.is_some() {
            loaded_keys.insert(reshape_key);
        }
        matched = Some(LoadedWeightAdapter {
            adapter: NativeWeightAdapter::Lora {
                up: up.clone(),
                down,
                alpha: request.alpha,
                mid,
                dora_scale: request.dora_scale.clone(),
                reshape,
            },
            loaded_keys,
        });
    }
    Ok(matched)
}

fn load_loha(
    request: &WeightAdapterLoadRequest,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    let first_up_key = format!("{}.hada_w1_a", request.prefix);
    let Some(first_up) = request.tensors.get(&first_up_key) else {
        return Ok(None);
    };
    let first_down_key = format!("{}.hada_w1_b", request.prefix);
    let second_up_key = format!("{}.hada_w2_a", request.prefix);
    let second_down_key = format!("{}.hada_w2_b", request.prefix);
    let first_tucker_key = format!("{}.hada_t1", request.prefix);
    let second_tucker_key = format!("{}.hada_t2", request.prefix);
    let first_tucker = request.tensors.get(&first_tucker_key).cloned();
    let second_tucker = match &first_tucker {
        Some(_) => Some(required_tensor(
            request,
            &second_tucker_key,
            AdapterFamily::Loha,
        )?),
        None => None,
    };
    let mut loaded_keys = BTreeSet::from([
        first_up_key,
        first_down_key.clone(),
        second_up_key.clone(),
        second_down_key.clone(),
    ]);
    if first_tucker.is_some() {
        loaded_keys.insert(first_tucker_key);
        loaded_keys.insert(second_tucker_key);
    }
    Ok(Some(LoadedWeightAdapter {
        adapter: NativeWeightAdapter::Loha {
            first_up: first_up.clone(),
            first_down: required_tensor(request, &first_down_key, AdapterFamily::Loha)?,
            second_up: required_tensor(request, &second_up_key, AdapterFamily::Loha)?,
            second_down: required_tensor(request, &second_down_key, AdapterFamily::Loha)?,
            first_tucker,
            second_tucker,
            alpha: request.alpha,
            dora_scale: request.dora_scale.clone(),
        },
        loaded_keys,
    }))
}

fn load_lokr(
    request: &WeightAdapterLoadRequest,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    let key = |suffix: &str| format!("{}{}", request.prefix, suffix);
    let first_key = key(".lokr_w1");
    let second_key = key(".lokr_w2");
    let first_up_key = key(".lokr_w1_a");
    let first_down_key = key(".lokr_w1_b");
    let second_up_key = key(".lokr_w2_a");
    let second_down_key = key(".lokr_w2_b");
    let second_tucker_key = key(".lokr_t2");
    let first = request.tensors.get(&first_key).cloned();
    let second = request.tensors.get(&second_key).cloned();
    let first_up = request.tensors.get(&first_up_key).cloned();
    let first_down = request.tensors.get(&first_down_key).cloned();
    let second_up = request.tensors.get(&second_up_key).cloned();
    let second_down = request.tensors.get(&second_down_key).cloned();
    let second_tucker = request.tensors.get(&second_tucker_key).cloned();
    if first.is_none() && second.is_none() && first_up.is_none() && second_up.is_none() {
        return Ok(None);
    }
    validate_direct_or_pair("LoKr first factor", &first, &first_up, &first_down)?;
    validate_direct_or_pair("LoKr second factor", &second, &second_up, &second_down)?;
    let loaded_keys = [
        (&first_key, &first),
        (&second_key, &second),
        (&first_up_key, &first_up),
        (&first_down_key, &first_down),
        (&second_up_key, &second_up),
        (&second_down_key, &second_down),
        (&second_tucker_key, &second_tucker),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.is_some().then_some(key.clone()))
    .collect();
    Ok(Some(LoadedWeightAdapter {
        adapter: NativeWeightAdapter::Lokr {
            first,
            second,
            first_up,
            first_down,
            second_up,
            second_down,
            second_tucker,
            alpha: request.alpha,
            dora_scale: request.dora_scale.clone(),
        },
        loaded_keys,
    }))
}

fn load_glora(
    request: &WeightAdapterLoadRequest,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    let first_a_key = format!("{}.a1.weight", request.prefix);
    let Some(first_a) = request.tensors.get(&first_a_key) else {
        return Ok(None);
    };
    let second_a_key = format!("{}.a2.weight", request.prefix);
    let first_b_key = format!("{}.b1.weight", request.prefix);
    let second_b_key = format!("{}.b2.weight", request.prefix);
    Ok(Some(LoadedWeightAdapter {
        adapter: NativeWeightAdapter::Glora {
            first_a: first_a.clone(),
            second_a: required_tensor(request, &second_a_key, AdapterFamily::Glora)?,
            first_b: required_tensor(request, &first_b_key, AdapterFamily::Glora)?,
            second_b: required_tensor(request, &second_b_key, AdapterFamily::Glora)?,
            alpha: request.alpha,
            dora_scale: request.dora_scale.clone(),
        },
        loaded_keys: BTreeSet::from([first_a_key, second_a_key, first_b_key, second_b_key]),
    }))
}

fn load_oft(
    request: &WeightAdapterLoadRequest,
    butterfly: bool,
) -> Result<Option<LoadedWeightAdapter>, WeightAdapterError> {
    let blocks_key = format!("{}.oft_blocks", request.prefix);
    let Some(blocks) = request.tensors.get(&blocks_key) else {
        return Ok(None);
    };
    let expected_rank = if butterfly { 4 } else { 3 };
    if blocks.shape()?.len() != expected_rank {
        return Ok(None);
    }
    let rescale_key = format!("{}.rescale", request.prefix);
    let rescale = request.tensors.get(&rescale_key).cloned();
    let mut loaded_keys = BTreeSet::from([blocks_key]);
    if rescale.is_some() {
        loaded_keys.insert(rescale_key);
    }
    let adapter = if butterfly {
        NativeWeightAdapter::Boft {
            blocks: blocks.clone(),
            rescale,
            constraint: request.alpha,
            dora_scale: request.dora_scale.clone(),
        }
    } else {
        NativeWeightAdapter::Oft {
            blocks: blocks.clone(),
            rescale,
            constraint: request.alpha,
            dora_scale: request.dora_scale.clone(),
        }
    };
    Ok(Some(LoadedWeightAdapter {
        adapter,
        loaded_keys,
    }))
}

fn required_tensor(
    request: &WeightAdapterLoadRequest,
    key: &str,
    family: AdapterFamily,
) -> Result<AdapterTensor, WeightAdapterError> {
    request
        .tensors
        .get(key)
        .cloned()
        .ok_or_else(|| WeightAdapterError::MissingCompanion {
            family,
            key: key.to_owned(),
        })
}

fn decode_integral_shape(tensor: &AdapterTensor) -> Result<Vec<u64>, WeightAdapterError> {
    let AdapterTensor::Dense(tensor) = tensor else {
        return Err(WeightAdapterError::InvalidShape(
            "LoRA reshape metadata cannot be quantized".into(),
        ));
    };
    if tensor.descriptor().device().kind() != DeviceKind::Cpu {
        return Err(WeightAdapterError::UnsupportedDevice {
            operation: "LoRA reshape metadata",
            device: tensor.descriptor().device(),
        });
    }
    let count = tensor.descriptor().element_count()?;
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(checked_usize(count, "reshape metadata length")?)
        .map_err(|_| WeightAdapterError::ShapeOverflow)?;
    for index in 0..count {
        let decoded = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?;
        let value = match decoded {
            DecodedScalar::Unsigned(value) => value,
            DecodedScalar::Signed(value) if value > 0 => {
                u64::try_from(value).map_err(|_| WeightAdapterError::ShapeOverflow)?
            }
            DecodedScalar::Real(value)
                if value.is_finite()
                    && value > 0.0
                    && value < 18_446_744_073_709_551_616.0
                    && value.fract() == 0.0 =>
            {
                value as u64
            }
            _ => {
                return Err(WeightAdapterError::InvalidShape(
                    "LoRA reshape metadata must contain positive integers".into(),
                ));
            }
        };
        if value == 0 {
            return Err(WeightAdapterError::InvalidShape(
                "LoRA reshape metadata must contain positive integers".into(),
            ));
        }
        shape.push(value);
    }
    if shape.is_empty() {
        return Err(WeightAdapterError::InvalidShape(
            "LoRA reshape metadata is empty".into(),
        ));
    }
    Ok(shape)
}

fn validate_layer_compatibility(
    adapter: &NativeWeightAdapter,
    module: &ModuleTypeInfo,
) -> Result<(), WeightAdapterError> {
    adapter.validate()?;
    if !module.has_weight {
        return Err(WeightAdapterError::InvalidModule(
            "bypass target has no weight".into(),
        ));
    }
    if let LayerKind::Convolution { dimensions } = module.kind
        && !(1..=3).contains(&dimensions)
    {
        return Err(WeightAdapterError::InvalidModule(
            "unsupported convolution dimension".into(),
        ));
    }
    Ok(())
}

fn validate_matrix_pair(
    up: &AdapterTensor,
    down: &AdapterTensor,
    name: &'static str,
) -> Result<(), WeightAdapterError> {
    let up_shape = require_rank_two(up, name)?;
    let down_shape = require_rank_two(down, name)?;
    if up_shape[1] != down_shape[0] {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} ranks are incompatible"
        )));
    }
    Ok(())
}

fn validate_tucker_factor(
    tucker: &AdapterTensor,
    up: &AdapterTensor,
    down: &AdapterTensor,
    name: &'static str,
) -> Result<Vec<u64>, WeightAdapterError> {
    let tucker_shape = tucker.shape()?;
    let up_shape = require_rank_two(up, name)?;
    let down_shape = require_rank_two(down, name)?;
    if !(3..=5).contains(&tucker_shape.len()) || tucker_shape.contains(&0) {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} Tucker core must have rank three through five"
        )));
    }
    if tucker_shape[0] != up_shape[0] {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} Tucker core and up-factor ranks are incompatible"
        )));
    }
    if tucker_shape[1] != down_shape[0] {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} Tucker core and down-factor ranks are incompatible"
        )));
    }
    let mut output_shape = vec![up_shape[1], down_shape[1]];
    output_shape.extend_from_slice(&tucker_shape[2..]);
    Ok(output_shape)
}

fn validate_convolution_tucker(
    tucker: &AdapterTensor,
    up: &AdapterTensor,
    down: &AdapterTensor,
    name: &'static str,
) -> Result<Vec<u64>, WeightAdapterError> {
    let tucker_shape = tucker.shape()?;
    let up_shape = require_rank_two(up, name)?;
    let down_shape = require_rank_two(down, name)?;
    if !(2..=5).contains(&tucker_shape.len()) || tucker_shape.contains(&0) {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} core must have rank two through five"
        )));
    }
    if tucker_shape[0] != up_shape[1] || tucker_shape[1] != down_shape[0] {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} core and factor ranks are incompatible"
        )));
    }
    let mut output_shape = vec![up_shape[0], down_shape[1]];
    output_shape.extend_from_slice(&tucker_shape[2..]);
    Ok(output_shape)
}

fn validate_direct_or_pair(
    name: &'static str,
    direct: &Option<AdapterTensor>,
    up: &Option<AdapterTensor>,
    down: &Option<AdapterTensor>,
) -> Result<(), WeightAdapterError> {
    match (direct, up, down) {
        (Some(direct), None, None) => {
            if direct.shape()?.len() < 2 {
                return Err(WeightAdapterError::InvalidShape(format!(
                    "{name} direct tensor must have rank at least two"
                )));
            }
            Ok(())
        }
        (None, Some(up), Some(down)) => validate_matrix_pair(up, down, name),
        _ => Err(WeightAdapterError::InvalidShape(format!(
            "{name} must be either one direct tensor or one complete factor pair"
        ))),
    }
}

fn require_rank_two(
    tensor: &AdapterTensor,
    name: &'static str,
) -> Result<Vec<u64>, WeightAdapterError> {
    let shape = tensor.shape()?;
    if shape.len() != 2 || shape.contains(&0) {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} must be a nonempty rank-two tensor"
        )));
    }
    Ok(shape)
}

fn require_square_blocks(
    tensor: &AdapterTensor,
    rank: usize,
    name: &'static str,
) -> Result<(), WeightAdapterError> {
    let shape = tensor.shape()?;
    if shape.len() != rank || shape.contains(&0) || shape[rank - 1] != shape[rank - 2] {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} must contain nonempty square blocks"
        )));
    }
    Ok(())
}

fn validate_glora_layout(
    first_a: &AdapterTensor,
    second_a: &AdapterTensor,
    first_b: &AdapterTensor,
    second_b: &AdapterTensor,
) -> Result<(), WeightAdapterError> {
    let first_a = first_a.shape()?;
    let second_a = second_a.shape()?;
    let first_b = first_b.shape()?;
    let second_b = second_b.shape()?;
    let old = second_b[1] == first_b[0] && first_b[0] == first_a[0] && first_a[0] == second_a[1];
    let new = second_b[0] == first_b[1] && first_b[1] == first_a[1] && first_a[1] == second_a[0];
    if !old && !new {
        return Err(WeightAdapterError::InvalidShape(
            "GLoRA factors match neither source layout".into(),
        ));
    }
    Ok(())
}

fn glora_old_layout(
    first_a: &Tensor,
    second_a: &Tensor,
    first_b: &Tensor,
    second_b: &Tensor,
) -> Result<bool, WeightAdapterError> {
    let a1 = require_dense_rank_two(first_a, "GLoRA first A")?;
    let a2 = require_dense_rank_two(second_a, "GLoRA second A")?;
    let b1 = require_dense_rank_two(first_b, "GLoRA first B")?;
    let b2 = require_dense_rank_two(second_b, "GLoRA second B")?;
    Ok(b2[1] == b1[0] && b1[0] == a1[0] && a1[0] == a2[1])
}

fn glora_new_layout(
    first_a: &Tensor,
    second_a: &Tensor,
    first_b: &Tensor,
    second_b: &Tensor,
) -> Result<bool, WeightAdapterError> {
    let a1 = require_dense_rank_two(first_a, "GLoRA first A")?;
    let a2 = require_dense_rank_two(second_a, "GLoRA second A")?;
    let b1 = require_dense_rank_two(first_b, "GLoRA first B")?;
    let b2 = require_dense_rank_two(second_b, "GLoRA second B")?;
    Ok(b2[0] == b1[1] && b1[1] == a1[1] && a1[1] == a2[0])
}

fn require_dense_rank_two(
    tensor: &Tensor,
    name: &'static str,
) -> Result<[u64; 2], WeightAdapterError> {
    let [rows, columns] = tensor.descriptor().shape() else {
        return Err(WeightAdapterError::InvalidShape(format!(
            "{name} must have rank two"
        )));
    };
    Ok([*rows, *columns])
}

fn require_execution_input(
    input: &Tensor,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), WeightAdapterError> {
    require_tensor_boundary(input, backend, context, "adapter input")?;
    require_f32(input, "adapter input")?;
    if input.descriptor().layout() != Layout::Contiguous
        && input.descriptor().layout() != Layout::Strided
    {
        return Err(WeightAdapterError::UnsupportedLayout(
            input.descriptor().layout(),
        ));
    }
    Ok(())
}

fn validate_base_output(
    output: &Tensor,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), WeightAdapterError> {
    require_tensor_boundary(output, backend, context, "base output")?;
    require_f32(output, "base output")
}

fn require_tensor_boundary(
    tensor: &Tensor,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
    name: &'static str,
) -> Result<(), WeightAdapterError> {
    context.check()?;
    if tensor.descriptor().device() != backend.device() {
        return Err(WeightAdapterError::DeviceMismatch {
            name,
            expected: backend.device(),
            actual: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().stream() != context.stream {
        return Err(WeightAdapterError::StreamMismatch {
            name,
            expected: context.stream.get(),
            actual: tensor.descriptor().stream().get(),
        });
    }
    Ok(())
}

fn require_same_device_stream(
    left: &Tensor,
    right: &Tensor,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), WeightAdapterError> {
    require_tensor_boundary(left, backend, context, "left tensor")?;
    require_tensor_boundary(right, backend, context, "right tensor")?;
    Ok(())
}

fn require_f32(tensor: &Tensor, name: &'static str) -> Result<(), WeightAdapterError> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(WeightAdapterError::UnsupportedDType {
            name,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn first_extent(tensor: &Tensor, name: &'static str) -> Result<u64, WeightAdapterError> {
    tensor
        .descriptor()
        .shape()
        .first()
        .copied()
        .filter(|extent| *extent > 0)
        .ok_or_else(|| WeightAdapterError::InvalidShape(format!("{name} is empty")))
}

fn prefix_shape(shape: &[u64]) -> Result<Vec<u64>, WeightAdapterError> {
    if shape.is_empty() {
        return Err(WeightAdapterError::InvalidShape("tensor is scalar".into()));
    }
    Ok(shape[..shape.len() - 1].to_vec())
}

fn broadcast_shape(left: &[u64], right: &[u64]) -> Result<Vec<u64>, WeightAdapterError> {
    let rank = left.len().max(right.len());
    let mut output = vec![1; rank];
    for offset in 0..rank {
        let left_dimension = left
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| left.get(index))
            .copied()
            .unwrap_or(1);
        let right_dimension = right
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| right.get(index))
            .copied()
            .unwrap_or(1);
        if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
            return Err(WeightAdapterError::InvalidShape(
                "tensor shapes cannot broadcast".into(),
            ));
        }
        output[rank - offset - 1] = left_dimension.max(right_dimension);
    }
    Ok(output)
}

fn patch_optional(
    tensor: &Option<AdapterTensor>,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<Option<PatchTensor>, WeightAdapterError> {
    tensor
        .as_ref()
        .map(|tensor| tensor.to_patch_tensor(backend, context))
        .transpose()
}

fn validate_finite(name: &'static str, value: f32) -> Result<(), WeightAdapterError> {
    if !value.is_finite() {
        return Err(WeightAdapterError::NonFinite(name));
    }
    Ok(())
}

fn validate_optional_finite(
    name: &'static str,
    value: Option<f32>,
) -> Result<(), WeightAdapterError> {
    if let Some(value) = value {
        validate_finite(name, value)?;
    }
    Ok(())
}

fn validate_optional_nonnegative(
    name: &'static str,
    value: Option<f32>,
) -> Result<(), WeightAdapterError> {
    validate_optional_finite(name, value)?;
    if value.is_some_and(|value| value < 0.0) {
        return Err(WeightAdapterError::InvalidPlan(format!(
            "{name} must be nonnegative"
        )));
    }
    Ok(())
}

fn validate_module_key(value: &str) -> Result<(), WeightAdapterError> {
    if value.is_empty()
        || value.len() > MAX_MODULE_KEY_BYTES
        || value.starts_with('.')
        || value.ends_with('.')
        || value.contains("..")
        || value.split('.').any(|part| {
            part.is_empty()
                || !part
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
        })
    {
        return Err(WeightAdapterError::InvalidModuleKey(value.to_owned()));
    }
    Ok(())
}

fn normalize_module_key(mut value: String) -> Result<String, WeightAdapterError> {
    if let Some(stripped) = value.strip_suffix(".weight") {
        value = stripped.to_owned();
    }
    validate_module_key(&value)?;
    Ok(value)
}

fn checked_product(values: &[u64]) -> Result<u64, WeightAdapterError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(WeightAdapterError::ShapeOverflow)
    })
}

fn checked_usize(value: u64, name: &'static str) -> Result<usize, WeightAdapterError> {
    usize::try_from(value).map_err(|_| WeightAdapterError::InvalidShape(format!("{name} overflow")))
}

fn checked_u64(value: usize, name: &'static str) -> Result<u64, WeightAdapterError> {
    u64::try_from(value).map_err(|_| WeightAdapterError::InvalidShape(format!("{name} overflow")))
}

#[derive(Debug, Error)]
pub enum WeightAdapterError {
    #[error("weight-adapter shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("weight-adapter shape is invalid: {0}")]
    InvalidShape(String),
    #[error("weight-adapter plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("weight-adapter module is invalid: {0}")]
    InvalidModule(String),
    #[error("weight-adapter module key is invalid: {0}")]
    InvalidModuleKey(String),
    #[error("weight-adapter value {0} must be finite")]
    NonFinite(&'static str),
    #[error("weight-adapter family {family:?} is missing companion tensor {key}")]
    MissingCompanion { family: AdapterFamily, key: String },
    #[error("weight-adapter payload matches both {first:?} and {second:?}")]
    AmbiguousFamilies {
        first: AdapterFamily,
        second: AdapterFamily,
    },
    #[error("weight-adapter payload matches multiple {0:?} source variants")]
    AmbiguousVariants(AdapterFamily),
    #[error("weight-adapter family {0:?} has no source trainable Diff class")]
    UnsupportedTrainableFamily(AdapterFamily),
    #[error("weight-adapter trainable tensor {name} cannot use quantized storage")]
    QuantizedTrainableTensor { name: &'static str },
    #[error("weight-adapter canonical operation {operation} failed: {reason}")]
    CanonicalOperation {
        operation: &'static str,
        reason: String,
    },
    #[error("weight-adapter {name} uses unsupported dtype {dtype:?}")]
    UnsupportedDType { name: &'static str, dtype: DType },
    #[error("weight-adapter layout {0:?} is unsupported")]
    UnsupportedLayout(Layout),
    #[error("weight-adapter operation {operation} is unsupported on device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("weight-adapter {name} device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        name: &'static str,
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error("weight-adapter {name} stream mismatch: expected {expected}, got {actual}")]
    StreamMismatch {
        name: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("weight-adapter binding count exceeds {0}")]
    TooManyBindings(usize),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Quantization(#[from] QuantizationError),
    #[error(transparent)]
    PatchGraph(#[from] PatchGraphError),
    #[error(transparent)]
    Cancellation(#[from] sim_types::CancellationError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    AutogradBreadth(#[from] AutogradBreadthError),
    #[error(transparent)]
    Rng(#[from] RngError),
}
