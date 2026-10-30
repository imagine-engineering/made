use crate::{
    AutocastPolicy, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, MutationWitness, NumericClass, Scalar, Tensor, TensorBackend,
    TensorDescriptor, TensorError, TensorId,
    generated_elementwise_or_runtime_operation_06::checkpoint_execution_from_outputs_exact_native,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

pub mod breadth;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct TapeNodeId(u64);

impl TapeNodeId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct OutputSlot {
    pub node: TapeNodeId,
    pub output: u32,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "LeafIdWire")]
pub struct LeafId(String);

#[derive(Deserialize)]
struct LeafIdWire(String);

impl TryFrom<LeafIdWire> for LeafId {
    type Error = AutogradError;

    fn try_from(value: LeafIdWire) -> Result<Self, Self::Error> {
        Self::new(value.0)
    }
}

impl LeafId {
    pub fn new(value: impl Into<String>) -> Result<Self, AutogradError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(AutogradError::InvalidGraph {
                reason: "leaf identifier is empty".to_owned(),
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum AutogradInput {
    Node(OutputSlot),
    Leaf(LeafId),
    Constant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientMode {
    Enabled,
    NoGrad,
    Inference,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradScalerConfig {
    pub initial_scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: u64,
    pub enabled: bool,
}

impl Default for GradScalerConfig {
    fn default() -> Self {
        Self {
            initial_scale: 65_536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2_000,
            enabled: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradScalerOptimizerDecision {
    Run,
    SkipNonFinite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GradScalerStage {
    Ready,
    Unscaled { found_non_finite: bool },
    Stepped { found_non_finite: bool },
}

#[derive(Clone, Debug)]
pub struct NativeGradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: u64,
    growth_tracker: u64,
    enabled: bool,
    stage: GradScalerStage,
}

impl NativeGradScaler {
    pub fn new(config: GradScalerConfig) -> Result<Self, GradScalerError> {
        if !config.initial_scale.is_finite()
            || config.initial_scale <= 0.0
            || !config.growth_factor.is_finite()
            || config.growth_factor <= 1.0
            || !config.backoff_factor.is_finite()
            || !(0.0..1.0).contains(&config.backoff_factor)
            || config.growth_interval == 0
        {
            return Err(GradScalerError::InvalidConfiguration);
        }
        Ok(Self {
            scale: config.initial_scale,
            growth_factor: config.growth_factor,
            backoff_factor: config.backoff_factor,
            growth_interval: config.growth_interval,
            growth_tracker: 0,
            enabled: config.enabled,
            stage: GradScalerStage::Ready,
        })
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn growth_tracker(&self) -> u64 {
        self.growth_tracker
    }

    pub fn scale_loss_exact_native(
        &self,
        loss: &Tensor,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, GradScalerError> {
        if !self.enabled {
            cancellation_check_scaler(cancellation)?;
            return Ok(loss.clone());
        }
        map_floating_tensor(loss, self.scale, cancellation)
    }

    pub fn unscale_gradients_exact_native(
        &mut self,
        gradients: &mut [Tensor],
        cancellation: &CancellationToken,
    ) -> Result<bool, GradScalerError> {
        if self.stage != GradScalerStage::Ready {
            return Err(GradScalerError::InvalidStage);
        }
        cancellation_check_scaler(cancellation)?;
        if !self.enabled {
            self.stage = GradScalerStage::Unscaled {
                found_non_finite: false,
            };
            return Ok(false);
        }
        if gradients.is_empty() {
            return Err(GradScalerError::NoGradients);
        }
        let reciprocal = self.scale.recip();
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(gradients.len())
            .map_err(|_| GradScalerError::AllocationFailed)?;
        let mut found_non_finite = false;
        for gradient in gradients.iter() {
            let (candidate, candidate_non_finite) =
                unscale_floating_tensor(gradient, reciprocal, cancellation)?;
            found_non_finite |= candidate_non_finite;
            staged.push(candidate);
        }
        cancellation_check_scaler(cancellation)?;
        for (gradient, candidate) in gradients.iter_mut().zip(staged) {
            gradient.commit_in_place(candidate)?;
        }
        self.stage = GradScalerStage::Unscaled { found_non_finite };
        Ok(found_non_finite)
    }

    pub fn optimizer_step_decision_exact_native(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<GradScalerOptimizerDecision, GradScalerError> {
        cancellation_check_scaler(cancellation)?;
        let GradScalerStage::Unscaled { found_non_finite } = self.stage else {
            return Err(GradScalerError::InvalidStage);
        };
        self.stage = GradScalerStage::Stepped { found_non_finite };
        Ok(if found_non_finite {
            GradScalerOptimizerDecision::SkipNonFinite
        } else {
            GradScalerOptimizerDecision::Run
        })
    }

    pub fn update_exact_native(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<(), GradScalerError> {
        cancellation_check_scaler(cancellation)?;
        let GradScalerStage::Stepped { found_non_finite } = self.stage else {
            return Err(GradScalerError::InvalidStage);
        };
        let mut next_scale = self.scale;
        let mut next_growth_tracker = self.growth_tracker;
        if self.enabled {
            if found_non_finite {
                next_scale *= self.backoff_factor;
                if next_scale == 0.0 {
                    return Err(GradScalerError::ScaleUnderflow);
                }
                next_growth_tracker = 0;
            } else {
                next_growth_tracker = next_growth_tracker
                    .checked_add(1)
                    .ok_or(GradScalerError::CounterOverflow)?;
                if next_growth_tracker >= self.growth_interval {
                    let grown_scale = next_scale * self.growth_factor;
                    if !grown_scale.is_finite() {
                        return Err(GradScalerError::ScaleOverflow);
                    }
                    next_scale = grown_scale;
                    next_growth_tracker = 0;
                }
            }
        }
        cancellation_check_scaler(cancellation)?;
        self.scale = next_scale;
        self.growth_tracker = next_growth_tracker;
        self.stage = GradScalerStage::Ready;
        Ok(())
    }
}

fn map_floating_tensor(
    input: &Tensor,
    factor: f64,
    cancellation: &CancellationToken,
) -> Result<Tensor, GradScalerError> {
    let (candidate, _) = map_floating_tensor_checked(input, factor, cancellation)?;
    Ok(candidate)
}

fn unscale_floating_tensor(
    input: &Tensor,
    factor: f64,
    cancellation: &CancellationToken,
) -> Result<(Tensor, bool), GradScalerError> {
    map_floating_tensor_checked(input, factor, cancellation)
}

fn map_floating_tensor_checked(
    input: &Tensor,
    factor: f64,
    cancellation: &CancellationToken,
) -> Result<(Tensor, bool), GradScalerError> {
    let descriptor = input.descriptor();
    if descriptor.device() != DeviceId::CPU {
        return Err(GradScalerError::UnsupportedDevice(descriptor.device()));
    }
    if descriptor.dtype().class() != NumericClass::FloatingPoint {
        return Err(GradScalerError::UnsupportedDType(descriptor.dtype()));
    }
    let shape = descriptor.shape().to_vec();
    let element_count = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(GradScalerError::ShapeOverflow)
    })?;
    let mut candidate = input.clone();
    let mut write = candidate.write()?;
    let mut found_non_finite = false;
    for linear_index in 0..element_count {
        if linear_index % 1_024 == 0 {
            cancellation_check_scaler(cancellation)?;
        }
        let indices = scaler_unravel_index(linear_index, &shape)?;
        let value = match descriptor
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?
        {
            DecodedScalar::Real(value) => value,
            _ => return Err(GradScalerError::UnsupportedDType(descriptor.dtype())),
        };
        found_non_finite |= !value.is_finite();
        let encoded = descriptor.dtype().encode_scalar(
            Scalar::Float(value * factor),
            "torch.amp.GradScaler",
            DeviceId::CPU,
        )?;
        write.element_bytes_mut(&indices)?.copy_from_slice(&encoded);
    }
    drop(write);
    cancellation_check_scaler(cancellation)?;
    Ok((candidate, found_non_finite))
}

fn scaler_unravel_index(mut linear_index: u64, shape: &[u64]) -> Result<Vec<u64>, GradScalerError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.len())
        .map_err(|_| GradScalerError::AllocationFailed)?;
    indices.resize(shape.len(), 0);
    for dimension_index in (0..shape.len()).rev() {
        let dimension = shape[dimension_index];
        if dimension == 0 {
            return Err(GradScalerError::ShapeOverflow);
        }
        indices[dimension_index] = linear_index % dimension;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn cancellation_check_scaler(cancellation: &CancellationToken) -> Result<(), GradScalerError> {
    if cancellation.is_cancelled() {
        Err(GradScalerError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum GradScalerError {
    #[error("gradient scaler configuration is invalid")]
    InvalidConfiguration,
    #[error("gradient scaler method order is invalid for its current optimizer stage")]
    InvalidStage,
    #[error("gradient scaler supports no native operation for device {0:?}")]
    UnsupportedDevice(DeviceId),
    #[error("gradient scaler requires a floating-point tensor, received {0:?}")]
    UnsupportedDType(crate::DType),
    #[error("gradient scaler tensor shape overflowed")]
    ShapeOverflow,
    #[error("gradient scaler allocation failed")]
    AllocationFailed,
    #[error("gradient scaler growth tracker overflowed")]
    CounterOverflow,
    #[error("gradient scaler scale overflowed")]
    ScaleOverflow,
    #[error("gradient scaler scale underflowed to zero")]
    ScaleUnderflow,
    #[error("gradient scaler received no gradients for non-finite checking")]
    NoGradients,
    #[error("gradient scaler operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[derive(Clone, Debug)]
pub struct SavedTensor {
    tensor: Tensor,
    witness: MutationWitness,
}

impl SavedTensor {
    pub fn capture(tensor: &Tensor) -> Self {
        Self {
            tensor: tensor.clone(),
            witness: tensor.mutation_witness(),
        }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn validate(&self) -> Result<(), AutogradError> {
        let actual_version = self.witness.actual_epoch();
        if !self.witness.is_current() {
            return Err(AutogradError::SavedTensorModified {
                tensor: self.witness.tensor_id(),
                expected_version: self.witness.expected_epoch(),
                actual_version,
            });
        }
        Ok(())
    }
}

pub trait BackwardRule: Send + Sync {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        saved_tensors: &[SavedTensor],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError>;

    fn vjp_with_context(
        &self,
        output_gradients: &[Option<Tensor>],
        saved_tensors: &[SavedTensor],
        _backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        self.vjp(output_gradients, saved_tensors, execution.cancellation)
    }

    fn higher_order_policy(&self) -> breadth::HigherOrderPolicy {
        breadth::HigherOrderPolicy::FirstOrderOnly
    }

    fn symbol(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    fn vjp_higher_order(
        &self,
        _output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        match self.higher_order_policy() {
            breadth::HigherOrderPolicy::Analytical => Err(AutogradError::MissingHigherOrderRule {
                symbol: self.symbol(),
            }),
            policy => Err(AutogradError::HigherOrderUnavailable {
                symbol: self.symbol(),
                policy,
            }),
        }
    }
}

pub trait GradientReducer {
    fn add(
        &self,
        left: Tensor,
        right: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError>;

    fn add_higher_order(
        &self,
        _left: Tensor,
        _right: Tensor,
        _context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Tensor, AutogradError> {
        Err(AutogradError::MissingHigherOrderRule {
            symbol: "gradient accumulation",
        })
    }
}

#[derive(Clone)]
struct TapeNode {
    id: TapeNodeId,
    inputs: Vec<AutogradInput>,
    output_count: u32,
    saved_tensors: Vec<SavedTensor>,
    rule: Arc<dyn BackwardRule>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum TapeState {
    Active,
    Completed,
    Cancelled(String),
    Faulted(String),
}

pub struct AutogradTape {
    mode: GradientMode,
    state: TapeState,
    nodes: Vec<TapeNode>,
    leaf_bindings: HashMap<TensorId, LeafId>,
    value_bindings: HashMap<TensorId, AutogradInput>,
    next_node: u64,
    scalar_nodes: Vec<ScalarAutogradNode>,
    scalar_leaves: HashMap<LeafId, ScalarAutogradLeaf>,
    scalar_maximum_nodes: Option<usize>,
    scalar_autocast_dtype: Option<DType>,
}

#[derive(Clone, Copy, Debug)]
pub struct ScalarAutogradValue {
    value: f32,
    node: Option<u32>,
}

impl ScalarAutogradValue {
    pub const fn constant(value: f32) -> Self {
        Self { value, node: None }
    }

    pub const fn value(self) -> f32 {
        self.value
    }
}

struct ScalarAutogradNode {
    first_node: u32,
    first_derivative: f32,
    second_node: u32,
    second_derivative: f32,
}

struct ScalarAutogradLeaf {
    descriptor: TensorDescriptor,
    nodes: Vec<u32>,
}

pub struct HigherOrderContext<'a, 'execution> {
    backend: &'a CpuBackend,
    execution: &'a ExecutionContext<'execution>,
    tape: &'a mut AutogradTape,
}

impl<'a, 'execution> HigherOrderContext<'a, 'execution> {
    pub fn backend(&self) -> &CpuBackend {
        self.backend
    }

    pub fn execution(&self) -> &ExecutionContext<'execution> {
        self.execution
    }

    pub fn record_operation(
        &mut self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        differentiable_outputs: &[bool],
        saved_tensors: Vec<SavedTensor>,
        rule: Arc<dyn BackwardRule>,
    ) -> Result<Option<Vec<OutputSlot>>, AutogradError> {
        cancellation_check(self.execution.cancellation)?;
        self.tape
            .record_operation(inputs, outputs, differentiable_outputs, saved_tensors, rule)
    }
}

impl AutogradTape {
    pub fn new(mode: GradientMode) -> Self {
        Self {
            mode,
            state: TapeState::Active,
            nodes: Vec::new(),
            leaf_bindings: HashMap::new(),
            value_bindings: HashMap::new(),
            next_node: 0,
            scalar_nodes: Vec::new(),
            scalar_leaves: HashMap::new(),
            scalar_maximum_nodes: None,
            scalar_autocast_dtype: None,
        }
    }

    pub fn mode(&self) -> GradientMode {
        self.mode
    }

    pub fn with_mode<T>(
        &mut self,
        mode: GradientMode,
        cancellation: &CancellationToken,
        operation: impl FnOnce(&mut Self) -> Result<T, AutogradError>,
    ) -> Result<T, AutogradError> {
        self.require_active()?;
        cancellation_check(cancellation)?;
        let previous = self.mode;
        self.mode = mode;
        let result = operation(self);
        self.mode = previous;
        cancellation_check(cancellation)?;
        result
    }

    pub fn state(&self) -> &TapeState {
        &self.state
    }

    pub fn retained_node_count(&self) -> usize {
        self.nodes.len().saturating_add(self.scalar_nodes.len())
    }

    pub fn record_scalar_parameter(
        &mut self,
        tensor: &Tensor,
        leaf: LeafId,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<ScalarAutogradValue>, AutogradError> {
        self.require_active()?;
        cancellation_check(context.cancellation)?;
        if tensor.descriptor().dtype() != DType::F32
            || tensor.descriptor().device() != DeviceId::CPU
            || self.scalar_leaves.contains_key(&leaf)
        {
            return Err(AutogradError::InvalidGraph {
                reason: "scalar autograd parameters must be unique CPU F32 leaves".to_owned(),
            });
        }
        let maximum = usize::try_from(context.scratch.bytes())
            .ok()
            .and_then(|bytes| bytes.checked_div(std::mem::size_of::<ScalarAutogradNode>()))
            .map(|maximum| maximum.min(u32::MAX as usize))
            .filter(|maximum| *maximum > 0)
            .ok_or(AutogradError::AllocationFailed)?;
        self.scalar_maximum_nodes = Some(
            self.scalar_maximum_nodes
                .map_or(maximum, |existing| existing.min(maximum)),
        );
        let element_count = usize::try_from(tensor.descriptor().element_count()?)
            .map_err(|_| AutogradError::AllocationFailed)?;
        let shape = tensor.descriptor().shape().to_vec();
        let mut values = Vec::new();
        let mut nodes = Vec::new();
        values
            .try_reserve_exact(element_count)
            .map_err(|_| AutogradError::AllocationFailed)?;
        nodes
            .try_reserve_exact(element_count)
            .map_err(|_| AutogradError::AllocationFailed)?;
        for index in 0..element_count {
            if index.is_multiple_of(1024) {
                cancellation_check(context.cancellation)?;
            }
            let index = u64::try_from(index).map_err(|_| AutogradError::AllocationFailed)?;
            let indices =
                training_unravel_index(index, &shape).map_err(training_error_as_autograd)?;
            let value = training_read_f32(tensor, &indices).map_err(training_error_as_autograd)?;
            let recorded = self.record_scalar_node(value, None, None)?;
            let node = recorded.node.ok_or(AutogradError::IdentifierOverflow)?;
            values.push(recorded);
            nodes.push(node);
        }
        self.scalar_leaves.insert(
            leaf,
            ScalarAutogradLeaf {
                descriptor: tensor.descriptor().clone(),
                nodes,
            },
        );
        Ok(values)
    }

    pub fn set_scalar_autocast_dtype(&mut self, dtype: Option<DType>) -> Result<(), AutogradError> {
        self.require_active()?;
        if dtype.is_some_and(|dtype| !matches!(dtype, DType::F32 | DType::Bf16)) {
            return Err(AutogradError::InvalidGraph {
                reason: "scalar autograd autocast supports F32 and BF16 only".to_owned(),
            });
        }
        self.scalar_autocast_dtype = dtype;
        Ok(())
    }

    pub fn scalar_add(
        &mut self,
        left: ScalarAutogradValue,
        right: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        match (left.node, right.node) {
            (None, None) => Ok(ScalarAutogradValue::constant(left.value + right.value)),
            (first, second) => self.record_scalar_node(
                left.value + right.value,
                first.map(|node| (node, 1.0)),
                second.map(|node| (node, 1.0)),
            ),
        }
    }

    pub fn scalar_multiply(
        &mut self,
        left: ScalarAutogradValue,
        right: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        match (left.node, right.node) {
            (None, None) => Ok(ScalarAutogradValue::constant(left.value * right.value)),
            (first, second) => self.record_scalar_node(
                left.value * right.value,
                first.map(|node| (node, right.value)),
                second.map(|node| (node, left.value)),
            ),
        }
    }

    pub fn scalar_scale(
        &mut self,
        value: ScalarAutogradValue,
        scale: f32,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        self.scalar_multiply(value, ScalarAutogradValue::constant(scale))
    }

    pub fn scalar_reciprocal(
        &mut self,
        value: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        let output = value.value.recip();
        match value.node {
            None => Ok(ScalarAutogradValue::constant(output)),
            Some(node) => self.record_scalar_node(output, Some((node, -output * output)), None),
        }
    }

    pub fn scalar_square_root(
        &mut self,
        value: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        let output = value.value.sqrt();
        match value.node {
            None => Ok(ScalarAutogradValue::constant(output)),
            Some(node) => self.record_scalar_node(output, Some((node, 0.5 / output)), None),
        }
    }

    pub fn scalar_exponential(
        &mut self,
        value: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        let output = value.value.exp();
        match value.node {
            None => Ok(ScalarAutogradValue::constant(output)),
            Some(node) => self.record_scalar_node(output, Some((node, output)), None),
        }
    }

    pub fn scalar_absolute(
        &mut self,
        value: ScalarAutogradValue,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        let output = value.value.abs();
        match value.node {
            None => Ok(ScalarAutogradValue::constant(output)),
            Some(node) => self.record_scalar_node(output, Some((node, value.value.signum())), None),
        }
    }

    pub fn scalar_straight_through(
        &mut self,
        value: ScalarAutogradValue,
        rounded: f32,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        match value.node {
            None => Ok(ScalarAutogradValue::constant(rounded)),
            Some(node) => self.record_scalar_node(rounded, Some((node, 1.0)), None),
        }
    }

    fn record_scalar_node(
        &mut self,
        value: f32,
        first: Option<(u32, f32)>,
        second: Option<(u32, f32)>,
    ) -> Result<ScalarAutogradValue, AutogradError> {
        let value = match self.scalar_autocast_dtype {
            Some(DType::Bf16) => half::bf16::from_f32(value).to_f32(),
            Some(DType::F32) | None => value,
            Some(_) => {
                return Err(AutogradError::InvalidGraph {
                    reason: "scalar autograd autocast dtype changed after validation".to_owned(),
                });
            }
        };
        if !value.is_finite() {
            return Err(AutogradError::InvalidGraph {
                reason: "scalar autograd produced a non-finite value".to_owned(),
            });
        }
        let maximum = self
            .scalar_maximum_nodes
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "scalar autograd has no bounded parameter leaf".to_owned(),
            })?;
        if self.scalar_nodes.len() >= maximum {
            return Err(AutogradError::AllocationFailed);
        }
        self.scalar_nodes
            .try_reserve(1)
            .map_err(|_| AutogradError::AllocationFailed)?;
        let node = u32::try_from(self.scalar_nodes.len())
            .map_err(|_| AutogradError::IdentifierOverflow)?;
        self.scalar_nodes.push(ScalarAutogradNode {
            first_node: first.map_or(u32::MAX, |value| value.0),
            first_derivative: first.map_or(0.0, |value| value.1),
            second_node: second.map_or(u32::MAX, |value| value.0),
            second_derivative: second.map_or(0.0, |value| value.1),
        });
        Ok(ScalarAutogradValue {
            value,
            node: Some(node),
        })
    }

    pub fn reverse_scalar_and_publish(
        &mut self,
        loss: ScalarAutogradValue,
        reducer: &dyn GradientReducer,
        store: &mut GradientStore,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(), AutogradError> {
        self.require_active()?;
        cancellation_check(context.cancellation)?;
        let loss_node = loss.node.ok_or_else(|| AutogradError::InvalidGraph {
            reason: "scalar training loss is detached from its parameters".to_owned(),
        })?;
        let mut scalar_gradients = Vec::new();
        scalar_gradients
            .try_reserve_exact(self.scalar_nodes.len())
            .map_err(|_| AutogradError::AllocationFailed)?;
        scalar_gradients.resize(self.scalar_nodes.len(), 0.0_f32);
        scalar_gradients[loss_node as usize] = 1.0;
        for node_index in (0..self.scalar_nodes.len()).rev() {
            if node_index.is_multiple_of(1024) {
                cancellation_check(context.cancellation)?;
            }
            let gradient = scalar_gradients[node_index];
            let node = &self.scalar_nodes[node_index];
            if node.first_node != u32::MAX {
                scalar_gradients[node.first_node as usize] += gradient * node.first_derivative;
            }
            if node.second_node != u32::MAX {
                scalar_gradients[node.second_node as usize] += gradient * node.second_derivative;
            }
        }
        let mut published = HashMap::new();
        for (leaf, record) in &self.scalar_leaves {
            cancellation_check(context.cancellation)?;
            let mut gradient = backend
                .fill(Scalar::Float(0.0), record.descriptor.clone(), context)?
                .0;
            let shape = record.descriptor.shape();
            let mut write = gradient.write()?;
            for (index, node) in record.nodes.iter().enumerate() {
                let index = u64::try_from(index).map_err(|_| AutogradError::AllocationFailed)?;
                let indices =
                    training_unravel_index(index, shape).map_err(training_error_as_autograd)?;
                write
                    .element_bytes_mut(&indices)?
                    .copy_from_slice(&scalar_gradients[*node as usize].to_le_bytes());
            }
            drop(write);
            published.insert(leaf.clone(), gradient);
        }
        store.accumulate(published, reducer, context.cancellation)?;
        self.release_recorded_nodes();
        self.state = TapeState::Completed;
        Ok(())
    }

    pub fn release_scalar_activations(&mut self) -> Result<(), AutogradError> {
        self.require_active()?;
        self.scalar_nodes.clear();
        self.scalar_leaves.clear();
        Ok(())
    }

    pub fn set_requires_grad(
        &mut self,
        tensor: &Tensor,
        leaf: Option<LeafId>,
        requires_grad: bool,
        cancellation: &CancellationToken,
    ) -> Result<(), AutogradError> {
        self.require_active()?;
        cancellation_check(cancellation)?;
        if requires_grad {
            if !matches!(
                tensor.descriptor().dtype().class(),
                NumericClass::FloatingPoint | NumericClass::Complex
            ) {
                return Err(AutogradError::InvalidRequiresGradDType {
                    dtype: tensor.descriptor().dtype(),
                });
            }
            let leaf = leaf.ok_or_else(|| AutogradError::InvalidGraph {
                reason: "enabling gradients requires a checked leaf identifier".to_owned(),
            })?;
            if let Some(existing) = self.leaf_bindings.get(&tensor.tensor_id())
                && existing != &leaf
            {
                return Err(AutogradError::InvalidGraph {
                    reason: format!(
                        "logical tensor {} is already bound to leaf {:?}",
                        tensor.tensor_id().get(),
                        existing.as_str()
                    ),
                });
            }
            cancellation_check(cancellation)?;
            self.leaf_bindings.insert(tensor.tensor_id(), leaf.clone());
            self.value_bindings
                .insert(tensor.tensor_id(), AutogradInput::Leaf(leaf));
        } else {
            cancellation_check(cancellation)?;
            self.leaf_bindings.remove(&tensor.tensor_id());
            self.value_bindings.remove(&tensor.tensor_id());
        }
        Ok(())
    }

    pub fn requires_grad(&self, tensor: &Tensor) -> bool {
        self.leaf_bindings.contains_key(&tensor.tensor_id())
    }

    pub fn leaf_binding(&self, tensor: &Tensor) -> Option<&LeafId> {
        self.leaf_bindings.get(&tensor.tensor_id())
    }

    pub fn output_slot(&self, tensor: &Tensor) -> Option<OutputSlot> {
        match self.value_bindings.get(&tensor.tensor_id()) {
            Some(AutogradInput::Node(slot)) => Some(*slot),
            Some(AutogradInput::Leaf(_) | AutogradInput::Constant) | None => None,
        }
    }

    pub fn record_operation(
        &mut self,
        input_tensors: &[&Tensor],
        output_tensors: &[&Tensor],
        differentiable_outputs: &[bool],
        saved_tensors: Vec<SavedTensor>,
        rule: Arc<dyn BackwardRule>,
    ) -> Result<Option<Vec<OutputSlot>>, AutogradError> {
        if output_tensors.len() != differentiable_outputs.len() {
            return Err(AutogradError::InvalidGraph {
                reason: "recorded output and differentiability arity differ".to_owned(),
            });
        }
        let output_count =
            u32::try_from(output_tensors.len()).map_err(|_| AutogradError::InvalidGraph {
                reason: "recorded output arity exceeds the native tape limit".to_owned(),
            })?;
        let inputs = input_tensors
            .iter()
            .map(|tensor| {
                self.value_bindings
                    .get(&tensor.tensor_id())
                    .cloned()
                    .unwrap_or(AutogradInput::Constant)
            })
            .collect::<Vec<_>>();
        self.value_bindings
            .try_reserve(output_tensors.len())
            .map_err(|_| AutogradError::AllocationFailed)?;
        let Some(slots) = self.record(inputs, output_count, saved_tensors, rule)? else {
            return Ok(None);
        };
        for ((tensor, differentiable), slot) in output_tensors
            .iter()
            .zip(differentiable_outputs)
            .zip(&slots)
        {
            if *differentiable {
                self.value_bindings
                    .insert(tensor.tensor_id(), AutogradInput::Node(*slot));
            } else {
                self.value_bindings.remove(&tensor.tensor_id());
            }
        }
        Ok(Some(slots))
    }

    pub fn record(
        &mut self,
        inputs: Vec<AutogradInput>,
        output_count: u32,
        saved_tensors: Vec<SavedTensor>,
        rule: Arc<dyn BackwardRule>,
    ) -> Result<Option<Vec<OutputSlot>>, AutogradError> {
        self.require_active()?;
        if self.mode != GradientMode::Enabled {
            return Ok(None);
        }
        if output_count == 0 {
            return Err(AutogradError::InvalidGraph {
                reason: "a recorded autograd node must have at least one output".to_owned(),
            });
        }
        let id = TapeNodeId(self.next_node);
        let next_node = self
            .next_node
            .checked_add(1)
            .ok_or(AutogradError::IdentifierOverflow)?;
        for saved in &saved_tensors {
            saved.validate()?;
        }
        for input in &inputs {
            if let AutogradInput::Node(slot) = input {
                self.validate_slot(*slot)?;
                if slot.node.get() >= id.get() {
                    return Err(AutogradError::InvalidGraph {
                        reason: "autograd inputs must reference an earlier node".to_owned(),
                    });
                }
            }
        }
        self.next_node = next_node;
        self.nodes.push(TapeNode {
            id,
            inputs,
            output_count,
            saved_tensors,
            rule,
        });
        let outputs = (0..output_count)
            .map(|output| OutputSlot { node: id, output })
            .collect();
        Ok(Some(outputs))
    }

    pub fn backward(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        self.reverse(seeds, reducer, cancellation, false)
    }

    pub fn backward_retain_graph(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
        retain_graph: bool,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        self.reverse(seeds, reducer, cancellation, retain_graph)
    }

    pub fn reverse(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
        retain_graph: bool,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        self.require_active()?;
        let result = self.backward_active(seeds, reducer, cancellation);
        match result {
            Ok(gradients) if retain_graph => Ok(gradients),
            Ok(gradients) => {
                self.release_recorded_nodes();
                self.state = TapeState::Completed;
                Ok(gradients)
            }
            Err(error) => {
                self.release_recorded_nodes();
                self.state = if error == AutogradError::Cancelled {
                    TapeState::Cancelled(error.to_string())
                } else {
                    TapeState::Faulted(error.to_string())
                };
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reverse_with_context(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        retain_graph: bool,
        create_graph: bool,
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        if !create_graph {
            self.require_active()?;
            let result = self.backward_active_with_context(seeds, reducer, backend, execution);
            return match result {
                Ok(gradients) if retain_graph => Ok(gradients),
                Ok(gradients) => {
                    self.release_recorded_nodes();
                    self.state = TapeState::Completed;
                    Ok(gradients)
                }
                Err(error) => {
                    self.release_recorded_nodes();
                    self.state = if error == AutogradError::Cancelled {
                        TapeState::Cancelled(error.to_string())
                    } else {
                        TapeState::Faulted(error.to_string())
                    };
                    Err(error)
                }
            };
        }
        self.require_active()?;
        cancellation_check(execution.cancellation)?;
        let mut derivative_graph = self.fork_active()?;
        let result = self.backward_active_higher_order(
            seeds,
            reducer,
            backend,
            execution,
            &mut derivative_graph,
        );
        match result {
            Ok(gradients) => {
                if let Err(error) = cancellation_check(execution.cancellation) {
                    self.release_recorded_nodes();
                    self.state = TapeState::Cancelled(error.to_string());
                    return Err(error);
                }
                *self = derivative_graph;
                Ok(gradients)
            }
            Err(error) => {
                self.release_recorded_nodes();
                self.state = if error == AutogradError::Cancelled {
                    TapeState::Cancelled(error.to_string())
                } else {
                    TapeState::Faulted(error.to_string())
                };
                Err(error)
            }
        }
    }

    pub fn reverse_and_publish(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
        retain_graph: bool,
        inputs: Option<&[LeafId]>,
        store: &mut GradientStore,
    ) -> Result<(), AutogradError> {
        self.require_active()?;
        let result = self.backward_active(seeds, reducer, cancellation);
        match result {
            Ok(mut gradients) => {
                if let Some(inputs) = inputs {
                    gradients.retain(|leaf, _| inputs.contains(leaf));
                }
                let publication = cancellation_check(cancellation)
                    .and_then(|()| store.accumulate(gradients, reducer, cancellation));
                match publication {
                    Ok(()) => {
                        if !retain_graph {
                            self.release_recorded_nodes();
                            self.state = TapeState::Completed;
                        }
                        Ok(())
                    }
                    Err(error) => {
                        self.release_recorded_nodes();
                        self.state = if error == AutogradError::Cancelled {
                            TapeState::Cancelled(error.to_string())
                        } else {
                            TapeState::Faulted(error.to_string())
                        };
                        Err(error)
                    }
                }
            }
            Err(error) => {
                self.release_recorded_nodes();
                self.state = if error == AutogradError::Cancelled {
                    TapeState::Cancelled(error.to_string())
                } else {
                    TapeState::Faulted(error.to_string())
                };
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reverse_and_publish_with_context(
        &mut self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        retain_graph: bool,
        create_graph: bool,
        inputs: Option<&[LeafId]>,
        store: &mut GradientStore,
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<(), AutogradError> {
        if !create_graph {
            self.require_active()?;
            let result = self.backward_active_with_context(seeds, reducer, backend, execution);
            return match result {
                Ok(mut gradients) => {
                    if let Some(inputs) = inputs {
                        gradients.retain(|leaf, _| inputs.contains(leaf));
                    }
                    let publication = cancellation_check(execution.cancellation).and_then(|()| {
                        store.accumulate(gradients, reducer, execution.cancellation)
                    });
                    match publication {
                        Ok(()) => {
                            if !retain_graph {
                                self.release_recorded_nodes();
                                self.state = TapeState::Completed;
                            }
                            Ok(())
                        }
                        Err(error) => {
                            self.release_recorded_nodes();
                            self.state = if error == AutogradError::Cancelled {
                                TapeState::Cancelled(error.to_string())
                            } else {
                                TapeState::Faulted(error.to_string())
                            };
                            Err(error)
                        }
                    }
                }
                Err(error) => {
                    self.release_recorded_nodes();
                    self.state = if error == AutogradError::Cancelled {
                        TapeState::Cancelled(error.to_string())
                    } else {
                        TapeState::Faulted(error.to_string())
                    };
                    Err(error)
                }
            };
        }
        self.require_active()?;
        cancellation_check(execution.cancellation)?;
        let mut derivative_graph = self.fork_active()?;
        let result = self.backward_active_higher_order(
            seeds,
            reducer,
            backend,
            execution,
            &mut derivative_graph,
        );
        match result {
            Ok(mut gradients) => {
                if let Some(inputs) = inputs {
                    gradients.retain(|leaf, _| inputs.contains(leaf));
                }
                let publication = store.accumulate_higher_order(
                    gradients,
                    reducer,
                    backend,
                    execution,
                    &mut derivative_graph,
                );
                match publication {
                    Ok(staged) => {
                        if let Err(error) = cancellation_check(execution.cancellation) {
                            self.release_recorded_nodes();
                            self.state = TapeState::Cancelled(error.to_string());
                            return Err(error);
                        }
                        store.gradients = staged;
                        *self = derivative_graph;
                        Ok(())
                    }
                    Err(error) => {
                        self.release_recorded_nodes();
                        self.state = if error == AutogradError::Cancelled {
                            TapeState::Cancelled(error.to_string())
                        } else {
                            TapeState::Faulted(error.to_string())
                        };
                        Err(error)
                    }
                }
            }
            Err(error) => {
                self.release_recorded_nodes();
                self.state = if error == AutogradError::Cancelled {
                    TapeState::Cancelled(error.to_string())
                } else {
                    TapeState::Faulted(error.to_string())
                };
                Err(error)
            }
        }
    }

    pub fn cancel(&mut self, reason: impl Into<String>) -> Result<(), AutogradError> {
        self.require_active()?;
        self.release_recorded_nodes();
        self.state = TapeState::Cancelled(reason.into());
        Ok(())
    }

    pub fn fault(&mut self, reason: impl Into<String>) -> Result<(), AutogradError> {
        self.require_active()?;
        self.release_recorded_nodes();
        self.state = TapeState::Faulted(reason.into());
        Ok(())
    }

    fn backward_active(
        &self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        cancellation_check(cancellation)?;
        let mut node_gradients = HashMap::new();
        for (slot, gradient) in seeds {
            self.validate_slot(slot)?;
            accumulate(&mut node_gradients, slot, gradient, reducer, cancellation)?;
        }
        let mut leaf_gradients = HashMap::new();
        for node in self.nodes.iter().rev() {
            cancellation_check(cancellation)?;
            for saved in &node.saved_tensors {
                saved.validate()?;
            }
            let output_gradients = (0..node.output_count)
                .map(|output| {
                    node_gradients.remove(&OutputSlot {
                        node: node.id,
                        output,
                    })
                })
                .collect::<Vec<_>>();
            if output_gradients.iter().all(Option::is_none) {
                continue;
            }
            let input_gradients =
                node.rule
                    .vjp(&output_gradients, &node.saved_tensors, cancellation)?;
            if input_gradients.len() != node.inputs.len() {
                return Err(AutogradError::GradientArity {
                    expected: node.inputs.len(),
                    actual: input_gradients.len(),
                });
            }
            for (input, gradient) in node.inputs.iter().zip(input_gradients) {
                let Some(gradient) = gradient else {
                    continue;
                };
                match input {
                    AutogradInput::Node(slot) => {
                        accumulate(&mut node_gradients, *slot, gradient, reducer, cancellation)?;
                    }
                    AutogradInput::Leaf(leaf) => {
                        accumulate(
                            &mut leaf_gradients,
                            leaf.clone(),
                            gradient,
                            reducer,
                            cancellation,
                        )?;
                    }
                    AutogradInput::Constant => {}
                }
            }
        }
        Ok(leaf_gradients)
    }

    fn backward_active_higher_order(
        &self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
        derivative_graph: &mut AutogradTape,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        cancellation_check(execution.cancellation)?;
        let mut node_gradients = HashMap::new();
        for (slot, gradient) in seeds {
            self.validate_slot(slot)?;
            accumulate_higher_order(
                &mut node_gradients,
                slot,
                gradient,
                reducer,
                backend,
                execution,
                derivative_graph,
            )?;
        }
        let mut leaf_gradients = HashMap::new();
        for node in self.nodes.iter().rev() {
            cancellation_check(execution.cancellation)?;
            for saved in &node.saved_tensors {
                saved.validate()?;
            }
            let output_gradients = (0..node.output_count)
                .map(|output| {
                    node_gradients.remove(&OutputSlot {
                        node: node.id,
                        output,
                    })
                })
                .collect::<Vec<_>>();
            if output_gradients.iter().all(Option::is_none) {
                continue;
            }
            let policy = node.rule.higher_order_policy();
            if policy != breadth::HigherOrderPolicy::Analytical {
                return Err(AutogradError::HigherOrderUnavailable {
                    symbol: node.rule.symbol(),
                    policy,
                });
            }
            let input_gradients = {
                let mut context = HigherOrderContext {
                    backend,
                    execution,
                    tape: derivative_graph,
                };
                node.rule
                    .vjp_higher_order(&output_gradients, &node.saved_tensors, &mut context)?
            };
            if input_gradients.len() != node.inputs.len() {
                return Err(AutogradError::GradientArity {
                    expected: node.inputs.len(),
                    actual: input_gradients.len(),
                });
            }
            for (input, gradient) in node.inputs.iter().zip(input_gradients) {
                let Some(gradient) = gradient else {
                    continue;
                };
                match input {
                    AutogradInput::Node(slot) => accumulate_higher_order(
                        &mut node_gradients,
                        *slot,
                        gradient,
                        reducer,
                        backend,
                        execution,
                        derivative_graph,
                    )?,
                    AutogradInput::Leaf(leaf) => accumulate_higher_order(
                        &mut leaf_gradients,
                        leaf.clone(),
                        gradient,
                        reducer,
                        backend,
                        execution,
                        derivative_graph,
                    )?,
                    AutogradInput::Constant => {}
                }
            }
        }
        Ok(leaf_gradients)
    }

    fn backward_active_with_context(
        &self,
        seeds: Vec<(OutputSlot, Tensor)>,
        reducer: &dyn GradientReducer,
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        cancellation_check(execution.cancellation)?;
        let mut node_gradients = HashMap::new();
        for (slot, gradient) in seeds {
            self.validate_slot(slot)?;
            accumulate(
                &mut node_gradients,
                slot,
                gradient,
                reducer,
                execution.cancellation,
            )?;
        }
        let mut leaf_gradients = HashMap::new();
        for node in self.nodes.iter().rev() {
            cancellation_check(execution.cancellation)?;
            for saved in &node.saved_tensors {
                saved.validate()?;
            }
            let output_gradients = (0..node.output_count)
                .map(|output| {
                    node_gradients.remove(&OutputSlot {
                        node: node.id,
                        output,
                    })
                })
                .collect::<Vec<_>>();
            if output_gradients.iter().all(Option::is_none) {
                continue;
            }
            let input_gradients = node.rule.vjp_with_context(
                &output_gradients,
                &node.saved_tensors,
                backend,
                execution,
            )?;
            if input_gradients.len() != node.inputs.len() {
                return Err(AutogradError::GradientArity {
                    expected: node.inputs.len(),
                    actual: input_gradients.len(),
                });
            }
            for (input, gradient) in node.inputs.iter().zip(input_gradients) {
                let Some(gradient) = gradient else {
                    continue;
                };
                match input {
                    AutogradInput::Node(slot) => accumulate(
                        &mut node_gradients,
                        *slot,
                        gradient,
                        reducer,
                        execution.cancellation,
                    )?,
                    AutogradInput::Leaf(leaf) => accumulate(
                        &mut leaf_gradients,
                        leaf.clone(),
                        gradient,
                        reducer,
                        execution.cancellation,
                    )?,
                    AutogradInput::Constant => {}
                }
            }
        }
        Ok(leaf_gradients)
    }

    fn fork_active(&self) -> Result<Self, AutogradError> {
        self.require_active()?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| AutogradError::AllocationFailed)?;
        nodes.extend(self.nodes.iter().cloned());
        let mut leaf_bindings = HashMap::new();
        leaf_bindings
            .try_reserve(self.leaf_bindings.len())
            .map_err(|_| AutogradError::AllocationFailed)?;
        leaf_bindings.extend(
            self.leaf_bindings
                .iter()
                .map(|(tensor, leaf)| (*tensor, leaf.clone())),
        );
        let mut value_bindings = HashMap::new();
        value_bindings
            .try_reserve(self.value_bindings.len())
            .map_err(|_| AutogradError::AllocationFailed)?;
        value_bindings.extend(
            self.value_bindings
                .iter()
                .map(|(tensor, input)| (*tensor, input.clone())),
        );
        Ok(Self {
            mode: self.mode,
            state: TapeState::Active,
            nodes,
            leaf_bindings,
            value_bindings,
            next_node: self.next_node,
            scalar_nodes: Vec::new(),
            scalar_leaves: HashMap::new(),
            scalar_maximum_nodes: self.scalar_maximum_nodes,
            scalar_autocast_dtype: self.scalar_autocast_dtype,
        })
    }

    fn release_recorded_nodes(&mut self) {
        self.nodes.clear();
        self.scalar_nodes.clear();
        self.scalar_leaves.clear();
        self.value_bindings
            .retain(|_, input| matches!(input, AutogradInput::Leaf(_)));
    }

    fn validate_slot(&self, slot: OutputSlot) -> Result<(), AutogradError> {
        let node = self.nodes.iter().find(|node| node.id == slot.node).ok_or(
            AutogradError::InvalidGraph {
                reason: format!("seed references unknown node {}", slot.node.get()),
            },
        )?;
        if slot.output >= node.output_count {
            return Err(AutogradError::InvalidGraph {
                reason: format!(
                    "seed output {} exceeds node {} output count {}",
                    slot.output,
                    slot.node.get(),
                    node.output_count
                ),
            });
        }
        Ok(())
    }

    fn require_active(&self) -> Result<(), AutogradError> {
        if self.state == TapeState::Active {
            Ok(())
        } else {
            Err(AutogradError::TerminalTape {
                state: self.state.clone(),
            })
        }
    }
}

#[derive(Debug)]
pub struct CheckpointRecord {
    saved: Vec<SavedTensor>,
    needs_input_grad: Vec<bool>,
    released: bool,
}

impl CheckpointRecord {
    pub fn capture(saved: &[Tensor], needs_input_grad: Vec<bool>) -> Result<Self, AutogradError> {
        if needs_input_grad.len() != saved.len() {
            return Err(AutogradError::GradientArity {
                expected: saved.len(),
                actual: needs_input_grad.len(),
            });
        }
        Ok(Self {
            saved: saved.iter().map(SavedTensor::capture).collect(),
            needs_input_grad,
            released: false,
        })
    }

    pub fn saved_tensors(&self) -> Result<Vec<Tensor>, AutogradError> {
        if self.released {
            return Err(AutogradError::ReleasedCheckpoint);
        }
        self.saved
            .iter()
            .map(|saved| {
                saved.validate()?;
                Ok(saved.tensor().clone())
            })
            .collect()
    }

    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.needs_input_grad.get(index).copied().unwrap_or(false)
    }

    pub fn saved_tensor_count(&self) -> usize {
        self.saved.len()
    }

    pub fn release(&mut self) {
        self.saved.clear();
        self.needs_input_grad.clear();
        self.released = true;
    }
}

#[derive(Default)]
pub struct GradientStore {
    gradients: HashMap<LeafId, Tensor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTrainingOptimizer {
    AdamW,
    Adam,
    Sgd,
    Rmsprop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTrainingLoss {
    Mse,
    L1,
    Huber,
    SmoothL1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTrainingComputeDType {
    Bf16,
    F32,
    Preserve,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeTrainingExample {
    pub feature: f32,
    pub target: f32,
    pub base_prediction: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeTrainingPlan {
    pub optimizer: NativeTrainingOptimizer,
    pub loss: NativeTrainingLoss,
    pub compute_dtype: NativeTrainingComputeDType,
    pub learning_rate: f32,
    pub optimizer_steps: u64,
    pub total_microsteps: u64,
    pub gradient_accumulation_steps: u64,
    pub batch_size: u64,
    pub seed: u64,
    pub gradient_checkpointing: bool,
    pub checkpoint_depth: u8,
    pub offloading: bool,
    pub quantized_backward: bool,
    pub checkpoint_failure_at_microstep: Option<u64>,
    pub cancellation_at_microstep: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct NativeTrainingResult {
    pub parameters: Vec<Tensor>,
    pub losses: Vec<f32>,
    pub sampled_indices: Vec<usize>,
    pub completed_optimizer_steps: u64,
    pub checkpoint_boundaries: Vec<u64>,
}

pub struct NativeTrainingObjectiveEvaluation {
    pub loss: f32,
    pub scalar_loss: ScalarAutogradValue,
}

pub trait NativeTrainingObjective {
    fn sample_index(
        &self,
        seed: u64,
        microstep: u64,
        batch_ordinal: u64,
        dataset_length: usize,
    ) -> Result<usize, NativeTrainingError> {
        native_training_sample_index(
            seed ^ batch_ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            microstep,
            dataset_length,
        )
    }

    fn evaluate(
        &mut self,
        backend: &CpuBackend,
        parameters: &[Tensor],
        scalar_parameters: &[Vec<ScalarAutogradValue>],
        sampled_indices: &[usize],
        microstep: u64,
        plan: &NativeTrainingPlan,
        tape: &mut AutogradTape,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeTrainingObjectiveEvaluation, NativeTrainingError>;

    fn recompute_loss(
        &mut self,
        backend: &CpuBackend,
        parameters: &[Tensor],
        sampled_indices: &[usize],
        microstep: u64,
        plan: &NativeTrainingPlan,
        context: &ExecutionContext<'_>,
    ) -> Result<f32, NativeTrainingError> {
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_scalar_autocast_dtype(match plan.compute_dtype {
            NativeTrainingComputeDType::Bf16 => Some(DType::Bf16),
            NativeTrainingComputeDType::F32 | NativeTrainingComputeDType::Preserve => None,
        })?;
        let scalar_parameters = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| {
                let leaf = LeafId::new(format!("recompute_parameter_{index}"))?;
                tape.record_scalar_parameter(parameter, leaf, context)
                    .map_err(NativeTrainingError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .evaluate(
                backend,
                parameters,
                &scalar_parameters,
                sampled_indices,
                microstep,
                plan,
                &mut tape,
                context,
            )?
            .loss)
    }
}

struct NativeScalarTrainingObjective<'a> {
    examples: &'a [NativeTrainingExample],
}

impl NativeTrainingObjective for NativeScalarTrainingObjective<'_> {
    fn evaluate(
        &mut self,
        _backend: &CpuBackend,
        _parameters: &[Tensor],
        scalar_parameters: &[Vec<ScalarAutogradValue>],
        sampled_indices: &[usize],
        microstep: u64,
        plan: &NativeTrainingPlan,
        tape: &mut AutogradTape,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeTrainingObjectiveEvaluation, NativeTrainingError> {
        let parameter_elements =
            scalar_parameters
                .iter()
                .try_fold(0_usize, |total, parameter| {
                    total
                        .checked_add(parameter.len())
                        .ok_or(NativeTrainingError::ShapeOverflow)
                })?;
        if parameter_elements == 0 {
            return Err(NativeTrainingError::Invalid(
                "training parameters are empty",
            ));
        }
        let mut parameter_sum = ScalarAutogradValue::constant(0.0);
        for parameter in scalar_parameters {
            for value in parameter {
                parameter_sum = tape.scalar_add(parameter_sum, *value)?;
            }
        }
        let parameter_mean = tape.scalar_scale(parameter_sum, 1.0 / parameter_elements as f32)?;
        let mut loss = ScalarAutogradValue::constant(0.0);
        for (batch_ordinal, sample) in sampled_indices.iter().copied().enumerate() {
            context.check().map_err(map_training_tensor_error)?;
            let phase =
                u64::try_from(batch_ordinal).map_err(|_| NativeTrainingError::ShapeOverflow)?;
            let example = self
                .examples
                .get(sample)
                .ok_or(NativeTrainingError::Invalid(
                    "sampled batch dataset index is unavailable",
                ))?;
            let random = native_training_unit_random(plan.seed, microstep, 1 + phase * 2);
            let sigma = native_training_unit_random(plan.seed, microstep, 2 + phase * 2);
            let feature = quantize_training_value(
                example.feature + (random * 2.0 - 1.0) * sigma,
                plan.compute_dtype,
            );
            let product = tape.scalar_scale(parameter_mean, feature)?;
            let prediction = tape.scalar_add(
                ScalarAutogradValue::constant(example.base_prediction),
                product,
            )?;
            let rounded = quantize_training_value(prediction.value(), plan.compute_dtype);
            let prediction = tape.scalar_straight_through(prediction, rounded)?;
            let error =
                tape.scalar_add(prediction, ScalarAutogradValue::constant(-example.target))?;
            let absolute = tape.scalar_absolute(error)?;
            let sample_loss = match plan.loss {
                NativeTrainingLoss::Mse => tape.scalar_multiply(error, error)?,
                NativeTrainingLoss::L1 => absolute,
                NativeTrainingLoss::Huber | NativeTrainingLoss::SmoothL1
                    if absolute.value() <= 1.0 =>
                {
                    let square = tape.scalar_multiply(error, error)?;
                    tape.scalar_scale(square, 0.5)?
                }
                NativeTrainingLoss::Huber | NativeTrainingLoss::SmoothL1 => {
                    tape.scalar_add(absolute, ScalarAutogradValue::constant(-0.5))?
                }
            };
            let normalized = tape.scalar_scale(sample_loss, 1.0 / sampled_indices.len() as f32)?;
            loss = tape.scalar_add(loss, normalized)?;
        }
        Ok(NativeTrainingObjectiveEvaluation {
            loss: loss.value(),
            scalar_loss: loss,
        })
    }
}

#[derive(Debug, Error)]
pub enum NativeTrainingError {
    #[error("native training request is invalid: {0}")]
    Invalid(&'static str),
    #[error("native training encountered a non-finite loss")]
    NonFiniteLoss,
    #[error("native training checkpoint recomputation failed at microstep {0}")]
    CheckpointFailure(u64),
    #[error("native training was cancelled")]
    Cancelled,
    #[error("native training allocation failed")]
    AllocationFailed,
    #[error("native training shape overflowed")]
    ShapeOverflow,
    #[error("native training autograd failed: {0}")]
    Autograd(String),
    #[error("native training optimizer failed: {0}")]
    Optimizer(String),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<AutogradError> for NativeTrainingError {
    fn from(error: AutogradError) -> Self {
        match error {
            AutogradError::Cancelled => Self::Cancelled,
            AutogradError::AllocationFailed => Self::AllocationFailed,
            AutogradError::InvalidGraph { ref reason } if reason.contains("non-finite") => {
                Self::NonFiniteLoss
            }
            error => Self::Autograd(error.to_string()),
        }
    }
}

enum NativeTrainingOptimizerState {
    AdamW(crate::generated_elementwise_or_runtime_operation_09::NativeAdamW),
    Adam(crate::generated_elementwise_or_runtime_operation_11::NativeAdam),
    Sgd(crate::generated_elementwise_or_runtime_operation_07::NativeSgd),
    Rmsprop(crate::generated_elementwise_or_runtime_operation_23::NativeRmsprop),
}

impl NativeTrainingOptimizerState {
    fn checked(
        optimizer: NativeTrainingOptimizer,
        parameters: &[Tensor],
        learning_rate: f32,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeTrainingError> {
        Ok(match optimizer {
            NativeTrainingOptimizer::AdamW => Self::AdamW(
                crate::generated_elementwise_or_runtime_operation_09::NativeAdamW::new_with_context_exact_native(
                    backend, parameters, learning_rate, 0.9, 0.999, 1.0e-8, 0.01, false,
                    false, context,
                )
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string()))?,
            ),
            NativeTrainingOptimizer::Adam => Self::Adam(
                crate::generated_elementwise_or_runtime_operation_11::NativeAdam::new_with_context_exact_native(
                    backend, parameters, learning_rate, context,
                )
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string()))?,
            ),
            NativeTrainingOptimizer::Sgd => Self::Sgd(
                crate::generated_elementwise_or_runtime_operation_07::NativeSgd::new_exact_native(
                    parameters.len(), learning_rate, 0.0, 0.0, 0.0, false, false,
                    context.cancellation,
                )
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string()))?,
            ),
            NativeTrainingOptimizer::Rmsprop => Self::Rmsprop(
                crate::generated_elementwise_or_runtime_operation_23::NativeRmsprop::new_with_context_exact_native(
                    backend, parameters, learning_rate, 0.99, 1.0e-8, 0.0, 0.0, false,
                    false, context,
                )
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string()))?,
            ),
        })
    }

    fn step(
        &mut self,
        backend: &CpuBackend,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeTrainingError> {
        match self {
            Self::AdamW(optimizer) => optimizer
                .step_with_context_exact_native(backend, parameters, gradients, context)
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string())),
            Self::Adam(optimizer) => optimizer
                .step_with_context_exact_native(backend, parameters, gradients, context)
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string())),
            Self::Sgd(optimizer) => optimizer
                .step_with_context_exact_native(backend, parameters, gradients, context)
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string())),
            Self::Rmsprop(optimizer) => optimizer
                .step_with_context_exact_native(backend, parameters, gradients, context)
                .map_err(|error| NativeTrainingError::Optimizer(error.to_string())),
        }
    }
}

struct NativeTrainingGradientReducer;

impl GradientReducer for NativeTrainingGradientReducer {
    fn add(
        &self,
        left: Tensor,
        right: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError> {
        cancellation_check(cancellation)?;
        if left.descriptor() != right.descriptor() || left.descriptor().dtype() != DType::F32 {
            return Err(AutogradError::InvalidGraph {
                reason: "training gradient descriptors differ".to_owned(),
            });
        }
        let mut output = left.clone();
        let element_count = output.descriptor().element_count()?;
        let shape = output.descriptor().shape().to_vec();
        let mut write = output.write()?;
        for index in 0..element_count {
            if index.is_multiple_of(1024) {
                cancellation_check(cancellation)?;
            }
            let indices =
                training_unravel_index(index, &shape).map_err(training_error_as_autograd)?;
            let left_value =
                training_read_f32(&left, &indices).map_err(training_error_as_autograd)?;
            let right_value =
                training_read_f32(&right, &indices).map_err(training_error_as_autograd)?;
            write
                .element_bytes_mut(&indices)?
                .copy_from_slice(&(left_value + right_value).to_le_bytes());
        }
        drop(write);
        Ok(output)
    }
}

pub fn train_lora_parameters_exact_native(
    backend: &CpuBackend,
    parameters: Vec<Tensor>,
    examples: &[NativeTrainingExample],
    plan: &NativeTrainingPlan,
    context: &ExecutionContext<'_>,
) -> Result<NativeTrainingResult, NativeTrainingError> {
    if examples.iter().any(|example| {
        !example.feature.is_finite()
            || !example.target.is_finite()
            || !example.base_prediction.is_finite()
    }) {
        return Err(NativeTrainingError::Invalid(
            "training examples contain non-finite values",
        ));
    }
    let mut objective = NativeScalarTrainingObjective { examples };
    train_lora_parameters_with_objective_exact_native(
        backend,
        parameters,
        examples.len(),
        &mut objective,
        plan,
        context,
    )
}

pub fn train_lora_parameters_with_objective_exact_native(
    backend: &CpuBackend,
    mut parameters: Vec<Tensor>,
    dataset_length: usize,
    objective: &mut dyn NativeTrainingObjective,
    plan: &NativeTrainingPlan,
    context: &ExecutionContext<'_>,
) -> Result<NativeTrainingResult, NativeTrainingError> {
    context.check().map_err(map_training_tensor_error)?;
    validate_native_training(parameters.as_slice(), dataset_length, plan)?;
    let mut optimizer = NativeTrainingOptimizerState::checked(
        plan.optimizer,
        &parameters,
        plan.learning_rate,
        backend,
        context,
    )?;
    let total_microsteps = plan.total_microsteps;
    let loss_capacity =
        usize::try_from(total_microsteps).map_err(|_| NativeTrainingError::ShapeOverflow)?;
    let mut losses = Vec::new();
    losses
        .try_reserve_exact(loss_capacity)
        .map_err(|_| NativeTrainingError::AllocationFailed)?;
    let mut sampled_indices = Vec::new();
    let sample_capacity = loss_capacity
        .checked_mul(
            usize::try_from(plan.batch_size).map_err(|_| NativeTrainingError::ShapeOverflow)?,
        )
        .ok_or(NativeTrainingError::ShapeOverflow)?;
    sampled_indices
        .try_reserve_exact(sample_capacity)
        .map_err(|_| NativeTrainingError::AllocationFailed)?;
    let mut checkpoint_boundaries = Vec::new();
    checkpoint_boundaries
        .try_reserve_exact(
            usize::try_from(plan.optimizer_steps)
                .map_err(|_| NativeTrainingError::ShapeOverflow)?,
        )
        .map_err(|_| NativeTrainingError::AllocationFailed)?;
    let leaves = (0..parameters.len())
        .map(|index| LeafId::new(format!("lora_parameter_{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut gradient_store = GradientStore::default();
    let reducer = NativeTrainingGradientReducer;
    let batch_size =
        usize::try_from(plan.batch_size).map_err(|_| NativeTrainingError::ShapeOverflow)?;
    let mut completed_optimizer_steps = 0_u64;
    for microstep in 0..total_microsteps {
        context.check().map_err(map_training_tensor_error)?;
        if plan.cancellation_at_microstep == Some(microstep) {
            return Err(NativeTrainingError::Cancelled);
        }
        if plan.checkpoint_failure_at_microstep == Some(microstep) {
            return Err(NativeTrainingError::CheckpointFailure(microstep));
        }
        let batch_start = sampled_indices.len();
        for batch_ordinal in 0..batch_size {
            context.check().map_err(map_training_tensor_error)?;
            let phase =
                u64::try_from(batch_ordinal).map_err(|_| NativeTrainingError::ShapeOverflow)?;
            let sample = objective.sample_index(plan.seed, microstep, phase, dataset_length)?;
            sampled_indices.push(sample);
        }
        let batch_indices = sampled_indices
            .get(batch_start..)
            .ok_or(NativeTrainingError::ShapeOverflow)?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_scalar_autocast_dtype(match plan.compute_dtype {
            NativeTrainingComputeDType::Bf16 => Some(DType::Bf16),
            NativeTrainingComputeDType::F32 | NativeTrainingComputeDType::Preserve => None,
        })?;
        let scalar_parameters = parameters
            .iter()
            .zip(&leaves)
            .map(|(parameter, leaf)| {
                tape.record_scalar_parameter(parameter, leaf.clone(), context)
                    .map_err(NativeTrainingError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut evaluation = objective.evaluate(
            backend,
            &parameters,
            &scalar_parameters,
            batch_indices,
            microstep,
            plan,
            &mut tape,
            context,
        )?;
        if !evaluation.loss.is_finite() {
            return Err(NativeTrainingError::NonFiniteLoss);
        }
        if plan.gradient_checkpointing {
            let loss_descriptor = TensorDescriptor::contiguous(
                vec![1],
                DType::F32,
                DeviceId::CPU,
                parameters[0].descriptor().stream(),
            )?;
            let loss_tensor = backend
                .fill(
                    Scalar::Float(f64::from(evaluation.loss)),
                    loss_descriptor,
                    context,
                )?
                .0;
            let autocast_dtype = match plan.compute_dtype {
                NativeTrainingComputeDType::Bf16 => DType::Bf16,
                NativeTrainingComputeDType::F32 | NativeTrainingComputeDType::Preserve => {
                    DType::F32
                }
            };
            let autocast = AutocastPolicy::new(
                plan.compute_dtype == NativeTrainingComputeDType::Bf16,
                autocast_dtype,
                !plan.offloading,
            )?;
            let mut checkpoint = checkpoint_execution_from_outputs_exact_native(
                &parameters,
                vec![loss_tensor],
                vec![true; parameters.len()],
                true,
                autocast,
                context.cancellation,
            )
            .map_err(|_| NativeTrainingError::CheckpointFailure(microstep))?;
            let expected_loss = evaluation.loss;
            for _ in 0..plan.checkpoint_depth {
                let recompute_parameters = checkpoint
                    .shallow_recompute_inputs_exact_native(context.cancellation)
                    .map_err(|_| NativeTrainingError::CheckpointFailure(microstep))?;
                tape.release_scalar_activations()?;
                tape = AutogradTape::new(GradientMode::Enabled);
                tape.set_scalar_autocast_dtype(match plan.compute_dtype {
                    NativeTrainingComputeDType::Bf16 => Some(DType::Bf16),
                    NativeTrainingComputeDType::F32 | NativeTrainingComputeDType::Preserve => None,
                })?;
                let recompute_scalars = recompute_parameters
                    .iter()
                    .zip(&leaves)
                    .map(|(parameter, leaf)| {
                        tape.record_scalar_parameter(parameter, leaf.clone(), context)
                            .map_err(NativeTrainingError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                evaluation = objective.evaluate(
                    backend,
                    &recompute_parameters,
                    &recompute_scalars,
                    batch_indices,
                    microstep,
                    plan,
                    &mut tape,
                    context,
                )?;
                if evaluation.loss.to_bits() != expected_loss.to_bits() {
                    return Err(NativeTrainingError::CheckpointFailure(microstep));
                }
            }
            checkpoint.release();
        }
        losses.push(evaluation.loss);
        tape.reverse_scalar_and_publish(
            evaluation.scalar_loss,
            &reducer,
            &mut gradient_store,
            backend,
            context,
        )?;
        let accumulation_boundary = (microstep + 1)
            .is_multiple_of(plan.gradient_accumulation_steps)
            || microstep + 1 == total_microsteps;
        if accumulation_boundary {
            let accumulated = (microstep % plan.gradient_accumulation_steps) + 1;
            let gradients = leaves
                .iter()
                .map(|leaf| {
                    let gradient = gradient_store.gradient(leaf).ok_or_else(|| {
                        NativeTrainingError::Autograd(
                            "training gradient was not published".to_owned(),
                        )
                    })?;
                    training_scaled(backend, gradient, 1.0 / accumulated as f32, context)
                })
                .collect::<Result<Vec<_>, _>>()?;
            optimizer.step(backend, &mut parameters, &gradients, context)?;
            completed_optimizer_steps = completed_optimizer_steps
                .checked_add(1)
                .ok_or(NativeTrainingError::ShapeOverflow)?;
            gradient_store.zero_grad(backend, context, true)?;
            if plan.gradient_checkpointing {
                checkpoint_boundaries.push(microstep + 1);
            }
        }
    }
    context.check().map_err(map_training_tensor_error)?;
    Ok(NativeTrainingResult {
        parameters,
        losses,
        sampled_indices,
        completed_optimizer_steps,
        checkpoint_boundaries,
    })
}

fn validate_native_training(
    parameters: &[Tensor],
    dataset_length: usize,
    plan: &NativeTrainingPlan,
) -> Result<(), NativeTrainingError> {
    if parameters.is_empty()
        || dataset_length == 0
        || dataset_length > 1_000_000
        || plan.optimizer_steps == 0
        || plan.optimizer_steps > 100_000
        || plan.total_microsteps == 0
        || plan.total_microsteps
            > plan
                .optimizer_steps
                .saturating_mul(plan.gradient_accumulation_steps)
        || plan.gradient_accumulation_steps == 0
        || plan.gradient_accumulation_steps > 1_024
        || plan.batch_size == 0
        || plan.batch_size > 10_000
        || !plan.learning_rate.is_finite()
        || !(1.0e-7..=1.0).contains(&plan.learning_rate)
        || !(1..=5).contains(&plan.checkpoint_depth)
        || parameters.iter().any(|parameter| {
            parameter.descriptor().dtype() != DType::F32
                || parameter.descriptor().device() != DeviceId::CPU
                || parameter.descriptor().element_count().is_err()
                || parameter.descriptor().shape().contains(&0)
        })
    {
        return Err(NativeTrainingError::Invalid(
            "training plan, dataset, or parameters are unsupported",
        ));
    }
    Ok(())
}

fn quantize_training_value(value: f32, dtype: NativeTrainingComputeDType) -> f32 {
    match dtype {
        NativeTrainingComputeDType::Bf16 => f32::from_bits(value.to_bits() & 0xffff_0000),
        NativeTrainingComputeDType::F32 | NativeTrainingComputeDType::Preserve => value,
    }
}

fn native_training_sample_index(
    seed: u64,
    microstep: u64,
    length: usize,
) -> Result<usize, NativeTrainingError> {
    let length = u64::try_from(length).map_err(|_| NativeTrainingError::ShapeOverflow)?;
    if length == 0 {
        return Err(NativeTrainingError::Invalid("training dataset is empty"));
    }
    let value = native_training_random(seed, microstep, 0);
    usize::try_from(value % length).map_err(|_| NativeTrainingError::ShapeOverflow)
}

fn native_training_unit_random(seed: u64, microstep: u64, phase: u64) -> f32 {
    let value = native_training_random(seed, microstep, phase) >> 40;
    value as f32 / ((1_u32 << 24) - 1) as f32
}

fn native_training_random(seed: u64, microstep: u64, phase: u64) -> u64 {
    let mut value = seed
        .wrapping_add(microstep.wrapping_mul(1_000))
        .wrapping_add(phase.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn training_scaled(
    backend: &CpuBackend,
    tensor: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeTrainingError> {
    let mut output = backend
        .fill(Scalar::Float(0.0), tensor.descriptor().clone(), context)?
        .0;
    let element_count = tensor.descriptor().element_count()?;
    let mut write = output.write()?;
    for index in 0..element_count {
        if index.is_multiple_of(1024) {
            context.check().map_err(map_training_tensor_error)?;
        }
        let indices = training_unravel_index(index, tensor.descriptor().shape())?;
        write
            .element_bytes_mut(&indices)?
            .copy_from_slice(&(training_read_f32(tensor, &indices)? * scale).to_le_bytes());
    }
    drop(write);
    Ok(output)
}

fn training_read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, NativeTrainingError> {
    let bytes = tensor.element_bytes(indices)?;
    let bytes = <[u8; 4]>::try_from(bytes).map_err(|_| {
        NativeTrainingError::Invalid("training tensor does not contain F32 elements")
    })?;
    Ok(f32::from_le_bytes(bytes))
}

fn training_unravel_index(
    mut linear_index: u64,
    shape: &[u64],
) -> Result<Vec<u64>, NativeTrainingError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.len())
        .map_err(|_| NativeTrainingError::AllocationFailed)?;
    indices.resize(shape.len(), 0);
    for dimension_index in (0..shape.len()).rev() {
        let dimension = shape[dimension_index];
        if dimension == 0 {
            return Err(NativeTrainingError::ShapeOverflow);
        }
        indices[dimension_index] = linear_index % dimension;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn map_training_tensor_error(error: TensorError) -> NativeTrainingError {
    match error {
        TensorError::Cancelled => NativeTrainingError::Cancelled,
        error => NativeTrainingError::Tensor(error),
    }
}

fn training_error_as_autograd(error: NativeTrainingError) -> AutogradError {
    match error {
        NativeTrainingError::Tensor(error) => AutogradError::Tensor(error),
        NativeTrainingError::Cancelled => AutogradError::Cancelled,
        error => AutogradError::InvalidGraph {
            reason: error.to_string(),
        },
    }
}

impl GradientStore {
    pub fn publish(
        &mut self,
        gradients: HashMap<LeafId, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<(), AutogradError> {
        cancellation_check(cancellation)?;
        self.gradients = gradients;
        Ok(())
    }

    pub fn gradient(&self, leaf: &LeafId) -> Option<&Tensor> {
        self.gradients.get(leaf)
    }

    pub fn accumulate(
        &mut self,
        gradients: HashMap<LeafId, Tensor>,
        reducer: &dyn GradientReducer,
        cancellation: &CancellationToken,
    ) -> Result<(), AutogradError> {
        cancellation_check(cancellation)?;
        let mut staged = self.gradients.clone();
        for (leaf, gradient) in gradients {
            accumulate(&mut staged, leaf, gradient, reducer, cancellation)?;
        }
        cancellation_check(cancellation)?;
        self.gradients = staged;
        Ok(())
    }

    fn accumulate_higher_order(
        &self,
        gradients: HashMap<LeafId, Tensor>,
        reducer: &dyn GradientReducer,
        backend: &CpuBackend,
        execution: &ExecutionContext<'_>,
        derivative_graph: &mut AutogradTape,
    ) -> Result<HashMap<LeafId, Tensor>, AutogradError> {
        cancellation_check(execution.cancellation)?;
        let mut staged = self.gradients.clone();
        for (leaf, gradient) in gradients {
            accumulate_higher_order(
                &mut staged,
                leaf,
                gradient,
                reducer,
                backend,
                execution,
                derivative_graph,
            )?;
        }
        cancellation_check(execution.cancellation)?;
        Ok(staged)
    }

    pub fn zero_grad(
        &mut self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        set_to_none: bool,
    ) -> Result<(), AutogradError> {
        context.check()?;
        if set_to_none {
            self.gradients.clear();
            return Ok(());
        }
        let mut staged = HashMap::with_capacity(self.gradients.len());
        for (leaf, gradient) in &self.gradients {
            let (zero, _) = backend.fill(
                crate::Scalar::Float(0.0),
                gradient.descriptor().clone(),
                context,
            )?;
            staged.insert(leaf.clone(), zero);
        }
        context.check()?;
        self.gradients = staged;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }
}

fn accumulate<Key: std::hash::Hash + Eq>(
    gradients: &mut HashMap<Key, Tensor>,
    key: Key,
    gradient: Tensor,
    reducer: &dyn GradientReducer,
    cancellation: &CancellationToken,
) -> Result<(), AutogradError> {
    cancellation_check(cancellation)?;
    let value = if let Some(existing) = gradients.remove(&key) {
        if existing.descriptor() != gradient.descriptor() {
            return Err(AutogradError::GradientDescriptorMismatch);
        }
        let descriptor = existing.descriptor().clone();
        let reduced = reducer.add(existing, gradient, cancellation)?;
        if reduced.descriptor() != &descriptor {
            return Err(AutogradError::GradientDescriptorMismatch);
        }
        reduced
    } else {
        gradient
    };
    gradients.insert(key, value);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_higher_order<Key: std::hash::Hash + Eq>(
    gradients: &mut HashMap<Key, Tensor>,
    key: Key,
    gradient: Tensor,
    reducer: &dyn GradientReducer,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
    derivative_graph: &mut AutogradTape,
) -> Result<(), AutogradError> {
    cancellation_check(execution.cancellation)?;
    let value = if let Some(existing) = gradients.remove(&key) {
        if existing.descriptor() != gradient.descriptor() {
            return Err(AutogradError::GradientDescriptorMismatch);
        }
        let descriptor = existing.descriptor().clone();
        let reduced = {
            let mut context = HigherOrderContext {
                backend,
                execution,
                tape: derivative_graph,
            };
            reducer.add_higher_order(existing, gradient, &mut context)?
        };
        if reduced.descriptor() != &descriptor {
            return Err(AutogradError::GradientDescriptorMismatch);
        }
        reduced
    } else {
        gradient
    };
    gradients.insert(key, value);
    Ok(())
}

fn cancellation_check(cancellation: &CancellationToken) -> Result<(), AutogradError> {
    if cancellation.is_cancelled() {
        Err(AutogradError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AutogradError {
    #[error("autograd graph is invalid: {reason}")]
    InvalidGraph { reason: String },
    #[error("requires_grad is valid only for floating-point or complex tensors, not {dtype:?}")]
    InvalidRequiresGradDType { dtype: crate::DType },
    #[error("autograd node identifier overflowed")]
    IdentifierOverflow,
    #[error("autograd allocation failed")]
    AllocationFailed,
    #[error("autograd rule {symbol} rejects create_graph under {policy:?} policy")]
    HigherOrderUnavailable {
        symbol: &'static str,
        policy: breadth::HigherOrderPolicy,
    },
    #[error("analytical autograd rule {symbol} has no recorded higher-order implementation")]
    MissingHigherOrderRule { symbol: &'static str },
    #[error(
        "saved logical tensor {tensor:?} changed from version {expected_version} to {actual_version}"
    )]
    SavedTensorModified {
        tensor: TensorId,
        expected_version: u64,
        actual_version: u64,
    },
    #[error("checkpoint state has already been released")]
    ReleasedCheckpoint,
    #[error("VJP returned {actual} input gradients, expected {expected}")]
    GradientArity { expected: usize, actual: usize },
    #[error("gradient accumulation changed or combined incompatible tensor descriptors")]
    GradientDescriptorMismatch,
    #[error("autograd tape is terminal: {state:?}")]
    TerminalTape { state: TapeState },
    #[error("autograd execution was cancelled")]
    Cancelled,
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuWorkspaceAuthority, DType, DeviceId, StreamId, TensorDescriptor};
    use std::{
        collections::{BTreeMap, BTreeSet},
        error::Error,
    };

    fn scalar(value: f32) -> Tensor {
        let descriptor = match TensorDescriptor::contiguous(
            vec![],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        ) {
            Ok(value) => value,
            Err(error) => panic!("scalar descriptor failed: {error}"),
        };
        match Tensor::from_bytes(descriptor, value.to_le_bytes().to_vec()) {
            Ok(value) => value,
            Err(error) => panic!("scalar tensor failed: {error}"),
        }
    }

    fn scalar_value(tensor: &Tensor) -> f32 {
        let bytes = match tensor.contiguous_bytes() {
            Ok(value) => value,
            Err(error) => panic!("scalar read failed: {error}"),
        };
        let Some(bytes) = bytes.get(..4) else {
            panic!("scalar requires four bytes");
        };
        let Ok(bytes) = <[u8; 4]>::try_from(bytes) else {
            panic!("scalar requires four bytes");
        };
        f32::from_le_bytes(bytes)
    }

    #[test]
    fn native_lora_training_matrix_is_deterministic_and_flushes_remainder()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(8 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let examples = [
            NativeTrainingExample {
                feature: 0.25,
                target: -0.5,
                base_prediction: 0.1,
            },
            NativeTrainingExample {
                feature: -0.75,
                target: 0.25,
                base_prediction: -0.2,
            },
            NativeTrainingExample {
                feature: 0.5,
                target: 0.75,
                base_prediction: 0.4,
            },
        ];
        for optimizer in [
            NativeTrainingOptimizer::AdamW,
            NativeTrainingOptimizer::Adam,
            NativeTrainingOptimizer::Sgd,
            NativeTrainingOptimizer::Rmsprop,
        ] {
            for loss in [
                NativeTrainingLoss::Mse,
                NativeTrainingLoss::L1,
                NativeTrainingLoss::Huber,
                NativeTrainingLoss::SmoothL1,
            ] {
                for compute_dtype in [
                    NativeTrainingComputeDType::Bf16,
                    NativeTrainingComputeDType::F32,
                    NativeTrainingComputeDType::Preserve,
                ] {
                    let plan = NativeTrainingPlan {
                        optimizer,
                        loss,
                        compute_dtype,
                        learning_rate: 0.01,
                        optimizer_steps: 2,
                        total_microsteps: 3,
                        gradient_accumulation_steps: 2,
                        batch_size: 2,
                        seed: 44,
                        gradient_checkpointing: true,
                        checkpoint_depth: 2,
                        offloading: true,
                        quantized_backward: false,
                        checkpoint_failure_at_microstep: None,
                        cancellation_at_microstep: None,
                    };
                    let first = train_lora_parameters_exact_native(
                        &backend,
                        vec![scalar(0.0)],
                        &examples,
                        &plan,
                        &context,
                    )?;
                    let second = train_lora_parameters_exact_native(
                        &backend,
                        vec![scalar(0.0)],
                        &examples,
                        &plan,
                        &context,
                    )?;
                    assert_eq!(first.losses, second.losses);
                    assert_eq!(first.sampled_indices, second.sampled_indices);
                    assert_eq!(first.sampled_indices.len(), 6);
                    assert_eq!(first.checkpoint_boundaries, [2, 3]);
                    assert_eq!(first.completed_optimizer_steps, 2);
                    let first_parameter = first
                        .parameters
                        .first()
                        .ok_or("first training result has no parameter")?;
                    let second_parameter = second
                        .parameters
                        .first()
                        .ok_or("second training result has no parameter")?;
                    assert_eq!(
                        first_parameter.contiguous_bytes()?,
                        second_parameter.contiguous_bytes()?
                    );
                    assert_ne!(scalar_value(first_parameter), 0.0);
                    if compute_dtype == NativeTrainingComputeDType::F32 {
                        let expected = match (optimizer, loss) {
                            (NativeTrainingOptimizer::AdamW, NativeTrainingLoss::Mse) => {
                                0x3b9f_8075
                            }
                            (NativeTrainingOptimizer::AdamW, NativeTrainingLoss::L1) => 0x3bc2_db45,
                            (NativeTrainingOptimizer::AdamW, NativeTrainingLoss::Huber) => {
                                0x3b9f_8075
                            }
                            (NativeTrainingOptimizer::AdamW, NativeTrainingLoss::SmoothL1) => {
                                0x3b9f_8075
                            }
                            (NativeTrainingOptimizer::Adam, NativeTrainingLoss::Mse) => 0x3b9f_88d9,
                            (NativeTrainingOptimizer::Adam, NativeTrainingLoss::L1) => 0x3bc2_e3a9,
                            (NativeTrainingOptimizer::Adam, NativeTrainingLoss::Huber) => {
                                0x3b9f_88d9
                            }
                            (NativeTrainingOptimizer::Adam, NativeTrainingLoss::SmoothL1) => {
                                0x3b9f_88d9
                            }
                            (NativeTrainingOptimizer::Sgd, NativeTrainingLoss::Mse) => 0xbb9c_dc24,
                            (NativeTrainingOptimizer::Sgd, NativeTrainingLoss::L1) => 0xbb84_9430,
                            (NativeTrainingOptimizer::Sgd, NativeTrainingLoss::Huber) => {
                                0xbb1c_4b42
                            }
                            (NativeTrainingOptimizer::Sgd, NativeTrainingLoss::SmoothL1) => {
                                0xbb1c_4b42
                            }
                            (NativeTrainingOptimizer::Rmsprop, NativeTrainingLoss::Mse) => {
                                0x3b4d_8fe0
                            }
                            (NativeTrainingOptimizer::Rmsprop, NativeTrainingLoss::L1) => {
                                0x3c0d_fe78
                            }
                            (NativeTrainingOptimizer::Rmsprop, NativeTrainingLoss::Huber) => {
                                0x3b4d_8f60
                            }
                            (NativeTrainingOptimizer::Rmsprop, NativeTrainingLoss::SmoothL1) => {
                                0x3b4d_8f60
                            }
                        };
                        assert_eq!(scalar_value(first_parameter).to_bits(), expected);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_lora_training_rejects_each_atomic_failure_boundary() -> Result<(), Box<dyn Error>> {
        struct DivergingCheckpointObjective {
            evaluations: usize,
        }

        impl NativeTrainingObjective for DivergingCheckpointObjective {
            fn evaluate(
                &mut self,
                _backend: &CpuBackend,
                _parameters: &[Tensor],
                scalar_parameters: &[Vec<ScalarAutogradValue>],
                _sampled_indices: &[usize],
                _microstep: u64,
                _plan: &NativeTrainingPlan,
                _tape: &mut AutogradTape,
                _context: &ExecutionContext<'_>,
            ) -> Result<NativeTrainingObjectiveEvaluation, NativeTrainingError> {
                let loss = if self.evaluations == 0 { 1.0 } else { 2.0 };
                self.evaluations = self.evaluations.saturating_add(1);
                Ok(NativeTrainingObjectiveEvaluation {
                    loss,
                    scalar_loss: scalar_parameters
                        .first()
                        .and_then(|parameter| parameter.first())
                        .copied()
                        .ok_or(NativeTrainingError::Invalid("missing scalar parameter"))?,
                })
            }
        }

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let example = NativeTrainingExample {
            feature: 0.5,
            target: 0.25,
            base_prediction: -0.5,
        };
        let plan = NativeTrainingPlan {
            optimizer: NativeTrainingOptimizer::Sgd,
            loss: NativeTrainingLoss::Mse,
            compute_dtype: NativeTrainingComputeDType::F32,
            learning_rate: 0.01,
            optimizer_steps: 2,
            total_microsteps: 4,
            gradient_accumulation_steps: 2,
            batch_size: 1,
            seed: 4,
            gradient_checkpointing: true,
            checkpoint_depth: 1,
            offloading: false,
            quantized_backward: false,
            checkpoint_failure_at_microstep: None,
            cancellation_at_microstep: None,
        };
        for microstep in 0..plan.total_microsteps {
            let mut cancelled = plan.clone();
            cancelled.cancellation_at_microstep = Some(microstep);
            assert!(matches!(
                train_lora_parameters_exact_native(
                    &backend,
                    vec![scalar(0.0)],
                    &[example],
                    &cancelled,
                    &context,
                ),
                Err(NativeTrainingError::Cancelled)
            ));
        }
        let mut checkpoint_failure = plan.clone();
        checkpoint_failure.checkpoint_failure_at_microstep = Some(2);
        assert!(matches!(
            train_lora_parameters_exact_native(
                &backend,
                vec![scalar(0.0)],
                &[example],
                &checkpoint_failure,
                &context,
            ),
            Err(NativeTrainingError::CheckpointFailure(2))
        ));
        assert!(matches!(
            train_lora_parameters_with_objective_exact_native(
                &backend,
                vec![scalar(0.0)],
                1,
                &mut DivergingCheckpointObjective { evaluations: 0 },
                &plan,
                &context,
            ),
            Err(NativeTrainingError::CheckpointFailure(0))
        ));
        let non_finite = NativeTrainingExample {
            feature: f32::MAX,
            target: -f32::MAX,
            base_prediction: f32::MAX,
        };
        assert!(matches!(
            train_lora_parameters_exact_native(
                &backend,
                vec![scalar(f32::MAX)],
                &[non_finite],
                &plan,
                &context,
            ),
            Err(NativeTrainingError::NonFiniteLoss)
        ));
        let (small_backend, small_authority) = CpuWorkspaceAuthority::create_backend(64)?;
        let small_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: small_authority.authorize_workspace(8)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let constrained = train_lora_parameters_exact_native(
            &small_backend,
            vec![scalar(0.0)],
            &[example],
            &plan,
            &small_context,
        );
        assert!(
            matches!(
                constrained,
                Err(NativeTrainingError::Tensor(
                    TensorError::WorkspaceAuthorizationExceeded { .. }
                )) | Err(NativeTrainingError::Optimizer(_))
                    | Err(NativeTrainingError::AllocationFailed)
            ),
            "unexpected constrained-training result: {constrained:?}",
        );
        Ok(())
    }

    struct AddReducer;

    impl GradientReducer for AddReducer {
        fn add(
            &self,
            left: Tensor,
            right: Tensor,
            cancellation: &CancellationToken,
        ) -> Result<Tensor, AutogradError> {
            cancellation_check(cancellation)?;
            Ok(scalar(scalar_value(&left) + scalar_value(&right)))
        }
    }

    struct IdentityRule;

    impl BackwardRule for IdentityRule {
        fn vjp(
            &self,
            output_gradients: &[Option<Tensor>],
            _saved_tensors: &[SavedTensor],
            cancellation: &CancellationToken,
        ) -> Result<Vec<Option<Tensor>>, AutogradError> {
            cancellation_check(cancellation)?;
            Ok(vec![output_gradients.first().cloned().flatten()])
        }
    }

    #[test]
    fn leaf_identifier_wire_adapter_revalidates_invariants() {
        assert!(LeafId::try_from(LeafIdWire("\t".to_owned())).is_err());
        assert!(matches!(
            LeafId::try_from(LeafIdWire("weight".to_owned())),
            Ok(identifier) if identifier.as_str() == "weight"
        ));
    }

    #[test]
    fn reverse_mode_traverses_nodes_and_accumulates_leaf_gradients() {
        let leaf = match LeafId::new("x") {
            Ok(value) => value,
            Err(error) => panic!("leaf failed: {error}"),
        };
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let first = match tape.record(
            vec![AutogradInput::Leaf(leaf.clone())],
            1,
            vec![],
            Arc::new(IdentityRule),
        ) {
            Ok(Some(value)) => value,
            Ok(None) => panic!("enabled tape should record"),
            Err(error) => panic!("record failed: {error}"),
        };
        let Some(first_output) = first.first().copied() else {
            panic!("first node should have an output");
        };
        let second = match tape.record(
            vec![AutogradInput::Node(first_output)],
            1,
            vec![],
            Arc::new(IdentityRule),
        ) {
            Ok(Some(value)) => value,
            Ok(None) => panic!("enabled tape should record"),
            Err(error) => panic!("record failed: {error}"),
        };
        let Some(second_output) = second.first().copied() else {
            panic!("second node should have an output");
        };
        let gradients = match tape.backward(
            vec![(second_output, scalar(3.0))],
            &AddReducer,
            &CancellationToken::default(),
        ) {
            Ok(value) => value,
            Err(error) => panic!("backward failed: {error}"),
        };
        let Some(gradient) = gradients.get(&leaf) else {
            panic!("leaf gradient should exist");
        };
        assert_eq!(scalar_value(gradient), 3.0);
        assert_eq!(tape.state(), &TapeState::Completed);
        assert_eq!(tape.retained_node_count(), 0);
    }

    #[test]
    fn branches_accumulate_with_the_declared_reducer() {
        let leaf = match LeafId::new("x") {
            Ok(value) => value,
            Err(error) => panic!("leaf failed: {error}"),
        };
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let recorded = match tape.record(
                vec![AutogradInput::Leaf(leaf.clone())],
                1,
                vec![],
                Arc::new(IdentityRule),
            ) {
                Ok(Some(value)) => value,
                Ok(None) => panic!("enabled tape should record"),
                Err(error) => panic!("record failed: {error}"),
            };
            let Some(output) = recorded.first().copied() else {
                panic!("recorded node should have output");
            };
            outputs.push(output);
        }
        let seeds = outputs
            .into_iter()
            .map(|output| (output, scalar(2.0)))
            .collect();
        let gradients = match tape.backward(seeds, &AddReducer, &CancellationToken::default()) {
            Ok(value) => value,
            Err(error) => panic!("backward failed: {error}"),
        };
        let Some(gradient) = gradients.get(&leaf) else {
            panic!("leaf gradient should exist");
        };
        assert_eq!(scalar_value(gradient), 4.0);
    }

    #[test]
    fn cancellation_releases_saved_tensors_and_is_terminal() {
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let recorded = tape.record(
            vec![AutogradInput::Constant],
            1,
            vec![SavedTensor::capture(&scalar(1.0))],
            Arc::new(IdentityRule),
        );
        assert!(recorded.is_ok());
        let token = CancellationToken::default();
        token.cancel();
        let result = tape.backward(vec![], &AddReducer, &token);
        assert!(matches!(result, Err(AutogradError::Cancelled)));
        assert!(matches!(tape.state(), TapeState::Cancelled(_)));
        assert_eq!(tape.retained_node_count(), 0);
        assert!(matches!(
            tape.record(
                vec![AutogradInput::Constant],
                1,
                vec![],
                Arc::new(IdentityRule),
            ),
            Err(AutogradError::TerminalTape { .. })
        ));
    }

    #[test]
    fn no_grad_and_inference_modes_do_not_retain_rules() {
        for mode in [GradientMode::NoGrad, GradientMode::Inference] {
            let mut tape = AutogradTape::new(mode);
            let result = tape.record(
                vec![AutogradInput::Constant],
                1,
                vec![],
                Arc::new(IdentityRule),
            );
            assert!(matches!(result, Ok(None)));
            assert_eq!(tape.retained_node_count(), 0);
        }
    }

    #[test]
    fn recording_rejects_unknown_or_forward_edges_without_mutation() {
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let result = tape.record(
            vec![AutogradInput::Node(OutputSlot {
                node: TapeNodeId(9),
                output: 0,
            })],
            1,
            vec![],
            Arc::new(IdentityRule),
        );
        assert!(matches!(result, Err(AutogradError::InvalidGraph { .. })));
        assert_eq!(tape.retained_node_count(), 0);
        let valid = tape.record(
            vec![AutogradInput::Constant],
            1,
            vec![],
            Arc::new(IdentityRule),
        );
        assert!(matches!(
            valid,
            Ok(Some(outputs)) if outputs.first().is_some_and(|slot| slot.node.get() == 0)
        ));
    }

    #[test]
    fn core_autograd_fixture_registry_and_state_contracts_are_complete()
    -> Result<(), Box<dyn Error>> {
        let leaf = LeafId::new("validation-leaf")?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let first = tape
            .record(
                vec![AutogradInput::Leaf(leaf.clone())],
                1,
                vec![SavedTensor::capture(&scalar(1.0))],
                Arc::new(IdentityRule),
            )?
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "enabled validation tape did not record".to_owned(),
            })?;
        let first_output = first
            .first()
            .copied()
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "validation node has no output".to_owned(),
            })?;
        let second = tape
            .record(
                vec![
                    AutogradInput::Node(first_output),
                    AutogradInput::Leaf(leaf.clone()),
                ],
                1,
                vec![],
                Arc::new(PassthroughBothRule),
            )?
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "enabled validation tape did not record its branch".to_owned(),
            })?;
        let second_output = second
            .first()
            .copied()
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "validation branch has no output".to_owned(),
            })?;
        let gradients = tape.backward(
            vec![(second_output, scalar(2.0))],
            &AddReducer,
            &CancellationToken::default(),
        )?;
        let accumulated = gradients
            .get(&leaf)
            .is_some_and(|gradient| scalar_value(gradient) == 4.0);

        let mut cancelled_tape = AutogradTape::new(GradientMode::Enabled);
        cancelled_tape.record(
            vec![AutogradInput::Constant],
            1,
            vec![SavedTensor::capture(&scalar(1.0))],
            Arc::new(IdentityRule),
        )?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let cancelled = matches!(
            cancelled_tape.backward(vec![], &AddReducer, &cancellation),
            Err(AutogradError::Cancelled)
        ) && matches!(cancelled_tape.state(), TapeState::Cancelled(_))
            && cancelled_tape.retained_node_count() == 0;

        let modes_do_not_record = [GradientMode::NoGrad, GradientMode::Inference]
            .into_iter()
            .all(|mode| {
                let mut tape = AutogradTape::new(mode);
                matches!(
                    tape.record(
                        vec![AutogradInput::Constant],
                        1,
                        vec![],
                        Arc::new(IdentityRule),
                    ),
                    Ok(None)
                ) && tape.retained_node_count() == 0
            });

        let breadth_ids = breadth::AUTOGRAD_CONSTRUCTS
            .iter()
            .map(|contract| contract.id)
            .collect::<BTreeSet<_>>();
        let breadth_symbols = breadth::AUTOGRAD_CONSTRUCTS
            .iter()
            .map(|contract| contract.symbol)
            .collect::<BTreeSet<_>>();
        let breadth_catalog_is_closed = breadth::AUTOGRAD_CONSTRUCTS.len() == 36
            && breadth_ids.len() == 36
            && breadth_symbols.len() == 36;
        let custom_function_catalog_is_closed = breadth::AUTOGRAD_CONSTRUCTS
            .iter()
            .filter(|contract| contract.construct == "custom-autograd-function")
            .count()
            == 7;
        let breadth_fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/autograd/breadth-v1.json"
        ))?;
        let fixture_cases = breadth_fixture["catalog_cases"]
            .as_array()
            .ok_or("autograd breadth fixture catalog_cases is not an array")?;
        let fixture_ids = fixture_cases
            .iter()
            .filter_map(|case| case["id"].as_str())
            .collect::<BTreeSet<_>>();
        let fixture_symbols = fixture_cases
            .iter()
            .filter_map(|case| case["symbol"].as_str())
            .collect::<BTreeSet<_>>();
        let fixture_execution_cases = fixture_cases
            .iter()
            .filter_map(|case| case["execution_case"].as_str())
            .collect::<BTreeSet<_>>();
        let strict_fixture_registry = fixture_cases.len() == 36
            && fixture_ids == breadth_ids
            && fixture_symbols == breadth_symbols
            && fixture_execution_cases.len() == 36
            && breadth_fixture["schema_version"].as_u64() == Some(1)
            && breadth_fixture["owner_task_id"].as_str()
                == Some("comfy-parity-native-autograd-breadth")
            && breadth_fixture["oracle"]["development_only"].as_bool() == Some(true)
            && breadth_fixture["oracle"]["source_files"]
                .as_object()
                .is_some_and(|files| files.len() == 6)
            && fixture_cases.iter().all(|case| {
                case["execution_case"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty())
                    && case["source_observations"]
                        .as_array()
                        .is_some_and(|observations| !observations.is_empty())
            });
        let exact_custom_contracts = breadth::CUSTOM_FUNCTIONS
            .iter()
            .map(|contract| {
                (
                    contract.symbol,
                    contract.forward_arity,
                    contract.variadic_inputs,
                    contract.forward_outputs,
                    contract.backward_inputs,
                    contract.backward_outputs,
                    contract.higher_order,
                    contract.fixture,
                )
            })
            .eq([
                (
                    "vector_quantize",
                    2,
                    false,
                    2,
                    2,
                    2,
                    breadth::HigherOrderPolicy::Analytical,
                    "breadth-v1.json#vector_quantize",
                ),
                (
                    "CheckpointFunction",
                    3,
                    true,
                    1,
                    1,
                    3,
                    breadth::HigherOrderPolicy::FirstOrderOnly,
                    "breadth-v1.json#checkpoint_function",
                ),
                (
                    "QuantLinearFunc",
                    6,
                    false,
                    1,
                    1,
                    6,
                    breadth::HigherOrderPolicy::OnceDifferentiable,
                    ".agents/specs/comfy-parity/fixtures/quant-linear-source-oracle.json#callable",
                ),
                (
                    "HadaWeightTucker",
                    7,
                    false,
                    1,
                    1,
                    7,
                    breadth::HigherOrderPolicy::Analytical,
                    "breadth-v1.json#hada_weight_tucker",
                ),
                (
                    "AddAuxLoss",
                    2,
                    false,
                    1,
                    1,
                    2,
                    breadth::HigherOrderPolicy::Analytical,
                    "breadth-v1.json#add_aux_loss",
                ),
                (
                    "OffloadCheckpointFunction",
                    2,
                    false,
                    1,
                    1,
                    2,
                    breadth::HigherOrderPolicy::FirstOrderOnly,
                    "breadth-v1.json#offload_checkpoint",
                ),
                (
                    "HadaWeight",
                    5,
                    false,
                    1,
                    1,
                    5,
                    breadth::HigherOrderPolicy::Analytical,
                    "breadth-v1.json#hada_weight",
                ),
            ]);

        let cases = BTreeMap::from([
            (
                "autograd_breadth_catalog_has_exact_unique_coverage",
                breadth_catalog_is_closed,
            ),
            ("branch_gradients_accumulate", accumulated),
            (
                "cancellation_is_terminal_and_releases_saved_tensors",
                cancelled,
            ),
            (
                "completed_tapes_release_nodes",
                tape.retained_node_count() == 0,
            ),
            (
                "invariant_bearing_leaf_ids_use_checked_wire_conversion",
                LeafId::try_from(LeafIdWire(" ".to_owned())).is_err(),
            ),
            (
                "strict_36_row_source_fixture_registry_is_closed",
                strict_fixture_registry,
            ),
            (
                "seven_custom_function_contracts_are_exact",
                custom_function_catalog_is_closed && exact_custom_contracts,
            ),
            (
                "fixture_COMFY_AUTOGRAD_0164A83D79F9",
                fixture_ids.contains("COMFY-AUTOGRAD-0164A83D79F9"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_08DA3A226CB4",
                fixture_ids.contains("COMFY-AUTOGRAD-08DA3A226CB4"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_0BDFE52B87F3",
                fixture_ids.contains("COMFY-AUTOGRAD-0BDFE52B87F3"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_0C5FA58D517B",
                fixture_ids.contains("COMFY-AUTOGRAD-0C5FA58D517B"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_104D91298DF9",
                fixture_ids.contains("COMFY-AUTOGRAD-104D91298DF9"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_1691472B873D",
                fixture_ids.contains("COMFY-AUTOGRAD-1691472B873D"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_2682346109CE",
                fixture_ids.contains("COMFY-AUTOGRAD-2682346109CE"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_285F07173F3E",
                fixture_ids.contains("COMFY-AUTOGRAD-285F07173F3E"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_30043B9C2264",
                fixture_ids.contains("COMFY-AUTOGRAD-30043B9C2264"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_304CC342AC2A",
                fixture_ids.contains("COMFY-AUTOGRAD-304CC342AC2A"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_35DAFB8F8753",
                fixture_ids.contains("COMFY-AUTOGRAD-35DAFB8F8753"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_3CBCCC7F6931",
                fixture_ids.contains("COMFY-AUTOGRAD-3CBCCC7F6931"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_4CF4D676FFBB",
                fixture_ids.contains("COMFY-AUTOGRAD-4CF4D676FFBB"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_58A5B3D9CFE8",
                fixture_ids.contains("COMFY-AUTOGRAD-58A5B3D9CFE8"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_617621E1EEBE",
                fixture_ids.contains("COMFY-AUTOGRAD-617621E1EEBE"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_619FFDF53F34",
                fixture_ids.contains("COMFY-AUTOGRAD-619FFDF53F34"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_640C4BF17167",
                fixture_ids.contains("COMFY-AUTOGRAD-640C4BF17167"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_75400A23E6BE",
                fixture_ids.contains("COMFY-AUTOGRAD-75400A23E6BE"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_77E715FA8F5B",
                fixture_ids.contains("COMFY-AUTOGRAD-77E715FA8F5B"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_885F94147CD4",
                fixture_ids.contains("COMFY-AUTOGRAD-885F94147CD4"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_97F154ABF757",
                fixture_ids.contains("COMFY-AUTOGRAD-97F154ABF757"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_9A036C261AF5",
                fixture_ids.contains("COMFY-AUTOGRAD-9A036C261AF5"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_A1ACCD3F23F9",
                fixture_ids.contains("COMFY-AUTOGRAD-A1ACCD3F23F9"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_A1FE605E0A41",
                fixture_ids.contains("COMFY-AUTOGRAD-A1FE605E0A41"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_A50883A5EA1D",
                fixture_ids.contains("COMFY-AUTOGRAD-A50883A5EA1D"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_ABC8AAD8B0B5",
                fixture_ids.contains("COMFY-AUTOGRAD-ABC8AAD8B0B5"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_B16B6C3AAC27",
                fixture_ids.contains("COMFY-AUTOGRAD-B16B6C3AAC27"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_B575430CB29A",
                fixture_ids.contains("COMFY-AUTOGRAD-B575430CB29A"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_B6C63329EB83",
                fixture_ids.contains("COMFY-AUTOGRAD-B6C63329EB83"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_B93A2676328D",
                fixture_ids.contains("COMFY-AUTOGRAD-B93A2676328D"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_BC03B0A46C6A",
                fixture_ids.contains("COMFY-AUTOGRAD-BC03B0A46C6A"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_C235B5282FB7",
                fixture_ids.contains("COMFY-AUTOGRAD-C235B5282FB7"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_CBC045CBB408",
                fixture_ids.contains("COMFY-AUTOGRAD-CBC045CBB408"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_E31FB2A11AFF",
                fixture_ids.contains("COMFY-AUTOGRAD-E31FB2A11AFF"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_E50C96639633",
                fixture_ids.contains("COMFY-AUTOGRAD-E50C96639633"),
            ),
            (
                "fixture_COMFY_AUTOGRAD_F5EB56FAE2E4",
                fixture_ids.contains("COMFY-AUTOGRAD-F5EB56FAE2E4"),
            ),
            ("no_grad_and_inference_do_not_record", modes_do_not_record),
        ]);
        let failed = cases
            .iter()
            .filter_map(|(case, passed)| (!passed).then_some(*case))
            .collect::<Vec<_>>();
        assert!(failed.is_empty(), "core autograd cases failed: {failed:?}");
        Ok(())
    }

    struct PassthroughBothRule;

    impl BackwardRule for PassthroughBothRule {
        fn vjp(
            &self,
            output_gradients: &[Option<Tensor>],
            _saved_tensors: &[SavedTensor],
            cancellation: &CancellationToken,
        ) -> Result<Vec<Option<Tensor>>, AutogradError> {
            cancellation_check(cancellation)?;
            let gradient = output_gradients.first().cloned().flatten();
            Ok(vec![gradient.clone(), gradient])
        }
    }
}
