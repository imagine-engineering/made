use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_model::{
    QuantLinearError, QuantLinearLayout, QuantLinearOptions, QuantLinearScale, QuantLinearWeight,
    quant_linear_forward_exact_native, quantize_linear_matrix,
};
use sim_tensor::{
    AutocastPolicy, AutogradError, AutogradInput, AutogradTape, BackwardRule, CancellationToken,
    CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, GradScalerConfig,
    GradScalerOptimizerDecision, GradientMode, GradientReducer, GradientStore, Layout, LeafId,
    MemoryFormatReference, NativeGradScaler, SavedTensor, StreamId, TapeState, Tensor,
    TensorDescriptor, TensorError,
    autograd::breadth::{
        AUTOGRAD_CONSTRUCTS, AddAuxLossFunction, AutogradBreadthError, CUSTOM_FUNCTIONS,
        CheckpointCallable, CheckpointFunction, FunctionContext, HadaWeightFunction,
        HadaWeightTuckerFunction, HigherOrderPolicy, NativeAdam, NativeAdamW, NativeRmsprop,
        NativeSgd, OffloadCheckpointFunction, VectorQuantizeFunction,
    },
    generated_elementwise_or_runtime_operation_06::checkpoint_exact_native,
    generated_elementwise_or_runtime_operation_08::autograd_grad_exact_native,
    generated_elementwise_or_runtime_operation_14::{autocast_exact_native, detach_exact_native},
    generated_elementwise_or_runtime_operation_17::requires_grad_method_exact_native,
    generated_elementwise_or_runtime_operation_21::backward_method_with_context_exact_native,
    generated_elementwise_or_runtime_operation_22::cuda_amp_autocast_exact_native,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_storage_dtype_device_01::clone_with_context_exact_native,
    generated_tensor_creation_01::ones_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

const BREADTH_FIXTURE: &[u8] = include_bytes!("../fixtures/autograd/breadth-v1.json");
const BREADTH_FIXTURE_SHA256: &str =
    "d8d0c41d9873dcc376489403fcc3bf9a19428719ac49eaee9ecc6fd0dee99c1a";
const AUTOGRAD_CATALOG_PATH: &str = ".agents/specs/comfy-parity/catalogs/backend-autograd.csv";
const AUTOGRAD_CATALOG: &[u8] =
    include_bytes!("../../../.agents/specs/comfy-parity/catalogs/backend-autograd.csv");
const AUTOGRAD_CATALOG_SHA256: &str =
    "d51ff8465e2a161bef2093bbdb37f7547a6d6157d0fa1c4d6f0a30b8fd682670";
const QUANT_LINEAR_FIXTURE_PATH: &str =
    ".agents/specs/comfy-parity/fixtures/quant-linear-source-oracle.json";
const QUANT_LINEAR_FIXTURE: &[u8] =
    include_bytes!("../../../.agents/specs/comfy-parity/fixtures/quant-linear-source-oracle.json");
const QUANT_LINEAR_FIXTURE_SHA256: &str =
    "74acf934871befe3a87a91de6aea430a7ea9a16a821441bd716768dfb1919d0c";
const QUANT_LINEAR_CATALOG_ID: &str = "COMFY-AUTOGRAD-30043B9C2264";
const QUANT_LINEAR_SYMBOL: &str = "QuantLinearFunc";
const QUANT_LINEAR_EXECUTION_CASE: &str = "quant_linear_model_adapter";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BreadthFixture {
    schema_version: u32,
    owner_task_id: String,
    oracle: SourceOracle,
    catalog_cases: Vec<FixtureCase>,
    custom_functions: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceOracle {
    comfyui_version: String,
    comfyui_tree_sha256: String,
    source_files: BTreeMap<String, String>,
    development_only: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCase {
    id: String,
    symbol: String,
    execution_case: String,
    source_observations: Vec<SourceObservation>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
struct SourceObservation {
    case: String,
    expected: String,
    #[serde(default)]
    sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ObservationReceipt {
    case: &'static str,
    expected: &'static str,
    sha256: Option<&'static str>,
}

impl ObservationReceipt {
    const fn new(case: &'static str, expected: &'static str) -> Self {
        Self {
            case,
            expected,
            sha256: None,
        }
    }

    const fn with_sha256(case: &'static str, expected: &'static str, sha256: &'static str) -> Self {
        Self {
            case,
            expected,
            sha256: Some(sha256),
        }
    }

    fn owned(self) -> SourceObservation {
        SourceObservation {
            case: self.case.to_owned(),
            expected: self.expected.to_owned(),
            sha256: self.sha256.map(str::to_owned),
        }
    }
}

#[derive(Debug)]
struct CaseExecution {
    observations: BTreeSet<SourceObservation>,
    native_receipts: BTreeSet<String>,
}

impl CaseExecution {
    fn observations(receipts: &[ObservationReceipt]) -> Self {
        Self {
            observations: receipts
                .iter()
                .copied()
                .map(ObservationReceipt::owned)
                .collect(),
            native_receipts: BTreeSet::new(),
        }
    }

    fn quant_linear(receipt: ObservationReceipt, native_receipts: BTreeSet<String>) -> Self {
        Self {
            observations: BTreeSet::from([receipt.owned()]),
            native_receipts,
        }
    }
}

fn allowed_observations(execution_case: &str) -> Option<Vec<ObservationReceipt>> {
    let receipts: &[ObservationReceipt] = match execution_case {
        "tape_requires_grad" => &[ObservationReceipt::new("state", "checked_leaf_binding")],
        "vector_quantize" => &[ObservationReceipt::new(
            "forward_vjp",
            "nearest_code_and_two_gradient_slots",
        )],
        "scaler_step" => &[
            ObservationReceipt::new("finite_step", "optimizer_runs"),
            ObservationReceipt::new("nonfinite_step", "optimizer_skips"),
        ],
        "optimizer_adam" => &[ObservationReceipt::new(
            "state",
            "transactional_step_counter",
        )],
        "checkpoint_function" => &[
            ObservationReceipt::new("forward_backward", "no_grad_then_enabled_autocast_replay"),
            ObservationReceipt::new("arity", "two_metadata_none_slots"),
        ],
        "function_context_needs" => &[ObservationReceipt::new("state", "out_of_range_false")],
        "mode_enable" => &[ObservationReceipt::new(
            "nested_scope",
            "recording_enabled_then_restored",
        )],
        "grad_scaler" => &[
            ObservationReceipt::new("finite_cycle", "growth"),
            ObservationReceipt::new("nonfinite_cycle", "backoff"),
        ],
        QUANT_LINEAR_EXECUTION_CASE => &[ObservationReceipt::with_sha256(
            "delegated_fixture",
            QUANT_LINEAR_FIXTURE_PATH,
            QUANT_LINEAR_FIXTURE_SHA256,
        )],
        "hada_weight_tucker" => &[
            ObservationReceipt::new("forward_vjp", "seven_slots_scale_none"),
            ObservationReceipt::new("higher_order", "analytical"),
        ],
        "tape_backward" => &[ObservationReceipt::new(
            "reverse",
            "leaf_accumulation_and_terminal_release",
        )],
        "tape_grad" => &[ObservationReceipt::new(
            "reverse",
            "selected_leaf_return_without_publication",
        )],
        "optimizer_rmsprop" => &[ObservationReceipt::new(
            "state",
            "transactional_square_average",
        )],
        "mode_inference" => &[ObservationReceipt::new(
            "nested_scope",
            "recording_suppressed_then_restored",
        )],
        "tape_requires_grad_mutation" => &[ObservationReceipt::new(
            "mutation",
            "alias_and_checked_dtype",
        )],
        "detach_alias" => &[ObservationReceipt::new(
            "alias",
            "new_identity_shared_lineage_no_leaf",
        )],
        "add_aux_loss" => &[ObservationReceipt::new(
            "forward_vjp",
            "input_alias_and_loss_dtype_one",
        )],
        "scaler_scale" => &[ObservationReceipt::new("scale", "loss_times_current_scale")],
        "autocast_cuda_alias" => &[ObservationReceipt::new(
            "policy",
            "typed_legacy_namespace_projection",
        )],
        "gradient_store_lookup" => &[ObservationReceipt::new("lookup", "canonical_leaf_gradient")],
        "optimizer_adamw_functional" => {
            &[ObservationReceipt::new("equation", "canonical_adamw_step")]
        }
        "function_context_save" => &[ObservationReceipt::new(
            "mutation",
            "saved_version_rejection",
        )],
        "offload_checkpoint" => &[
            ObservationReceipt::new(
                "forward_backward",
                "callable_released_before_recompute_completion",
            ),
            ObservationReceipt::new("arity", "grad_x_and_forward_fn_none"),
        ],
        "optimizer_sgd" => &[ObservationReceipt::new("state", "transactional_step")],
        "hada_weight" => &[
            ObservationReceipt::new("forward_vjp", "five_slots_scale_none"),
            ObservationReceipt::new("higher_order", "analytical"),
        ],
        "gradient_store_zero" => &[ObservationReceipt::new(
            "zero",
            "zero_or_set_to_none_transaction",
        )],
        "scaler_update" => &[ObservationReceipt::new(
            "update",
            "growth_or_backoff_then_ready",
        )],
        "data_alias" => &[ObservationReceipt::new(
            "alias",
            "new_identity_shared_mutation_lineage",
        )],
        "autocast_policy" => &[ObservationReceipt::new(
            "policy",
            "typed_dtype_enabled_cache_scope",
        )],
        "function_context_mark" => &[ObservationReceipt::new(
            "output",
            "indices_gradient_slot_ignored",
        )],
        "scaler_unscale" => &[ObservationReceipt::new(
            "unscale",
            "atomic_and_nonfinite_detection",
        )],
        "function_context_saved" => &[ObservationReceipt::new(
            "lifetime",
            "validated_until_terminal_release",
        )],
        "optimizer_adamw" => &[ObservationReceipt::new(
            "state",
            "transactional_moments_and_step",
        )],
        "mode_no_grad" => &[ObservationReceipt::new(
            "nested_scope",
            "recording_suppressed_then_restored",
        )],
        "checkpoint_api" => &[ObservationReceipt::new(
            "policy",
            "single_checkpoint_execution_owner",
        )],
        "factory_requires_grad" => &[ObservationReceipt::new(
            "factory",
            "checked_floating_or_complex_leaf_registration",
        )],
        _ => return None,
    };
    Some(receipts.to_vec())
}

fn validate_case_observations(case: &FixtureCase) -> Result<(), Box<dyn Error>> {
    let allowed = allowed_observations(&case.execution_case).ok_or_else(|| {
        io::Error::other(format!(
            "unreferenced autograd execution case {}",
            case.execution_case
        ))
    })?;
    let actual = case
        .source_observations
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual.len() != case.source_observations.len() {
        return Err(
            io::Error::other(format!("duplicate autograd observation for {}", case.id)).into(),
        );
    }
    let expected = allowed
        .iter()
        .copied()
        .map(ObservationReceipt::owned)
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(io::Error::other(format!(
            "unknown, prose-only, missing, or unreferenced autograd observation for {}",
            case.id
        ))
        .into());
    }
    Ok(())
}

fn validate_execution_receipts(
    case: &FixtureCase,
    execution: &CaseExecution,
) -> Result<(), Box<dyn Error>> {
    let declared = case
        .source_observations
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if execution.observations != declared {
        return Err(io::Error::other(format!(
            "unexecuted or undeclared autograd observation for {}: declared {declared:?}, executed {:?}",
            case.id, execution.observations
        ))
        .into());
    }
    Ok(())
}

fn validate_custom_function_keys(value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    let object = value
        .as_object()
        .ok_or_else(|| io::Error::other("custom_functions must be an object"))?;
    let expected = BTreeSet::from([
        "add_aux_loss",
        "checkpoint_function",
        "hada_weight",
        "hada_weight_tucker",
        "offload_checkpoint",
        "vector_quantize",
    ]);
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(io::Error::other("custom function fixture keys changed").into());
    }
    let expected_fields: [(&str, &[&str]); 6] = [
        (
            "add_aux_loss",
            &[
                "aux_gradient_dtype",
                "aux_gradient_value",
                "backward_outputs",
                "forward_alias",
                "forward_arity",
                "higher_order",
            ],
        ),
        (
            "checkpoint_function",
            &[
                "autocast_replayed",
                "backward_fixture_outputs",
                "backward_metadata_slots",
                "forward_fixture_arity",
                "forward_mode",
                "higher_order",
                "recompute_mode",
                "shallow_views",
                "variadic_tensor_inputs",
            ],
        ),
        (
            "hada_weight",
            &[
                "backward_outputs",
                "forward_arity",
                "higher_order",
                "output",
                "terminal_gradient",
            ],
        ),
        (
            "hada_weight_tucker",
            &[
                "backward_outputs",
                "degenerate_output",
                "forward_arity",
                "higher_order",
                "terminal_gradient",
            ],
        ),
        (
            "offload_checkpoint",
            &[
                "backward_outputs",
                "clear_callable_before_recompute",
                "forward_arity",
                "higher_order",
                "terminal_gradient",
            ],
        ),
        (
            "vector_quantize",
            &[
                "backward_inputs",
                "backward_outputs",
                "codebook",
                "forward_arity",
                "forward_outputs",
                "higher_order",
                "indices",
                "input",
                "output",
            ],
        ),
    ];
    for (function, expected) in expected_fields {
        let fields = object
            .get(function)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| io::Error::other(format!("{function} fixture must be an object")))?;
        let actual = fields.keys().map(String::as_str).collect::<BTreeSet<_>>();
        let expected = expected.iter().copied().collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(io::Error::other(format!(
                "{function} fixture has missing or unknown fields"
            ))
            .into());
        }
    }
    Ok(())
}

fn parse_and_validate_breadth_fixture(bytes: &[u8]) -> Result<BreadthFixture, Box<dyn Error>> {
    let fixture: BreadthFixture = serde_json::from_slice(bytes)?;
    if fixture.schema_version != 1
        || fixture.owner_task_id != "comfy-parity-native-autograd-breadth"
    {
        return Err(io::Error::other("autograd breadth fixture schema or owner changed").into());
    }
    validate_custom_function_keys(&fixture.custom_functions)?;
    if fixture.catalog_cases.len() != AUTOGRAD_CONSTRUCTS.len() {
        return Err(io::Error::other("autograd breadth fixture row count changed").into());
    }
    let contracts = AUTOGRAD_CONSTRUCTS
        .iter()
        .map(|contract| (contract.id, contract.symbol))
        .collect::<BTreeSet<_>>();
    let rows = fixture
        .catalog_cases
        .iter()
        .map(|case| (case.id.as_str(), case.symbol.as_str()))
        .collect::<BTreeSet<_>>();
    let execution_cases = fixture
        .catalog_cases
        .iter()
        .map(|case| case.execution_case.as_str())
        .collect::<BTreeSet<_>>();
    if contracts != rows
        || rows.len() != fixture.catalog_cases.len()
        || execution_cases.len() != fixture.catalog_cases.len()
    {
        return Err(io::Error::other(
            "autograd fixture has missing, duplicate, or unreferenced catalog rows",
        )
        .into());
    }
    for case in &fixture.catalog_cases {
        validate_case_observations(case)?;
    }
    Ok(fixture)
}

#[derive(Deserialize)]
struct QuantLinearOracleInputs {
    input_shape: Vec<u64>,
    input: Vec<f32>,
    weight_shape: Vec<u64>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    output_gradient_shape: Vec<u64>,
    output_gradient: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum QuantLinearOracleScale {
    Default,
    Explicit { value: f32 },
    Recalculate,
}

impl QuantLinearOracleScale {
    fn as_native(&self) -> QuantLinearScale {
        match self {
            Self::Default => QuantLinearScale::Default,
            Self::Explicit { value } => QuantLinearScale::Explicit(*value),
            Self::Recalculate => QuantLinearScale::Recalculate,
        }
    }

    fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }
}

#[derive(Deserialize)]
struct QuantLinearOracleCase {
    id: String,
    source_layout: Option<String>,
    input_scale: QuantLinearOracleScale,
    weight_layout: Option<String>,
    weight_scale: QuantLinearOracleScale,
    compute_dtype: String,
    fp8_backward: bool,
    weight_requires_grad: bool,
    weight_runtime_type: String,
    output: Vec<f32>,
    input_gradient: Vec<f32>,
    weight_gradient: Option<Vec<f32>>,
    bias_gradient: Vec<f32>,
    output_dtype: String,
    gradient_dtypes: Vec<Option<String>>,
}

#[derive(Deserialize)]
struct QuantLinearOracle {
    schema_version: u16,
    owner_task_id: String,
    fixture_inputs: QuantLinearOracleInputs,
    execution_cases: Vec<QuantLinearOracleCase>,
}

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl std::ops::Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn backend() -> Result<TestBackend, TensorError> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    Ok(TestBackend { backend, authority })
}

fn context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(32 * 1024 * 1024)?,
        cancellation,
    ))
}

fn tensor(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        DeviceId::CPU,
        &context(backend, cancellation)?,
    )?)
}

fn values(
    backend: &TestBackend,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &context(backend, cancellation)?,
    )?)
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "{actual} != {expected}"
        );
    }
}

struct IdentityRule;

impl BackwardRule for IdentityRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, sim_tensor::AutogradError> {
        if cancellation.is_cancelled() {
            return Err(sim_tensor::AutogradError::Cancelled);
        }
        Ok(vec![output_gradients.first().cloned().flatten()])
    }
}

struct AddReducer;

impl GradientReducer for AddReducer {
    fn add(
        &self,
        left: Tensor,
        right: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, sim_tensor::AutogradError> {
        if cancellation.is_cancelled() {
            return Err(sim_tensor::AutogradError::Cancelled);
        }
        let backend = backend()?;
        let read = |tensor: &Tensor| -> Result<Vec<f32>, sim_tensor::AutogradError> {
            let count = usize::try_from(tensor.descriptor().element_count()?)
                .map_err(|_| TensorError::ShapeOverflow)?;
            let mut output = Vec::with_capacity(count);
            for index in 0..count {
                let bytes = tensor.linear_element_bytes(
                    u64::try_from(index).map_err(|_| TensorError::ShapeOverflow)?,
                )?;
                let bytes = <[u8; 4]>::try_from(bytes).map_err(|_| TensorError::DTypeMismatch {
                    expected: DType::F32,
                    actual: tensor.descriptor().dtype(),
                })?;
                output.push(f32::from_ne_bytes(bytes));
            }
            Ok(output)
        };
        let left_values = read(&left)?;
        let right_values = read(&right)?;
        let sums = left_values
            .iter()
            .zip(right_values)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(
            left.descriptor().shape().to_vec(),
            left.descriptor().dtype(),
            left.descriptor().device(),
            left.descriptor().stream(),
        )?;
        Ok(backend
            .upload_f32(descriptor, &sums, &context(&backend, cancellation)?)?
            .0)
    }
}

struct SquareCallable;

impl CheckpointCallable for SquareCallable {
    fn forward(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::NoGrad);
        assert!(!autocast.enabled());
        let input = inputs.first().ok_or_else(|| {
            AutogradBreadthError::InvalidInput("checkpoint input is missing".to_owned())
        })?;
        let source = input
            .host_storage_bytes()?
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .map(|value| value * value)
            .collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            execution.stream,
        )?;
        Ok(vec![backend.upload_f32(descriptor, &source, execution)?.0])
    }

    fn recompute_vjp(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        _parameters: &[Tensor],
        output_gradients: &[Option<Tensor>],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::Enabled);
        assert!(!autocast.enabled());
        let input = inputs.first().ok_or_else(|| {
            AutogradBreadthError::InvalidInput("checkpoint input is missing".to_owned())
        })?;
        let gradient = output_gradients
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                AutogradBreadthError::InvalidInput("checkpoint gradient is missing".to_owned())
            })?;
        let input_values = input
            .host_storage_bytes()?
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        let gradient_values = gradient
            .host_storage_bytes()?
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        let output = input_values
            .zip(gradient_values)
            .map(|(input, gradient)| 2.0 * input * gradient)
            .collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            execution.stream,
        )?;
        Ok(vec![Some(
            backend.upload_f32(descriptor, &output, execution)?.0,
        )])
    }
}

struct EffectCountingSquareCallable {
    forward_calls: Arc<AtomicUsize>,
    recompute_calls: Arc<AtomicUsize>,
}

impl CheckpointCallable for EffectCountingSquareCallable {
    fn forward(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        SquareCallable.forward(backend, inputs, mode, autocast, execution)
    }

    fn recompute_vjp(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        parameters: &[Tensor],
        output_gradients: &[Option<Tensor>],
        mode: GradientMode,
        autocast: &AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        self.recompute_calls.fetch_add(1, Ordering::SeqCst);
        SquareCallable.recompute_vjp(
            backend,
            inputs,
            parameters,
            output_gradients,
            mode,
            autocast,
            execution,
        )
    }
}

fn probe_tape(case: &str, backend: &TestBackend) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let input = tensor(backend, &[], &[2.0], DType::F32, &cancellation)?;
    let leaf = LeafId::new(format!("leaf-{case}"))?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    if case.contains("requires_grad") || case == "factory_requires_grad" {
        if case == "factory_requires_grad" {
            let output = ones_with_context_exact_native(
                backend,
                &[1],
                DType::F32,
                Layout::Strided,
                DeviceId::CPU,
                true,
                Some((&mut tape, leaf)),
                &context(backend, &cancellation)?,
            )?;
            assert!(tape.requires_grad(&output));
            let integer = tensor(backend, &[1], &[1.0], DType::I64, &cancellation)?;
            assert!(
                requires_grad_method_exact_native(
                    &mut tape,
                    &integer,
                    Some(LeafId::new("invalid-factory-leaf")?),
                    true,
                    &cancellation,
                )
                .is_err()
            );
        } else {
            let alias = requires_grad_method_exact_native(
                &mut tape,
                &input,
                Some(leaf),
                true,
                &cancellation,
            )?;
            assert!(tape.requires_grad(&input));
            assert_eq!(alias.tensor_id(), input.tensor_id());
            if case == "tape_requires_grad_mutation" {
                let integer = tensor(backend, &[1], &[1.0], DType::I64, &cancellation)?;
                assert!(
                    requires_grad_method_exact_native(
                        &mut tape,
                        &integer,
                        Some(LeafId::new("invalid-integral-leaf")?),
                        true,
                        &cancellation,
                    )
                    .is_err()
                );
                assert!(!tape.requires_grad(&integer));
            }
        }
        return Ok(());
    }
    if case.starts_with("mode_") {
        let mode = match case {
            "mode_enable" => GradientMode::Enabled,
            "mode_no_grad" => GradientMode::NoGrad,
            "mode_inference" => GradientMode::Inference,
            _ => return Err("unknown mode fixture".into()),
        };
        tape.with_mode(mode, &cancellation, |tape| {
            assert_eq!(tape.mode(), mode);
            Ok(())
        })?;
        assert_eq!(tape.mode(), GradientMode::Enabled);
        return Ok(());
    }
    let outputs = tape
        .record(
            vec![AutogradInput::Leaf(leaf.clone())],
            1,
            vec![],
            Arc::new(IdentityRule),
        )?
        .ok_or("tape did not record")?;
    let gradient = tensor(backend, &[], &[3.0], DType::F32, &cancellation)?;
    if case == "tape_grad" {
        let gradients = autograd_grad_exact_native(
            &mut tape,
            vec![(outputs[0], gradient)],
            std::slice::from_ref(&leaf),
            &AddReducer,
            false,
            &cancellation,
        )?;
        close(
            &values(
                backend,
                gradients[0].as_ref().ok_or("missing selected gradient")?,
                &cancellation,
            )?,
            &[3.0],
        );
        assert_eq!(tape.state(), &TapeState::Completed);
    } else {
        let mut store = GradientStore::default();
        backward_method_with_context_exact_native(
            backend,
            &mut tape,
            outputs[0],
            &input,
            Some(gradient),
            None,
            &AddReducer,
            &mut store,
            false,
            false,
            &context(backend, &cancellation)?,
        )?;
        close(
            &values(
                backend,
                store.gradient(&leaf).ok_or("missing published gradient")?,
                &cancellation,
            )?,
            &[3.0],
        );
        assert_eq!(tape.state(), &TapeState::Completed);
        assert_eq!(tape.retained_node_count(), 0);
    }
    Ok(())
}

fn probe_context(case: &str, backend: &TestBackend) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let mut input = tensor(backend, &[1], &[2.0], DType::F32, &cancellation)?;
    let mut function = FunctionContext::new(vec![true, false]);
    function.save_for_backward(&[&input])?;
    if case == "function_context_mark" {
        function.mark_non_differentiable(1)?;
        assert!(function.is_non_differentiable(1));
    } else if case == "function_context_needs" {
        assert!(function.needs_input_grad(0));
        assert!(!function.needs_input_grad(2));
    } else if case == "function_context_save" {
        input
            .write()?
            .bytes_mut()?
            .copy_from_slice(&3.0_f32.to_ne_bytes());
        assert!(matches!(
            function.saved_tensors(),
            Err(AutogradBreadthError::Autograd(
                AutogradError::SavedTensorModified { .. }
            ))
        ));
    } else {
        assert_eq!(function.saved_tensors()?.len(), 1);
    }
    function.release();
    assert_eq!(function.retained_tensor_count(), 0);
    assert!(matches!(
        function.saved_tensors(),
        Err(AutogradBreadthError::ReleasedContext)
    ));
    Ok(())
}

fn probe_scaler(case: &str, backend: &TestBackend) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let loss = tensor(backend, &[], &[2.0], DType::F32, &cancellation)?;
    if case == "scaler_scale" {
        let scaler = NativeGradScaler::new(GradScalerConfig {
            initial_scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            enabled: true,
        })?;
        close(
            &values(
                backend,
                &scaler.scale_loss_exact_native(&loss, &cancellation)?,
                &cancellation,
            )?,
            &[16.0],
        );
        return Ok(());
    }
    for nonfinite in [false, true] {
        let mut scaler = NativeGradScaler::new(GradScalerConfig {
            initial_scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1,
            enabled: true,
        })?;
        let mut gradients = vec![tensor(
            backend,
            &[1],
            &[if nonfinite { f32::INFINITY } else { 4.0 }],
            DType::F32,
            &cancellation,
        )?];
        let found = scaler.unscale_gradients_exact_native(&mut gradients, &cancellation)?;
        assert_eq!(found, nonfinite);
        if nonfinite {
            assert!(values(backend, &gradients[0], &cancellation)?[0].is_infinite());
        } else {
            close(&values(backend, &gradients[0], &cancellation)?, &[0.5]);
        }
        assert_eq!(
            scaler.optimizer_step_decision_exact_native(&cancellation)?,
            if nonfinite {
                GradScalerOptimizerDecision::SkipNonFinite
            } else {
                GradScalerOptimizerDecision::Run
            }
        );
        scaler.update_exact_native(&cancellation)?;
        assert_eq!(scaler.scale(), if nonfinite { 4.0 } else { 16.0 });
    }
    Ok(())
}

fn probe_optimizer(case: &str, backend: &TestBackend) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let execution = context(backend, &cancellation)?;
    let parameter = tensor(backend, &[1], &[1.0], DType::F32, &cancellation)?;
    let gradient = tensor(backend, &[1], &[0.5], DType::F32, &cancellation)?;
    let mut parameters = vec![parameter.clone()];
    let expected_parameter = match case {
        "optimizer_sgd" => {
            let mut optimizer =
                NativeSgd::new_exact_native(1, 0.1, 0.0, 0.0, 0.0, false, false, &cancellation)?;
            optimizer.step_with_context_exact_native(
                backend,
                &mut parameters,
                std::slice::from_ref(&gradient),
                &execution,
            )?;
            0.95
        }
        "optimizer_adam" => {
            let mut optimizer = NativeAdam::new_with_context_exact_native(
                backend,
                std::slice::from_ref(&parameter),
                0.1,
                &execution,
            )?;
            optimizer.step_with_context_exact_native(
                backend,
                &mut parameters,
                std::slice::from_ref(&gradient),
                &execution,
            )?;
            assert_eq!(optimizer.steps(), [1]);
            0.9
        }
        "optimizer_rmsprop" => {
            let mut optimizer = NativeRmsprop::new_with_context_exact_native(
                backend,
                std::slice::from_ref(&parameter),
                0.01,
                0.99,
                1.0e-8,
                0.0,
                0.0,
                false,
                false,
                &execution,
            )?;
            optimizer.step_with_context_exact_native(
                backend,
                &mut parameters,
                std::slice::from_ref(&gradient),
                &execution,
            )?;
            assert_eq!(optimizer.steps(), [1]);
            close(
                &values(backend, &optimizer.square_averages()[0], &cancellation)?,
                &[0.0025],
            );
            0.9
        }
        "optimizer_adamw" | "optimizer_adamw_functional" => {
            let mut optimizer = NativeAdamW::new_with_context_exact_native(
                backend,
                std::slice::from_ref(&parameter),
                0.1,
                0.9,
                0.999,
                1.0e-8,
                0.01,
                false,
                false,
                &execution,
            )?;
            optimizer.step_with_context_exact_native(
                backend,
                &mut parameters,
                std::slice::from_ref(&gradient),
                &execution,
            )?;
            assert_eq!(optimizer.steps(), [1]);
            0.899
        }
        _ => return Err("unknown optimizer fixture".into()),
    };
    close(
        &values(backend, &parameters[0], &cancellation)?,
        &[expected_parameter],
    );
    Ok(())
}

fn probe_alias_or_store(case: &str, backend: &TestBackend) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let input = tensor(backend, &[1], &[2.0], DType::F32, &cancellation)?;
    if case == "detach_alias" {
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_requires_grad(
            &input,
            Some(LeafId::new("detach-source")?),
            true,
            &cancellation,
        )?;
        let detached = detach_exact_native(&input, &cancellation)?;
        assert_ne!(detached.tensor_id(), input.tensor_id());
        assert_eq!(
            detached.mutation_witness().actual_epoch(),
            input.mutation_witness().actual_epoch()
        );
        assert!(!tape.requires_grad(&detached));
        return Ok(());
    }
    if case == "data_alias" {
        let alias = input.data_alias()?;
        assert_ne!(alias.tensor_id(), input.tensor_id());
        assert_eq!(
            alias.mutation_witness().actual_epoch(),
            input.mutation_witness().actual_epoch()
        );
        return Ok(());
    }
    let leaf = LeafId::new("stored")?;
    let mut store = GradientStore::default();
    store.publish(
        std::collections::HashMap::from([(leaf.clone(), input.clone())]),
        &cancellation,
    )?;
    if case == "gradient_store_zero" {
        store.zero_grad(backend, &context(backend, &cancellation)?, false)?;
        close(
            &values(
                backend,
                store.gradient(&leaf).ok_or("missing zero gradient")?,
                &cancellation,
            )?,
            &[0.0],
        );
        store.publish(
            std::collections::HashMap::from([(leaf, input)]),
            &cancellation,
        )?;
        store.zero_grad(backend, &context(backend, &cancellation)?, true)?;
        assert!(store.is_empty());
    } else {
        close(
            &values(
                backend,
                store.gradient(&leaf).ok_or("missing stored gradient")?,
                &cancellation,
            )?,
            &[2.0],
        );
    }
    Ok(())
}

fn probe_autocast(case: &str) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let policy = if case == "autocast_cuda_alias" {
        cuda_amp_autocast_exact_native(true, Some(DType::F16), Some(false), &cancellation)?
    } else {
        autocast_exact_native(
            DeviceKind::Cpu,
            Some(DType::Bf16),
            true,
            Some(true),
            &cancellation,
        )?
    };
    assert!(policy.enabled());
    if case == "autocast_cuda_alias" {
        assert_eq!(policy.dtype(), DType::F16);
        assert!(!policy.cache_enabled());
    } else {
        assert_eq!(policy.dtype(), DType::Bf16);
        assert!(policy.cache_enabled());
    }
    Ok(())
}

fn probe_custom(
    case: &str,
    backend: &TestBackend,
) -> Result<Vec<ObservationReceipt>, Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let execution = context(backend, &cancellation)?;
    if case == "vector_quantize" {
        let input = tensor(
            backend,
            &[3, 2],
            &[0.1, 0.2, 2.8, 3.2, 0.8, 1.1],
            DType::F32,
            &cancellation,
        )?;
        let codebook = tensor(
            backend,
            &[3, 2],
            &[0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
            DType::F32,
            &cancellation,
        )?;
        let incoming = tensor(
            backend,
            &[3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::F32,
            &cancellation,
        )?;
        let second_seed = tensor(backend, &[3, 2], &[1.0; 6], DType::F32, &cancellation)?;
        let input_leaf = LeafId::new("aggregate-vq-input")?;
        let codebook_leaf = LeafId::new("aggregate-vq-codebook")?;
        let incoming_leaf = LeafId::new("aggregate-vq-incoming")?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_requires_grad(&input, Some(input_leaf.clone()), true, &cancellation)?;
        tape.set_requires_grad(&codebook, Some(codebook_leaf.clone()), true, &cancellation)?;
        tape.set_requires_grad(&incoming, Some(incoming_leaf.clone()), true, &cancellation)?;
        let (_function, output, _indices, slot) = VectorQuantizeFunction::forward_recorded(
            backend,
            &mut tape,
            &input,
            &codebook,
            [true, true],
            &execution,
        )?;
        close(
            &values(backend, &output, &cancellation)?,
            &[0.0, 0.0, 3.0, 3.0, 1.0, 1.0],
        );
        let first = tape.reverse_with_context(
            vec![(slot.ok_or("vector_quantize was not recorded")?, incoming)],
            &AddReducer,
            false,
            true,
            backend,
            &execution,
        )?;
        close(
            &values(
                backend,
                first
                    .get(&input_leaf)
                    .ok_or("missing vector_quantize input gradient")?,
                &cancellation,
            )?,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let codebook_gradient = first
            .get(&codebook_leaf)
            .ok_or("missing vector_quantize codebook gradient")?;
        close(
            &values(backend, codebook_gradient, &cancellation)?,
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0],
        );
        let gradient_slot = tape
            .output_slot(codebook_gradient)
            .ok_or("vector_quantize gradient lacks create_graph provenance")?;
        let second = tape.reverse_with_context(
            vec![(gradient_slot, second_seed)],
            &AddReducer,
            false,
            false,
            backend,
            &execution,
        )?;
        close(
            &values(
                backend,
                second
                    .get(&incoming_leaf)
                    .ok_or("missing vector_quantize second backward provenance")?,
                &cancellation,
            )?,
            &[1.0; 6],
        );
        return Ok(vec![ObservationReceipt::new(
            "forward_vjp",
            "nearest_code_and_two_gradient_slots",
        )]);
    }
    if case == "add_aux_loss" {
        let input = tensor(backend, &[1], &[2.0], DType::F32, &cancellation)?;
        let loss = tensor(backend, &[1], &[7.0], DType::F32, &cancellation)?;
        let incoming = tensor(backend, &[1], &[1.0], DType::F32, &cancellation)?;
        let second_seed = tensor(backend, &[1], &[1.0], DType::F32, &cancellation)?;
        let input_leaf = LeafId::new("aggregate-add-aux-input")?;
        let incoming_leaf = LeafId::new("aggregate-add-aux-incoming")?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_requires_grad(&input, Some(input_leaf.clone()), true, &cancellation)?;
        tape.set_requires_grad(&incoming, Some(incoming_leaf.clone()), true, &cancellation)?;
        let (_function, output, slot) = AddAuxLossFunction::forward_recorded(
            backend, &mut tape, &input, &loss, true, &execution,
        )?;
        assert_eq!(input.storage_id(), output.storage_id());
        let first = tape.reverse_with_context(
            vec![(slot.ok_or("AddAuxLoss was not recorded")?, incoming)],
            &AddReducer,
            false,
            true,
            backend,
            &execution,
        )?;
        let input_gradient = first
            .get(&input_leaf)
            .ok_or("missing AddAuxLoss input gradient")?;
        close(&values(backend, input_gradient, &cancellation)?, &[1.0]);
        let gradient_slot = tape
            .output_slot(input_gradient)
            .ok_or("AddAuxLoss gradient lacks create_graph provenance")?;
        let second = tape.reverse_with_context(
            vec![(gradient_slot, second_seed)],
            &AddReducer,
            false,
            false,
            backend,
            &execution,
        )?;
        close(
            &values(
                backend,
                second
                    .get(&incoming_leaf)
                    .ok_or("missing AddAuxLoss second backward provenance")?,
                &cancellation,
            )?,
            &[1.0],
        );
        return Ok(vec![ObservationReceipt::new(
            "forward_vjp",
            "input_alias_and_loss_dtype_one",
        )]);
    }
    if case == "hada_weight" {
        let factors = [
            tensor(backend, &[1, 1], &[2.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[3.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[5.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[7.0], DType::F32, &cancellation)?,
        ];
        let scale = tensor(backend, &[1, 1], &[11.0], DType::F32, &cancellation)?;
        let seed = tensor(backend, &[1, 1], &[1.0], DType::F32, &cancellation)?;
        let leaves = (0..5)
            .map(|index| LeafId::new(format!("aggregate-hada-{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        for (factor, leaf) in factors.iter().chain(std::iter::once(&scale)).zip(&leaves) {
            tape.set_requires_grad(factor, Some(leaf.clone()), true, &cancellation)?;
        }
        let (_function, output, slot) = HadaWeightFunction::forward_recorded(
            backend,
            &mut tape,
            [&factors[0], &factors[1], &factors[2], &factors[3]],
            &scale,
            [true; 5],
            &execution,
        )?;
        close(&values(backend, &output, &cancellation)?, &[2310.0]);
        let first = tape.reverse_with_context(
            vec![(slot.ok_or("HadaWeight was not recorded")?, seed.clone())],
            &AddReducer,
            false,
            true,
            backend,
            &execution,
        )?;
        let first_gradient = first
            .get(&leaves[0])
            .ok_or("missing HadaWeight first gradient")?;
        close(&values(backend, first_gradient, &cancellation)?, &[1155.0]);
        let gradient_slot = tape
            .output_slot(first_gradient)
            .ok_or("HadaWeight gradient lacks create_graph provenance")?;
        let second = tape.reverse_with_context(
            vec![(gradient_slot, seed)],
            &AddReducer,
            false,
            false,
            backend,
            &execution,
        )?;
        close(
            &values(
                backend,
                second
                    .get(&leaves[1])
                    .ok_or("missing HadaWeight mixed second derivative")?,
                &cancellation,
            )?,
            &[385.0],
        );
        return Ok(vec![
            ObservationReceipt::new("forward_vjp", "five_slots_scale_none"),
            ObservationReceipt::new("higher_order", "analytical"),
        ]);
    }
    if case == "hada_weight_tucker" {
        let tensors = [
            tensor(backend, &[1, 1], &[2.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[3.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[4.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[5.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[6.0], DType::F32, &cancellation)?,
            tensor(backend, &[1, 1], &[7.0], DType::F32, &cancellation)?,
        ];
        let scale = tensor(backend, &[1, 1], &[11.0], DType::F32, &cancellation)?;
        let seed = tensor(backend, &[1, 1], &[1.0], DType::F32, &cancellation)?;
        let leaves = (0..7)
            .map(|index| LeafId::new(format!("aggregate-tucker-{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        for (tensor, leaf) in tensors.iter().chain(std::iter::once(&scale)).zip(&leaves) {
            tape.set_requires_grad(tensor, Some(leaf.clone()), true, &cancellation)?;
        }
        let (_function, output, slot) = HadaWeightTuckerFunction::forward_recorded(
            backend,
            &mut tape,
            [
                &tensors[0],
                &tensors[1],
                &tensors[2],
                &tensors[3],
                &tensors[4],
                &tensors[5],
            ],
            &scale,
            [true; 7],
            &execution,
        )?;
        close(&values(backend, &output, &cancellation)?, &[55440.0]);
        let first = tape.reverse_with_context(
            vec![(
                slot.ok_or("HadaWeightTucker was not recorded")?,
                seed.clone(),
            )],
            &AddReducer,
            false,
            true,
            backend,
            &execution,
        )?;
        let first_gradient = first
            .get(&leaves[0])
            .ok_or("missing HadaWeightTucker first gradient")?;
        close(&values(backend, first_gradient, &cancellation)?, &[27720.0]);
        let gradient_slot = tape
            .output_slot(first_gradient)
            .ok_or("HadaWeightTucker gradient lacks create_graph provenance")?;
        let second = tape.reverse_with_context(
            vec![(gradient_slot, seed)],
            &AddReducer,
            false,
            false,
            backend,
            &execution,
        )?;
        close(
            &values(
                backend,
                second
                    .get(&leaves[1])
                    .ok_or("missing HadaWeightTucker mixed second derivative")?,
                &cancellation,
            )?,
            &[9240.0],
        );
        return Ok(vec![
            ObservationReceipt::new("forward_vjp", "seven_slots_scale_none"),
            ObservationReceipt::new("higher_order", "analytical"),
        ]);
    }
    if matches!(
        case,
        "checkpoint_function" | "offload_checkpoint" | "checkpoint_api"
    ) {
        let input = tensor(backend, &[2], &[2.0, 3.0], DType::F32, &cancellation)?;
        if case == "checkpoint_api" {
            let checkpoint = checkpoint_exact_native(
                std::slice::from_ref(&input),
                true,
                &cancellation,
                |inputs, mode, _| {
                    assert_eq!(mode, GradientMode::NoGrad);
                    Ok(inputs.to_vec())
                },
            )?;
            assert_eq!(checkpoint.saved_input_count(), 1);
            assert_eq!(checkpoint.forward_mode(), GradientMode::NoGrad);
            assert_eq!(checkpoint.recompute_mode(), GradientMode::Enabled);
            let recomputed =
                checkpoint.recompute_exact_native(&cancellation, |inputs, mode, _| {
                    assert_eq!(mode, GradientMode::Enabled);
                    Ok(inputs.to_vec())
                })?;
            close(
                &values(backend, &recomputed[0], &cancellation)?,
                &[2.0, 3.0],
            );
            return Ok(vec![ObservationReceipt::new(
                "policy",
                "single_checkpoint_execution_owner",
            )]);
        }
        let gradient = tensor(backend, &[2], &[1.0, 2.0], DType::F32, &cancellation)?;
        let autocast = AutocastPolicy::new(false, DType::F32, true)?;
        let forward_calls = Arc::new(AtomicUsize::new(0));
        let recompute_calls = Arc::new(AtomicUsize::new(0));
        let rejection_callable = Arc::new(EffectCountingSquareCallable {
            forward_calls: forward_calls.clone(),
            recompute_calls: recompute_calls.clone(),
        });
        if case == "offload_checkpoint" {
            let (function, _) = OffloadCheckpointFunction::forward(
                backend,
                rejection_callable,
                &input,
                true,
                autocast,
                &execution,
            )?;
            assert_eq!(forward_calls.load(Ordering::SeqCst), 1);
            assert_eq!(recompute_calls.load(Ordering::SeqCst), 0);
            assert!(matches!(
                function.backward_with_options(backend, Some(gradient.clone()), true, &execution,),
                Err(AutogradBreadthError::HigherOrderUnavailable {
                    symbol: "OffloadCheckpointFunction",
                    policy: HigherOrderPolicy::FirstOrderOnly,
                })
            ));
        } else {
            let (function, _) = CheckpointFunction::forward(
                backend,
                rejection_callable,
                std::slice::from_ref(&input),
                &[],
                vec![true],
                autocast,
                &execution,
            )?;
            assert_eq!(forward_calls.load(Ordering::SeqCst), 1);
            assert_eq!(recompute_calls.load(Ordering::SeqCst), 0);
            assert!(matches!(
                function.backward_with_options(
                    backend,
                    &[Some(gradient.clone())],
                    true,
                    &execution,
                ),
                Err(AutogradBreadthError::HigherOrderUnavailable {
                    symbol: "CheckpointFunction",
                    policy: HigherOrderPolicy::FirstOrderOnly,
                })
            ));
        }
        assert_eq!(forward_calls.load(Ordering::SeqCst), 1);
        assert_eq!(recompute_calls.load(Ordering::SeqCst), 0);

        let callable = Arc::new(SquareCallable);
        if case == "offload_checkpoint" {
            let (function, _) = OffloadCheckpointFunction::forward(
                backend,
                callable.clone(),
                &input,
                true,
                autocast,
                &execution,
            )?;
            assert_eq!(Arc::strong_count(&callable), 2);
            let gradients = function.backward(backend, Some(gradient), &execution)?;
            assert!(gradients[0].is_some() && gradients[1].is_none());
            assert_eq!(Arc::strong_count(&callable), 1);
            return Ok(vec![
                ObservationReceipt::new(
                    "forward_backward",
                    "callable_released_before_recompute_completion",
                ),
                ObservationReceipt::new("arity", "grad_x_and_forward_fn_none"),
            ]);
        } else {
            let (function, _) = CheckpointFunction::forward(
                backend,
                callable.clone(),
                std::slice::from_ref(&input),
                &[],
                vec![true],
                autocast,
                &execution,
            )?;
            assert_eq!(Arc::strong_count(&callable), 2);
            let gradients =
                function.backward_source_arity(backend, &[Some(gradient)], &execution)?;
            assert_eq!(gradients.len(), 3);
            assert!(gradients[0].is_none() && gradients[1].is_none());
            assert_eq!(Arc::strong_count(&callable), 1);
            return Ok(vec![
                ObservationReceipt::new("forward_backward", "no_grad_then_enabled_autocast_replay"),
                ObservationReceipt::new("arity", "two_metadata_none_slots"),
            ]);
        }
    }
    Err("unknown custom-function fixture".into())
}

fn invalid_quant_linear_delegation(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::other(message.into()).into()
}

fn expected_quant_linear_case_ids() -> BTreeSet<&'static str> {
    BTreeSet::from([
        "fp8-e4m3-explicit-scale",
        "fp8-e4m3-fp8-backward",
        "fp8-e4m3-ordinary",
        "fp8-e4m3-recalculated-scale",
        "fp8-e5m2-fp8-backward",
        "fp8-e5m2-ordinary",
        "mxfp8-fp8-backward",
        "mxfp8-ordinary",
        "nvfp4-fp8-backward",
        "nvfp4-ordinary",
        "quantized-weight-fp8-e4m3-fp8-backward",
        "quantized-weight-fp8-e4m3-ordinary",
        "quantized-weight-fp8-e5m2-fp8-backward",
        "quantized-weight-fp8-e5m2-ordinary",
        "quantized-weight-mxfp8-fp8-backward",
        "quantized-weight-mxfp8-ordinary",
        "quantized-weight-nvfp4-fp8-backward",
        "quantized-weight-nvfp4-ordinary",
        "unquantized-bf16-ordinary",
        "unquantized-f16-ordinary",
        "unquantized-f32-fp8-backward",
        "unquantized-f32-ordinary",
    ])
}

fn validate_quant_linear_oracle_structure(
    oracle: &QuantLinearOracle,
) -> Result<(), Box<dyn Error>> {
    if oracle.schema_version != 4
        || oracle.owner_task_id != "comfy-parity-quantized-autograd-adapter"
    {
        return Err(invalid_quant_linear_delegation(
            "QuantLinear delegated fixture schema or owner changed",
        ));
    }
    let actual_case_ids = oracle
        .execution_cases
        .iter()
        .map(|execution_case| execution_case.id.as_str())
        .collect::<BTreeSet<_>>();
    let expected_case_ids = expected_quant_linear_case_ids();
    if oracle.execution_cases.len() != expected_case_ids.len()
        || actual_case_ids != expected_case_ids
    {
        return Err(invalid_quant_linear_delegation(
            "QuantLinear delegated fixture case closure changed",
        ));
    }
    Ok(())
}

fn load_quant_linear_oracle(
    case: &FixtureCase,
    fixture: &[u8],
) -> Result<QuantLinearOracle, Box<dyn Error>> {
    if case.id != QUANT_LINEAR_CATALOG_ID
        || case.symbol != QUANT_LINEAR_SYMBOL
        || case.execution_case != QUANT_LINEAR_EXECUTION_CASE
    {
        return Err(invalid_quant_linear_delegation(
            "QuantLinear delegated fixture is attached to an unexpected catalog contract",
        ));
    }
    let expected_observations = vec![
        ObservationReceipt::with_sha256(
            "delegated_fixture",
            QUANT_LINEAR_FIXTURE_PATH,
            QUANT_LINEAR_FIXTURE_SHA256,
        )
        .owned(),
    ];
    if case.source_observations != expected_observations {
        return Err(invalid_quant_linear_delegation(
            "QuantLinear catalog contract does not name exactly the canonical delegated fixture",
        ));
    }
    if format!("{:x}", Sha256::digest(fixture)) != QUANT_LINEAR_FIXTURE_SHA256 {
        return Err(invalid_quant_linear_delegation(
            "QuantLinear delegated fixture digest is stale",
        ));
    }
    let oracle: QuantLinearOracle = serde_json::from_slice(fixture)?;
    validate_quant_linear_oracle_structure(&oracle)?;
    Ok(oracle)
}

fn quant_linear_oracle_dtype(name: &str) -> Result<DType, Box<dyn Error>> {
    match name {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::Bf16),
        "f32" => Ok(DType::F32),
        _ => Err(invalid_quant_linear_delegation(format!(
            "unsupported QuantLinear oracle compute dtype {name}"
        ))),
    }
}

fn close_quant_linear_oracle_case(case_id: &str, field: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "QuantLinear oracle case {case_id} {field} length"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1.0e-5,
            "QuantLinear oracle case {case_id} {field}[{index}]: expected {expected}, got {actual}"
        );
    }
}

fn probe_quant_linear(
    case: &FixtureCase,
    backend: &TestBackend,
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let oracle = load_quant_linear_oracle(case, QUANT_LINEAR_FIXTURE)?;
    let cancellation = CancellationToken::default();
    let mut executed_case_ids = BTreeSet::new();
    for execution_case in &oracle.execution_cases {
        if !executed_case_ids.insert(execution_case.id.as_str()) {
            return Err(invalid_quant_linear_delegation(format!(
                "duplicate QuantLinear oracle case {}",
                execution_case.id
            )));
        }
        let input = tensor(
            backend,
            &oracle.fixture_inputs.input_shape,
            &oracle.fixture_inputs.input,
            DType::F32,
            &cancellation,
        )?;
        let bias = tensor(
            backend,
            &[u64::try_from(oracle.fixture_inputs.bias.len())?],
            &oracle.fixture_inputs.bias,
            DType::F32,
            &cancellation,
        )?;
        let output_gradient = tensor(
            backend,
            &oracle.fixture_inputs.output_gradient_shape,
            &oracle.fixture_inputs.output_gradient,
            DType::F32,
            &cancellation,
        )?;
        let compute_dtype = quant_linear_oracle_dtype(&execution_case.compute_dtype)?;
        if !execution_case.weight_scale.is_default()
            || execution_case.weight_requires_grad != execution_case.weight_layout.is_none()
        {
            return Err(invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} has an invalid weight contract",
                execution_case.id
            )));
        }
        let expected_weight_runtime_type = if execution_case.weight_layout.is_some() {
            "comfy_kitchen.tensor.base.QuantizedTensor"
        } else {
            "torch.Tensor"
        };
        if execution_case.weight_runtime_type != expected_weight_runtime_type {
            return Err(invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} changed weight runtime type",
                execution_case.id
            )));
        }
        let native_weight = if let Some(weight_layout) = execution_case.weight_layout.as_deref() {
            let layout = QuantLinearLayout::from_source_name(weight_layout).ok_or_else(|| {
                invalid_quant_linear_delegation(format!(
                    "unsupported QuantLinear oracle weight layout {weight_layout}"
                ))
            })?;
            let rows = *oracle.fixture_inputs.weight_shape.first().ok_or_else(|| {
                invalid_quant_linear_delegation("QuantLinear oracle weight shape has no rows")
            })?;
            let columns = *oracle.fixture_inputs.weight_shape.get(1).ok_or_else(|| {
                invalid_quant_linear_delegation("QuantLinear oracle weight shape has no columns")
            })?;
            QuantLinearWeight::Quantized(quantize_linear_matrix(
                layout,
                compute_dtype,
                &oracle.fixture_inputs.weight,
                usize::try_from(rows)?,
                usize::try_from(columns)?,
                execution_case.weight_scale.as_native(),
                &cancellation,
            )?)
        } else {
            QuantLinearWeight::Dense(tensor(
                backend,
                &oracle.fixture_inputs.weight_shape,
                &oracle.fixture_inputs.weight,
                DType::F32,
                &cancellation,
            )?)
        };
        let options = QuantLinearOptions::from_source_layout(
            execution_case.source_layout.as_deref(),
            execution_case.input_scale.as_native(),
            compute_dtype,
            execution_case.weight_requires_grad,
            execution_case.fp8_backward,
        )?;
        let mut higher_order_execution = quant_linear_forward_exact_native(
            &**backend,
            &input,
            native_weight.clone(),
            Some(&bias),
            options,
            &context(backend, &cancellation)?,
        )?;
        assert!(matches!(
            higher_order_execution.backward(
                &**backend,
                &output_gradient,
                true,
                &context(backend, &cancellation)?,
            ),
            Err(QuantLinearError::OnceDifferentiable)
        ));
        let mut execution = quant_linear_forward_exact_native(
            &**backend,
            &input,
            native_weight,
            Some(&bias),
            options,
            &context(backend, &cancellation)?,
        )?;
        if execution.output().descriptor().dtype() != compute_dtype
            || execution_case.output_dtype != execution_case.compute_dtype
        {
            return Err(invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} changed output dtype",
                execution_case.id
            )));
        }
        close_quant_linear_oracle_case(
            &execution_case.id,
            "output",
            &values(backend, execution.output(), &cancellation)?,
            &execution_case.output,
        );
        let gradients = execution.backward(
            &**backend,
            &output_gradient,
            false,
            &context(backend, &cancellation)?,
        )?;
        if gradients.input_arity() != 6 || !gradients.as_slice()[3..].iter().all(Option::is_none) {
            return Err(invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} changed exact gradient arity",
                execution_case.id
            )));
        }
        let expected_gradient_dtypes = vec![
            Some(execution_case.compute_dtype.clone()),
            execution_case
                .weight_requires_grad
                .then(|| execution_case.compute_dtype.clone()),
            Some(execution_case.compute_dtype.clone()),
        ];
        if execution_case.gradient_dtypes != expected_gradient_dtypes {
            return Err(invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} changed gradient dtypes",
                execution_case.id
            )));
        }
        let input_gradient = gradients.input().ok_or_else(|| {
            invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} omitted input gradient",
                execution_case.id
            ))
        })?;
        let bias_gradient = gradients.bias().ok_or_else(|| {
            invalid_quant_linear_delegation(format!(
                "QuantLinear oracle case {} omitted bias gradient",
                execution_case.id
            ))
        })?;
        close_quant_linear_oracle_case(
            &execution_case.id,
            "input_gradient",
            &values(backend, input_gradient, &cancellation)?,
            &execution_case.input_gradient,
        );
        match (
            gradients.weight(),
            execution_case.weight_gradient.as_deref(),
        ) {
            (Some(weight_gradient), Some(expected)) => close_quant_linear_oracle_case(
                &execution_case.id,
                "weight_gradient",
                &values(backend, weight_gradient, &cancellation)?,
                expected,
            ),
            (None, None) => {}
            _ => {
                return Err(invalid_quant_linear_delegation(format!(
                    "QuantLinear oracle case {} changed weight-gradient presence",
                    execution_case.id
                )));
            }
        }
        close_quant_linear_oracle_case(
            &execution_case.id,
            "bias_gradient",
            &values(backend, bias_gradient, &cancellation)?,
            &execution_case.bias_gradient,
        );
    }
    Ok(executed_case_ids.into_iter().map(str::to_owned).collect())
}

fn execute_case(
    case: &FixtureCase,
    backend: &TestBackend,
) -> Result<CaseExecution, Box<dyn Error>> {
    let allowed = allowed_observations(&case.execution_case).ok_or_else(|| {
        io::Error::other(format!(
            "unhandled autograd execution case {}",
            case.execution_case
        ))
    })?;
    let execution = match case.execution_case.as_str() {
        "tape_requires_grad"
        | "tape_requires_grad_mutation"
        | "factory_requires_grad"
        | "mode_enable"
        | "mode_no_grad"
        | "mode_inference"
        | "tape_backward"
        | "tape_grad" => probe_tape(&case.execution_case, backend),
        "function_context_needs"
        | "function_context_save"
        | "function_context_mark"
        | "function_context_saved" => probe_context(&case.execution_case, backend),
        "scaler_step" | "grad_scaler" | "scaler_scale" | "scaler_update" | "scaler_unscale" => {
            probe_scaler(&case.execution_case, backend)
        }
        "optimizer_sgd"
        | "optimizer_adam"
        | "optimizer_rmsprop"
        | "optimizer_adamw"
        | "optimizer_adamw_functional" => probe_optimizer(&case.execution_case, backend),
        "detach_alias" | "data_alias" | "gradient_store_lookup" | "gradient_store_zero" => {
            probe_alias_or_store(&case.execution_case, backend)
        }
        "autocast_cuda_alias" | "autocast_policy" => probe_autocast(&case.execution_case),
        "vector_quantize"
        | "checkpoint_function"
        | "hada_weight_tucker"
        | "add_aux_loss"
        | "offload_checkpoint"
        | "hada_weight"
        | "checkpoint_api" => {
            let receipts = probe_custom(&case.execution_case, backend)?;
            return Ok(CaseExecution::observations(&receipts));
        }
        QUANT_LINEAR_EXECUTION_CASE => {
            let native_receipts = probe_quant_linear(case, backend)?;
            let receipt = allowed
                .first()
                .copied()
                .ok_or_else(|| io::Error::other("QuantLinear observation receipt is missing"))?;
            return Ok(CaseExecution::quant_linear(receipt, native_receipts));
        }
        unhandled => Err(format!("unhandled autograd execution case {unhandled}").into()),
    };
    execution.map(|()| CaseExecution::observations(&allowed))
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn target_directory(root: &Path) -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(directory) => {
            let directory = PathBuf::from(directory);
            if directory.is_absolute() {
                directory
            } else {
                root.join(directory)
            }
        }
        None => root.join("target"),
    }
}

fn write_autograd_artifact(
    cases: &BTreeMap<String, bool>,
    execution_receipts: &BTreeMap<String, serde_json::Value>,
) -> Result<(), Box<dyn Error>> {
    if let Some((case, _)) = cases.iter().find(|(_, passed)| !**passed) {
        return Err(
            io::Error::other(format!("VAL-AUTOGRAD-001 validation case failed: {case}")).into(),
        );
    }
    let scope = "Task 7/101/102/103 native reverse-mode tape, strict 36-row source fixtures, seven exact custom-function contracts, accumulation, cancellation, and release closure";
    let artifact = serde_json::json!({
        "validation_id": "VAL-AUTOGRAD-001",
        "validation": "VAL-AUTOGRAD-001",
        "scope": scope,
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "development_oracle_executed": false,
        },
        "fixture_digests": {
            AUTOGRAD_CATALOG_PATH: AUTOGRAD_CATALOG_SHA256,
            QUANT_LINEAR_FIXTURE_PATH: QUANT_LINEAR_FIXTURE_SHA256,
            "crates/sim_test_support/fixtures/autograd/breadth-v1.json": BREADTH_FIXTURE_SHA256,
        },
        "summary": {"passed": cases.len(), "failed": 0, "skipped": 0},
        "cases": cases,
        "execution_receipts": execution_receipts,
        "skipped": [],
        "validation_closure": {
            "claimed": true,
            "stage": "task-7-101-102-103-autograd-closure",
            "validated_scope": scope,
        },
        "release_closure_claimed": false,
        "release_closure_required": true,
        "remaining_release_gates": ["comfy-parity-final-validation"],
    });
    let root = workspace_root()?;
    let directory = target_directory(&root).join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let path = directory.join("val-autograd-001.json");
    let temporary = directory.join("val-autograd-001.json.tmp");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[test]
fn every_catalog_row_executes_its_pinned_native_oracle_case() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        format!("{:x}", Sha256::digest(BREADTH_FIXTURE)),
        BREADTH_FIXTURE_SHA256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(AUTOGRAD_CATALOG)),
        AUTOGRAD_CATALOG_SHA256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(QUANT_LINEAR_FIXTURE)),
        QUANT_LINEAR_FIXTURE_SHA256
    );
    let fixture = parse_and_validate_breadth_fixture(BREADTH_FIXTURE)?;
    assert_eq!(fixture.oracle.comfyui_version, "0.27.1");
    assert_eq!(
        fixture.oracle.comfyui_tree_sha256,
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert!(fixture.oracle.development_only);
    assert_eq!(fixture.oracle.source_files.len(), 6);
    let expected_sources = [
        (
            "comfy/ldm/cascade/stage_a.py",
            "8c1399846647a3376738daa90578ad8f6af224640e74dfdd23250c36676dc546",
        ),
        (
            "comfy/ldm/modules/diffusionmodules/util.py",
            "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
        ),
        (
            "comfy/ops.py",
            "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
        ),
        (
            "comfy/weight_adapter/loha.py",
            "579ca1e33e0d244e0d7eedd30fb727913341f8e7bfbd74b51221f567612286d5",
        ),
        (
            "comfy/ldm/hunyuan3dv2_1/hunyuandit.py",
            "183c09de7cadde60417916836a9be5bb8512553ea804d9cfdacac0ab2cdf0e45",
        ),
        (
            "comfy_extras/nodes_train.py",
            "d88aabbc72da32e1e5d934b6e4f3a6587438fe7a7840199771530644a21916ae",
        ),
    ];
    for (path, digest) in expected_sources {
        assert_eq!(
            fixture.oracle.source_files.get(path).map(String::as_str),
            Some(digest)
        );
    }
    let backend = backend()?;
    let mut artifact_cases = BTreeMap::new();
    let mut artifact_receipts = BTreeMap::new();
    for case in &fixture.catalog_cases {
        let execution = execute_case(case, &backend)
            .map_err(|error| format!("{} {}: {error}", case.id, case.symbol))?;
        validate_execution_receipts(case, &execution)?;
        if case.execution_case == QUANT_LINEAR_EXECUTION_CASE {
            let expected = expected_quant_linear_case_ids()
                .into_iter()
                .map(str::to_owned)
                .collect::<BTreeSet<_>>();
            if execution.native_receipts != expected {
                return Err(io::Error::other(
                    "QuantLinear did not execute the exact 22 native oracle receipts",
                )
                .into());
            }
        } else if !execution.native_receipts.is_empty() {
            return Err(io::Error::other(format!(
                "non-QuantLinear row {} returned undeclared native subcase receipts",
                case.id
            ))
            .into());
        }
        let observations = execution
            .observations
            .iter()
            .map(|observation| {
                serde_json::json!({
                    "case": observation.case,
                    "expected": observation.expected,
                    "sha256": observation.sha256,
                })
            })
            .collect::<Vec<_>>();
        artifact_receipts.insert(
            case.id.clone(),
            serde_json::json!({
                "execution_case": case.execution_case,
                "observations": observations,
                "native_receipts": execution.native_receipts,
            }),
        );
        artifact_cases.insert(format!("fixture_{}", case.id.replace('-', "_")), true);
    }
    assert_eq!(artifact_cases.len(), 36);

    let branch_leaf = LeafId::new("artifact-branch")?;
    let mut branch_tape = AutogradTape::new(GradientMode::Enabled);
    let branch_outputs = (0..2)
        .map(|_| {
            branch_tape
                .record(
                    vec![AutogradInput::Leaf(branch_leaf.clone())],
                    1,
                    Vec::new(),
                    Arc::new(IdentityRule),
                )?
                .and_then(|outputs| outputs.first().copied())
                .ok_or_else(|| AutogradError::InvalidGraph {
                    reason: "artifact branch did not record".to_owned(),
                })
        })
        .collect::<Result<Vec<_>, AutogradError>>()?;
    let branch_gradients = branch_tape.backward(
        branch_outputs
            .into_iter()
            .map(|output| {
                Ok((
                    output,
                    tensor(
                        &backend,
                        &[],
                        &[2.0],
                        DType::F32,
                        &CancellationToken::default(),
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
        &AddReducer,
        &CancellationToken::default(),
    )?;
    close(
        &values(
            &backend,
            branch_gradients
                .get(&branch_leaf)
                .ok_or("missing branch gradient")?,
            &CancellationToken::default(),
        )?,
        &[4.0],
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let mut cancelled_tape = AutogradTape::new(GradientMode::Enabled);
    cancelled_tape.record(
        vec![AutogradInput::Constant],
        1,
        Vec::new(),
        Arc::new(IdentityRule),
    )?;
    assert!(matches!(
        cancelled_tape.backward(Vec::new(), &AddReducer, &cancelled),
        Err(AutogradError::Cancelled)
    ));
    assert!(matches!(cancelled_tape.state(), TapeState::Cancelled(_)));
    assert_eq!(cancelled_tape.retained_node_count(), 0);

    artifact_cases.extend([
        (
            "autograd_breadth_catalog_has_exact_unique_coverage".to_owned(),
            true,
        ),
        ("branch_gradients_accumulate".to_owned(), true),
        (
            "cancellation_is_terminal_and_releases_saved_tensors".to_owned(),
            true,
        ),
        ("completed_tapes_release_nodes".to_owned(), true),
        (
            "invariant_bearing_leaf_ids_use_checked_wire_conversion".to_owned(),
            LeafId::new(" ").is_err(),
        ),
        ("no_grad_and_inference_do_not_record".to_owned(), true),
        (
            "seven_custom_function_contracts_are_exact".to_owned(),
            CUSTOM_FUNCTIONS.len() == 7
                && CUSTOM_FUNCTIONS
                    .iter()
                    .map(|contract| contract.id)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == 7,
        ),
        (
            "strict_36_row_source_fixture_registry_is_closed".to_owned(),
            true,
        ),
    ]);
    assert_eq!(artifact_cases.len(), 44);
    write_autograd_artifact(&artifact_cases, &artifact_receipts)?;
    Ok(())
}

fn assert_breadth_mutation_rejected(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Result<(), Box<dyn Error>> {
    let mut value: serde_json::Value = serde_json::from_slice(BREADTH_FIXTURE)?;
    mutate(&mut value);
    let bytes = serde_json::to_vec(&value)?;
    assert!(parse_and_validate_breadth_fixture(&bytes).is_err());
    Ok(())
}

#[test]
fn strict_general_registry_rejects_missing_duplicate_prose_unknown_and_unexecuted_observations()
-> Result<(), Box<dyn Error>> {
    assert_breadth_mutation_rejected(|value| {
        if let Some(observations) = value["catalog_cases"]
            .as_array_mut()
            .and_then(|cases| cases.first_mut())
            .and_then(|case| case["source_observations"].as_array_mut())
        {
            observations.clear();
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(observations) = value["catalog_cases"]
            .as_array_mut()
            .and_then(|cases| cases.first_mut())
            .and_then(|case| case["source_observations"].as_array_mut())
            && let Some(duplicate) = observations.first().cloned()
        {
            observations.push(duplicate);
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(expected) = value["catalog_cases"]
            .as_array_mut()
            .and_then(|cases| cases.first_mut())
            .and_then(|case| case["source_observations"].as_array_mut())
            .and_then(|observations| observations.first_mut())
            .and_then(|observation| observation.get_mut("expected"))
        {
            *expected = serde_json::Value::String("descriptive prose only".to_owned());
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(observation) = value["catalog_cases"]
            .as_array_mut()
            .and_then(|cases| cases.first_mut())
            .and_then(|case| case["source_observations"].as_array_mut())
            .and_then(|observations| observations.first_mut())
            .and_then(serde_json::Value::as_object_mut)
        {
            observation.insert("unknown".to_owned(), serde_json::Value::Bool(true));
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(case) = value["catalog_cases"]
            .as_array_mut()
            .and_then(|cases| cases.first_mut())
        {
            case["execution_case"] =
                serde_json::Value::String("unreferenced_execution_case".to_owned());
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(cases) = value["catalog_cases"].as_array_mut() {
            cases.pop();
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        if let Some(cases) = value["catalog_cases"].as_array_mut()
            && let Some(duplicate) = cases.first().cloned()
        {
            cases.push(duplicate);
        }
    })?;
    assert_breadth_mutation_rejected(|value| {
        value["unknown_root_field"] = serde_json::Value::Bool(true);
    })?;
    assert_breadth_mutation_rejected(|value| {
        value["custom_functions"]["unknown_function"] = serde_json::json!({});
    })?;
    assert_breadth_mutation_rejected(|value| {
        value["custom_functions"]["vector_quantize"]["unknown_field"] =
            serde_json::Value::Bool(true);
    })?;

    let fixture = parse_and_validate_breadth_fixture(BREADTH_FIXTURE)?;
    let case = fixture
        .catalog_cases
        .first()
        .ok_or("missing canonical breadth case")?;
    let unexecuted = CaseExecution {
        observations: BTreeSet::new(),
        native_receipts: BTreeSet::new(),
    };
    assert!(validate_execution_receipts(case, &unexecuted).is_err());
    let undeclared =
        CaseExecution::observations(&[ObservationReceipt::new("undeclared", "prose_only_receipt")]);
    assert!(validate_execution_receipts(case, &undeclared).is_err());
    Ok(())
}

#[test]
fn quant_linear_delegation_rejects_missing_stale_renamed_or_unused_fixture()
-> Result<(), Box<dyn Error>> {
    let fixture: BreadthFixture = serde_json::from_slice(BREADTH_FIXTURE)?;
    let canonical_case = fixture
        .catalog_cases
        .into_iter()
        .find(|case| case.id == QUANT_LINEAR_CATALOG_ID)
        .ok_or_else(|| invalid_quant_linear_delegation("missing QuantLinear catalog case"))?;
    assert!(load_quant_linear_oracle(&canonical_case, QUANT_LINEAR_FIXTURE).is_ok());

    let mut missing = canonical_case.clone();
    missing.source_observations.clear();
    assert!(load_quant_linear_oracle(&missing, QUANT_LINEAR_FIXTURE).is_err());

    let mut renamed = canonical_case.clone();
    renamed.source_observations = vec![SourceObservation {
        case: "delegated_fixture".to_owned(),
        expected: ".agents/specs/comfy-parity/fixtures/renamed-quant-linear-source-oracle.json"
            .to_owned(),
        sha256: Some(QUANT_LINEAR_FIXTURE_SHA256.to_owned()),
    }];
    assert!(load_quant_linear_oracle(&renamed, QUANT_LINEAR_FIXTURE).is_err());

    let mut stale_claim = canonical_case.clone();
    stale_claim.source_observations = vec![SourceObservation {
        case: "delegated_fixture".to_owned(),
        expected: QUANT_LINEAR_FIXTURE_PATH.to_owned(),
        sha256: Some("0000000000000000000000000000000000000000000000000000000000000000".to_owned()),
    }];
    assert!(load_quant_linear_oracle(&stale_claim, QUANT_LINEAR_FIXTURE).is_err());

    let mut stale_fixture = QUANT_LINEAR_FIXTURE.to_vec();
    let last = stale_fixture
        .last_mut()
        .ok_or_else(|| invalid_quant_linear_delegation("empty QuantLinear fixture"))?;
    *last ^= 1;
    assert!(load_quant_linear_oracle(&canonical_case, &stale_fixture).is_err());

    let mut unused = canonical_case;
    unused.execution_case = "tape_backward".to_owned();
    assert!(load_quant_linear_oracle(&unused, QUANT_LINEAR_FIXTURE).is_err());

    let mut wrong_schema: QuantLinearOracle = serde_json::from_slice(QUANT_LINEAR_FIXTURE)?;
    wrong_schema.schema_version = 5;
    assert!(validate_quant_linear_oracle_structure(&wrong_schema).is_err());

    let mut wrong_owner: QuantLinearOracle = serde_json::from_slice(QUANT_LINEAR_FIXTURE)?;
    wrong_owner.owner_task_id = "renamed-quant-linear-owner".to_owned();
    assert!(validate_quant_linear_oracle_structure(&wrong_owner).is_err());

    let mut deleted_case: QuantLinearOracle = serde_json::from_slice(QUANT_LINEAR_FIXTURE)?;
    deleted_case
        .execution_cases
        .pop()
        .ok_or_else(|| invalid_quant_linear_delegation("missing QuantLinear execution case"))?;
    assert!(validate_quant_linear_oracle_structure(&deleted_case).is_err());

    let mut renamed_case: QuantLinearOracle = serde_json::from_slice(QUANT_LINEAR_FIXTURE)?;
    renamed_case
        .execution_cases
        .first_mut()
        .ok_or_else(|| invalid_quant_linear_delegation("missing QuantLinear execution case"))?
        .id = "renamed-quant-linear-case".to_owned();
    assert!(validate_quant_linear_oracle_structure(&renamed_case).is_err());

    let mut duplicate_case: QuantLinearOracle = serde_json::from_slice(QUANT_LINEAR_FIXTURE)?;
    let duplicate_id = duplicate_case
        .execution_cases
        .first()
        .ok_or_else(|| invalid_quant_linear_delegation("missing QuantLinear execution case"))?
        .id
        .clone();
    duplicate_case
        .execution_cases
        .get_mut(1)
        .ok_or_else(|| {
            invalid_quant_linear_delegation("missing second QuantLinear execution case")
        })?
        .id = duplicate_id;
    assert!(validate_quant_linear_oracle_structure(&duplicate_case).is_err());
    Ok(())
}

#[test]
fn cancellation_and_true_copy_are_preserved_by_custom_adapters() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let live = CancellationToken::default();
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &live)?;
    let clone = clone_with_context_exact_native(
        &backend,
        &input,
        MemoryFormatReference::PreserveFormat,
        &context(&backend, &live)?,
    )?;
    assert_ne!(clone.tensor_id(), input.tensor_id());
    assert_ne!(clone.storage_id(), input.storage_id());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let codebook = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &live)?;
    assert!(matches!(
        VectorQuantizeFunction::forward(
            &backend,
            &input,
            &codebook,
            [true, true],
            &context(&backend, &cancelled)?,
        ),
        Err(AutogradBreadthError::Cancelled)
    ));
    Ok(())
}
