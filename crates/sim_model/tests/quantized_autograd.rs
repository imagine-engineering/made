use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_model::{
    NativeModule, QuantLinearError, QuantLinearLayout, QuantLinearOptions, QuantLinearScale,
    QuantLinearWeight, QuantizationError, QuantizationKind, quant_linear_forward_exact_native,
    quantize_linear_matrix, quantize_matrix,
};
use sim_tensor::{
    AutogradError, BackendCapabilityMatrix, BackendWorkspaceLease, BinaryOperation,
    CachedAllocationOwner, CancellationToken, ConvolutionSpec, CpuBackend, CpuWorkspaceAuthority,
    CustomKernelId, DType, DeviceId, EventFence, ExecutionContext, IndexSpec, Layout,
    LinearAlgebraOperation, ReductionSpec, ResizeSpec, Scalar, ScalarSide, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError, UnaryOperation, ViewAccess,
    autograd::breadth::AutogradBreadthError,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use std::collections::BTreeSet;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

const ORACLE_FIXTURE: &[u8] = include_bytes!(
    "../../../crates/sim_test_support/parity/fixtures/quant-linear-source-oracle.json"
);
const ORACLE_FIXTURE_SHA256: &str =
    "74acf934871befe3a87a91de6aea430a7ea9a16a821441bd716768dfb1919d0c";
const ORACLE_GENERATOR: &[u8] = include_bytes!(
    "../../../crates/sim_test_support/parity/oracles/generate_quant_linear_oracle.py"
);
const ORACLE_GENERATOR_SHA256: &str =
    "beac9521a7dc3e1676e049504c665a4c372868e8c4cf3fe52cc24b9dae44c662";
const ORACLE_MANIFEST: &[u8] = include_bytes!(
    "../../../crates/sim_test_support/parity/oracles/quant_linear_oracle_manifest.json"
);
const ORACLE_MANIFEST_SHA256: &str =
    "0f1dc92eb5987737003e536f5a9841b8d0893e6ba3d028243fce39a5df5940dd";

#[derive(Debug, Deserialize)]
struct OracleInputs {
    input_shape: Vec<u64>,
    input: Vec<f32>,
    weight_shape: Vec<u64>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    output_gradient_shape: Vec<u64>,
    output_gradient: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OracleScale {
    Default,
    Explicit { value: f32 },
    Recalculate,
}

impl OracleScale {
    fn quant_linear_scale(&self) -> QuantLinearScale {
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

#[derive(Debug, Deserialize)]
struct OracleExecutionCase {
    id: String,
    source_layout: Option<String>,
    input_scale: OracleScale,
    weight_layout: Option<String>,
    weight_scale: OracleScale,
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

#[derive(Debug, Deserialize)]
struct QuantLinearOracle {
    schema_version: u16,
    owner_task_id: String,
    fixture_inputs: OracleInputs,
    execution_cases: Vec<OracleExecutionCase>,
}

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
}

struct DelegatingTestBackend<'a> {
    backend: &'a CpuBackend,
    device: DeviceId,
    cancellation_on_reserve: Option<&'a CancellationToken>,
    reserve_calls: AtomicUsize,
}

impl CachedAllocationOwner for DelegatingTestBackend<'_> {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        self.backend.release_cached_allocations(cancellation)
    }
}

impl TensorBackend for DelegatingTestBackend<'_> {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn capabilities(&self) -> &BackendCapabilityMatrix {
        self.backend.capabilities()
    }

    fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        self.reserve_calls.fetch_add(1, Ordering::AcqRel);
        let workspace = self.backend.reserve_workspace(context, requested)?;
        if let Some(cancellation) = self.cancellation_on_reserve {
            cancellation.cancel();
        }
        Ok(workspace)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.allocate(descriptor, context)
    }

    fn copy(
        &self,
        source: &Tensor,
        destination: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.copy(source, destination, context)
    }

    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        self.backend.record_event(context)
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.backend.wait_event(event, context)
    }

    fn fill(
        &self,
        value: Scalar,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.fill(value, output, context)
    }

    fn unary(
        &self,
        operation: UnaryOperation,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.unary(operation, input, output, context)
    }

    fn binary(
        &self,
        operation: BinaryOperation,
        left: &Tensor,
        right: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.binary(operation, left, right, output, context)
    }

    fn binary_scalar(
        &self,
        operation: BinaryOperation,
        input: &Tensor,
        scalar: Scalar,
        scalar_side: ScalarSide,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend
            .binary_scalar(operation, input, scalar, scalar_side, output, context)
    }

    fn reduction(
        &self,
        operation: &ReductionSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.reduction(operation, input, output, context)
    }

    fn indexing(
        &self,
        operation: &IndexSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.indexing(operation, inputs, output, context)
    }

    fn resize(
        &self,
        operation: ResizeSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.resize(operation, input, output, context)
    }

    fn convolution(
        &self,
        operation: &ConvolutionSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend.convolution(operation, inputs, output, context)
    }

    fn linear_algebra(
        &self,
        operation: LinearAlgebraOperation,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.backend
            .linear_algebra(operation, inputs, output, context)
    }

    fn custom_kernel(
        &self,
        kernel: &CustomKernelId,
        inputs: &[Tensor],
        outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.backend.custom_kernel(kernel, inputs, outputs, context)
    }
}

impl std::ops::Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn backend() -> Result<TestBackend, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(16 * 1024 * 1024)?,
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

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn dense_options(compute_dtype: DType) -> QuantLinearOptions {
    QuantLinearOptions {
        layout: None,
        input_scale: QuantLinearScale::Default,
        compute_dtype,
        weight_requires_grad: true,
        fp8_backward: false,
    }
}

#[test]
fn source_oracle_fixture_is_pinned_and_declares_the_exact_callable_contract()
-> Result<(), Box<dyn Error>> {
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_FIXTURE)),
        ORACLE_FIXTURE_SHA256
    );
    let fixture: serde_json::Value = serde_json::from_slice(ORACLE_FIXTURE)?;
    assert_eq!(fixture["schema_version"], 4);
    assert_eq!(
        fixture["owner_task_id"],
        "comfy-parity-quantized-autograd-adapter"
    );
    assert_eq!(fixture["oracle"]["simui_version"], "0.27.1");
    assert_eq!(fixture["oracle"]["simui_file_count"], 949);
    assert_eq!(
        fixture["oracle"]["simui_tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(fixture["oracle"]["python_major_minor"], "3.12");
    assert!(fixture["oracle"].get("python_version").is_none());
    assert!(fixture["oracle"].get("platform").is_none());
    assert_eq!(
        fixture["oracle"]["runtime_profile"]["python_implementation"],
        "CPython"
    );
    assert_eq!(
        fixture["oracle"]["runtime_profile"]["python_cache_tag"],
        "cpython-312"
    );
    assert_eq!(fixture["oracle"]["runtime_profile"]["python_abi"], "cp312");
    assert_eq!(
        fixture["oracle"]["runtime_profile"]["platform_system"],
        "Darwin"
    );
    assert_eq!(
        fixture["oracle"]["runtime_profile"]["platform_machine"],
        "arm64"
    );
    assert_eq!(
        fixture["oracle"]["runtime_profile"]["sysconfig_platform"],
        "macosx-11.0-arm64"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["version"],
        "2.10.0"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["python_source_file_count"],
        2156
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["python_source_sha256"],
        "1b3ec473cbc22443afedce5567e5a1bdf9618944e3b0aec78b3061758848332b"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["record_sha256"],
        "afed3d2c56b90ca742f83433f791ff1f0e7433eb4faa1a25bc6695d3af51d40d"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["record_entry_count"],
        14541
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["record_hashed_entry_count"],
        12341
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["supplemental_bytecode_file_count"],
        2199
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["supplemental_bytecode_sha256"],
        "2a2eec5dc29068a730fd34b549647f0acdbf62e6669073c8972b8faab0e4da79"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["wheel_sha256"],
        "29d9da686c1260684d7454c4edac4e2dc5a14da0b7ab0fbf1767f6b5a0022e41"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["torch"]["wheel_tags"],
        serde_json::json!(["cp312-none-macosx_11_0_arm64"])
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["version"],
        "0.2.16"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["python_source_file_count"],
        27
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["python_source_sha256"],
        "a19c43ef5e77a99e4a964c46cf9b19dc7ec809a97f9089aebce1550870603d66"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["record_sha256"],
        "139d248ca695d590822196738916ac1fd29e1c254bec81e9404a07d8ffa5a58e"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["record_entry_count"],
        66
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["record_hashed_entry_count"],
        38
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["supplemental_bytecode_file_count"],
        27
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["supplemental_bytecode_sha256"],
        "2838d41338422e326773affb00e2e29dbf3d6896834d36e34e0cdf4cf999425c"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["wheel_sha256"],
        "69e6228a0d35958183cc1812f07c565ce837b95eb51bd8a33acba9fa4f68d6c9"
    );
    assert_eq!(
        fixture["oracle"]["dependencies"]["sim-kitchen"]["wheel_tags"],
        serde_json::json!(["py3-none-any"])
    );
    assert_eq!(
        fixture["oracle"]["quant_linear_symbol_sha256"],
        "dc91a7dbcc41fb6dd94b65c4c445fb6e89bcd3fb55ae85413fb66b2b9b4024fb"
    );
    assert_eq!(
        fixture["oracle"]["generator"],
        "crates/sim_test_support/parity/oracles/generate_quant_linear_oracle.py"
    );
    assert_eq!(
        fixture["oracle"]["generator_sha256"],
        ORACLE_GENERATOR_SHA256
    );
    assert_eq!(
        fixture["oracle"]["manifest"],
        "crates/sim_test_support/parity/oracles/quant_linear_oracle_manifest.json"
    );
    assert_eq!(fixture["oracle"]["manifest_sha256"], ORACLE_MANIFEST_SHA256);
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_GENERATOR)),
        ORACLE_GENERATOR_SHA256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_MANIFEST)),
        ORACLE_MANIFEST_SHA256
    );
    assert_eq!(fixture["oracle"]["determinism"]["algorithms"], true);
    assert_eq!(fixture["oracle"]["determinism"]["random_seed"], 0);
    assert_eq!(fixture["oracle"]["determinism"]["threads"], 1);
    assert_eq!(fixture["oracle"]["determinism"]["interop_threads"], 1);
    assert_eq!(fixture["callable"]["class_name"], "QuantLinearFunc");
    assert_eq!(
        fixture["callable"]["forward_inputs"]
            .as_array()
            .ok_or("forward inputs are missing")?
            .len(),
        6
    );
    assert_eq!(
        fixture["callable"]["backward_outputs"]
            .as_array()
            .ok_or("backward outputs are missing")?
            .len(),
        6
    );
    assert_eq!(
        fixture["callable"]["backward_outputs"][3],
        serde_json::Value::Null
    );
    assert_eq!(
        fixture["callable"]["higher_order_decorator"],
        "torch.autograd.function.once_differentiable"
    );
    assert_eq!(
        fixture["execution_cases"]
            .as_array()
            .ok_or("execution cases are missing")?
            .len(),
        22
    );
    assert_eq!(fixture["coverage"]["case_count"], 22);
    assert_eq!(fixture["coverage"]["dense_weight_cases"], 14);
    assert_eq!(fixture["coverage"]["quantized_weight_cases"], 8);
    assert_eq!(
        fixture["source_probes"]["runtime_backward"]["output_arity"],
        6
    );
    assert_eq!(
        fixture["source_probes"]["runtime_backward"]["none_indexes"],
        serde_json::json!([3, 4, 5])
    );
    assert_eq!(
        fixture["source_probes"]["once_differentiable"]["first_order_requires_grad"],
        serde_json::json!([false, false, false])
    );
    assert_eq!(
        fixture["source_probes"]["once_differentiable"]["second_order_rejected"],
        true
    );
    assert_eq!(
        fixture["source_probes"]["released_state"]["second_backward_rejected"],
        true
    );
    assert_eq!(
        fixture["source_probes"]["unsupported_layout"]["source_name"],
        "TensorCoreINT4Layout"
    );
    assert_eq!(
        fixture["source_probes"]["unsupported_layout"]["exception_type"],
        "KeyError"
    );
    Ok(())
}

fn base_tensors(
    backend: &TestBackend,
    cancellation: &CancellationToken,
) -> Result<(Tensor, Tensor, Tensor, Tensor), Box<dyn Error>> {
    Ok((
        tensor(
            backend,
            &[2, 3],
            &[1.0, -2.0, 0.5, 0.0, 3.0, -1.0],
            DType::F32,
            cancellation,
        )?,
        tensor(
            backend,
            &[2, 3],
            &[2.0, -1.0, 0.25, -0.5, 1.0, 2.0],
            DType::F32,
            cancellation,
        )?,
        tensor(backend, &[2], &[0.5, -0.25], DType::F32, cancellation)?,
        tensor(
            backend,
            &[2, 2],
            &[1.0, -2.0, 0.5, 3.0],
            DType::F32,
            cancellation,
        )?,
    ))
}

#[test]
fn dense_forward_backward_preserves_source_semantics_and_six_input_arity()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let (input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let mut execution = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight),
        Some(&bias),
        dense_options(DType::F32),
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(execution.output().descriptor().shape(), &[2, 2]);
    close(
        &values(&backend, execution.output(), &cancellation)?,
        &[4.625, -1.75, -2.75, 0.75],
        1.0e-6,
    );

    let gradients = execution.backward(
        &*backend,
        &output_gradient,
        false,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(gradients.input_arity(), 6);
    close(
        &values(
            &backend,
            gradients.input().ok_or("missing input gradient")?,
            &cancellation,
        )?,
        &[3.0, -3.0, -3.75, -0.5, 2.5, 6.125],
        1.0e-6,
    );
    close(
        &values(
            &backend,
            gradients.weight().ok_or("missing weight gradient")?,
            &cancellation,
        )?,
        &[1.0, -0.5, 0.0, -2.0, 13.0, -4.0],
        1.0e-6,
    );
    close(
        &values(
            &backend,
            gradients.bias().ok_or("missing bias gradient")?,
            &cancellation,
        )?,
        &[1.5, 1.0],
        1.0e-6,
    );
    assert!(gradients.as_slice()[3..].iter().all(Option::is_none));
    assert!(matches!(
        execution.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::Autograd(
            AutogradBreadthError::ReleasedContext
        ))
    ));
    Ok(())
}

#[test]
fn leading_dimensions_are_flattened_and_compute_dtypes_are_explicit() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let input = tensor(
            &backend,
            &[2, 1, 3],
            &[1.0, -2.0, 0.5, 0.0, 3.0, -1.0],
            DType::F32,
            &cancellation,
        )?;
        let weight = tensor(
            &backend,
            &[2, 3],
            &[2.0, -1.0, 0.25, -0.5, 1.0, 2.0],
            DType::F32,
            &cancellation,
        )?;
        let execution = quant_linear_forward_exact_native(
            &*backend,
            &input,
            QuantLinearWeight::Dense(weight),
            None,
            dense_options(dtype),
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(execution.output().descriptor().shape(), &[2, 1, 2]);
        assert_eq!(execution.output().descriptor().dtype(), dtype);
        close(
            &values(&backend, execution.output(), &cancellation)?,
            &[4.125, -1.5, -3.25, 1.0],
            if dtype == DType::F32 { 1.0e-6 } else { 0.03 },
        );
    }

    let storage = tensor(
        &backend,
        &[2, 3, 2],
        &[
            1.0, 99.0, -2.0, 99.0, 0.5, 99.0, 0.0, 99.0, 3.0, 99.0, -1.0, 99.0,
        ],
        DType::F32,
        &cancellation,
    )?;
    let noncontiguous = storage.view(
        TensorDescriptor::new_strided(
            vec![2, 3],
            vec![6, 2],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        ViewAccess::ReadOnly,
    )?;
    let weight = tensor(
        &backend,
        &[2, 3],
        &[2.0, -1.0, 0.25, -0.5, 1.0, 2.0],
        DType::F32,
        &cancellation,
    )?;
    let execution = quant_linear_forward_exact_native(
        &*backend,
        &noncontiguous,
        QuantLinearWeight::Dense(weight),
        None,
        dense_options(DType::F32),
        &context(&backend, &cancellation)?,
    )?;
    close(
        &values(&backend, execution.output(), &cancellation)?,
        &[4.125, -1.5, -3.25, 1.0],
        1.0e-6,
    );
    Ok(())
}

#[test]
fn every_layout_uses_the_canonical_quantizer_and_legacy_alias_maps_to_e4m3()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let source = [1.0, -2.0, 0.5, 0.0, 3.0, -1.0];
    assert_eq!(
        QuantLinearLayout::from_source_name("TensorCoreFP8Layout"),
        Some(QuantLinearLayout::TensorCoreFp8E4M3)
    );
    assert_eq!(QuantLinearLayout::from_source_name("unknown"), None);
    for layout in [
        QuantLinearLayout::TensorCoreFp8E4M3,
        QuantLinearLayout::TensorCoreFp8E5M2,
        QuantLinearLayout::TensorCoreMxFp8,
        QuantLinearLayout::TensorCoreNvFp4,
    ] {
        let quantized = quantize_linear_matrix(
            layout,
            DType::F32,
            &source,
            2,
            3,
            QuantLinearScale::Explicit(0.5),
            &cancellation,
        )?;
        assert_eq!(quantized.layout(), layout);
        assert_eq!(quantized.rows(), 2);
        assert_eq!(quantized.columns(), 3);
        assert!(quantized.storage_bytes() > 0);
        assert!(
            quantized
                .dequantize(&cancellation)?
                .iter()
                .all(|value| value.is_finite())
        );
    }

    let mx_default = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreMxFp8,
        DType::F32,
        &source,
        2,
        3,
        QuantLinearScale::Default,
        &cancellation,
    )?;
    let mx_explicit = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreMxFp8,
        DType::F32,
        &source,
        2,
        3,
        QuantLinearScale::Explicit(0.125),
        &cancellation,
    )?;
    assert_eq!(
        mx_default.dequantize(&cancellation)?,
        mx_explicit.dequantize(&cancellation)?
    );
    assert!(matches!(
        quantize_linear_matrix(
            QuantLinearLayout::TensorCoreFp8E4M3,
            DType::F32,
            &source,
            2,
            3,
            QuantLinearScale::Explicit(0.0),
            &cancellation,
        ),
        Err(QuantizationError::InvalidScale)
    ));
    let recalculated = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &source,
        2,
        3,
        QuantLinearScale::Recalculate,
        &cancellation,
    )?;
    close(&recalculated.dequantize(&cancellation)?, &source, 0.08);
    let f16_zero = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F16,
        &[0.0; 6],
        2,
        3,
        QuantLinearScale::Recalculate,
        &cancellation,
    )?;
    assert_eq!(f16_zero.dequantize(&cancellation)?, vec![0.0; 6]);
    Ok(())
}

#[test]
fn canonical_identity_and_scoped_materialization_bind_source_and_workspace()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let first = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &[1.0, 0.100_00],
        1,
        2,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    let same = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &[1.0, 0.100_00],
        1,
        2,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    let distinct_source_same_encoding = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &[1.0, 0.100_01],
        1,
        2,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    let different_scale = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &[1.0, 0.100_00],
        1,
        2,
        QuantLinearScale::Explicit(0.5),
        &cancellation,
    )?;
    let different_layout = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E5M2,
        DType::F32,
        &[1.0, 0.100_00],
        1,
        2,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    let different_dtype = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F16,
        &[1.0, 0.100_00],
        1,
        2,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    let different_shape = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &[1.0, 0.100_00],
        2,
        1,
        QuantLinearScale::Explicit(1.0),
        &cancellation,
    )?;
    assert_eq!(first.source_identity(), same.source_identity());
    assert_eq!(first.content_identity(), same.content_identity());
    assert_eq!(
        first.dequantize(&cancellation)?,
        distinct_source_same_encoding.dequantize(&cancellation)?
    );
    assert_ne!(
        first.source_identity(),
        distinct_source_same_encoding.source_identity()
    );
    assert_ne!(
        first.content_identity(),
        distinct_source_same_encoding.content_identity()
    );
    assert_eq!(first.source_identity(), different_scale.source_identity());
    assert_ne!(first.content_identity(), different_scale.content_identity());
    assert_eq!(first.source_identity(), different_layout.source_identity());
    assert_ne!(
        first.content_identity(),
        different_layout.content_identity()
    );
    assert_ne!(first.source_identity(), different_dtype.source_identity());
    assert_ne!(first.content_identity(), different_dtype.content_identity());
    assert_ne!(first.source_identity(), different_shape.source_identity());
    assert_ne!(first.content_identity(), different_shape.content_identity());
    assert_eq!(first.source_identity().to_hex().len(), 64);
    assert_eq!(first.content_identity().to_hex().len(), 64);

    let exact_context = context(&backend, &cancellation)?;
    let materialization = first.materialize(&*backend, &exact_context)?;
    assert_eq!(materialization.values(), first.dequantize(&cancellation)?);
    assert_eq!(materialization.content_identity(), first.content_identity());
    assert_eq!(exact_context.scratch.in_use_bytes(), 8);
    assert_eq!(exact_context.scratch.peak_bytes(), 8);
    drop(materialization);
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        backend.workspace_authority.authorize_workspace(7)?,
        &cancellation,
    );
    assert!(matches!(
        first.materialize(&*backend, &insufficient),
        Err(QuantizationError::MaterializationCapacity { requested: 8 })
    ));
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let device_mismatched_backend = DelegatingTestBackend {
        backend: &backend,
        device: DeviceId::new(DeviceKind::Metal, 0),
        cancellation_on_reserve: None,
        reserve_calls: AtomicUsize::new(0),
    };
    assert!(matches!(
        first.materialize(&device_mismatched_backend, &exact_context),
        Err(QuantizationError::MaterializationUnsupportedDevice { device })
            if device == DeviceId::new(DeviceKind::Metal, 0)
    ));
    assert_eq!(
        device_mismatched_backend
            .reserve_calls
            .load(Ordering::Acquire),
        0
    );
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let materialization_cancellation = CancellationToken::default();
    let materialization_context = context(&backend, &materialization_cancellation)?;
    let cancelling_backend = DelegatingTestBackend {
        backend: &backend,
        device: DeviceId::CPU,
        cancellation_on_reserve: Some(&materialization_cancellation),
        reserve_calls: AtomicUsize::new(0),
    };
    assert!(matches!(
        first.materialize(&cancelling_backend, &materialization_context),
        Err(QuantizationError::Cancelled)
    ));
    assert_eq!(cancelling_backend.reserve_calls.load(Ordering::Acquire), 1);
    assert_eq!(materialization_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(matches!(
        quantize_linear_matrix(
            QuantLinearLayout::TensorCoreFp8E4M3,
            DType::F32,
            &vec![1.0; 4_096],
            64,
            64,
            QuantLinearScale::Default,
            &cancelled,
        ),
        Err(QuantizationError::Cancelled)
    ));
    Ok(())
}

#[test]
fn fp8_backward_uses_quantized_matmuls_but_unquantized_bias_gradient() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let (input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let quantized_weight = quantize_linear_matrix(
        QuantLinearLayout::TensorCoreFp8E4M3,
        DType::F32,
        &values(&backend, &weight, &cancellation)?,
        2,
        3,
        QuantLinearScale::Default,
        &cancellation,
    )?;
    let mut execution = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Quantized(quantized_weight),
        Some(&bias),
        QuantLinearOptions {
            layout: Some(QuantLinearLayout::TensorCoreFp8E4M3),
            input_scale: QuantLinearScale::Default,
            compute_dtype: DType::F32,
            weight_requires_grad: false,
            fp8_backward: true,
        },
        &context(&backend, &cancellation)?,
    )?;
    let gradients = execution.backward(
        &*backend,
        &output_gradient,
        false,
        &context(&backend, &cancellation)?,
    )?;
    assert!(gradients.weight().is_none());
    close(
        &values(
            &backend,
            gradients.bias().ok_or("missing bias gradient")?,
            &cancellation,
        )?,
        &[1.5, 1.0],
        1.0e-6,
    );
    assert!(
        values(
            &backend,
            gradients.input().ok_or("missing input gradient")?,
            &cancellation,
        )?
        .iter()
        .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn all_layouts_execute_ordinary_and_optional_fp8_backward() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    for layout in [
        QuantLinearLayout::TensorCoreFp8E4M3,
        QuantLinearLayout::TensorCoreFp8E5M2,
        QuantLinearLayout::TensorCoreMxFp8,
        QuantLinearLayout::TensorCoreNvFp4,
    ] {
        for fp8_backward in [false, true] {
            let (input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
            let mut execution = quant_linear_forward_exact_native(
                &*backend,
                &input,
                QuantLinearWeight::Dense(weight),
                Some(&bias),
                QuantLinearOptions {
                    layout: Some(layout),
                    input_scale: QuantLinearScale::Default,
                    compute_dtype: DType::F32,
                    weight_requires_grad: true,
                    fp8_backward,
                },
                &context(&backend, &cancellation)?,
            )?;
            assert!(
                values(&backend, execution.output(), &cancellation)?
                    .iter()
                    .all(|value| value.is_finite())
            );
            let gradients = execution.backward(
                &*backend,
                &output_gradient,
                false,
                &context(&backend, &cancellation)?,
            )?;
            assert_eq!(gradients.input_arity(), 6);
            close(
                &values(
                    &backend,
                    gradients.bias().ok_or("missing bias gradient")?,
                    &cancellation,
                )?,
                &[1.5, 1.0],
                1.0e-6,
            );
            if !fp8_backward {
                close(
                    &values(
                        &backend,
                        gradients.weight().ok_or("missing weight gradient")?,
                        &cancellation,
                    )?,
                    &[1.0, -0.5, 0.0, -2.0, 13.0, -4.0],
                    1.0e-6,
                );
            }
        }
    }

    let (input, weight, _bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let mut execution = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight),
        None,
        QuantLinearOptions {
            weight_requires_grad: false,
            ..dense_options(DType::F32)
        },
        &context(&backend, &cancellation)?,
    )?;
    let gradients = execution.backward(
        &*backend,
        &output_gradient,
        false,
        &context(&backend, &cancellation)?,
    )?;
    assert!(gradients.weight().is_none());
    assert!(gradients.bias().is_none());
    Ok(())
}

#[test]
fn saved_tensor_versions_follow_the_source_fp8_cache_policy() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let (mut input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let mut ordinary = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight),
        Some(&bias),
        dense_options(DType::F32),
        &context(&backend, &cancellation)?,
    )?;
    let replacement = tensor(&backend, &[2, 3], &[0.0; 6], DType::F32, &cancellation)?;
    input.replace_data(replacement)?;
    assert!(matches!(
        ordinary.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::Autograd(AutogradBreadthError::Autograd(
            AutogradError::SavedTensorModified { .. }
        )))
    ));

    let (mut input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let mut fp8 = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight),
        Some(&bias),
        QuantLinearOptions {
            layout: None,
            input_scale: QuantLinearScale::Default,
            compute_dtype: DType::F32,
            weight_requires_grad: true,
            fp8_backward: true,
        },
        &context(&backend, &cancellation)?,
    )?;
    input.replace_data(tensor(
        &backend,
        &[2, 3],
        &[10.0; 6],
        DType::F32,
        &cancellation,
    )?)?;
    assert!(
        fp8.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancellation)?,
        )
        .is_ok()
    );

    for fp8_backward in [false, true] {
        let (input, mut weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
        let mut execution = quant_linear_forward_exact_native(
            &*backend,
            &input,
            QuantLinearWeight::Dense(weight.clone()),
            Some(&bias),
            QuantLinearOptions {
                fp8_backward,
                ..dense_options(DType::F32)
            },
            &context(&backend, &cancellation)?,
        )?;
        weight.replace_data(tensor(
            &backend,
            &[2, 3],
            &[0.0; 6],
            DType::F32,
            &cancellation,
        )?)?;
        assert!(matches!(
            execution.backward(
                &*backend,
                &output_gradient,
                false,
                &context(&backend, &cancellation)?,
            ),
            Err(QuantLinearError::Autograd(AutogradBreadthError::Autograd(
                AutogradError::SavedTensorModified { .. }
            )))
        ));
    }
    Ok(())
}

#[test]
fn once_differentiable_cancellation_and_shape_errors_fail_closed() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let (input, weight, bias, output_gradient) = base_tensors(&backend, &cancellation)?;
    let mut execution = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight.clone()),
        Some(&bias),
        dense_options(DType::F32),
        &context(&backend, &cancellation)?,
    )?;
    assert!(matches!(
        execution.backward(
            &*backend,
            &output_gradient,
            true,
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::OnceDifferentiable)
    ));

    let mut execution = quant_linear_forward_exact_native(
        &*backend,
        &input,
        QuantLinearWeight::Dense(weight.clone()),
        Some(&bias),
        dense_options(DType::F32),
        &context(&backend, &cancellation)?,
    )?;
    let cancelled_backward = CancellationToken::default();
    assert!(cancelled_backward.cancel());
    assert!(matches!(
        execution.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancelled_backward)?,
        ),
        Err(QuantLinearError::Cancelled)
    ));
    assert!(matches!(
        execution.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::Autograd(
            AutogradBreadthError::ReleasedContext
        ))
    ));

    let rank_one = tensor(&backend, &[3], &[1.0; 3], DType::F32, &cancellation)?;
    assert!(matches!(
        quant_linear_forward_exact_native(
            &*backend,
            &rank_one,
            QuantLinearWeight::Dense(weight.clone()),
            Some(&bias),
            dense_options(DType::F32),
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::InvalidInputRank)
    ));
    let invalid_dtype = tensor(&backend, &[2, 3], &[1.0; 6], DType::I32, &cancellation)?;
    assert!(matches!(
        quant_linear_forward_exact_native(
            &*backend,
            &invalid_dtype,
            QuantLinearWeight::Dense(weight.clone()),
            Some(&bias),
            dense_options(DType::F32),
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::UnsupportedDType { dtype: DType::I32 })
    ));
    let mismatched_weight = tensor(&backend, &[2, 2], &[1.0; 4], DType::F32, &cancellation)?;
    assert!(matches!(
        quant_linear_forward_exact_native(
            &*backend,
            &input,
            QuantLinearWeight::Dense(mismatched_weight),
            Some(&bias),
            dense_options(DType::F32),
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::ShapeMismatch)
    ));
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(matches!(
        quant_linear_forward_exact_native(
            &*backend,
            &input,
            QuantLinearWeight::Dense(weight),
            Some(&bias),
            dense_options(DType::F32),
            &context(&backend, &cancelled)?,
        ),
        Err(QuantLinearError::Cancelled)
    ));
    Ok(())
}

#[test]
fn compute_dtype_and_forward_device_rejections_precede_adapter_effects()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let (input, weight, bias, _) = base_tensors(&backend, &cancellation)?;
    assert!(matches!(
        quant_linear_forward_exact_native(
            &*backend,
            &input,
            QuantLinearWeight::Dense(weight.clone()),
            Some(&bias),
            dense_options(DType::I32),
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::UnsupportedDType { dtype: DType::I32 })
    ));

    let mismatched_backend = DelegatingTestBackend {
        backend: &backend,
        device: DeviceId::new(DeviceKind::Metal, 0),
        cancellation_on_reserve: None,
        reserve_calls: AtomicUsize::new(0),
    };
    assert!(matches!(
        quant_linear_forward_exact_native(
            &mismatched_backend,
            &input,
            QuantLinearWeight::Dense(weight),
            Some(&bias),
            dense_options(DType::F32),
            &context(&backend, &cancellation)?,
        ),
        Err(QuantLinearError::UnsupportedDevice {
            device: DeviceId::CPU
        })
    ));
    assert_eq!(mismatched_backend.reserve_calls.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn native_module_adapter_reuses_canonical_quantized_parameter_storage() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let weight = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[2.0, -1.0, 0.25, -0.5, 1.0, 2.0],
        2,
        3,
        &cancellation,
    )?;
    let bias = tensor(&backend, &[2], &[0.5, -0.25], DType::F32, &cancellation)?;
    let mut module = NativeModule::linear("adapter", 3, 2, true, false)?;
    module.load_quantized_linear_parameters(weight, Some(bias))?;
    let input = tensor(
        &backend,
        &[1, 3],
        &[1.0, -2.0, 0.5],
        DType::F32,
        &cancellation,
    )?;
    let execution = module.forward_quantized_autograd_with_context(
        &*backend,
        &input,
        QuantLinearLayout::TensorCoreFp8E4M3,
        QuantLinearScale::Default,
        false,
        true,
        &context(&backend, &cancellation)?,
    )?;
    close(
        &values(&backend, execution.output(), &cancellation)?,
        &[4.625, -1.75],
        0.04,
    );
    Ok(())
}

fn oracle_dtype(name: &str) -> Result<DType, Box<dyn Error>> {
    match name {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::Bf16),
        "f32" => Ok(DType::F32),
        _ => Err(std::io::Error::other(format!("unsupported oracle compute dtype {name}")).into()),
    }
}

fn close_oracle_case(case_id: &str, field: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "oracle case {case_id} {field} length"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1.0e-5,
            "oracle case {case_id} {field}[{index}]: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn val_task102_pinned_quant_linear_source_oracle() -> Result<(), Box<dyn Error>> {
    let oracle: QuantLinearOracle = serde_json::from_slice(ORACLE_FIXTURE)?;
    assert_eq!(oracle.schema_version, 4);
    assert_eq!(
        oracle.owner_task_id,
        "comfy-parity-quantized-autograd-adapter"
    );
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut executed_case_ids = BTreeSet::new();

    for case in &oracle.execution_cases {
        assert!(
            executed_case_ids.insert(case.id.as_str()),
            "duplicate oracle case {}",
            case.id
        );
        let input = tensor(
            &backend,
            &oracle.fixture_inputs.input_shape,
            &oracle.fixture_inputs.input,
            DType::F32,
            &cancellation,
        )?;
        let bias = tensor(
            &backend,
            &[u64::try_from(oracle.fixture_inputs.bias.len())?],
            &oracle.fixture_inputs.bias,
            DType::F32,
            &cancellation,
        )?;
        let output_gradient = tensor(
            &backend,
            &oracle.fixture_inputs.output_gradient_shape,
            &oracle.fixture_inputs.output_gradient,
            DType::F32,
            &cancellation,
        )?;
        let compute_dtype = oracle_dtype(&case.compute_dtype)?;
        assert!(case.weight_scale.is_default());
        assert_eq!(case.weight_requires_grad, case.weight_layout.is_none());
        assert_eq!(
            case.weight_runtime_type,
            if case.weight_layout.is_some() {
                "comfy_kitchen.tensor.base.QuantizedTensor"
            } else {
                "torch.Tensor"
            }
        );
        let native_weight = if let Some(weight_layout) = case.weight_layout.as_deref() {
            let layout = QuantLinearLayout::from_source_name(weight_layout).ok_or_else(|| {
                std::io::Error::other(format!("unsupported oracle weight layout {weight_layout}"))
            })?;
            QuantLinearWeight::Quantized(quantize_linear_matrix(
                layout,
                compute_dtype,
                &oracle.fixture_inputs.weight,
                usize::try_from(oracle.fixture_inputs.weight_shape[0])?,
                usize::try_from(oracle.fixture_inputs.weight_shape[1])?,
                case.weight_scale.quant_linear_scale(),
                &cancellation,
            )?)
        } else {
            QuantLinearWeight::Dense(tensor(
                &backend,
                &oracle.fixture_inputs.weight_shape,
                &oracle.fixture_inputs.weight,
                DType::F32,
                &cancellation,
            )?)
        };
        let options = QuantLinearOptions::from_source_layout(
            case.source_layout.as_deref(),
            case.input_scale.quant_linear_scale(),
            compute_dtype,
            case.weight_requires_grad,
            case.fp8_backward,
        )?;
        let mut execution = quant_linear_forward_exact_native(
            &*backend,
            &input,
            native_weight,
            Some(&bias),
            options,
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(execution.output().descriptor().dtype(), compute_dtype);
        assert_eq!(case.output_dtype, case.compute_dtype);
        close_oracle_case(
            &case.id,
            "output",
            &values(&backend, execution.output(), &cancellation)?,
            &case.output,
        );

        let gradients = execution.backward(
            &*backend,
            &output_gradient,
            false,
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(gradients.input_arity(), 6);
        assert!(gradients.as_slice()[3..].iter().all(Option::is_none));
        let input_gradient = gradients.input().ok_or("missing input gradient")?;
        let bias_gradient = gradients.bias().ok_or("missing bias gradient")?;
        assert_eq!(input_gradient.descriptor().dtype(), compute_dtype);
        assert_eq!(bias_gradient.descriptor().dtype(), compute_dtype);
        assert_eq!(
            case.gradient_dtypes,
            vec![
                Some(case.compute_dtype.clone()),
                case.weight_requires_grad
                    .then(|| case.compute_dtype.clone()),
                Some(case.compute_dtype.clone()),
            ]
        );
        close_oracle_case(
            &case.id,
            "input_gradient",
            &values(&backend, input_gradient, &cancellation)?,
            &case.input_gradient,
        );
        match (gradients.weight(), case.weight_gradient.as_deref()) {
            (Some(weight_gradient), Some(expected)) => {
                assert_eq!(weight_gradient.descriptor().dtype(), compute_dtype);
                close_oracle_case(
                    &case.id,
                    "weight_gradient",
                    &values(&backend, weight_gradient, &cancellation)?,
                    expected,
                );
            }
            (None, None) => {}
            _ => panic!("oracle case {} weight-gradient presence differs", case.id),
        }
        close_oracle_case(
            &case.id,
            "bias_gradient",
            &values(&backend, bias_gradient, &cancellation)?,
            &case.bias_gradient,
        );
    }

    let expected_case_ids = BTreeSet::from([
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
    ]);
    assert_eq!(executed_case_ids, expected_case_ids);

    let unsupported_source_name = "TensorCoreINT4Layout";
    assert!(matches!(
        QuantLinearOptions::from_source_layout(
            Some(unsupported_source_name),
            QuantLinearScale::Default,
            DType::F32,
            true,
            false,
        ),
        Err(QuantLinearError::UnsupportedLayout { source_name })
            if source_name == unsupported_source_name
    ));
    Ok(())
}
