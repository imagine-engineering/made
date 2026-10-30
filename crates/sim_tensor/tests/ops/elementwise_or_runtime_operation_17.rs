use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    AutogradTape, BackendCapabilityMatrix, BinaryOperation, CachedAllocationOwner,
    CancellationToken, ConvolutionSpec, CpuBackend, CpuWorkspaceAuthority, CustomKernelId, DType,
    DeviceId, EventFence, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    GradientMode, IndexSpec, LeafId, LinearAlgebraOperation, OperationSupport, ReductionSpec,
    ResizeSpec, Scalar, ScalarSide, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    UnaryOperation,
    generated_elementwise_or_runtime_operation_17::{
        ElementwiseRuntimePartSeventeenError, TensorSplitSpec, atanh_jvp_with_context_exact_native,
        atanh_vjp_with_context_exact_native, atanh_with_context_exact_native,
        clip_jvp_with_context_exact_native, clip_vjp_with_context_exact_native,
        clip_with_context_exact_native, requires_grad_method_exact_native,
        roll_jvp_with_context_exact_native, roll_vjp_with_context_exact_native,
        roll_with_context_exact_native, tensor_split_exact_native, tensor_split_jvp_exact_native,
        tensor_split_vjp_with_context_exact_native, xpu_synchronize_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-BDAC082E9091",
    "COMFY-TENSOR-OP-BA7930023140",
    "COMFY-TENSOR-OP-BB442559BFF4",
    "COMFY-TENSOR-OP-BBE4FD70D20E",
    "COMFY-TENSOR-OP-BF7BF3AA74D7",
    "COMFY-TENSOR-OP-BB1114038F65",
    "COMFY-TENSOR-OP-B96B1B025618",
    "COMFY-TENSOR-OP-BD0C27F1B551",
    "COMFY-TENSOR-OP-BE1F415B5A74",
    "COMFY-TENSOR-OP-BF0B50BCC3B4",
    "COMFY-TENSOR-OP-B91A910A5AF9",
];

const EXTERNAL_NUMPY_ROLL_ID: &str = "COMFY-TENSOR-OP-BE67DCC5B9C6";

#[test]
fn part_seventeen_workspace_is_exact_bounded_and_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_i64(
        &backend,
        &workspace_authority,
        &[4],
        &[1, 2, 3, 4],
        &cancellation,
    )?;
    let bytes = 4 * u64::try_from(std::mem::size_of::<i64>())?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancellation,
    );
    roll_with_context_exact_native(&backend, &input, &[1], Some(&[0]), &exact)?;
    assert_eq!(exact.scratch.peak_bytes(), bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes - 1)?,
        &cancellation,
    );
    assert!(
        roll_with_context_exact_native(&backend, &input, &[1], Some(&[0]), &insufficient).is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancelled,
    );
    assert!(
        roll_with_context_exact_native(&backend, &input, &[1], Some(&[0]), &cancelled_context)
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn context<'a>(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, sim_tensor::TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        cancellation,
    ))
}

fn upload_f32(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
}

fn upload_i64(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
}

fn f32_values(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let byte_count = tensor
        .descriptor()
        .element_count()?
        .checked_mul(4)
        .ok_or("tensor-to-f32 workspace overflow")?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(byte_count)?,
        cancellation,
    );
    Ok(tensor_to_f32_with_context_exact_native(
        backend, tensor, &execution,
    )?)
}

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn Error>> {
    let count = tensor.descriptor().element_count()?;
    (0..count)
        .map(|index| {
            let bytes: [u8; 8] = tensor
                .element_bytes(&[index])?
                .try_into()
                .map_err(|_| "invalid I64 element width")?;
            Ok(i64::from_ne_bytes(bytes))
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn resolution_slice_seals_only_executable_contracts_and_external_roll_stays_unclaimed()
-> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_17")
        .ok_or("Task 60 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    assert_eq!(
        slice
            .contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect::<BTreeSet<_>>()
    );
    assert!(
        slice
            .contracts
            .iter()
            .all(|contract| contract.operation_id != EXTERNAL_NUMPY_ROLL_ID)
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let mut overloads = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(overloads.insert(contract.overload_id));
        assert!(digests.insert(contract.evidence_fixture_sha256));
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            contract.evidence_fixture_sha256
        );
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            fixture["operation_id"].as_str(),
            Some(contract.operation_id)
        );
        assert_eq!(fixture["overload_id"].as_str(), Some(contract.overload_id));
    }

    let disposition = fs::read_to_string(
        workspace.join(".agents/specs/comfy-parity/catalogs/backend-tensor-operations.csv"),
    )?;
    let row = disposition
        .lines()
        .find(|line| line.starts_with(EXTERNAL_NUMPY_ROLL_ID))
        .ok_or("external Tensor.roll disposition is missing")?;
    assert!(row.contains("numpy.roll"));
    Ok(())
}

#[test]
fn requires_grad_mutates_only_the_caller_owned_autograd_tape() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[1], &[2.0], &cancellation)?;
    let leaf = LeafId::new("task-60-leaf")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let returned = requires_grad_method_exact_native(
        &mut tape,
        &input,
        Some(leaf.clone()),
        true,
        &cancellation,
    )?;
    assert_eq!(returned.storage_id(), input.storage_id());
    assert!(tape.requires_grad(&input));
    assert_eq!(tape.leaf_binding(&input), Some(&leaf));
    assert!(
        requires_grad_method_exact_native(
            &mut tape,
            &input,
            Some(LeafId::new("different-leaf")?),
            true,
            &cancellation,
        )
        .is_err()
    );
    requires_grad_method_exact_native(&mut tape, &input, None, false, &cancellation)?;
    assert!(!tape.requires_grad(&input));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        requires_grad_method_exact_native(&mut tape, &input, Some(leaf), true, &cancelled,)
            .is_err()
    );
    assert!(!tape.requires_grad(&input));
    Ok(())
}

#[test]
fn atanh_and_clip_reuse_canonical_primitives_with_analytical_maps() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[-0.5, 0.0, 0.5],
        &cancellation,
    )?;
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &atanh_with_context_exact_native(&backend, &input, &execution)?,
            &cancellation,
        )?,
        &[(-0.5_f32).atanh(), 0.0, 0.5_f32.atanh()],
    );
    let expected_gradient = [4.0 / 3.0, 2.0, 4.0];
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &atanh_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &expected_gradient,
    );
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &atanh_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &expected_gradient,
    );

    let clipped =
        clip_with_context_exact_native(&backend, &tangent, Some(1.5), Some(2.5), &execution)?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &clipped, &cancellation)?,
        &[1.5, 2.0, 2.5],
    );
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let expected_clip_gradient = [0.0, 5.0, 0.0];
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &clip_vjp_with_context_exact_native(
                &backend,
                &tangent,
                Some(1.5),
                Some(2.5),
                &output_gradient,
                &execution,
            )?,
            &cancellation,
        )?,
        &expected_clip_gradient,
    );
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &clip_jvp_with_context_exact_native(
                &backend,
                &tangent,
                Some(1.5),
                Some(2.5),
                &output_gradient,
                &execution,
            )?,
            &cancellation,
        )?,
        &expected_clip_gradient,
    );
    Ok(())
}

#[test]
fn roll_is_byte_preserving_and_its_derivative_is_the_inverse_permutation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let integers = upload_i64(
        &backend,
        &workspace_authority,
        &[5],
        &[1, 2, 3, 4, 5],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let rolled = roll_with_context_exact_native(&backend, &integers, &[2], Some(&[0]), &execution)?;
    assert_eq!(i64_values(&rolled)?, vec![4, 5, 1, 2, 3]);

    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let rolled =
        roll_with_context_exact_native(&backend, &input, &[1, -1], Some(&[0, 1]), &execution)?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &rolled, &cancellation)?,
        &[5.0, 6.0, 4.0, 2.0, 3.0, 1.0],
    );
    let restored =
        roll_vjp_with_context_exact_native(&backend, &rolled, &[1, -1], Some(&[0, 1]), &execution)?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &restored, &cancellation)?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &roll_jvp_with_context_exact_native(&backend, &input, &[2], None, &execution)?,
            &cancellation,
        )?,
        &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0],
    );
    Ok(())
}

#[test]
fn tensor_split_returns_read_only_views_and_scatter_gathers_gradients() -> Result<(), Box<dyn Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 5],
        &(0..10).map(|v| v as f32).collect::<Vec<_>>(),
        &cancellation,
    )?;
    let mut outputs =
        tensor_split_exact_native(&input, &TensorSplitSpec::Sections(3), 1, &cancellation)?;
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.descriptor().shape())
            .collect::<Vec<_>>(),
        vec![&[2, 2][..], &[2, 2][..], &[2, 1][..]]
    );
    assert!(
        outputs
            .iter()
            .all(|output| output.storage_id() == input.storage_id())
    );
    assert!(outputs.iter_mut().all(|output| output.write().is_err()));
    assert_close(
        &f32_values(&backend, &workspace_authority, &outputs[0], &cancellation)?,
        &[0.0, 1.0, 5.0, 6.0],
    );

    let explicit_sizes = tensor_split_exact_native(
        &input,
        &TensorSplitSpec::Sizes(vec![1, 0, 4]),
        1,
        &cancellation,
    )?;
    assert_eq!(
        explicit_sizes
            .iter()
            .map(|output| output.descriptor().shape())
            .collect::<Vec<_>>(),
        vec![&[2, 1][..], &[2, 0][..], &[2, 4][..]]
    );
    assert!(
        tensor_split_exact_native(
            &input,
            &TensorSplitSpec::Sizes(vec![1, 3]),
            1,
            &cancellation,
        )
        .is_err()
    );

    let tangent_outputs = tensor_split_jvp_exact_native(
        &input,
        &TensorSplitSpec::Indices(vec![2, 4]),
        1,
        &cancellation,
    )?;
    assert_eq!(tangent_outputs.len(), 3);
    let gradients = vec![
        upload_f32(
            &backend,
            &workspace_authority,
            &[2, 2],
            &[1.0; 4],
            &cancellation,
        )?,
        upload_f32(
            &backend,
            &workspace_authority,
            &[2, 2],
            &[2.0; 4],
            &cancellation,
        )?,
        upload_f32(
            &backend,
            &workspace_authority,
            &[2, 1],
            &[3.0; 2],
            &cancellation,
        )?,
    ];
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let combined = tensor_split_vjp_with_context_exact_native(
        &backend,
        &input,
        &gradients,
        &TensorSplitSpec::Indices(vec![2, 4]),
        1,
        &execution,
    )?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &combined, &cancellation)?,
        &[1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    );
    Ok(())
}

struct XpuEventBackend {
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    event_owner: CpuBackend,
}

impl CachedAllocationOwner for XpuEventBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

fn unused_backend_operation<T>() -> Result<T, TensorError> {
    Err(TensorError::Faulted {
        reason: "fixture backend exposes only event synchronization".to_owned(),
    })
}

impl TensorBackend for XpuEventBackend {
    fn device(&self) -> DeviceId {
        self.device
    }
    fn capabilities(&self) -> &BackendCapabilityMatrix {
        &self.capabilities
    }
    fn allocate(
        &self,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn copy(
        &self,
        _: &Tensor,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        self.event_owner.record_event(context)
    }
    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.event_owner.wait_event(event, context)
    }
    fn fill(
        &self,
        _: Scalar,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn unary(
        &self,
        _: UnaryOperation,
        _: &Tensor,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn binary(
        &self,
        _: BinaryOperation,
        _: &Tensor,
        _: &Tensor,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn binary_scalar(
        &self,
        _: BinaryOperation,
        _: &Tensor,
        _: Scalar,
        _: ScalarSide,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn reduction(
        &self,
        _: &ReductionSpec,
        _: &Tensor,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn indexing(
        &self,
        _: &IndexSpec,
        _: &[Tensor],
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn resize(
        &self,
        _: ResizeSpec,
        _: &Tensor,
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn convolution(
        &self,
        _: &ConvolutionSpec,
        _: &[Tensor],
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn linear_algebra(
        &self,
        _: LinearAlgebraOperation,
        _: &[Tensor],
        _: TensorDescriptor,
        _: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        unused_backend_operation()
    }
    fn custom_kernel(
        &self,
        _: &CustomKernelId,
        _: &[Tensor],
        _: &[TensorDescriptor],
        _: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        unused_backend_operation()
    }
}

#[test]
fn xpu_synchronize_delegates_to_capabilities_context_and_backend_events()
-> Result<(), Box<dyn Error>> {
    let device = DeviceId::new(DeviceKind::Xpu, 0);
    let capabilities = BackendCapabilityMatrix::new(
        device,
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
    )?;
    let (event_owner, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let backend = XpuEventBackend {
        device,
        capabilities: capabilities.clone(),
        event_owner,
    };
    let cancellation = CancellationToken::default();
    xpu_synchronize_exact_native(
        &backend,
        &capabilities,
        &context(&backend.event_owner, &workspace_authority, &cancellation)?,
    )?;

    let missing = BackendCapabilityMatrix::new(device, vec![], vec![])?;
    assert!(
        xpu_synchronize_exact_native(
            &backend,
            &missing,
            &context(&backend.event_owner, &workspace_authority, &cancellation)?,
        )
        .is_err()
    );
    let wrong = BackendCapabilityMatrix::new(DeviceId::CPU, vec![], vec![])?;
    assert!(
        xpu_synchronize_exact_native(
            &backend,
            &wrong,
            &context(&backend.event_owner, &workspace_authority, &cancellation)?,
        )
        .is_err()
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        xpu_synchronize_exact_native(
            &backend,
            &capabilities,
            &context(&backend.event_owner, &workspace_authority, &cancelled)?,
        )
        .is_err()
    );
    Ok(())
}

fn require_task60_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartSeventeenError>,
) -> Result<(), Box<dyn Error>> {
    if matches!(result, Err(ElementwiseRuntimePartSeventeenError::Cancelled)) {
        Ok(())
    } else {
        Err("Task 60 public adapter did not give cancellation precedence".into())
    }
}

#[test]
fn every_public_task60_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let active = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[1], &[0.25], &active)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(&backend, &workspace_authority, &cancelled)?;

    let mut tape = AutogradTape::new(GradientMode::Enabled);
    require_task60_cancelled(requires_grad_method_exact_native(
        &mut tape,
        &input,
        Some(LeafId::new("task-60-cancelled-leaf")?),
        true,
        &cancelled,
    ))?;
    assert!(!tape.requires_grad(&input));

    require_task60_cancelled(atanh_with_context_exact_native(
        &backend,
        &input,
        &cancelled_context,
    ))?;
    require_task60_cancelled(atanh_vjp_with_context_exact_native(
        &backend,
        &input,
        &input,
        &cancelled_context,
    ))?;
    require_task60_cancelled(atanh_jvp_with_context_exact_native(
        &backend,
        &input,
        &input,
        &cancelled_context,
    ))?;
    require_task60_cancelled(clip_with_context_exact_native(
        &backend,
        &input,
        Some(2.0),
        Some(1.0),
        &cancelled_context,
    ))?;
    require_task60_cancelled(clip_vjp_with_context_exact_native(
        &backend,
        &input,
        Some(2.0),
        Some(1.0),
        &input,
        &cancelled_context,
    ))?;
    require_task60_cancelled(clip_jvp_with_context_exact_native(
        &backend,
        &input,
        Some(2.0),
        Some(1.0),
        &input,
        &cancelled_context,
    ))?;
    require_task60_cancelled(roll_with_context_exact_native(
        &backend,
        &input,
        &[],
        Some(&[99]),
        &cancelled_context,
    ))?;
    require_task60_cancelled(roll_vjp_with_context_exact_native(
        &backend,
        &input,
        &[],
        Some(&[99]),
        &cancelled_context,
    ))?;
    require_task60_cancelled(roll_jvp_with_context_exact_native(
        &backend,
        &input,
        &[],
        Some(&[99]),
        &cancelled_context,
    ))?;
    require_task60_cancelled(tensor_split_exact_native(
        &input,
        &TensorSplitSpec::Sections(0),
        99,
        &cancelled,
    ))?;
    require_task60_cancelled(tensor_split_vjp_with_context_exact_native(
        &backend,
        &input,
        &[],
        &TensorSplitSpec::Sections(0),
        99,
        &cancelled_context,
    ))?;
    require_task60_cancelled(tensor_split_jvp_exact_native(
        &input,
        &TensorSplitSpec::Sections(0),
        99,
        &cancelled,
    ))?;

    let xpu = DeviceId::new(DeviceKind::Xpu, 0);
    let valid_capabilities = BackendCapabilityMatrix::new(
        xpu,
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
    )?;
    let xpu_backend = XpuEventBackend {
        device: xpu,
        capabilities: valid_capabilities,
        event_owner: backend,
    };
    let invalid_capabilities = BackendCapabilityMatrix::new(DeviceId::CPU, vec![], vec![])?;
    require_task60_cancelled(xpu_synchronize_exact_native(
        &xpu_backend,
        &invalid_capabilities,
        &cancelled_context,
    ))?;
    Ok(())
}
