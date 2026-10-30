use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, BinaryOperation, CachedAllocationOwner, CancellationToken,
    ConvolutionSpec, CpuBackend, CpuWorkspaceAuthority, CustomKernelId, DType, DeviceId,
    EventFence, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, IndexSpec,
    LinearAlgebraOperation, NativeDeviceProperties, NativeStreamRegistry, OperationSupport,
    ReductionSpec, ResizeSpec, Scalar, ScalarSide, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, UnaryOperation,
    generated_elementwise_or_runtime_operation_13::softmax_function_with_context_exact_native,
    generated_elementwise_or_runtime_operation_20::{
        ElementwiseRuntimePartTwentyError, broadcast_tensor_jvp_exact_native,
        broadcast_tensor_vjp_with_context_exact_native as broadcast_tensor_vjp_exact_native,
        broadcast_tensors_exact_native,
        cross_jvp_with_context_exact_native as cross_jvp_exact_native,
        cross_vjp_with_context_exact_native as cross_vjp_exact_native,
        cross_with_context_exact_native as cross_exact_native, cross_with_context_exact_native,
        cuda_stream_exact_native, cuda_synchronize_exact_native, directml_device_name_exact_native,
        flip_jvp_with_context_exact_native, flip_vjp_with_context_exact_native,
        flip_with_context_exact_native,
        int_method_with_context_exact_native as int_method_exact_native,
        softmax_method_jvp_with_context_exact_native, softmax_method_vjp_with_context_exact_native,
        softmax_method_with_context_exact_native, swapaxes_exact_native, swapaxes_jvp_exact_native,
        swapaxes_vjp_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-DDAAD49116D0",
    "COMFY-TENSOR-OP-D5D333C89A34",
    "COMFY-TENSOR-OP-E430CCED2202",
    "COMFY-TENSOR-OP-D54F52B18FB1",
    "COMFY-TENSOR-OP-E644686B4E0F",
    "COMFY-TENSOR-OP-D6F57272FC58",
    "COMFY-TENSOR-OP-E237E236E06A",
    "COMFY-TENSOR-OP-D718FC279D3E",
    "COMFY-TENSOR-OP-E07BBEBA226B",
    "COMFY-TENSOR-OP-D54AF27B4D70",
    "COMFY-TENSOR-OP-E0C529F06769",
    "COMFY-TENSOR-OP-E01C0CE81BB1",
];

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

fn authorized_context<'a>(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, sim_tensor::TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
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

fn values(
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[track_caller]
fn assert_cancelled<T>(result: Result<T, ElementwiseRuntimePartTwentyError>) {
    assert!(matches!(
        result,
        Err(ElementwiseRuntimePartTwentyError::Cancelled)
    ));
}

#[test]
fn workspace_cross_has_exact_peak_underauthorization_and_cancel_atomicity()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let left = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &cancellation,
    )?;
    let right = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &cancellation,
    )?;

    let scratch = workspace_authority.authorize_workspace(24)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let output = cross_with_context_exact_native(&backend, &left, &right, Some(1), &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    );
    assert_eq!(scratch.peak_bytes(), 24);
    assert_eq!(scratch.in_use_bytes(), 0);

    let too_small = workspace_authority.authorize_workspace(23)?;
    let execution = backend.execution_context(StreamId::DEFAULT, too_small.clone(), &cancellation);
    assert!(matches!(
        cross_with_context_exact_native(&backend, &left, &right, Some(1), &execution),
        Err(sim_tensor::generated_elementwise_or_runtime_operation_20::ElementwiseRuntimePartTwentyError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(too_small.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_scratch = workspace_authority.authorize_workspace(24)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, cancelled_scratch.clone(), &cancelled);
    assert!(cross_with_context_exact_native(&backend, &left, &right, Some(1), &execution).is_err());
    assert_eq!(cancelled_scratch.peak_bytes(), 0);
    assert_eq!(cancelled_scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task_63_resolution_slice_seals_all_twelve_unique_contracts() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_20")
        .ok_or("Task 63 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    assert_eq!(
        slice
            .contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root")?;
    let mut overloads = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(overloads.insert(contract.overload_id));
        assert!(digests.insert(contract.evidence_fixture_sha256));
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}

#[test]
fn int_and_softmax_delegate_their_canonical_owners() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.9, -2.9, 0.0, 1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let integer = int_method_exact_native(
        &backend,
        &input,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(integer.descriptor().dtype(), DType::I32);
    assert_close(
        &values(&backend, &workspace_authority, &integer, &cancellation)?,
        &[1.0, -2.0, 0.0, 1.0, 2.0, 3.0],
    );

    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let adapted = softmax_method_with_context_exact_native(&backend, &input, -1, &execution)?;
    let canonical = softmax_function_with_context_exact_native(&backend, &input, -1, &execution)?;
    assert_eq!(
        adapted.host_storage_bytes()?,
        canonical.host_storage_bytes()?
    );
    Ok(())
}

#[test]
fn broadcast_views_preserve_aliases_and_inverse_gradient_geometry() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let left = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let right = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[10.0, 20.0],
        &cancellation,
    )?;
    let mut outputs = broadcast_tensors_exact_native(&[left.clone(), right], &cancellation)?;
    assert_eq!(outputs[0].descriptor().shape(), [2, 3]);
    assert_eq!(outputs[0].descriptor().strides(), [0, 1]);
    assert_eq!(outputs[0].storage_id(), left.storage_id());
    assert!(outputs[0].write().is_err());
    assert_close(
        &values(&backend, &workspace_authority, &outputs[0], &cancellation)?,
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    );

    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let reduced = broadcast_tensor_vjp_exact_native(
        &backend,
        &left,
        &gradient,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &reduced, &cancellation)?,
        &[5.0, 7.0, 9.0],
    );
    let mut tangent = broadcast_tensor_jvp_exact_native(&left, &[2, 3], &cancellation)?;
    assert_eq!(tangent.storage_id(), left.storage_id());
    assert!(tangent.write().is_err());
    assert_close(
        &values(&backend, &workspace_authority, &tangent, &cancellation)?,
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    );
    Ok(())
}

#[test]
fn cross_forward_vjp_and_jvp_use_right_handed_equations() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 0.0, 0.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, 1.0, 0.0],
        &cancellation,
    )?;
    let output = cross_exact_native(
        &backend,
        &input,
        &other,
        None,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[0.0, 0.0, 1.0],
    );
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, 0.0, 1.0],
        &cancellation,
    )?;
    let (input_gradient, other_gradient) = cross_vjp_exact_native(
        &backend,
        &input,
        &other,
        &gradient,
        None,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &input_gradient,
            &cancellation,
        )?,
        &[1.0, 0.0, 0.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &other_gradient,
            &cancellation,
        )?,
        &[0.0, 1.0, 0.0],
    );
    let input_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, 1.0, 0.0],
        &cancellation,
    )?;
    let other_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, 0.0, 1.0],
        &cancellation,
    )?;
    let tangent = cross_jvp_exact_native(
        &backend,
        &input,
        &other,
        &input_tangent,
        &other_tangent,
        None,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &tangent, &cancellation)?,
        &[0.0, -1.0, 0.0],
    );
    Ok(())
}

#[test]
fn flip_and_swapaxes_preserve_canonical_copy_and_view_semantics() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(24)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let flipped = flip_with_context_exact_native(&backend, &input, &[0, 1], &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &flipped, &cancellation)?,
        &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    );
    assert_ne!(flipped.storage_id(), input.storage_id());

    let mut swapped = swapaxes_exact_native(&input, 0, 1, &cancellation)?;
    assert_eq!(swapped.descriptor().shape(), [3, 2]);
    assert_eq!(swapped.descriptor().strides(), [1, 3]);
    assert_eq!(swapped.storage_id(), input.storage_id());
    assert!(swapped.write().is_err());
    assert_close(
        &values(&backend, &workspace_authority, &swapped, &cancellation)?,
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
    let inverse = swapaxes_vjp_exact_native(&swapped, 0, 1, &cancellation)?;
    let tangent = swapaxes_jvp_exact_native(&input, 0, 1, &cancellation)?;
    assert_eq!(inverse.descriptor().shape(), input.descriptor().shape());
    assert_eq!(tangent.descriptor(), swapped.descriptor());
    Ok(())
}

struct EventOnlyBackend {
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    event_owner: CpuBackend,
}

impl CachedAllocationOwner for EventOnlyBackend {
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

impl TensorBackend for EventOnlyBackend {
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
fn device_stream_name_and_synchronize_are_thin_canonical_adapters() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let cuda = DeviceId::new(DeviceKind::Cuda, 1);
    let capabilities = BackendCapabilityMatrix::new(
        cuda,
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
    )?;
    let registry = NativeStreamRegistry::default();
    let stream = cuda_stream_exact_native(&registry, &capabilities, cuda, -2, &cancellation)?;
    assert_eq!(stream.device(), cuda);
    assert_eq!(stream.priority(), -2);
    let (event_owner, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let backend = EventOnlyBackend {
        device: cuda,
        capabilities: capabilities.clone(),
        event_owner,
    };
    cuda_synchronize_exact_native(
        &backend,
        &capabilities,
        &context(&backend.event_owner, &workspace_authority, &cancellation)?,
    )?;

    let directml = DeviceId::new(DeviceKind::DirectMl, 4);
    let properties =
        NativeDeviceProperties::new(directml, "DirectML Native 4", 1024, 1, 0, None, true)?;
    let directml_capabilities = BackendCapabilityMatrix::new_with_properties(
        directml,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        directml_device_name_exact_native(&directml_capabilities, directml, &cancellation)?,
        "DirectML Native 4"
    );
    assert!(directml_device_name_exact_native(&capabilities, cuda, &cancellation).is_err());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(cuda_stream_exact_native(&registry, &capabilities, cuda, 0, &cancelled).is_err());
    assert!(
        cuda_synchronize_exact_native(
            &backend,
            &capabilities,
            &context(&backend.event_owner, &workspace_authority, &cancelled)?,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn task_63_every_public_tensor_adapter_observes_cancellation_before_validation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &live,
    )?;
    let mismatched = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &live)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = authorized_context(&backend, &workspace_authority, &cancelled)?;

    assert_cancelled(int_method_exact_native(&backend, &input, &execution));
    assert_cancelled(softmax_method_with_context_exact_native(
        &backend, &input, 4, &execution,
    ));
    assert_cancelled(softmax_method_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        4,
        &execution,
    ));
    assert_cancelled(softmax_method_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        4,
        &execution,
    ));

    assert_cancelled(broadcast_tensors_exact_native(&[], &cancelled));
    assert_cancelled(broadcast_tensor_vjp_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(broadcast_tensor_jvp_exact_native(
        &input,
        &[2, 2],
        &cancelled,
    ));

    assert_cancelled(cross_exact_native(
        &backend,
        &input,
        &mismatched,
        Some(8),
        &execution,
    ));
    assert_cancelled(cross_vjp_exact_native(
        &backend,
        &input,
        &mismatched,
        &mismatched,
        Some(8),
        &execution,
    ));
    assert_cancelled(cross_jvp_exact_native(
        &backend,
        &input,
        &mismatched,
        &mismatched,
        &input,
        Some(8),
        &execution,
    ));

    assert_cancelled(flip_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));
    assert_cancelled(flip_vjp_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));
    assert_cancelled(flip_jvp_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));
    assert_cancelled(swapaxes_exact_native(&input, 0, 9, &cancelled));
    assert_cancelled(swapaxes_vjp_exact_native(&input, 0, 9, &cancelled));
    assert_cancelled(swapaxes_jvp_exact_native(&input, 0, 9, &cancelled));

    let cpu_capabilities = BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?;
    let registry = NativeStreamRegistry::default();
    assert_cancelled(cuda_stream_exact_native(
        &registry,
        &cpu_capabilities,
        DeviceId::CPU,
        0,
        &cancelled,
    ));
    assert_cancelled(directml_device_name_exact_native(
        &cpu_capabilities,
        DeviceId::CPU,
        &cancelled,
    ));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let cuda_capabilities = BackendCapabilityMatrix::new(
        cuda,
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
        vec![
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ],
    )?;
    let event_backend = EventOnlyBackend {
        device: cuda,
        capabilities: cuda_capabilities.clone(),
        event_owner: backend,
    };
    assert_cancelled(cuda_synchronize_exact_native(
        &event_backend,
        &cuda_capabilities,
        &execution,
    ));
    let first = cuda_stream_exact_native(&registry, &cuda_capabilities, cuda, 0, &live)?;
    assert_eq!(first.id().get(), 1);
    Ok(())
}
