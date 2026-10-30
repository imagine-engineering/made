use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    AutocastPolicy, BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority,
    DType, DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    NativeStreamRegistry, Scalar, StreamId, Tensor, TensorDescriptor,
    generated_activation_normalization_functional_01::{
        log_softmax_jvp_with_context_exact_native as canonical_log_softmax_jvp,
        log_softmax_vjp_with_context_exact_native as canonical_log_softmax_vjp,
        log_softmax_with_context_exact_native as canonical_log_softmax,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_18::{
        ElementwiseRuntimePartEighteenError, add_scalar_tensor_jvp_with_context_exact_native,
        add_scalar_tensor_vjp_with_context_exact_native,
        add_scalar_tensor_with_context_exact_native, bincount_with_context_exact_native,
        bucketize_with_context_exact_native, byte_tensor_with_context_exact_native,
        element_size_exact_native, is_autocast_enabled_exact_native,
        library_custom_op_exact_native, log_softmax_exact_native, log_softmax_jvp_exact_native,
        log_softmax_vjp_exact_native, numel_function_exact_native,
        sub_method_jvp_with_context_exact_native, sub_method_vjp_with_context_exact_native,
        sub_method_with_context_exact_native, xpu_stream_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-C579143F7B56",
    "COMFY-TENSOR-OP-C83ECA429710",
    "COMFY-TENSOR-OP-C1FAC5999B98",
    "COMFY-TENSOR-OP-C575200CD790",
    "COMFY-TENSOR-OP-C7A255E21877",
    "COMFY-TENSOR-OP-C7B72CD0ABE7",
    "COMFY-TENSOR-OP-C0D4EF19ED71",
    "COMFY-TENSOR-OP-C64E630A756E",
    "COMFY-TENSOR-OP-C63B343CD3EF",
    "COMFY-TENSOR-OP-C46EB25624FB",
    "COMFY-TENSOR-OP-C1BBE55AA3A0",
    "COMFY-TENSOR-OP-C05FE0730305",
];

#[test]
fn part_eighteen_workspace_is_exact_bounded_and_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_i64(
        &backend,
        &workspace_authority,
        &[4],
        &[0, 1, 1, 3],
        &cancellation,
    )?;
    let bytes = 3 * 4 * u64::try_from(std::mem::size_of::<i64>())?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancellation,
    );
    bincount_with_context_exact_native(&backend, &input, 4, &exact)?;
    assert_eq!(exact.scratch.peak_bytes(), bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes - 1)?,
        &cancellation,
    );
    assert!(bincount_with_context_exact_native(&backend, &input, 4, &insufficient).is_err());
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancelled,
    );
    assert!(bincount_with_context_exact_native(&backend, &input, 4, &cancelled_context).is_err());
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

fn authorized_context<'a>(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
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
            let bytes: [u8; 8] = tensor.element_bytes(&[index])?.try_into()?;
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

fn assert_cancelled<T>(result: Result<T, ElementwiseRuntimePartEighteenError>) {
    assert!(matches!(
        result,
        Err(ElementwiseRuntimePartEighteenError::Cancelled)
    ));
}

#[test]
fn task_61_every_public_tensor_adapter_observes_cancellation_before_validation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let active = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &active)?;
    let indices = upload_i64(&backend, &workspace_authority, &[2], &[0, 1], &active)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );

    assert_cancelled(byte_tensor_with_context_exact_native(
        &backend,
        &[],
        &execution,
    ));
    assert_cancelled(element_size_exact_native(&input, &cancelled));
    assert_cancelled(sub_method_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(1.0)),
        f32::NAN,
        &execution,
    ));
    assert_cancelled(sub_method_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(1.0)),
        f32::NAN,
        &input,
        &execution,
    ));
    assert_cancelled(sub_method_jvp_with_context_exact_native(
        &backend,
        &input,
        None,
        f32::NAN,
        &execution,
    ));
    assert_cancelled(add_scalar_tensor_with_context_exact_native(
        &backend,
        f32::NAN,
        &indices,
        f32::NAN,
        &execution,
    ));
    assert_cancelled(add_scalar_tensor_vjp_with_context_exact_native(
        &backend,
        &indices,
        f32::NAN,
        &input,
        &execution,
    ));
    assert_cancelled(add_scalar_tensor_jvp_with_context_exact_native(
        &backend,
        &indices,
        f32::NAN,
        &execution,
    ));
    assert_cancelled(bincount_with_context_exact_native(
        &backend, &input, 0, &execution,
    ));
    assert_cancelled(bucketize_with_context_exact_native(
        &backend, &indices, &indices, false, false, &execution,
    ));
    assert_cancelled(is_autocast_enabled_exact_native(
        &AutocastPolicy::new(true, DType::F16, false)?,
        &cancelled,
    ));
    assert_cancelled(library_custom_op_exact_native(
        "invalid",
        &[0, 0],
        &cancelled,
    ));
    assert_cancelled(log_softmax_exact_native(
        &backend,
        &indices,
        i64::MAX,
        &execution,
    ));
    assert_cancelled(log_softmax_vjp_exact_native(
        &backend,
        &indices,
        &input,
        i64::MAX,
        &execution,
    ));
    assert_cancelled(log_softmax_jvp_exact_native(
        &backend,
        &indices,
        &input,
        i64::MAX,
        &execution,
    ));
    assert_cancelled(numel_function_exact_native(&input, &cancelled));

    let registry = NativeStreamRegistry::default();
    let xpu = DeviceId::new(DeviceKind::Xpu, 0);
    let capabilities = BackendCapabilityMatrix::new(xpu, Vec::new(), Vec::new())?;
    assert_cancelled(xpu_stream_exact_native(
        &registry,
        &capabilities,
        DeviceId::new(DeviceKind::Cuda, 0),
        0,
        &cancelled,
    ));
    let stream = xpu_stream_exact_native(&registry, &capabilities, xpu, 0, &active)?;
    assert_eq!(stream.id(), StreamId::new(1));
    Ok(())
}

#[test]
fn task_61_resolution_slice_and_fixture_digests_are_exact() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_18")
        .ok_or("Task 61 resolution slice is missing")?;
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
    for contract in slice.contracts {
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            contract.evidence_fixture_sha256
        );
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            fixture
                .get("operation_id")
                .and_then(serde_json::Value::as_str),
            Some(contract.operation_id)
        );
        assert_eq!(
            fixture
                .get("owner_task_id")
                .and_then(serde_json::Value::as_str),
            Some(contract.owner_task_id)
        );
    }
    Ok(())
}

#[test]
fn task_61_constructor_element_size_and_numel_reuse_canonical_descriptor_owners()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let tensor = byte_tensor_with_context_exact_native(&backend, &[0, 255, 17], &execution)?;
    assert_eq!(tensor.descriptor().dtype(), DType::U8);
    assert_eq!(tensor.descriptor().shape(), [3]);
    assert_eq!(tensor.host_storage_bytes()?, [0, 255, 17]);
    assert_eq!(element_size_exact_native(&tensor, &cancellation)?, 1);
    assert_eq!(numel_function_exact_native(&tensor, &cancellation)?, 3);
    Ok(())
}

#[test]
fn task_61_add_and_sub_adapters_preserve_analytical_maps() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[3.0, 4.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let subtracted = sub_method_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        0.5,
        &execution,
    )?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &subtracted, &cancellation)?,
        &[2.0, 3.0],
    );
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let gradients = sub_method_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        0.5,
        &output_gradient,
        &execution,
    )?;
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[1.0, 2.0],
    );
    assert!(gradients.other.is_none());
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 1.0],
        &cancellation,
    )?;
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &sub_method_jvp_with_context_exact_native(&backend, &tangent, None, 0.5, &execution)?,
            &cancellation,
        )?,
        &[1.0, 1.0],
    );

    let added =
        add_scalar_tensor_with_context_exact_native(&backend, 0.5, &input, 0.25, &execution)?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &added, &cancellation)?,
        &[1.25, 1.5],
    );
    let add_gradient = add_scalar_tensor_vjp_with_context_exact_native(
        &backend,
        &input,
        0.25,
        &output_gradient,
        &execution,
    )?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &add_gradient, &cancellation)?,
        &[0.25, 0.5],
    );
    let add_tangent =
        add_scalar_tensor_jvp_with_context_exact_native(&backend, &tangent, 0.25, &execution)?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &add_tangent, &cancellation)?,
        &[0.25, 0.25],
    );
    Ok(())
}

#[test]
fn task_61_bincount_and_bucketize_match_integer_boundary_semantics() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let indices = upload_i64(
        &backend,
        &workspace_authority,
        &[4],
        &[0, 1, 1, 3],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let counts = bincount_with_context_exact_native(&backend, &indices, 6, &execution)?;
    assert_eq!(i64_values(&counts)?, [1, 2, 0, 1, 0, 0]);

    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[6],
        &[0.0, 1.0, 1.5, 2.0, 3.0, f32::NAN],
        &cancellation,
    )?;
    let boundaries = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    assert_eq!(
        i64_values(&bucketize_with_context_exact_native(
            &backend,
            &input,
            &boundaries,
            false,
            false,
            &execution,
        )?)?,
        [0, 0, 1, 1, 2, 2]
    );
    assert_eq!(
        i64_values(&bucketize_with_context_exact_native(
            &backend,
            &input,
            &boundaries,
            true,
            false,
            &execution,
        )?)?,
        [0, 1, 1, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn task_61_autocast_and_custom_operator_are_typed_canonical_adapters() -> Result<(), Box<dyn Error>>
{
    let cancellation = CancellationToken::default();
    let enabled = AutocastPolicy::new(true, DType::F16, false)?;
    let disabled = AutocastPolicy::new(false, DType::F32, true)?;
    assert!(is_autocast_enabled_exact_native(&enabled, &cancellation)?);
    assert!(!is_autocast_enabled_exact_native(&disabled, &cancellation)?);
    let declaration =
        library_custom_op_exact_native("flash_attention::flash_attn", &[], &cancellation)?;
    assert_eq!(declaration.qualified_name(), "flash_attention::flash_attn");
    assert_eq!(declaration.kernel().as_str(), "flash_attention.flash_attn");
    assert!(declaration.mutates_arguments().is_empty());
    assert!(library_custom_op_exact_native("invalid", &[], &cancellation).is_err());
    assert!(library_custom_op_exact_native("valid::operator", &[0, 0], &cancellation).is_err());
    Ok(())
}

#[test]
fn task_61_log_softmax_forward_vjp_and_jvp_are_stable() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        &cancellation,
    )?;
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let output = log_softmax_exact_native(&backend, &input, -1, &execution)?;
    let canonical_output = canonical_log_softmax(
        &backend,
        &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        &[2, 3],
        -1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &output, &cancellation)?,
        &canonical_output,
    );
    assert_close(
        &canonical_output,
        &[
            -2.407606, -1.407606, -0.407606, -1.0986123, -1.0986123, -1.0986123,
        ],
    );
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 0.0, -1.0, 1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let expected = [1.0, 0.0, -1.0, -1.0, 0.0, 1.0];
    let canonical_gradient = canonical_log_softmax_vjp(
        &backend,
        &canonical_output,
        &[1.0, 0.0, -1.0, 1.0, 2.0, 3.0],
        &[2, 3],
        -1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_close(&canonical_gradient, &expected);
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &log_softmax_vjp_exact_native(&backend, &output, &tangent, -1, &execution)?,
            &cancellation,
        )?,
        &canonical_gradient,
    );
    let canonical_tangent = canonical_log_softmax_jvp(
        &backend,
        &canonical_output,
        &[1.0, 0.0, -1.0, 1.0, 2.0, 3.0],
        &[2, 3],
        -1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_close(
        &canonical_tangent,
        &[1.5752103, 0.5752103, -0.4247897, -1.0, 0.0, 1.0],
    );
    assert_ne!(canonical_gradient, canonical_tangent);
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &log_softmax_jvp_exact_native(&backend, &output, &tangent, -1, &execution)?,
            &cancellation,
        )?,
        &canonical_tangent,
    );
    Ok(())
}

#[test]
fn task_61_xpu_streams_use_the_only_native_stream_registry() -> Result<(), Box<dyn Error>> {
    let device = DeviceId::new(DeviceKind::Xpu, 2);
    let capabilities = BackendCapabilityMatrix::new(device, Vec::new(), Vec::new())?;
    let registry = NativeStreamRegistry::default();
    let cancellation = CancellationToken::default();
    let first = xpu_stream_exact_native(&registry, &capabilities, device, 0, &cancellation)?;
    let second = xpu_stream_exact_native(&registry, &capabilities, device, -1, &cancellation)?;
    assert_eq!(first.id(), StreamId::new(1));
    assert_eq!(second.id(), StreamId::new(2));
    assert_eq!(second.device(), device);
    assert_eq!(second.priority(), -1);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(xpu_stream_exact_native(&registry, &capabilities, device, 0, &cancelled).is_err());
    let third = xpu_stream_exact_native(&registry, &capabilities, device, 3, &cancellation)?;
    assert_eq!(third.id(), StreamId::new(3));
    assert!(
        xpu_stream_exact_native(
            &registry,
            &capabilities,
            DeviceId::new(DeviceKind::Cuda, 2),
            0,
            &cancellation,
        )
        .is_err()
    );
    Ok(())
}
