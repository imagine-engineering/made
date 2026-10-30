use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType,
    DeterministicOperationDisposition, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, OperationSupport, StreamId, Tensor,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_13::{
        ElementwiseRuntimePartThirteenError,
        addmm_jvp_with_context_exact_native as addmm_jvp_exact_native,
        addmm_vjp_with_context_exact_native as addmm_vjp_exact_native,
        addmm_with_context_exact_native as addmm_exact_native,
        allclose_with_context_exact_native as allclose_exact_native,
        cuda_is_bf16_supported_exact_native,
        empty_like_with_context_exact_native as empty_like_exact_native,
        lerp_jvp_with_context_exact_native as lerp_jvp_exact_native,
        lerp_vjp_with_context_exact_native as lerp_vjp_exact_native,
        lerp_with_context_exact_native as lerp_exact_native,
        quantile_jvp_with_context_exact_native as quantile_jvp_exact_native,
        quantile_vjp_with_context_exact_native as quantile_vjp_exact_native,
        quantile_with_context_exact_native as quantile_exact_native,
        softmax_function_jvp_with_context_exact_native as softmax_function_jvp_exact_native,
        softmax_function_vjp_with_context_exact_native as softmax_function_vjp_exact_native,
        softmax_function_with_context_exact_native as softmax_function_exact_native,
        tensor_sign_jvp_with_context_exact_native as tensor_sign_jvp_exact_native,
        tensor_sign_vjp_with_context_exact_native as tensor_sign_vjp_exact_native,
        tensor_sign_with_context_exact_native as tensor_sign_exact_native,
        use_deterministic_algorithms_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-9C679FFC6CCF",
    "COMFY-TENSOR-OP-9547009E77B6",
    "COMFY-TENSOR-OP-9443F2A50F6D",
    "COMFY-TENSOR-OP-9285B877ECB7",
    "COMFY-TENSOR-OP-9CC2489AFA14",
    "COMFY-TENSOR-OP-9B68023167CF",
    "COMFY-TENSOR-OP-97058675DD67",
    "COMFY-TENSOR-OP-8F0ACDA02879",
    "COMFY-TENSOR-OP-8F6CC1A0A7AC",
    "COMFY-TENSOR-OP-9BE377B59853",
    "COMFY-TENSOR-OP-999B3107D90B",
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
) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(2 * 1024 * 1024)?,
        cancellation,
    ))
}

fn upload(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
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
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

fn require_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartThirteenError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(ElementwiseRuntimePartThirteenError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected canonical cancellation, got {error}").into()),
        Ok(_) => Err("expected canonical cancellation, got success".into()),
    }
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_13")
        .ok_or("Task 56 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    let expected = IDS.into_iter().collect::<BTreeSet<_>>();
    let actual = slice
        .contracts
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
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
            fixture
                .get("operation_id")
                .and_then(serde_json::Value::as_str),
            Some(contract.operation_id)
        );
        assert_eq!(
            fixture
                .get("overload_id")
                .and_then(serde_json::Value::as_str),
            Some(contract.overload_id)
        );
    }
    Ok(())
}

#[test]
fn view_allocation_device_and_determinism_adapters_reuse_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[6],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &cancellation,
    )?;
    let empty = empty_like_exact_native(&backend, &input, Some(DType::F16), None, &execution)?;
    assert_ne!(empty.storage_id(), input.storage_id());
    assert_eq!(empty.descriptor().shape(), input.descriptor().shape());
    assert_eq!(empty.descriptor().strides(), input.descriptor().strides());
    assert_eq!(empty.descriptor().dtype(), DType::F16);

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let bf16 = OperationSupport::allocation(DType::Bf16, Layout::Contiguous);
    let capabilities = BackendCapabilityMatrix::new(cuda, vec![bf16], vec![])?;
    assert!(cuda_is_bf16_supported_exact_native(
        &capabilities,
        &cancellation
    )?);
    assert!(
        cuda_is_bf16_supported_exact_native(&CpuBackend::capability_matrix(), &cancellation)
            .is_err()
    );
    let warning_policy = use_deterministic_algorithms_exact_native(true, true, &cancellation)?;
    assert_eq!(
        warning_policy.disposition(&capabilities, bf16),
        DeterministicOperationDisposition::Warn
    );
    let rejecting_policy = use_deterministic_algorithms_exact_native(true, false, &cancellation)?;
    assert_eq!(
        rejecting_policy.disposition(&capabilities, bf16),
        DeterministicOperationDisposition::Reject
    );
    let disabled_policy = use_deterministic_algorithms_exact_native(false, false, &cancellation)?;
    assert_eq!(
        disabled_policy.disposition(&capabilities, bf16),
        DeterministicOperationDisposition::Allowed
    );
    Ok(())
}

#[test]
fn lerp_sign_and_allclose_preserve_broadcast_and_derivative_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[0.0, 4.0],
        &cancellation,
    )?;
    let end = upload(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[8.0, 12.0],
        &cancellation,
    )?;
    let output = lerp_exact_native(&backend, &input, &end, 0.25, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &output, &cancellation)?,
        [2.0, 3.0, 5.0, 6.0]
    );
    let output_gradient = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients =
        lerp_vjp_exact_native(&backend, &input, &end, 0.25, &output_gradient, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation
        )?,
        [1.5, 1.5]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &gradients.end,
            &cancellation
        )?,
        [0.5, 0.5]
    );
    let tangent = lerp_jvp_exact_native(&backend, &input, &end, &input, &end, 0.25, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &tangent, &cancellation)?,
        [2.0, 3.0, 5.0, 6.0]
    );

    let signed = upload(
        &backend,
        &workspace_authority,
        &[4],
        &[-2.0, -0.0, 0.0, 3.0],
        &cancellation,
    )?;
    let sign = tensor_sign_exact_native(&backend, &signed, &execution)?;
    let sign_values = values(&backend, &workspace_authority, &sign, &cancellation)?;
    assert_eq!(sign_values[0], -1.0);
    assert!(sign_values[1].is_sign_negative());
    assert_eq!(sign_values[2..], [0.0, 1.0]);
    assert!(allclose_exact_native(
        &backend,
        &output,
        &upload(
            &backend,
            &workspace_authority,
            &[2, 2],
            &[2.0, 3.0, 5.0, 6.000_001],
            &cancellation
        )?,
        1.0e-5,
        1.0e-8,
        false,
        &execution,
    )?);
    assert!(!allclose_exact_native(
        &backend,
        &output,
        &upload(
            &backend,
            &workspace_authority,
            &[2, 2],
            &[2.0, 3.0, 5.0, 6.1],
            &cancellation
        )?,
        1.0e-5,
        1.0e-8,
        false,
        &execution,
    )?);
    Ok(())
}

#[test]
fn addmm_reuses_linear_equations_with_broadcast_vjp_and_jvp()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let matrix1 = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let matrix2 = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let output = addmm_exact_native(&backend, &input, &matrix1, &matrix2, 2.0, 0.5, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &output, &cancellation)?,
        [11.5, 15.0, 23.5, 29.0]
    );
    let output_gradient = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients = addmm_vjp_exact_native(
        &backend,
        &input,
        &matrix1,
        &matrix2,
        2.0,
        0.5,
        &output_gradient,
        &execution,
    )?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation
        )?,
        [4.0, 4.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &gradients.matrix1,
            &cancellation
        )?,
        [5.5, 7.5, 5.5, 7.5]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &gradients.matrix2,
            &cancellation
        )?,
        [2.0, 2.0, 3.0, 3.0]
    );
    let zeros = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[0.0; 4],
        &cancellation,
    )?;
    let input_tangent = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0; 2],
        &cancellation,
    )?;
    let tangent = addmm_jvp_exact_native(
        &backend,
        &input,
        &matrix1,
        &matrix2,
        &input_tangent,
        &zeros,
        &zeros,
        2.0,
        0.5,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &workspace_authority, &tangent, &cancellation)?,
        [2.0; 4]
    );
    Ok(())
}

#[test]
fn quantile_and_softmax_cover_axes_boundaries_nan_derivatives_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &cancellation,
    )?;
    let median = quantile_exact_native(&backend, &input, 0.5, Some(-1), &execution)?;
    assert_eq!(median.descriptor().shape(), [2]);
    assert_eq!(
        values(&backend, &workspace_authority, &median, &cancellation)?,
        [1.5, 5.5]
    );
    let flattened = quantile_exact_native(&backend, &input, 0.25, None, &execution)?;
    assert!(flattened.descriptor().shape().is_empty());
    assert_eq!(
        values(&backend, &workspace_authority, &flattened, &cancellation)?,
        [1.75]
    );
    let output_gradient = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 1.0],
        &cancellation,
    )?;
    let gradient =
        quantile_vjp_exact_native(&backend, &input, 0.5, Some(1), &output_gradient, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &gradient, &cancellation)?,
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0]
    );
    let tangent = quantile_jvp_exact_native(&backend, &input, &input, 0.5, Some(1), &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &tangent, &cancellation)?,
        [1.5, 5.5]
    );
    let nan_input = upload(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, f32::NAN, 2.0],
        &cancellation,
    )?;
    assert!(
        values(
            &backend,
            &workspace_authority,
            &quantile_exact_native(&backend, &nan_input, 0.0, None, &execution)?,
            &cancellation,
        )?[0]
            .is_nan()
    );

    let softmax = softmax_function_exact_native(&backend, &input, 1, &execution)?;
    let softmax_values = values(&backend, &workspace_authority, &softmax, &cancellation)?;
    for row in softmax_values.chunks_exact(4) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }
    let ones = upload(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0; 8],
        &cancellation,
    )?;
    let softmax_vjp = softmax_function_vjp_exact_native(&backend, &input, &ones, 1, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &softmax_vjp, &cancellation)?,
        &[0.0; 8],
    );
    let softmax_jvp = softmax_function_jvp_exact_native(&backend, &input, &ones, 1, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &softmax_jvp, &cancellation)?,
        &[0.0; 8],
    );

    assert!(quantile_exact_native(&backend, &input, 1.1, None, &execution).is_err());
    assert!(softmax_function_exact_native(&backend, &input, 2, &execution).is_err());
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = authorized_context(&backend, &workspace_authority, &cancelled)?;
    assert!(quantile_exact_native(&backend, &input, 0.5, None, &cancelled_execution).is_err());
    assert!(use_deterministic_algorithms_exact_native(true, true, &cancelled).is_err());
    Ok(())
}

#[test]
fn every_local_task56_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload(&backend, &workspace_authority, &[2], &[1.0, 2.0], &live)?;
    let original_storage = input.storage_id();
    let original_version = input.storage_version();

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = context(&backend, &workspace_authority, &cancelled)?;

    require_cancelled(lerp_exact_native(&backend, &input, &input, 0.5, &execution))?;
    require_cancelled(lerp_vjp_exact_native(
        &backend, &input, &input, 0.5, &input, &execution,
    ))?;
    require_cancelled(lerp_jvp_exact_native(
        &backend, &input, &input, &input, &input, 0.5, &execution,
    ))?;
    require_cancelled(tensor_sign_exact_native(&backend, &input, &execution))?;
    require_cancelled(tensor_sign_vjp_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(tensor_sign_jvp_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(addmm_exact_native(
        &backend, &input, &input, &input, 1.0, 1.0, &execution,
    ))?;
    require_cancelled(addmm_vjp_exact_native(
        &backend, &input, &input, &input, 1.0, 1.0, &input, &execution,
    ))?;
    require_cancelled(addmm_jvp_exact_native(
        &backend, &input, &input, &input, &input, &input, &input, 1.0, 1.0, &execution,
    ))?;
    require_cancelled(allclose_exact_native(
        &backend, &input, &input, 0.0, -1.0, false, &execution,
    ))?;
    require_cancelled(cuda_is_bf16_supported_exact_native(
        &CpuBackend::capability_matrix(),
        &cancelled,
    ))?;
    require_cancelled(empty_like_exact_native(
        &backend,
        &input,
        None,
        Some(DeviceId::new(DeviceKind::Cuda, 0)),
        &execution,
    ))?;
    require_cancelled(quantile_exact_native(
        &backend,
        &input,
        1.1,
        Some(9),
        &execution,
    ))?;
    require_cancelled(quantile_vjp_exact_native(
        &backend,
        &input,
        1.1,
        Some(9),
        &input,
        &execution,
    ))?;
    require_cancelled(quantile_jvp_exact_native(
        &backend,
        &input,
        &input,
        1.1,
        Some(9),
        &execution,
    ))?;
    require_cancelled(softmax_function_exact_native(
        &backend, &input, 9, &execution,
    ))?;
    require_cancelled(softmax_function_vjp_exact_native(
        &backend, &input, &input, 9, &execution,
    ))?;
    require_cancelled(softmax_function_jvp_exact_native(
        &backend, &input, &input, 9, &execution,
    ))?;
    require_cancelled(use_deterministic_algorithms_exact_native(
        true, true, &cancelled,
    ))?;

    assert_eq!(input.storage_id(), original_storage);
    assert_eq!(input.storage_version(), original_version);
    Ok(())
}
