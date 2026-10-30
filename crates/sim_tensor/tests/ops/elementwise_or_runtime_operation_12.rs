use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    AutocastPolicy, BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority,
    DType, DecodedScalar, DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    StreamId, Tensor, TensorDescriptor,
    generated_elementwise_or_runtime_operation_12::{
        ElementwiseRuntimePartTwelveError, calculate_fan_in_and_fan_out_exact_native,
        cuda_current_stream_exact_native,
        cumsum_function_jvp_with_context_exact_native as cumsum_function_jvp_exact_native,
        cumsum_function_vjp_with_context_exact_native as cumsum_function_vjp_exact_native,
        cumsum_function_with_context_exact_native as cumsum_function_exact_native,
        get_autocast_gpu_dtype_exact_native, is_contiguous_exact_native,
        is_floating_point_exact_native, numel_exact_native,
        stft_jvp_with_context_exact_native as stft_jvp_exact_native,
        stft_vjp_with_context_exact_native as stft_vjp_exact_native,
        stft_with_context_exact_native as stft_exact_native,
        tensor_pow_jvp_with_context_exact_native as tensor_pow_jvp_exact_native,
        tensor_pow_vjp_with_context_exact_native as tensor_pow_vjp_exact_native,
        tensor_pow_with_context_exact_native as tensor_pow_exact_native,
        topk_jvp_with_context_exact_native as topk_jvp_exact_native,
        topk_vjp_with_context_exact_native as topk_vjp_exact_native,
        topk_with_context_exact_native as topk_exact_native,
        tril_jvp_with_context_exact_native as tril_jvp_exact_native,
        tril_vjp_with_context_exact_native as tril_vjp_exact_native,
        tril_with_context_exact_native as tril_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-8A892AD7A3C2",
    "COMFY-TENSOR-OP-8AE4F174E7A1",
    "COMFY-TENSOR-OP-8E0C873ABBAD",
    "COMFY-TENSOR-OP-8C351B65C789",
    "COMFY-TENSOR-OP-861EE6173859",
    "COMFY-TENSOR-OP-88FE050115A9",
    "COMFY-TENSOR-OP-868F72C2BE67",
    "COMFY-TENSOR-OP-8B5439D32B8F",
    "COMFY-TENSOR-OP-8E5582D70F18",
    "COMFY-TENSOR-OP-8C29B75AEA2A",
    "COMFY-TENSOR-OP-8DF974B2A77C",
    "COMFY-TENSOR-OP-874C83BCB8C5",
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

fn upload_f32(
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

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut result = Vec::with_capacity(count);
    for linear in 0..count {
        let columns = usize::try_from(*tensor.descriptor().shape().last().ok_or("rank zero")?)?;
        let indices = if tensor.descriptor().rank() == 1 {
            vec![u64::try_from(linear)?]
        } else {
            vec![
                u64::try_from(linear / columns)?,
                u64::try_from(linear % columns)?,
            ]
        };
        let DecodedScalar::Signed(value) = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.element_bytes(&indices)?)?
        else {
            return Err("expected signed tensor".into());
        };
        result.push(value);
    }
    Ok(result)
}

fn complex_values(tensor: &Tensor) -> Result<Vec<(f32, f32)>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let columns = usize::try_from(*tensor.descriptor().shape().last().ok_or("rank zero")?)?;
    let mut result = Vec::with_capacity(count);
    for linear in 0..count {
        let indices = vec![
            u64::try_from(linear / columns)?,
            u64::try_from(linear % columns)?,
        ];
        let DecodedScalar::Complex { real, imaginary } = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.element_bytes(&indices)?)?
        else {
            return Err("expected complex tensor".into());
        };
        result.push((real as f32, imaginary as f32));
    }
    Ok(result)
}

fn require_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartTwelveError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(ElementwiseRuntimePartTwelveError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected cancellation, received {error}").into()),
        Ok(_) => Err("cancelled operation unexpectedly succeeded".into()),
    }
}

#[test]
fn metadata_policy_and_stream_adapters_reuse_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let tensor = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3, 4],
        &[0.0; 24],
        &cancellation,
    )?;
    assert!(is_contiguous_exact_native(&tensor, &cancellation)?);
    assert_eq!(numel_exact_native(&tensor, &cancellation)?, 24);
    assert!(is_floating_point_exact_native(&tensor, &cancellation)?);
    let empty = upload_f32(&backend, &workspace_authority, &[0, 3], &[], &cancellation)?;
    assert!(is_contiguous_exact_native(&empty, &cancellation)?);
    assert_eq!(numel_exact_native(&empty, &cancellation)?, 0);
    assert_eq!(
        calculate_fan_in_and_fan_out_exact_native(&tensor, &cancellation)?,
        sim_tensor::generated_elementwise_or_runtime_operation_12::FanInAndFanOut {
            fan_in: 12,
            fan_out: 8,
        }
    );
    let policy = AutocastPolicy::new(true, DType::F16, false)?;
    assert_eq!(
        get_autocast_gpu_dtype_exact_native(&policy, &cancellation)?,
        DType::F16
    );
    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let capabilities = BackendCapabilityMatrix::new(cuda, vec![], vec![])?;
    let stream = StreamId::new(19);
    let execution = backend.execution_context(
        stream,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    assert_eq!(
        cuda_current_stream_exact_native(&capabilities, cuda, &execution)?,
        stream
    );
    assert!(cuda_current_stream_exact_native(&capabilities, DeviceId::CPU, &execution).is_err());
    Ok(())
}

#[test]
fn canonical_power_scan_and_triangular_adapters_preserve_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let base = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let exponent = upload_f32(&backend, &workspace_authority, &[1], &[2.0], &cancellation)?;
    let powered = tensor_pow_exact_native(&backend, &base, &exponent, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &powered, &cancellation)?,
        [1.0, 4.0, 9.0, 16.0]
    );
    let ones = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let zero_exponent_tangent =
        upload_f32(&backend, &workspace_authority, &[1], &[0.0], &cancellation)?;
    let power_gradients =
        tensor_pow_vjp_exact_native(&backend, &base, &exponent, &ones, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &power_gradients.left,
            &cancellation
        )?,
        [2.0, 4.0, 6.0, 8.0]
    );
    let expected_exponent_gradient =
        4.0_f32 * 2.0_f32.ln() + 9.0 * 3.0_f32.ln() + 16.0 * 4.0_f32.ln();
    let exponent_gradient = values(
        &backend,
        &workspace_authority,
        &power_gradients.right,
        &cancellation,
    )?;
    assert!((exponent_gradient[0] - expected_exponent_gradient).abs() < 1.0e-4);
    let power_tangent = tensor_pow_jvp_exact_native(
        &backend,
        &base,
        &exponent,
        &ones,
        &zero_exponent_tangent,
        &execution,
    )?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &power_tangent,
            &cancellation
        )?,
        [2.0, 4.0, 6.0, 8.0]
    );
    let cumulative = cumsum_function_exact_native(&backend, &base, 1, None, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &cumulative, &cancellation)?,
        [1.0, 3.0, 3.0, 7.0]
    );
    let cumulative_gradient = cumsum_function_vjp_exact_native(&backend, &ones, 1, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &cumulative_gradient,
            &cancellation
        )?,
        [2.0, 1.0, 2.0, 1.0]
    );
    let cumulative_tangent = cumsum_function_jvp_exact_native(&backend, &ones, 1, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &cumulative_tangent,
            &cancellation
        )?,
        [1.0, 2.0, 1.0, 2.0]
    );
    let lower = tril_exact_native(&backend, &base, 0, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &lower, &cancellation)?,
        [1.0, 0.0, 3.0, 4.0]
    );
    let lower_gradient = tril_vjp_exact_native(&backend, &ones, 0, &execution)?;
    let lower_tangent = tril_jvp_exact_native(&backend, &ones, 0, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &lower_gradient,
            &cancellation
        )?,
        [1.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &lower_tangent,
            &cancellation
        )?,
        [1.0, 0.0, 1.0, 1.0]
    );
    Ok(())
}

#[test]
fn stft_fft_and_derivatives_are_deterministic_and_checked() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let impulse = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 0.0, 0.0, 0.0],
        &cancellation,
    )?;
    let spectrum = stft_exact_native(
        &backend,
        &impulse,
        4,
        Some(4),
        Some(4),
        None,
        false,
        false,
        true,
        &execution,
    )?;
    assert_eq!(spectrum.descriptor().shape(), [3, 1]);
    for (real, imaginary) in complex_values(&spectrum)? {
        assert!((real - 1.0).abs() < 1.0e-6);
        assert!(imaginary.abs() < 1.0e-6);
    }
    let tangent = stft_jvp_exact_native(
        &backend,
        &impulse,
        4,
        Some(4),
        Some(4),
        None,
        false,
        false,
        true,
        &execution,
    )?;
    assert_eq!(complex_values(&tangent)?, complex_values(&spectrum)?);
    let gradient = stft_vjp_exact_native(
        &backend,
        &impulse,
        &spectrum,
        4,
        Some(4),
        Some(4),
        None,
        false,
        false,
        true,
        &execution,
    )?;
    let gradient = values(&backend, &workspace_authority, &gradient, &cancellation)?;
    assert!((gradient[0] - 3.0).abs() < 1.0e-5);
    assert!(gradient[1].abs() < 1.0e-5);
    assert!((gradient[2] - 1.0).abs() < 1.0e-5);
    assert!(gradient[3].abs() < 1.0e-5);

    let non_power_of_two = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 0.0, 0.0],
        &cancellation,
    )?;
    let full_spectrum = stft_exact_native(
        &backend,
        &non_power_of_two,
        3,
        Some(3),
        Some(3),
        None,
        false,
        false,
        false,
        &execution,
    )?;
    assert_eq!(full_spectrum.descriptor().shape(), [3, 1]);
    for (real, imaginary) in complex_values(&full_spectrum)? {
        assert!((real - 1.0).abs() < 1.0e-6);
        assert!(imaginary.abs() < 1.0e-6);
    }
    let window = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 0.5, 0.5, 1.0],
        &cancellation,
    )?;
    let waveform = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let centered = stft_exact_native(
        &backend,
        &waveform,
        4,
        Some(2),
        Some(4),
        Some(&window),
        true,
        true,
        true,
        &execution,
    )?;
    assert_eq!(centered.descriptor().shape(), [3, 3]);
    assert!(
        complex_values(&centered)?
            .iter()
            .all(|(real, imaginary)| { real.is_finite() && imaginary.is_finite() })
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_execution = authorized_context(&backend, &workspace_authority, &cancelled)?;
    assert!(
        stft_exact_native(
            &backend,
            &impulse,
            4,
            Some(4),
            Some(4),
            None,
            false,
            false,
            true,
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn topk_gradients_and_resolution_evidence_are_complete() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0, 5.0, 3.0, 4.0, -1.0, 2.0, 8.0, 7.0],
        &cancellation,
    )?;
    let selected = topk_exact_native(&backend, &input, 2, -1, true, true, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &selected.values,
            &cancellation
        )?,
        [5.0, 4.0, 8.0, 7.0]
    );
    assert_eq!(i64_values(&selected.indices)?, [1, 3, 2, 3]);
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[10.0, 20.0, 30.0, 40.0],
        &cancellation,
    )?;
    let input_gradient = topk_vjp_exact_native(
        &backend,
        &input,
        &selected.indices,
        &output_gradient,
        -1,
        &execution,
    )?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &input_gradient,
            &cancellation
        )?,
        [0.0, 10.0, 0.0, 20.0, 0.0, 0.0, 30.0, 40.0]
    );
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let selected_tangent =
        topk_jvp_exact_native(&backend, &tangent, &selected.indices, -1, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &selected_tangent,
            &cancellation
        )?,
        [2.0, 4.0, 7.0, 8.0]
    );
    let ties_and_nan = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[f32::NAN, 5.0, 5.0, 1.0],
        &cancellation,
    )?;
    let deterministic = topk_exact_native(&backend, &ties_and_nan, 3, 0, true, true, &execution)?;
    let deterministic_values = values(
        &backend,
        &workspace_authority,
        &deterministic.values,
        &cancellation,
    )?;
    assert!(deterministic_values[0].is_nan());
    assert_eq!(&deterministic_values[1..], &[5.0, 5.0]);
    assert_eq!(i64_values(&deterministic.indices)?, [0, 1, 2]);
    let smallest = topk_exact_native(&backend, &ties_and_nan, 2, 0, false, false, &execution)?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &smallest.values,
            &cancellation
        )?,
        [5.0, 1.0]
    );
    assert_eq!(i64_values(&smallest.indices)?, [1, 3]);
    assert!(topk_exact_native(&backend, &input, 5, -1, true, true, &execution).is_err());

    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_12")
        .ok_or("Task 55 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    let ids = slice
        .contracts
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(ids, IDS.into_iter().collect());
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(!contract.rust_signature.contains("&ExecutionContext)"));
        assert!(
            !contract.rust_signature.contains("ExecutionContext")
                || contract.rust_signature.contains("ExecutionContext<'_>")
        );
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-861ee6173859"
        );
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
fn every_local_task55_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &live)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let execution = authorized_context(&backend, &workspace_authority, &cancelled)?;

    require_cancelled(is_contiguous_exact_native(&input, &cancelled))?;
    require_cancelled(numel_exact_native(&input, &cancelled))?;
    require_cancelled(tensor_pow_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(tensor_pow_vjp_exact_native(
        &backend, &input, &input, &input, &execution,
    ))?;
    require_cancelled(tensor_pow_jvp_exact_native(
        &backend, &input, &input, &input, &input, &execution,
    ))?;
    require_cancelled(cuda_current_stream_exact_native(
        &BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?,
        DeviceId::CPU,
        &execution,
    ))?;
    require_cancelled(cumsum_function_exact_native(
        &backend,
        &input,
        i64::MAX,
        Some(DType::Bool),
        &execution,
    ))?;
    require_cancelled(cumsum_function_vjp_exact_native(
        &backend,
        &input,
        i64::MAX,
        &execution,
    ))?;
    require_cancelled(cumsum_function_jvp_exact_native(
        &backend,
        &input,
        i64::MAX,
        &execution,
    ))?;
    let policy = AutocastPolicy::new(true, DType::F16, false)?;
    require_cancelled(get_autocast_gpu_dtype_exact_native(&policy, &cancelled))?;
    require_cancelled(is_floating_point_exact_native(&input, &cancelled))?;
    require_cancelled(calculate_fan_in_and_fan_out_exact_native(
        &input, &cancelled,
    ))?;
    require_cancelled(stft_exact_native(
        &backend,
        &input,
        0,
        Some(0),
        Some(0),
        None,
        true,
        false,
        true,
        &execution,
    ))?;
    require_cancelled(stft_jvp_exact_native(
        &backend,
        &input,
        0,
        Some(0),
        Some(0),
        None,
        true,
        false,
        true,
        &execution,
    ))?;
    require_cancelled(stft_vjp_exact_native(
        &backend,
        &input,
        &input,
        0,
        Some(0),
        Some(0),
        None,
        true,
        false,
        true,
        &execution,
    ))?;
    require_cancelled(topk_exact_native(
        &backend,
        &input,
        usize::MAX,
        i64::MAX,
        true,
        true,
        &execution,
    ))?;
    require_cancelled(topk_vjp_exact_native(
        &backend,
        &input,
        &input,
        &input,
        i64::MAX,
        &execution,
    ))?;
    require_cancelled(topk_jvp_exact_native(
        &backend,
        &input,
        &input,
        i64::MAX,
        &execution,
    ))?;
    require_cancelled(tril_exact_native(&backend, &input, isize::MAX, &execution))?;
    require_cancelled(tril_vjp_exact_native(
        &backend,
        &input,
        isize::MAX,
        &execution,
    ))?;
    require_cancelled(tril_jvp_exact_native(
        &backend,
        &input,
        isize::MAX,
        &execution,
    ))?;

    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(
        values(&backend, &workspace_authority, &input, &live)?,
        [1.0, 2.0]
    );
    Ok(())
}

#[test]
fn workspace_authorization_is_exact_atomic_and_convergent() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[4.0, 1.0, 3.0, 2.0],
        &cancellation,
    )?;
    let original_bytes = input.contiguous_bytes()?.to_vec();
    let simultaneous_bytes = 2 * std::mem::size_of::<f32>()
        + 2 * std::mem::size_of::<i64>()
        + 4 * std::mem::size_of::<(f32, usize)>();
    let simultaneous_bytes = u64::try_from(simultaneous_bytes)?;

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(simultaneous_bytes)?,
        &cancellation,
    );
    let selected = topk_exact_native(&backend, &input, 2, 0, true, true, &exact)?;
    assert_eq!(i64_values(&selected.indices)?, [0, 2]);
    assert_eq!(exact.scratch.peak_bytes(), simultaneous_bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(simultaneous_bytes - 1)?,
        &cancellation,
    );
    assert!(topk_exact_native(&backend, &input, 2, 0, true, true, &insufficient).is_err());
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, original_bytes);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(simultaneous_bytes)?,
        &cancelled,
    );
    assert!(topk_exact_native(&backend, &input, 2, 0, true, true, &cancelled_context).is_err());
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, original_bytes);
    Ok(())
}
