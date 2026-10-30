use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, MemoryFormatReference,
    NativeDeviceProperties, StreamId, Tensor, TensorDescriptor,
    generated_elementwise_or_runtime_operation_22::{
        ElementwiseRuntimePartTwentyTwoError, arccos_function_jvp_with_context_exact_native,
        arccos_function_vjp_with_context_exact_native, arccos_function_with_context_exact_native,
        argsort_method_with_context_exact_native, cuda_amp_autocast_exact_native,
        cuda_get_device_name_exact_native, exp_function_jvp_with_context_exact_native,
        exp_function_vjp_with_context_exact_native,
        exp_function_with_context_exact_native as exp_function_exact_native,
        exp_function_with_context_exact_native, floor_method_jvp_with_context_exact_native,
        floor_method_vjp_with_context_exact_native, floor_method_with_context_exact_native,
        lerp_function_jvp_with_context_exact_native, lerp_function_vjp_with_context_exact_native,
        lerp_function_with_context_exact_native, t_method_exact_native, t_method_jvp_exact_native,
        t_method_vjp_exact_native, xpu_set_device_exact_native,
        zeros_like_with_context_exact_native as zeros_like_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-FB884955DE1E",
    "COMFY-TENSOR-OP-F2D7AE6E8F48",
    "COMFY-TENSOR-OP-F1F71360D559",
    "COMFY-TENSOR-OP-F56BA497ED13",
    "COMFY-TENSOR-OP-FBA06A1411DE",
    "COMFY-TENSOR-OP-F9ED42F7BFDF",
    "COMFY-TENSOR-OP-F18E1AE1B857",
    "COMFY-TENSOR-OP-FA7DD244B7CA",
    "COMFY-TENSOR-OP-F27D07B4E10D",
    "COMFY-TENSOR-OP-F15A2D8A6BD4",
    "COMFY-TENSOR-OP-FACB7FC5B252",
    "COMFY-TENSOR-OP-F3D0014DD82A",
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
fn assert_cancelled<T>(result: Result<T, ElementwiseRuntimePartTwentyTwoError>) {
    assert!(matches!(
        result,
        Err(ElementwiseRuntimePartTwentyTwoError::Cancelled)
    ));
}

#[test]
fn task_65_resolution_slice_seals_all_twelve_unique_contracts() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_22")
        .ok_or("Task 65 resolution slice is missing")?;
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
fn canonical_numeric_adapters_preserve_forward_vjp_and_jvp() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.25, 0.5, 0.75],
        &cancellation,
    )?;
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);

    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &floor_method_with_context_exact_native(&backend, &input, &execution)?,
            &cancellation,
        )?,
        &[0.0, 0.0, 0.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &floor_method_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &[0.0, 0.0, 0.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &floor_method_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &[0.0, 0.0, 0.0],
    );

    let acos = arccos_function_with_context_exact_native(&backend, &input, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &acos, &cancellation)?,
        &[0.25_f32.acos(), 0.5_f32.acos(), 0.75_f32.acos()],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &arccos_function_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &values(
            &backend,
            &workspace_authority,
            &arccos_function_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
    );

    let exponential = exp_function_with_context_exact_native(&backend, &input, None, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &exponential, &cancellation)?,
        &[0.25_f32.exp(), 0.5_f32.exp(), 0.75_f32.exp()],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &exp_function_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
        &values(
            &backend,
            &workspace_authority,
            &exp_function_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
            &cancellation,
        )?,
    );

    let end = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.25, 1.5, 1.75],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &lerp_function_with_context_exact_native(&backend, &input, &end, 0.25, &execution)?,
            &cancellation,
        )?,
        &[0.5, 0.75, 1.0],
    );
    let gradients = lerp_function_vjp_with_context_exact_native(
        &backend, &input, &end, 0.25, &tangent, &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[0.75, 1.5, 2.25],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.end,
            &cancellation,
        )?,
        &[0.25, 0.5, 0.75],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &lerp_function_jvp_with_context_exact_native(
                &backend, &input, &end, &tangent, &tangent, 0.25, &execution,
            )?,
            &cancellation,
        )?,
        &[1.0, 2.0, 3.0],
    );
    Ok(())
}

#[test]
fn method_sort_and_t_views_reuse_canonical_owners() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[3.0, 1.0, 2.0, 6.0, 4.0, 5.0],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &argsort_method_with_context_exact_native(
                &backend, &input, -1, false, true, &execution,
            )?,
            &cancellation,
        )?,
        &[1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
    );

    let mut transposed = t_method_exact_native(&input, &cancellation)?;
    assert_eq!(transposed.descriptor().shape(), &[3, 2]);
    assert_eq!(transposed.descriptor().strides(), &[1, 3]);
    assert_eq!(transposed.storage_id(), input.storage_id());
    assert!(transposed.write().is_err());
    assert_close(
        &values(&backend, &workspace_authority, &transposed, &cancellation)?,
        &[3.0, 6.0, 1.0, 4.0, 2.0, 5.0],
    );
    assert_eq!(
        t_method_vjp_exact_native(&transposed, &cancellation)?
            .descriptor()
            .shape(),
        &[2, 3]
    );
    assert_eq!(
        t_method_jvp_exact_native(&input, &cancellation)?
            .descriptor()
            .shape(),
        &[3, 2]
    );
    let rank_three = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 1, 1],
        &[1.0],
        &cancellation,
    )?;
    assert!(t_method_exact_native(&rank_three, &cancellation).is_err());
    Ok(())
}

#[test]
fn policy_and_device_adapters_project_canonical_state_only() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let policy = cuda_amp_autocast_exact_native(true, None, Some(false), &cancellation)?;
    assert!(policy.enabled());
    assert_eq!(policy.dtype(), DType::F16);
    assert!(!policy.cache_enabled());

    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let properties = NativeDeviceProperties::new(
        cuda,
        "Zed Native CUDA Fixture",
        16 * 1024 * 1024,
        9,
        0,
        Some("sm_90".to_owned()),
        true,
    )?;
    let cuda_capabilities = BackendCapabilityMatrix::new_with_properties(
        cuda,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        cuda_get_device_name_exact_native(&cuda_capabilities, cuda, &cancellation)?,
        "Zed Native CUDA Fixture"
    );
    assert!(
        cuda_get_device_name_exact_native(&cuda_capabilities, DeviceId::CPU, &cancellation)
            .is_err()
    );

    let xpu = DeviceId::new(DeviceKind::Xpu, 4);
    let available = [BackendCapabilityMatrix::new(xpu, Vec::new(), Vec::new())?];
    assert_eq!(
        xpu_set_device_exact_native(&available, xpu, &cancellation)?.device(),
        xpu
    );
    let duplicate = [available[0].clone(), available[0].clone()];
    assert!(xpu_set_device_exact_native(&duplicate, xpu, &cancellation).is_err());
    Ok(())
}

#[test]
fn zeros_like_and_exp_out_publish_atomically_with_requested_format() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let transposed = t_method_exact_native(&input, &cancellation)?;
    let zeros = zeros_like_exact_native(
        &backend,
        &transposed,
        None,
        None,
        None,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(zeros.descriptor().shape(), &[3, 2]);
    assert_eq!(zeros.descriptor().strides(), &[1, 3]);
    assert_eq!(zeros.descriptor().layout(), Layout::Strided);
    assert_close(
        &values(&backend, &workspace_authority, &zeros, &cancellation)?,
        &[0.0; 6],
    );

    let channels_last = zeros_like_exact_native(
        &backend,
        &upload_f32(
            &backend,
            &workspace_authority,
            &[1, 2, 2, 2],
            &[1.0; 8],
            &cancellation,
        )?,
        Some(DType::F16),
        None,
        Some(MemoryFormatReference::Layout(Layout::ChannelsLast)),
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(channels_last.descriptor().layout(), Layout::ChannelsLast);
    assert_eq!(channels_last.descriptor().dtype(), DType::F16);
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &channels_last,
            &cancellation,
        )?,
        &[0.0; 8],
    );

    let mut out = upload_f32(&backend, &workspace_authority, &[1], &[-1.0], &cancellation)?;
    let returned = exp_function_exact_native(
        &backend,
        &input,
        Some(&mut out),
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(returned.storage_id(), out.storage_id());
    assert_eq!(out.descriptor().shape(), &[2, 3]);
    assert_close(
        &values(&backend, &workspace_authority, &out, &cancellation)?,
        &[
            1.0_f32.exp(),
            2.0_f32.exp(),
            3.0_f32.exp(),
            4.0_f32.exp(),
            5.0_f32.exp(),
            6.0_f32.exp(),
        ],
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        zeros_like_exact_native(
            &backend,
            &input,
            None,
            None,
            None,
            &authorized_context(&backend, &workspace_authority, &cancelled)?
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn task_65_every_public_tensor_adapter_observes_cancellation_before_validation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[0.25, 0.5], &live)?;
    let mismatched = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &live)?;
    let rank_three = upload_f32(&backend, &workspace_authority, &[1, 1, 1], &[1.0], &live)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancelled);

    assert_cancelled(argsort_method_with_context_exact_native(
        &backend, &input, 99, false, true, &execution,
    ));
    assert_cancelled(floor_method_with_context_exact_native(
        &backend, &input, &execution,
    ));
    assert_cancelled(floor_method_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(floor_method_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(t_method_exact_native(&rank_three, &cancelled));
    assert_cancelled(t_method_vjp_exact_native(&rank_three, &cancelled));
    assert_cancelled(t_method_jvp_exact_native(&rank_three, &cancelled));

    assert_cancelled(arccos_function_with_context_exact_native(
        &backend, &input, &execution,
    ));
    assert_cancelled(arccos_function_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(arccos_function_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(cuda_amp_autocast_exact_native(
        true,
        Some(DType::F32),
        None,
        &cancelled,
    ));
    let cpu_capabilities = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert_cancelled(cuda_get_device_name_exact_native(
        &cpu_capabilities,
        DeviceId::CPU,
        &cancelled,
    ));

    let mut out = mismatched.clone();
    let original_out = out.storage_id();
    assert_cancelled(exp_function_exact_native(
        &backend,
        &input,
        Some(&mut out),
        &execution,
    ));
    assert_eq!(out.storage_id(), original_out);
    assert_cancelled(exp_function_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(exp_function_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(lerp_function_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        0.5,
        &execution,
    ));
    assert_cancelled(lerp_function_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        0.5,
        &mismatched,
        &execution,
    ));
    assert_cancelled(lerp_function_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &mismatched,
        &input,
        0.5,
        &execution,
    ));
    assert_cancelled(xpu_set_device_exact_native(
        std::slice::from_ref(&cpu_capabilities),
        DeviceId::CPU,
        &cancelled,
    ));
    assert_cancelled(zeros_like_exact_native(
        &backend,
        &rank_three,
        None,
        Some(DeviceId::new(DeviceKind::Cuda, 0)),
        Some(MemoryFormatReference::Layout(Layout::Strided)),
        &execution,
    ));
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}
