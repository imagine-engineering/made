use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    AutocastPolicy, BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority,
    DType, DecodedScalar, DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    Layout, Scalar, StreamId, Tensor, TensorDescriptor, ViewAccess,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_14::{
        ElementwiseRuntimePartFourteenError,
        abs_function_jvp_with_context_exact_native as abs_function_jvp_exact_native,
        abs_function_vjp_with_context_exact_native as abs_function_vjp_exact_native,
        abs_function_with_context_exact_native as abs_function_exact_native,
        argsort_with_context_exact_native as argsort_exact_native, autocast_exact_native,
        concat_jvp_with_context_exact_native as concat_jvp_exact_native,
        concat_vjp_with_context_exact_native as concat_vjp_exact_native,
        concat_with_context_exact_native as concat_exact_native, cudnn_version_exact_native,
        detach_exact_native, detach_jvp_exact_native, detach_vjp_exact_native,
        isposinf_with_context_exact_native as isposinf_exact_native,
        mul_in_place_with_context_exact_native as mul_in_place_exact_native,
        view_as_real_exact_native, view_as_real_jvp_exact_native, view_as_real_vjp_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-A2B7298E8EB4",
    "COMFY-TENSOR-OP-A496777C1987",
    "COMFY-TENSOR-OP-A3A6638578F7",
    "COMFY-TENSOR-OP-A46ED7068064",
    "COMFY-TENSOR-OP-A59D885AD4F9",
    "COMFY-TENSOR-OP-9F0E29E01970",
    "COMFY-TENSOR-OP-9E2C8B750099",
    "COMFY-TENSOR-OP-9CD229514F61",
    "COMFY-TENSOR-OP-9D504472EFE5",
    "COMFY-TENSOR-OP-A680A8A7456F",
    "COMFY-TENSOR-OP-A67546895304",
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

fn upload_complex64(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    values: &[(f32, f32)],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::Complex64,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(
        values
            .len()
            .checked_mul(8)
            .ok_or("complex fixture overflow")?,
    )?;
    for (real, imaginary) in values {
        bytes.extend_from_slice(&real.to_ne_bytes());
        bytes.extend_from_slice(&imaginary.to_ne_bytes());
    }
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
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
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    for index in 0..count {
        let index = [u64::try_from(index)?];
        match DType::I64.decode_scalar(tensor.element_bytes(&index)?)? {
            DecodedScalar::Signed(value) => result.push(value),
            value => return Err(format!("unexpected argsort scalar {value:?}").into()),
        }
    }
    Ok(result)
}

fn require_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartFourteenError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(ElementwiseRuntimePartFourteenError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected canonical cancellation, got {error}").into()),
        Ok(_) => Err("expected canonical cancellation, got success".into()),
    }
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_14")
        .ok_or("Task 57 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    assert_eq!(
        slice
            .contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect::<BTreeSet<_>>()
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
    Ok(())
}

#[test]
fn detach_mul_and_cancellation_preserve_atomic_alias_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let detached = detach_exact_native(&input, &cancellation)?;
    assert_eq!(detached.storage_id(), input.storage_id());
    assert!(detach_vjp_exact_native(&cancellation)?.is_none());
    assert!(detach_jvp_exact_native(&cancellation)?.is_none());

    let mut multiplied = input.clone();
    mul_in_place_exact_native(
        &backend,
        &mut multiplied,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &workspace_authority, &multiplied, &cancellation)?,
        [2.0, 4.0, 6.0]
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = authorized_context(&backend, &workspace_authority, &cancelled)?;
    let before = values(&backend, &workspace_authority, &input, &cancellation)?;
    let mut unchanged = input;
    assert!(
        mul_in_place_exact_native(
            &backend,
            &mut unchanged,
            ElementwiseOperand::Scalar(Scalar::Float(3.0)),
            &cancelled_execution,
        )
        .is_err()
    );
    assert_eq!(
        values(&backend, &workspace_authority, &unchanged, &cancellation)?,
        before
    );
    Ok(())
}

#[test]
fn canonical_abs_and_concat_adapters_preserve_analytical_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[0.0, 3.0],
        &cancellation,
    )?;
    let gradient = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[2.0, 2.0],
        &cancellation,
    )?;
    let signed = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[-2.0, 3.0],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &abs_function_exact_native(&backend, &signed, &execution)?,
            &cancellation
        )?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &abs_function_vjp_exact_native(&backend, &signed, &gradient, &execution)?,
            &cancellation
        )?,
        [-2.0, 2.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &abs_function_jvp_exact_native(&backend, &signed, &gradient, &execution)?,
            &cancellation
        )?,
        [-2.0, 2.0]
    );

    let joined = concat_exact_native(&backend, &[input.clone(), signed.clone()], 0, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &joined, &cancellation)?,
        [0.0, 3.0, -2.0, 3.0]
    );
    let joined_gradient = upload(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let gradients = concat_vjp_exact_native(
        &backend,
        &[input.clone(), signed.clone()],
        0,
        &joined_gradient,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &workspace_authority, &gradients[0], &cancellation)?,
        [1.0, 2.0]
    );
    assert_eq!(
        values(&backend, &workspace_authority, &gradients[1], &cancellation)?,
        [3.0, 4.0]
    );
    let tangent = concat_jvp_exact_native(
        &backend,
        &[input, signed],
        &[gradient.clone(), gradient],
        0,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &workspace_authority, &tangent, &cancellation)?,
        [2.0; 4]
    );
    Ok(())
}

#[test]
fn argsort_isposinf_autocast_and_cudnn_are_deterministic_typed_projections()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[5],
        &[3.0, f32::NAN, 1.0, 3.0, -2.0],
        &cancellation,
    )?;
    assert_eq!(
        i64_values(&argsort_exact_native(
            &backend, &input, -1, false, true, &execution
        )?)?,
        [4, 2, 0, 3, 1]
    );
    assert_eq!(
        i64_values(&argsort_exact_native(
            &backend, &input, 0, true, true, &execution
        )?)?,
        [1, 0, 3, 2, 4]
    );

    let exceptional = upload(
        &backend,
        &workspace_authority,
        &[5],
        &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, -0.0],
        &cancellation,
    )?;
    let flags = isposinf_exact_native(&backend, &exceptional, &execution)?;
    assert_eq!(flags.descriptor().dtype(), DType::Bool);
    assert_eq!(flags.contiguous_bytes()?, &[1, 0, 0, 0, 0]);

    assert_eq!(
        autocast_exact_native(DeviceKind::Cpu, None, true, None, &cancellation)?,
        AutocastPolicy::new(true, DType::Bf16, true)?
    );
    assert_eq!(
        autocast_exact_native(DeviceKind::Cuda, None, false, Some(false), &cancellation)?,
        AutocastPolicy::new(false, DType::F16, false)?
    );
    assert!(
        autocast_exact_native(DeviceKind::Cpu, Some(DType::I64), true, None, &cancellation)
            .is_err()
    );
    let cuda = BackendCapabilityMatrix::new(DeviceId::new(DeviceKind::Cuda, 0), vec![], vec![])?;
    assert_eq!(cudnn_version_exact_native(&cuda, &cancellation)?, None);
    Ok(())
}

#[test]
fn view_as_real_preserves_strided_storage_and_inverse_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let base = upload_complex64(
        &backend,
        &workspace_authority,
        &[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        &cancellation,
    )?;
    let descriptor = TensorDescriptor::new_strided(
        vec![2],
        vec![2],
        0,
        DType::Complex64,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let complex = base.view(descriptor, ViewAccess::ReadOnly)?;
    let real = view_as_real_exact_native(&complex, &cancellation)?;
    assert_eq!(real.storage_id(), base.storage_id());
    assert_eq!(real.descriptor().shape(), [2, 2]);
    assert_eq!(real.descriptor().strides(), [4, 1]);
    assert_eq!(real.access(), ViewAccess::ReadOnly);
    assert_eq!(
        DType::F32.decode_scalar(real.element_bytes(&[0, 0])?)?,
        DecodedScalar::Real(1.0)
    );
    assert_eq!(
        DType::F32.decode_scalar(real.element_bytes(&[1, 1])?)?,
        DecodedScalar::Real(6.0)
    );

    let inverse = view_as_real_vjp_exact_native(&real, &cancellation)?;
    assert_eq!(inverse.descriptor(), complex.descriptor());
    assert_eq!(inverse.storage_id(), real.storage_id());
    let tangent = view_as_real_jvp_exact_native(&complex, &cancellation)?;
    assert_eq!(tangent.descriptor(), real.descriptor());
    Ok(())
}

#[test]
fn every_local_task57_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload(&backend, &workspace_authority, &[2], &[1.0, 2.0], &live)?;
    let mut mutable = input.clone();
    let original_storage = mutable.storage_id();
    let original_version = mutable.storage_version();

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = context(&backend, &workspace_authority, &cancelled)?;

    require_cancelled(detach_exact_native(&input, &cancelled))?;
    require_cancelled(detach_vjp_exact_native(&cancelled))?;
    require_cancelled(detach_jvp_exact_native(&cancelled))?;
    require_cancelled(mul_in_place_exact_native(
        &backend,
        &mut mutable,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        &execution,
    ))?;
    require_cancelled(abs_function_exact_native(&backend, &input, &execution))?;
    require_cancelled(abs_function_vjp_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(abs_function_jvp_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(argsort_exact_native(
        &backend, &input, 9, false, true, &execution,
    ))?;
    require_cancelled(autocast_exact_native(
        DeviceKind::Cpu,
        Some(DType::I64),
        true,
        None,
        &cancelled,
    ))?;
    require_cancelled(cudnn_version_exact_native(
        &CpuBackend::capability_matrix(),
        &cancelled,
    ))?;
    require_cancelled(concat_exact_native(&backend, &[], 9, &execution))?;
    require_cancelled(concat_vjp_exact_native(
        &backend,
        &[],
        9,
        &input,
        &execution,
    ))?;
    require_cancelled(concat_jvp_exact_native(&backend, &[], &[], 9, &execution))?;
    require_cancelled(isposinf_exact_native(&backend, &input, &execution))?;
    require_cancelled(view_as_real_exact_native(&input, &cancelled))?;
    require_cancelled(view_as_real_vjp_exact_native(&input, &cancelled))?;
    require_cancelled(view_as_real_jvp_exact_native(&input, &cancelled))?;

    assert_eq!(mutable.storage_id(), original_storage);
    assert_eq!(mutable.storage_version(), original_version);
    Ok(())
}
