use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Scalar, StreamId, Tensor,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_jvp_with_context_exact_native,
        add_method_vjp_with_context_exact_native, add_method_with_context_exact_native,
        atan_jvp_with_context_exact_native, atan_vjp_with_context_exact_native,
        atan_with_context_exact_native, bitwise_and_with_context_exact_native,
        is_tensor_exact_native, kaiser_window_with_context_exact_native,
        logaddexp_jvp_with_context_exact_native, logaddexp_vjp_with_context_exact_native,
        logaddexp_with_context_exact_native, mul_method_jvp_with_context_exact_native,
        mul_method_vjp_with_context_exact_native, mul_method_with_context_exact_native,
        square_method_jvp_with_context_exact_native, square_method_vjp_with_context_exact_native,
        square_method_with_context_exact_native, tile_jvp_with_context_exact_native,
        tile_vjp_with_context_exact_native, tile_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-B74E6E64A97F",
    "COMFY-TENSOR-OP-B82E0C11E45D",
    "COMFY-TENSOR-OP-B40398C137FF",
    "COMFY-TENSOR-OP-B4F8F3B2B2E6",
    "COMFY-TENSOR-OP-B4B7266D14A9",
    "COMFY-TENSOR-OP-B153098D5C48",
    "COMFY-TENSOR-OP-B1A79A94DDE6",
    "COMFY-TENSOR-OP-B296530D4BB3",
    "COMFY-TENSOR-OP-B7955A0A7AC9",
    "COMFY-TENSOR-OP-B2699A727A6C",
    "COMFY-TENSOR-OP-B30CBD7D8727",
    "COMFY-TENSOR-OP-B088976A05AB",
];

#[test]
fn part_sixteen_workspace_is_exact_bounded_and_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[-0.5, 0.0, 0.5],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 1.0, 1.0],
        &cancellation,
    )?;
    let bytes = 3 * u64::try_from(std::mem::size_of::<f32>())?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancellation,
    );
    atan_vjp_with_context_exact_native(&backend, &input, &gradient, &exact)?;
    assert_eq!(exact.scratch.peak_bytes(), bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes - 1)?,
        &cancellation,
    );
    assert!(
        atan_vjp_with_context_exact_native(&backend, &input, &gradient, &insufficient).is_err()
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
        atan_vjp_with_context_exact_native(&backend, &input, &gradient, &cancelled_context)
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

fn upload_scalars(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    dtype: DType,
    values: &[Scalar],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend(dtype.encode_scalar(*value, "task-59-fixture", DeviceId::CPU)?);
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, StreamId::DEFAULT)?;
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

fn require_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartSixteenError>,
) -> Result<(), Box<dyn Error>> {
    match result {
        Err(ElementwiseRuntimePartSixteenError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected cancellation, got {error}").into()),
        Ok(_) => Err("pre-cancelled adapter published a result".into()),
    }
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures() -> Result<(), Box<dyn Error>>
{
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_16")
        .ok_or("Task 59 resolution slice is missing")?;
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
fn method_adapters_reuse_canonical_add_mul_and_square_semantics() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[3.0, 4.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let output = add_method_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&other),
        0.5,
        &execution,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[2.5, 3.0, 3.5, 4.0],
    );
    let infinite = add_method_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(f64::INFINITY)),
        1.0,
        &execution,
    )?;
    assert!(
        values(&backend, &workspace_authority, &infinite, &cancellation)?
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_positive())
    );
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients = add_method_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&other),
        0.5,
        &output_gradient,
        &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[2.0, 2.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            gradients.other.as_ref().ok_or("tensor gradient missing")?,
            &cancellation,
        )?,
        &[1.0, 1.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &add_method_jvp_with_context_exact_native(
                &backend,
                &input,
                Some(&other),
                0.5,
                &execution,
            )?,
            &cancellation,
        )?,
        &[2.5, 3.0, 3.5, 4.0],
    );

    let multiplied = mul_method_with_context_exact_native(&backend, &input, &other, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &multiplied, &cancellation)?,
        &[3.0, 4.0, 6.0, 8.0],
    );
    let mul_gradients = mul_method_vjp_with_context_exact_native(
        &backend,
        &input,
        &other,
        &output_gradient,
        &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &mul_gradients.left,
            &cancellation,
        )?,
        &[7.0, 7.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &mul_gradients.right,
            &cancellation,
        )?,
        &[3.0, 3.0],
    );
    let mul_jvp = mul_method_jvp_with_context_exact_native(
        &backend, &input, &other, &input, &other, &execution,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &mul_jvp, &cancellation)?,
        &[6.0, 8.0, 12.0, 16.0],
    );

    let squared = square_method_with_context_exact_native(&backend, &input, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &squared, &cancellation)?,
        &[1.0, 4.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &square_method_vjp_with_context_exact_native(&backend, &input, &input, &execution)?,
            &cancellation,
        )?,
        &[2.0, 8.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &square_method_jvp_with_context_exact_native(&backend, &input, &input, &execution)?,
            &cancellation,
        )?,
        &[2.0, 8.0],
    );
    Ok(())
}

#[test]
fn atan_and_logaddexp_have_stable_forward_and_derivative_maps() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[-1.0, 0.0, 2.0],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[2.0, 3.0, 5.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let atan = atan_with_context_exact_native(&backend, &input, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &atan, &cancellation)?,
        &[(-1.0_f32).atan(), 0.0, 2.0_f32.atan()],
    );
    let expected = [1.0, 3.0, 1.0];
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &atan_vjp_with_context_exact_native(&backend, &input, &gradient, &execution)?,
            &cancellation,
        )?,
        &expected,
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &atan_jvp_with_context_exact_native(&backend, &input, &gradient, &execution)?,
            &cancellation,
        )?,
        &expected,
    );

    let left = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[1_000.0, f32::NEG_INFINITY],
        &cancellation,
    )?;
    let right = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[999.0, f32::NEG_INFINITY],
        &cancellation,
    )?;
    let output = logaddexp_with_context_exact_native(&backend, &left, &right, &execution)?;
    let output_values = values(&backend, &workspace_authority, &output, &cancellation)?;
    assert!((output_values[0] - (1_000.0 + (-1.0_f32).exp().ln_1p())).abs() <= 1.0e-5);
    assert_eq!(output_values[1], 1_000.0);
    assert_eq!(output_values[2], 999.0);
    assert_eq!(output_values[3], f32::NEG_INFINITY);
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients = logaddexp_vjp_with_context_exact_native(
        &backend,
        &left,
        &right,
        &output_gradient,
        &execution,
    )?;
    let left_gradient = values(
        &backend,
        &workspace_authority,
        &gradients.left,
        &cancellation,
    )?;
    let right_gradient = values(
        &backend,
        &workspace_authority,
        &gradients.right,
        &cancellation,
    )?;
    assert_close(&left_gradient, &[1.731_058_6, 0.5]);
    assert_close(&right_gradient, &[1.268_941_4, 0.5]);
    let left_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[2.0, 4.0],
        &cancellation,
    )?;
    let right_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[3.0, 6.0],
        &cancellation,
    )?;
    let tangent = logaddexp_jvp_with_context_exact_native(
        &backend,
        &left,
        &right,
        &left_tangent,
        &right_tangent,
        &execution,
    )?;
    let tangent_values = values(&backend, &workspace_authority, &tangent, &cancellation)?;
    assert!(tangent_values.iter().all(|value| value.is_finite()));
    assert_close(&tangent_values[3..], &[5.0]);
    Ok(())
}

#[test]
fn bitwise_kaiser_tile_identity_and_cancellation_are_exact() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let left = upload_scalars(
        &backend,
        &workspace_authority,
        &[2, 1],
        DType::I16,
        &[Scalar::Signed(7), Scalar::Signed(10)],
        &cancellation,
    )?;
    let right = upload_scalars(
        &backend,
        &workspace_authority,
        &[1, 2],
        DType::U8,
        &[Scalar::Unsigned(3), Scalar::Unsigned(12)],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let bitwise = bitwise_and_with_context_exact_native(&backend, &left, &right, &execution)?;
    assert_eq!(bitwise.descriptor().dtype(), DType::I16);
    let expected = [3_i64, 4, 2, 8];
    for (linear, expected) in expected.into_iter().enumerate() {
        let indices = [u64::try_from(linear / 2)?, u64::try_from(linear % 2)?];
        assert_eq!(
            DType::I16.decode_scalar(bitwise.element_bytes(&indices)?)?,
            DecodedScalar::Signed(expected)
        );
    }

    assert!(is_tensor_exact_native(Some(&left), &cancellation)?);
    assert!(!is_tensor_exact_native(None, &cancellation)?);
    let singleton =
        kaiser_window_with_context_exact_native(&backend, 1, false, 12.0, DType::F32, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &singleton, &cancellation)?,
        &[1.0],
    );
    let window =
        kaiser_window_with_context_exact_native(&backend, 5, false, 8.0, DType::F32, &execution)?;
    let window_values = values(&backend, &workspace_authority, &window, &cancellation)?;
    assert_close(&window_values[0..1], &window_values[4..5]);
    assert_close(&window_values[1..2], &window_values[3..4]);
    assert_close(&window_values[2..3], &[1.0]);

    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let tiled = tile_with_context_exact_native(&backend, &input, &[2, 3], &execution)?;
    assert_eq!(tiled.descriptor().shape(), &[2, 6]);
    assert_close(
        &values(&backend, &workspace_authority, &tiled, &cancellation)?,
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 6],
        &[1.0; 12],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &tile_vjp_with_context_exact_native(
                &backend,
                &input,
                &[2, 3],
                &output_gradient,
                &execution,
            )?,
            &cancellation,
        )?,
        &[6.0, 6.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &tile_jvp_with_context_exact_native(&backend, &input, &input, &[2, 3], &execution)?,
            &cancellation,
        )?,
        &values(&backend, &workspace_authority, &tiled, &cancellation)?,
    );
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    assert!(
        tile_with_context_exact_native(&backend, &input, &[2, 3], &cancelled_execution).is_err()
    );
    assert!(
        kaiser_window_with_context_exact_native(
            &backend,
            5,
            false,
            8.0,
            DType::F32,
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn every_public_task59_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let active = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &active)?;
    let before = input.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );

    require_cancelled(add_method_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&input),
        1.0,
        &execution,
    ))?;
    require_cancelled(add_method_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&input),
        1.0,
        &input,
        &execution,
    ))?;
    require_cancelled(add_method_jvp_with_context_exact_native(
        &backend,
        &input,
        Some(&input),
        1.0,
        &execution,
    ))?;
    require_cancelled(mul_method_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(mul_method_vjp_with_context_exact_native(
        &backend, &input, &input, &input, &execution,
    ))?;
    require_cancelled(mul_method_jvp_with_context_exact_native(
        &backend, &input, &input, &input, &input, &execution,
    ))?;
    require_cancelled(square_method_with_context_exact_native(
        &backend, &input, &execution,
    ))?;
    require_cancelled(square_method_vjp_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(square_method_jvp_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(atan_with_context_exact_native(&backend, &input, &execution))?;
    require_cancelled(atan_vjp_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(atan_jvp_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(bitwise_and_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(is_tensor_exact_native(Some(&input), &cancelled))?;
    require_cancelled(kaiser_window_with_context_exact_native(
        &backend,
        5,
        false,
        8.0,
        DType::F32,
        &execution,
    ))?;
    require_cancelled(logaddexp_with_context_exact_native(
        &backend, &input, &input, &execution,
    ))?;
    require_cancelled(logaddexp_vjp_with_context_exact_native(
        &backend, &input, &input, &input, &execution,
    ))?;
    require_cancelled(logaddexp_jvp_with_context_exact_native(
        &backend, &input, &input, &input, &input, &execution,
    ))?;
    require_cancelled(tile_with_context_exact_native(
        &backend,
        &input,
        &[2],
        &execution,
    ))?;
    require_cancelled(tile_vjp_with_context_exact_native(
        &backend,
        &input,
        &[2],
        &input,
        &execution,
    ))?;
    require_cancelled(tile_jvp_with_context_exact_native(
        &backend,
        &input,
        &input,
        &[2],
        &execution,
    ))?;
    assert_eq!(input.contiguous_bytes()?, before);
    Ok(())
}
