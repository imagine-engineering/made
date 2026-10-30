use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_15::{
        ElementwiseRuntimePartFifteenError, FLIPLR_OPERATION_ID,
        atan2_jvp_with_context_exact_native as atan2_jvp_exact_native,
        atan2_vjp_with_context_exact_native as atan2_vjp_exact_native,
        atan2_with_context_exact_native as atan2_exact_native,
        baddbmm_jvp_with_context_exact_native as baddbmm_jvp_exact_native,
        baddbmm_vjp_with_context_exact_native as baddbmm_vjp_exact_native,
        baddbmm_with_context_exact_native as baddbmm_exact_native,
        ceil_function_with_context_exact_native as ceil_function_exact_native,
        diag_jvp_with_context_exact_native as diag_jvp_exact_native,
        diag_vjp_with_context_exact_native as diag_vjp_exact_native,
        diag_with_context_exact_native as diag_exact_native,
        flip_dimensions_with_context_exact_native as flip_dimensions_exact_native,
        fliplr_jvp_with_context_exact_native as fliplr_jvp_exact_native,
        fliplr_vjp_with_context_exact_native as fliplr_vjp_exact_native,
        fliplr_with_context_exact_native as fliplr_exact_native,
        float_tensor_with_context_exact_native as float_tensor_exact_native,
        long_with_context_exact_native as long_exact_native,
        ones_like_with_context_exact_native as ones_like_exact_native,
        searchsorted_with_context_exact_native as searchsorted_exact_native,
        tan_jvp_with_context_exact_native as tan_jvp_exact_native,
        tan_vjp_with_context_exact_native as tan_vjp_exact_native,
        tan_with_context_exact_native as tan_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-AB3C563E635F",
    "COMFY-TENSOR-OP-AA014D6FD446",
    "COMFY-TENSOR-OP-AC4C80016C2B",
    "COMFY-TENSOR-OP-AB6C1D5013D1",
    "COMFY-TENSOR-OP-AA097B951CB6",
    "COMFY-TENSOR-OP-A91E1CFDC489",
    "COMFY-TENSOR-OP-A8B4A9E79500",
    "COMFY-TENSOR-OP-AC979D604DAA",
    "COMFY-TENSOR-OP-A69CEE614EB5",
    "COMFY-TENSOR-OP-AA36FFD0433B",
    "COMFY-TENSOR-OP-AAB5F10B20F5",
    "COMFY-TENSOR-OP-A68AE691163C",
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

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    let shape = tensor.descriptor().shape();
    for linear in 0..count {
        let mut remaining = linear;
        let mut indices = vec![0; shape.len()];
        for (axis, dimension) in shape.iter().enumerate().rev() {
            let dimension = usize::try_from(*dimension)?;
            if dimension != 0 {
                indices[axis] = u64::try_from(remaining % dimension)?;
                remaining /= dimension;
            }
        }
        match DType::I64.decode_scalar(tensor.element_bytes(&indices)?)? {
            DecodedScalar::Signed(value) => result.push(value),
            scalar => return Err(format!("unexpected I64 scalar {scalar:?}").into()),
        }
    }
    Ok(result)
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
    result: Result<T, ElementwiseRuntimePartFifteenError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(ElementwiseRuntimePartFifteenError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected cancellation, got {error}").into()),
        Ok(_) => Err("pre-cancelled adapter published a result".into()),
    }
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_15")
        .ok_or("Task 58 resolution slice is missing")?;
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
fn constructor_cast_ceil_and_ones_reuse_canonical_owners() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = float_tensor_exact_native(
        &backend,
        &[-1.25, 0.0, 2.75, 4.0],
        &[2, 2],
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(input.descriptor().dtype(), DType::F32);
    assert_eq!(
        values(&backend, &workspace_authority, &input, &cancellation)?,
        [-1.25, 0.0, 2.75, 4.0]
    );

    let long = long_exact_native(
        &backend,
        &input,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(long.descriptor().dtype(), DType::I64);
    assert_eq!(i64_values(&long)?, [-1, 0, 2, 4]);

    let ceiling = ceil_function_exact_native(&backend, &input, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &ceiling, &cancellation)?,
        [-1.0, 0.0, 3.0, 4.0]
    );
    let ones = ones_like_exact_native(&backend, &input, Some(DType::F32), &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &ones, &cancellation)?,
        [1.0; 4]
    );
    assert_ne!(ones.storage_id(), input.storage_id());
    Ok(())
}

#[test]
fn atan2_and_tan_have_deterministic_forward_vjp_and_jvp_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let other = upload(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[1.0, 3.0],
        &cancellation,
    )?;
    let output = atan2_exact_native(&backend, &input, &other, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[
            1.0_f32.atan2(1.0),
            1.0_f32.atan2(3.0),
            2.0_f32.atan2(1.0),
            2.0_f32.atan2(3.0),
        ],
    );
    let output_gradient = upload(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients = atan2_vjp_exact_native(&backend, &input, &other, &output_gradient, &execution)?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[0.5 + 0.3, 0.2 + 3.0 / 13.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.other,
            &cancellation,
        )?,
        &[-0.5 - 0.4, -0.1 - 2.0 / 13.0],
    );
    let input_tangent = upload(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[0.25, 0.5],
        &cancellation,
    )?;
    let other_tangent = upload(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[0.1, 0.2],
        &cancellation,
    )?;
    let tangent = atan2_jvp_exact_native(
        &backend,
        &input,
        &other,
        &input_tangent,
        &other_tangent,
        &execution,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &tangent, &cancellation)?,
        &[0.075, 0.055, 0.06, 1.1 / 13.0],
    );

    let angles = upload(
        &backend,
        &workspace_authority,
        &[3],
        &[-0.25, 0.0, 0.5],
        &cancellation,
    )?;
    let directions = upload(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let tangent_values = [-0.25_f32.tan(), 0.0, 0.5_f32.tan()];
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &tan_exact_native(&backend, &angles, &execution)?,
            &cancellation,
        )?,
        &tangent_values,
    );
    let expected_derivative = [
        1.0 / (-0.25_f32).cos().powi(2),
        2.0,
        3.0 / 0.5_f32.cos().powi(2),
    ];
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &tan_vjp_exact_native(&backend, &angles, &directions, &execution)?,
            &cancellation,
        )?,
        &expected_derivative,
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &tan_jvp_exact_native(&backend, &angles, &directions, &execution)?,
            &cancellation,
        )?,
        &expected_derivative,
    );
    Ok(())
}

#[test]
fn baddbmm_composes_canonical_batch_matmul_and_analytical_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let input = upload(&backend, &workspace_authority, &[1], &[10.0], &cancellation)?;
    let batch1 = upload(
        &backend,
        &workspace_authority,
        &[1, 2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let batch2 = upload(
        &backend,
        &workspace_authority,
        &[1, 2, 2],
        &[5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let output = baddbmm_exact_native(&backend, &input, &batch1, &batch2, 0.5, 2.0, &execution)?;
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[43.0, 49.0, 91.0, 105.0],
    );
    let ignored_input = upload(
        &backend,
        &workspace_authority,
        &[1],
        &[f32::NAN],
        &cancellation,
    )?;
    let without_input = baddbmm_exact_native(
        &backend,
        &ignored_input,
        &batch1,
        &batch2,
        0.0,
        1.0,
        &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &without_input,
            &cancellation,
        )?,
        &[19.0, 22.0, 43.0, 50.0],
    );

    let gradient = upload(
        &backend,
        &workspace_authority,
        &[1, 2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let gradients = baddbmm_vjp_exact_native(
        &backend, &input, &batch1, &batch2, 0.5, 2.0, &gradient, &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[2.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.batch1,
            &cancellation,
        )?,
        &[22.0, 30.0, 22.0, 30.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.batch2,
            &cancellation,
        )?,
        &[8.0, 8.0, 12.0, 12.0],
    );

    let input_tangent = upload(&backend, &workspace_authority, &[1], &[2.0], &cancellation)?;
    let batch1_tangent = upload(
        &backend,
        &workspace_authority,
        &[1, 2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let batch2_tangent = upload(
        &backend,
        &workspace_authority,
        &[1, 2, 2],
        &[0.5; 4],
        &cancellation,
    )?;
    let tangent = baddbmm_jvp_exact_native(
        &backend,
        &input,
        &batch1,
        &batch2,
        &input_tangent,
        &batch1_tangent,
        &batch2_tangent,
        0.5,
        2.0,
        &execution,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &tangent, &cancellation)?,
        &[28.0, 32.0, 32.0, 36.0],
    );
    Ok(())
}

#[test]
fn diag_fliplr_and_searchsorted_cover_structured_edges_and_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &workspace_authority, &cancellation)?;
    let vector = upload(
        &backend,
        &workspace_authority,
        &[2],
        &[2.0, 3.0],
        &cancellation,
    )?;
    let matrix = diag_exact_native(&backend, &vector, 1, &execution)?;
    assert_eq!(matrix.descriptor().shape(), [3, 3]);
    assert_eq!(
        values(&backend, &workspace_authority, &matrix, &cancellation)?,
        [0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &diag_vjp_exact_native(&backend, &vector, 1, &matrix, &execution)?,
            &cancellation,
        )?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &diag_jvp_exact_native(&backend, &vector, 1, &vector, &execution)?,
            &cancellation,
        )?,
        values(&backend, &workspace_authority, &matrix, &cancellation)?
    );
    let extracted = diag_exact_native(&backend, &matrix, 1, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &extracted, &cancellation,)?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &diag_jvp_exact_native(&backend, &matrix, 1, &matrix, &execution)?,
            &cancellation,
        )?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &diag_vjp_exact_native(&backend, &matrix, 1, &vector, &execution)?,
            &cancellation,
        )?,
        values(&backend, &workspace_authority, &matrix, &cancellation)?
    );
    assert!(
        diag_vjp_exact_native(&backend, &vector, 1, &vector, &execution).is_err(),
        "diag VJP must reject a gradient that does not match the forward output"
    );

    let rows = upload(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let flipped = fliplr_exact_native(&backend, &rows, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &flipped, &cancellation)?,
        [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &fliplr_vjp_exact_native(&backend, &flipped, &execution)?,
            &cancellation,
        )?,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        values(
            &backend,
            &workspace_authority,
            &fliplr_jvp_exact_native(&backend, &rows, &execution)?,
            &cancellation,
        )?,
        [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );

    let volume = upload(
        &backend,
        &workspace_authority,
        &[2, 2, 3],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &cancellation,
    )?;
    let generic_flip =
        flip_dimensions_exact_native(&backend, &volume, &[0, -1], FLIPLR_OPERATION_ID, &execution)?;
    assert_eq!(
        values(&backend, &workspace_authority, &generic_flip, &cancellation)?,
        [
            9.0, 8.0, 7.0, 12.0, 11.0, 10.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0
        ]
    );
    let integer_rows = long_exact_native(&backend, &rows, &execution)?;
    assert_eq!(
        i64_values(&flip_dimensions_exact_native(
            &backend,
            &integer_rows,
            &[1],
            FLIPLR_OPERATION_ID,
            &execution,
        )?)?,
        [3, 2, 1, 6, 5, 4]
    );
    assert!(
        flip_dimensions_exact_native(&backend, &volume, &[0, -3], FLIPLR_OPERATION_ID, &execution,)
            .is_err(),
        "normalized duplicate dimensions must fail"
    );

    let boundaries = upload(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 3.0, 3.0, 8.0],
        &cancellation,
    )?;
    let probes = upload(
        &backend,
        &workspace_authority,
        &[5],
        &[0.0, 3.0, 4.0, f32::NAN, 9.0],
        &cancellation,
    )?;
    assert_eq!(
        i64_values(&searchsorted_exact_native(
            &backend,
            &boundaries,
            &probes,
            false,
            &execution,
        )?)?,
        [0, 1, 3, 4, 4]
    );
    assert_eq!(
        i64_values(&searchsorted_exact_native(
            &backend,
            &boundaries,
            &probes,
            true,
            &execution,
        )?)?,
        [0, 3, 3, 4, 4]
    );
    let malformed = upload(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 0.0, 2.0],
        &cancellation,
    )?;
    assert!(searchsorted_exact_native(&backend, &malformed, &probes, false, &execution).is_err());

    let row_boundaries = upload(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0, 2.0, 4.0, 8.0, -2.0, 0.0, 5.0, 5.0],
        &cancellation,
    )?;
    let row_probes = upload(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[0.0, 4.0, 9.0, -2.0, 5.0, 6.0],
        &cancellation,
    )?;
    assert_eq!(
        i64_values(&searchsorted_exact_native(
            &backend,
            &row_boundaries,
            &row_probes,
            false,
            &execution,
        )?)?,
        [0, 2, 4, 0, 2, 4]
    );
    assert_eq!(
        i64_values(&searchsorted_exact_native(
            &backend,
            &row_boundaries,
            &row_probes,
            true,
            &execution,
        )?)?,
        [0, 3, 4, 1, 4, 4]
    );
    Ok(())
}

#[test]
fn every_public_task58_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let active = CancellationToken::default();
    let input = upload(&backend, &workspace_authority, &[2], &[0.25, 0.5], &active)?;
    let before = input.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = authorized_context(&backend, &workspace_authority, &cancelled)?;
    require_cancelled(float_tensor_exact_native(
        &backend,
        &[],
        &[1],
        &cancelled_execution,
    ))?;
    require_cancelled(long_exact_native(&backend, &input, &cancelled_execution))?;
    require_cancelled(atan2_exact_native(
        &backend,
        &input,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(atan2_vjp_exact_native(
        &backend,
        &input,
        &input,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(atan2_jvp_exact_native(
        &backend,
        &input,
        &input,
        &input,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(baddbmm_exact_native(
        &backend,
        &input,
        &input,
        &input,
        1.0,
        1.0,
        &cancelled_execution,
    ))?;
    require_cancelled(baddbmm_vjp_exact_native(
        &backend,
        &input,
        &input,
        &input,
        1.0,
        1.0,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(baddbmm_jvp_exact_native(
        &backend,
        &input,
        &input,
        &input,
        &input,
        &input,
        &input,
        1.0,
        1.0,
        &cancelled_execution,
    ))?;
    require_cancelled(ceil_function_exact_native(
        &backend,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(diag_exact_native(&backend, &input, 0, &cancelled_execution))?;
    require_cancelled(diag_vjp_exact_native(
        &backend,
        &input,
        0,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(diag_jvp_exact_native(
        &backend,
        &input,
        0,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(fliplr_exact_native(&backend, &input, &cancelled_execution))?;
    require_cancelled(flip_dimensions_exact_native(
        &backend,
        &input,
        &[0, 0],
        FLIPLR_OPERATION_ID,
        &cancelled_execution,
    ))?;
    require_cancelled(fliplr_vjp_exact_native(
        &backend,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(fliplr_jvp_exact_native(
        &backend,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(ones_like_exact_native(
        &backend,
        &input,
        Some(DType::F32),
        &cancelled_execution,
    ))?;
    require_cancelled(searchsorted_exact_native(
        &backend,
        &input,
        &input,
        false,
        &cancelled_execution,
    ))?;
    require_cancelled(tan_exact_native(&backend, &input, &cancelled_execution))?;
    require_cancelled(tan_vjp_exact_native(
        &backend,
        &input,
        &input,
        &cancelled_execution,
    ))?;
    require_cancelled(tan_jvp_exact_native(
        &backend,
        &input,
        &input,
        &cancelled_execution,
    ))?;
    assert_eq!(input.contiguous_bytes()?, before);
    Ok(())
}
