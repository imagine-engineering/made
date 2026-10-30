use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OperationContractId, StreamId,
    Tensor, TensorDescriptor, TensorError, ViewAccess,
    generated_indexing_masking_01::{
        IndexingMaskingPartOneError, NonzeroOutput,
        gather_function_with_context_exact_native as gather_function_exact_native,
        gather_jvp_with_context_exact_native as gather_jvp_exact_native,
        gather_method_with_context_exact_native as gather_method_exact_native,
        gather_vjp_with_context_exact_native as gather_vjp_exact_native,
        index_add_in_place_with_context_exact_native as index_add_in_place_exact_native,
        index_add_jvp_with_context_exact_native as index_add_jvp_exact_native,
        index_add_vjp_with_context_exact_native as index_add_vjp_exact_native,
        masked_fill_in_place_with_context_exact_native as masked_fill_in_place_exact_native,
        masked_fill_jvp_with_context_exact_native as masked_fill_jvp_exact_native,
        masked_fill_vjp_with_context_exact_native as masked_fill_vjp_exact_native,
        narrow_function_exact_native, narrow_jvp_exact_native, narrow_method_exact_native,
        narrow_vjp_with_context_exact_native as narrow_vjp_exact_native,
        nonzero_with_context_exact_native as nonzero_exact_native,
        scatter_function_with_context_exact_native as scatter_function_exact_native,
        scatter_in_place_with_context_exact_native as scatter_in_place_exact_native,
        scatter_jvp_with_context_exact_native as scatter_jvp_exact_native,
        scatter_method_with_context_exact_native as scatter_method_exact_native,
        scatter_vjp_with_context_exact_native as scatter_vjp_exact_native,
        where_jvp_with_context_exact_native as where_jvp_exact_native,
        where_nonzero_with_context_exact_native as where_nonzero_exact_native,
        where_vjp_with_context_exact_native as where_vjp_exact_native,
        where_with_context_exact_native as where_exact_native,
    },
};
use std::{collections::BTreeSet, fs, path::Path};

const EXECUTABLE_IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-650A7E36398C",
    "COMFY-TENSOR-OP-787E82C83CB5",
    "COMFY-TENSOR-OP-923E7CBA8F2A",
    "COMFY-TENSOR-OP-006E05C5DAAF",
    "COMFY-TENSOR-OP-3710A378E57B",
    "COMFY-TENSOR-OP-6CEB132BD4F8",
    "COMFY-TENSOR-OP-301932E71E58",
    "COMFY-TENSOR-OP-A29830647789",
    "COMFY-TENSOR-OP-3885D52BE05C",
    "COMFY-TENSOR-OP-2CC6738611B8",
    "COMFY-TENSOR-OP-40CEC38A1D1F",
];

#[test]
fn workspace_scatter_vjp_tracks_simultaneous_staging_and_converges()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let source = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let index = upload_i64(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[0, 1, 0, 1],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0; 4],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(32)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let gradients =
        scatter_vjp_exact_native(&backend, &input, 1, &index, &source, &gradient, &execution)?;
    assert_eq!(
        decoded(&gradients.input)?,
        [
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(0.0),
        ]
    );
    assert_eq!(
        decoded(&gradients.source)?,
        [
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(1.0),
        ]
    );
    assert_eq!(scratch.peak_bytes(), 32);
    assert_eq!(scratch.in_use_bytes(), 0);

    let too_small = workspace_authority.authorize_workspace(31)?;
    let execution = backend.execution_context(StreamId::DEFAULT, too_small.clone(), &cancellation);
    assert!(matches!(
        scatter_vjp_exact_native(&backend, &input, 1, &index, &source, &gradient, &execution,),
        Err(
            sim_tensor::generated_indexing_masking_01::IndexingMaskingPartOneError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            )
        )
    ));
    assert_eq!(too_small.in_use_bytes(), 0);
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

fn upload_values(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    dtype: DType,
    values: &[DecodedScalar],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&dtype.encode_decoded_scalar(
            *value,
            "indexing masking test upload",
            DeviceId::CPU,
        )?);
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

fn upload_i64(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    upload_values(
        backend,
        workspace_authority,
        shape,
        DType::I64,
        &values
            .iter()
            .copied()
            .map(DecodedScalar::Signed)
            .collect::<Vec<_>>(),
        cancellation,
    )
}

fn upload_bool(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[bool],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    upload_values(
        backend,
        workspace_authority,
        shape,
        DType::Bool,
        &values
            .iter()
            .copied()
            .map(DecodedScalar::Boolean)
            .collect::<Vec<_>>(),
        cancellation,
    )
}

fn decoded(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut values = Vec::with_capacity(count);
    for linear in 0..count {
        let mut remainder = linear;
        let mut indices = vec![0; tensor.descriptor().rank()];
        for (index, dimension) in indices.iter_mut().zip(tensor.descriptor().shape()).rev() {
            let dimension = usize::try_from(*dimension)?;
            *index = u64::try_from(remainder % dimension)?;
            remainder /= dimension;
        }
        values.push(
            tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.element_bytes(&indices)?)?,
        );
    }
    Ok(values)
}

fn real(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    decoded(tensor)?
        .into_iter()
        .map(|value| match value {
            DecodedScalar::Real(value) => Ok(value as f32),
            other => Err(format!("expected real scalar, got {other:?}").into()),
        })
        .collect()
}

#[test]
fn gather_preserves_generic_values_and_has_exact_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        &cancellation,
    )?;
    let index = upload_i64(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[2, 0, 1, 1],
        &cancellation,
    )?;
    assert_eq!(
        real(&gather_function_exact_native(
            &backend,
            &input,
            1,
            &index,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?)?,
        [12.0, 10.0, 21.0, 21.0]
    );
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    assert_eq!(
        real(&gather_jvp_exact_native(
            &backend,
            &input,
            &tangent,
            1,
            &index,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?)?,
        [3.0, 1.0, 5.0, 5.0]
    );
    let upstream = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    assert_eq!(
        real(&gather_vjp_exact_native(
            &backend,
            &input,
            1,
            &index,
            &upstream,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?)?,
        [2.0, 0.0, 1.0, 0.0, 7.0, 0.0]
    );
    Ok(())
}

#[test]
fn index_add_and_masked_fill_stage_before_publication() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut input = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[10.0, 20.0, 30.0, 40.0],
        &cancellation,
    )?;
    let index = upload_i64(
        &backend,
        &workspace_authority,
        &[3],
        &[2, 0, 2],
        &cancellation,
    )?;
    let source = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    index_add_in_place_exact_native(
        &backend,
        &mut input,
        0,
        &index,
        &source,
        1.0,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&input)?, [12.0, 20.0, 34.0, 40.0]);

    let mask = upload_bool(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[true, false],
        &cancellation,
    )?;
    let mut matrix = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    masked_fill_in_place_exact_native(
        &backend,
        &mut matrix,
        &mask,
        sim_tensor::Scalar::Float(-1.0),
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&matrix)?, [-1.0, -1.0, 3.0, 4.0]);
    let upstream = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    assert_eq!(
        real(&masked_fill_vjp_exact_native(
            &backend,
            &matrix,
            &mask,
            &upstream,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?)?,
        [0.0, 0.0, 7.0, 8.0]
    );
    Ok(())
}

#[test]
fn narrow_is_the_canonical_read_only_view_and_vjp_scatter() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let view = narrow_function_exact_native(&input, -1, -2, 1, &cancellation)?;
    assert_eq!(view.storage_id(), input.storage_id());
    assert_eq!(view.access(), ViewAccess::ReadOnly);
    assert_eq!(real(&view)?, [2.0, 5.0]);
    let upstream = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[7.0, 8.0],
        &cancellation,
    )?;
    assert_eq!(
        real(&narrow_vjp_exact_native(
            &backend,
            &input,
            -1,
            -2,
            1,
            &upstream,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?)?,
        [0.0, 7.0, 0.0, 0.0, 8.0, 0.0]
    );
    Ok(())
}

#[test]
fn scatter_is_deterministic_and_derivatives_follow_written_destinations()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        &cancellation,
    )?;
    let index = upload_i64(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[2, 0, 1, 1],
        &cancellation,
    )?;
    let source = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let output = scatter_function_exact_native(
        &backend,
        &input,
        1,
        &index,
        &source,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&output)?, [2.0, 11.0, 1.0, 20.0, 4.0, 22.0]);
    let mut in_place = input.clone();
    scatter_in_place_exact_native(
        &backend,
        &mut in_place,
        1,
        &index,
        &source,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&in_place)?, real(&output)?);
    let upstream = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let gradients = scatter_vjp_exact_native(
        &backend,
        &input,
        1,
        &index,
        &source,
        &upstream,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&gradients.input)?, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    assert_eq!(real(&gradients.source)?, [3.0, 1.0, 5.0, 5.0]);
    Ok(())
}

#[test]
fn nonzero_delegates_coordinate_order_and_tuple_projection()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[0.0, 2.0, 0.0, 4.0, 0.0, 6.0],
        &cancellation,
    )?;
    let NonzeroOutput::Matrix(matrix) = nonzero_exact_native(
        &backend,
        &input,
        false,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?
    else {
        return Err("expected matrix nonzero output".into());
    };
    assert_eq!(
        decoded(&matrix)?,
        [
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
        ]
    );
    let NonzeroOutput::Tuple(columns) = where_nonzero_exact_native(
        &backend,
        &input,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?
    else {
        return Err("expected tuple nonzero output".into());
    };
    assert_eq!(columns.len(), 2);
    assert_eq!(
        decoded(&columns[0])?,
        [
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(1)
        ]
    );
    assert_eq!(
        decoded(&columns[1])?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(2)
        ]
    );
    Ok(())
}

#[test]
fn where_broadcasts_promotes_and_reduces_vjps() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let condition = upload_bool(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[true, false],
        &cancellation,
    )?;
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[10.0, 20.0],
        &cancellation,
    )?;
    let output = where_exact_native(
        &backend,
        &condition,
        &input,
        &other,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&output)?, [1.0, 2.0, 20.0, 20.0]);
    let upstream = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let gradients = where_vjp_exact_native(
        &backend,
        &condition,
        &input,
        &other,
        &upstream,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(real(&gradients.input)?, [1.0, 2.0]);
    assert_eq!(real(&gradients.other)?, [0.0, 7.0]);

    let integer = upload_values(
        &backend,
        &workspace_authority,
        &[1],
        DType::I16,
        &[DecodedScalar::Signed(7)],
        &cancellation,
    )?;
    let promoted = where_exact_native(
        &backend,
        &upload_bool(&backend, &workspace_authority, &[1], &[true], &cancellation)?,
        &integer,
        &upload_f32(&backend, &workspace_authority, &[1], &[2.5], &cancellation)?,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(promoted.descriptor().dtype(), DType::F32);
    assert_eq!(real(&promoted)?, [7.0]);
    Ok(())
}

#[test]
fn cancellation_wins_before_validation_and_mutations_publish_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let setup = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &setup)?;
    let index = upload_i64(&backend, &workspace_authority, &[1], &[9], &setup)?;
    let source = upload_f32(&backend, &workspace_authority, &[1], &[5.0], &setup)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let scratch = workspace_authority.authorize_workspace(16 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);

    macro_rules! assert_cancelled {
        ($name:literal, $expression:expr) => {
            let error = $expression.expect_err(concat!(
                $name,
                " must fail before deliberately invalid argument validation"
            ));
            assert!(
                matches!(error, IndexingMaskingPartOneError::Cancelled),
                "{} returned the wrong pre-cancellation error: {error:?}",
                $name
            );
        };
    }

    assert_cancelled!(
        "gather method",
        gather_method_exact_native(&backend, &input, 99, &index, &execution)
    );
    assert_cancelled!(
        "gather function",
        gather_function_exact_native(&backend, &input, 99, &index, &execution)
    );

    let mut index_add_input = input.clone();
    let index_add_before = real(&index_add_input)?;
    assert_cancelled!(
        "index_add_",
        index_add_in_place_exact_native(
            &backend,
            &mut index_add_input,
            99,
            &index,
            &source,
            1.0,
            &execution,
        )
    );
    assert_eq!(real(&index_add_input)?, index_add_before);

    let mut masked_fill_input = input.clone();
    let masked_fill_before = real(&masked_fill_input)?;
    assert_cancelled!(
        "masked_fill_",
        masked_fill_in_place_exact_native(
            &backend,
            &mut masked_fill_input,
            &input,
            sim_tensor::Scalar::Float(0.0),
            &execution,
        )
    );
    assert_eq!(real(&masked_fill_input)?, masked_fill_before);

    assert_cancelled!(
        "narrow method",
        narrow_method_exact_native(&input, 99, -9, u64::MAX, &cancellation)
    );
    assert_cancelled!(
        "narrow function",
        narrow_function_exact_native(&input, 99, -9, u64::MAX, &cancellation)
    );
    assert_cancelled!(
        "scatter method",
        scatter_method_exact_native(&backend, &input, 99, &index, &source, &execution)
    );
    assert_cancelled!(
        "scatter function",
        scatter_function_exact_native(&backend, &input, 99, &index, &source, &execution)
    );

    let mut scatter_input = input.clone();
    let scatter_before = real(&scatter_input)?;
    assert_cancelled!(
        "scatter_",
        scatter_in_place_exact_native(
            &backend,
            &mut scatter_input,
            99,
            &index,
            &source,
            &execution,
        )
    );
    assert_eq!(real(&scatter_input)?, scatter_before);

    assert_cancelled!(
        "nonzero",
        nonzero_exact_native(&backend, &input, false, &execution)
    );
    assert_cancelled!(
        "where",
        where_exact_native(&backend, &input, &input, &source, &execution)
    );
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn derivative_adapters_check_cancellation_before_validation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let setup = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &setup)?;
    let mismatched = upload_f32(&backend, &workspace_authority, &[1], &[5.0], &setup)?;
    let index = upload_i64(&backend, &workspace_authority, &[1], &[9], &setup)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let scratch = workspace_authority.authorize_workspace(16 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);

    macro_rules! assert_cancelled {
        ($name:literal, $expression:expr) => {
            let error = $expression.expect_err(concat!($name, " must check cancellation first"));
            assert!(
                matches!(error, IndexingMaskingPartOneError::Cancelled),
                "{} returned the wrong pre-cancellation error: {error:?}",
                $name
            );
        };
    }

    assert_cancelled!(
        "gather VJP",
        gather_vjp_exact_native(&backend, &input, 99, &index, &mismatched, &execution)
    );
    assert_cancelled!(
        "gather JVP",
        gather_jvp_exact_native(&backend, &input, &mismatched, 99, &index, &execution)
    );
    assert_cancelled!(
        "index_add_ VJP",
        index_add_vjp_exact_native(
            &backend,
            &input,
            99,
            &index,
            &mismatched,
            1.0,
            &mismatched,
            &execution,
        )
    );
    assert_cancelled!(
        "index_add_ JVP",
        index_add_jvp_exact_native(&backend, &input, &mismatched, 99, &index, 1.0, &execution,)
    );
    assert_cancelled!(
        "masked_fill_ VJP",
        masked_fill_vjp_exact_native(&backend, &input, &input, &mismatched, &execution)
    );
    assert_cancelled!(
        "masked_fill_ JVP",
        masked_fill_jvp_exact_native(&backend, &input, &input, &mismatched, &execution)
    );
    assert_cancelled!(
        "narrow VJP",
        narrow_vjp_exact_native(&backend, &input, 99, -9, u64::MAX, &mismatched, &execution)
    );
    assert_cancelled!(
        "narrow JVP",
        narrow_jvp_exact_native(&input, &mismatched, 99, -9, u64::MAX, &cancellation)
    );
    assert_cancelled!(
        "scatter VJP",
        scatter_vjp_exact_native(
            &backend,
            &input,
            99,
            &index,
            &mismatched,
            &mismatched,
            &execution,
        )
    );
    assert_cancelled!(
        "scatter JVP",
        scatter_jvp_exact_native(
            &backend,
            &input,
            &mismatched,
            99,
            &index,
            &mismatched,
            &input,
            &execution,
        )
    );
    assert_cancelled!(
        "where VJP",
        where_vjp_exact_native(
            &backend,
            &input,
            &input,
            &mismatched,
            &mismatched,
            &execution
        )
    );
    assert_cancelled!(
        "where JVP",
        where_jvp_exact_native(
            &backend,
            &input,
            &input,
            &mismatched,
            &mismatched,
            &input,
            &execution,
        )
    );
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn invalid_indices_and_masks_are_rejected_without_partial_results()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let negative = upload_i64(&backend, &workspace_authority, &[1], &[-1], &cancellation)?;
    assert!(
        gather_function_exact_native(
            &backend,
            &input,
            0,
            &negative,
            &authorized_context(&backend, &workspace_authority, &cancellation)?
        )
        .is_err()
    );
    let numeric_mask = upload_i64(&backend, &workspace_authority, &[1], &[1], &cancellation)?;
    assert!(
        where_exact_native(
            &backend,
            &numeric_mask,
            &input,
            &input,
            &authorized_context(&backend, &workspace_authority, &cancellation)?
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn resolution_slice_seals_every_executable_and_keeps_sqlalchemy_select_external()
-> Result<(), Box<dyn std::error::Error>> {
    let owner = "comfy-parity-tensor-ops-indexing-masking-comfy-tensor-op-006e05c5daaf";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "indexing_masking_01")
        .ok_or("indexing/masking part-one resolution slice is missing")?;
    assert_eq!(slice.len(), EXECUTABLE_IDS.len());
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        EXECUTABLE_IDS.into_iter().collect()
    );
    let mut overloads = BTreeSet::new();
    for contract in slice.iter() {
        assert_eq!(contract.owner_task_id, owner);
        assert!(overloads.insert(contract.overload_id));
        let fixture = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(contract.evidence_fixture),
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&fixture)),
            contract.evidence_fixture_sha256
        );
        let document: serde_json::Value = serde_json::from_slice(&fixture)?;
        assert_eq!(document["operation_id"], contract.operation_id);
        assert_eq!(document["overload_id"], contract.overload_id);
        assert_eq!(document["owner_task_id"], owner);
    }
    assert!(OperationContractId::new("COMFY-TENSOR-OP-C56873FC70F9").is_err());
    Ok(())
}
