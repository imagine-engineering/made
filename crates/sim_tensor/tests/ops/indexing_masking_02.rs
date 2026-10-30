use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor,
    TensorDescriptor,
    generated_indexing_masking_01::{
        IndexingMaskingPartOneError, NonzeroOutput,
        nonzero_with_context_exact_native as nonzero_exact_native,
    },
    generated_indexing_masking_02::{
        IndexingMaskingPartTwoError,
        masked_fill_method_jvp_with_context_exact_native as masked_fill_method_jvp_exact_native,
        masked_fill_method_vjp_with_context_exact_native as masked_fill_method_vjp_exact_native,
        masked_fill_method_with_context_exact_native as masked_fill_method_exact_native,
        nonzero_method_with_context_exact_native as nonzero_method_exact_native,
    },
};
use std::{collections::BTreeSet, fs, path::Path};

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

fn upload_bool(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[bool],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::Bool,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let bytes = values.iter().copied().map(u8::from).collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
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

#[test]
fn workspace_masked_fill_adapter_preserves_canonical_lease_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let mask = upload_bool(
        &backend,
        &workspace_authority,
        &[4],
        &[true, false, true, false],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[1.0; 4],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(16)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let output =
        masked_fill_method_vjp_exact_native(&backend, &input, &mask, &gradient, &execution)?;
    assert_eq!(
        decoded(&output)?,
        [
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(1.0),
        ]
    );
    assert_eq!(scratch.peak_bytes(), 16);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn masked_fill_method_is_fresh_and_delegates_forward_vjp_and_jvp()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let mask = upload_bool(
        &backend,
        &workspace_authority,
        &[2, 1],
        &[true, false],
        &cancellation,
    )?;
    let output = masked_fill_method_exact_native(
        &backend,
        &input,
        &mask,
        sim_tensor::Scalar::Float(-1.0),
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_ne!(output.storage_id(), input.storage_id());
    assert_eq!(
        decoded(&input)?,
        [
            DecodedScalar::Real(1.0),
            DecodedScalar::Real(2.0),
            DecodedScalar::Real(3.0),
            DecodedScalar::Real(4.0),
        ]
    );
    assert_eq!(
        decoded(&output)?,
        [
            DecodedScalar::Real(-1.0),
            DecodedScalar::Real(-1.0),
            DecodedScalar::Real(3.0),
            DecodedScalar::Real(4.0),
        ]
    );
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let vjp = masked_fill_method_vjp_exact_native(
        &backend,
        &input,
        &mask,
        &tangent,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    let jvp = masked_fill_method_jvp_exact_native(
        &backend,
        &input,
        &mask,
        &tangent,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(decoded(&vjp)?, decoded(&jvp)?);
    assert_eq!(
        decoded(&vjp)?,
        [
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(0.0),
            DecodedScalar::Real(7.0),
            DecodedScalar::Real(8.0),
        ]
    );
    Ok(())
}

#[test]
fn nonzero_method_is_exactly_the_task71_projection() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[0.0, 2.0, 3.0, 0.0],
        &cancellation,
    )?;
    for as_tuple in [false, true] {
        let canonical = nonzero_exact_native(
            &backend,
            &input,
            as_tuple,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?;
        let method = nonzero_method_exact_native(
            &backend,
            &input,
            as_tuple,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?;
        match (canonical, method) {
            (NonzeroOutput::Matrix(canonical), NonzeroOutput::Matrix(method)) => {
                assert_eq!(decoded(&canonical)?, decoded(&method)?);
            }
            (NonzeroOutput::Tuple(canonical), NonzeroOutput::Tuple(method)) => {
                assert_eq!(canonical.len(), method.len());
                for (canonical, method) in canonical.iter().zip(&method) {
                    assert_eq!(decoded(canonical)?, decoded(method)?);
                }
            }
            _ => return Err("nonzero adapter changed its output kind".into()),
        }
    }
    Ok(())
}

#[test]
fn adapters_preserve_cancellation_precedence_and_sealed_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let setup = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &setup)?;
    let invalid_mask = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &setup)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let input_before = decoded(&input)?;

    macro_rules! assert_canonical_cancelled {
        ($name:literal, $expression:expr) => {
            let error = $expression.expect_err(concat!($name, " must delegate cancellation first"));
            assert!(
                matches!(
                    error,
                    IndexingMaskingPartTwoError::Canonical(IndexingMaskingPartOneError::Cancelled)
                ),
                "{} returned the wrong cancellation error: {error:?}",
                $name
            );
        };
    }

    assert_canonical_cancelled!(
        "masked-fill forward",
        masked_fill_method_exact_native(
            &backend,
            &input,
            &invalid_mask,
            sim_tensor::Scalar::Float(0.0),
            &execution,
        )
    );
    assert_canonical_cancelled!(
        "masked-fill VJP",
        masked_fill_method_vjp_exact_native(&backend, &input, &invalid_mask, &input, &execution,)
    );
    assert_canonical_cancelled!(
        "masked-fill JVP",
        masked_fill_method_jvp_exact_native(&backend, &input, &invalid_mask, &input, &execution,)
    );
    assert_canonical_cancelled!(
        "nonzero method",
        nonzero_method_exact_native(&backend, &input, true, &execution)
    );
    assert_eq!(decoded(&input)?, input_before);
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);

    let owner = "comfy-parity-tensor-ops-indexing-masking-comfy-tensor-op-e9a313720d5d";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "indexing_masking_02")
        .ok_or("indexing/masking part-two resolution slice is missing")?;
    assert_eq!(slice.len(), 2);
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        [
            "COMFY-TENSOR-OP-E9A313720D5D",
            "COMFY-TENSOR-OP-F76D5ACB74F3"
        ]
        .into_iter()
        .collect()
    );
    for contract in slice.iter() {
        assert_eq!(contract.owner_task_id, owner);
        let fixture = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(contract.evidence_fixture),
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&fixture)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
