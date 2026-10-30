use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    AutocastPolicy, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor,
    TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_10::{
        ElementwiseRuntimePartTenError, NativeCumulativeOperation, PairInput, TensorSize,
        cumsum_jvp_with_context_exact_native, cumsum_vjp_with_context_exact_native,
        cumsum_with_context_exact_native, cumulative_with_context_exact_native,
        fmod_jvp_with_context_exact_native, fmod_vjp_with_context_exact_native,
        fmod_with_context_exact_native, hann_window_with_context_exact_native, iinfo_exact_native,
        is_autocast_cache_enabled_exact_native, log2_jvp_with_context_exact_native,
        log2_vjp_with_context_exact_native, log2_with_context_exact_native, pair_exact_native,
        size_exact_native, unique_consecutive_with_context_exact_native,
        unique_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-75127DF334F2",
    "COMFY-TENSOR-OP-77C67CAAC4AD",
    "COMFY-TENSOR-OP-78FD9BB26FAF",
    "COMFY-TENSOR-OP-73EF9076727A",
    "COMFY-TENSOR-OP-75F89A81FD21",
    "COMFY-TENSOR-OP-6F8F8AE14084",
    "COMFY-TENSOR-OP-6D6C617423EA",
    "COMFY-TENSOR-OP-73E8932FDF3A",
    "COMFY-TENSOR-OP-6E17F49E5F14",
    "COMFY-TENSOR-OP-706EE92A3AD0",
    "COMFY-TENSOR-OP-78CD1B8EFCEC",
    "COMFY-TENSOR-OP-6DEA145A655F",
];

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
}

fn upload_f32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
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
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn upload_i64(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn f32_values(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            cancellation,
        ),
    )?)
}

fn decoded_values(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let shape = tensor.descriptor().shape();
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut values = Vec::new();
    for linear in 0..count {
        let mut remainder = linear;
        let mut indices = vec![0; shape.len()];
        for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
            let dimension = usize::try_from(*dimension)?;
            *slot = u64::try_from(remainder % dimension)?;
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
fn cumulative_shape_dtype_pair_and_autocast_queries_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024)?,
        &cancellation,
    );
    let input = upload_i64(
        &backend,
        &authority,
        &[2, 3],
        &[1, 2, 3, 4, 5, 6],
        &cancellation,
    )?;
    let cumulative = cumsum_with_context_exact_native(&backend, &input, 1, None, &execution)?;
    assert_eq!(
        decoded_values(&cumulative)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(3),
            DecodedScalar::Signed(6),
            DecodedScalar::Signed(4),
            DecodedScalar::Signed(9),
            DecodedScalar::Signed(15),
        ]
    );
    let narrow = upload_i64(&backend, &authority, &[3], &[120, 10, 1], &cancellation)?;
    let narrow =
        cumsum_with_context_exact_native(&backend, &narrow, 0, Some(DType::I8), &execution)?;
    assert_eq!(
        decoded_values(&narrow)?,
        [
            DecodedScalar::Signed(120),
            DecodedScalar::Signed(-126),
            DecodedScalar::Signed(-125),
        ]
    );
    assert_eq!(
        size_exact_native(&input, None, &cancellation)?,
        TensorSize::Shape(vec![2, 3])
    );
    assert_eq!(
        size_exact_native(&input, Some(-1), &cancellation)?,
        TensorSize::Dimension(3)
    );
    let info = iinfo_exact_native(DType::I16, &cancellation)?;
    assert_eq!(
        (info.bits(), info.minimum(), info.maximum()),
        (16, -32_768, 32_767)
    );
    assert_eq!(
        pair_exact_native(PairInput::Scalar(5), &cancellation)?,
        [5, 5]
    );
    assert_eq!(
        pair_exact_native(PairInput::Pair([3, 7]), &cancellation)?,
        [3, 7]
    );
    let autocast = AutocastPolicy::new(true, DType::F16, false)?;
    assert!(!is_autocast_cache_enabled_exact_native(
        &autocast,
        &cancellation
    )?);
    Ok(())
}

#[test]
fn cumulative_and_derivative_workspace_authority_is_exact_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_i64(
        &backend,
        &authority,
        &[2, 3],
        &[1, 2, 3, 4, 5, 6],
        &cancellation,
    )?;
    let gradient = upload_f32(&backend, &authority, &[2, 3], &[1.0; 6], &cancellation)?;
    let baseline = backend.memory_snapshot().current_bytes;
    let cumulative_bytes = u64::try_from(2 * std::mem::size_of::<DecodedScalar>())?
        .checked_add(6 * 8)
        .ok_or("cumulative workspace overflow")?;
    let cumulative_underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(cumulative_bytes - 1)?,
        &cancellation,
    );
    assert!(
        cumsum_with_context_exact_native(&backend, &input, 1, None, &cumulative_underauthorized,)
            .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    let cumulative_exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(cumulative_bytes)?,
        &cancellation,
    );
    let output = cumsum_with_context_exact_native(&backend, &input, 1, None, &cumulative_exact)?;
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let derivative_bytes = (2_u64 * 4) + (6_u64 * 4);
    let derivative_underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(derivative_bytes - 1)?,
        &cancellation,
    );
    assert!(
        cumsum_jvp_with_context_exact_native(&backend, &gradient, 1, &derivative_underauthorized,)
            .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    let derivative_exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(derivative_bytes)?,
        &cancellation,
    );
    let output = cumsum_vjp_with_context_exact_native(&backend, &gradient, 1, &derivative_exact)?;
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let map_underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace((6 * 4) - 1)?,
        &cancellation,
    );
    assert!(
        log2_vjp_with_context_exact_native(&backend, &gradient, &gradient, &map_underauthorized,)
            .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    let map_exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(6 * 4)?,
        &cancellation,
    );
    let output = log2_jvp_with_context_exact_native(&backend, &gradient, &gradient, &map_exact)?;
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(cumulative_bytes)?,
        &cancelled,
    );
    assert!(
        cumsum_with_context_exact_native(&backend, &input, 1, None, &cancelled_context).is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn canonical_fmod_log2_hann_and_derivatives_are_deterministic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64)?,
        &cancellation,
    );
    let zero_scratch = context(&backend, &authority, &cancellation)?;
    let input = upload_f32(&backend, &authority, &[3], &[-5.5, 4.0, 8.0], &cancellation)?;
    let fmod = f32_values(
        &backend,
        &authority,
        &fmod_with_context_exact_native(&backend, &input, 3.0, &zero_scratch)?,
        &cancellation,
    )?;
    assert_eq!(fmod, [-2.5, 1.0, 2.0]);
    let zero_divisor = f32_values(
        &backend,
        &authority,
        &fmod_with_context_exact_native(&backend, &input, 0.0, &zero_scratch)?,
        &cancellation,
    )?;
    assert!(zero_divisor.iter().all(|value| value.is_nan()));
    let tangent = upload_f32(
        &backend,
        &authority,
        &[3],
        &[0.25, 0.5, 0.75],
        &cancellation,
    )?;
    assert_eq!(
        f32_values(
            &backend,
            &authority,
            &fmod_vjp_with_context_exact_native(&backend, &input, &tangent, &zero_scratch)?,
            &cancellation,
        )?,
        f32_values(
            &backend,
            &authority,
            &fmod_jvp_with_context_exact_native(&backend, &input, &tangent, &zero_scratch)?,
            &cancellation,
        )?
    );
    let positive = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 8.0], &cancellation)?;
    assert_eq!(
        f32_values(
            &backend,
            &authority,
            &log2_with_context_exact_native(&backend, &positive, &zero_scratch)?,
            &cancellation,
        )?,
        [0.0, 1.0, 3.0]
    );
    assert_eq!(
        f32_values(
            &backend,
            &authority,
            &log2_vjp_with_context_exact_native(&backend, &positive, &tangent, &execution)?,
            &cancellation,
        )?,
        f32_values(
            &backend,
            &authority,
            &log2_jvp_with_context_exact_native(&backend, &positive, &tangent, &execution)?,
            &cancellation,
        )?
    );
    let hann = hann_window_with_context_exact_native(
        &backend,
        4,
        true,
        DType::F32,
        StreamId::DEFAULT,
        &execution,
    )?;
    let values = f32_values(&backend, &authority, &hann, &cancellation)?;
    assert!((values[0] - 0.0).abs() < f32::EPSILON);
    assert!((values[1] - 0.5).abs() < 1e-6);
    assert!((values[2] - 1.0).abs() < 1e-6);
    assert!((values[3] - 0.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn hann_workspace_authority_is_exact_and_failure_atomic() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let baseline = backend.memory_snapshot().current_bytes;
    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    assert!(
        hann_window_with_context_exact_native(
            &backend,
            4,
            true,
            DType::F32,
            StreamId::DEFAULT,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancellation,
    );
    let output = hann_window_with_context_exact_native(
        &backend,
        4,
        true,
        DType::F32,
        StreamId::DEFAULT,
        &exact,
    )?;
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancelled,
    );
    assert!(
        hann_window_with_context_exact_native(
            &backend,
            4,
            true,
            DType::F32,
            StreamId::DEFAULT,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn unique_variants_preserve_inverse_count_and_dimension_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024)?,
        &cancellation,
    );
    let input = upload_i64(
        &backend,
        &authority,
        &[6],
        &[3, 1, 3, 2, 1, 1],
        &cancellation,
    )?;
    let result =
        unique_with_context_exact_native(&backend, &input, true, true, true, None, &execution)?;
    assert_eq!(
        decoded_values(&result.values)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(3)
        ]
    );
    assert_eq!(
        decoded_values(result.inverse_indices.as_ref().ok_or("inverse missing")?)?,
        [
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(0),
        ]
    );
    assert_eq!(
        decoded_values(result.counts.as_ref().ok_or("counts missing")?)?,
        [
            DecodedScalar::Signed(3),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
        ]
    );

    let rows = upload_i64(
        &backend,
        &authority,
        &[4, 2],
        &[1, 2, 1, 2, 3, 4, 1, 2],
        &cancellation,
    )?;
    let rows =
        unique_with_context_exact_native(&backend, &rows, false, true, true, Some(0), &execution)?;
    assert_eq!(rows.values.descriptor().shape(), &[2, 2]);
    assert_eq!(
        decoded_values(&rows.values)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(3),
            DecodedScalar::Signed(4),
        ]
    );
    let consecutive = upload_i64(
        &backend,
        &authority,
        &[6],
        &[1, 1, 2, 2, 1, 1],
        &cancellation,
    )?;
    let consecutive = unique_consecutive_with_context_exact_native(
        &backend,
        &consecutive,
        true,
        true,
        None,
        &execution,
    )?;
    assert_eq!(
        decoded_values(&consecutive.values)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(1),
        ]
    );

    let gradient = upload_f32(&backend, &authority, &[2, 3], &[1.0; 6], &cancellation)?;
    assert_eq!(
        f32_values(
            &backend,
            &authority,
            &cumsum_vjp_with_context_exact_native(&backend, &gradient, 1, &execution)?,
            &cancellation,
        )?,
        [3.0, 2.0, 1.0, 3.0, 2.0, 1.0]
    );
    assert_eq!(
        f32_values(
            &backend,
            &authority,
            &cumsum_jvp_with_context_exact_native(&backend, &gradient, 1, &execution)?,
            &cancellation,
        )?,
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
    Ok(())
}

#[test]
fn unique_workspace_authority_is_exact_and_failure_atomic() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_i64(
        &backend,
        &authority,
        &[6],
        &[3, 1, 3, 2, 1, 1],
        &cancellation,
    )?;
    let baseline = backend.memory_snapshot().current_bytes;
    let item_count = 6_u64;
    let decoded_bytes = item_count
        .checked_mul(u64::try_from(std::mem::size_of::<DecodedScalar>())?)
        .ok_or("unique workspace overflow")?;
    let index_bytes = item_count
        .checked_mul(4)
        .and_then(|count| count.checked_mul(8))
        .ok_or("unique workspace overflow")?;
    let exact_bytes = decoded_bytes
        .checked_add(index_bytes)
        .ok_or("unique workspace overflow")?;

    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(exact_bytes - 1)?,
        &cancellation,
    );
    assert!(
        unique_with_context_exact_native(
            &backend,
            &input,
            true,
            true,
            true,
            None,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(exact_bytes)?,
        &cancellation,
    );
    let output =
        unique_with_context_exact_native(&backend, &input, true, true, true, None, &exact)?;
    drop(output);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(exact_bytes)?,
        &cancelled,
    );
    assert!(
        unique_with_context_exact_native(
            &backend,
            &input,
            true,
            true,
            true,
            None,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn resolution_contracts_are_unique_and_sealed_by_their_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let owner =
        "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-6d6c617423ea";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_10")
        .ok_or("part-ten resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    let ids = IDS.into_iter().collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), IDS.len());
    let mut overloads = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(ids.contains(contract.operation_id));
        assert_eq!(contract.owner_task_id, owner);
        assert!(overloads.insert(contract.overload_id));
        assert!(digests.insert(contract.evidence_fixture_sha256));
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root missing")?
            .join(contract.evidence_fixture);
        let bytes = fs::read(path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-75127DF334F2" => "cumsum_with_context_exact_native",
            "COMFY-TENSOR-OP-77C67CAAC4AD" => "size_exact_native",
            "COMFY-TENSOR-OP-78FD9BB26FAF" => "fmod_with_context_exact_native",
            "COMFY-TENSOR-OP-73EF9076727A" => "hann_window_with_context_exact_native",
            "COMFY-TENSOR-OP-75F89A81FD21" => "iinfo_exact_native",
            "COMFY-TENSOR-OP-6F8F8AE14084" => "is_autocast_cache_enabled_exact_native",
            "COMFY-TENSOR-OP-6D6C617423EA" => "log2_with_context_exact_native",
            "COMFY-TENSOR-OP-73E8932FDF3A" => "mlu_device_count_exact_native",
            "COMFY-TENSOR-OP-6E17F49E5F14" => "pair_exact_native",
            "COMFY-TENSOR-OP-706EE92A3AD0" => "unique_with_context_exact_native",
            "COMFY-TENSOR-OP-78CD1B8EFCEC" => "unique_consecutive_with_context_exact_native",
            "COMFY-TENSOR-OP-6DEA145A655F" => "xpu_empty_cache_exact_native",
            _ => return Err("unexpected Task 53 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn every_local_task53_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[1], &[1.0], &live)?;
    let integer = upload_i64(&backend, &authority, &[1], &[1], &live)?;
    let input_bytes = input.contiguous_bytes()?.to_vec();
    let integer_bytes = integer.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );

    macro_rules! assert_cancelled {
        ($expression:expr) => {
            assert!(matches!(
                $expression,
                Err(ElementwiseRuntimePartTenError::Cancelled)
            ));
        };
    }

    assert_cancelled!(cumsum_with_context_exact_native(
        &backend,
        &integer,
        99,
        Some(DType::Bool),
        &execution,
    ));
    assert_cancelled!(cumulative_with_context_exact_native(
        &backend,
        &integer,
        99,
        Some(DType::Bool),
        NativeCumulativeOperation::Sum,
        "cancelled-test",
        &execution,
    ));
    assert_cancelled!(cumsum_vjp_with_context_exact_native(
        &backend, &integer, 99, &execution,
    ));
    assert_cancelled!(cumsum_jvp_with_context_exact_native(
        &backend, &integer, 99, &execution,
    ));
    assert_cancelled!(size_exact_native(&integer, Some(99), &cancelled));
    assert_cancelled!(fmod_with_context_exact_native(
        &backend,
        &integer,
        f32::NAN,
        &execution,
    ));
    assert_cancelled!(fmod_vjp_with_context_exact_native(
        &backend, &input, &integer, &execution,
    ));
    assert_cancelled!(fmod_jvp_with_context_exact_native(
        &backend, &input, &integer, &execution,
    ));
    assert_cancelled!(hann_window_with_context_exact_native(
        &backend,
        usize::MAX,
        true,
        DType::Bool,
        StreamId::DEFAULT,
        &execution,
    ));
    assert_cancelled!(iinfo_exact_native(DType::F32, &cancelled));
    let policy = AutocastPolicy::new(true, DType::F16, true)?;
    assert_cancelled!(is_autocast_cache_enabled_exact_native(&policy, &cancelled,));
    assert_cancelled!(log2_with_context_exact_native(
        &backend, &integer, &execution,
    ));
    assert_cancelled!(log2_vjp_with_context_exact_native(
        &backend, &input, &integer, &execution,
    ));
    assert_cancelled!(log2_jvp_with_context_exact_native(
        &backend, &input, &integer, &execution,
    ));
    assert_cancelled!(pair_exact_native(PairInput::Scalar(1), &cancelled));
    assert_cancelled!(unique_with_context_exact_native(
        &backend,
        &integer,
        true,
        true,
        true,
        Some(99),
        &execution,
    ));
    assert_cancelled!(unique_consecutive_with_context_exact_native(
        &backend,
        &integer,
        true,
        true,
        Some(99),
        &execution,
    ));

    assert_eq!(execution.scratch.peak_bytes(), 0);
    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, input_bytes);
    assert_eq!(integer.contiguous_bytes()?, integer_bytes);
    Ok(())
}
