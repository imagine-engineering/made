use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, NativeDeviceProperties,
    NativeStreamRegistry, Scalar, StreamId, Tensor, TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_08::{
        cos_jvp_with_context_exact_native as canonical_cos_jvp,
        cos_vjp_with_context_exact_native as canonical_cos_vjp,
        cos_with_context_exact_native as canonical_cos,
    },
    generated_elementwise_or_runtime_operation_19::{
        ElementwiseRuntimePartNineteenError, cos_method_jvp_with_context_exact_native,
        cos_method_vjp_with_context_exact_native, cos_method_with_context_exact_native,
        cuda_current_device_exact_native, cuda_stream_exact_native, directml_device_exact_native,
        div_in_place_with_context_exact_native, equal_exact_native,
        flip_jvp_with_context_exact_native, flip_vjp_with_context_exact_native,
        flip_with_context_exact_native, mlu_get_device_name_exact_native,
        sort_jvp_with_context_exact_native, sort_vjp_with_context_exact_native,
        sort_with_context_exact_native, tensor_size_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 11] = [
    "COMFY-TENSOR-OP-CC875F3A9DF9",
    "COMFY-TENSOR-OP-CD54624C2360",
    "COMFY-TENSOR-OP-C9CC06A648EC",
    "COMFY-TENSOR-OP-C9C8310F80B5",
    "COMFY-TENSOR-OP-C93863D94FF9",
    "COMFY-TENSOR-OP-CA8F43C066B1",
    "COMFY-TENSOR-OP-C9765FFEEB7F",
    "COMFY-TENSOR-OP-C905902EB028",
    "COMFY-TENSOR-OP-CC4DC3D17ADD",
    "COMFY-TENSOR-OP-C8BA6CE3159C",
    "COMFY-TENSOR-OP-CE66E20937C0",
];

const EXTERNAL_MATH_ATAN2_ID: &str = "COMFY-TENSOR-OP-CA83DE14D96E";

#[test]
fn part_nineteen_workspace_is_exact_bounded_and_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[4],
        &[3.0, 1.0, 4.0, 2.0],
        &cancellation,
    )?;
    let probe = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(4096)?,
        &cancellation,
    );
    sort_with_context_exact_native(&backend, &input, 0, false, true, &probe)?;
    let bytes = probe.scratch.peak_bytes();
    assert!(bytes >= 16);
    assert_eq!(probe.scratch.in_use_bytes(), 0);
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes)?,
        &cancellation,
    );
    sort_with_context_exact_native(&backend, &input, 0, false, true, &exact)?;
    assert_eq!(exact.scratch.peak_bytes(), bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(bytes - 1)?,
        &cancellation,
    );
    assert!(
        sort_with_context_exact_native(&backend, &input, 0, false, true, &insufficient).is_err()
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
        sort_with_context_exact_native(&backend, &input, 0, false, true, &cancelled_context)
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
        bytes.extend(dtype.encode_scalar(*value, "task-62-fixture", DeviceId::CPU)?);
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

fn upload_complex64(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    values: &[(f32, f32)],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|(real, imaginary)| {
            real.to_ne_bytes()
                .into_iter()
                .chain(imaginary.to_ne_bytes())
        })
        .collect::<Vec<_>>();
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::Complex64,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
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
        .map(|linear| {
            let width = usize::try_from(tensor.descriptor().shape().last().copied().unwrap_or(1))?;
            let linear = usize::try_from(linear)?;
            let indices = if tensor.descriptor().rank() == 2 {
                vec![
                    u64::try_from(linear / width)?,
                    u64::try_from(linear % width)?,
                ]
            } else {
                vec![u64::try_from(linear)?]
            };
            let bytes: [u8; 8] = tensor.element_bytes(&indices)?.try_into()?;
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

#[track_caller]
fn assert_cancelled<T>(result: Result<T, ElementwiseRuntimePartNineteenError>) {
    assert!(matches!(
        result,
        Err(ElementwiseRuntimePartNineteenError::Cancelled)
    ));
}

#[test]
fn task_62_resolution_slice_seals_executables_and_keeps_math_atan2_external()
-> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_19")
        .ok_or("Task 62 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    assert_eq!(
        slice
            .contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    assert!(
        slice
            .contracts
            .iter()
            .all(|contract| contract.operation_id != EXTERNAL_MATH_ATAN2_ID)
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
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/native-tensor-operation-contracts.csv"),
    )?;
    let row = catalog
        .lines()
        .find(|line| line.starts_with(EXTERNAL_MATH_ATAN2_ID))
        .ok_or("math.atan2 disposition is missing")?;
    assert!(row.contains("reclassified_external"));
    assert!(row.contains("math.atan2"));
    Ok(())
}

#[test]
fn task_62_every_public_tensor_adapter_observes_cancellation_before_validation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &live)?;
    let mismatched = upload_f32(&backend, &workspace_authority, &[1], &[3.0], &live)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );

    assert_cancelled(tensor_size_exact_native(&[2], &cancelled));
    assert_cancelled(cos_method_with_context_exact_native(
        &backend, &input, &execution,
    ));
    assert_cancelled(cos_method_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(cos_method_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));

    let mut dividend = input.clone();
    assert_cancelled(div_in_place_with_context_exact_native(
        &backend,
        &mut dividend,
        ElementwiseOperand::Tensor(&mismatched),
        &execution,
    ));
    assert_cancelled(flip_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));
    assert_cancelled(flip_vjp_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));
    assert_cancelled(flip_jvp_with_context_exact_native(
        &backend,
        &input,
        &[0, 0],
        &execution,
    ));

    let cpu_capabilities = BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?;
    assert_cancelled(cuda_current_device_exact_native(
        &cpu_capabilities,
        &cancelled,
    ));
    let registry = NativeStreamRegistry::default();
    assert_cancelled(cuda_stream_exact_native(
        &registry,
        &cpu_capabilities,
        DeviceId::CPU,
        0,
        &cancelled,
    ));
    assert_cancelled(equal_exact_native(&input, &mismatched, &cancelled));
    assert_cancelled(mlu_get_device_name_exact_native(
        &cpu_capabilities,
        DeviceId::CPU,
        &cancelled,
    ));

    assert_cancelled(sort_with_context_exact_native(
        &backend, &input, 4, false, true, &execution,
    ));
    assert_cancelled(sort_vjp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        4,
        false,
        true,
        &execution,
    ));
    assert_cancelled(sort_jvp_with_context_exact_native(
        &backend,
        &input,
        &mismatched,
        4,
        false,
        true,
        &execution,
    ));
    assert_cancelled(directml_device_exact_native(
        &cpu_capabilities,
        0,
        &cancelled,
    ));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let cuda_capabilities = BackendCapabilityMatrix::new(cuda, Vec::new(), Vec::new())?;
    let first = cuda_stream_exact_native(&registry, &cuda_capabilities, cuda, 0, &live)?;
    assert_eq!(first.id().get(), 1);
    Ok(())
}

#[test]
fn task_62_size_cosine_and_division_are_only_canonical_adapters() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let size = tensor_size_exact_native(&[2, 0, 4], &cancellation)?;
    assert_eq!(size.as_ref(), [2, 0, 4]);

    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[0.0, 0.5, 1.0],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let adapted = cos_method_with_context_exact_native(&backend, &input, &execution)?;
    let canonical = canonical_cos(&backend, &input, &execution)?;
    assert_eq!(
        adapted.host_storage_bytes()?,
        canonical.host_storage_bytes()?
    );
    let adapted_gradient =
        cos_method_vjp_with_context_exact_native(&backend, &input, &gradient, &execution)?;
    let canonical_gradient = canonical_cos_vjp(&backend, &input, &gradient, &execution)?;
    assert_eq!(
        adapted_gradient.host_storage_bytes()?,
        canonical_gradient.host_storage_bytes()?
    );
    let adapted_tangent =
        cos_method_jvp_with_context_exact_native(&backend, &input, &gradient, &execution)?;
    let canonical_tangent = canonical_cos_jvp(&backend, &input, &gradient, &execution)?;
    assert_eq!(
        adapted_tangent.host_storage_bytes()?,
        canonical_tangent.host_storage_bytes()?
    );

    let cancelled_size = CancellationToken::default();
    cancelled_size.cancel();
    assert!(tensor_size_exact_native(&[1, 2], &cancelled_size).is_err());

    let mut dividend = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[6.0, 9.0],
        &cancellation,
    )?;
    div_in_place_with_context_exact_native(
        &backend,
        &mut dividend,
        ElementwiseOperand::Scalar(Scalar::Float(3.0)),
        &execution,
    )?;
    assert_close(
        &f32_values(&backend, &workspace_authority, &dividend, &cancellation)?,
        &[2.0, 3.0],
    );
    let before = dividend.host_storage_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        div_in_place_with_context_exact_native(
            &backend,
            &mut dividend,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            &backend.execution_context(
                StreamId::DEFAULT,
                workspace_authority.authorize_workspace(1024 * 1024)?,
                &cancelled
            ),
        )
        .is_err()
    );
    assert_eq!(dividend.host_storage_bytes()?, before);
    Ok(())
}

#[test]
fn task_62_flip_reuses_task_58_axis_reversal_and_preserves_maps() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_i64(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1, 2, 3, 4, 5, 6],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let flipped = flip_with_context_exact_native(&backend, &input, &[0, -1], &execution)?;
    assert_eq!(i64_values(&flipped)?, [6, 5, 4, 3, 2, 1]);
    assert_eq!(
        i64_values(&flip_vjp_with_context_exact_native(
            &backend,
            &flipped,
            &[0, 1],
            &execution,
        )?)?,
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        i64_values(&flip_jvp_with_context_exact_native(
            &backend,
            &input,
            &[1],
            &execution,
        )?)?,
        [3, 2, 1, 6, 5, 4]
    );
    assert!(flip_with_context_exact_native(&backend, &input, &[1, -1], &execution).is_err());
    Ok(())
}

#[test]
fn task_62_equal_supports_exact_cross_dtype_values_and_nan_rules() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let floating = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let integer = upload_i64(&backend, &workspace_authority, &[2], &[1, 2], &cancellation)?;
    assert!(equal_exact_native(&floating, &integer, &cancellation)?);
    let different = upload_i64(&backend, &workspace_authority, &[2], &[1, 3], &cancellation)?;
    assert!(!equal_exact_native(&floating, &different, &cancellation)?);
    let different_shape = upload_i64(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[1, 2],
        &cancellation,
    )?;
    assert!(!equal_exact_native(
        &floating,
        &different_shape,
        &cancellation
    )?);
    let nan_left = upload_f32(
        &backend,
        &workspace_authority,
        &[1],
        &[f32::NAN],
        &cancellation,
    )?;
    let nan_right = upload_f32(
        &backend,
        &workspace_authority,
        &[1],
        &[f32::NAN],
        &cancellation,
    )?;
    assert!(!equal_exact_native(&nan_left, &nan_right, &cancellation)?);

    let exact_integer = upload_scalars(
        &backend,
        &workspace_authority,
        &[1],
        DType::U64,
        &[Scalar::Unsigned(9_007_199_254_740_992)],
        &cancellation,
    )?;
    let exact_real = upload_scalars(
        &backend,
        &workspace_authority,
        &[1],
        DType::F64,
        &[Scalar::Float(9_007_199_254_740_992.0)],
        &cancellation,
    )?;
    assert!(equal_exact_native(
        &exact_integer,
        &exact_real,
        &cancellation
    )?);
    let inexact_integer = upload_scalars(
        &backend,
        &workspace_authority,
        &[1],
        DType::U64,
        &[Scalar::Unsigned(9_007_199_254_740_993)],
        &cancellation,
    )?;
    assert!(!equal_exact_native(
        &inexact_integer,
        &exact_real,
        &cancellation
    )?);
    let negative = upload_i64(&backend, &workspace_authority, &[1], &[-1], &cancellation)?;
    let unsigned = upload_scalars(
        &backend,
        &workspace_authority,
        &[1],
        DType::U64,
        &[Scalar::Unsigned(u64::MAX)],
        &cancellation,
    )?;
    assert!(!equal_exact_native(&negative, &unsigned, &cancellation)?);

    let complex_real =
        upload_complex64(&backend, &workspace_authority, &[(1.0, 0.0)], &cancellation)?;
    let real_one = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &cancellation)?;
    assert!(equal_exact_native(&complex_real, &real_one, &cancellation)?);
    let complex_non_real =
        upload_complex64(&backend, &workspace_authority, &[(1.0, 2.0)], &cancellation)?;
    assert!(!equal_exact_native(
        &complex_non_real,
        &real_one,
        &cancellation
    )?);
    Ok(())
}

#[test]
fn task_62_sort_reuses_argsort_and_applies_inverse_vjp_and_permutation_jvp()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[3.0, 1.0, 2.0, 0.0, 4.0, -1.0],
        &cancellation,
    )?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let sorted = sort_with_context_exact_native(&backend, &input, 1, false, true, &execution)?;
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &sorted.values,
            &cancellation,
        )?,
        &[1.0, 2.0, 3.0, -1.0, 0.0, 4.0],
    );
    assert_eq!(i64_values(&sorted.indices)?, [1, 2, 0, 2, 0, 1]);
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &cancellation,
    )?;
    let input_gradient = sort_vjp_with_context_exact_native(
        &backend,
        &input,
        &output_gradient,
        1,
        false,
        true,
        &execution,
    )?;
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &input_gradient,
            &cancellation,
        )?,
        &[30.0, 10.0, 20.0, 50.0, 60.0, 40.0],
    );
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let output_tangent =
        sort_jvp_with_context_exact_native(&backend, &input, &tangent, 1, false, true, &execution)?;
    assert_close(
        &f32_values(
            &backend,
            &workspace_authority,
            &output_tangent,
            &cancellation,
        )?,
        &[2.0, 3.0, 1.0, 6.0, 4.0, 5.0],
    );
    Ok(())
}

#[test]
fn task_62_device_adapters_use_capability_and_stream_owners() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let cuda_capabilities = BackendCapabilityMatrix::new(cuda, Vec::new(), Vec::new())?;
    assert_eq!(
        cuda_current_device_exact_native(&cuda_capabilities, &cancellation)?,
        2
    );
    let registry = NativeStreamRegistry::default();
    let stream = cuda_stream_exact_native(&registry, &cuda_capabilities, cuda, -1, &cancellation)?;
    assert_eq!(stream.device(), cuda);
    assert_eq!(stream.priority(), -1);

    let mlu = DeviceId::new(DeviceKind::Mlu, 3);
    let properties = NativeDeviceProperties::new(mlu, "MLU Native 3", 1024, 1, 0, None, true)?;
    let mlu_capabilities = BackendCapabilityMatrix::new_with_properties(
        mlu,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        mlu_get_device_name_exact_native(&mlu_capabilities, mlu, &cancellation)?,
        "MLU Native 3"
    );

    let directml = DeviceId::new(DeviceKind::DirectMl, 1);
    let directml_capabilities = BackendCapabilityMatrix::new(directml, Vec::new(), Vec::new())?;
    assert_eq!(
        directml_device_exact_native(&directml_capabilities, 1, &cancellation)?,
        directml
    );
    assert!(directml_device_exact_native(&cuda_capabilities, 1, &cancellation).is_err());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(cuda_stream_exact_native(&registry, &cuda_capabilities, cuda, 0, &cancelled).is_err());
    let next = cuda_stream_exact_native(&registry, &cuda_capabilities, cuda, 0, &cancellation)?;
    assert_eq!(next.id().get(), stream.id().get() + 1);
    Ok(())
}
