use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor,
    generated_linear_algebra_01::{
        LinearAlgebraPartOneError, vector_norm_jvp_with_context_exact_native,
        vector_norm_vjp_with_context_exact_native, vector_norm_with_context_exact_native,
    },
    generated_linear_algebra_02::norm_with_context_exact_native,
    generated_reduction_02::torch_norm_with_context_exact_native,
    generated_reduction_03::{
        ReductionPartThreeError, TENSOR_NORM_OPERATION_ID,
        tensor_norm_jvp_with_context_exact_native, tensor_norm_vjp_with_context_exact_native,
        tensor_norm_with_context_exact_native,
    },
};
use sha2::{Digest, Sha256};
use std::{fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(16 * 1024 * 1024)?,
            cancellation,
        ))
    }

    fn tensor(
        &self,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::F32,
            sim_tensor::DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self.backend.upload_f32(descriptor, values, context)?.0)
    }

    fn boolean_tensor(
        &self,
        shape: &[u64],
        values: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::Bool,
            sim_tensor::DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self.backend.upload_bytes(descriptor, values, context)?.0)
    }
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| Ok(f32::from_ne_bytes(bytes.try_into()?)))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn tensor_norm_matches_source_used_axes_keepdim_and_default_order()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0], &context)?;

    let rows = tensor_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        Some(&[-1]),
        true,
        None,
        &context,
    )?;
    assert_eq!(rows.descriptor().shape(), &[2, 1]);
    assert_ne!(rows.storage_id(), input.storage_id());
    assert_close(&values(&rows)?, &[5.0, 13.0]);

    let all = tensor_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        None,
        false,
        None,
        &context,
    )?;
    assert!(all.descriptor().shape().is_empty());
    assert_close(&values(&all)?, &[194.0_f32.sqrt()]);
    Ok(())
}

#[test]
fn all_norm_facades_share_task_73_default_axis_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0], &context)?;
    let canonical = vector_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        &[0, 1],
        false,
        None,
        &context,
    )?;
    let linear_algebra = norm_with_context_exact_native(
        &backend, &input, 2.0, None, false, None, &context,
    )?;
    let function = torch_norm_with_context_exact_native(
        &backend, &input, 2.0, None, false, None, &context,
    )?;
    let method = tensor_norm_with_context_exact_native(
        &backend, &input, 2.0, None, false, None, &context,
    )?;
    let expected = values(&canonical)?;
    assert_eq!(values(&linear_algebra)?, expected);
    assert_eq!(values(&function)?, expected);
    assert_eq!(values(&method)?, expected);
    Ok(())
}

#[test]
fn tensor_norm_vjp_and_jvp_are_exact_task_73_adapters()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0], &context)?;
    let tangent = backend.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?;
    let upstream = backend.tensor(&[2], &[2.0, 3.0], &context)?;

    let adapted_vjp = tensor_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        2.0,
        Some(&[1]),
        false,
        None,
        &context,
    )?;
    let canonical_vjp = vector_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        2.0,
        &[1],
        false,
        None,
        &context,
    )?;
    assert_close(&values(&adapted_vjp)?, &values(&canonical_vjp)?);

    let adapted_jvp = tensor_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        2.0,
        Some(&[1]),
        false,
        None,
        &context,
    )?;
    let canonical_jvp = vector_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        2.0,
        &[1],
        false,
        None,
        &context,
    )?;
    assert_close(&values(&adapted_jvp)?, &values(&canonical_jvp)?);
    Ok(())
}

#[test]
fn cancellation_precedes_dimension_validation_and_unsupported_dtype_is_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let active = CancellationToken::default();
    let active_context = backend.execution(&active)?;
    let input = backend.tensor(&[2], &[1.0, 2.0], &active_context)?;
    let boolean = backend.boolean_tensor(&[2], &[1, 0], &active_context)?;

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution(&cancelled)?;
    let cancelled_result = tensor_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        Some(&[99]),
        false,
        None,
        &cancelled_context,
    );
    assert!(matches!(
        cancelled_result,
        Err(ReductionPartThreeError::LinearAlgebra(
            LinearAlgebraPartOneError::Cancelled
        ))
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);

    let unsupported = tensor_norm_with_context_exact_native(
        &backend,
        &boolean,
        2.0,
        None,
        false,
        None,
        &active_context,
    );
    assert!(matches!(
        unsupported,
        Err(ReductionPartThreeError::LinearAlgebra(
            LinearAlgebraPartOneError::UnsupportedDType { .. }
        ))
    ));
    Ok(())
}

#[test]
fn tensor_norm_resolution_is_unique_source_profiled_and_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.contracts.iter())
        .filter(|contract| contract.operation_id == TENSOR_NORM_OPERATION_ID)
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), 1);
    let contract = contracts.first().ok_or("missing Tensor.norm resolution")?;
    assert_eq!(contract.resolution_module, "reduction_03");
    assert_eq!(
        contract.owner_task_id,
        "comfy-parity-tensor-ops-reduction-comfy-tensor-op-ff3f06b4b591"
    );
    assert!(contract.numeric_rule.contains("Task 73 exclusively owns"));

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        contract.evidence_fixture_sha256
    );
    Ok(())
}
