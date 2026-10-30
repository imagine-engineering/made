use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor,
    generated_linear_algebra_01::{
        vector_norm_jvp_with_context_exact_native, vector_norm_vjp_with_context_exact_native,
        vector_norm_with_context_exact_native,
    },
    generated_reduction_02::{
        DifferentiableReductionPartTwo, ReductionPartTwoError, TENSOR_AMAX_OPERATION_ID,
        TENSOR_AMIN_OPERATION_ID, TENSOR_ARGMIN_OPERATION_ID, TENSOR_STD_OPERATION_ID,
        TENSOR_SUM_OPERATION_ID, TORCH_AMAX_OPERATION_ID, TORCH_ANY_OPERATION_ID,
        TORCH_ARGMAX_OPERATION_ID, TORCH_MEAN_OPERATION_ID, TORCH_MIN_OPERATION_ID,
        TORCH_NORM_OPERATION_ID, TORCH_SUM_OPERATION_ID, TorchMinimumArgument,
        reduction_part_two_jvp_with_context_exact_native,
        reduction_part_two_vjp_with_context_exact_native,
        tensor_amax_with_context_exact_native, tensor_amin_with_context_exact_native,
        tensor_argmin_with_context_exact_native, tensor_std_with_context_exact_native,
        tensor_sum_with_context_exact_native, torch_amax_with_context_exact_native,
        torch_any_with_context_exact_native, torch_argmax_with_context_exact_native,
        torch_mean_with_context_exact_native, torch_min_with_context_exact_native,
        torch_norm_jvp_with_context_exact_native, torch_norm_vjp_with_context_exact_native,
        torch_norm_with_context_exact_native, torch_sum_with_context_exact_native,
    },
};
use sha2::{Digest, Sha256};
use std::{fs, ops::Deref, path::Path};

const OPERATION_IDS: [&str; 12] = [
    TENSOR_AMAX_OPERATION_ID,
    TENSOR_AMIN_OPERATION_ID,
    TENSOR_ARGMIN_OPERATION_ID,
    TENSOR_STD_OPERATION_ID,
    TENSOR_SUM_OPERATION_ID,
    TORCH_AMAX_OPERATION_ID,
    TORCH_ANY_OPERATION_ID,
    TORCH_ARGMAX_OPERATION_ID,
    TORCH_MEAN_OPERATION_ID,
    TORCH_MIN_OPERATION_ID,
    TORCH_NORM_OPERATION_ID,
    TORCH_SUM_OPERATION_ID,
];

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(32 * 1024 * 1024)?,
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

    fn tensor_bytes(
        &self,
        shape: &[u64],
        dtype: DType,
        bytes: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            dtype,
            sim_tensor::DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self.backend.upload_bytes(descriptor, bytes, context)?.0)
    }
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| Ok(f32::from_ne_bytes(bytes.try_into()?)))
        .collect()
}

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(8)
        .map(|bytes| Ok(i64::from_ne_bytes(bytes.try_into()?)))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{actual} != {expected}");
    }
}

#[test]
fn tensor_facades_reuse_canonical_extrema_arg_sum_and_standard_deviation()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[1.0, 4.0, 4.0, -1.0, 2.0, 3.0], &context)?;

    let maximum =
        tensor_amax_with_context_exact_native(&backend, &input, Some(&[0, 1]), true, &context)?;
    assert_eq!(maximum.descriptor().shape(), [1, 1]);
    assert_eq!(f32_values(&maximum)?, [4.0]);
    let minimum = tensor_amin_with_context_exact_native(&backend, &input, Some(&[-1]), false, &context)?;
    assert_eq!(f32_values(&minimum)?, [1.0, -1.0]);
    let argmin = tensor_argmin_with_context_exact_native(&backend, &input, Some(1), true, &context)?;
    assert_eq!(argmin.descriptor().shape(), [2, 1]);
    assert_eq!(i64_values(&argmin)?, [0, 0]);
    let sum = tensor_sum_with_context_exact_native(
        &backend,
        &input,
        Some(&[1]),
        false,
        None,
        &context,
    )?;
    assert_eq!(f32_values(&sum)?, [9.0, 4.0]);
    let standard_deviation = tensor_std_with_context_exact_native(
        &backend,
        &input,
        Some(&[1]),
        0,
        false,
        &context,
    )?;
    assert_close(
        &f32_values(&standard_deviation)?,
        &[std::f32::consts::SQRT_2, 1.6996732],
    );
    Ok(())
}

#[test]
fn torch_facades_preserve_function_and_method_overloads()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[0.0, 4.0, 4.0, -1.0, 2.0, 3.0], &context)?;

    assert_eq!(
        torch_any_with_context_exact_native(&backend, &input, Some(&[1]), false, &context)?
            .contiguous_bytes()?,
        &[1, 1]
    );
    assert_eq!(
        i64_values(&torch_argmax_with_context_exact_native(
            &backend,
            &input,
            Some(1),
            false,
            &context,
        )?)?,
        [1, 2]
    );
    assert_eq!(
        f32_values(&torch_amax_with_context_exact_native(
            &backend,
            &input,
            Some(&[0]),
            false,
            &context,
        )?)?,
        [0.0, 4.0, 4.0]
    );
    assert_close(
        &f32_values(&torch_mean_with_context_exact_native(
            &backend,
            &input,
            Some(&[0]),
            false,
            None,
            &context,
        )?)?,
        &[-0.5, 3.0, 3.5],
    );
    assert_eq!(
        f32_values(&torch_sum_with_context_exact_native(
            &backend,
            &input,
            None,
            false,
            None,
            &context,
        )?)?,
        [12.0]
    );
    Ok(())
}

#[test]
fn sum_preserves_source_boolean_and_integral_promotion() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let boolean = backend.tensor_bytes(&[3], DType::Bool, &[1, 0, 1], &context)?;
    let integer_bytes = [1_i32, -2, 3]
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect::<Vec<_>>();
    let integer = backend.tensor_bytes(&[3], DType::I32, &integer_bytes, &context)?;

    let boolean_sum = torch_sum_with_context_exact_native(
        &backend,
        &boolean,
        None,
        false,
        None,
        &context,
    )?;
    assert_eq!(boolean_sum.descriptor().dtype(), DType::I64);
    assert_eq!(i64_values(&boolean_sum)?, [2]);

    let integer_sum = tensor_sum_with_context_exact_native(
        &backend,
        &integer,
        None,
        false,
        None,
        &context,
    )?;
    assert_eq!(integer_sum.descriptor().dtype(), DType::I64);
    assert_eq!(i64_values(&integer_sum)?, [2]);

    let converted = torch_sum_with_context_exact_native(
        &backend,
        &boolean,
        None,
        false,
        Some(DType::F32),
        &context,
    )?;
    assert_eq!(f32_values(&converted)?, [2.0]);
    Ok(())
}

#[test]
fn torch_min_delegates_reduction_and_binary_minimum_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[0.0, 4.0, 4.0, -1.0, 2.0, 3.0], &context)?;
    let reduced = torch_min_with_context_exact_native(
        &backend,
        &input,
        TorchMinimumArgument::Dimension(1),
        false,
        &context,
    )?;
    assert_eq!(f32_values(&reduced.values)?, [0.0, -1.0]);
    assert_eq!(i64_values(reduced.indices.as_ref().ok_or("missing indices")?)?, [0, 0]);

    let other = backend.tensor(&[1, 3], &[1.0, 1.0, 5.0], &context)?;
    let elementwise = torch_min_with_context_exact_native(
        &backend,
        &input,
        TorchMinimumArgument::Tensor(&other),
        false,
        &context,
    )?;
    assert_eq!(f32_values(&elementwise.values)?, [0.0, 1.0, 4.0, -1.0, 1.0, 3.0]);
    assert!(elementwise.indices.is_none());
    Ok(())
}

#[test]
fn analytical_maps_delegate_sum_and_tied_extrema_to_task_83()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[4.0, 4.0, 1.0, 2.0, 3.0, 3.0], &context)?;
    let tangent = backend.tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &context)?;
    let sum = tensor_sum_with_context_exact_native(
        &backend,
        &input,
        Some(&[1]),
        false,
        None,
        &context,
    )?;
    let upstream = backend.tensor(&[2], &[2.0, 4.0], &context)?;
    let sum_vjp = reduction_part_two_vjp_with_context_exact_native(
        &backend,
        &input,
        &sum,
        None,
        &upstream,
        Some(&[1]),
        false,
        DifferentiableReductionPartTwo::TensorSum,
        &context,
    )?;
    assert_eq!(f32_values(&sum_vjp)?, [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]);
    let sum_jvp = reduction_part_two_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &sum,
        None,
        Some(&[1]),
        false,
        DifferentiableReductionPartTwo::TensorSum,
        &context,
    )?;
    assert_eq!(f32_values(&sum_jvp)?, [6.0, 15.0]);

    let maximum = tensor_amax_with_context_exact_native(
        &backend,
        &input,
        Some(&[1]),
        false,
        &context,
    )?;
    let maximum_vjp = reduction_part_two_vjp_with_context_exact_native(
        &backend,
        &input,
        &maximum,
        None,
        &upstream,
        Some(&[1]),
        false,
        DifferentiableReductionPartTwo::TensorAmax,
        &context,
    )?;
    assert_eq!(f32_values(&maximum_vjp)?, [1.0, 1.0, 0.0, 0.0, 2.0, 2.0]);
    Ok(())
}

#[test]
fn torch_norm_is_a_thin_task_73_forward_vjp_and_jvp_adapter()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 2], &[3.0, 4.0, 5.0, 12.0], &context)?;
    let tangent = backend.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?;
    let upstream = backend.tensor(&[2], &[2.0, 3.0], &context)?;

    let adapted = torch_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        Some(&[1]),
        false,
        None,
        &context,
    )?;
    let canonical = vector_norm_with_context_exact_native(
        &backend,
        &input,
        2.0,
        &[1],
        false,
        None,
        &context,
    )?;
    assert_eq!(f32_values(&adapted)?, f32_values(&canonical)?);
    assert_eq!(f32_values(&adapted)?, [5.0, 13.0]);

    let adapted_vjp = torch_norm_vjp_with_context_exact_native(
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
    assert_close(&f32_values(&adapted_vjp)?, &f32_values(&canonical_vjp)?);

    let adapted_jvp = torch_norm_jvp_with_context_exact_native(
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
    assert_close(&f32_values(&adapted_jvp)?, &f32_values(&canonical_jvp)?);
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_dimensions_and_publishes_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let active = CancellationToken::default();
    let active_context = backend.execution(&active)?;
    let input = backend.tensor(&[2], &[1.0, 2.0], &active_context)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let context = backend.execution(&cancelled)?;
    let result = tensor_sum_with_context_exact_native(
        &backend,
        &input,
        Some(&[99]),
        false,
        None,
        &context,
    );
    assert!(matches!(
        result,
        Err(ReductionPartTwoError::Reduction(
            sim_tensor::generated_reduction_01::ReductionPartOneError::Tensor(
                sim_tensor::TensorError::Cancelled
            )
        ))
    ));
    Ok(())
}

#[test]
fn reduction_part_two_resolutions_are_unique_source_profiled_and_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    for operation_id in OPERATION_IDS {
        let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.contracts.iter())
            .filter(|contract| contract.operation_id == operation_id)
            .collect::<Vec<_>>();
        assert_eq!(contracts.len(), 1, "{operation_id}");
        let contract = contracts.first().ok_or("missing reduction resolution")?;
        assert_eq!(contract.resolution_module, "reduction_02");
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-reduction-comfy-tensor-op-9f681b3616f6"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
