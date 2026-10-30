use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, PrimitiveOperation, ReductionOperation,
    StreamId, Tensor, TensorDescriptor,
    generated_reduction_01::{
        DifferentiableReduction, ReductionPartOneError, TENSOR_ALL_OPERATION_ID,
        TENSOR_ANY_OPERATION_ID, TENSOR_ARGMAX_OPERATION_ID, TENSOR_MAX_OPERATION_ID,
        TENSOR_MEAN_OPERATION_ID, TENSOR_MIN_OPERATION_ID, TENSOR_PROD_OPERATION_ID,
        TENSOR_VAR_OPERATION_ID, TORCH_ALL_OPERATION_ID, TORCH_ARGMIN_OPERATION_ID,
        TORCH_MAX_OPERATION_ID, TORCH_STD_OPERATION_ID, TorchMaximumArgument,
        reduction_jvp_with_context_exact_native, reduction_vjp_with_context_exact_native,
        tensor_all_with_context_exact_native, tensor_any_with_context_exact_native,
        tensor_argmax_with_context_exact_native, tensor_max_with_context_exact_native,
        tensor_mean_with_context_exact_native, tensor_min_with_context_exact_native,
        tensor_prod_with_context_exact_native, tensor_var_with_context_exact_native,
        torch_all_with_context_exact_native, torch_argmin_with_context_exact_native,
        torch_max_with_context_exact_native, torch_std_with_context_exact_native,
    },
};
use sha2::{Digest, Sha256};
use std::{fs, ops::Deref, path::Path};

const OPERATION_IDS: [&str; 12] = [
    TENSOR_ALL_OPERATION_ID,
    TENSOR_ANY_OPERATION_ID,
    TENSOR_ARGMAX_OPERATION_ID,
    TENSOR_MAX_OPERATION_ID,
    TENSOR_MEAN_OPERATION_ID,
    TENSOR_MIN_OPERATION_ID,
    TENSOR_PROD_OPERATION_ID,
    TENSOR_VAR_OPERATION_ID,
    TORCH_ALL_OPERATION_ID,
    TORCH_ARGMIN_OPERATION_ID,
    TORCH_MAX_OPERATION_ID,
    TORCH_STD_OPERATION_ID,
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
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn canonical_backend_executes_logical_arg_and_extrema_reductions()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[1.0, 0.0, 3.0, 4.0, -1.0, 4.0], &context)?;

    let all = tensor_all_with_context_exact_native(&backend, &input, Some(&[1]), false, &context)?;
    assert_eq!(all.contiguous_bytes()?, &[0, 1]);
    let any = tensor_any_with_context_exact_native(&backend, &input, Some(&[0]), true, &context)?;
    assert_eq!(any.descriptor().shape(), [1, 3]);
    assert_eq!(any.contiguous_bytes()?, &[1, 1, 1]);

    let argmax =
        tensor_argmax_with_context_exact_native(&backend, &input, Some(-1), false, &context)?;
    assert_eq!(i64_values(&argmax)?, [2, 0]);
    let maximum = tensor_max_with_context_exact_native(&backend, &input, Some(1), true, &context)?;
    assert_eq!(maximum.values.descriptor().shape(), [2, 1]);
    assert_eq!(f32_values(&maximum.values)?, [3.0, 4.0]);
    assert_eq!(
        i64_values(maximum.indices.as_ref().ok_or("missing max indices")?)?,
        [2, 0]
    );

    let minimum = tensor_min_with_context_exact_native(&backend, &input, Some(1), false, &context)?;
    assert_eq!(f32_values(&minimum.values)?, [0.0, -1.0]);
    assert_eq!(
        i64_values(minimum.indices.as_ref().ok_or("missing min indices")?)?,
        [1, 1]
    );
    Ok(())
}

#[test]
fn canonical_backend_executes_mean_product_variance_and_standard_deviation()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[1.0, 0.0, 3.0, 4.0, -1.0, 4.0], &context)?;

    let mean =
        tensor_mean_with_context_exact_native(&backend, &input, Some(&[0]), false, None, &context)?;
    assert_eq!(f32_values(&mean)?, [2.5, -0.5, 3.5]);
    let product =
        tensor_prod_with_context_exact_native(&backend, &input, Some(1), false, None, &context)?;
    assert_eq!(f32_values(&product)?, [0.0, -16.0]);

    let variance =
        tensor_var_with_context_exact_native(&backend, &input, None, 1, false, &context)?;
    assert_close(&f32_values(&variance)?, &[4.5666666]);
    let standard_deviation =
        torch_std_with_context_exact_native(&backend, &input, Some(&[1]), 0, false, &context)?;
    assert_close(&f32_values(&standard_deviation)?, &[1.2472191, 2.3570225]);
    Ok(())
}

#[test]
fn analytical_vjp_and_jvp_cover_mean_product_variance_std_and_extrema()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
    let tangent = backend.tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
    let mean =
        tensor_mean_with_context_exact_native(&backend, &input, Some(&[1]), false, None, &context)?;
    let upstream = backend.tensor(&[2], &[2.0, 4.0], &context)?;
    let mean_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &input,
        &mean,
        None,
        &upstream,
        Some(&[1]),
        false,
        DifferentiableReduction::Mean,
        &context,
    )?;
    assert_eq!(f32_values(&mean_vjp)?, [1.0, 1.0, 2.0, 2.0]);
    let mean_jvp = reduction_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &mean,
        None,
        Some(&[1]),
        false,
        DifferentiableReduction::Mean,
        &context,
    )?;
    assert_eq!(f32_values(&mean_jvp)?, [1.5, 3.5]);

    let product_input = backend.tensor(&[2, 2], &[2.0, 0.0, 3.0, 4.0], &context)?;
    let product = tensor_prod_with_context_exact_native(
        &backend,
        &product_input,
        Some(1),
        false,
        None,
        &context,
    )?;
    let product_upstream = backend.tensor(&[2], &[1.0, 1.0], &context)?;
    let product_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &product_input,
        &product,
        None,
        &product_upstream,
        Some(&[1]),
        false,
        DifferentiableReduction::Product,
        &context,
    )?;
    assert_eq!(f32_values(&product_vjp)?, [0.0, 2.0, 4.0, 3.0]);

    let variance =
        tensor_var_with_context_exact_native(&backend, &input, Some(&[1]), 0, false, &context)?;
    let unit_upstream = backend.tensor(&[2], &[1.0, 1.0], &context)?;
    let variance_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &input,
        &variance,
        None,
        &unit_upstream,
        Some(&[1]),
        false,
        DifferentiableReduction::Variance { correction: 0 },
        &context,
    )?;
    assert_eq!(f32_values(&variance_vjp)?, [-0.5, 0.5, -0.5, 0.5]);
    let standard_deviation =
        torch_std_with_context_exact_native(&backend, &input, Some(&[1]), 0, false, &context)?;
    let standard_deviation_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &input,
        &standard_deviation,
        None,
        &unit_upstream,
        Some(&[1]),
        false,
        DifferentiableReduction::StandardDeviation { correction: 0 },
        &context,
    )?;
    assert_eq!(f32_values(&standard_deviation_vjp)?, [-0.5, 0.5, -0.5, 0.5]);

    let maximum = tensor_max_with_context_exact_native(&backend, &input, Some(1), false, &context)?;
    let maximum_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &input,
        &maximum.values,
        maximum.indices.as_ref(),
        &upstream,
        Some(&[1]),
        false,
        DifferentiableReduction::Maximum,
        &context,
    )?;
    assert_eq!(f32_values(&maximum_vjp)?, [0.0, 2.0, 0.0, 4.0]);

    let tied = backend.tensor(&[4], &[1.0, 4.0, 4.0, 2.0], &context)?;
    let tied_maximum =
        tensor_max_with_context_exact_native(&backend, &tied, None, false, &context)?;
    let scalar_upstream = backend.tensor(&[], &[2.0], &context)?;
    let tied_vjp = reduction_vjp_with_context_exact_native(
        &backend,
        &tied,
        &tied_maximum.values,
        None,
        &scalar_upstream,
        None,
        false,
        DifferentiableReduction::Maximum,
        &context,
    )?;
    assert_eq!(f32_values(&tied_vjp)?, [0.0, 1.0, 1.0, 0.0]);
    Ok(())
}

#[test]
fn torch_facades_delegate_to_canonical_reduction_and_elementwise_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = backend.tensor(&[2, 3], &[1.0, 0.0, 3.0, 4.0, -1.0, 4.0], &context)?;
    let all = torch_all_with_context_exact_native(&backend, &input, None, false, &context)?;
    assert_eq!(all.contiguous_bytes()?, &[0]);
    let argmin = torch_argmin_with_context_exact_native(&backend, &input, None, false, &context)?;
    assert_eq!(i64_values(&argmin)?, [4]);
    let maximum = torch_max_with_context_exact_native(
        &backend,
        &input,
        TorchMaximumArgument::Dimension(1),
        false,
        &context,
    )?;
    assert_eq!(f32_values(&maximum.values)?, [3.0, 4.0]);
    let other = backend.tensor(&[2, 3], &[2.0; 6], &context)?;
    let elementwise = torch_max_with_context_exact_native(
        &backend,
        &input,
        TorchMaximumArgument::Tensor(&other),
        false,
        &context,
    )?;
    assert_eq!(
        f32_values(&elementwise.values)?,
        [2.0, 2.0, 3.0, 4.0, 2.0, 4.0]
    );
    assert!(elementwise.indices.is_none());
    Ok(())
}

#[test]
fn empty_domains_identities_invalid_dimensions_and_cancellation_match_contract()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let empty = backend.tensor(&[0], &[], &context)?;
    assert_eq!(
        tensor_all_with_context_exact_native(&backend, &empty, None, false, &context)?
            .contiguous_bytes()?,
        &[1]
    );
    assert_eq!(
        tensor_any_with_context_exact_native(&backend, &empty, None, false, &context)?
            .contiguous_bytes()?,
        &[0]
    );
    assert!(tensor_max_with_context_exact_native(&backend, &empty, None, false, &context).is_err());

    let input = backend.tensor(&[2], &[1.0, 2.0], &context)?;
    assert!(matches!(
        tensor_mean_with_context_exact_native(&backend, &input, Some(&[1]), false, None, &context,),
        Err(ReductionPartOneError::InvalidDimension { .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution(&cancelled)?;
    assert!(matches!(
        tensor_mean_with_context_exact_native(
            &backend,
            &input,
            Some(&[99]),
            false,
            None,
            &cancelled_context,
        ),
        Err(ReductionPartOneError::Tensor(
            sim_tensor::TensorError::Cancelled
        ))
    ));
    Ok(())
}

#[test]
fn reduction_capabilities_and_wire_round_trip_preserve_the_canonical_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let matrix = CpuBackend::capability_matrix();
    for operation in [
        ReductionOperation::All,
        ReductionOperation::ArgMaximum,
        ReductionOperation::Mean,
        ReductionOperation::StandardDeviation,
    ] {
        assert!(matrix.supported().iter().any(|support| {
            support.primitive() == PrimitiveOperation::Reduction(operation)
                && support.layout() == Some(Layout::Contiguous)
        }));
    }
    let worker = matrix.to_worker_capabilities()?;
    let json = serde_json::to_string(&worker)?;
    assert!(json.contains("reduction"));
    let decoded: sim_types::WorkerBackendCapabilities = serde_json::from_str(&json)?;
    let restored = sim_tensor::BackendCapabilityMatrix::try_from(decoded)?;
    assert_eq!(matrix, restored);
    Ok(())
}

#[test]
fn reduction_resolutions_are_unique_source_profiled_and_hash_sealed()
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
        assert_eq!(contract.resolution_module, "reduction_01");
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-reduction-comfy-tensor-op-00e998458e0c"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}

#[test]
fn numpy_contiguous_scalar_sums_match_original_source_bits()
-> Result<(), Box<dyn std::error::Error>> {
    use sim_tensor::generated_reduction_01::{
        numpy_contiguous_sum_f32_with_context_exact_native,
        numpy_contiguous_sum_f64_with_context_exact_native,
    };
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(0)?,
        &cancellation,
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x60ad78ec
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x40466666
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x60ad78ec
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x40466666
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x60ad78ec
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x40466666
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x60ad78ec
    );
    let cycle = [
        f32::from_bits(0x60ad78ec),
        f32::from_bits(0x3f800000),
        f32::from_bits(0xe0ad78ec),
        f32::from_bits(0x40400000),
        f32::from_bits(0x80000000),
        f32::from_bits(0x00000000),
        f32::from_bits(0x3dcccccd),
        f32::from_bits(0xbdcccccd),
    ];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x3f800001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x40400002
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x40e00001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x41000001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x41100001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x42fe0001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x43000001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x43010001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x437f0002
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x43808001
    );
    let cycle = [f32::from_bits(0x3f800001)];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x46000001
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [f32::from_bits(0x80000000)];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context)?.to_bits(),
        0x00000000
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x7e37e43c8800759c
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4008cccccccccccd
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x7e37e43c8800759c
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4008cccccccccccd
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x7e37e43c8800759c
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4008cccccccccccd
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x7e37e43c8800759c
    );
    let cycle = [
        f64::from_bits(0x7e37e43c8800759c),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xfe37e43c8800759c),
        f64::from_bits(0x4008000000000000),
        f64::from_bits(0x8000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3fb999999999999a),
        f64::from_bits(0xbfb999999999999a),
    ];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x3ff0000000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4008000000000002
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x401c000000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4020000000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4022000000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x405fc00000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4060000000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4060200000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x406fe00000000002
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x4070100000000001
    );
    let cycle = [f64::from_bits(0x3ff0000000000001)];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x40c0000000000001
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(0).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(1).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(3).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(7).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(8).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(9).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(127).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(128).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(129).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(255).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(257).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    let cycle = [f64::from_bits(0x8000000000000000)];
    let values = cycle.into_iter().cycle().take(8192).collect::<Vec<_>>();
    assert_eq!(
        numpy_contiguous_sum_f64_with_context_exact_native(&values, &context)?.to_bits(),
        0x0000000000000000
    );
    assert_eq!(context.scratch.peak_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn numpy_contiguous_scalar_sums_enforce_limits_and_cancel_without_workspace()
-> Result<(), Box<dyn std::error::Error>> {
    use sim_tensor::{
        TensorError,
        generated_reduction_01::{
            numpy_contiguous_sum_f32_with_context_exact_native,
            numpy_contiguous_sum_f64_with_context_exact_native,
        },
    };
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(0)?,
        &cancellation,
    );
    let values = vec![1.0_f32; 8193];
    assert!(matches!(
        numpy_contiguous_sum_f32_with_context_exact_native(&values, &context),
        Err(ReductionPartOneError::ScalarInputLimit {
            count: 8193,
            limit: 8192
        })
    ));
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let stopped = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        numpy_contiguous_sum_f64_with_context_exact_native(&[1.0, 2.0], &stopped),
        Err(ReductionPartOneError::Tensor(TensorError::Cancelled))
    ));
    assert_eq!(stopped.scratch.peak_bytes(), 0);
    assert_eq!(
        numpy_contiguous_sum_f32_with_context_exact_native(&[1.0, 2.0], &context)?,
        3.0
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}
