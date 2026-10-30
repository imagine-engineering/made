use std::collections::BTreeMap;
use std::{fs, path::Path};

use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor,
    generated_shape_layout_transform_01::{
        RepeatInterleaveSpec, ShapeLayoutTransformPartOneError, TensorViewSpec,
        chunk_jvp_exact_native,
        chunk_vjp_with_context_exact_native, einops_repeat_jvp_with_context_exact_native,
        einops_repeat_vjp_with_context_exact_native, einops_repeat_with_context_exact_native,
        expand_jvp_exact_native, expand_vjp_with_context_exact_native,
        flatten_vjp_exact_native, movedim_jvp_exact_native, movedim_vjp_exact_native,
        repeat_interleave_jvp_with_context_exact_native,
        repeat_interleave_vjp_with_context_exact_native, tensor_chunk_exact_native,
        tensor_expand_as_exact_native, tensor_expand_exact_native,
        tensor_flatten_with_context_exact_native, tensor_movedim_exact_native,
        tensor_repeat_interleave_with_context_exact_native, tensor_unsqueeze_exact_native,
        tensor_view_exact_native, torch_chunk_exact_native,
        torch_repeat_interleave_with_context_exact_native, torch_unsqueeze_exact_native,
        unsqueeze_vjp_exact_native, view_vjp_exact_native,
    },
};
use sha2::{Digest, Sha256};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn context<'a>(
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
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self.backend.upload_f32(descriptor, values, context)?.0)
    }

    fn i64_tensor(
        &self,
        shape: &[u64],
        values: &[i64],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::I64,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        Ok(self.backend.upload_bytes(descriptor, &bytes, context)?.0)
    }
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    (0..count)
        .map(|linear| {
            let bytes: [u8; 4] = tensor
                .linear_element_bytes(u64::try_from(linear)?)?
                .try_into()?;
            Ok(f32::from_ne_bytes(bytes))
        })
        .collect()
}

#[test]
fn chunk_reuses_canonical_split_views_and_gradients() -> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[4], &[1., 2., 3., 4.], &context)?;

    let chunks = tensor_chunk_exact_native(&input, 3, 0, &cancellation)?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].descriptor().shape(), &[2]);
    assert_eq!(chunks[1].descriptor().shape(), &[2]);
    assert!(chunks.iter().all(|chunk| chunk.storage_id() == input.storage_id()));
    let function_chunks = torch_chunk_exact_native(&input, 3, -1, &cancellation)?;
    assert_eq!(values(&function_chunks[1])?, vec![3., 4.]);
    let empty = test.tensor(&[0], &[], &context)?;
    let empty_chunks = tensor_chunk_exact_native(&empty, 3, 0, &cancellation)?;
    assert_eq!(empty_chunks.len(), 3);
    assert!(empty_chunks
        .iter()
        .all(|chunk| chunk.descriptor().shape() == [0]));

    let first_gradient = test.tensor(&[2], &[10., 20.], &context)?;
    let second_gradient = test.tensor(&[2], &[30., 40.], &context)?;
    let gradient = chunk_vjp_with_context_exact_native(
        &test.backend,
        &input,
        &[first_gradient, second_gradient],
        3,
        0,
        &context,
    )?;
    assert_eq!(values(&gradient)?, vec![10., 20., 30., 40.]);
    assert_eq!(chunk_jvp_exact_native(&input, 3, 0, &cancellation)?.len(), 2);
    Ok(())
}

#[test]
fn expand_is_a_read_only_zero_stride_view_with_sum_vjp() -> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[1, 2], &[2., 5.], &context)?;
    let expanded = tensor_expand_exact_native(&input, &[3, -1], &cancellation)?;
    assert_eq!(expanded.descriptor().shape(), &[3, 2]);
    assert_eq!(expanded.descriptor().strides(), &[0, 1]);
    assert_eq!(expanded.storage_id(), input.storage_id());
    assert_eq!(values(&expanded)?, vec![2., 5., 2., 5., 2., 5.]);

    let other = test.tensor(&[4, 2], &[0.; 8], &context)?;
    let expanded_as = tensor_expand_as_exact_native(&input, &other, &cancellation)?;
    assert_eq!(expanded_as.descriptor().shape(), &[4, 2]);
    let output_gradient = test.tensor(&[3, 2], &[1., 2., 3., 4., 5., 6.], &context)?;
    let gradient = expand_vjp_with_context_exact_native(
        &test.backend,
        &input,
        &output_gradient,
        &context,
    )?;
    assert_eq!(values(&gradient)?, vec![9., 12.]);
    let tangent = expand_jvp_exact_native(&input, &[3, 2], &cancellation)?;
    assert_eq!(tangent.storage_id(), input.storage_id());
    Ok(())
}

#[test]
fn view_and_flatten_use_descriptor_compatibility_and_copy_only_when_required()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.], &context)?;
    let viewed = tensor_view_exact_native(
        &input,
        TensorViewSpec::Shape(&[3, -1]),
        &cancellation,
    )?;
    assert_eq!(viewed.descriptor().shape(), &[3, 2]);
    assert_eq!(viewed.storage_id(), input.storage_id());
    let view_gradient = view_vjp_exact_native(&viewed, &[2, 3], &cancellation)?;
    assert_eq!(view_gradient.descriptor().shape(), &[2, 3]);

    let moved = tensor_movedim_exact_native(&input, &[0], &[1], &cancellation)?;
    assert!(matches!(
        tensor_view_exact_native(&moved, TensorViewSpec::Shape(&[6]), &cancellation),
        Err(ShapeLayoutTransformPartOneError::Invalid { .. })
    ));
    let byte_view = tensor_view_exact_native(
        &input,
        TensorViewSpec::DType(DType::U8),
        &cancellation,
    )?;
    assert_eq!(byte_view.descriptor().shape(), &[2, 12]);
    assert_eq!(byte_view.descriptor().strides(), &[12, 1]);
    assert_eq!(byte_view.storage_id(), input.storage_id());
    let round_trip = tensor_view_exact_native(
        &byte_view,
        TensorViewSpec::DType(DType::F32),
        &cancellation,
    )?;
    assert_eq!(round_trip.descriptor(), input.descriptor());
    let flattened = tensor_flatten_with_context_exact_native(
        &test.backend,
        &moved,
        0,
        -1,
        &context,
    )?;
    assert_ne!(flattened.storage_id(), moved.storage_id());
    assert_eq!(values(&flattened)?, vec![1., 4., 2., 5., 3., 6.]);
    let flat_gradient = flatten_vjp_exact_native(&flattened, &[3, 2], &cancellation)?;
    assert_eq!(flat_gradient.descriptor().shape(), &[3, 2]);
    Ok(())
}

#[test]
fn movedim_and_unsqueeze_are_inverse_read_only_views() -> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3, 4], &(0..24).map(|v| v as f32).collect::<Vec<_>>(), &context)?;
    let moved = tensor_movedim_exact_native(&input, &[0, 2], &[2, 0], &cancellation)?;
    assert_eq!(moved.descriptor().shape(), &[4, 3, 2]);
    assert_eq!(moved.storage_id(), input.storage_id());
    let inverse = movedim_vjp_exact_native(&moved, &[0, 2], &[2, 0], &cancellation)?;
    assert_eq!(inverse.descriptor().shape(), input.descriptor().shape());
    assert_eq!(values(&inverse)?, values(&input)?);
    assert_eq!(movedim_jvp_exact_native(&input, &[0], &[2], &cancellation)?.descriptor().shape(), &[3, 4, 2]);

    let unsqueezed = tensor_unsqueeze_exact_native(&input, -1, &cancellation)?;
    assert_eq!(unsqueezed.descriptor().shape(), &[2, 3, 4, 1]);
    assert_eq!(unsqueezed.storage_id(), input.storage_id());
    assert_eq!(torch_unsqueeze_exact_native(&input, 0, &cancellation)?.descriptor().shape(), &[1, 2, 3, 4]);
    assert_eq!(unsqueeze_vjp_exact_native(&unsqueezed, &[2, 3, 4], &cancellation)?.descriptor().shape(), &[2, 3, 4]);
    Ok(())
}

#[test]
fn repeat_interleave_supports_scalar_per_element_function_and_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.], &context)?;
    let repeated = tensor_repeat_interleave_with_context_exact_native(
        &test.backend,
        &input,
        RepeatInterleaveSpec::PerElement(&[1, 2, 0]),
        Some(1),
        Some(3),
        &context,
    )?;
    assert_eq!(repeated.descriptor().shape(), &[2, 3]);
    assert_eq!(values(&repeated)?, vec![1., 2., 2., 4., 5., 5.]);
    let function = torch_repeat_interleave_with_context_exact_native(
        &test.backend,
        &input,
        RepeatInterleaveSpec::Scalar(2),
        None,
        Some(12),
        &context,
    )?;
    assert_eq!(function.descriptor().shape(), &[12]);
    assert_eq!(values(&function)?[..4], [1., 1., 2., 2.]);
    let repeat_counts = test.i64_tensor(&[3], &[1, 0, 2], &context)?;
    let tensor_counts = torch_repeat_interleave_with_context_exact_native(
        &test.backend,
        &input,
        RepeatInterleaveSpec::Tensor(&repeat_counts),
        Some(1),
        Some(3),
        &context,
    )?;
    assert_eq!(values(&tensor_counts)?, vec![1., 3., 3., 4., 6., 6.]);
    let floating_counts = test.tensor(&[3], &[1., 0., 2.], &context)?;
    assert!(matches!(
        torch_repeat_interleave_with_context_exact_native(
            &test.backend,
            &input,
            RepeatInterleaveSpec::Tensor(&floating_counts),
            Some(1),
            None,
            &context,
        ),
        Err(ShapeLayoutTransformPartOneError::UnsupportedDType {
            operation: "COMFY-TENSOR-OP-0C2E0712DA68",
            dtype: DType::F32,
        })
    ));

    let output_gradient = test.tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.], &context)?;
    let gradient = repeat_interleave_vjp_with_context_exact_native(
        &test.backend,
        &input,
        RepeatInterleaveSpec::PerElement(&[1, 2, 0]),
        Some(1),
        &output_gradient,
        &context,
    )?;
    assert_eq!(values(&gradient)?, vec![1., 5., 0., 4., 11., 0.]);
    let tangent = repeat_interleave_jvp_with_context_exact_native(
        &test.backend,
        &input,
        RepeatInterleaveSpec::Scalar(2),
        Some(0),
        Some(4),
        &context,
    )?;
    assert_eq!(tangent.descriptor().shape(), &[4, 3]);
    Ok(())
}

#[test]
fn einops_repeat_reuses_the_canonical_grammar_owner_and_sums_vjp()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.], &context)?;
    let axes = BTreeMap::from([("r".to_owned(), 2)]);
    let repeated = einops_repeat_with_context_exact_native(
        &test.backend,
        &input,
        "b c -> b r c",
        &axes,
        &context,
    )?;
    assert_eq!(repeated.descriptor().shape(), &[2, 2, 3]);
    assert_eq!(values(&repeated)?, vec![1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.]);
    let output_gradient = test.tensor(&[2, 2, 3], &[1.; 12], &context)?;
    let gradient = einops_repeat_vjp_with_context_exact_native(
        &test.backend,
        &output_gradient,
        &[2, 3],
        "b c -> b r c",
        &axes,
        &context,
    )?;
    assert_eq!(values(&gradient)?, vec![2.; 6]);
    assert_eq!(
        einops_repeat_jvp_with_context_exact_native(
            &test.backend,
            &input,
            "b c -> b r c",
            &axes,
            &context,
        )?
        .descriptor()
        .shape(),
        &[2, 2, 3]
    );
    let error = einops_repeat_with_context_exact_native(
        &test.backend,
        &input,
        "b c b r c",
        &axes,
        &context,
    )
    .expect_err("invalid repeat syntax must fail");
    assert!(error.to_string().contains("COMFY-TENSOR-OP-71DB8F99EAAC"));
    Ok(())
}

#[test]
fn cancellation_precedes_all_shape_validation_and_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let active = CancellationToken::default();
    let active_context = test.context(&active)?;
    let input = test.tensor(&[2], &[1., 2.], &active_context)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = test.context(&cancellation)?;
    assert!(matches!(
        tensor_chunk_exact_native(&input, 0, 9, &cancellation),
        Err(ShapeLayoutTransformPartOneError::Cancelled)
    ));
    assert!(matches!(
        tensor_repeat_interleave_with_context_exact_native(
            &test.backend,
            &input,
            RepeatInterleaveSpec::Scalar(2),
            Some(9),
            None,
            &context,
        ),
        Err(ShapeLayoutTransformPartOneError::Cancelled)
    ));
    assert!(matches!(
        einops_repeat_with_context_exact_native(
            &test.backend,
            &input,
            "invalid",
            &BTreeMap::new(),
            &context,
        ),
        Err(ShapeLayoutTransformPartOneError::Cancelled)
    ));
    Ok(())
}

#[test]
fn all_exact_contracts_are_build_sealed_to_distinct_runtime_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let expected = [
        ("COMFY-TENSOR-OP-0C2E0712DA68", "torch_repeat_interleave.json", "84b976ca3d1d375d84eb73e2ff8845b2fc0d2d80d53b651cf1d450632055ffa6"),
        ("COMFY-TENSOR-OP-17E7C823A86F", "tensor_unsqueeze.json", "08d02403e18d95597e919e2a95ce12c34412774c00cefed4b26b3b66950d02aa"),
        ("COMFY-TENSOR-OP-25362A66A957", "tensor_expand_as.json", "29a86c4d45df833e165fc1332d5bc04df14a5dc0df56662782518c9005614c6a"),
        ("COMFY-TENSOR-OP-3D13DA91C9F3", "tensor_expand.json", "0f39d4f58f4e5420fe921da5cd57f34b51cdbeea9d387c299bf7e910c1a7863f"),
        ("COMFY-TENSOR-OP-3E6301EB6AA6", "tensor_repeat_interleave.json", "b9b464137e583aaf0e48682176ced785560077bcf68ec871430f657274c28d79"),
        ("COMFY-TENSOR-OP-3E9A0E130935", "torch_unsqueeze.json", "a7e99dd9faf53d3a763f7a8e774e5bb12e079191c3e5dc45e90dc7dcd99f852c"),
        ("COMFY-TENSOR-OP-47B154B1D223", "torch_chunk.json", "9ac46855af3970a6bc45d9d4fbf4de14f44d1e363b4f6184513b5c7a52a4cf1c"),
        ("COMFY-TENSOR-OP-5380FDF9E668", "tensor_view.json", "5c238f1689e8aeab586c95aad522159e1f296e824632a6da06ad74e6bd1d4f90"),
        ("COMFY-TENSOR-OP-5A4B8BBBFD81", "tensor_chunk.json", "d4ba3ab03cf6f3e5c80aef26f424777fdc7ec53d5e571d69c552276fc75da3c5"),
        ("COMFY-TENSOR-OP-67D2FDD707E0", "tensor_flatten.json", "6118e65459c71abac9ad762c6a1c5f12a8018ad35af1555e4fcfae7a51bd9f39"),
        ("COMFY-TENSOR-OP-71DB8F99EAAC", "einops_repeat.json", "567544b1017baf74f5335a03af15a1ee788e0dabc4f4074c136f0937af606f8e"),
        ("COMFY-TENSOR-OP-73D179A8CEB9", "tensor_movedim.json", "dcef471212dca303e3da6b1f894326fc828b97dc6b51d21f753aecd54399acc6"),
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let contracts = sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "shape_layout_transform_01")
        .ok_or("Task 86 resolution slice is absent")?;
    assert_eq!(contracts.contracts.len(), expected.len());
    for (operation, file, digest) in expected {
        let path = root
            .join("crates/sim_test_support/fixtures/tensor_operations/shape_layout_transform_01")
            .join(file);
        let bytes = fs::read(path)?;
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest);
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(fixture["operation_id"], operation);
        let contract = contracts
            .contracts
            .iter()
            .find(|contract| contract.operation_id == operation)
            .ok_or("exact operation is absent from Task 86 slice")?;
        assert_eq!(contract.evidence_fixture_sha256, digest);
    }
    Ok(())
}
