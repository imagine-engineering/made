use std::{fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, StreamId, Tensor, TensorDescriptor,
    generated_elementwise_or_runtime_operation_17::TensorSplitSpec,
    generated_shape_layout_transform_03::{
        FunctionalPadMode, ShapeLayoutTransformPartThreeError, TENSOR_TRANSPOSE_OPERATION_ID,
        functional_pad_with_context_exact_native, pad_jvp_with_context_exact_native,
        pad_vjp_with_context_exact_native, permute_jvp_exact_native, permute_vjp_exact_native,
        split_jvp_exact_native, split_vjp_with_context_exact_native, squeeze_jvp_exact_native,
        squeeze_vjp_with_context_exact_native, tensor_permute_exact_native,
        tensor_split_exact_native_part_three, tensor_squeeze_exact_native,
        tensor_transpose_exact_native, transpose_jvp_exact_native, transpose_vjp_exact_native,
    },
};

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
fn method_permute_and_transpose_are_descriptor_owned_inverse_views()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(
        &[2, 3, 4],
        &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
        &context,
    )?;
    let permuted = tensor_permute_exact_native(&input, &[2, 0, 1], &cancellation)?;
    assert_eq!(permuted.descriptor().shape(), &[4, 2, 3]);
    assert_eq!(permuted.storage_id(), input.storage_id());
    let inverse = permute_vjp_exact_native(&permuted, &[2, 0, 1], &cancellation)?;
    assert_eq!(inverse.descriptor().shape(), input.descriptor().shape());
    assert_eq!(values(&inverse)?, values(&input)?);
    assert_eq!(
        permute_jvp_exact_native(&input, &[2, 0, 1], &cancellation)?
            .descriptor()
            .shape(),
        &[4, 2, 3]
    );

    let transposed = tensor_transpose_exact_native(&input, 0, -1, &cancellation)?;
    assert_eq!(transposed.descriptor().shape(), &[4, 3, 2]);
    assert_eq!(transposed.storage_id(), input.storage_id());
    assert_eq!(
        transpose_vjp_exact_native(&transposed, 0, -1, &cancellation)?
            .descriptor()
            .shape(),
        input.descriptor().shape()
    );
    assert_eq!(
        transpose_jvp_exact_native(&input, 0, -1, &cancellation)?
            .descriptor()
            .shape(),
        &[4, 3, 2]
    );
    Ok(())
}

#[test]
fn method_split_delegates_sizes_views_and_gradients_to_task_60()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[5], &[1., 2., 3., 4., 5.], &context)?;
    let specification = TensorSplitSpec::Sizes(vec![1, 0, 4]);
    let outputs = tensor_split_exact_native_part_three(&input, &specification, 0, &cancellation)?;
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.descriptor().shape().first().copied())
            .collect::<Vec<_>>(),
        vec![Some(1), Some(0), Some(4)]
    );
    assert!(
        outputs
            .iter()
            .all(|output| output.storage_id() == input.storage_id())
    );
    let gradient = split_vjp_with_context_exact_native(
        &test.backend,
        &input,
        &outputs,
        &specification,
        0,
        &context,
    )?;
    assert_eq!(values(&gradient)?, values(&input)?);
    assert_eq!(
        split_jvp_exact_native(&input, &TensorSplitSpec::Size(2), 0, &cancellation)?.len(),
        3
    );
    Ok(())
}

#[test]
fn squeeze_preserves_read_only_aliases_and_uses_reshape_for_its_inverse()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[1, 2, 1], &[3., 4.], &context)?;
    let squeezed = tensor_squeeze_exact_native(&input, None, &cancellation)?;
    assert_eq!(squeezed.descriptor().shape(), &[2]);
    assert_eq!(squeezed.storage_id(), input.storage_id());
    assert_eq!(
        tensor_squeeze_exact_native(&input, Some(&[0]), &cancellation)?
            .descriptor()
            .shape(),
        &[2, 1]
    );
    assert_eq!(
        tensor_squeeze_exact_native(&input, Some(&[1]), &cancellation)?
            .descriptor()
            .shape(),
        &[1, 2, 1]
    );
    let gradient = squeeze_vjp_with_context_exact_native(
        &test.backend,
        &squeezed,
        input.descriptor().shape(),
        &context,
    )?;
    assert_eq!(gradient.descriptor().shape(), input.descriptor().shape());
    assert_eq!(
        squeeze_jvp_exact_native(&input, None, &cancellation)?
            .descriptor()
            .shape(),
        &[2]
    );
    Ok(())
}

#[test]
fn constant_pad_supports_negative_crop_fill_and_zero_tangent_boundaries()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[3], &[1., 2., 3.], &context)?;
    let output = functional_pad_with_context_exact_native(
        &test.backend,
        &input,
        &[-1, 2],
        FunctionalPadMode::Constant,
        Some(DecodedScalar::Real(9.0)),
        &context,
    )?;
    assert_eq!(output.descriptor().shape(), &[4]);
    assert_eq!(values(&output)?, vec![2., 3., 9., 9.]);
    let tangent = pad_jvp_with_context_exact_native(
        &test.backend,
        &input,
        &[-1, 2],
        FunctionalPadMode::Constant,
        &context,
    )?;
    assert_eq!(values(&tangent)?, vec![2., 3., 0., 0.]);
    let gradient = test.tensor(&[4], &[1., 2., 3., 4.], &context)?;
    assert_eq!(
        values(&pad_vjp_with_context_exact_native(
            &test.backend,
            &gradient,
            &[3],
            &[-1, 2],
            FunctionalPadMode::Constant,
            &context,
        )?)?,
        vec![0., 1., 2.]
    );
    Ok(())
}

#[test]
fn reflect_replicate_and_circular_pad_share_canonical_coordinate_mapping()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[3], &[1., 2., 3.], &context)?;
    for (mode, expected) in [
        (FunctionalPadMode::Reflect, vec![2., 1., 2., 3., 2.]),
        (FunctionalPadMode::Replicate, vec![1., 1., 2., 3., 3.]),
        (FunctionalPadMode::Circular, vec![3., 1., 2., 3., 1.]),
    ] {
        let output = functional_pad_with_context_exact_native(
            &test.backend,
            &input,
            &[1, 1],
            mode,
            None,
            &context,
        )?;
        assert_eq!(values(&output)?, expected);
    }
    let output_gradient = test.tensor(&[5], &[1.; 5], &context)?;
    let input_gradient = pad_vjp_with_context_exact_native(
        &test.backend,
        &output_gradient,
        &[3],
        &[1, 1],
        FunctionalPadMode::Reflect,
        &context,
    )?;
    assert_eq!(values(&input_gradient)?, vec![1., 3., 1.]);
    Ok(())
}

#[test]
fn cancellation_invalid_boundaries_and_build_sealed_contracts_cover_all_ids()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[1], &[0.], &context)?;
    cancellation.cancel();
    assert!(matches!(
        functional_pad_with_context_exact_native(
            &test.backend,
            &input,
            &[1],
            FunctionalPadMode::Constant,
            None,
            &context,
        ),
        Err(ShapeLayoutTransformPartThreeError::Cancelled)
    ));

    let active = CancellationToken::default();
    assert!(matches!(
        tensor_transpose_exact_native(&input, 0, 2, &active),
        Err(ShapeLayoutTransformPartThreeError::Invalid {
            operation: TENSOR_TRANSPOSE_OPERATION_ID,
            ..
        }) | Err(ShapeLayoutTransformPartThreeError::CanonicalOwner {
            operation: TENSOR_TRANSPOSE_OPERATION_ID,
            ..
        })
    ));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixtures =
        root.join("crates/sim_test_support/fixtures/tensor_operations/shape_layout_transform_03");
    let mut runtime_digests = Vec::new();
    for entry in fs::read_dir(fixtures)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            runtime_digests.push(format!("{:x}", Sha256::digest(fs::read(path)?)));
        }
    }
    assert_eq!(runtime_digests.len(), 5);
    assert_eq!(
        sim_tensor::GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS
            .iter()
            .filter(|(_, _, module, digest)| {
                *module == "shape_layout_transform_03"
                    && runtime_digests.iter().any(|runtime| runtime == digest)
            })
            .count(),
        5
    );
    assert_eq!(
        sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .filter(|slice| slice.module_name == "shape_layout_transform_03")
            .flat_map(|slice| slice.contracts)
            .count(),
        5
    );
    Ok(())
}
