use std::{fs, path::Path};

use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor,
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, TENSOR_RESHAPE_OPERATION_ID,
        TENSOR_UNBIND_OPERATION_ID, TORCH_RESHAPE_OPERATION_ID, TORCH_STACK_OPERATION_ID,
        TORCH_UNBIND_OPERATION_ID, TorchSplitSpec, cat_jvp_with_context_exact_native,
        cat_vjp_with_context_exact_native, flatten_jvp_with_context_exact_native,
        flatten_vjp_with_context_exact_native, movedim_jvp_exact_native,
        movedim_vjp_exact_native, permute_jvp_exact_native, permute_vjp_exact_native,
        repeat_jvp_with_context_exact_native, repeat_vjp_with_context_exact_native,
        reshape_jvp_with_context_exact_native, reshape_vjp_with_context_exact_native,
        split_jvp_exact_native,
        split_vjp_with_context_exact_native, stack_jvp_with_context_exact_native,
        stack_vjp_exact_native, tensor_repeat_with_context_exact_native,
        tensor_reshape_with_context_exact_native, tensor_unbind_exact_native,
        tensor_view_as_exact_native, torch_cat_with_context_exact_native,
        torch_flatten_with_context_exact_native, torch_movedim_exact_native,
        torch_permute_exact_native, torch_reshape_with_context_exact_native,
        torch_split_exact_native, torch_stack_with_context_exact_native,
        torch_unbind_exact_native, unbind_jvp_exact_native, unbind_vjp_with_context_exact_native,
        view_as_jvp_exact_native, view_as_vjp_exact_native,
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
fn repeat_reuses_tile_values_and_analytical_maps() -> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2], &[1., 2.], &context)?;
    let repeated =
        tensor_repeat_with_context_exact_native(&test.backend, &input, &[2], &context)?;
    assert_eq!(values(&repeated)?, vec![1., 2., 1., 2.]);
    let output_gradient = test.tensor(&[4], &[1., 2., 3., 4.], &context)?;
    let gradient = repeat_vjp_with_context_exact_native(
        &test.backend,
        &input,
        &[2],
        &output_gradient,
        &context,
    )?;
    assert_eq!(values(&gradient)?, vec![4., 6.]);
    let tangent = repeat_jvp_with_context_exact_native(
        &test.backend,
        &input,
        &input,
        &[2],
        &context,
    )?;
    assert_eq!(values(&tangent)?, values(&repeated)?);
    assert!(matches!(
        tensor_repeat_with_context_exact_native(&test.backend, &input, &[], &context),
        Err(ShapeLayoutTransformPartTwoError::Invalid { .. })
    ));
    Ok(())
}

#[test]
fn reshape_and_view_as_preserve_aliases_or_copy_only_when_required()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.], &context)?;
    let reshaped =
        tensor_reshape_with_context_exact_native(&test.backend, &input, &[3, -1], &context)?;
    assert_eq!(reshaped.descriptor().shape(), &[3, 2]);
    assert_eq!(reshaped.storage_id(), input.storage_id());
    let other = test.tensor(&[6], &[0.; 6], &context)?;
    let viewed = tensor_view_as_exact_native(&input, &other, &cancellation)?;
    assert_eq!(viewed.descriptor().shape(), &[6]);
    assert_eq!(viewed.storage_id(), input.storage_id());
    assert_eq!(
        view_as_vjp_exact_native(&viewed, &[2, 3], &cancellation)?
            .descriptor()
            .shape(),
        &[2, 3]
    );
    assert_eq!(
        view_as_jvp_exact_native(&input, &other, &cancellation)?
            .descriptor()
            .shape(),
        &[6]
    );

    let transposed = torch_permute_exact_native(&input, &[1, 0], &cancellation)?;
    let copied =
        torch_reshape_with_context_exact_native(&test.backend, &transposed, &[6], &context)?;
    assert_ne!(copied.storage_id(), transposed.storage_id());
    assert_eq!(values(&copied)?, vec![1., 4., 2., 5., 3., 6.]);
    let gradient = reshape_vjp_with_context_exact_native(
        &test.backend,
        &reshaped,
        &[2, 3],
        TENSOR_RESHAPE_OPERATION_ID,
        &context,
    )?;
    assert_eq!(gradient.descriptor().shape(), &[2, 3]);
    assert_eq!(
        reshape_jvp_with_context_exact_native(
            &test.backend,
            &input,
            &[3, 2],
            TORCH_RESHAPE_OPERATION_ID,
            &context,
        )?
        .descriptor()
        .shape(),
        &[3, 2]
    );
    Ok(())
}

#[test]
fn unbind_returns_read_only_views_and_stack_is_its_vjp()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 2], &[1., 2., 3., 4.], &context)?;
    let outputs = tensor_unbind_exact_native(&input, 0, &cancellation)?;
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|output| output.storage_id() == input.storage_id()));
    assert_eq!(values(&outputs[1])?, vec![3., 4.]);
    let function_outputs = torch_unbind_exact_native(&input, -1, &cancellation)?;
    assert_eq!(values(&function_outputs[0])?, vec![1., 3.]);
    let reconstructed = unbind_vjp_with_context_exact_native(
        &test.backend,
        &outputs,
        0,
        TENSOR_UNBIND_OPERATION_ID,
        &context,
    )?;
    assert_eq!(values(&reconstructed)?, values(&input)?);
    assert_eq!(
        unbind_jvp_exact_native(&input, 0, TORCH_UNBIND_OPERATION_ID, &cancellation)?.len(),
        2
    );
    Ok(())
}

#[test]
fn cat_delegates_forward_vjp_and_jvp_to_the_concatenation_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let left = test.tensor(&[1, 2], &[1., 2.], &context)?;
    let right = test.tensor(&[2, 2], &[3., 4., 5., 6.], &context)?;
    let inputs = [left, right];
    let output = torch_cat_with_context_exact_native(&test.backend, &inputs, 0, &context)?;
    assert_eq!(output.descriptor().shape(), &[3, 2]);
    assert_eq!(values(&output)?, vec![1., 2., 3., 4., 5., 6.]);
    let gradients =
        cat_vjp_with_context_exact_native(&test.backend, &inputs, 0, &output, &context)?;
    assert_eq!(values(&gradients[0])?, vec![1., 2.]);
    assert_eq!(values(&gradients[1])?, vec![3., 4., 5., 6.]);
    let tangent = cat_jvp_with_context_exact_native(
        &test.backend,
        &inputs,
        &inputs,
        0,
        &context,
    )?;
    assert_eq!(values(&tangent)?, values(&output)?);
    Ok(())
}

#[test]
fn flatten_movedim_and_permute_share_task_86_and_descriptor_views()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3, 4], &(0..24).map(|v| v as f32).collect::<Vec<_>>(), &context)?;
    let flattened =
        torch_flatten_with_context_exact_native(&test.backend, &input, 1, -1, &context)?;
    assert_eq!(flattened.descriptor().shape(), &[2, 12]);
    assert_eq!(flattened.storage_id(), input.storage_id());
    assert_eq!(
        flatten_vjp_with_context_exact_native(
            &test.backend,
            &flattened,
            &[2, 3, 4],
            &context,
        )?
        .descriptor()
        .shape(),
        &[2, 3, 4]
    );
    assert_eq!(
        flatten_jvp_with_context_exact_native(&test.backend, &input, 1, -1, &context)?
            .descriptor()
            .shape(),
        &[2, 12]
    );
    let moved = torch_movedim_exact_native(&input, &[0], &[2], &cancellation)?;
    assert_eq!(moved.descriptor().shape(), &[3, 4, 2]);
    assert_eq!(
        movedim_vjp_exact_native(&moved, &[0], &[2], &cancellation)?
            .descriptor()
            .shape(),
        &[2, 3, 4]
    );
    assert_eq!(
        movedim_jvp_exact_native(&input, &[0], &[2], &cancellation)?
            .descriptor()
            .shape(),
        &[3, 4, 2]
    );
    let permuted = torch_permute_exact_native(&input, &[2, 0, 1], &cancellation)?;
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
    Ok(())
}

#[test]
fn split_supports_sizes_and_explicit_sections_through_task_60()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[5], &[1., 2., 3., 4., 5.], &context)?;
    let sized = torch_split_exact_native(&input, TorchSplitSpec::Size(2), 0, &cancellation)?;
    assert_eq!(sized.len(), 3);
    assert_eq!(sized[2].descriptor().shape(), &[1]);
    assert!(sized.iter().all(|output| output.storage_id() == input.storage_id()));
    let sections =
        torch_split_exact_native(&input, TorchSplitSpec::Sizes(&[1, 0, 4]), 0, &cancellation)?;
    assert_eq!(sections.iter().map(|output| output.descriptor().shape()[0]).collect::<Vec<_>>(), vec![1, 0, 4]);
    let gradient = split_vjp_with_context_exact_native(
        &test.backend,
        &input,
        &sized,
        TorchSplitSpec::Size(2),
        0,
        &context,
    )?;
    assert_eq!(values(&gradient)?, values(&input)?);
    assert_eq!(
        split_jvp_exact_native(&input, TorchSplitSpec::Size(2), 0, &cancellation)?.len(),
        3
    );
    Ok(())
}

#[test]
fn stack_composes_unsqueeze_and_canonical_concatenation_with_inverse_vjp()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let first = test.tensor(&[2], &[1., 2.], &context)?;
    let second = test.tensor(&[2], &[3., 4.], &context)?;
    let inputs = [first, second];
    let output = torch_stack_with_context_exact_native(&test.backend, &inputs, 1, &context)?;
    assert_eq!(output.descriptor().shape(), &[2, 2]);
    assert_eq!(values(&output)?, vec![1., 3., 2., 4.]);
    let gradients = stack_vjp_exact_native(&output, 1, &cancellation)?;
    assert_eq!(values(&gradients[0])?, vec![1., 2.]);
    assert_eq!(values(&gradients[1])?, vec![3., 4.]);
    let tangent =
        stack_jvp_with_context_exact_native(&test.backend, &inputs, 1, &context)?;
    assert_eq!(values(&tangent)?, values(&output)?);
    let round_trip = unbind_vjp_with_context_exact_native(
        &test.backend,
        &gradients,
        1,
        TORCH_STACK_OPERATION_ID,
        &context,
    )?;
    assert_eq!(values(&round_trip)?, values(&output)?);
    Ok(())
}

#[test]
fn cancellation_and_build_sealed_contracts_cover_all_exact_ids()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[1], &[0.], &context)?;
    cancellation.cancel();
    assert!(matches!(
        torch_reshape_with_context_exact_native(&test.backend, &input, &[1], &context),
        Err(ShapeLayoutTransformPartTwoError::Cancelled)
    ));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixtures = root.join(
        "crates/sim_test_support/fixtures/tensor_operations/shape_layout_transform_02",
    );
    let mut runtime_digests = Vec::new();
    for entry in fs::read_dir(fixtures)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            runtime_digests.push(format!("{:x}", Sha256::digest(fs::read(path)?)));
        }
    }
    assert_eq!(runtime_digests.len(), 12);
    assert_eq!(
        sim_tensor::GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS
            .iter()
            .filter(|(_, _, module, digest)| {
                *module == "shape_layout_transform_02"
                    && runtime_digests.iter().any(|runtime| runtime == digest)
            })
            .count(),
        12
    );
    assert_eq!(
        sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .filter(|slice| slice.module_name == "shape_layout_transform_02")
            .flat_map(|slice| slice.contracts)
            .count(),
        12
    );
    assert!(
        sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.contracts)
            .any(|contract| contract.operation_id == TORCH_RESHAPE_OPERATION_ID)
    );
    Ok(())
}
