use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor,
    autograd::{AutogradTape, GradientMode, LeafId},
    generated_activation_normalization_functional_01::GeluApproximation,
    generated_neural_network_module_01::LossReduction,
    generated_neural_network_module_03::{
        NeuralNetworkModulePartThreeError, conv_3d_jvp_with_context_exact_native,
        conv_3d_vjp_with_context_exact_native, conv_3d_with_context_exact_native,
        gelu_module_jvp_with_context_exact_native, gelu_module_vjp_with_context_exact_native,
        gelu_module_with_context_exact_native, l1_loss_jvp_with_context_exact_native,
        l1_loss_vjp_with_context_exact_native, l1_loss_with_context_exact_native,
        max_pool_2d_jvp_with_context_exact_native, max_pool_2d_vjp_with_context_exact_native,
        max_pool_2d_with_context_exact_native, parameter_exact_native,
        pixel_shuffle_module_jvp_with_context_exact_native,
        pixel_shuffle_module_vjp_with_context_exact_native,
        pixel_shuffle_module_with_context_exact_native,
        pixel_unshuffle_module_with_context_exact_native, relu_6_jvp_with_context_exact_native,
        relu_6_vjp_with_context_exact_native, relu_6_with_context_exact_native,
        relu_module_jvp_with_context_exact_native, relu_module_vjp_with_context_exact_native,
        relu_module_with_context_exact_native, zero_pad_2d_jvp_with_context_exact_native,
        zero_pad_2d_vjp_with_context_exact_native, zero_pad_2d_with_context_exact_native,
    },
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(8 * 1024 * 1024)?,
            cancellation,
        ))
    }
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(descriptor, values, &backend.execution(cancellation)?)?
        .0)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn convolution_gelu_and_l1_delegate_canonical_owners() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
    let weight = vec![1.0; 8];
    let convolution = conv_3d_with_context_exact_native(
        &backend,
        &input,
        &[1, 1, 2, 2, 2],
        &weight,
        &[1, 1, 2, 2, 2],
        Some(&[1.0]),
        [1; 3],
        [0; 3],
        [1; 3],
        1,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(convolution.shape, [1, 1, 1, 1, 1]);
    close(&convolution.values, &[37.0], 0.0);
    let vjp = conv_3d_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1, 1, 2, 2, 2],
        &weight,
        &[1, 1, 2, 2, 2],
        Some(&[1.0]),
        [1; 3],
        [0; 3],
        [1; 3],
        1,
        &[2.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[2.0; 8], 0.0);
    close(
        &vjp.weight,
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        0.0,
    );
    close(
        vjp.bias.as_deref().ok_or("missing bias gradient")?,
        &[2.0],
        0.0,
    );
    let jvp = conv_3d_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 8],
        &[1, 1, 2, 2, 2],
        &weight,
        &[0.0; 8],
        &[1, 1, 2, 2, 2],
        Some(&[1.0]),
        Some(&[0.0]),
        [1; 3],
        [0; 3],
        [1; 3],
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[8.0], 0.0);

    let activation_input = [-1.0, 0.0, 1.0];
    let gelu = gelu_module_with_context_exact_native(
        &backend,
        &activation_input,
        GeluApproximation::None,
        DeviceId::CPU,
        &context,
    )?;
    close(&gelu, &[-0.15865526, 0.0, 0.8413447], 1.0e-6);
    let gelu_vjp = gelu_module_vjp_with_context_exact_native(
        &backend,
        &activation_input,
        &[1.0; 3],
        GeluApproximation::None,
        DeviceId::CPU,
        &context,
    )?;
    let gelu_jvp = gelu_module_jvp_with_context_exact_native(
        &backend,
        &activation_input,
        &[1.0; 3],
        GeluApproximation::None,
        DeviceId::CPU,
        &context,
    )?;
    close(&gelu_vjp, &gelu_jvp, 0.0);

    let loss = l1_loss_with_context_exact_native(
        &backend,
        &[-1.0, 2.0],
        &[1.0, 0.0],
        LossReduction::Mean,
        DeviceId::CPU,
        &context,
    )?;
    close(&loss, &[2.0], 0.0);
    let loss_vjp = l1_loss_vjp_with_context_exact_native(
        &backend,
        &[-1.0, 2.0],
        &[1.0, 0.0],
        LossReduction::Mean,
        &[1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&loss_vjp, &[-0.5, 0.5], 0.0);
    let loss_jvp = l1_loss_jvp_with_context_exact_native(
        &backend,
        &[-1.0, 2.0],
        &[1.0, 1.0],
        &[1.0, 0.0],
        &[0.0, 0.0],
        LossReduction::Mean,
        DeviceId::CPU,
        &context,
    )?;
    close(&loss_jvp, &[0.0], 0.0);
    Ok(())
}

#[test]
fn max_pool_and_zero_pad_share_the_existing_geometry_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = (1..=9).map(|value| value as f32).collect::<Vec<_>>();
    let pooled = max_pool_2d_with_context_exact_native(
        &input,
        &[1, 1, 3, 3],
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        true,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(pooled.shape, [1, 1, 2, 2]);
    close(&pooled.values, &[5.0, 6.0, 8.0, 9.0], 0.0);
    let pool_vjp = max_pool_2d_vjp_with_context_exact_native(
        &input,
        &[1, 1, 3, 3],
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        true,
        &[1.0; 4],
        DeviceId::CPU,
        &context,
    )?;
    close(
        &pool_vjp.input,
        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        0.0,
    );
    let tangent = (10..=18).map(|value| value as f32).collect::<Vec<_>>();
    let pool_jvp = max_pool_2d_jvp_with_context_exact_native(
        &input,
        &tangent,
        &[1, 1, 3, 3],
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        true,
        DeviceId::CPU,
        &context,
    )?;
    close(&pool_jvp.values, &[14.0, 15.0, 17.0, 18.0], 0.0);

    let padded = zero_pad_2d_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        [1, 0, 1, 0],
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(padded.shape, [1, 3, 3]);
    close(
        &padded.values,
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        0.0,
    );
    let pad_vjp = zero_pad_2d_vjp_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        [1, 0, 1, 0],
        &[1.0; 9],
        DeviceId::CPU,
        &context,
    )?;
    close(&pad_vjp, &[1.0; 4], 0.0);
    let pad_jvp = zero_pad_2d_jvp_with_context_exact_native(
        &[2.0; 4],
        &[1, 2, 2],
        [1, 0, 1, 0],
        DeviceId::CPU,
        &context,
    )?;
    close(
        &pad_jvp.values,
        &[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0],
        0.0,
    );
    Ok(())
}

#[test]
fn pixel_rearrangement_and_relu_adapters_preserve_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = upload_f32(
        &backend,
        &[1, 4, 1, 1],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let shuffled = pixel_shuffle_module_with_context_exact_native(&backend, &input, 2, &context)?;
    assert_eq!(shuffled.descriptor().shape(), &[1, 1, 2, 2]);
    close(&f32_values(&shuffled)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let restored =
        pixel_unshuffle_module_with_context_exact_native(&backend, &shuffled, 2, &context)?;
    assert_eq!(restored.descriptor().shape(), input.descriptor().shape());
    assert_eq!(restored.descriptor().dtype(), input.descriptor().dtype());
    assert_eq!(restored.descriptor().device(), input.descriptor().device());
    close(&f32_values(&restored)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let shuffled_jvp =
        pixel_shuffle_module_jvp_with_context_exact_native(&backend, &input, 2, &context)?;
    close(&f32_values(&shuffled_jvp)?, &f32_values(&shuffled)?, 0.0);
    let shuffled_vjp = pixel_shuffle_module_vjp_with_context_exact_native(
        &backend,
        &shuffled,
        input.descriptor().shape(),
        2,
        &context,
    )?;
    close(&f32_values(&shuffled_vjp)?, &[1.0, 2.0, 3.0, 4.0], 0.0);

    let values = [-1.0, 0.0, 2.0, 8.0, f32::NAN];
    let relu = relu_module_with_context_exact_native(&backend, &values, DeviceId::CPU, &context)?;
    assert_eq!(&relu[..4], &[0.0, 0.0, 2.0, 8.0]);
    assert!(relu[4].is_nan());
    let relu_vjp = relu_module_vjp_with_context_exact_native(
        &backend,
        &values,
        &[1.0; 5],
        DeviceId::CPU,
        &context,
    )?;
    let relu_jvp = relu_module_jvp_with_context_exact_native(
        &backend,
        &values,
        &[1.0; 5],
        DeviceId::CPU,
        &context,
    )?;
    close(&relu_vjp, &relu_jvp, 0.0);
    let relu_6 = relu_6_with_context_exact_native(&backend, &values, DeviceId::CPU, &context)?;
    assert_eq!(&relu_6[..4], &[0.0, 0.0, 2.0, 6.0]);
    assert!(relu_6[4].is_nan());
    let relu_6_vjp = relu_6_vjp_with_context_exact_native(
        &backend,
        &values,
        &[1.0; 5],
        DeviceId::CPU,
        &context,
    )?;
    let relu_6_jvp = relu_6_jvp_with_context_exact_native(
        &backend,
        &values,
        &[1.0; 5],
        DeviceId::CPU,
        &context,
    )?;
    close(&relu_6_vjp, &relu_6_jvp, 0.0);
    assert_eq!(&relu_6_vjp[..4], &[0.0, 0.0, 1.0, 0.0]);
    Ok(())
}

#[test]
fn parameter_uses_the_caller_owned_autograd_tape() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &[1], &[2.0], &cancellation)?;
    let leaf = LeafId::new("task-79-parameter")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let parameter =
        parameter_exact_native(&mut tape, &input, Some(leaf.clone()), true, &cancellation)?;
    assert_eq!(parameter.storage_id(), input.storage_id());
    assert_eq!(tape.leaf_binding(&input), Some(&leaf));
    parameter_exact_native(&mut tape, &input, None, false, &cancellation)?;
    assert!(!tape.requires_grad(&input));
    Ok(())
}

#[test]
fn cancellation_precedes_part_three_validation() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    assert!(matches!(
        max_pool_2d_with_context_exact_native(
            &[],
            &[],
            [0; 2],
            [0; 2],
            [usize::MAX; 2],
            [0; 2],
            true,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartThreeError::Cancelled)
    ));
    assert!(matches!(
        relu_6_with_context_exact_native(&backend, &[], DeviceId::CPU, &context),
        Err(NeuralNetworkModulePartThreeError::Cancelled)
    ));
    assert!(matches!(
        zero_pad_2d_with_context_exact_native(&[], &[], [usize::MAX; 4], DeviceId::CPU, &context,),
        Err(NeuralNetworkModulePartThreeError::Cancelled)
    ));
    Ok(())
}

#[test]
fn all_twelve_part_three_resolutions_are_unique_and_runtime_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "neural_network_module_03")
        .ok_or("neural-network module part-three resolution slice is missing")?;
    assert_eq!(slice.len(), 12);
    let mut identifiers = BTreeSet::new();
    for contract in slice.contracts {
        assert!(identifiers.insert(contract.operation_id));
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-neural-network-module-comfy-tensor-op-904c1e14bae4"
        );
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(contract.evidence_fixture);
        let bytes = fs::read(path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
