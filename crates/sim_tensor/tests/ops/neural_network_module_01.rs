use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor,
    generated_neural_network_module_01::{
        LossReduction, NeuralNetworkModuleError, UpsampleMode,
        average_pool_2d_jvp_with_context_exact_native,
        average_pool_2d_vjp_with_context_exact_native,
        average_pool_2d_with_context_exact_native, conv1d_jvp_with_context_exact_native,
        conv1d_vjp_with_context_exact_native, conv1d_with_context_exact_native,
        group_norm_module_jvp_with_context_exact_native,
        group_norm_module_vjp_with_context_exact_native,
        group_norm_module_with_context_exact_native,
        layer_norm_module_jvp_with_context_exact_native,
        layer_norm_module_vjp_with_context_exact_native,
        layer_norm_module_with_context_exact_native, prelu_jvp_with_context_exact_native,
        prelu_vjp_with_context_exact_native, prelu_with_context_exact_native,
        silu_module_jvp_with_context_exact_native, silu_module_vjp_with_context_exact_native,
        silu_module_with_context_exact_native, smooth_l1_loss_jvp_with_context_exact_native,
        smooth_l1_loss_vjp_with_context_exact_native,
        smooth_l1_loss_with_context_exact_native,
        softmax_module_jvp_with_context_exact_native,
        softmax_module_vjp_with_context_exact_native,
        softmax_module_with_context_exact_native, tanh_module_jvp_with_context_exact_native,
        tanh_module_vjp_with_context_exact_native, tanh_module_with_context_exact_native,
        upsample_jvp_with_context_exact_native, upsample_vjp_with_context_exact_native,
        upsample_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
    workspace_limit: u64,
}

impl TestBackend {
    fn new(workspace_limit: u64) -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(workspace_limit)?;
        Ok(Self {
            backend,
            workspace_authority,
            workspace_limit,
        })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.workspace_authority
                .authorize_workspace(self.workspace_limit)?,
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

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= tolerance, "{actual} != {expected}");
    }
}

#[test]
fn average_pool_forward_and_derivatives_are_exact() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let output = average_pool_2d_with_context_exact_native(
        &backend,
        &input,
        &[1, 1, 3, 3],
        [2, 2],
        [1, 1],
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(output.shape, [1, 1, 2, 2]);
    close(&output.values, &[3.0, 4.0, 6.0, 7.0], 0.0);
    let vjp = average_pool_2d_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1, 1, 3, 3],
        [2, 2],
        [1, 1],
        &[1.0; 4],
        DeviceId::CPU,
        &context,
    )?;
    close(
        &vjp.input,
        &[0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25],
        0.0,
    );
    let jvp = average_pool_2d_jvp_with_context_exact_native(
        &backend,
        &[1.0; 9],
        &[1, 1, 3, 3],
        [2, 2],
        [1, 1],
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[1.0; 4], 0.0);
    Ok(())
}

#[test]
fn convolution_and_normalization_delegate_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let convolution = conv1d_with_context_exact_native(
        &backend,
        &[1.0, 2.0, 3.0],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[1, 1, 2],
        Some(&[0.5]),
        1,
        0,
        1,
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&convolution.values, &[4.5, 7.5], 0.0);
    let convolution_vjp = conv1d_vjp_with_context_exact_native(
        &backend,
        &[1.0, 2.0, 3.0],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[1, 1, 2],
        Some(&[0.5]),
        &[1.0, 1.0],
        1,
        0,
        1,
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&convolution_vjp.input, &[2.0, 3.0, 1.0], 0.0);
    let convolution_jvp = conv1d_jvp_with_context_exact_native(
        &backend,
        &[1.0, 2.0, 3.0],
        &[1.0; 3],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[0.0; 2],
        &[1, 1, 2],
        Some(&[0.5]),
        Some(&[0.0]),
        1,
        0,
        1,
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&convolution_jvp.values, &[3.0, 3.0], 0.0);

    let input = [1.0, 3.0, 2.0, 4.0];
    let layer = layer_norm_module_with_context_exact_native(
        &backend,
        &input,
        &[2, 2],
        &[2],
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    close(&layer, &[-0.999995, 0.999995, -0.999995, 0.999995], 1.0e-5);
    let layer_vjp = layer_norm_module_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1.0, -1.0, 1.0, -1.0],
        &[2, 2],
        &[2],
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(layer_vjp.input.len(), 4);
    let layer_jvp = layer_norm_module_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0, 0.0, 1.0, 0.0],
        &[2, 2],
        &[2],
        None,
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(layer_jvp.len(), 4);

    let group = group_norm_module_with_context_exact_native(
        &backend,
        &input,
        &[1, 2, 2],
        2,
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(group.len(), 4);
    let group_vjp = group_norm_module_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 4],
        &[1, 2, 2],
        2,
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(group_vjp.input.len(), 4);
    let group_jvp = group_norm_module_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 4],
        &[1, 2, 2],
        2,
        None,
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(group_jvp.len(), 4);
    Ok(())
}

#[test]
fn prelu_activation_loss_and_softmax_have_analytic_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [-2.0, 1.0, -4.0, 3.0];
    let prelu = prelu_with_context_exact_native(
        &backend,
        &input,
        &[1, 2, 2],
        &[0.5, 0.25],
        DeviceId::CPU,
        &context,
    )?;
    close(&prelu, &[-1.0, 1.0, -1.0, 3.0], 0.0);
    let prelu_vjp = prelu_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1, 2, 2],
        &[0.5, 0.25],
        &[1.0; 4],
        DeviceId::CPU,
        &context,
    )?;
    close(&prelu_vjp.input, &[0.5, 1.0, 0.25, 1.0], 0.0);
    close(&prelu_vjp.weight, &[-2.0, -4.0], 0.0);
    let prelu_jvp = prelu_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 4],
        &[1, 2, 2],
        &[0.5, 0.25],
        &[0.1, 0.2],
        DeviceId::CPU,
        &context,
    )?;
    close(&prelu_jvp, &[0.3, 1.0, -0.55, 1.0], 1.0e-6);

    let silu = silu_module_with_context_exact_native(
        &backend,
        &[0.0, 1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&silu, &[0.0, 0.7310586], 1.0e-6);
    assert_eq!(
        silu_module_vjp_with_context_exact_native(
            &backend,
            &[0.0, 1.0],
            &[1.0; 2],
            DeviceId::CPU,
            &context,
        )?
        .len(),
        2
    );
    assert_eq!(
        silu_module_jvp_with_context_exact_native(
            &backend,
            &[0.0, 1.0],
            &[1.0; 2],
            DeviceId::CPU,
            &context,
        )?
        .len(),
        2
    );

    let loss = smooth_l1_loss_with_context_exact_native(
        &backend,
        &[0.0, 2.0],
        &[1.0, 0.0],
        1.0,
        LossReduction::Mean,
        DeviceId::CPU,
        &context,
    )?;
    close(&loss, &[1.0], 0.0);
    let loss_vjp = smooth_l1_loss_vjp_with_context_exact_native(
        &backend,
        &[0.0, 2.0],
        &[1.0, 0.0],
        1.0,
        LossReduction::Mean,
        &[1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&loss_vjp, &[-0.5, 0.5], 0.0);
    let loss_jvp = smooth_l1_loss_jvp_with_context_exact_native(
        &backend,
        &[0.0, 2.0],
        &[1.0, 1.0],
        &[1.0, 0.0],
        &[0.0, 0.0],
        1.0,
        LossReduction::Mean,
        DeviceId::CPU,
        &context,
    )?;
    close(&loss_jvp, &[0.0], 0.0);

    let softmax = softmax_module_with_context_exact_native(
        &backend,
        &[1.0, 2.0],
        &[1, 2],
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&softmax, &[0.26894143, 0.7310586], 1.0e-6);
    assert_eq!(
        softmax_module_vjp_with_context_exact_native(
            &backend,
            &[1.0, 2.0],
            &[1.0, 0.0],
            &[1, 2],
            1,
            DeviceId::CPU,
            &context,
        )?
        .len(),
        2
    );
    assert_eq!(
        softmax_module_jvp_with_context_exact_native(
            &backend,
            &[1.0, 2.0],
            &[1.0, 0.0],
            &[1, 2],
            1,
            DeviceId::CPU,
            &context,
        )?
        .len(),
        2
    );
    Ok(())
}

#[test]
fn tanh_and_upsample_delegate_tensor_owners_including_aligned_coordinates()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[1, 1, 2, 2], &[0.0, 1.0, 2.0, 3.0], &cancellation)?;
    let tangent = upload_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &cancellation)?;
    let tanh = tanh_module_with_context_exact_native(&backend, &input, &context)?;
    close(&values(&tanh)?, &[0.0, 0.7615942, 0.9640276, 0.9950548], 1.0e-6);
    assert_eq!(
        tanh_module_vjp_with_context_exact_native(&backend, &input, &tangent, &context)?
            .descriptor(),
        input.descriptor()
    );
    assert_eq!(
        tanh_module_jvp_with_context_exact_native(&backend, &input, &tangent, &context)?
            .descriptor(),
        input.descriptor()
    );
    let aligned = upsample_with_context_exact_native(
        &backend,
        &input,
        3,
        3,
        UpsampleMode::Bilinear,
        Some(true),
        &context,
    )?;
    close(
        &values(&aligned)?,
        &[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0],
        1.0e-6,
    );
    let aligned_gradient = upload_f32(&backend, &[1, 1, 3, 3], &[1.0; 9], &cancellation)?;
    let vjp = upsample_vjp_with_context_exact_native(
        &backend,
        &input,
        &aligned_gradient,
        3,
        3,
        UpsampleMode::Bilinear,
        Some(true),
        &context,
    )?;
    close(&values(&vjp)?, &[2.25; 4], 1.0e-6);
    let jvp = upsample_jvp_with_context_exact_native(
        &backend,
        &tangent,
        3,
        3,
        UpsampleMode::Bilinear,
        Some(true),
        &context,
    )?;
    close(&values(&jvp)?, &[1.0; 9], 1.0e-6);
    let nearest = upsample_with_context_exact_native(
        &backend,
        &input,
        4,
        4,
        UpsampleMode::Nearest,
        None,
        &context,
    )?;
    assert_eq!(nearest.descriptor().shape(), &[1, 1, 4, 4]);
    Ok(())
}

#[test]
fn unsupported_devices_and_cancelled_invalid_inputs_fail_closed()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        average_pool_2d_with_context_exact_native(
            &backend,
            &[1.0; 4],
            &[1, 1, 2, 2],
            [2, 2],
            [2, 2],
            cuda,
            &context,
        ),
        Err(NeuralNetworkModuleError::UnsupportedDevice { .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution(&cancelled)?;
    let assert_cancelled = |result: Result<(), NeuralNetworkModuleError>| {
        assert_eq!(result, Err(NeuralNetworkModuleError::Cancelled));
        assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    };
    assert_cancelled(
        average_pool_2d_with_context_exact_native(
            &backend,
            &[],
            &[],
            [0, 0],
            [0, 0],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        conv1d_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            0,
            0,
            0,
            0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        group_norm_module_with_context_exact_native(
            &backend,
            &[],
            &[],
            0,
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        layer_norm_module_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        prelu_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        silu_module_with_context_exact_native(&backend, &[], cuda, &cancelled_context)
            .map(|_| ()),
    );
    assert_cancelled(
        smooth_l1_loss_with_context_exact_native(
            &backend,
            &[],
            &[1.0],
            -1.0,
            LossReduction::Mean,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        softmax_module_with_context_exact_native(
            &backend,
            &[],
            &[],
            99,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        average_pool_2d_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            [0, 0],
            [0, 0],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        average_pool_2d_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            [0, 0],
            [0, 0],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        conv1d_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            &[],
            0,
            0,
            0,
            0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        conv1d_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            None,
            None,
            0,
            0,
            0,
            0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        group_norm_module_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            0,
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        group_norm_module_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            0,
            None,
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        layer_norm_module_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        layer_norm_module_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            None,
            None,
            -1.0,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        prelu_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        prelu_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        silu_module_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        silu_module_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        smooth_l1_loss_vjp_with_context_exact_native(
            &backend,
            &[],
            &[1.0],
            -1.0,
            LossReduction::Mean,
            &[],
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        smooth_l1_loss_jvp_with_context_exact_native(
            &backend,
            &[],
            &[1.0],
            &[1.0],
            &[],
            -1.0,
            LossReduction::Mean,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        softmax_module_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            99,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        softmax_module_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            99,
            cuda,
            &cancelled_context,
        )
        .map(|_| ()),
    );

    let invalid_descriptor = TensorDescriptor::contiguous(
        vec![1],
        DType::I64,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let invalid_tensor = backend
        .upload_bytes(
            invalid_descriptor,
            &1_i64.to_ne_bytes(),
            &backend.execution(&cancellation)?,
        )?
        .0;
    let invalid_bytes = invalid_tensor.contiguous_bytes()?.to_vec();
    assert_cancelled(
        tanh_module_with_context_exact_native(&backend, &invalid_tensor, &cancelled_context)
            .map(|_| ()),
    );
    assert_cancelled(
        tanh_module_vjp_with_context_exact_native(
            &backend,
            &invalid_tensor,
            &invalid_tensor,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        tanh_module_jvp_with_context_exact_native(
            &backend,
            &invalid_tensor,
            &invalid_tensor,
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        upsample_with_context_exact_native(
            &backend,
            &invalid_tensor,
            0,
            0,
            UpsampleMode::Nearest,
            Some(true),
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        upsample_vjp_with_context_exact_native(
            &backend,
            &invalid_tensor,
            &invalid_tensor,
            0,
            0,
            UpsampleMode::Nearest,
            Some(true),
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_cancelled(
        upsample_jvp_with_context_exact_native(
            &backend,
            &invalid_tensor,
            0,
            0,
            UpsampleMode::Nearest,
            Some(true),
            &cancelled_context,
        )
        .map(|_| ()),
    );
    assert_eq!(invalid_tensor.contiguous_bytes()?, invalid_bytes);
    Ok(())
}

#[test]
fn all_twelve_resolutions_are_unique_and_runtime_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "neural_network_module_01")
        .ok_or("neural-network module part-one resolution slice is missing")?;
    assert_eq!(slice.len(), 12);
    let mut ids = BTreeSet::new();
    for contract in slice.contracts {
        assert!(ids.insert(contract.operation_id));
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-neural-network-module-comfy-tensor-op-0e602e58360a"
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
