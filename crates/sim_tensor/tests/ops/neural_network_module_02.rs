use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_08::ElementwiseRuntimePartEightError,
    generated_neural_network_functional_01::{EmbeddingOptions, NeuralNetworkFunctionalError},
    generated_neural_network_module_01::LossReduction,
    generated_neural_network_module_02::{
        BATCH_NORM_1D_OPERATION_ID, NeuralNetworkModulePartTwoError,
        adaptive_average_pool_2d_module_jvp_with_context_exact_native,
        adaptive_average_pool_2d_module_vjp_with_context_exact_native,
        adaptive_average_pool_2d_module_with_context_exact_native,
        average_pool_3d_module_jvp_with_context_exact_native,
        average_pool_3d_module_vjp_with_context_exact_native,
        average_pool_3d_module_with_context_exact_native,
        batch_norm_module_jvp_with_context_exact_native,
        batch_norm_module_vjp_with_context_exact_native,
        batch_norm_module_with_context_exact_native, conv_2d_jvp_with_context_exact_native,
        conv_2d_vjp_with_context_exact_native, conv_2d_with_context_exact_native,
        embedding_module_jvp_with_context_exact_native,
        embedding_module_vjp_with_context_exact_native,
        embedding_module_with_context_exact_native, huber_loss_jvp_with_context_exact_native,
        huber_loss_vjp_with_context_exact_native, huber_loss_with_context_exact_native,
        instance_norm_2d_jvp_with_context_exact_native,
        instance_norm_2d_tensor_with_context_exact_native,
        instance_norm_2d_vjp_with_context_exact_native,
        instance_norm_2d_with_context_exact_native,
        leaky_relu_module_jvp_with_context_exact_native,
        leaky_relu_module_vjp_with_context_exact_native,
        leaky_relu_module_with_context_exact_native,
        linear_module_jvp_with_context_exact_native,
        linear_module_vjp_with_context_exact_native,
        linear_module_with_context_exact_native,
        multihead_attention_projected_jvp_with_context_exact_native,
        multihead_attention_projected_vjp_with_context_exact_native,
        multihead_attention_projected_with_context_exact_native,
        replication_pad_2d_jvp_with_context_exact_native,
        replication_pad_2d_tensor_with_context_exact_native,
        replication_pad_2d_vjp_with_context_exact_native,
        replication_pad_2d_with_context_exact_native,
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

fn upload_i64(
    backend: &TestBackend,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(descriptor, &bytes, &backend.execution(cancellation)?)?
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
        assert!((actual - expected).abs() <= tolerance, "{actual} != {expected}");
    }
}

#[test]
fn pooling_and_replication_padding_share_canonical_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let values = (1..=9).map(|value| value as f32).collect::<Vec<_>>();
    let adaptive = adaptive_average_pool_2d_module_with_context_exact_native(
        &backend,
        &values,
        &[1, 1, 3, 3],
        [2, 2],
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(adaptive.shape, [1, 1, 2, 2]);
    close(&adaptive.values, &[3.0, 4.0, 6.0, 7.0], 0.0);
    let adaptive_vjp = adaptive_average_pool_2d_module_vjp_with_context_exact_native(
        &backend,
        &values,
        &[1, 1, 3, 3],
        [2, 2],
        &[1.0; 4],
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(adaptive_vjp.input.len(), 9);
    let adaptive_jvp = adaptive_average_pool_2d_module_jvp_with_context_exact_native(
        &backend,
        &[1.0; 9],
        &[1, 1, 3, 3],
        [2, 2],
        DeviceId::CPU,
        &context,
    )?;
    close(&adaptive_jvp.values, &[1.0; 4], 0.0);

    let cube = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
    let average = average_pool_3d_module_with_context_exact_native(
        &backend,
        &cube,
        &[1, 1, 2, 2, 2],
        [2, 2, 2],
        [1, 1, 1],
        DeviceId::CPU,
        &context,
    )?;
    close(&average.values, &[4.5], 0.0);
    let average_vjp = average_pool_3d_module_vjp_with_context_exact_native(
        &backend,
        &cube,
        &[1, 1, 2, 2, 2],
        [2, 2, 2],
        [1, 1, 1],
        &[1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&average_vjp.input, &[0.125; 8], 0.0);
    let average_jvp = average_pool_3d_module_jvp_with_context_exact_native(
        &backend,
        &[1.0; 8],
        &[1, 1, 2, 2, 2],
        [2, 2, 2],
        [1, 1, 1],
        DeviceId::CPU,
        &context,
    )?;
    close(&average_jvp.values, &[1.0], 0.0);

    let padded = replication_pad_2d_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 1, 2],
        [1, 1, 1, 0],
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(padded.shape, [1, 2, 4]);
    close(&padded.values, &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], 0.0);
    let padding_vjp = replication_pad_2d_vjp_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 1, 2],
        [1, 1, 1, 0],
        &[1.0; 8],
        DeviceId::CPU,
        &context,
    )?;
    close(&padding_vjp, &[4.0, 4.0], 0.0);
    let padding_jvp = replication_pad_2d_jvp_with_context_exact_native(
        &[3.0, 5.0],
        &[1, 1, 2],
        [1, 1, 1, 0],
        DeviceId::CPU,
        &context,
    )?;
    close(&padding_jvp.values, &[3.0, 3.0, 5.0, 5.0, 3.0, 3.0, 5.0, 5.0], 0.0);
    let tensor_input = upload_f32(&backend, &[1, 1, 3, 3], &values, &cancellation)?;
    let tensor_padded = replication_pad_2d_tensor_with_context_exact_native(
        &*backend,
        &tensor_input,
        [1, 2, 1, 0],
        &context,
    )?;
    assert_eq!(tensor_padded.descriptor().shape(), [1, 1, 4, 6]);
    assert_eq!(tensor_padded.descriptor().dtype(), tensor_input.descriptor().dtype());
    assert_eq!(tensor_padded.descriptor().device(), tensor_input.descriptor().device());
    assert_eq!(tensor_padded.descriptor().stream(), context.stream);
    close(
        &f32_values(&tensor_padded)?,
        &[
            1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0,
            4.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 8.0, 9.0, 9.0, 9.0,
        ],
        0.0,
    );
    Ok(())
}

#[test]
fn normalization_activation_and_huber_delegate_existing_math() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut mean = [0.0, 0.0];
    let mut variance = [1.0, 1.0];
    let batch = batch_norm_module_with_context_exact_native(
        &backend,
        &input,
        &[2, 2],
        2,
        Some(&mut mean),
        Some(&mut variance),
        None,
        None,
        true,
        0.1,
        1.0e-5,
        BATCH_NORM_1D_OPERATION_ID,
        DeviceId::CPU,
        &context,
    )?;
    close(&batch, &[-0.999995, -0.999995, 0.999995, 0.999995], 1.0e-5);
    assert!(mean.iter().any(|value| *value != 0.0));
    let batch_vjp = batch_norm_module_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 4],
        &[2, 2],
        2,
        Some(&mean),
        Some(&variance),
        None,
        None,
        true,
        1.0e-5,
        BATCH_NORM_1D_OPERATION_ID,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(batch_vjp.input.len(), 4);
    let batch_jvp = batch_norm_module_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0, 0.0, 1.0, 0.0],
        &[2, 2],
        2,
        Some(&mean),
        Some(&variance),
        None,
        None,
        None,
        true,
        1.0e-5,
        BATCH_NORM_1D_OPERATION_ID,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(batch_jvp.len(), 4);

    let instance = instance_norm_2d_with_context_exact_native(
        &backend,
        &input,
        &[1, 2, 1, 2],
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &context,
    )?;
    close(&instance, &[-0.99998, 0.99998, -0.99998, 0.99998], 1.0e-4);
    let instance_input = upload_f32(
        &backend,
        &[1, 2, 1, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let instance_tensor = instance_norm_2d_tensor_with_context_exact_native(
        &*backend,
        &instance_input,
        1.0e-5,
        &context,
    )?;
    close(
        &f32_values(&instance_tensor)?,
        &[-0.99998, 0.99998, -0.99998, 0.99998],
        1.0e-4,
    );
    assert_ne!(instance_tensor.storage_id(), instance_input.storage_id());
    assert_eq!(
        instance_norm_2d_vjp_with_context_exact_native(
            &backend,
            &input,
            &[1.0; 4],
            &[1, 2, 1, 2],
            None,
            None,
            1.0e-5,
            DeviceId::CPU,
            &context,
        )?
        .input
        .len(),
        4
    );
    assert_eq!(
        instance_norm_2d_jvp_with_context_exact_native(
            &backend,
            &input,
            &[1.0; 4],
            &[1, 2, 1, 2],
            None,
            None,
            None,
            1.0e-5,
            DeviceId::CPU,
            &context,
        )?
        .len(),
        4
    );

    let activation = leaky_relu_module_with_context_exact_native(
        &backend,
        &[-2.0, 3.0],
        0.1,
        DeviceId::CPU,
        &context,
    )?;
    close(&activation, &[-0.2, 3.0], 0.0);
    close(
        &leaky_relu_module_vjp_with_context_exact_native(
            &backend,
            &[-2.0, 3.0],
            &[2.0, 2.0],
            0.1,
            DeviceId::CPU,
            &context,
        )?,
        &[0.2, 2.0],
        0.0,
    );
    close(
        &leaky_relu_module_jvp_with_context_exact_native(
            &backend,
            &[-2.0, 3.0],
            &[2.0, 2.0],
            0.1,
            DeviceId::CPU,
            &context,
        )?,
        &[0.2, 2.0],
        0.0,
    );

    let huber = huber_loss_with_context_exact_native(
        &backend,
        &[1.0, 3.0],
        &[0.0, 0.0],
        2.0,
        LossReduction::Mean,
        DeviceId::CPU,
        &context,
    )?;
    close(&huber, &[2.25], 0.0);
    close(
        &huber_loss_vjp_with_context_exact_native(
            &backend,
            &[1.0, 3.0],
            &[0.0, 0.0],
            2.0,
            LossReduction::Mean,
            &[1.0],
            DeviceId::CPU,
            &context,
        )?,
        &[0.5, 1.0],
        0.0,
    );
    close(
        &huber_loss_jvp_with_context_exact_native(
            &backend,
            &[1.0, 3.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            2.0,
            LossReduction::Mean,
            DeviceId::CPU,
            &context,
        )?,
        &[1.5],
        0.0,
    );
    Ok(())
}

#[test]
fn convolution_linear_embedding_and_attention_preserve_derivatives() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let convolution = conv_2d_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        &[1.0; 4],
        &[1, 1, 2, 2],
        Some(&[0.5]),
        [1, 1],
        [0, 0],
        [1, 1],
        1,
        DeviceId::CPU,
        &context,
    )?;
    close(&convolution.values, &[10.5], 0.0);
    assert_eq!(
        conv_2d_vjp_with_context_exact_native(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 2],
            &[1.0; 4],
            &[1, 1, 2, 2],
            Some(&[0.5]),
            [1, 1],
            [0, 0],
            [1, 1],
            1,
            &[1.0],
            DeviceId::CPU,
            &context,
        )?
        .input,
        [1.0; 4]
    );
    close(
        &conv_2d_jvp_with_context_exact_native(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0; 4],
            &[1, 1, 2, 2],
            &[1.0; 4],
            &[0.0; 4],
            &[1, 1, 2, 2],
            Some(&[0.5]),
            Some(&[0.0]),
            [1, 1],
            [0, 0],
            [1, 1],
            1,
            DeviceId::CPU,
            &context,
        )?
        .values,
        &[4.0],
        0.0,
    );

    let linear = linear_module_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 2],
        &[1.0, 0.0, 0.0, 2.0],
        &[2, 2],
        Some(&[0.5, -1.0]),
        DeviceId::CPU,
        &context,
    )?;
    close(&linear.values, &[1.5, 3.0], 0.0);
    assert_eq!(
        linear_module_vjp_with_context_exact_native(
            &[1.0, 2.0],
            &[1, 2],
            &[1.0, 0.0, 0.0, 2.0],
            &[2, 2],
            Some(&[0.5, -1.0]),
            &[1.0, 1.0],
            DeviceId::CPU,
            &context,
        )?
        .input,
        [1.0, 2.0]
    );
    assert_eq!(
        linear_module_jvp_with_context_exact_native(
            &[1.0, 2.0],
            &[1.0, 1.0],
            &[1, 2],
            &[1.0, 0.0, 0.0, 2.0],
            &[0.0; 4],
            &[2, 2],
            Some(&[0.5, -1.0]),
            Some(&[0.0, 0.0]),
            DeviceId::CPU,
            &context,
        )?
        .values,
        [1.0, 2.0]
    );

    let indices = upload_i64(&backend, &[2], &[0, 2], &cancellation)?;
    let mut weight = upload_f32(
        &backend,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let embedding = embedding_module_with_context_exact_native(
        &backend,
        &indices,
        &mut weight,
        EmbeddingOptions::default(),
        &context,
    )?;
    close(&f32_values(&embedding)?, &[1.0, 2.0, 5.0, 6.0], 0.0);
    let embedding_gradient = upload_f32(&backend, &[2, 2], &[1.0; 4], &cancellation)?;
    let embedding_vjp = embedding_module_vjp_with_context_exact_native(
        &backend,
        &indices,
        &weight,
        EmbeddingOptions::default(),
        &embedding_gradient,
        &context,
    )?;
    close(&f32_values(&embedding_vjp)?, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0], 0.0);
    let weight_tangent = upload_f32(&backend, &[3, 2], &[1.0; 6], &cancellation)?;
    let embedding_jvp = embedding_module_jvp_with_context_exact_native(
        &backend,
        &indices,
        &weight,
        &weight_tangent,
        EmbeddingOptions::default(),
        &context,
    )?;
    close(&f32_values(&embedding_jvp)?, &[1.0; 4], 0.0);

    let query = [1.0, 0.0];
    let key = [1.0, 0.0, 0.0, 1.0];
    let value = [2.0, 0.0, 0.0, 4.0];
    let attention = multihead_attention_projected_with_context_exact_native(
        &backend,
        &query,
        &[1, 1, 2],
        &key,
        &[2, 1, 2],
        &value,
        &[2, 1, 2],
        1,
        &context,
    )?;
    close(&attention.values, &[1.3395231, 1.3209538], 1.0e-5);
    let attention_vjp = multihead_attention_projected_vjp_with_context_exact_native(
        &backend,
        &query,
        &[1, 1, 2],
        &key,
        &[2, 1, 2],
        &value,
        &[2, 1, 2],
        1,
        &[1.0, 1.0],
        &context,
    )?;
    assert_eq!(attention_vjp.query.len(), query.len());
    assert_eq!(attention_vjp.key.len(), key.len());
    assert_eq!(attention_vjp.value.len(), value.len());
    let attention_jvp = multihead_attention_projected_jvp_with_context_exact_native(
        &backend,
        &query,
        &[1.0, 0.0],
        &[1, 1, 2],
        &key,
        &[0.0; 4],
        &[2, 1, 2],
        &value,
        &[0.0; 4],
        &[2, 1, 2],
        1,
        &context,
    )?;
    assert!(attention_jvp.values.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn cancellation_precedes_part_two_validation() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    assert!(matches!(
        replication_pad_2d_with_context_exact_native(
            &[],
            &[],
            [usize::MAX; 4],
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartTwoError::Cancelled)
    ));
    assert!(matches!(
        huber_loss_with_context_exact_native(
            &backend,
            &[],
            &[],
            -1.0,
            LossReduction::Mean,
            DeviceId::CPU,
            &context,
        ),
        Err(NeuralNetworkModulePartTwoError::Cancelled)
    ));
    assert!(matches!(
        multihead_attention_projected_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            0,
            &context,
        ),
        Err(NeuralNetworkModulePartTwoError::Cancelled)
    ));
    Ok(())
}

#[test]
fn embedding_preserves_typed_workspace_exhaustion() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let indices = upload_i64(&backend, &[2], &[0, 2], &cancellation)?;
    let mut weight = upload_f32(
        &backend,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let workspace = backend.authority.authorize_workspace(0)?;
    let context =
        backend
            .backend
            .execution_context(StreamId::DEFAULT, workspace, &cancellation);
    assert!(matches!(
        embedding_module_with_context_exact_native(
            &backend,
            &indices,
            &mut weight,
            EmbeddingOptions::default(),
            &context,
        ),
        Err(NeuralNetworkModulePartTwoError::Tensor(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
                | sim_tensor::TensorError::AllocationFailed { .. }
                | sim_tensor::TensorError::ResourceLimitExceeded { .. }
        ))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(
        NeuralNetworkModulePartTwoError::from(NeuralNetworkFunctionalError::Tensor(
            TensorError::Cancelled,
        )),
        NeuralNetworkModulePartTwoError::Cancelled
    );
    assert_eq!(
        NeuralNetworkModulePartTwoError::from(NeuralNetworkFunctionalError::IndexSelect(
            ElementwiseRuntimePartEightError::Cancelled,
        )),
        NeuralNetworkModulePartTwoError::Cancelled
    );
    assert!(matches!(
        NeuralNetworkModulePartTwoError::from(NeuralNetworkFunctionalError::IndexSelect(
            ElementwiseRuntimePartEightError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested: 64,
                    authorized: 32,
                    in_use: 0,
                },
            ),
        )),
        NeuralNetworkModulePartTwoError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded {
                requested: 64,
                authorized: 32,
                in_use: 0,
            }
        )
    ));
    Ok(())
}

#[test]
fn all_twelve_part_two_resolutions_are_unique_and_runtime_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "neural_network_module_02")
        .ok_or("neural-network module part-two resolution slice is missing")?;
    assert_eq!(slice.len(), 12);
    let mut ids = BTreeSet::new();
    for contract in slice.contracts {
        assert!(ids.insert(contract.operation_id));
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-neural-network-module-comfy-tensor-op-5b8ce1451811"
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
