use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Scalar, StreamId, Tensor, TensorDescriptor,
    generated_accelerated_attention_kernel_01::{
        AttentionKernelKind, AttentionKernelRequest, AttentionLayout, AttentionShape,
    },
    generated_neural_network_functional_01::{
        EmbeddingOptions, NeuralNetworkFunctionalError, SpatialParameters2d,
        cosine_similarity_jvp_with_context_exact_native,
        cosine_similarity_vjp_with_context_exact_native,
        cosine_similarity_with_context_exact_native, embedding_jvp_with_context_exact_native,
        embedding_vjp_with_context_exact_native, embedding_with_context_exact_native,
        fold_jvp_with_context_exact_native, fold_vjp_with_context_exact_native,
        fold_with_context_exact_native, glu_jvp_with_context_exact_native,
        glu_vjp_with_context_exact_native, glu_with_context_exact_native,
        linear_jvp_with_context_exact_native, linear_vjp_with_context_exact_native,
        linear_with_context_exact_native, one_hot_with_context_exact_native,
        pixel_shuffle_jvp_with_context_exact_native, pixel_shuffle_vjp_with_context_exact_native,
        pixel_shuffle_tensor_with_context_exact_native, pixel_shuffle_with_context_exact_native,
        pixel_unshuffle_jvp_with_context_exact_native,
        pixel_unshuffle_vjp_with_context_exact_native, pixel_unshuffle_with_context_exact_native,
        pixel_unshuffle_tensor_with_context_exact_native,
        scaled_dot_product_attention_jvp_with_context_exact_native,
        scaled_dot_product_attention_vjp_with_context_exact_native,
        scaled_dot_product_attention_with_context_exact_native,
        sigmoid_jvp_with_context_exact_native, sigmoid_vjp_with_context_exact_native,
        sigmoid_with_context_exact_native, softplus_jvp_with_context_exact_native,
        softplus_vjp_with_context_exact_native, softplus_with_context_exact_native,
        unfold_jvp_with_context_exact_native, unfold_vjp_with_context_exact_native,
        unfold_with_context_exact_native,
    },
};
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
        self.execution_with_workspace(self.workspace_limit, cancellation)
    }

    fn execution_with_workspace<'a>(
        &self,
        bytes: u64,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.workspace_authority.authorize_workspace(bytes)?,
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
        .map(|bytes| {
            let mut value = [0_u8; 4];
            value.copy_from_slice(bytes);
            f32::from_ne_bytes(value)
        })
        .collect())
}

fn close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }
}

fn assert_cancelled<T>(
    result: Result<T, NeuralNetworkFunctionalError>,
    context: &ExecutionContext<'_>,
) {
    assert!(matches!(
        result,
        Err(NeuralNetworkFunctionalError::Cancelled)
    ));
    assert_eq!(context.scratch.peak_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
}

#[test]
fn cosine_similarity_composes_canonical_normalization_and_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_with_workspace(0, &cancellation)?;
    let input_one = [1.0, 0.0, 0.0, 1.0];
    let input_two = [1.0, 0.0, 1.0, 0.0];
    let output = cosine_similarity_with_context_exact_native(
        &backend,
        &input_one,
        &[2, 2],
        &input_two,
        &[2, 2],
        1,
        1.0e-8,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(output.shape, [2]);
    close(&output.values, &[1.0, 0.0], 1.0e-6);
    let gradients = cosine_similarity_vjp_with_context_exact_native(
        &backend,
        &input_one,
        &[2, 2],
        &input_two,
        &[2, 2],
        1,
        1.0e-8,
        &[1.0, 1.0],
        DeviceId::CPU,
        &execution,
    )?;
    close(&gradients.input_one, &[0.0, 0.0, 1.0, 0.0], 1.0e-5);
    let tangent = cosine_similarity_jvp_with_context_exact_native(
        &backend,
        &input_one,
        &[0.0, 0.0, 1.0, 0.0],
        &[2, 2],
        &input_two,
        &[0.0; 4],
        &[2, 2],
        1,
        1.0e-8,
        DeviceId::CPU,
        &execution,
    )?;
    close(&tangent.values, &[0.0, 1.0], 1.0e-5);
    Ok(())
}

#[test]
fn embedding_delegates_index_select_and_preserves_padding_gradient_rules()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let indices = upload_i64(&backend, &[2, 2], &[0, 2, 2, 1], &cancellation)?;
    let mut weight = upload_f32(
        &backend,
        &[3, 2],
        &[10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let options = EmbeddingOptions {
        padding_index: Some(0),
        scale_gradient_by_frequency: true,
        ..EmbeddingOptions::default()
    };
    let output =
        embedding_with_context_exact_native(&backend, &indices, &mut weight, options, &execution)?;
    assert_eq!(output.descriptor().shape(), &[2, 2, 2]);
    close(
        &f32_values(&output)?,
        &[10.0, 20.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0],
        0.0,
    );
    let output_gradient = upload_f32(&backend, &[2, 2, 2], &[1.0; 8], &cancellation)?;
    let gradient = embedding_vjp_with_context_exact_native(
        &backend,
        &indices,
        &weight,
        options,
        &output_gradient,
        &execution,
    )?;
    close(
        &f32_values(&gradient)?,
        &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        1.0e-6,
    );
    let tangent = upload_f32(&backend, &[3, 2], &[1.0; 6], &cancellation)?;
    let tangent_output = embedding_jvp_with_context_exact_native(
        &backend, &indices, &weight, &tangent, options, &execution,
    )?;
    close(&f32_values(&tangent_output)?, &[1.0; 8], 0.0);
    Ok(())
}

#[test]
fn canonical_cosine_derivatives_are_zero_scratch_and_match_legacy_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_with_workspace(0, &cancellation)?;
    let input_one = [1.0, 0.0, 0.0, 1.0];
    let input_two = [1.0, 0.0, 1.0, 0.0];
    let output = cosine_similarity_with_context_exact_native(
        &backend,
        &input_one,
        &[2, 2],
        &input_two,
        &[2, 2],
        1,
        1.0e-8,
        DeviceId::CPU,
        &execution,
    )?;
    close(&output.values, &[1.0, 0.0], 1.0e-6);
    let gradients = cosine_similarity_vjp_with_context_exact_native(
        &backend,
        &input_one,
        &[2, 2],
        &input_two,
        &[2, 2],
        1,
        1.0e-8,
        &[1.0, 1.0],
        DeviceId::CPU,
        &execution,
    )?;
    close(&gradients.input_one, &[0.0, 0.0, 1.0, 0.0], 1.0e-5);
    let tangent = cosine_similarity_jvp_with_context_exact_native(
        &backend,
        &input_one,
        &[0.0, 0.0, 1.0, 0.0],
        &[2, 2],
        &input_two,
        &[0.0; 4],
        &[2, 2],
        1,
        1.0e-8,
        DeviceId::CPU,
        &execution,
    )?;
    close(&tangent.values, &[0.0, 1.0], 1.0e-5);
    assert_eq!(execution.scratch.peak_bytes(), 0);
    Ok(())
}

#[test]
fn canonical_embedding_leases_exact_staging_and_is_atomic() -> Result<(), Box<dyn std::error::Error>>
{
    const GENEROUS: u64 = 1024 * 1024;

    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let indices = upload_i64(&backend, &[2, 2], &[0, 2, 2, 1], &cancellation)?;
    let original_weight = upload_f32(
        &backend,
        &[3, 2],
        &[10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let options = EmbeddingOptions {
        padding_index: Some(0),
        max_norm: Some(5.0),
        scale_gradient_by_frequency: true,
        ..EmbeddingOptions::default()
    };

    let mut weight = original_weight.clone();
    let forward_probe = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let output = embedding_with_context_exact_native(
        &backend,
        &indices,
        &mut weight,
        options,
        &forward_probe,
    )?;
    assert_eq!(output.descriptor().shape(), &[2, 2, 2]);
    let forward_peak = forward_probe.scratch.peak_bytes();
    assert!(forward_peak > 0);
    assert_eq!(forward_probe.scratch.in_use_bytes(), 0);

    let mut exact_weight = original_weight.clone();
    let forward_exact = backend.execution_with_workspace(forward_peak, &cancellation)?;
    embedding_with_context_exact_native(
        &backend,
        &indices,
        &mut exact_weight,
        options,
        &forward_exact,
    )?;
    assert_eq!(forward_exact.scratch.peak_bytes(), forward_peak);

    let before = f32_values(&original_weight)?;
    let mut insufficient_weight = original_weight.clone();
    let insufficient = backend.execution_with_workspace(forward_peak - 1, &cancellation)?;
    assert!(
        embedding_with_context_exact_native(
            &backend,
            &indices,
            &mut insufficient_weight,
            options,
            &insufficient,
        )
        .is_err()
    );
    close(&f32_values(&insufficient_weight)?, &before, 0.0);
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let output_gradient = upload_f32(&backend, &[2, 2, 2], &[1.0; 8], &cancellation)?;
    let vjp_probe = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let gradient = embedding_vjp_with_context_exact_native(
        &backend,
        &indices,
        &original_weight,
        options,
        &output_gradient,
        &vjp_probe,
    )?;
    close(
        &f32_values(&gradient)?,
        &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        1.0e-6,
    );
    let vjp_peak = vjp_probe.scratch.peak_bytes();
    assert!(vjp_peak > 0);
    assert_eq!(vjp_probe.scratch.in_use_bytes(), 0);

    let zero_width_weight = upload_f32(&backend, &[3, 0], &[], &cancellation)?;
    let zero_width_gradient = upload_f32(&backend, &[2, 2, 0], &[], &cancellation)?;
    let zero_width_context = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let zero_width_vjp = embedding_vjp_with_context_exact_native(
        &backend,
        &indices,
        &zero_width_weight,
        options,
        &zero_width_gradient,
        &zero_width_context,
    )?;
    assert_eq!(zero_width_vjp.descriptor().shape(), &[3, 0]);
    assert_eq!(zero_width_context.scratch.in_use_bytes(), 0);

    let tangent = upload_f32(&backend, &[3, 2], &[1.0; 6], &cancellation)?;
    let jvp_context = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let tangent_output = embedding_jvp_with_context_exact_native(
        &backend,
        &indices,
        &original_weight,
        &tangent,
        options,
        &jvp_context,
    )?;
    close(&f32_values(&tangent_output)?, &[1.0; 8], 0.0);
    assert_eq!(jvp_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_with_workspace(forward_peak, &cancelled)?;
    let mut cancelled_weight = original_weight;
    assert!(
        embedding_with_context_exact_native(
            &backend,
            &indices,
            &mut cancelled_weight,
            options,
            &cancelled_context,
        )
        .is_err()
    );
    close(&f32_values(&cancelled_weight)?, &before, 0.0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn fold_and_unfold_share_checked_im2col_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let parameters = SpatialParameters2d::new([2, 2]);
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let columns = unfold_with_context_exact_native(
        &backend,
        &input,
        [1, 1, 3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(columns.shape, [1, 4, 4]);
    close(
        &columns.values,
        &[
            1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0,
        ],
        0.0,
    );
    let folded = fold_with_context_exact_native(
        &backend,
        &columns.values,
        [1, 4, 4],
        [3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    close(
        &folded.values,
        &[1.0, 4.0, 3.0, 8.0, 20.0, 12.0, 7.0, 16.0, 9.0],
        0.0,
    );
    let unfold_vjp = unfold_vjp_with_context_exact_native(
        &backend,
        &[1.0; 16],
        [1, 1, 3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    close(
        &unfold_vjp.values,
        &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        0.0,
    );
    let unfold_jvp = unfold_jvp_with_context_exact_native(
        &backend,
        &input,
        [1, 1, 3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(unfold_jvp, columns);
    let fold_vjp = fold_vjp_with_context_exact_native(
        &backend,
        &[1.0; 9],
        [1, 4, 4],
        [3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(fold_vjp.values.len(), 16);
    let fold_jvp = fold_jvp_with_context_exact_native(
        &backend,
        &columns.values,
        [1, 4, 4],
        [3, 3],
        parameters,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(fold_jvp, folded);
    let unsupported = DeviceId::from_source_device("cuda")?;
    let unsupported_unfold = unfold_vjp_with_context_exact_native(
        &backend,
        &[1.0; 16],
        [1, 1, 3, 3],
        parameters,
        unsupported,
        &execution,
    )
    .expect_err("unfold must reject unsupported devices");
    assert!(
        matches!(
            &unsupported_unfold,
            NeuralNetworkFunctionalError::UnsupportedDevice { operation, .. }
                if *operation == "COMFY-TENSOR-OP-87C10166BCF5"
        ),
        "unexpected unfold error: {unsupported_unfold:?}"
    );
    assert!(matches!(
        fold_vjp_with_context_exact_native(
            &backend,
            &[1.0; 9],
            [1, 4, 4],
            [3, 3],
            parameters,
            unsupported,
            &execution,
        ),
        Err(NeuralNetworkFunctionalError::UnsupportedDevice { operation, .. })
            if operation == "COMFY-TENSOR-OP-3D194029352B"
    ));
    assert!(matches!(
        unfold_vjp_with_context_exact_native(
            &backend,
            &[1.0; 15],
            [1, 1, 3, 3],
            parameters,
            DeviceId::CPU,
            &execution,
        ),
        Err(NeuralNetworkFunctionalError::Invalid { operation, .. })
            if operation == "COMFY-TENSOR-OP-87C10166BCF5"
    ));
    Ok(())
}

#[test]
fn glu_softplus_and_one_hot_cover_values_boundaries_and_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = [2.0, 3.0, 0.0, 0.0];
    let glu =
        glu_with_context_exact_native(&backend, &input, &[1, 4], -1, DeviceId::CPU, &execution)?;
    close(&glu.values, &[1.0, 1.5], 1.0e-6);
    let glu_vjp = glu_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1, 4],
        -1,
        &[1.0, 1.0],
        DeviceId::CPU,
        &execution,
    )?;
    let glu_jvp = glu_jvp_with_context_exact_native(
        &backend,
        &input,
        &[1.0; 4],
        &[1, 4],
        -1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(glu_vjp.len(), 4);
    assert_eq!(glu_jvp.shape, [1, 2]);
    let softplus = softplus_with_context_exact_native(
        &backend,
        &[-2.0, 0.0, 30.0],
        1.0,
        20.0,
        DeviceId::CPU,
        &execution,
    )?;
    close(&softplus, &[0.126928, std::f32::consts::LN_2, 30.0], 1.0e-5);
    let softplus_vjp = softplus_vjp_with_context_exact_native(
        &backend,
        &[0.0],
        &[2.0],
        1.0,
        20.0,
        DeviceId::CPU,
        &execution,
    )?;
    let softplus_jvp = softplus_jvp_with_context_exact_native(
        &backend,
        &[0.0],
        &[2.0],
        1.0,
        20.0,
        DeviceId::CPU,
        &execution,
    )?;
    close(&softplus_vjp, &[1.0], 1.0e-6);
    assert_eq!(softplus_vjp, softplus_jvp);
    let one_hot =
        one_hot_with_context_exact_native(&backend, &[2, 0], &[2], -1, DeviceId::CPU, &execution)?;
    assert_eq!(one_hot.shape, [2, 3]);
    assert_eq!(one_hot.values, [0, 0, 1, 1, 0, 0]);
    Ok(())
}

#[test]
fn linear_facade_preserves_canonical_forward_vjp_and_jvp() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let output = linear_with_context_exact_native(
        &backend,
        &[1.0, 2.0],
        &[1, 2],
        &[3.0, 4.0],
        &[1, 2],
        Some(&[5.0]),
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(output.shape, [1, 1]);
    close(&output.values, &[16.0], 0.0);
    let gradients = linear_vjp_with_context_exact_native(
        &backend,
        &[1.0, 2.0],
        &[1, 2],
        &[3.0, 4.0],
        &[1, 2],
        Some(&[5.0]),
        &[2.0],
        DeviceId::CPU,
        &execution,
    )?;
    close(&gradients.input, &[6.0, 8.0], 0.0);
    close(&gradients.weight, &[2.0, 4.0], 0.0);
    let tangent = linear_jvp_with_context_exact_native(
        &backend,
        &[1.0, 2.0],
        &[1.0, 1.0],
        &[1, 2],
        &[3.0, 4.0],
        &[0.0, 0.0],
        &[1, 2],
        Some(&[5.0]),
        Some(&[1.0]),
        DeviceId::CPU,
        &execution,
    )?;
    close(&tangent.values, &[8.0], 0.0);
    Ok(())
}

#[test]
fn pixel_shuffle_and_unshuffle_use_inverse_canonical_rearrangements()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(
        &backend,
        &[1, 4, 1, 1],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let shuffled =
        pixel_shuffle_tensor_with_context_exact_native(&*backend, &input, 2, &execution)?;
    assert_eq!(shuffled.descriptor().shape(), &[1, 1, 2, 2]);
    close(&f32_values(&shuffled)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let restored =
        pixel_unshuffle_tensor_with_context_exact_native(&*backend, &shuffled, 2, &execution)?;
    close(&f32_values(&restored)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let shuffle_jvp = pixel_shuffle_jvp_with_context_exact_native(&backend, &input, 2, &execution)?;
    let shuffle_vjp = pixel_shuffle_vjp_with_context_exact_native(
        &backend,
        &shuffled,
        &[1, 4, 1, 1],
        2,
        &execution,
    )?;
    close(&f32_values(&shuffle_jvp)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    close(&f32_values(&shuffle_vjp)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let unshuffle_jvp =
        pixel_unshuffle_jvp_with_context_exact_native(&backend, &shuffled, 2, &execution)?;
    let unshuffle_vjp = pixel_unshuffle_vjp_with_context_exact_native(
        &backend,
        &input,
        &[1, 1, 2, 2],
        2,
        &execution,
    )?;
    close(&f32_values(&unshuffle_jvp)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    close(&f32_values(&unshuffle_vjp)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let half_bytes = [1.0_f64, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|value| DType::F16.encode_scalar(Scalar::Float(value), "pixel-shuffle-test", DeviceId::CPU))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let half_descriptor = TensorDescriptor::contiguous(
        vec![1, 4, 1, 1],
        DType::F16,
        DeviceId::CPU,
        execution.stream,
    )?;
    let half = backend.upload_bytes(half_descriptor, &half_bytes, &execution)?.0;
    let shuffled_half =
        pixel_shuffle_tensor_with_context_exact_native(&*backend, &half, 2, &execution)?;
    assert_eq!(shuffled_half.descriptor().dtype(), DType::F16);
    assert_eq!(shuffled_half.descriptor().device(), DeviceId::CPU);
    assert_eq!(shuffled_half.descriptor().stream(), execution.stream);
    assert_eq!(shuffled_half.contiguous_bytes()?, half_bytes);
    Ok(())
}

#[test]
fn sigmoid_and_attention_facades_preserve_canonical_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[1], &[0.0], &cancellation)?;
    let gradient = upload_f32(&backend, &[1], &[2.0], &cancellation)?;
    close(
        &f32_values(&sigmoid_with_context_exact_native(
            &backend, &input, &execution,
        )?)?,
        &[0.5],
        1.0e-6,
    );
    let sigmoid_vjp =
        sigmoid_vjp_with_context_exact_native(&backend, &input, &gradient, &execution)?;
    let sigmoid_jvp =
        sigmoid_jvp_with_context_exact_native(&backend, &input, &gradient, &execution)?;
    close(&f32_values(&sigmoid_vjp)?, &[0.5], 1.0e-6);
    assert_eq!(f32_values(&sigmoid_vjp)?, f32_values(&sigmoid_jvp)?);
    let request = AttentionKernelRequest {
        kind: AttentionKernelKind::ReferenceSdp,
        device: DeviceId::CPU,
        layout: AttentionLayout::Nhd,
        shape: AttentionShape {
            batch: 1,
            query_tokens: 1,
            key_tokens: 1,
            heads: 1,
            head_dimension: 1,
            value_dimension: 1,
        },
        scale: None,
        causal: false,
        dropout_probability: 0.0,
    };
    close(
        &scaled_dot_product_attention_with_context_exact_native(
            &backend,
            request,
            &[2.0],
            &[3.0],
            &[4.0],
            None,
            &execution,
        )?,
        &[4.0],
        0.0,
    );
    let attention_vjp = scaled_dot_product_attention_vjp_with_context_exact_native(
        &backend,
        request,
        &[2.0],
        &[3.0],
        &[4.0],
        None,
        &[2.0],
        &execution,
    )?;
    close(&attention_vjp.value, &[2.0], 0.0);
    close(
        &scaled_dot_product_attention_jvp_with_context_exact_native(
            &backend,
            request,
            &[2.0],
            &[3.0],
            &[4.0],
            None,
            &[0.0],
            &[0.0],
            &[2.0],
            &execution,
        )?,
        &[2.0],
        0.0,
    );
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_input_and_resolution_evidence_is_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let live = CancellationToken::default();
    let tensor = upload_f32(&backend, &[1], &[1.0], &live)?;
    let indices = upload_i64(&backend, &[1], &[0], &live)?;
    let original_tensor = tensor.contiguous_bytes()?.to_vec();
    let original_indices = indices.contiguous_bytes()?.to_vec();
    let mut weight = tensor.clone();
    let original_weight = weight.contiguous_bytes()?.to_vec();

    let cancellation = CancellationToken::default();
    assert!(cancellation.cancel());
    let execution = backend.execution(&cancellation)?;
    let spatial = SpatialParameters2d::new([0, 0]);

    assert_cancelled(
        cosine_similarity_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            0,
            0.0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        cosine_similarity_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            0,
            0.0,
            &[],
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        cosine_similarity_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            0,
            0.0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        embedding_with_context_exact_native(
            &backend,
            &indices,
            &mut weight,
            EmbeddingOptions::default(),
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        embedding_vjp_with_context_exact_native(
            &backend,
            &indices,
            &tensor,
            EmbeddingOptions::default(),
            &tensor,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        embedding_jvp_with_context_exact_native(
            &backend,
            &indices,
            &tensor,
            &tensor,
            EmbeddingOptions::default(),
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        unfold_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        unfold_vjp_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        unfold_jvp_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        fold_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0],
            [0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        fold_vjp_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0],
            [0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        fold_jvp_with_context_exact_native(
            &backend,
            &[],
            [0, 0, 0],
            [0, 0],
            spatial,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        glu_with_context_exact_native(&backend, &[], &[], 0, DeviceId::CPU, &execution),
        &execution,
    );
    assert_cancelled(
        glu_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            0,
            &[],
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        glu_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        linear_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        linear_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            None,
            &[],
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        linear_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            None,
            None,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        one_hot_with_context_exact_native(
            &backend,
            &[],
            &[],
            0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        pixel_shuffle_with_context_exact_native(&backend, &tensor, 0, &execution),
        &execution,
    );
    assert_cancelled(
        pixel_shuffle_vjp_with_context_exact_native(&backend, &tensor, &[], 0, &execution),
        &execution,
    );
    assert_cancelled(
        pixel_shuffle_jvp_with_context_exact_native(&backend, &tensor, 0, &execution),
        &execution,
    );
    assert_cancelled(
        pixel_unshuffle_with_context_exact_native(&backend, &tensor, 0, &execution),
        &execution,
    );
    assert_cancelled(
        pixel_unshuffle_vjp_with_context_exact_native(&backend, &tensor, &[], 0, &execution),
        &execution,
    );
    assert_cancelled(
        pixel_unshuffle_jvp_with_context_exact_native(&backend, &tensor, 0, &execution),
        &execution,
    );

    let request = AttentionKernelRequest {
        kind: AttentionKernelKind::ReferenceSdp,
        device: DeviceId::CPU,
        layout: AttentionLayout::Nhd,
        shape: AttentionShape {
            batch: 0,
            query_tokens: 0,
            key_tokens: 0,
            heads: 0,
            head_dimension: 0,
            value_dimension: 0,
        },
        scale: None,
        causal: false,
        dropout_probability: 0.0,
    };
    assert_cancelled(
        scaled_dot_product_attention_with_context_exact_native(
            &backend,
            request,
            &[],
            &[],
            &[],
            None,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        scaled_dot_product_attention_vjp_with_context_exact_native(
            &backend,
            request,
            &[],
            &[],
            &[],
            None,
            &[],
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        scaled_dot_product_attention_jvp_with_context_exact_native(
            &backend,
            request,
            &[],
            &[],
            &[],
            None,
            &[],
            &[],
            &[],
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        sigmoid_with_context_exact_native(&backend, &tensor, &execution),
        &execution,
    );
    assert_cancelled(
        sigmoid_vjp_with_context_exact_native(&backend, &tensor, &tensor, &execution),
        &execution,
    );
    assert_cancelled(
        sigmoid_jvp_with_context_exact_native(&backend, &tensor, &tensor, &execution),
        &execution,
    );
    assert_cancelled(
        softplus_with_context_exact_native(
            &backend,
            &[],
            0.0,
            0.0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        softplus_vjp_with_context_exact_native(
            &backend,
            &[],
            &[],
            0.0,
            0.0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        softplus_jvp_with_context_exact_native(
            &backend,
            &[],
            &[],
            0.0,
            0.0,
            DeviceId::CPU,
            &execution,
        ),
        &execution,
    );

    assert_eq!(weight.contiguous_bytes()?, original_weight.as_slice());
    assert_eq!(tensor.contiguous_bytes()?, original_tensor.as_slice());
    assert_eq!(indices.contiguous_bytes()?, original_indices.as_slice());

    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "neural_network_functional_01")
        .ok_or("neural-network functional resolution slice is missing")?;
    assert_eq!(slice.len(), 12);
    let mut ids = BTreeSet::new();
    for contract in slice.contracts {
        assert!(ids.insert(contract.operation_id));
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-neural-network-functional-comfy-tensor-op-13df18f5f426"
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
