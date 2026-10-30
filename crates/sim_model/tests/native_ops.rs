use sim_model::{
    ConvolutionAutopad, EmbeddingOptions, GeluApproximation, LayerQuantizationV1, LossReduction,
    NativeModule, NativeModuleSpec, NativeOpsError, QuantizationKind, UpsampleMode,
    adaptive_average_pool_2d_module_exact_native, average_pool_1d_module_exact_native,
    average_pool_2d_module_exact_native, average_pool_3d_module_exact_native,
    batch_norm_1d_module_exact_native, batch_norm_2d_module_exact_native,
    buffer_module_exact_native, cast_modules_with_vbar_with_context_exact_native,
    conv_2d_module_exact_native, conv_3d_module_exact_native, conv1d_module_exact_native,
    disable_weight_init_convolution_exact_native, disable_weight_init_linear_exact_native,
    dropout_module_exact_native, elu_module_exact_native, embedding_module_exact_native,
    gelu_module_exact_native, group_norm_module_exact_native, huber_loss_module_exact_native,
    identity_module_exact_native, instance_norm_2d_module_exact_native,
    l1_loss_module_exact_native, layer_norm_module_exact_native, leaky_relu_module_exact_native,
    linear_module_exact_native, manual_cast_layer_norm_exact_native,
    max_pool_2d_module_exact_native, mixed_precision_ops_exact_native, module_dict_exact_native,
    module_exact_native, module_list_exact_native, mse_loss_module_exact_native,
    multihead_attention_module_exact_native, pixel_shuffle_module_exact_native,
    pixel_unshuffle_module_exact_native, prelu_module_exact_native, quantize_matrix,
    relu_6_module_exact_native, relu_module_exact_native, replication_pad_2d_module_exact_native,
    sequential_module_exact_native, sigmoid_module_exact_native, silu_module_exact_native,
    smooth_l1_loss_module_exact_native, softmax_module_exact_native, tanh_module_exact_native,
    upsample_module_exact_native, zero_pad_2d_module_exact_native,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, tensor_to_f32_with_context_exact_native,
    },
    rng::{RetryRngPolicy, RngAlgorithm, RngProfileVersion, RngStream, RngStreamAddress},
};
use sim_types::DeviceKind;
use std::collections::{BTreeMap, BTreeSet};

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
}

impl std::ops::Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn backend() -> Result<TestBackend, Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(16 * 1024 * 1024)?,
        cancellation,
    ))
}

fn tensor(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = context(backend, cancellation)?;
    Ok(backend.upload_f32(descriptor, values, &context)?.0)
}

fn tensor_with_dtype(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    Ok(sim_tensor::generated_sim_operator_indirection_01::tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        DeviceId::CPU,
        &context(backend, cancellation)?,
    )?)
}

fn integer_tensor(
    backend: &TestBackend,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(descriptor, &bytes, &context(backend, cancellation)?)?
        .0)
}

fn tensor_values(
    backend: &TestBackend,
    tensor: &sim_tensor::Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &context(backend, cancellation)?,
    )?)
}

#[test]
fn neural_network_module_part_one_uses_the_canonical_lifecycle_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(
        &backend,
        &[1, 1, 2, 2],
        &[0.0, 1.0, 2.0, 3.0],
        &cancellation,
    )?;

    let buffer = buffer_module_exact_native("state.running", input.clone(), &cancellation)?;
    assert!(matches!(buffer.spec(), NativeModuleSpec::Buffer));
    assert_eq!(
        buffer
            .registered_buffer()
            .ok_or("registered buffer missing")?
            .contiguous_bytes()?,
        input.contiguous_bytes()?
    );

    let tanh = tanh_module_exact_native("activation.tanh", &cancellation)?;
    let silu = silu_module_exact_native("activation.silu", false, &cancellation)?;
    let mut sequential =
        sequential_module_exact_native("activation.sequence", vec![tanh, silu], &cancellation)?;
    assert_eq!(sequential.children().len(), 2);
    let output =
        sequential.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(output.descriptor(), input.descriptor());
    assert_eq!(sequential.generation(), 1);

    let mut average = average_pool_2d_module_exact_native("pool", [2, 2], None, &cancellation)?;
    let pooled =
        average.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(pooled.descriptor().shape(), &[1, 1, 1, 1]);
    assert_eq!(tensor_values(&backend, &pooled, &cancellation)?, [1.5]);

    let mut prelu = prelu_module_exact_native("prelu", 1, &cancellation)?;
    prelu.load_dense_parameters(tensor(&backend, &[1], &[0.25], &cancellation)?, None)?;
    let signed = tensor(&backend, &[1, 1, 1, 2], &[-4.0, 2.0], &cancellation)?;
    let prelu_output =
        prelu.forward_with_context(&backend, &signed, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_values(&backend, &prelu_output, &cancellation)?,
        [-1.0, 2.0]
    );

    let mut softmax = softmax_module_exact_native("softmax", -1, &cancellation)?;
    let normalized =
        softmax.forward_with_context(&backend, &signed, &context(&backend, &cancellation)?)?;
    let normalized_values = tensor_values(&backend, &normalized, &cancellation)?;
    assert!((normalized_values.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);

    let mut upsample = upsample_module_exact_native(
        "upsample",
        [1.5, 1.5],
        UpsampleMode::Bilinear,
        Some(true),
        &cancellation,
    )?;
    let enlarged =
        upsample.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(enlarged.descriptor().shape(), &[1, 1, 3, 3]);

    let mut loss =
        smooth_l1_loss_module_exact_native("loss", 1.0, LossReduction::Mean, &cancellation)?;
    let target = tensor(
        &backend,
        &[1, 1, 2, 2],
        &[1.0, 0.0, 2.0, 1.0],
        &cancellation,
    )?;
    let loss_output = loss.forward_loss_with_context(
        &backend,
        &input,
        &target,
        &context(&backend, &cancellation)?,
    )?;
    assert!(loss_output.descriptor().shape().is_empty());
    assert_eq!(
        tensor_values(&backend, &loss_output, &cancellation)?,
        [0.625]
    );

    let convolution = conv1d_module_exact_native("conv", 1, 2, 3, 1, 1, 1, 1, true, &cancellation)?;
    assert!(matches!(
        convolution.spec(),
        NativeModuleSpec::Convolution { .. }
    ));
    assert!(matches!(
        group_norm_module_exact_native("group", 2, 4, 1.0e-5, true, &cancellation)?.spec(),
        NativeModuleSpec::GroupNorm { .. }
    ));
    assert!(matches!(
        layer_norm_module_exact_native("layer", vec![4], 1.0e-5, true, true, &cancellation,)?
            .spec(),
        NativeModuleSpec::LayerNorm { .. }
    ));
    Ok(())
}

#[test]
fn cancelled_module_construction_and_sequential_execution_publish_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[1], &[1.0], &cancellation)?;
    let mut sequential = sequential_module_exact_native(
        "sequence",
        vec![tanh_module_exact_native("tanh", &cancellation)?],
        &cancellation,
    )?;
    let generation = sequential.generation();
    let child_generation = sequential.children()[0].generation();

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        sequential.forward_with_context(&backend, &input, &context(&backend, &cancelled)?,),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(sequential.generation(), generation);
    assert_eq!(sequential.children()[0].generation(), child_generation);
    assert!(matches!(
        average_pool_2d_module_exact_native("pool", [0, 0], None, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        buffer_module_exact_native("buffer", input, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        conv1d_module_exact_native("conv", 0, 0, 0, 0, 0, 0, 0, false, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        group_norm_module_exact_native("group", 0, 0, -1.0, false, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        layer_norm_module_exact_native("layer", vec![], -1.0, false, true, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        prelu_module_exact_native("prelu", 0, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        sequential_module_exact_native("sequence", vec![], &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        silu_module_exact_native("silu", true, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        smooth_l1_loss_module_exact_native("loss", -1.0, LossReduction::Mean, &cancelled,),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        softmax_module_exact_native("softmax", 99, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        tanh_module_exact_native("tanh", &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        upsample_module_exact_native(
            "upsample",
            [0.0, 0.0],
            UpsampleMode::Nearest,
            Some(true),
            &cancelled,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn unloaded_modules_commit_checked_parameters_atomically_and_execute()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[2, 2], &[1.0, -2.0, 0.5, 3.0], &cancellation)?;
    let mut module = disable_weight_init_linear_exact_native("encoder.proj", 2, 3, true)?;
    assert!(matches!(
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?),
        Err(NativeOpsError::ParametersNotLoaded)
    ));
    let wrong_weight = tensor(&backend, &[2, 2], &[1.0; 4], &cancellation)?;
    let bias = tensor(&backend, &[3], &[0.25, -0.75, 1.0], &cancellation)?;
    assert!(matches!(
        module.load_dense_parameters(wrong_weight, Some(bias.clone())),
        Err(NativeOpsError::Invalid(_))
    ));
    assert!(matches!(
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?),
        Err(NativeOpsError::ParametersNotLoaded)
    ));
    let weight = tensor(
        &backend,
        &[3, 2],
        &[0.5, 1.0, -1.0, 2.0, 0.25, -0.5],
        &cancellation,
    )?;
    module.load_dense_parameters(weight, Some(bias))?;
    let output =
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?
        )?,
        vec![-1.25, -5.75, 2.25, 3.5, 4.75, -0.375]
    );
    Ok(())
}

#[test]
fn immutable_dense_inference_preserves_module_state_and_tensor_placement()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[2, 2], &[1.0, -2.0, 0.5, 3.0], &cancellation)?;

    let mut linear = disable_weight_init_linear_exact_native("linear", 2, 3, true)?;
    linear.load_dense_parameters(
        tensor(
            &backend,
            &[3, 2],
            &[0.5, 1.0, -1.0, 2.0, 0.25, -0.5],
            &cancellation,
        )?,
        Some(tensor(&backend, &[3], &[0.25, -0.75, 1.0], &cancellation)?),
    )?;

    let geometry =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?;
    let mut convolution =
        disable_weight_init_convolution_exact_native("conv2d", 1, 1, vec![2, 2], true, geometry)?;
    convolution.load_dense_parameters(
        tensor(
            &backend,
            &[1, 1, 2, 2],
            &[1.0, 0.0, 0.0, -1.0],
            &cancellation,
        )?,
        Some(tensor(&backend, &[1], &[0.5], &cancellation)?),
    )?;
    let convolution_input = tensor(
        &backend,
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &cancellation,
    )?;

    let transposed_geometry =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, true, vec![0, 0])?;
    let mut transposed = disable_weight_init_convolution_exact_native(
        "conv_transpose",
        1,
        1,
        vec![2, 2],
        false,
        transposed_geometry,
    )?;
    transposed.load_dense_parameters(
        tensor(
            &backend,
            &[1, 1, 2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            &cancellation,
        )?,
        None,
    )?;
    let transposed_input = tensor(&backend, &[1, 1, 1, 1], &[2.0], &cancellation)?;

    let mut layer_norm =
        layer_norm_module_exact_native("layer_norm", vec![2], 1.0e-5, true, true, &cancellation)?;
    layer_norm.load_dense_parameters(
        tensor(&backend, &[2], &[1.25, 0.75], &cancellation)?,
        Some(tensor(&backend, &[2], &[0.5, -0.25], &cancellation)?),
    )?;

    let mut group_norm =
        group_norm_module_exact_native("group_norm", 1, 2, 1.0e-5, true, &cancellation)?;
    group_norm.load_dense_parameters(
        tensor(&backend, &[2], &[1.5, 0.5], &cancellation)?,
        Some(tensor(&backend, &[2], &[0.25, -0.75], &cancellation)?),
    )?;
    let group_input = tensor(
        &backend,
        &[1, 2, 1, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;

    let silu = silu_module_exact_native("silu", false, &cancellation)?;
    let gelu = gelu_module_exact_native("gelu", GeluApproximation::Tanh, &cancellation)?;
    let instance_norm =
        instance_norm_2d_module_exact_native("instance_norm", 1, 1.0e-5, false, &cancellation)?;
    let instance_input = tensor(
        &backend,
        &[1, 1, 2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;

    for (module, module_input) in [
        (&linear, &input),
        (&convolution, &convolution_input),
        (&transposed, &transposed_input),
        (&layer_norm, &input),
        (&group_norm, &group_input),
        (&instance_norm, &instance_input),
        (&silu, &input),
        (&gelu, &input),
    ] {
        let generation = module.generation();
        let prefetched = module.prefetched_dtype_device();
        let digest = module.semantic_state_digest(&cancellation)?;
        let immutable = module.forward_dense_inference_with_context(
            &backend,
            module_input,
            &context(&backend, &cancellation)?,
        )?;
        let mut mutable = module.clone();
        let ordinary = mutable.forward_with_context(
            &backend,
            module_input,
            &context(&backend, &cancellation)?,
        )?;

        let immutable_values = tensor_values(&backend, &immutable, &cancellation)?;
        let ordinary_values = tensor_values(&backend, &ordinary, &cancellation)?;
        assert_eq!(immutable_values.len(), ordinary_values.len());
        for (immutable, ordinary) in immutable_values.iter().zip(&ordinary_values) {
            assert!((immutable - ordinary).abs() <= 1.0e-6);
        }
        assert_eq!(
            immutable.descriptor().dtype(),
            module_input.descriptor().dtype()
        );
        assert_eq!(
            immutable.descriptor().device(),
            module_input.descriptor().device()
        );
        assert_eq!(
            immutable.descriptor().stream(),
            module_input.descriptor().stream()
        );
        assert_eq!(module.generation(), generation);
        assert_eq!(module.prefetched_dtype_device(), prefetched);
        assert_eq!(module.semantic_state_digest(&cancellation)?, digest);
    }

    assert_eq!(
        tensor_values(
            &backend,
            &transposed.forward_dense_inference_with_context(
                &backend,
                &transposed_input,
                &context(&backend, &cancellation)?,
            )?,
            &cancellation,
        )?,
        [2.0, 4.0, 6.0, 8.0]
    );
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let input = tensor_with_dtype(&backend, &[1, 2], &[-1.0, 1.0], dtype, &cancellation)?;
        let output = silu.forward_dense_inference_with_context(
            &backend,
            &input,
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(output.descriptor().dtype(), dtype);
        let values = tensor_values(&backend, &output, &cancellation)?;
        assert!((values[0] - -0.268_941_43).abs() < 0.003);
        assert!((values[1] - 0.731_058_6).abs() < 0.003);
        assert_ne!(output.storage_id(), input.storage_id());

        let leaky = leaky_relu_module_exact_native("typed_leaky", 0.2, false, &cancellation)?;
        let leaky_generation = leaky.generation();
        let leaky_digest = leaky.semantic_state_digest(&cancellation)?;
        let leaky_output = leaky.forward_dense_inference_with_context(
            &backend,
            &input,
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(leaky_output.descriptor().dtype(), dtype);
        let leaky_values = tensor_values(&backend, &leaky_output, &cancellation)?;
        assert!((leaky_values[0] - -0.2).abs() < 0.004);
        assert!((leaky_values[1] - 1.0).abs() < 0.004);
        assert_ne!(leaky_output.storage_id(), input.storage_id());
        assert_eq!(leaky.generation(), leaky_generation);
        assert_eq!(leaky.semantic_state_digest(&cancellation)?, leaky_digest);

        let geometry =
            ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?;
        let mut convolution = disable_weight_init_convolution_exact_native(
            "typed_conv",
            1,
            1,
            vec![2, 2],
            true,
            geometry,
        )?;
        convolution.load_dense_parameters(
            tensor_with_dtype(&backend, &[1, 1, 2, 2], &[1.0; 4], dtype, &cancellation)?,
            Some(tensor_with_dtype(
                &backend,
                &[1],
                &[0.5],
                dtype,
                &cancellation,
            )?),
        )?;
        let convolution_input = tensor_with_dtype(
            &backend,
            &[1, 1, 2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            dtype,
            &cancellation,
        )?;
        let convolution_generation = convolution.generation();
        let convolution_prefetch = convolution.prefetched_dtype_device();
        let convolution_digest = convolution.semantic_state_digest(&cancellation)?;
        let convolution_allocations = convolution.resident_tensor_allocations();
        let convolution_context = context(&backend, &cancellation)?;
        let convolution_output = convolution.forward_dense_inference_with_context(
            &backend,
            &convolution_input,
            &convolution_context,
        )?;
        assert_eq!(convolution_output.descriptor().dtype(), dtype);
        assert!(
            (tensor_values(&backend, &convolution_output, &cancellation)?[0] - 10.5).abs() < 0.01
        );
        assert_ne!(
            convolution_output.storage_id(),
            convolution_input.storage_id()
        );
        assert!(
            convolution_allocations
                .iter()
                .all(|(storage_id, _)| *storage_id != convolution_output.storage_id())
        );
        assert_eq!(convolution_context.scratch.in_use_bytes(), 0);
        assert_eq!(convolution.generation(), convolution_generation);
        assert_eq!(convolution.prefetched_dtype_device(), convolution_prefetch);
        assert_eq!(
            convolution.semantic_state_digest(&cancellation)?,
            convolution_digest
        );

        let transposed_geometry =
            ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, true, vec![0, 0])?;
        let mut transposed = disable_weight_init_convolution_exact_native(
            "typed_transpose",
            1,
            1,
            vec![2, 2],
            false,
            transposed_geometry,
        )?;
        transposed.load_dense_parameters(
            tensor_with_dtype(
                &backend,
                &[1, 1, 2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                dtype,
                &cancellation,
            )?,
            None,
        )?;
        let transposed_input =
            tensor_with_dtype(&backend, &[1, 1, 1, 1], &[2.0], dtype, &cancellation)?;
        let transposed_generation = transposed.generation();
        let transposed_prefetch = transposed.prefetched_dtype_device();
        let transposed_digest = transposed.semantic_state_digest(&cancellation)?;
        let transposed_allocations = transposed.resident_tensor_allocations();
        let transposed_context = context(&backend, &cancellation)?;
        let transposed_output = transposed.forward_dense_inference_with_context(
            &backend,
            &transposed_input,
            &transposed_context,
        )?;
        assert_eq!(transposed_output.descriptor().dtype(), dtype);
        let values = tensor_values(&backend, &transposed_output, &cancellation)?;
        for (actual, expected) in values.iter().zip([2.0, 4.0, 6.0, 8.0]) {
            assert!((actual - expected).abs() < 0.01);
        }
        assert_ne!(
            transposed_output.storage_id(),
            transposed_input.storage_id()
        );
        assert!(
            transposed_allocations
                .iter()
                .all(|(storage_id, _)| *storage_id != transposed_output.storage_id())
        );
        assert_eq!(transposed_context.scratch.in_use_bytes(), 0);
        assert_eq!(transposed.generation(), transposed_generation);
        assert_eq!(transposed.prefetched_dtype_device(), transposed_prefetch);
        assert_eq!(
            transposed.semantic_state_digest(&cancellation)?,
            transposed_digest
        );

        let instance_input = tensor_with_dtype(
            &backend,
            &[1, 1, 2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            dtype,
            &cancellation,
        )?;
        let instance_output = instance_norm.forward_dense_inference_with_context(
            &backend,
            &instance_input,
            &context(&backend, &cancellation)?,
        )?;
        assert_eq!(instance_output.descriptor().dtype(), dtype);
        assert_ne!(instance_output.storage_id(), instance_input.storage_id());
        let values = tensor_values(&backend, &instance_output, &cancellation)?;
        for (actual, expected) in
            values
                .iter()
                .zip([-1.341_635_5, -0.447_211_8, 0.447_211_8, 1.341_635_5])
        {
            assert!((actual - expected).abs() < 0.01);
        }
    }
    Ok(())
}

#[test]
fn immutable_dense_inference_rejects_unsupported_state_and_rolls_back_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[1, 2], &[1.0, -1.0], &cancellation)?;
    let tanh = tanh_module_exact_native("tanh", &cancellation)?;
    assert!(matches!(
        tanh.forward_dense_inference_with_context(
            &backend,
            &input,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeOpsError::Invalid(
            "module does not support immutable dense inference"
        ))
    ));

    let mut linear = disable_weight_init_linear_exact_native("linear", 2, 2, false)?;
    linear.load_dense_parameters(
        tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], &cancellation)?,
        None,
    )?;
    let generation = linear.generation();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&backend, &cancelled)?;
    assert!(matches!(
        linear.forward_dense_inference_with_context(&backend, &input, &cancelled_context),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(linear.generation(), generation);
    assert!(linear.prefetched_dtype_device().is_none());

    let convolution_1d_geometry =
        ConvolutionGeometry::new(1, vec![1], vec![0], vec![1], 1, false, vec![0])?;
    let mut convolution_1d = disable_weight_init_convolution_exact_native(
        "conv1d",
        1,
        1,
        vec![1],
        false,
        convolution_1d_geometry,
    )?;
    convolution_1d
        .load_dense_parameters(tensor(&backend, &[1, 1, 1], &[1.0], &cancellation)?, None)?;
    let convolution_input = tensor(&backend, &[1, 1, 2], &[1.0, 2.0], &cancellation)?;
    assert!(matches!(
        convolution_1d.forward_dense_inference_with_context(
            &backend,
            &convolution_input,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeOpsError::Invalid(
            "immutable dense spatial inference supports only Conv2d modules"
        ))
    ));
    Ok(())
}

#[test]
fn canonical_module_owns_zero_weight_fast_forward_and_bias_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[2, 2], &[9.0, -7.0, 3.0, 4.0], &cancellation)?;
    let mut module = disable_weight_init_linear_exact_native("zero", 2, 3, true)?;
    module.load_dense_parameters(
        tensor(&backend, &[3, 2], &[0.0; 6], &cancellation)?,
        Some(tensor(&backend, &[3], &[0.25, -0.5, 1.5], &cancellation)?),
    )?;
    let fast = module
        .forward_if_dense_weight_is_zero_with_context(
            &backend,
            &input,
            &context(&backend, &cancellation)?,
        )?
        .ok_or("zero dense module did not use its canonical fast path")?;
    let regular =
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(fast.contiguous_bytes()?, regular.contiguous_bytes()?);
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &fast,
            &context(&backend, &cancellation)?
        )?,
        [0.25, -0.5, 1.5, 0.25, -0.5, 1.5]
    );
    for non_finite in [f32::NAN, f32::INFINITY] {
        let non_finite_input = tensor(&backend, &[1, 2], &[non_finite, 1.0], &cancellation)?;
        assert!(
            module
                .forward_if_dense_weight_is_zero_with_context(
                    &backend,
                    &non_finite_input,
                    &context(&backend, &cancellation)?
                )?
                .is_none()
        );
        let regular = module.forward_with_context(
            &backend,
            &non_finite_input,
            &context(&backend, &cancellation)?,
        )?;
        assert!(
            tensor_to_f32_with_context_exact_native(
                &backend,
                &regular,
                &context(&backend, &cancellation)?
            )?
            .iter()
            .all(|value| value.is_nan())
        );
    }
    module.load_dense_parameters(
        tensor(
            &backend,
            &[3, 2],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &cancellation,
        )?,
        Some(tensor(&backend, &[3], &[0.25, -0.5, 1.5], &cancellation)?),
    )?;
    assert!(
        module
            .forward_if_dense_weight_is_zero_with_context(
                &backend,
                &input,
                &context(&backend, &cancellation)?
            )?
            .is_none()
    );
    Ok(())
}

#[test]
fn cast_leases_are_single_completion_and_prefetch_is_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut first = disable_weight_init_linear_exact_native("first", 2, 2, true)?;
    first.load_dense_parameters(
        tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], &cancellation)?,
        Some(tensor(&backend, &[2], &[0.0, 0.0], &cancellation)?),
    )?;
    let mut casted = first.cast_bias_weight_with_context_exact_native(
        &backend,
        None,
        Some(DType::F32),
        Some(DeviceId::CPU),
        Some(DType::F32),
        true,
        None,
        false,
        &context(&backend, &cancellation)?,
    )?;
    assert!(casted.lease().offloadable());
    assert!(!casted.lease().is_completed());
    let cancelled_finish = CancellationToken::default();
    cancelled_finish.cancel();
    assert!(matches!(
        casted.finish(&cancelled_finish),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(!casted.lease().is_completed());
    casted.finish(&cancellation)?;
    assert!(casted.lease().is_completed());
    assert!(matches!(
        casted.finish(&cancellation),
        Err(NativeOpsError::LeaseAlreadyCompleted { .. })
    ));

    let generation_before_unsupported_cast = first.generation();
    assert!(
        first
            .cast_bias_weight_with_context_exact_native(
                &backend,
                None,
                Some(DType::F32),
                Some(DeviceId::new(DeviceKind::Cuda, 0)),
                Some(DType::F32),
                true,
                None,
                false,
                &context(&backend, &cancellation)?,
            )
            .is_err()
    );
    assert_eq!(first.generation(), generation_before_unsupported_cast);

    let mut second = disable_weight_init_linear_exact_native("second", 2, 2, false)?;
    second.load_dense_parameters(
        tensor(&backend, &[2, 2], &[2.0, 0.0, 0.0, 2.0], &cancellation)?,
        None,
    )?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let mut modules = vec![first, second];
    assert!(
        cast_modules_with_vbar_with_context_exact_native(
            &mut modules,
            &backend,
            DType::F16,
            DeviceId::new(DeviceKind::Cuda, 0),
            DType::F16,
            true,
            &context(&backend, &cancellation)?,
        )
        .is_err()
    );
    assert!(
        modules
            .iter()
            .all(|module| module.prefetched_dtype_device().is_none())
    );
    assert!(matches!(
        cast_modules_with_vbar_with_context_exact_native(
            &mut modules,
            &backend,
            DType::F16,
            DeviceId::CPU,
            DType::F16,
            true,
            &context(&backend, &cancelled)?,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(
        modules
            .iter()
            .all(|module| module.prefetched_dtype_device().is_none())
    );
    let receipt = cast_modules_with_vbar_with_context_exact_native(
        &mut modules,
        &backend,
        DType::F16,
        DeviceId::CPU,
        DType::F16,
        true,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(receipt.module_generations.len(), 2);
    assert!(
        modules.iter().all(|module| {
            module.prefetched_dtype_device() == Some((DType::F16, DeviceId::CPU))
        })
    );

    let probe = backend
        .workspace_authority
        .authorize_workspace(1024 * 1024)?;
    let probe_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: probe,
        rng_phase: None,
        cancellation: &cancellation,
    };
    sim_model::native_ops::cast_modules_with_vbar_with_context_exact_native(
        &mut modules,
        &backend,
        DType::F16,
        DeviceId::CPU,
        DType::F16,
        false,
        &probe_context,
    )?;
    let peak = probe_context.scratch.peak_bytes();
    assert!(peak > 0);
    assert_eq!(probe_context.scratch.in_use_bytes(), 0);

    let exact = backend.workspace_authority.authorize_workspace(peak)?;
    let exact_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: exact,
        rng_phase: None,
        cancellation: &cancellation,
    };
    sim_model::native_ops::cast_modules_with_vbar_with_context_exact_native(
        &mut modules,
        &backend,
        DType::F16,
        DeviceId::CPU,
        DType::F16,
        false,
        &exact_context,
    )?;
    assert_eq!(exact_context.scratch.peak_bytes(), peak);
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let generations = modules
        .iter()
        .map(NativeModule::generation)
        .collect::<Vec<_>>();
    let insufficient = backend.workspace_authority.authorize_workspace(peak - 1)?;
    let insufficient_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: insufficient,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(
        sim_model::native_ops::cast_modules_with_vbar_with_context_exact_native(
            &mut modules,
            &backend,
            DType::F16,
            DeviceId::CPU,
            DType::F16,
            false,
            &insufficient_context,
        )
        .is_err()
    );
    assert_eq!(
        modules
            .iter()
            .map(NativeModule::generation)
            .collect::<Vec<_>>(),
        generations
    );
    assert_eq!(insufficient_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn mixed_precision_operation_set_delegates_quantized_values_to_canonical_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut config = BTreeMap::new();
    config.insert(
        "decoder.proj".to_owned(),
        LayerQuantizationV1 {
            algorithm: QuantizationKind::Int8Tensorwise,
            original_dtype: DType::F32,
        },
    );
    let operation_set =
        mixed_precision_ops_exact_native(config, DType::Bf16, false, BTreeSet::new())?;
    assert_eq!(operation_set.compute_dtype(), DType::Bf16);
    assert_eq!(
        operation_set
            .quantization_for_layer("decoder.proj")
            .ok_or("quantization selection is missing")?
            .algorithm,
        QuantizationKind::Int8Tensorwise
    );
    let quantized = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[1.0, 0.0, 0.0, 2.0],
        2,
        2,
        &cancellation,
    )?;
    let mut module = operation_set.linear("decoder.proj", 2, 2, false)?;
    module.load_quantized_linear_parameters(quantized, None)?;
    let input = tensor(&backend, &[1, 2], &[3.0, -1.0], &cancellation)?;
    let mut casted = module.cast_bias_weight_with_context_exact_native(
        &backend,
        Some(&input),
        Some(DType::F32),
        Some(DeviceId::CPU),
        Some(DType::F32),
        true,
        Some(DType::F32),
        true,
        &context(&backend, &cancellation)?,
    )?;
    assert!(casted.requantized_weight.is_some());
    casted.finish(&cancellation)?;
    let output =
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    let values = tensor_to_f32_with_context_exact_native(
        &backend,
        &output,
        &context(&backend, &cancellation)?,
    )?;
    assert!((values[0] - 3.0).abs() < 0.03);
    assert!((values[1] + 2.0).abs() < 0.03);
    Ok(())
}

#[test]
fn convolution_and_manual_layer_norm_modules_are_focused_kernel_adapters()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let geometry =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?;
    let mut convolution = disable_weight_init_convolution_exact_native(
        "decoder.conv",
        1,
        1,
        vec![2, 2],
        false,
        geometry,
    )?;
    convolution.load_dense_parameters(
        tensor(&backend, &[1, 1, 2, 2], &[1.0; 4], &cancellation)?,
        None,
    )?;
    let input = tensor(
        &backend,
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &cancellation,
    )?;
    let output =
        convolution.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?
        )?,
        vec![12.0, 16.0, 24.0, 28.0]
    );

    let mut layer_norm =
        manual_cast_layer_norm_exact_native("decoder.norm", vec![2], 0.00001, true, true)?;
    layer_norm.load_dense_parameters(
        tensor(&backend, &[2], &[1.0, 2.0], &cancellation)?,
        Some(tensor(&backend, &[2], &[0.5, -0.5], &cancellation)?),
    )?;
    let norm_input = tensor(&backend, &[1, 2], &[1.0, 3.0], &cancellation)?;
    let normalized = layer_norm.forward_with_context(
        &backend,
        &norm_input,
        &context(&backend, &cancellation)?,
    )?;
    let values = tensor_to_f32_with_context_exact_native(
        &backend,
        &normalized,
        &context(&backend, &cancellation)?,
    )?;
    assert!((values[0] + 0.499995).abs() < 0.00002);
    assert!((values[1] - 1.49999).abs() < 0.00003);

    let mut non_affine = manual_cast_layer_norm_exact_native(
        "decoder.norm_without_affine",
        vec![2],
        0.00001,
        false,
        false,
    )?;
    let normalized = non_affine.forward_with_context(
        &backend,
        &norm_input,
        &context(&backend, &cancellation)?,
    )?;
    let values = tensor_to_f32_with_context_exact_native(
        &backend,
        &normalized,
        &context(&backend, &cancellation)?,
    )?;
    assert!((values[0] + 0.999995).abs() < 0.00002);
    assert!((values[1] - 0.999995).abs() < 0.00002);

    let causal_geometry = ConvolutionGeometry::new(
        3,
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        1,
        false,
        vec![0, 0, 0],
    )?;
    let mut causal = disable_weight_init_convolution_exact_native(
        "decoder.causal",
        1,
        1,
        vec![3, 1, 1],
        false,
        causal_geometry,
    )?;
    causal.load_dense_parameters(
        tensor(&backend, &[1, 1, 3, 1, 1], &[1.0, 2.0, 3.0], &cancellation)?,
        None,
    )?;
    let causal_input = tensor(&backend, &[1, 1, 1, 1, 1], &[2.0], &cancellation)?;
    let causal_output = causal.forward_with_autopad_with_context(
        &backend,
        &causal_input,
        ConvolutionAutopad::CausalZero,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &causal_output,
            &context(&backend, &cancellation)?
        )?,
        vec![6.0]
    );
    Ok(())
}

#[test]
fn constructors_reject_invalid_shapes_without_creating_parallel_module_state() {
    assert!(NativeModule::linear("", 2, 2, false, false).is_err());
    assert!(NativeModule::linear("valid", 0, 2, false, false).is_err());
    assert!(
        ConvolutionGeometry::new(2, vec![0, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0],)
            .is_err()
    );
    assert!(
        mixed_precision_ops_exact_native(BTreeMap::new(), DType::I8, false, BTreeSet::new(),)
            .is_err()
    );
    assert!(
        manual_cast_layer_norm_exact_native("invalid.norm", vec![2], 1.0e-5, false, true).is_err()
    );
    let geometry =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])
            .unwrap_or_else(|error| panic!("valid convolution geometry failed: {error}"));
    assert!(
        disable_weight_init_convolution_exact_native(
            "invalid.conv",
            0,
            1,
            vec![1, 1],
            false,
            geometry,
        )
        .is_err()
    );
}

#[test]
fn module_context_forward_uses_exact_caller_workspace_and_is_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[1, 2], &[3.0, -1.0], &cancellation)?;
    let mut module = disable_weight_init_linear_exact_native("context.linear", 2, 2, false)?;
    module.load_dense_parameters(
        tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 2.0], &cancellation)?,
        None,
    )?;

    let probe = backend
        .workspace_authority
        .authorize_workspace(1024 * 1024)?;
    let probe_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: probe,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let output = module.forward_with_context(&backend, &input, &probe_context)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?
        )?,
        [3.0, -2.0]
    );
    let peak = probe_context.scratch.peak_bytes();
    assert!(peak > 0);
    assert_eq!(probe_context.scratch.in_use_bytes(), 0);

    let exact = backend.workspace_authority.authorize_workspace(peak)?;
    let exact_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: exact,
        rng_phase: None,
        cancellation: &cancellation,
    };
    module.forward_with_context(&backend, &input, &exact_context)?;
    assert_eq!(exact_context.scratch.peak_bytes(), peak);
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let generation = module.generation();
    let insufficient = backend.workspace_authority.authorize_workspace(peak - 1)?;
    let insufficient_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: insufficient,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(
        module
            .forward_with_context(&backend, &input, &insufficient_context)
            .is_err()
    );
    assert_eq!(module.generation(), generation);
    assert_eq!(insufficient_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_scratch = backend.workspace_authority.authorize_workspace(peak)?;
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: cancelled_scratch,
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(
        module
            .forward_with_context(&backend, &input, &cancelled_context)
            .is_err()
    );
    assert_eq!(module.generation(), generation);
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn neural_network_module_part_two_uses_one_native_lifecycle()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;

    let image = tensor(
        &backend,
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &cancellation,
    )?;
    let mut adaptive =
        adaptive_average_pool_2d_module_exact_native("adaptive", [2, 2], &cancellation)?;
    let output = adaptive.forward_with_context(&backend, &image, &execution)?;
    assert_eq!(output.descriptor().shape(), &[1, 1, 2, 2]);
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [3.0, 4.0, 6.0, 7.0]
    );

    let cube = tensor(
        &backend,
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let mut average =
        average_pool_3d_module_exact_native("average", [2, 2, 2], None, &cancellation)?;
    let output = average.forward_with_context(&backend, &cube, &execution)?;
    assert_eq!(tensor_values(&backend, &output, &cancellation)?, [4.5]);

    let signed = tensor(&backend, &[2], &[-2.0, 3.0], &cancellation)?;
    let mut activation = leaky_relu_module_exact_native("leaky", 0.1, false, &cancellation)?;
    let output = activation.forward_with_context(&backend, &signed, &execution)?;
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [-0.2, 3.0]
    );

    let line = tensor(&backend, &[1, 1, 2], &[1.0, 2.0], &cancellation)?;
    let mut padding =
        replication_pad_2d_module_exact_native("padding", [1, 1, 1, 0], &cancellation)?;
    let output = padding.forward_with_context(&backend, &line, &execution)?;
    assert_eq!(output.descriptor().shape(), &[1, 2, 4]);
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    );

    let target = tensor(&backend, &[2], &[0.0, 0.0], &cancellation)?;
    let input = tensor(&backend, &[2], &[1.0, 3.0], &cancellation)?;
    let mut loss =
        huber_loss_module_exact_native("huber", 2.0, LossReduction::Mean, &cancellation)?;
    let output = loss.forward_loss_with_context(&backend, &input, &target, &execution)?;
    assert_eq!(tensor_values(&backend, &output, &cancellation)?, [2.25]);
    Ok(())
}

#[test]
fn part_two_stateful_and_parameterized_modules_are_atomic() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;

    let input = tensor(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let mut batch =
        batch_norm_1d_module_exact_native("batch", 2, 1.0e-5, 0.1, false, true, &cancellation)?;
    let before = batch
        .running_statistics()
        .ok_or("batch statistics missing")?;
    assert_eq!(before.0, [0.0, 0.0]);
    batch.forward_with_context(&backend, &input, &execution)?;
    let after = batch
        .running_statistics()
        .ok_or("batch statistics missing")?;
    assert!(after.0.iter().any(|value| *value != 0.0));
    let stable_mean = after.0.to_vec();
    let stable_variance = after.1.to_vec();
    let generation = batch.generation();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        batch.forward_with_context(&backend, &input, &context(&backend, &cancelled)?),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(batch.generation(), generation);
    assert_eq!(
        batch.running_statistics(),
        Some((stable_mean.as_slice(), stable_variance.as_slice()))
    );

    let mut batch_2d =
        batch_norm_2d_module_exact_native("batch2d", 2, 1.0e-5, 0.1, false, false, &cancellation)?;
    let image = tensor(
        &backend,
        &[1, 2, 1, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    batch_2d.forward_with_context(&backend, &image, &execution)?;
    assert!(batch_2d.running_statistics().is_none());

    let mut instance =
        instance_norm_2d_module_exact_native("instance", 2, 1.0e-5, false, &cancellation)?;
    let normalized = instance.forward_with_context(&backend, &image, &execution)?;
    let normalized = tensor_values(&backend, &normalized, &cancellation)?;
    assert!((normalized[0] + 1.0).abs() < 1.0e-4);
    assert!((normalized[1] - 1.0).abs() < 1.0e-4);

    let mut convolution = conv_2d_module_exact_native(
        "conv2d",
        1,
        1,
        [2, 2],
        [1, 1],
        [0, 0],
        [1, 1],
        1,
        true,
        &cancellation,
    )?;
    convolution.load_dense_parameters(
        tensor(&backend, &[1, 1, 2, 2], &[1.0; 4], &cancellation)?,
        Some(tensor(&backend, &[1], &[0.5], &cancellation)?),
    )?;
    let convolution_input = tensor(
        &backend,
        &[1, 1, 2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let output = convolution.forward_with_context(&backend, &convolution_input, &execution)?;
    assert_eq!(tensor_values(&backend, &output, &cancellation)?, [10.5]);

    let mut linear = linear_module_exact_native("linear", 2, 2, true, &cancellation)?;
    linear.load_dense_parameters(
        tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 2.0], &cancellation)?,
        Some(tensor(&backend, &[2], &[0.5, -1.0], &cancellation)?),
    )?;
    let output = linear.forward_with_context(
        &backend,
        &tensor(&backend, &[1, 2], &[1.0, 2.0], &cancellation)?,
        &execution,
    )?;
    assert_eq!(tensor_values(&backend, &output, &cancellation)?, [1.5, 3.0]);

    let mut embedding = embedding_module_exact_native(
        "embedding",
        3,
        2,
        EmbeddingOptions::default(),
        &cancellation,
    )?;
    embedding.load_dense_parameters(
        tensor(
            &backend,
            &[3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &cancellation,
        )?,
        None,
    )?;
    let indices = integer_tensor(&backend, &[2], &[0, 2], &cancellation)?;
    let output = embedding.forward_with_context(&backend, &indices, &execution)?;
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [1.0, 2.0, 5.0, 6.0]
    );
    Ok(())
}

#[test]
fn multihead_attention_commits_all_projection_children_together()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;
    let mut attention =
        multihead_attention_module_exact_native("attention", 2, 1, false, &cancellation)?;
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        attention
            .child_mut(name)
            .ok_or("projection child missing")?
            .load_dense_parameters(
                tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], &cancellation)?,
                None,
            )?;
    }
    let query = tensor(&backend, &[1, 1, 2], &[1.0, 0.0], &cancellation)?;
    let key = tensor(&backend, &[2, 1, 2], &[1.0, 0.0, 0.0, 1.0], &cancellation)?;
    let value = tensor(&backend, &[2, 1, 2], &[2.0, 0.0, 0.0, 4.0], &cancellation)?;
    let output =
        attention.forward_attention_with_context(&backend, &query, &key, &value, &execution)?;
    let values = tensor_values(&backend, &output, &cancellation)?;
    assert!((values[0] - 1.3395231).abs() < 1.0e-5);
    assert!((values[1] - 1.3209538).abs() < 1.0e-5);
    assert_eq!(attention.generation(), 1);
    assert!(
        attention
            .children()
            .iter()
            .all(|child| child.generation() == 2)
    );

    let parent_generation = attention.generation();
    let child_generations = attention
        .children()
        .iter()
        .map(NativeModule::generation)
        .collect::<Vec<_>>();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        attention.forward_attention_with_context(
            &backend,
            &query,
            &key,
            &value,
            &context(&backend, &cancelled)?,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(attention.generation(), parent_generation);
    assert_eq!(
        attention
            .children()
            .iter()
            .map(NativeModule::generation)
            .collect::<Vec<_>>(),
        child_generations
    );
    Ok(())
}

#[test]
fn cancelled_part_two_constructors_validate_cancellation_first()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        adaptive_average_pool_2d_module_exact_native("adaptive", [0, 0], &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        average_pool_3d_module_exact_native("average", [0; 3], None, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        batch_norm_1d_module_exact_native("batch", 0, -1.0, -1.0, false, false, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        batch_norm_2d_module_exact_native("batch", 0, -1.0, -1.0, false, false, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        conv_2d_module_exact_native(
            "conv",
            0,
            0,
            [0; 2],
            [0; 2],
            [0; 2],
            [0; 2],
            0,
            false,
            &cancellation,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        embedding_module_exact_native(
            "embedding",
            0,
            0,
            EmbeddingOptions::default(),
            &cancellation,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        huber_loss_module_exact_native("huber", -1.0, LossReduction::Mean, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        instance_norm_2d_module_exact_native("instance", 0, -1.0, false, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        leaky_relu_module_exact_native("leaky", f32::NAN, true, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        linear_module_exact_native("linear", 0, 0, false, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        multihead_attention_module_exact_native("attention", 0, 0, false, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        replication_pad_2d_module_exact_native("padding", [usize::MAX; 4], &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn neural_network_module_part_three_extends_the_canonical_lifecycle()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;

    let mut convolution = conv_3d_module_exact_native(
        "conv3d",
        1,
        1,
        [2; 3],
        [1; 3],
        [0; 3],
        [1; 3],
        1,
        true,
        &cancellation,
    )?;
    convolution.load_dense_parameters(
        tensor(&backend, &[1, 1, 2, 2, 2], &[1.0; 8], &cancellation)?,
        Some(tensor(&backend, &[1], &[1.0], &cancellation)?),
    )?;
    let cube = tensor(
        &backend,
        &[1, 1, 2, 2, 2],
        &(1..=8).map(|value| value as f32).collect::<Vec<_>>(),
        &cancellation,
    )?;
    let convolved = convolution.forward_with_context(&backend, &cube, &execution)?;
    assert_eq!(tensor_values(&backend, &convolved, &cancellation)?, [37.0]);

    let input = tensor(
        &backend,
        &[1, 4, 1, 1],
        &[-1.0, 0.0, 2.0, 8.0],
        &cancellation,
    )?;
    let mut relu = relu_module_exact_native("relu", false, &cancellation)?;
    let output = relu.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [0.0, 0.0, 2.0, 8.0]
    );
    let mut relu_6 = relu_6_module_exact_native("relu6", false, &cancellation)?;
    let output = relu_6.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(
        tensor_values(&backend, &output, &cancellation)?,
        [0.0, 0.0, 2.0, 6.0]
    );
    let mut gelu = gelu_module_exact_native("gelu", GeluApproximation::None, &cancellation)?;
    let output = gelu.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(output.descriptor().shape(), input.descriptor().shape());

    let mut shuffle = pixel_shuffle_module_exact_native("shuffle", 2, &cancellation)?;
    let shuffled = shuffle.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(shuffled.descriptor().shape(), &[1, 1, 2, 2]);
    let mut unshuffle = pixel_unshuffle_module_exact_native("unshuffle", 2, &cancellation)?;
    let restored = unshuffle.forward_with_context(&backend, &shuffled, &execution)?;
    assert_eq!(restored.descriptor().shape(), input.descriptor().shape());
    assert_eq!(
        tensor_values(&backend, &restored, &cancellation)?,
        [-1.0, 0.0, 2.0, 8.0]
    );

    let image = tensor(
        &backend,
        &[1, 1, 3, 3],
        &(1..=9).map(|value| value as f32).collect::<Vec<_>>(),
        &cancellation,
    )?;
    let mut pool = max_pool_2d_module_exact_native(
        "pool",
        [2, 2],
        Some([2, 2]),
        [0, 0],
        [1, 1],
        true,
        &cancellation,
    )?;
    let pooled = pool.forward_with_context(&backend, &image, &execution)?;
    assert_eq!(
        tensor_values(&backend, &pooled, &cancellation)?,
        [5.0, 6.0, 8.0, 9.0]
    );
    let mut padding = zero_pad_2d_module_exact_native("pad", [1, 0, 1, 0], &cancellation)?;
    let padded = padding.forward_with_context(&backend, &pooled, &execution)?;
    assert_eq!(padded.descriptor().shape(), &[1, 1, 3, 3]);

    let target = tensor(&backend, &[2], &[1.0, 0.0], &cancellation)?;
    let loss_input = tensor(&backend, &[2], &[-1.0, 2.0], &cancellation)?;
    let mut loss = l1_loss_module_exact_native("loss", LossReduction::Mean, &cancellation)?;
    let loss_output = loss.forward_loss_with_context(&backend, &loss_input, &target, &execution)?;
    assert_eq!(tensor_values(&backend, &loss_output, &cancellation)?, [2.0]);

    let list = module_list_exact_native(
        "list",
        vec![
            relu_module_exact_native("same", false, &cancellation)?,
            relu_module_exact_native("same", false, &cancellation)?,
        ],
        &cancellation,
    )?;
    assert!(matches!(list.spec(), NativeModuleSpec::ModuleList));
    assert_eq!(
        list.child_at(1)
            .ok_or("module-list child missing")?
            .layer_name(),
        "same"
    );
    let dictionary = module_dict_exact_native(
        "dict",
        vec![
            relu_module_exact_native("first", false, &cancellation)?,
            relu_6_module_exact_native("second", false, &cancellation)?,
        ],
        &cancellation,
    )?;
    assert!(matches!(dictionary.spec(), NativeModuleSpec::ModuleDict));
    assert_eq!(dictionary.children().len(), 2);
    assert!(
        module_dict_exact_native(
            "invalid",
            vec![
                relu_module_exact_native("duplicate", false, &cancellation)?,
                relu_module_exact_native("duplicate", false, &cancellation)?,
            ],
            &cancellation,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn cancelled_part_three_constructors_validate_cancellation_first()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        conv_3d_module_exact_native(
            "conv",
            0,
            0,
            [0; 3],
            [0; 3],
            [0; 3],
            [0; 3],
            0,
            false,
            &cancellation,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        gelu_module_exact_native("gelu", GeluApproximation::Tanh, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        max_pool_2d_module_exact_native(
            "pool",
            [0; 2],
            Some([0; 2]),
            [usize::MAX; 2],
            [0; 2],
            true,
            &cancellation,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        module_dict_exact_native("dict", vec![], &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        pixel_shuffle_module_exact_native("shuffle", 0, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        relu_module_exact_native("relu", true, &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(matches!(
        zero_pad_2d_module_exact_native("pad", [usize::MAX; 4], &cancellation),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}

fn dropout_stream() -> Result<RngStream, Box<dyn std::error::Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V1,
        RngAlgorithm::Philox4x32_10,
        19,
        RngStreamAddress::new(
            "workflow",
            "attempt",
            "dropout",
            0,
            "forward",
            0,
            0,
            RetryRngPolicy::Replay,
        )?,
    )?)
}

#[test]
fn neural_network_module_part_four_extends_the_canonical_lifecycle_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;
    let input = tensor(&backend, &[1, 4], &[1.0, 3.0, 5.0, 7.0], &cancellation)?;

    let base = module_exact_native("base", &cancellation)?;
    assert!(matches!(base.spec(), NativeModuleSpec::Container));

    let mut pool = average_pool_1d_module_exact_native("pool", 2, Some(2), &cancellation)?;
    let pooled = pool.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(pooled.descriptor().shape(), &[1, 2]);
    assert_eq!(tensor_values(&backend, &pooled, &cancellation)?, [2.0, 6.0]);

    let signed = tensor(&backend, &[3], &[-1.0, 0.0, 1.0], &cancellation)?;
    let mut activation = sequential_module_exact_native(
        "activation",
        vec![
            elu_module_exact_native("elu", 1.0, false, &cancellation)?,
            sigmoid_module_exact_native("sigmoid", &cancellation)?,
            identity_module_exact_native("identity", &cancellation)?,
        ],
        &cancellation,
    )?;
    let activated = activation.forward_with_context(&backend, &signed, &execution)?;
    let activated_values = tensor_values(&backend, &activated, &cancellation)?;
    assert!((activated_values[0] - 0.34703022).abs() < 1.0e-6);
    assert_eq!(activated_values[1], 0.5);
    assert!((activated_values[2] - 0.7310586).abs() < 1.0e-6);

    let target = tensor(&backend, &[2], &[1.0, 0.0], &cancellation)?;
    let loss_input = tensor(&backend, &[2], &[0.0, 2.0], &cancellation)?;
    let mut loss = mse_loss_module_exact_native("mse", LossReduction::Mean, &cancellation)?;
    let loss = loss.forward_loss_with_context(&backend, &loss_input, &target, &execution)?;
    assert_eq!(tensor_values(&backend, &loss, &cancellation)?, [2.5]);
    Ok(())
}

#[test]
fn dropout_training_state_and_rng_publication_are_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation)?;
    let input = tensor(&backend, &[16], &[1.0; 16], &cancellation)?;
    let stream = dropout_stream()?;

    let mut first = dropout_module_exact_native("dropout", 0.5, false, &cancellation)?;
    let first_result =
        first.forward_with_rng_with_context(&backend, &input, stream.begin(None)?, &execution)?;
    assert_eq!(first.generation(), 1);
    let first_values = tensor_values(&backend, &first_result.output, &cancellation)?;

    let mut replay = dropout_module_exact_native("dropout", 0.5, false, &cancellation)?;
    let replay_result =
        replay.forward_with_rng_with_context(&backend, &input, stream.begin(None)?, &execution)?;
    assert_eq!(
        first_values,
        tensor_values(&backend, &replay_result.output, &cancellation)?
    );

    let checkpoint = first_result.transaction.commit();
    let mut advanced = dropout_module_exact_native("dropout", 0.5, false, &cancellation)?;
    let advanced_result = advanced.forward_with_rng_with_context(
        &backend,
        &input,
        stream.begin(Some(checkpoint))?,
        &execution,
    )?;
    assert_ne!(
        first_values,
        tensor_values(&backend, &advanced_result.output, &cancellation)?
    );

    replay.set_training(false);
    let evaluation = replay.forward_with_context(&backend, &input, &execution)?;
    assert_eq!(evaluation.storage_id(), input.storage_id());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_execution = context(&backend, &cancelled)?;
    let generation = advanced.generation();
    assert!(matches!(
        advanced.forward_with_rng_with_context(
            &backend,
            &input,
            stream.begin(None)?,
            &cancelled_execution,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(advanced.generation(), generation);
    assert!(matches!(
        dropout_module_exact_native("dropout", f32::NAN, true, &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}
