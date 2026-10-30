use sim_model::{
    LayerQuantizationV1, NativeModule, NativeOpsError, QuantizationKind,
    disable_weight_init_conv1d_exact_native, disable_weight_init_group_norm_exact_native,
    disable_weight_init_layer_norm_exact_native, manual_cast_linear_exact_native,
    pick_operations_exact_native, remove_parametrizations_with_context_exact_native,
    spectral_norm_exact_native, weight_norm_exact_native,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        ConvolutionPaddingMode, OperatorIndirectionError, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use std::collections::BTreeMap;

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

#[test]
fn zero_initialization_is_checked_atomic_and_requires_a_complete_module()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[1, 2], &[3.0, -4.0], &cancellation)?;
    let mut module = NativeModule::linear("zeroed", 2, 2, true, false)?;
    module.zero_init_parameter_with_context_exact_native(
        &backend,
        "weight",
        DType::F32,
        DeviceId::CPU,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(module.generation(), 1);
    assert!(matches!(
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?),
        Err(NativeOpsError::ParametersNotLoaded)
    ));
    module.zero_init_parameter_with_context_exact_native(
        &backend,
        "bias",
        DType::F32,
        DeviceId::CPU,
        &context(&backend, &cancellation)?,
    )?;
    let output =
        module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?
        )?,
        vec![0.0, 0.0]
    );

    let generation = module.generation();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        module.zero_init_parameter_with_context_exact_native(
            &backend,
            "weight",
            DType::F16,
            DeviceId::CPU,
            &context(&backend, &cancelled)?,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(module.generation(), generation);
    let mut unsupported = NativeModule::linear("zeroed.unsupported", 2, 2, false, false)?;
    assert!(matches!(
        unsupported.zero_init_parameter_with_context_exact_native(
            &backend,
            "weight",
            DType::F32,
            DeviceId::new(DeviceKind::Cuda, 0),
            &context(&backend, &cancellation)?,
        ),
        Err(NativeOpsError::Tensor(
            OperatorIndirectionError::UnsupportedDevice { .. }
        ))
    ));
    assert_eq!(unsupported.generation(), 0);
    assert!(
        module
            .zero_init_parameter_with_context_exact_native(
                &backend,
                "running_mean",
                DType::F32,
                DeviceId::CPU,
                &context(&backend, &cancellation)?,
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn conv1d_constructor_reuses_grouped_kernel_and_nonzero_padding_mapping()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut convolution = disable_weight_init_conv1d_exact_native(
        "audio.replicate",
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        false,
        ConvolutionPaddingMode::Replicate,
    )?;
    convolution.load_dense_parameters(
        tensor(&backend, &[1, 1, 3], &[1.0, 1.0, 1.0], &cancellation)?,
        None,
    )?;
    let input = tensor(&backend, &[1, 1, 3], &[1.0, 2.0, 3.0], &cancellation)?;
    let output =
        convolution.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?
        )?,
        vec![4.0, 6.0, 8.0]
    );

    let grouped = disable_weight_init_conv1d_exact_native(
        "audio.grouped",
        4,
        4,
        3,
        1,
        2,
        2,
        4,
        true,
        ConvolutionPaddingMode::Zeros,
    )?;
    assert!(matches!(
        grouped.spec(),
        sim_model::NativeModuleSpec::Convolution { geometry, .. }
            if geometry.groups() == 4 && geometry.dilation() == [2]
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        convolution.forward_with_context(&backend, &input, &context(&backend, &cancelled)?),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(
        disable_weight_init_conv1d_exact_native(
            "audio.invalid",
            0,
            1,
            3,
            1,
            0,
            1,
            1,
            false,
            ConvolutionPaddingMode::Zeros,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn disable_weight_init_normalizers_delegate_to_the_functional_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = tensor(&backend, &[1, 4, 1], &[1.0, 3.0, 2.0, 6.0], &cancellation)?;

    let mut group = disable_weight_init_group_norm_exact_native("group", 2, 4, 0.00001, true)?;
    group.load_dense_parameters(
        tensor(&backend, &[4], &[1.0; 4], &cancellation)?,
        Some(tensor(&backend, &[4], &[0.0; 4], &cancellation)?),
    )?;
    let output =
        group.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    let values = tensor_to_f32_with_context_exact_native(
        &backend,
        &output,
        &context(&backend, &cancellation)?,
    )?;
    for (actual, expected) in values
        .iter()
        .zip([-0.999995, 0.999995, -0.999999, 0.999999])
    {
        assert!((actual - expected).abs() < 0.00002);
    }

    let mut non_affine =
        disable_weight_init_group_norm_exact_native("group.non_affine", 2, 4, 0.00001, false)?;
    let output =
        non_affine.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(output.descriptor().shape(), [1, 4, 1]);

    let mut layer =
        disable_weight_init_layer_norm_exact_native("layer", vec![2], 0.00001, false, false)?;
    let layer_input = tensor(&backend, &[1, 2], &[1.0, 3.0], &cancellation)?;
    let output =
        layer.forward_with_context(&backend, &layer_input, &context(&backend, &cancellation)?)?;
    let values = tensor_to_f32_with_context_exact_native(
        &backend,
        &output,
        &context(&backend, &cancellation)?,
    )?;
    assert!((values[0] + 0.999995).abs() < 0.00002);
    assert!((values[1] - 0.999995).abs() < 0.00002);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        layer.forward_with_context(&backend, &layer_input, &context(&backend, &cancelled)?,),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(
        disable_weight_init_layer_norm_exact_native("layer.invalid", vec![], 0.00001, true, true)
            .is_err()
    );
    assert!(
        disable_weight_init_group_norm_exact_native("group.invalid", 3, 4, 0.00001, true).is_err()
    );
    Ok(())
}

#[test]
fn operation_selection_and_manual_linear_share_one_module_policy_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut linear = manual_cast_linear_exact_native("manual", 2, 3, true)?;
    assert!(linear.manual_cast());
    linear.load_dense_parameters(
        tensor(
            &backend,
            &[3, 2],
            &[1.0, 2.0, -1.0, 0.5, 0.0, 3.0],
            &cancellation,
        )?,
        Some(tensor(&backend, &[3], &[0.5, -1.0, 2.0], &cancellation)?),
    )?;
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], &cancellation)?;
    let output =
        linear.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &output,
            &context(&backend, &cancellation)?,
        )?,
        vec![5.5, -1.0, 8.0]
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        linear.forward_with_context(&backend, &input, &context(&backend, &cancelled)?),
        Err(NativeOpsError::Cancelled)
    ));

    let direct = pick_operations_exact_native(
        DType::F32,
        None,
        Some(DeviceId::CPU),
        false,
        false,
        None,
        &cancellation,
    )?;
    assert!(!direct.manual_cast());
    assert_eq!(direct.weight_dtype(), DType::F32);
    assert_eq!(direct.compute_dtype(), DType::F32);
    assert!(!direct.linear("direct", 2, 2, false)?.manual_cast());

    let cast = pick_operations_exact_native(
        DType::F16,
        Some(DType::F32),
        Some(DeviceId::CPU),
        true,
        true,
        None,
        &cancellation,
    )?;
    assert!(cast.manual_cast());
    assert!(
        cast.conv1d(
            "selected.conv",
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            false,
            ConvolutionPaddingMode::Zeros,
        )?
        .manual_cast()
    );

    let mut quantization = BTreeMap::new();
    quantization.insert(
        "selected.linear".to_owned(),
        LayerQuantizationV1 {
            algorithm: QuantizationKind::Int8Tensorwise,
            original_dtype: DType::F32,
        },
    );
    let mixed = pick_operations_exact_native(
        DType::I8,
        Some(DType::Bf16),
        Some(DeviceId::CPU),
        false,
        false,
        Some(quantization),
        &cancellation,
    )?;
    assert!(mixed.is_mixed_precision());
    assert!(mixed.manual_cast());

    let unsupported_device = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        pick_operations_exact_native(
            DType::F32,
            Some(DType::F32),
            Some(unsupported_device),
            false,
            false,
            None,
            &cancellation,
        ),
        Err(NativeOpsError::UnsupportedDevice { device }) if device == unsupported_device
    ));
    assert!(matches!(
        pick_operations_exact_native(
            DType::F32,
            None,
            Some(DeviceId::CPU),
            false,
            false,
            None,
            &cancelled,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn weight_parametrization_removal_materializes_once_through_native_module()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut module = NativeModule::linear("weight_norm", 2, 2, false, false)?;
    module.load_weight_norm_parameters_with_context_exact_native(
        &backend,
        tensor(&backend, &[2, 1], &[2.0, 3.0], &cancellation)?,
        tensor(&backend, &[2, 2], &[3.0, 4.0, 0.0, 5.0], &cancellation)?,
        None,
        Some(0),
        &context(&backend, &cancellation)?,
    )?;
    assert!(module.has_weight_parametrization());
    let input = tensor(&backend, &[1, 2], &[1.0, 1.0], &cancellation)?;
    let before = tensor_to_f32_with_context_exact_native(
        &backend,
        &module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?,
        &context(&backend, &cancellation)?,
    )?;
    let generation = module.generation();
    remove_parametrizations_with_context_exact_native(
        &backend,
        &mut module,
        "weight",
        true,
        &context(&backend, &cancellation)?,
    )?;
    assert!(!module.has_weight_parametrization());
    assert_eq!(module.generation(), generation + 1);
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?,
            &context(&backend, &cancellation)?,
        )?,
        before
    );

    let mut cancelled_module = NativeModule::linear("cancelled_norm", 2, 2, false, false)?;
    cancelled_module.load_weight_norm_parameters_with_context_exact_native(
        &backend,
        tensor(&backend, &[2, 1], &[2.0, 3.0], &cancellation)?,
        tensor(&backend, &[2, 2], &[3.0, 4.0, 0.0, 5.0], &cancellation)?,
        None,
        Some(0),
        &context(&backend, &cancellation)?,
    )?;
    let generation = cancelled_module.generation();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        remove_parametrizations_with_context_exact_native(
            &backend,
            &mut cancelled_module,
            "invalid-name",
            false,
            &context(&backend, &cancelled)?,
        ),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(cancelled_module.generation(), generation);
    assert!(cancelled_module.has_weight_parametrization());
    assert!(
        remove_parametrizations_with_context_exact_native(
            &backend,
            &mut cancelled_module,
            "weight",
            false,
            &context(&backend, &cancellation)?,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn weight_norm_registration_uses_native_module_parameter_lifecycle()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut module = NativeModule::linear("registered_weight_norm", 2, 2, false, false)?;
    let generation = module.generation();
    let returned = weight_norm_exact_native(&mut module, "weight", Some(0), &cancellation)?;
    assert_eq!(returned.generation(), generation + 1);
    assert!(returned.has_weight_parametrization());
    assert!(
        returned
            .load_dense_parameters(
                tensor(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], &cancellation)?,
                None,
            )
            .is_err()
    );
    returned.load_weight_norm_parameters_with_context_exact_native(
        &backend,
        tensor(&backend, &[2, 1], &[2.0, 3.0], &cancellation)?,
        tensor(&backend, &[2, 2], &[3.0, 4.0, 0.0, 5.0], &cancellation)?,
        None,
        Some(0),
        &context(&backend, &cancellation)?,
    )?;
    let input = tensor(&backend, &[1, 2], &[1.0, 1.0], &cancellation)?;
    let output = tensor_to_f32_with_context_exact_native(
        &backend,
        &returned.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?,
        &context(&backend, &cancellation)?,
    )?;
    assert!((output[0] - 2.8).abs() < 1e-6);
    assert_eq!(output[1], 3.0);

    let mut cancelled_module = NativeModule::linear("cancelled_registration", 2, 2, false, false)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        weight_norm_exact_native(&mut cancelled_module, "weight", Some(0), &cancelled).is_err()
    );
    assert!(!cancelled_module.has_weight_parametrization());
    assert_eq!(cancelled_module.generation(), 0);
    Ok(())
}

#[test]
fn task_60_spectral_norm_uses_native_module_parameter_lifecycle()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let mut module = NativeModule::linear("spectral_norm", 2, 2, false, false)?;
    spectral_norm_exact_native(&mut module, "weight", 8, 1.0e-12, Some(0), &cancellation)?;
    assert!(module.has_weight_parametrization());
    assert!(module.has_spectral_parametrization());
    assert_eq!(module.generation(), 1);
    module.load_dense_parameters(
        tensor(&backend, &[2, 2], &[2.0, 0.0, 0.0, 1.0], &cancellation)?,
        None,
    )?;
    assert_eq!(module.generation(), 2);
    let input = tensor(&backend, &[1, 2], &[1.0, 1.0], &cancellation)?;
    let cancelled_forward = CancellationToken::default();
    cancelled_forward.cancel();
    assert!(
        module
            .forward_with_context(&backend, &input, &context(&backend, &cancelled_forward)?)
            .is_err()
    );
    assert_eq!(module.generation(), 2);
    let output = tensor_to_f32_with_context_exact_native(
        &backend,
        &module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?,
        &context(&backend, &cancellation)?,
    )?;
    assert!((output[0] - 1.0).abs() < 1.0e-5);
    assert!((output[1] - 0.5).abs() < 1.0e-5);
    assert_eq!(module.generation(), 3);

    let output = tensor_to_f32_with_context_exact_native(
        &backend,
        &module.forward_with_context(&backend, &input, &context(&backend, &cancellation)?)?,
        &context(&backend, &cancellation)?,
    )?;
    assert!((output[0] - 1.0).abs() < 1.0e-5);
    assert!((output[1] - 0.5).abs() < 1.0e-5);
    assert_eq!(module.generation(), 4);

    let mut invalid = NativeModule::linear("invalid_spectral", 2, 2, false, false)?;
    assert!(
        spectral_norm_exact_native(&mut invalid, "weight", 0, 1.0e-12, Some(0), &cancellation,)
            .is_err()
    );
    assert_eq!(invalid.generation(), 0);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        spectral_norm_exact_native(&mut invalid, "weight", 0, -1.0, Some(99), &cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert!(!invalid.has_weight_parametrization());
    assert_eq!(invalid.generation(), 0);
    Ok(())
}

#[test]
fn weight_norm_context_leases_all_staging_and_preserves_state_on_oom()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let magnitude = tensor(&backend, &[2, 1], &[2.0, 3.0], &cancellation)?;
    let direction = tensor(&backend, &[2, 2], &[3.0, 4.0, 0.0, 5.0], &cancellation)?;

    let probe = backend
        .workspace_authority
        .authorize_workspace(1024 * 1024)?;
    let probe_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: probe,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let mut module = NativeModule::linear("weight_norm.context", 2, 2, false, false)?;
    module.load_weight_norm_parameters_with_context_exact_native(
        &backend,
        magnitude.clone(),
        direction.clone(),
        None,
        Some(0),
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
    let mut exact_module = NativeModule::linear("weight_norm.exact", 2, 2, false, false)?;
    exact_module.load_weight_norm_parameters_with_context_exact_native(
        &backend,
        magnitude.clone(),
        direction.clone(),
        None,
        Some(0),
        &exact_context,
    )?;
    assert_eq!(exact_context.scratch.peak_bytes(), peak);
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let insufficient = backend.workspace_authority.authorize_workspace(peak - 1)?;
    let insufficient_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: insufficient,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let mut failed = NativeModule::linear("weight_norm.oom", 2, 2, false, false)?;
    assert!(
        failed
            .load_weight_norm_parameters_with_context_exact_native(
                &backend,
                magnitude,
                direction,
                None,
                Some(0),
                &insufficient_context,
            )
            .is_err()
    );
    assert_eq!(failed.generation(), 0);
    assert!(!failed.has_weight_parametrization());
    assert_eq!(insufficient_context.scratch.in_use_bytes(), 0);
    Ok(())
}
