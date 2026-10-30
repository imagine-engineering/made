use sim_model::{
    AttentionError, MathSdpSelection, NativeModuleSpec, NativeOpsError, PeriodicActivation,
    alias_free_activation::AliasFreeActivationError, alias_free_activation_1d_exact_native,
    enable_mem_efficient_sdp_exact_native, module_init_exact_native,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use std::error::Error;

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

fn test_backend(memory_limit_bytes: u64) -> Result<TestBackend, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(memory_limit_bytes)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn tensor(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &ExecutionContext {
                stream: StreamId::DEFAULT,
                scratch: backend
                    .workspace_authority
                    .authorize_workspace(1024 * 1024)?,
                rng_phase: None,
                cancellation,
            },
        )?
        .0)
}

#[test]
fn module_init_extends_the_canonical_empty_module_lifecycle() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let mut module = module_init_exact_native(&cancellation)?;
    assert_eq!(module.spec(), &NativeModuleSpec::Container);
    assert_eq!(module.generation(), 0);
    assert!(!module.has_weight_parametrization());

    let backend = test_backend(1024 * 1024)?;
    let input = tensor(&backend, &[1], &[1.0], &cancellation)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    assert!(
        module
            .forward_with_context(&backend, &input, &context)
            .is_err()
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(matches!(
        module_init_exact_native(&cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn mem_efficient_sdp_is_an_immutable_attention_policy() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    assert!(matches!(
        enable_mem_efficient_sdp_exact_native(true, &cancellation)?,
        MathSdpSelection::Enabled(_)
    ));
    assert_eq!(
        enable_mem_efficient_sdp_exact_native(false, &cancellation)?,
        MathSdpSelection::Disabled
    );
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert_eq!(
        enable_mem_efficient_sdp_exact_native(true, &cancelled),
        Err(AttentionError::Cancelled)
    );
    Ok(())
}

#[test]
fn activation1d_executes_native_kaiser_resampling_and_snake() -> Result<(), Box<dyn Error>> {
    let backend = test_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let activation = alias_free_activation_1d_exact_native(
        PeriodicActivation::Snake {
            alpha: vec![1.0, 0.5],
            logscale: false,
        },
        2,
        2,
        12,
        12,
        &cancellation,
    )?;
    assert_eq!(activation.base().spec(), &NativeModuleSpec::Container);
    let input = tensor(
        &backend,
        &[1, 2, 4],
        &[1.0, 1.0, 1.0, 1.0, -0.5, 0.0, 0.5, 1.0],
        &cancellation,
    )?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let first = activation.forward_with_context(&backend, &input, &context)?;
    let second = activation.forward_with_context(&backend, &input, &context)?;
    assert_eq!(first.descriptor().shape(), [1, 2, 4]);
    assert_eq!(first.host_storage_bytes()?, second.host_storage_bytes()?);
    let values = tensor_to_f32_with_context_exact_native(&backend, &first, &context)?;
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values[..4].iter().all(|value| *value > 1.5 && *value < 1.9));
    assert!(values[4] < -0.2);
    assert!(values[7] > 1.1);

    let beta = alias_free_activation_1d_exact_native(
        PeriodicActivation::SnakeBeta {
            alpha: vec![1.0, 1.0],
            beta: vec![2.0, 2.0],
            logscale: false,
        },
        2,
        2,
        12,
        12,
        &cancellation,
    )?;
    let beta_output = beta.forward_with_context(&backend, &input, &context)?;
    assert_ne!(
        first.host_storage_bytes()?,
        beta_output.host_storage_bytes()?
    );

    let empty_batch = tensor(&backend, &[0, 2, 4], &[], &cancellation)?;
    let empty_output = activation.forward_with_context(&backend, &empty_batch, &context)?;
    assert_eq!(empty_output.descriptor().shape(), [0, 2, 4]);
    assert!(empty_output.host_storage_bytes()?.is_empty());

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        activation.forward_with_context(&backend, &input, &cancelled_context),
        Err(AliasFreeActivationError::Cancelled)
    ));
    Ok(())
}

#[test]
fn activation1d_context_uses_exact_caller_workspace_and_converges() -> Result<(), Box<dyn Error>> {
    let backend = test_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let activation = alias_free_activation_1d_exact_native(
        PeriodicActivation::Snake {
            alpha: vec![1.0, 0.5],
            logscale: false,
        },
        2,
        2,
        12,
        12,
        &cancellation,
    )?;
    let input = tensor(
        &backend,
        &[1, 2, 4],
        &[1.0, 1.0, 1.0, 1.0, -0.5, 0.0, 0.5, 1.0],
        &cancellation,
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
    let expected = activation.forward_with_context(&backend, &input, &probe_context)?;
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
    let repeated = activation.forward_with_context(&backend, &input, &exact_context)?;
    assert_eq!(
        expected.host_storage_bytes()?,
        repeated.host_storage_bytes()?
    );
    assert_eq!(exact_context.scratch.peak_bytes(), peak);
    assert_eq!(exact_context.scratch.in_use_bytes(), 0);

    let insufficient = backend.workspace_authority.authorize_workspace(peak - 1)?;
    let insufficient_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: insufficient,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(
        activation
            .forward_with_context(&backend, &input, &insufficient_context)
            .is_err()
    );
    assert_eq!(insufficient_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_scratch = backend.workspace_authority.authorize_workspace(peak)?;
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: cancelled_scratch,
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(matches!(
        activation.forward_with_context(&backend, &input, &cancelled_context),
        Err(AliasFreeActivationError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task_63_model_adapters_observe_cancellation_before_validation() -> Result<(), Box<dyn Error>> {
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());

    assert!(matches!(
        alias_free_activation_1d_exact_native(
            PeriodicActivation::Snake {
                alpha: Vec::new(),
                logscale: false,
            },
            0,
            0,
            0,
            0,
            &cancelled,
        ),
        Err(AliasFreeActivationError::Cancelled)
    ));
    assert!(matches!(
        module_init_exact_native(&cancelled),
        Err(NativeOpsError::Cancelled)
    ));
    assert_eq!(
        enable_mem_efficient_sdp_exact_native(true, &cancelled),
        Err(AttentionError::Cancelled)
    );

    let backend = test_backend(1024 * 1024)?;
    let live = CancellationToken::default();
    let invalid_input = tensor(&backend, &[1], &[1.0], &live)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    let activation = alias_free_activation_1d_exact_native(
        PeriodicActivation::Snake {
            alpha: vec![1.0],
            logscale: false,
        },
        1,
        1,
        3,
        3,
        &live,
    )?;
    assert!(matches!(
        activation.forward_with_context(&backend, &invalid_input, &context),
        Err(AliasFreeActivationError::Cancelled)
    ));
    assert_eq!(context.scratch.peak_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}
