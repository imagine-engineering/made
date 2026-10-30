use sim_tensor::{
    BackendWorkspaceAuthority, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType,
    DeviceId, ExecutionContext, GENERATED_MODULES, StreamId, TensorBackend, TensorDescriptor,
    TensorError,
    generated_backend_cpu_sim_model_0016::{
        CpuTensorBackend, CpuTensorWorkspaceAuthority, initialize_cpu_tensor_backend,
    },
};
use std::any::TypeId;

fn context<'a>(
    authority: &CpuTensorWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(0)?,
        rng_phase: None,
        cancellation,
    })
}

fn descriptor(length: u64) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(vec![length], DType::F32, DeviceId::CPU, StreamId::DEFAULT)
}

#[test]
fn cpu_adapter_is_a_compiled_alias_of_the_canonical_backend() -> Result<(), TensorError> {
    assert!(GENERATED_MODULES.contains(&"ops/backend_cpu_comfy_model_0016"));
    assert_eq!(TypeId::of::<CpuTensorBackend>(), TypeId::of::<CpuBackend>());
    assert_eq!(
        TypeId::of::<CpuTensorWorkspaceAuthority>(),
        TypeId::of::<CpuWorkspaceAuthority>()
    );

    let (backend, _authority) = initialize_cpu_tensor_backend(64)?;
    assert_eq!(backend.device(), DeviceId::CPU);
    let properties =
        backend
            .capabilities()
            .device_properties()
            .ok_or_else(|| TensorError::Faulted {
                reason: "constructed CPU backend has no native properties".to_owned(),
            })?;
    assert_eq!(properties.device(), DeviceId::CPU);
    assert_eq!(properties.name(), "Zed native Rust CPU");
    assert_eq!(properties.total_memory_bytes(), 64);
    assert_eq!(properties.architecture(), Some(std::env::consts::ARCH));
    assert!(!properties.has_fp16());
    let compatibility = CpuBackend::capability_matrix();
    assert_eq!(
        backend.capabilities().supported(),
        compatibility.supported()
    );
    assert_eq!(
        backend.capabilities().deterministic(),
        compatibility.deterministic()
    );
    assert!(compatibility.device_properties().is_none());
    Ok(())
}

#[test]
fn adapter_delegates_transfer_events_and_shared_accounting() -> Result<(), TensorError> {
    let (backend, authority) = initialize_cpu_tensor_backend(64)?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation)?;
    let (source, upload_event) = backend.upload_f32(descriptor(2)?, &[1.25, -2.5], &context)?;
    backend.wait_event(upload_event, &context)?;
    assert_eq!(backend.memory_snapshot(), authority.memory_snapshot());
    assert_eq!(backend.memory_snapshot().current_bytes, 16);

    let (copy, copy_event) = backend.copy(&source, descriptor(2)?, &context)?;
    backend.wait_event(copy_event, &context)?;
    assert_eq!(copy.contiguous_bytes()?, source.contiguous_bytes()?);
    assert_eq!(backend.memory_snapshot(), authority.memory_snapshot());
    assert_eq!(backend.memory_snapshot().current_bytes, 32);

    drop(copy);
    drop(source);
    assert_eq!(backend.memory_snapshot(), authority.memory_snapshot());
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn adapter_preserves_cancellation_and_authority_boundaries() -> Result<(), TensorError> {
    let (backend, authority) = initialize_cpu_tensor_backend(16)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let cancelled_context = context(&authority, &cancellation)?;
    assert!(matches!(
        backend.allocate(descriptor(1)?, &cancelled_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let (_other_backend, other_authority) = BackendWorkspaceAuthority::create_backend(16)?;
    let active = CancellationToken::default();
    let foreign_context = context(&other_authority, &active)?;
    assert!(matches!(
        backend.allocate(descriptor(1)?, &foreign_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn adapter_preserves_constant_space_monotonic_cpu_events() -> Result<(), TensorError> {
    let (backend, authority) = initialize_cpu_tensor_backend(1)?;
    let cancellation = CancellationToken::default();
    let scratch = authority.authorize_workspace(0)?;
    let mut last_sequence = 0;
    for ordinal in 0..10_000 {
        let context = ExecutionContext {
            stream: StreamId::new(ordinal),
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        let event = backend.record_event(&context)?;
        backend.wait_event(event, &context)?;
        last_sequence = event.sequence();
    }
    assert_eq!(last_sequence, 10_000);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}
