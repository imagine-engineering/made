use super::*;
use crate::ScratchReservation;
use half::f16;

fn context<'a>(
    scratch: ScratchReservation,
    cancellation: &'a CancellationToken,
) -> ExecutionContext<'a> {
    ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch,
        rng_phase: None,
        cancellation,
    }
}

fn test_backend_with_limits(
    total_memory_bytes: usize,
    maximum_allocation_bytes: usize,
    requested_memory_limit_bytes: u64,
) -> Result<(CudaTensorBackend, ScratchReservation), TensorError> {
    let session = CudaExecutionSession::for_test_harness(
        0,
        "Injected NVIDIA CUDA adapter",
        12_020,
        (12, 2),
        120_205,
        90_000,
        total_memory_bytes,
        maximum_allocation_bytes,
    )
    .map_err(|error| map_execution_error("zed.cuda.test-harness", error))?;
    let cancellation = CancellationToken::default();
    let (backend, authority) = CudaTensorBackend::from_certified_session(
        session,
        requested_memory_limit_bytes,
        &cancellation,
    )?;
    let effective_limit = requested_memory_limit_bytes
        .min(u64::try_from(total_memory_bytes).map_err(|_| TensorError::ShapeOverflow)?)
        .min(u64::try_from(maximum_allocation_bytes).map_err(|_| TensorError::ShapeOverflow)?);
    Ok((backend, authority.authorize_workspace(effective_limit)?))
}

fn test_backend() -> Result<(CudaTensorBackend, ScratchReservation), TensorError> {
    test_backend_with_limits(1024 * 1024, 1024 * 1024, 1024 * 1024)
}

fn descriptor(shape: Vec<u64>, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        shape,
        dtype,
        DeviceId::new(DeviceKind::Cuda, 0),
        StreamId::DEFAULT,
    )
}

fn offset_descriptor(
    shape: Vec<u64>,
    dtype: DType,
    offset_elements: u64,
) -> Result<TensorDescriptor, TensorError> {
    let strides = match shape.as_slice() {
        [] => Vec::new(),
        [length] => vec![i64::from(*length != 0)],
        _ => {
            return Err(TensorError::Faulted {
                reason: "CUDA offset fixture supports rank zero or one".to_owned(),
            });
        }
    };
    TensorDescriptor::new_strided(
        shape,
        strides,
        offset_elements,
        dtype,
        Layout::Contiguous,
        DeviceId::new(DeviceKind::Cuda, 0),
        StreamId::DEFAULT,
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bytes_f32(values: &[u8]) -> Result<Vec<f32>, TensorError> {
    values
        .chunks_exact(4)
        .map(|value| {
            let value: [u8; 4] = value.try_into().map_err(|_| TensorError::Faulted {
                reason: "downloaded f32 chunk has the wrong byte width".to_owned(),
            })?;
            Ok(f32::from_ne_bytes(value))
        })
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_ne_bytes())
        .collect()
}

fn bytes_f16(values: &[u8]) -> Result<Vec<f32>, TensorError> {
    values
        .chunks_exact(2)
        .map(|value| {
            let value: [u8; 2] = value.try_into().map_err(|_| TensorError::Faulted {
                reason: "downloaded f16 chunk has the wrong byte width".to_owned(),
            })?;
            Ok(f16::from_bits(u16::from_ne_bytes(value)).to_f32())
        })
        .collect()
}

fn dtype_bytes(dtype: DType, values: &[f32]) -> Result<Vec<u8>, TensorError> {
    match dtype {
        DType::F16 => Ok(f16_bytes(values)),
        DType::F32 => Ok(f32_bytes(values)),
        _ => Err(TensorError::Faulted {
            reason: "CUDA fixture requires f16 or f32".to_owned(),
        }),
    }
}

fn bytes_dtype(dtype: DType, values: &[u8]) -> Result<Vec<f32>, TensorError> {
    match dtype {
        DType::F16 => bytes_f16(values),
        DType::F32 => bytes_f32(values),
        _ => Err(TensorError::Faulted {
            reason: "CUDA fixture requires f16 or f32".to_owned(),
        }),
    }
}

#[test]
fn instance_matrix_and_properties_are_derived_from_the_certified_session()
-> Result<(), TensorError> {
    let (backend, _) = test_backend()?;
    assert_eq!(backend.device(), DeviceId::new(DeviceKind::Cuda, 0));
    assert_eq!(backend.capabilities().supported().len(), 10);
    for dtype in [DType::F16, DType::F32] {
        for support in [
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
        ] {
            assert!(backend.capabilities().supports(support));
            assert!(backend.capabilities().is_deterministic(support));
        }
    }
    for support in [
        OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous),
        OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous),
    ] {
        assert!(backend.capabilities().supports(support));
        assert!(backend.capabilities().is_deterministic(support));
    }
    assert!(!backend.capabilities().supports(OperationSupport::binary_input(
        BinaryOperation::Add,
        DType::F16,
        Layout::Contiguous,
    )));
    assert!(
        !backend
            .capabilities()
            .supports(OperationSupport::binary_input(
                BinaryOperation::Multiply,
                DType::F32,
                Layout::Contiguous,
            ))
    );
    let properties = backend
        .capabilities()
        .device_properties()
        .ok_or_else(|| TensorError::Faulted {
            reason: "CUDA test device properties are absent".to_owned(),
        })?;
    assert_eq!(properties.name(), "Injected NVIDIA CUDA adapter");
    assert_eq!(properties.total_memory_bytes(), 1024 * 1024);
    assert_eq!(properties.allocation_limit_bytes(), 1024 * 1024);
    assert_eq!(properties.major(), 12);
    assert_eq!(properties.minor(), 2);
    assert_eq!(
        properties.architecture(),
        Some("CUDA driver 12020; NVRTC 12.2; cuBLASLt 120205; cuDNN 90000")
    );
    assert!(properties.has_fp16());
    Ok(())
}

#[test]
fn certified_session_preserves_f16_f32_odd_scalar_empty_and_copy_semantics()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    for dtype in [DType::F16, DType::F32] {
        for (shape, left_values, right_values) in [
            (vec![3], vec![0.5, 1.0, -2.0], vec![0.25, 2.0, 4.0]),
            (Vec::new(), vec![1.5], vec![2.25]),
            (vec![0, 3], Vec::new(), Vec::new()),
        ] {
            let left_bytes = dtype_bytes(dtype, &left_values)?;
            let right_bytes = dtype_bytes(dtype, &right_values)?;
            let (left, left_event) = backend.upload_bytes(
                descriptor(shape.clone(), dtype)?,
                &left_bytes,
                &context,
            )?;
            backend.wait_event(left_event, &context)?;
            let (right, right_event) = backend.upload_bytes(
                descriptor(shape.clone(), dtype)?,
                &right_bytes,
                &context,
            )?;
            backend.wait_event(right_event, &context)?;

            let (copy, copy_event) =
                backend.copy(&left, descriptor(shape.clone(), dtype)?, &context)?;
            backend.wait_event(copy_event, &context)?;
            assert_eq!(backend.download_bytes(&copy, &context)?, left_bytes);

            if dtype == DType::F32 {
                let (sum, sum_event) = backend.binary(
                    BinaryOperation::Add,
                    &left,
                    &right,
                    descriptor(shape, dtype)?,
                    &context,
                )?;
                backend.wait_event(sum_event, &context)?;
                backend.wait_event(sum_event, &context)?;
                let expected = left_values
                    .iter()
                    .zip(&right_values)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>();
                assert_eq!(
                    bytes_dtype(dtype, &backend.download_bytes(&sum, &context)?)?,
                    expected
                );
            } else {
                assert!(matches!(
                    backend.binary(
                        BinaryOperation::Add,
                        &left,
                        &right,
                        descriptor(shape, dtype)?,
                        &context,
                    ),
                    Err(TensorError::UnsupportedCapability { .. })
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn offsets_use_zero_filled_backing_and_add_rejects_nonzero_offsets()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, event) = backend.upload_bytes(
        offset_descriptor(vec![2], DType::F32, 3)?,
        &f32_bytes(&[1.25, -2.5]),
        &context,
    )?;
    backend.wait_event(event, &context)?;
    assert_eq!(left.storage_byte_len(), 20);
    let backing = backend.download_storage_bytes(&left, &context)?;
    assert_eq!(backing.get(..12), Some([0_u8; 12].as_slice()));
    assert_eq!(backing.get(12..), Some(f32_bytes(&[1.25, -2.5]).as_slice()));

    let (right, right_event) = backend.upload_bytes(
        descriptor(vec![2], DType::F32)?,
        &f32_bytes(&[2.0, 3.0]),
        &context,
    )?;
    backend.wait_event(right_event, &context)?;
    let before = backend.memory_snapshot();
    let pending = backend.events.pending_len()?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor(vec![2], DType::F32)?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert_eq!(backend.memory_snapshot(), before);
    assert_eq!(backend.events.pending_len()?, pending);
    Ok(())
}

#[test]
fn logical_and_native_allocation_limits_fail_without_leaking_accounting()
-> Result<(), TensorError> {
    let (native_limited, scratch) = test_backend_with_limits(128, 16, 128)?;
    assert_eq!(native_limited.memory_snapshot().limit_bytes, 16);
    let cancellation = CancellationToken::default();
    let native_context = context(scratch, &cancellation);
    assert!(matches!(
        native_limited.allocate(descriptor(vec![5], DType::F32)?, &native_context),
        Err(TensorError::AllocationFailed { requested: 20, .. })
    ));
    assert_eq!(native_limited.memory_snapshot().current_bytes, 0);

    let (logical_limited, scratch) = test_backend_with_limits(128, 128, 16)?;
    assert_eq!(logical_limited.memory_snapshot().limit_bytes, 16);
    let logical_context = context(scratch, &cancellation);
    assert!(matches!(
        logical_limited.allocate(descriptor(vec![5], DType::F32)?, &logical_context),
        Err(TensorError::AllocationFailed { requested: 20, .. })
    ));
    assert_eq!(logical_limited.memory_snapshot().current_bytes, 0);
    assert!(matches!(
        logical_limited.reserve_workspace(&logical_context, 17),
        Err(TensorError::WorkspaceAuthorizationExceeded { .. })
    ));
    Ok(())
}

#[test]
fn cancellation_foreign_authority_storage_and_events_fail_closed() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let (tensor, event) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[3.0]),
        &live_context,
    )?;

    let (other, other_scratch) = test_backend()?;
    let other_context = context(other_scratch, &cancellation);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32)?, &other_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));
    assert!(matches!(
        other.download_bytes(&tensor, &other_context),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        other.wait_event(event, &other_context),
        Err(TensorError::Faulted { .. })
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(scratch, &cancelled);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32)?, &cancelled_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 4);
    Ok(())
}

#[test]
fn cpu_copy_and_unsupported_operations_are_effect_safe() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let source = Tensor::from_bytes(
        TensorDescriptor::contiguous(
            vec![3],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        f32_bytes(&[1.0, -2.0, 3.5]),
    )?;
    let (copy, event) = backend.copy(
        &source,
        offset_descriptor(vec![3], DType::F32, 2)?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![1.0, -2.0, 3.5]
    );

    let before = backend.memory_snapshot();
    let pending = backend.events.pending_len()?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Multiply,
            &copy,
            &copy,
            descriptor(vec![3], DType::F32)?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert_eq!(backend.memory_snapshot(), before);
    assert_eq!(backend.events.pending_len()?, pending);
    Ok(())
}

#[test]
fn canonical_registries_bound_semantic_streams_and_pending_events()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    for stream in 0..MAX_CUDA_STREAMS {
        backend.stream(StreamId::new(stream as u64), &cancellation)?;
    }
    assert!(matches!(
        backend.stream(StreamId::new(MAX_CUDA_STREAMS as u64), &cancellation),
        Err(TensorError::ResourceLimitExceeded {
            resource: "CUDA semantic streams",
            limit: MAX_CUDA_STREAMS,
        })
    ));

    let context = context(scratch, &cancellation);
    let mut events = Vec::with_capacity(MAX_CUDA_PENDING_EVENTS);
    for _ in 0..MAX_CUDA_PENDING_EVENTS {
        events.push(backend.record_event(&context)?);
    }
    assert!(matches!(
        backend.record_event(&context),
        Err(TensorError::ResourceLimitExceeded {
            resource: "CUDA pending events",
            limit: MAX_CUDA_PENDING_EVENTS,
        })
    ));
    for event in events {
        backend.wait_event(event, &context)?;
    }
    assert_eq!(backend.events.pending_len()?, 0);
    Ok(())
}

#[test]
fn execution_errors_map_to_stable_tensor_error_categories() {
    assert!(matches!(
        map_execution_error("allocate", CudaExecutionError::Cancelled),
        TensorError::Cancelled
    ));
    assert!(matches!(
        map_execution_error(
            "allocate",
            CudaExecutionError::OutOfMemory {
                requested: 9,
                limit: 8,
            },
        ),
        TensorError::AllocationFailed { requested: 9, .. }
    ));
    assert!(matches!(
        map_execution_error(
            "add",
            CudaExecutionError::DeviceLost {
                device: 2,
                operation: "cuLaunchKernel",
            },
        ),
        TensorError::DeviceLost { .. }
    ));
    assert!(matches!(
        map_execution_error("identifier", CudaExecutionError::IdentifierOverflow),
        TensorError::IdentifierOverflow
    ));
    assert!(matches!(
        map_execution_error("bounds", CudaExecutionError::ResourceBounds {
            offset: 3,
            length: 4,
            available: 6,
        }),
        TensorError::Faulted { .. }
    ));
}
