use super::*;
use crate::ScratchReservation;
use half::f16;

fn descriptor(shape: Vec<u64>, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        shape,
        dtype,
        DeviceId::new(DeviceKind::Metal, 0),
        StreamId::DEFAULT,
    )
}

fn offset_descriptor(
    shape: Vec<u64>,
    dtype: DType,
    offset_elements: u64,
    device: DeviceId,
    stream: StreamId,
) -> Result<TensorDescriptor, TensorError> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1_i64;
    for dimension in shape.iter().rev() {
        strides.push(stride);
        let dimension = i64::try_from(*dimension).map_err(|_| TensorError::ShapeOverflow)?;
        stride = stride
            .checked_mul(dimension)
            .ok_or(TensorError::ShapeOverflow)?;
    }
    strides.reverse();
    TensorDescriptor::new_strided(
        shape,
        strides,
        offset_elements,
        dtype,
        Layout::Contiguous,
        device,
        stream,
    )
}

fn execution_context<'a>(
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

fn test_backend(
    capacity: u64,
    host_memory: u64,
) -> Result<(MetalTensorBackend, ScratchReservation), TensorError> {
    let runtime = MetalRuntime::for_test_harness(capacity, true)
        .map_err(|error| map_execution_error("zed.metal.test-harness", error))?;
    let cancellation = CancellationToken::default();
    let (backend, authority) =
        MetalTensorBackend::from_certified_runtime(runtime, host_memory, capacity, &cancellation)?;
    Ok((backend, authority.authorize_workspace(capacity)?))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    bytes
        .chunks_exact(4)
        .map(|bytes| {
            <[u8; 4]>::try_from(bytes)
                .map(f32::from_le_bytes)
                .map_err(|_| TensorError::Faulted {
                    reason: "invalid f32 test lane".to_owned(),
                })
        })
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    bytes
        .chunks_exact(2)
        .map(|bytes| {
            <[u8; 2]>::try_from(bytes)
                .map(u16::from_le_bytes)
                .map(f16::from_bits)
                .map(f16::to_f32)
                .map_err(|_| TensorError::Faulted {
                    reason: "invalid f16 test lane".to_owned(),
                })
        })
        .collect()
}

fn encode(dtype: DType, values: &[f32]) -> Result<Vec<u8>, TensorError> {
    match dtype {
        DType::F16 => Ok(f16_bytes(values)),
        DType::F32 => Ok(f32_bytes(values)),
        _ => Err(TensorError::Faulted {
            reason: "Metal fixture accepts f16/f32 only".to_owned(),
        }),
    }
}

fn decode(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    match dtype {
        DType::F16 => decode_f16(bytes),
        DType::F32 => decode_f32(bytes),
        _ => Err(TensorError::Faulted {
            reason: "Metal fixture accepts f16/f32 only".to_owned(),
        }),
    }
}

#[test]
fn val_device_001_exact_twelve_row_matrix_and_source_visible_memory_facts_are_instance_derived()
-> Result<(), TensorError> {
    let (backend, _) = test_backend(4 * 1024, 32 * 1024)?;
    assert_eq!(backend.device(), DeviceId::new(DeviceKind::Metal, 0));
    assert_eq!(backend.capabilities().supported().len(), 12);
    for dtype in [DType::F16, DType::F32] {
        for row in [
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
            OperationSupport::binary_input(BinaryOperation::Add, dtype, Layout::Contiguous),
            OperationSupport::binary_output(BinaryOperation::Add, dtype, Layout::Contiguous),
        ] {
            assert!(backend.capabilities().supports(row));
            assert!(backend.capabilities().is_deterministic(row));
        }
    }
    assert!(
        backend
            .capabilities()
            .supports(OperationSupport::record_event())
    );
    assert!(
        backend
            .capabilities()
            .supports(OperationSupport::wait_event())
    );
    assert!(
        !backend
            .capabilities()
            .supports(OperationSupport::binary_input(
                BinaryOperation::Multiply,
                DType::F32,
                Layout::Contiguous,
            ))
    );
    let properties =
        backend
            .capabilities()
            .device_properties()
            .ok_or_else(|| TensorError::Faulted {
                reason: "Metal properties are missing".to_owned(),
            })?;
    assert_eq!(properties.total_memory_bytes(), 32 * 1024);
    assert_eq!(backend.memory_snapshot().limit_bytes, 4 * 1024);
    assert_eq!(backend.physical_memory_snapshot().limit_bytes, 4 * 1024);
    Ok(())
}

#[test]
fn val_tensor_001_genuine_runtime_path_covers_f16_f32_odd_scalar_empty_copy_and_aliased_add()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend(32 * 1024, 64 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(scratch, &cancellation);
    for dtype in [DType::F16, DType::F32] {
        for (shape, left_values, right_values) in [
            (vec![3], vec![0.5, 1.0, -2.0], vec![0.25, 2.0, 4.0]),
            (Vec::new(), vec![1.5], vec![2.25]),
            (vec![0, 3], Vec::new(), Vec::new()),
        ] {
            let (left, left_event) = backend.upload_bytes(
                descriptor(shape.clone(), dtype)?,
                &encode(dtype, &left_values)?,
                &context,
            )?;
            backend.wait_event(left_event, &context)?;
            let (right, right_event) = backend.upload_bytes(
                descriptor(shape.clone(), dtype)?,
                &encode(dtype, &right_values)?,
                &context,
            )?;
            backend.wait_event(right_event, &context)?;
            let (copy, copy_event) =
                backend.copy(&left, descriptor(shape.clone(), dtype)?, &context)?;
            backend.wait_event(copy_event, &context)?;
            assert_eq!(
                decode(dtype, &backend.download_bytes(&copy, &context)?)?,
                left_values
            );
            let (sum, sum_event) = backend.binary(
                BinaryOperation::Add,
                &left,
                &right,
                descriptor(shape.clone(), dtype)?,
                &context,
            )?;
            backend.wait_event(sum_event, &context)?;
            assert_eq!(
                decode(dtype, &backend.download_bytes(&sum, &context)?)?,
                left_values
                    .iter()
                    .zip(&right_values)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>()
            );
            let (aliased, aliased_event) = backend.binary(
                BinaryOperation::Add,
                &left,
                &left,
                descriptor(shape, dtype)?,
                &context,
            )?;
            backend.wait_event(aliased_event, &context)?;
            assert_eq!(
                decode(dtype, &backend.download_bytes(&aliased, &context)?)?,
                left_values.iter().map(|value| value * 2.0).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[test]
fn cancellation_preflight_and_terminal_wait_retirement_converge() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend(1024, 4096)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(scratch.clone(), &cancelled);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32)?, &cancelled_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);

    let live = CancellationToken::default();
    let live_context = execution_context(scratch.clone(), &live);
    let event = backend.record_event(&live_context)?;
    backend.wait_event(event, &live_context)?;
    backend.wait_event(event, &live_context)?;

    backend
        .runtime
        .inject_test_command_failure(11)
        .map_err(|error| map_execution_error("zed.metal.failure-injection", error))?;
    let terminal_event = backend.record_event(&live_context)?;
    let cancelled_after_dispatch = CancellationToken::default();
    cancelled_after_dispatch.cancel();
    let cancelled_after_dispatch_context =
        execution_context(scratch.clone(), &cancelled_after_dispatch);
    assert!(matches!(
        backend.wait_event(terminal_event, &cancelled_after_dispatch_context),
        Err(TensorError::Cancelled)
    ));
    let retry = CancellationToken::default();
    let retry_context = execution_context(scratch, &retry);
    backend.wait_event(terminal_event, &retry_context)?;
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn post_dispatch_cancellation_waits_retires_discards_and_returns_cancelled()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend(1024, 4096)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(scratch, &cancellation);
    let (input, input_event) = backend.upload_bytes(
        descriptor(vec![2], DType::F32)?,
        &f32_bytes(&[1.0, 2.0]),
        &context,
    )?;
    backend.wait_event(input_event, &context)?;
    let baseline_logical = backend.memory_snapshot().current_bytes;
    let baseline_physical = backend.physical_memory_snapshot().current_bytes;
    backend
        .runtime
        .inject_test_command_failure(11)
        .map_err(|error| map_execution_error("zed.metal.failure-injection", error))?;
    backend.cancel_after_next_native_event(cancellation.clone())?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &input,
            &input,
            descriptor(vec![2], DType::F32)?,
            &context,
        ),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.events.pending_len()?, 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline_logical);
    assert_eq!(
        backend.physical_memory_snapshot().current_bytes,
        baseline_physical
    );
    Ok(())
}

#[test]
fn preflight_shape_dtype_foreign_storage_and_ordinal_errors_have_no_native_effects()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend(4096, 8192)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(scratch, &cancellation);
    let (left, _) = backend.upload_bytes(
        descriptor(vec![2], DType::F32)?,
        &f32_bytes(&[1.0, 2.0]),
        &context,
    )?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![2], DType::F16)?,
        &f16_bytes(&[1.0, 2.0]),
        &context,
    )?;
    let peak = backend.physical_memory_snapshot().peak_bytes;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor(vec![2], DType::F32)?,
            &context,
        ),
        Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16
        })
    ));
    assert_eq!(backend.physical_memory_snapshot().peak_bytes, peak);
    assert!(matches!(
        backend.allocate(
            TensorDescriptor::contiguous(
                vec![1],
                DType::F32,
                DeviceId::new(DeviceKind::Metal, 1),
                StreamId::DEFAULT,
            )?,
            &context,
        ),
        Err(TensorError::DeviceMismatch { .. })
    ));

    let (other, other_scratch) = test_backend(1024, 8192)?;
    let other_context = execution_context(other_scratch, &cancellation);
    let (foreign, _) = other.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[9.0]),
        &other_context,
    )?;
    assert!(matches!(
        backend.copy(&foreign, descriptor(vec![1], DType::F32)?, &context),
        Err(TensorError::UnsupportedCapability { .. })
    ));

    let peak = backend.physical_memory_snapshot().peak_bytes;
    assert!(matches!(
        backend.copy(&left, descriptor(vec![3], DType::F32)?, &context),
        Err(TensorError::Faulted { .. })
    ));
    assert!(matches!(
        backend.binary(
            BinaryOperation::Multiply,
            &left,
            &left,
            descriptor(vec![2], DType::F32)?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &left,
            offset_descriptor(
                vec![2],
                DType::F32,
                1,
                DeviceId::new(DeviceKind::Metal, 0),
                StreamId::DEFAULT,
            )?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    let strided = TensorDescriptor::new_strided(
        vec![2],
        vec![2],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::new(DeviceKind::Metal, 0),
        StreamId::DEFAULT,
    )?;
    assert!(matches!(
        backend.allocate(strided, &context),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert_eq!(backend.physical_memory_snapshot().peak_bytes, peak);

    assert!(matches!(
        backend.record_event(&other_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));
    let foreign_event = other.record_event(&other_context)?;
    assert!(matches!(
        backend.wait_event(foreign_event, &context),
        Err(TensorError::Faulted { .. })
    ));
    let stream_one_context = ExecutionContext {
        stream: StreamId::new(1),
        scratch: context.scratch.clone(),
        rng_phase: None,
        cancellation: &cancellation,
    };
    let stream_one_event = backend.record_event(&stream_one_context)?;
    assert!(matches!(
        backend.wait_event(stream_one_event, &context),
        Err(TensorError::StreamMismatch { .. })
    ));
    backend.wait_event(stream_one_event, &stream_one_context)?;
    Ok(())
}

#[test]
fn val_memory_001_cpu_copy_staging_is_bounded_and_offset_gaps_are_zeroed()
-> Result<(), TensorError> {
    let (backend, authority) = {
        let runtime = MetalRuntime::for_test_harness(64, true)
            .map_err(|error| map_execution_error("zed.metal.test-harness", error))?;
        let cancellation = CancellationToken::default();
        MetalTensorBackend::from_certified_runtime(runtime, 4096, 64, &cancellation)?
    };
    let cancellation = CancellationToken::default();
    let source = Tensor::from_bytes(
        TensorDescriptor::contiguous(
            vec![3],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        f32_bytes(&[1.0, -2.0, 3.5]),
    )?;
    let underauthorized = execution_context(authority.authorize_workspace(11)?, &cancellation);
    let empty_channels_last = Tensor::from_bytes(
        TensorDescriptor::channels_last(
            vec![1, 0, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        Vec::new(),
    )?;
    assert!(matches!(
        backend.copy(
            &empty_channels_last,
            descriptor(vec![1, 0, 1, 1], DType::F32)?,
            &underauthorized,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        backend.copy(
            &source,
            offset_descriptor(
                vec![3],
                DType::F32,
                2,
                DeviceId::new(DeviceKind::Metal, 0),
                StreamId::DEFAULT,
            )?,
            &underauthorized,
        ),
        Err(TensorError::WorkspaceAuthorizationExceeded { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);

    let context = execution_context(authority.authorize_workspace(64)?, &cancellation);
    let destination = offset_descriptor(
        vec![3],
        DType::F32,
        2,
        DeviceId::new(DeviceKind::Metal, 0),
        StreamId::DEFAULT,
    )?;
    let (copy, event) = backend.copy(&source, destination, &context)?;
    backend.wait_event(event, &context)?;
    assert_eq!(copy.storage_byte_len(), 20);
    assert_eq!(
        decode_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![1.0, -2.0, 3.5]
    );
    let storage = backend.storage(&copy)?;
    let allocation = storage
        .allocation
        .as_ref()
        .ok_or_else(|| TensorError::Faulted {
            reason: "nonempty Metal copy has no allocation".to_owned(),
        })?;
    let stream = backend.stream(StreamId::DEFAULT, &cancellation)?;
    let mut backing = vec![u8::MAX; 20];
    backend
        .runtime
        .copy_device_to_host(&stream, allocation, 0, &mut backing)
        .map_err(|error| map_execution_error("zed.metal.test.backing-read", error))?;
    assert_eq!(&backing[..8], &[0; 8]);
    assert_eq!(&backing[8..], source.contiguous_bytes()?);
    drop(storage);
    drop(copy);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn initialization_symbol_pipeline_and_execution_errors_map_to_distinct_typed_domains() {
    for error in [
        MetalExecutionError::UnsupportedTarget {
            target: "unsupported-target".into(),
        },
        MetalExecutionError::NoSystemDevice,
        MetalExecutionError::InvalidAbi {
            reason: "abi-version".into(),
        },
        MetalExecutionError::InvalidCertifiedInputs {
            reason: "certificate".into(),
        },
        MetalExecutionError::MissingFunction {
            function: "zed_sim_metal_add_f32_v1".into(),
        },
        MetalExecutionError::PipelineCreation {
            function: "zed_sim_metal_add_f32_v1".into(),
            reason: "pipeline".into(),
        },
    ] {
        assert!(matches!(
            map_execution_error("zed.metal.initialize", error),
            TensorError::UnsupportedCapability { .. }
        ));
    }
    assert!(matches!(
        map_execution_error(
            "zed.metal.allocate",
            MetalExecutionError::OutOfMemory { requested: 17 },
        ),
        TensorError::AllocationFailed { requested: 17, .. }
    ));
    assert!(matches!(
        map_execution_error(
            "zed.metal.wait",
            MetalExecutionError::DeviceLost { code: 11 },
        ),
        TensorError::DeviceLost { .. }
    ));
    assert!(matches!(
        map_execution_error("zed.metal.copy", MetalExecutionError::ForeignResource),
        TensorError::UnsupportedCapability { .. }
    ));
    assert!(matches!(
        map_execution_error(
            "zed.metal.copy",
            MetalExecutionError::ResourceBounds {
                offset: 8,
                length: 8,
                available: 12,
            },
        ),
        TensorError::Faulted { .. }
    ));
}

#[test]
fn val_memory_001_logical_physical_accounting_oom_device_loss_and_drop_converge()
-> Result<(), TensorError> {
    let runtime = MetalRuntime::for_test_harness(24, false)
        .map_err(|error| map_execution_error("zed.metal.test-harness", error))?;
    let injector = runtime.clone();
    let cancellation = CancellationToken::default();
    let (backend, authority) =
        MetalTensorBackend::from_certified_runtime(runtime, 4096, 24, &cancellation)?;
    let context = execution_context(authority.authorize_workspace(24)?, &cancellation);
    let (left, _) = backend.upload_bytes(
        descriptor(vec![2], DType::F32)?,
        &f32_bytes(&[1.0, 2.0]),
        &context,
    )?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![2], DType::F32)?,
        &f32_bytes(&[3.0, 4.0]),
        &context,
    )?;
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 16);
    assert!(matches!(
        backend.allocate(descriptor(vec![3], DType::F32)?, &context),
        Err(TensorError::AllocationFailed { requested: 12, .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 16);

    injector
        .inject_test_command_failure(11)
        .map_err(|error| map_execution_error("zed.metal.failure-injection", error))?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        descriptor(vec![2], DType::F32)?,
        &context,
    )?;
    assert!(matches!(
        backend.wait_event(event, &context),
        Err(TensorError::DeviceLost { .. })
    ));
    backend.wait_event(event, &context)?;
    drop(sum);
    drop(left);
    drop(right);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    assert!(backend.memory_snapshot().peak_bytes >= 24);
    assert!(backend.physical_memory_snapshot().peak_bytes >= 24);
    Ok(())
}

#[test]
fn val_memory_001_stream_and_event_limits_reject_before_native_effects() -> Result<(), TensorError>
{
    let (stream_backend, scratch) = test_backend(1024, 4096)?;
    let cancellation = CancellationToken::default();
    let mut stream_events = Vec::with_capacity(MAX_METAL_STREAMS);
    for index in 0..MAX_METAL_STREAMS {
        let context = ExecutionContext {
            stream: StreamId::new(index as u64),
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        stream_events.push((stream_backend.record_event(&context)?, context.stream));
    }
    let overflowing_stream = ExecutionContext {
        stream: StreamId::new(MAX_METAL_STREAMS as u64),
        scratch: scratch.clone(),
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        stream_backend.record_event(&overflowing_stream),
        Err(TensorError::ResourceLimitExceeded {
            resource: "Metal streams",
            limit: MAX_METAL_STREAMS
        })
    ));
    for (event, stream) in stream_events {
        let context = ExecutionContext {
            stream,
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        stream_backend.wait_event(event, &context)?;
    }

    let (event_backend, event_scratch) = test_backend(1024, 4096)?;
    let event_context = execution_context(event_scratch, &cancellation);
    let mut events = Vec::with_capacity(MAX_METAL_PENDING_EVENTS);
    for _ in 0..MAX_METAL_PENDING_EVENTS {
        events.push(event_backend.record_event(&event_context)?);
    }
    assert!(matches!(
        event_backend.record_event(&event_context),
        Err(TensorError::ResourceLimitExceeded {
            resource: "Metal pending events",
            limit: MAX_METAL_PENDING_EVENTS
        })
    ));
    assert_eq!(event_backend.physical_memory_snapshot().peak_bytes, 0);
    for event in events {
        event_backend.wait_event(event, &event_context)?;
    }
    Ok(())
}
