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

fn test_backend() -> Result<(DirectMlTensorBackend, ScratchReservation), TensorError> {
    test_certified_backend(1024 * 1024, 1024 * 1024, true)
}

fn test_certified_backend(
    physical_capacity_bytes: u64,
    effective_memory_limit_bytes: u64,
    has_fp16: bool,
) -> Result<(DirectMlTensorBackend, ScratchReservation), TensorError> {
    let session = DirectMlExecutionSession::for_test_harness(physical_capacity_bytes, has_fp16)
        .map_err(|error| map_execution_error("zed.directml.test-harness", error))?;
    let cancellation = CancellationToken::default();
    let (backend, authority) = DirectMlTensorBackend::from_certified_session(
        session,
        effective_memory_limit_bytes,
        &cancellation,
    )?;
    Ok((
        backend,
        authority.authorize_workspace(effective_memory_limit_bytes)?,
    ))
}

fn test_certified_backend_with_control(
    physical_capacity_bytes: u64,
    effective_memory_limit_bytes: u64,
    has_fp16: bool,
) -> Result<
    (
        DirectMlTensorBackend,
        ScratchReservation,
        DirectMlTestControl,
    ),
    TensorError,
> {
    let (session, control) = DirectMlExecutionSession::for_test_harness_with_control(
        physical_capacity_bytes,
        has_fp16,
    )
    .map_err(|error| map_execution_error("zed.directml.test-harness", error))?;
    let cancellation = CancellationToken::default();
    let (backend, authority) = DirectMlTensorBackend::from_certified_session(
        session,
        effective_memory_limit_bytes,
        &cancellation,
    )?;
    let scratch = authority.authorize_workspace(effective_memory_limit_bytes)?;
    Ok((backend, scratch, control))
}

fn descriptor(shape: Vec<u64>, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        shape,
        dtype,
        DeviceId::new(DeviceKind::DirectMl, 0),
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
                reason: "DirectML offset fixture supports rank zero or one".to_owned(),
            });
        }
    };
    TensorDescriptor::new_strided(
        shape,
        strides,
        offset_elements,
        dtype,
        Layout::Contiguous,
        DeviceId::new(DeviceKind::DirectMl, 0),
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
            reason: "DirectML edge fixture requires f16 or f32".to_owned(),
        }),
    }
}

fn bytes_dtype(dtype: DType, values: &[u8]) -> Result<Vec<f32>, TensorError> {
    match dtype {
        DType::F16 => bytes_f16(values),
        DType::F32 => bytes_f32(values),
        _ => Err(TensorError::Faulted {
            reason: "DirectML edge fixture requires f16 or f32".to_owned(),
        }),
    }
}

#[derive(Debug)]
struct TestBoundaryStorage {
    device: DeviceId,
    byte_length: u64,
    host: Option<Vec<u8>>,
}

impl BackendStorage for TestBoundaryStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceId {
        self.device
    }

    fn byte_len(&self) -> u64 {
        self.byte_length
    }

    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError> {
        Err(TensorError::NonHostStorage)
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        self.host.as_deref()
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.host.as_deref_mut()
    }
}

fn boundary_tensor(
    descriptor: TensorDescriptor,
    host: Option<Vec<u8>>,
) -> Result<Tensor, TensorError> {
    let byte_length = descriptor.minimum_backing_byte_length()?;
    let device = descriptor.device();
    Tensor::from_backend_storage(
        descriptor,
        Box::new(TestBoundaryStorage {
            device,
            byte_length,
            host,
        }),
        ViewAccess::Writable,
    )
}

#[test]
fn instance_matrix_is_exactly_derived_from_device_fp16_support() -> Result<(), TensorError> {
    let (backend, _) = test_backend()?;
    assert_eq!(backend.device(), DeviceId::new(DeviceKind::DirectMl, 0));
    assert_eq!(backend.capabilities().supported().len(), 12);
    for dtype in [DType::F16, DType::F32] {
        for support in [
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
            OperationSupport::binary_input(BinaryOperation::Add, dtype, Layout::Contiguous),
            OperationSupport::binary_output(BinaryOperation::Add, dtype, Layout::Contiguous),
        ] {
            assert!(backend.capabilities().supports(support));
            assert!(backend.capabilities().is_deterministic(support));
        }
    }
    assert!(
        !backend
            .capabilities()
            .supports(OperationSupport::binary_input(
                BinaryOperation::Multiply,
                DType::F32,
                Layout::Contiguous,
            ))
    );
    assert!(
        !backend
            .capabilities()
            .supports(OperationSupport::fill(DType::F32, Layout::Contiguous))
    );
    let properties =
        backend
            .capabilities()
            .device_properties()
            .ok_or_else(|| TensorError::Faulted {
                reason: "DirectML test device properties are absent".to_owned(),
            })?;
    assert_eq!(properties.name(), "Injected DirectML adapter");
    assert_eq!(properties.total_memory_bytes(), 1024 * 1024);
    assert_eq!(properties.allocation_limit_bytes(), 1024 * 1024);
    assert_eq!(properties.major(), 1);
    assert_eq!(properties.minor(), 13);
    assert_eq!(
        properties.architecture(),
        Some("DXGI adapter LUID 0x1122334455667788")
    );
    assert!(properties.has_fp16());

    let (f32_only, _) = test_certified_backend(1024, 1024, false)?;
    assert_eq!(f32_only.capabilities().supported().len(), 7);
    assert!(!f32_only.capabilities().supports_dtype(DType::F16));
    Ok(())
}

#[test]
fn certified_properties_keep_physical_total_and_live_allocation_ceiling_distinct()
-> Result<(), TensorError> {
    let session = DirectMlExecutionSession::for_test_harness_with_memory_properties(
        96, 160, 64, true,
    )
    .map_err(|error| map_execution_error("zed.directml.test-memory-properties", error))?;
    let cancellation = CancellationToken::default();
    let (backend, authority) =
        DirectMlTensorBackend::from_certified_session(session, 128, &cancellation)?;
    let scratch = authority.authorize_workspace(64)?;
    let properties = backend
        .capabilities()
        .device_properties()
        .ok_or_else(|| TensorError::Faulted {
            reason: "DirectML memory properties are absent".to_owned(),
        })?;
    assert_eq!(properties.total_memory_bytes(), 256);
    assert_eq!(properties.allocation_limit_bytes(), 64);
    assert_eq!(backend.memory_snapshot().limit_bytes, 64);
    assert_eq!(backend.physical_memory_snapshot().limit_bytes, 64);

    let context = context(scratch, &cancellation);
    let (tensor, event) = backend.allocate(descriptor(vec![1], DType::F32)?, &context)?;
    backend.wait_event(event, &context)?;
    assert_eq!(backend.memory_snapshot().current_bytes, 4);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 4);
    drop(tensor);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn transfer_copy_add_scalar_and_empty_preserve_f32_semantics() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024 * 1024, 1024 * 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, upload_event) = backend.upload_bytes(
        descriptor(vec![2, 2], DType::F32)?,
        &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        &context,
    )?;
    backend.wait_event(upload_event, &context)?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![2, 2], DType::F32)?,
        &f32_bytes(&[10.0, 20.0, 30.0, 40.0]),
        &context,
    )?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        descriptor(vec![2, 2], DType::F32)?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&sum, &context)?)?,
        vec![11.0, 22.0, 33.0, 44.0]
    );
    let (copy, _) = backend.copy(&left, descriptor(vec![2, 2], DType::F32)?, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let (scalar_left, _) = backend.upload_bytes(
        descriptor(Vec::new(), DType::F32)?,
        &f32_bytes(&[1.5]),
        &context,
    )?;
    let (scalar_right, _) = backend.upload_bytes(
        descriptor(Vec::new(), DType::F32)?,
        &f32_bytes(&[2.25]),
        &context,
    )?;
    let (scalar, _) = backend.binary(
        BinaryOperation::Add,
        &scalar_left,
        &scalar_right,
        descriptor(Vec::new(), DType::F32)?,
        &context,
    )?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&scalar, &context)?)?,
        vec![3.75]
    );

    let (empty_left, _) =
        backend.upload_bytes(descriptor(vec![0, 3], DType::F32)?, &[], &context)?;
    let (empty_right, _) =
        backend.upload_bytes(descriptor(vec![0, 3], DType::F32)?, &[], &context)?;
    let (empty, empty_event) = backend.binary(
        BinaryOperation::Add,
        &empty_left,
        &empty_right,
        descriptor(vec![0, 3], DType::F32)?,
        &context,
    )?;
    backend.wait_event(empty_event, &context)?;
    assert!(backend.download_bytes(&empty, &context)?.is_empty());
    Ok(())
}

#[test]
fn f16_odd_element_add_uses_reviewed_native_element_type() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, _) = backend.upload_bytes(
        descriptor(vec![3], DType::F16)?,
        &f16_bytes(&[0.5, 1.0, 2.0]),
        &context,
    )?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![3], DType::F16)?,
        &f16_bytes(&[0.25, 2.0, 4.0]),
        &context,
    )?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        descriptor(vec![3], DType::F16)?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f16(&backend.download_bytes(&sum, &context)?)?,
        vec![0.75, 3.0, 6.0]
    );
    Ok(())
}

#[test]
fn certified_session_harness_covers_f16_f32_odd_scalar_and_empty_semantics()
-> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(16 * 1024, 16 * 1024, true)?;
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
            let (left, left_event) =
                backend.upload_bytes(descriptor(shape.clone(), dtype)?, &left_bytes, &context)?;
            backend.wait_event(left_event, &context)?;
            let (right, right_event) =
                backend.upload_bytes(descriptor(shape.clone(), dtype)?, &right_bytes, &context)?;
            backend.wait_event(right_event, &context)?;
            let (copy, copy_event) =
                backend.copy(&left, descriptor(shape.clone(), dtype)?, &context)?;
            backend.wait_event(copy_event, &context)?;
            assert_eq!(
                bytes_dtype(dtype, &backend.download_bytes(&copy, &context)?)?,
                left_values
            );
            let (sum, sum_event) = backend.binary(
                BinaryOperation::Add,
                &left,
                &right,
                descriptor(shape, dtype)?,
                &context,
            )?;
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
        }
    }
    Ok(())
}

#[test]
fn aliased_add_inputs_are_snapshotted_without_recursive_mutex_locking() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (input, _) = backend.upload_bytes(
        descriptor(vec![3], DType::F32)?,
        &f32_bytes(&[1.0, -2.0, 3.5]),
        &context,
    )?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &input,
        &input,
        descriptor(vec![3], DType::F32)?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&sum, &context)?)?,
        vec![2.0, -4.0, 7.0]
    );
    Ok(())
}

#[test]
fn certified_session_aliased_add_snapshots_the_input_once() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (input, input_event) = backend.upload_bytes(
        descriptor(vec![3], DType::F32)?,
        &f32_bytes(&[1.0, -2.0, 3.5]),
        &context,
    )?;
    backend.wait_event(input_event, &context)?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &input,
        &input,
        descriptor(vec![3], DType::F32)?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&sum, &context)?)?,
        vec![2.0, -4.0, 7.0]
    );
    Ok(())
}

#[test]
fn right_operand_dtype_mismatch_reports_the_right_dtype_before_effects() -> Result<(), TensorError>
{
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, _) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[1.0]),
        &context,
    )?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![1], DType::F16)?,
        &f16_bytes(&[2.0]),
        &context,
    )?;
    let logical_before = backend.memory_snapshot();
    let physical_before = backend.physical_memory_snapshot();
    let pending_before = backend.events.pending_len()?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor(vec![1], DType::F32)?,
            &context,
        ),
        Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        })
    ));
    assert_eq!(backend.memory_snapshot(), logical_before);
    assert_eq!(backend.physical_memory_snapshot(), physical_before);
    assert_eq!(backend.events.pending_len()?, pending_before);
    Ok(())
}

#[test]
fn nonzero_offset_transfers_use_logical_ranges_and_preserve_zeroed_gaps() -> Result<(), TensorError>
{
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let source_descriptor = offset_descriptor(vec![2], DType::F32, 2)?;
    let (source, _) =
        backend.upload_bytes(source_descriptor, &f32_bytes(&[1.25, -2.5]), &context)?;
    assert_eq!(source.storage_byte_len(), 16);
    assert_eq!(
        bytes_f32(&backend.download_bytes(&source, &context)?)?,
        vec![1.25, -2.5]
    );
    let source_bytes = backend.download_storage_bytes(&source, &context)?;
    assert_eq!(source_bytes.get(..8), Some([0_u8; 8].as_slice()));

    let (copy, _) = backend.copy(
        &source,
        offset_descriptor(vec![2], DType::F32, 3)?,
        &context,
    )?;
    assert_eq!(copy.storage_byte_len(), 20);
    assert_eq!(
        bytes_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![1.25, -2.5]
    );
    let copy_bytes = backend.download_storage_bytes(&copy, &context)?;
    assert_eq!(copy_bytes.get(..12), Some([0_u8; 12].as_slice()));
    assert_eq!(
        copy_bytes.get(12..),
        Some(f32_bytes(&[1.25, -2.5]).as_slice())
    );
    Ok(())
}

#[test]
fn nonzero_offset_add_is_rejected_before_allocation_event_or_dispatch() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, _) = backend.upload_bytes(
        offset_descriptor(vec![1], DType::F32, 1)?,
        &f32_bytes(&[2.0]),
        &context,
    )?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[3.0]),
        &context,
    )?;
    let logical_before = backend.memory_snapshot();
    let physical_before = backend.physical_memory_snapshot();
    let pending_before = backend.events.pending_len()?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor(vec![1], DType::F32)?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert_eq!(backend.memory_snapshot(), logical_before);
    assert_eq!(backend.physical_memory_snapshot(), physical_before);
    assert_eq!(backend.events.pending_len()?, pending_before);
    Ok(())
}

#[test]
fn offset_backing_uses_logical_accounting_physical_rounding_and_failure_convergence()
-> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(64, 64, true)?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch, &cancellation);
    let (tensor, _) = backend.upload_bytes(
        offset_descriptor(vec![1], DType::F16, 2)?,
        &f16_bytes(&[1.0]),
        &live_context,
    )?;
    assert_eq!(backend.memory_snapshot().current_bytes, 6);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 8);
    drop(tensor);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);

    let (failing_backend, failing_scratch) = test_certified_backend(7, 7, true)?;
    let failing_context = context(failing_scratch.clone(), &cancellation);
    assert!(matches!(
        failing_backend.upload_bytes(
            offset_descriptor(vec![1], DType::F16, 2)?,
            &f16_bytes(&[1.0]),
            &failing_context,
        ),
        Err(TensorError::AllocationFailed { .. })
    ));
    assert_eq!(failing_backend.memory_snapshot().current_bytes, 0);
    assert_eq!(failing_backend.physical_memory_snapshot().current_bytes, 0);
    let failed_snapshot = failing_backend.physical_memory_snapshot();
    assert_eq!(failed_snapshot.current_bytes, 0);
    assert_eq!(failed_snapshot.peak_bytes, 0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(failing_scratch, &cancelled);
    let logical_before = failing_backend.memory_snapshot();
    let physical_before = failing_backend.physical_memory_snapshot();
    assert!(matches!(
        failing_backend.upload_bytes(
            offset_descriptor(vec![1], DType::F16, 2)?,
            &f16_bytes(&[1.0]),
            &cancelled_context,
        ),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(failing_backend.memory_snapshot(), logical_before);
    assert_eq!(failing_backend.physical_memory_snapshot(), physical_before);
    Ok(())
}

#[test]
fn logical_and_dword_rounded_physical_capacity_are_distinct_and_converge() -> Result<(), TensorError>
{
    let (backend, scratch) = test_certified_backend(64, 48, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (tensor, _) = backend.upload_bytes(
        descriptor(vec![3], DType::F16)?,
        &f16_bytes(&[1.0, 2.0, 3.0]),
        &context,
    )?;
    assert_eq!(backend.memory_snapshot().limit_bytes, 48);
    assert_eq!(backend.memory_snapshot().current_bytes, 6);
    assert_eq!(backend.physical_memory_snapshot().limit_bytes, 64);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 8);
    drop(tensor);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn cancellation_foreign_authority_and_foreign_storage_fail_closed() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(64, 64, true)?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let (tensor, _) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[3.0]),
        &live_context,
    )?;

    let (other, other_scratch) = test_backend()?;
    let foreign_context = context(other_scratch, &cancellation);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32)?, &foreign_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));
    assert!(matches!(
        other.download_bytes(&tensor, &foreign_context),
        Err(TensorError::UnsupportedCapability { .. })
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(scratch, &cancelled);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32)?, &cancelled_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 4);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 4);
    Ok(())
}

#[test]
fn copy_rejects_shape_nonhost_strided_unsupported_and_foreign_sources_before_effects()
-> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(4096, 4096, true)?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch, &cancellation);
    let destination = || descriptor(vec![2, 2], DType::F32);

    let equal_bytes_wrong_shape = Tensor::from_bytes(
        TensorDescriptor::contiguous(vec![1, 4], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
        f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
    )?;
    let shape_error = backend
        .copy(&equal_bytes_wrong_shape, destination()?, &live_context)
        .expect_err("equal byte counts with different shapes must fail");
    assert!(matches!(shape_error, TensorError::Faulted { ref reason } if
        reason.contains("source shape [1, 4]") && reason.contains("destination shape [2, 2]")));

    let foreign_stream = StreamId::new(7);
    let cross_stream_cpu = Tensor::from_bytes(
        TensorDescriptor::contiguous(vec![2, 2], DType::F32, DeviceId::CPU, foreign_stream)?,
        f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
    )?;
    assert!(matches!(
        backend.copy(&cross_stream_cpu, destination()?, &live_context),
        Err(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        }) if actual == foreign_stream
    ));

    let nonhost_cpu = boundary_tensor(
        TensorDescriptor::contiguous(vec![2, 2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
        None,
    )?;
    assert!(matches!(
        backend.copy(&nonhost_cpu, destination()?, &live_context),
        Err(TensorError::NonHostStorage)
    ));

    let strided_descriptor = TensorDescriptor::new_strided(
        vec![2, 2],
        vec![3, 1],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let strided_bytes = usize::try_from(strided_descriptor.minimum_backing_byte_length()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let strided_cpu = Tensor::from_bytes(strided_descriptor, vec![0; strided_bytes])?;
    assert!(matches!(
        backend.copy(&strided_cpu, destination()?, &live_context),
        Err(TensorError::NonContiguousAccess)
    ));

    let unsupported_device = DeviceId::new(DeviceKind::Mlu, 0);
    let unsupported = boundary_tensor(
        TensorDescriptor::contiguous(
            vec![2, 2],
            DType::F32,
            unsupported_device,
            StreamId::DEFAULT,
        )?,
        None,
    )?;
    assert!(matches!(
        backend.copy(&unsupported, destination()?, &live_context),
        Err(TensorError::UnsupportedCapability { .. })
    ));

    let (foreign_backend, foreign_scratch) = test_backend()?;
    let foreign_context = context(foreign_scratch, &cancellation);
    let (foreign, _) = foreign_backend.upload_bytes(
        descriptor(vec![2, 2], DType::F32)?,
        &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        &foreign_context,
    )?;
    assert!(matches!(
        backend.copy(&foreign, destination()?, &live_context),
        Err(TensorError::UnsupportedCapability { .. })
    ));

    assert_eq!(backend.memory_snapshot(), BackendMemorySnapshot {
        limit_bytes: 4096,
        current_bytes: 0,
        peak_bytes: 0,
    });
    assert_eq!(backend.physical_memory_snapshot(), BackendMemorySnapshot {
        limit_bytes: 4096,
        current_bytes: 0,
        peak_bytes: 0,
    });
    assert_eq!(backend.events.pending_len()?, 0);
    Ok(())
}

#[test]
fn cpu_copy_is_fully_staged_before_destination_effects_and_preserves_values()
-> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let source = Tensor::from_bytes(
        TensorDescriptor::contiguous(vec![3], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
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
    assert_eq!(copy.storage_byte_len(), 20);
    Ok(())
}

#[test]
fn post_fence_cancellation_retires_event_and_allows_resource_reuse() -> Result<(), TensorError> {
    let (backend, scratch, control) = test_certified_backend_with_control(1024, 1024, true)?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let event = backend.record_event(&live_context)?;
    assert_eq!(backend.events.pending_len()?, 1);
    control
        .cancel_after_next_wait()
        .map_err(|error| map_execution_error("zed.directml.test.cancel-after-wait", error))?;
    assert!(matches!(
        backend.wait_event(event, &live_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.events.pending_len()?, 0);
    backend.wait_event(event, &live_context)?;

    let cancelled_event = backend.record_event(&live_context)?;
    assert!(cancellation.cancel());
    assert!(matches!(
        backend.wait_event(cancelled_event, &live_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.events.pending_len()?, 0);

    let active = CancellationToken::default();
    let active_context = context(scratch, &active);
    let reused = backend.record_event(&active_context)?;
    backend.wait_event(reused, &active_context)?;
    assert_eq!(backend.events.pending_len()?, 0);
    Ok(())
}

#[test]
fn physical_oom_device_loss_and_logical_rejection_release_capacity() -> Result<(), TensorError> {
    let (backend, scratch, control) = test_certified_backend_with_control(32, 32, true)?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch, &cancellation);
    let (live, live_event) = backend.allocate(descriptor(vec![1], DType::F32)?, &live_context)?;
    backend.wait_event(live_event, &live_context)?;
    control
        .fail_next_event_with_device_loss()
        .map_err(|error| map_execution_error("zed.directml.test.device-loss", error))?;
    let lost_event = backend.record_event(&live_context)?;
    assert!(matches!(
        backend.wait_event(lost_event, &live_context),
        Err(TensorError::DeviceLost { .. })
    ));
    assert_eq!(backend.events.pending_len()?, 0);
    drop(live);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);
    let recovered_event = backend.record_event(&live_context)?;
    backend.wait_event(recovered_event, &live_context)?;

    assert!(matches!(
        backend.allocate(descriptor(vec![9], DType::F16)?, &live_context),
        Err(TensorError::AllocationFailed { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.physical_memory_snapshot().current_bytes, 0);

    assert!(matches!(
        backend.reserve_workspace(&live_context, 33),
        Err(TensorError::WorkspaceAuthorizationExceeded { .. })
    ));
    Ok(())
}

#[test]
fn unsupported_dtype_layout_and_operation_never_reach_native_session() -> Result<(), TensorError> {
    let (backend, scratch) = test_certified_backend(1024, 1024, false)?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F16)?, &context),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    let strided = TensorDescriptor::new_strided(
        vec![2],
        vec![2],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::new(DeviceKind::DirectMl, 0),
        StreamId::DEFAULT,
    )?;
    assert!(matches!(
        backend.allocate(strided, &context),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    let (left, _) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[2.0]),
        &context,
    )?;
    let logical_before = backend.memory_snapshot();
    let physical_before = backend.physical_memory_snapshot();
    let pending_before = backend.events.pending_len()?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Multiply,
            &left,
            &left,
            descriptor(vec![1], DType::F32)?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    assert_eq!(backend.memory_snapshot(), logical_before);
    assert_eq!(backend.physical_memory_snapshot(), physical_before);
    assert_eq!(backend.events.pending_len()?, pending_before);
    Ok(())
}

#[test]
fn canonical_registries_bound_streams_and_pending_events() -> Result<(), TensorError> {
    let (backend, scratch, control) =
        test_certified_backend_with_control(1024 * 1024, 1024 * 1024, true)?;
    let cancellation = CancellationToken::default();
    for stream in 0..MAX_DIRECTML_STREAMS {
        backend.stream(StreamId::new(stream as u64), &cancellation)?;
    }
    assert!(matches!(
        backend.stream(StreamId::new(MAX_DIRECTML_STREAMS as u64), &cancellation),
        Err(TensorError::ResourceLimitExceeded {
            resource: "DirectML streams",
            limit: MAX_DIRECTML_STREAMS,
        })
    ));

    let context = context(scratch, &cancellation);
    let (left, left_event) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[2.0]),
        &context,
    )?;
    backend.wait_event(left_event, &context)?;
    let (right, right_event) = backend.upload_bytes(
        descriptor(vec![1], DType::F32)?,
        &f32_bytes(&[3.0]),
        &context,
    )?;
    backend.wait_event(right_event, &context)?;

    let mut events = Vec::with_capacity(MAX_DIRECTML_PENDING_EVENTS);
    for _ in 0..MAX_DIRECTML_PENDING_EVENTS {
        events.push(backend.record_event(&context)?);
    }
    let limit_error = backend
        .record_event(&context)
        .expect_err("pending DirectML event limit must reject the next event");
    assert!(matches!(
        limit_error,
        TensorError::ResourceLimitExceeded {
            resource: "DirectML pending events",
            limit: MAX_DIRECTML_PENDING_EVENTS,
        }
    ), "unexpected DirectML event limit error: {limit_error:?}");

    control
        .fail_next_event_with_command_failure()
        .map_err(|error| map_execution_error("zed.directml.test.command-failure", error))?;
    let logical_before = backend.memory_snapshot();
    let physical_before = backend.physical_memory_snapshot();
    let add_limit_error = backend
        .binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor(vec![1], DType::F32)?,
            &context,
        )
        .expect_err("canonical pending-event capacity must reject Add before native dispatch");
    assert!(matches!(
        add_limit_error,
        TensorError::ResourceLimitExceeded {
            resource: "DirectML pending events",
            limit: MAX_DIRECTML_PENDING_EVENTS,
        }
    ), "unexpected DirectML Add event-limit error: {add_limit_error:?}");
    assert_eq!(backend.memory_snapshot().current_bytes, logical_before.current_bytes);
    assert_eq!(
        backend.physical_memory_snapshot().current_bytes,
        physical_before.current_bytes
    );

    let latest = events.last().copied().ok_or_else(|| TensorError::Faulted {
        reason: "DirectML event fixture did not record events".to_owned(),
    })?;
    backend.wait_event(latest, &context)?;
    for event in events {
        backend.wait_event(event, &context)?;
    }
    assert_eq!(backend.events.pending_len()?, 0);

    let failed_event = backend.record_event(&context)?;
    let command_failure = backend
        .wait_event(failed_event, &context)
        .expect_err("queued non-device-loss command failure must survive rejected Add");
    assert!(
        matches!(command_failure, TensorError::Faulted { ref reason }
            if reason.contains("0x80004005") && reason.contains("injected fault")),
        "unexpected mapped DirectML command failure: {command_failure:?}"
    );
    assert_eq!(backend.events.pending_len()?, 0);

    let reusable_event = backend.record_event(&context)?;
    backend.wait_event(reusable_event, &context)?;
    assert_eq!(backend.events.pending_len()?, 0);
    assert_eq!(backend.memory_snapshot().current_bytes, logical_before.current_bytes);
    assert_eq!(
        backend.physical_memory_snapshot().current_bytes,
        physical_before.current_bytes
    );
    Ok(())
}
