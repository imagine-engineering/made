use super::*;
use crate::ScratchReservation;
use half::f16;
use std::sync::atomic::{AtomicU64, Ordering};

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

fn test_runtime(memory_limit_bytes: u64) -> Arc<TestRuntime> {
    Arc::new(TestRuntime {
        device_count: 1,
        memory_limit_bytes,
        failure: Mutex::new(None),
        cancel_after_next_synchronization: Mutex::new(None),
        allocation_calls: AtomicU64::new(0),
        synchronization_calls: AtomicU64::new(0),
    })
}

fn test_backend_with_runtime(
    runtime: Arc<TestRuntime>,
) -> Result<(NpuTensorBackend, ScratchReservation), TensorError> {
    let device = DeviceId::new(DeviceKind::Npu, 0);
    let properties = NativeDeviceProperties::new(
        device,
        "test Huawei Ascend NPU",
        runtime.memory_limit_bytes,
        1,
        20,
        Some("test-npu370".to_owned()),
        true,
    )?;
    let cancellation = CancellationToken::default();
    let memory_limit_bytes = runtime.memory_limit_bytes;
    let (backend, authority) = NpuTensorBackend::from_runtime(
        RuntimeAdapter::Test(runtime),
        device,
        1,
        properties,
        memory_limit_bytes,
        &cancellation,
    )?;
    Ok((backend, authority.authorize_workspace(memory_limit_bytes)?))
}

fn test_backend() -> Result<(NpuTensorBackend, ScratchReservation), TensorError> {
    test_backend_with_runtime(test_runtime(1024 * 1024))
}

fn descriptor(shape: Vec<u64>) -> Result<TensorDescriptor, TensorError> {
    typed_descriptor(shape, DType::F32)
}

fn typed_descriptor(shape: Vec<u64>, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        shape,
        dtype,
        DeviceId::new(DeviceKind::Npu, 0),
        StreamId::DEFAULT,
    )
}

fn offset_descriptor(
    shape: Vec<u64>,
    offset_elements: u64,
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
        DType::F32,
        Layout::Contiguous,
        DeviceId::new(DeviceKind::Npu, 0),
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
            reason: "NPU edge fixture requires f16 or f32".to_owned(),
        }),
    }
}

fn bytes_dtype(dtype: DType, values: &[u8]) -> Result<Vec<f32>, TensorError> {
    match dtype {
        DType::F16 => bytes_f16(values),
        DType::F32 => bytes_f32(values),
        _ => Err(TensorError::Faulted {
            reason: "NPU edge fixture requires f16 or f32".to_owned(),
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
fn instance_matrix_is_derived_from_the_certified_kernel_surface() -> Result<(), TensorError> {
    let (backend, _) = test_backend()?;
    assert_eq!(backend.device(), DeviceId::new(DeviceKind::Npu, 0));
    assert_eq!(backend.device_count(), 1);
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
                reason: "NPU test device properties are absent".to_owned(),
            })?;
    assert_eq!(properties.name(), "test Huawei Ascend NPU");
    assert_eq!(properties.total_memory_bytes(), 1024 * 1024);
    assert_eq!(properties.architecture(), Some("test-npu370"));
    assert!(properties.has_fp16());
    Ok(())
}

#[test]
fn certified_execution_session_harness_uses_the_production_adapter_path()
-> Result<(), TensorError> {
    let session = NpuExecutionSession::for_test_harness(
        0,
        1,
        "Ascend 910B test device",
        (8, 0, 3),
        4_096,
        2_048,
    )
    .map_err(|error| map_execution_error("zed.npu.test-harness", 0, error))?;
    let cancellation = CancellationToken::default();
    let test_control = session.clone();
    let (backend, authority) =
        NpuTensorBackend::from_certified_runtime(session, 0, 1_024, &cancellation)?;
    let scratch = authority.authorize_workspace(1_024)?;
    let execution = context(scratch, &cancellation);
    let (left, left_event) =
        backend.upload_bytes(descriptor(vec![2])?, &f32_bytes(&[1.25, -2.5]), &execution)?;
    backend.wait_event(left_event, &execution)?;
    let (right, right_event) =
        backend.upload_bytes(descriptor(vec![2])?, &f32_bytes(&[0.75, 4.0]), &execution)?;
    backend.wait_event(right_event, &execution)?;
    let (output, output_event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        descriptor(vec![2])?,
        &execution,
    )?;
    backend.wait_event(output_event, &execution)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&output, &execution)?)?,
        vec![2.0, 1.5]
    );
    assert_eq!(backend.device_count(), 1);
    let properties = backend
        .capabilities()
        .device_properties()
        .ok_or_else(|| TensorError::Faulted {
            reason: "NPU test harness matrix omitted certified properties".to_owned(),
        })?;
    assert_eq!(properties.name(), "Ascend 910B test device");
    assert_eq!(properties.total_memory_bytes(), 4_096);
    assert_eq!(properties.allocation_limit_bytes(), 1_024);

    test_control
        .fail_next_test_call_with_oom()
        .map_err(|error| map_execution_error("zed.npu.test-harness", 0, error))?;
    assert!(matches!(
        backend.allocate(descriptor(vec![1])?, &execution),
        Err(TensorError::AllocationFailed { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 24);

    test_control
        .fail_next_test_call_with_device_loss()
        .map_err(|error| map_execution_error("zed.npu.test-harness", 0, error))?;
    assert!(matches!(
        backend.allocate(descriptor(vec![1])?, &execution),
        Err(TensorError::DeviceLost { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 24);

    let injected_cancellation = CancellationToken::default();
    test_control
        .cancel_after_next_test_call(injected_cancellation.clone())
        .map_err(|error| map_execution_error("zed.npu.test-harness", 0, error))?;
    let cancelling_context = context(execution.scratch, &injected_cancellation);
    assert!(matches!(
        backend.allocate(descriptor(vec![1])?, &cancelling_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 24);
    drop(output);
    drop(right);
    drop(left);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn transfers_copy_add_and_events_preserve_f32_semantics() -> Result<(), TensorError> {
    let runtime = test_runtime(1024 * 1024);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, upload_event) = backend.upload_bytes(
        descriptor(vec![2, 2])?,
        &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        &context,
    )?;
    backend.wait_event(upload_event, &context)?;
    let (right, _) = backend.upload_bytes(
        descriptor(vec![2, 2])?,
        &f32_bytes(&[10.0, 20.0, 30.0, 40.0]),
        &context,
    )?;
    let (sum, event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        descriptor(vec![2, 2])?,
        &context,
    )?;
    backend.wait_event(event, &context)?;
    backend.wait_event(event, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&sum, &context)?)?,
        vec![11.0, 22.0, 33.0, 44.0],
    );
    let (copy, _) = backend.copy(&left, descriptor(vec![2, 2])?, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![1.0, 2.0, 3.0, 4.0],
    );
    assert!(runtime.allocation_calls.load(Ordering::Acquire) >= 4);
    assert!(runtime.synchronization_calls.load(Ordering::Acquire) >= 4);
    Ok(())
}

#[test]
fn f16_odd_scalar_empty_cancellation_and_dtype_boundaries_are_exact() -> Result<(), TensorError> {
    let runtime = test_runtime(1024);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let (left, _) = backend.upload_bytes(
        typed_descriptor(vec![3], DType::F16)?,
        &f16_bytes(&[0.5, 1.0, 2.0]),
        &live_context,
    )?;
    let (right, _) = backend.upload_bytes(
        typed_descriptor(vec![3], DType::F16)?,
        &f16_bytes(&[0.25, 2.0, 4.0]),
        &live_context,
    )?;
    let (sum, sum_event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        typed_descriptor(vec![3], DType::F16)?,
        &live_context,
    )?;
    backend.wait_event(sum_event, &live_context)?;
    assert_eq!(
        bytes_f16(&backend.download_bytes(&sum, &live_context)?)?,
        vec![0.75, 3.0, 6.0],
    );

    let (scalar_left, _) = backend.upload_bytes(
        typed_descriptor(Vec::new(), DType::F16)?,
        &f16_bytes(&[1.5]),
        &live_context,
    )?;
    let (scalar_right, _) = backend.upload_bytes(
        typed_descriptor(Vec::new(), DType::F16)?,
        &f16_bytes(&[2.25]),
        &live_context,
    )?;
    let (scalar_sum, scalar_event) = backend.binary(
        BinaryOperation::Add,
        &scalar_left,
        &scalar_right,
        typed_descriptor(Vec::new(), DType::F16)?,
        &live_context,
    )?;
    backend.wait_event(scalar_event, &live_context)?;
    assert_eq!(
        bytes_f16(&backend.download_bytes(&scalar_sum, &live_context)?)?,
        vec![3.75],
    );

    let (empty_left, _) = backend.upload_bytes(
        typed_descriptor(vec![0, 3], DType::F16)?,
        &[],
        &live_context,
    )?;
    let (empty_right, _) = backend.upload_bytes(
        typed_descriptor(vec![0, 3], DType::F16)?,
        &[],
        &live_context,
    )?;
    let (empty_sum, empty_event) = backend.binary(
        BinaryOperation::Add,
        &empty_left,
        &empty_right,
        typed_descriptor(vec![0, 3], DType::F16)?,
        &live_context,
    )?;
    backend.wait_event(empty_event, &live_context)?;
    assert!(
        backend
            .download_bytes(&empty_sum, &live_context)?
            .is_empty()
    );

    let (f32_input, _) = backend.upload_bytes(
        descriptor(vec![3])?,
        &f32_bytes(&[1.0, 2.0, 3.0]),
        &live_context,
    )?;
    let allocation_calls = runtime.allocation_calls.load(Ordering::Acquire);
    let accounted_bytes = backend.memory_snapshot().current_bytes;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &f32_input,
            typed_descriptor(vec![3], DType::F16)?,
            &live_context,
        ),
        Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: DType::F32,
        })
    ));
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &f32_input,
            &left,
            typed_descriptor(vec![3], DType::F16)?,
            &live_context,
        ),
        Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: DType::F32,
        })
    ));
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            typed_descriptor(vec![3], DType::F32)?,
            &live_context,
        ),
        Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        })
    ));
    let (short, _) = backend.upload_bytes(
        typed_descriptor(vec![2], DType::F16)?,
        &f16_bytes(&[1.0, 2.0]),
        &live_context,
    )?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &short,
            typed_descriptor(vec![2], DType::F16)?,
            &live_context,
        ),
        Err(TensorError::Faulted { reason })
            if reason == "NPU add left shape [3] does not match output shape [2]"
    ));
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &short,
            &left,
            typed_descriptor(vec![2], DType::F16)?,
            &live_context,
        ),
        Err(TensorError::Faulted { reason })
            if reason == "NPU add right shape [3] does not match output shape [2]"
    ));
    assert_eq!(
        runtime.allocation_calls.load(Ordering::Acquire),
        allocation_calls + 1
    );
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        accounted_bytes + short.descriptor().byte_len()?
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(scratch, &cancelled);
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            typed_descriptor(vec![3], DType::F16)?,
            &cancelled_context,
        ),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(
        runtime.allocation_calls.load(Ordering::Acquire),
        allocation_calls + 1
    );

    drop((
        left,
        right,
        sum,
        scalar_left,
        scalar_right,
        scalar_sum,
        empty_left,
        empty_right,
        empty_sum,
        f32_input,
        short,
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn f16_f32_odd_scalar_empty_add_alias_and_d2d_copy_are_exact() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);

    for dtype in [DType::F16, DType::F32] {
        let (odd_left, _) = backend.upload_bytes(
            typed_descriptor(vec![3], dtype)?,
            &dtype_bytes(dtype, &[0.5, 1.0, -2.0])?,
            &context,
        )?;
        let (odd_right, _) = backend.upload_bytes(
            typed_descriptor(vec![3], dtype)?,
            &dtype_bytes(dtype, &[0.25, 2.0, 4.0])?,
            &context,
        )?;
        let (odd_copy, copy_event) =
            backend.copy(&odd_left, typed_descriptor(vec![3], dtype)?, &context)?;
        backend.wait_event(copy_event, &context)?;
        assert_eq!(
            bytes_dtype(dtype, &backend.download_bytes(&odd_copy, &context)?)?,
            vec![0.5, 1.0, -2.0],
        );
        let (odd_sum, sum_event) = backend.binary(
            BinaryOperation::Add,
            &odd_left,
            &odd_right,
            typed_descriptor(vec![3], dtype)?,
            &context,
        )?;
        backend.wait_event(sum_event, &context)?;
        assert_eq!(
            bytes_dtype(dtype, &backend.download_bytes(&odd_sum, &context)?)?,
            vec![0.75, 3.0, 2.0],
        );
        let (alias_sum, alias_event) = backend.binary(
            BinaryOperation::Add,
            &odd_left,
            &odd_left,
            typed_descriptor(vec![3], dtype)?,
            &context,
        )?;
        backend.wait_event(alias_event, &context)?;
        assert_eq!(
            bytes_dtype(dtype, &backend.download_bytes(&alias_sum, &context)?)?,
            vec![1.0, 2.0, -4.0],
        );

        let (scalar_left, _) = backend.upload_bytes(
            typed_descriptor(Vec::new(), dtype)?,
            &dtype_bytes(dtype, &[1.5])?,
            &context,
        )?;
        let (scalar_right, _) = backend.upload_bytes(
            typed_descriptor(Vec::new(), dtype)?,
            &dtype_bytes(dtype, &[2.25])?,
            &context,
        )?;
        let (scalar_copy, scalar_copy_event) =
            backend.copy(&scalar_left, typed_descriptor(Vec::new(), dtype)?, &context)?;
        backend.wait_event(scalar_copy_event, &context)?;
        assert_eq!(
            bytes_dtype(dtype, &backend.download_bytes(&scalar_copy, &context)?)?,
            vec![1.5],
        );
        let (scalar_sum, scalar_event) = backend.binary(
            BinaryOperation::Add,
            &scalar_left,
            &scalar_right,
            typed_descriptor(Vec::new(), dtype)?,
            &context,
        )?;
        backend.wait_event(scalar_event, &context)?;
        assert_eq!(
            bytes_dtype(dtype, &backend.download_bytes(&scalar_sum, &context)?)?,
            vec![3.75],
        );

        let (empty_left, _) =
            backend.upload_bytes(typed_descriptor(vec![0, 3], dtype)?, &[], &context)?;
        let (empty_right, _) =
            backend.upload_bytes(typed_descriptor(vec![0, 3], dtype)?, &[], &context)?;
        let (empty_copy, empty_copy_event) =
            backend.copy(&empty_left, typed_descriptor(vec![0, 3], dtype)?, &context)?;
        backend.wait_event(empty_copy_event, &context)?;
        assert!(backend.download_bytes(&empty_copy, &context)?.is_empty());
        let (empty_sum, empty_event) = backend.binary(
            BinaryOperation::Add,
            &empty_left,
            &empty_right,
            typed_descriptor(vec![0, 3], dtype)?,
            &context,
        )?;
        backend.wait_event(empty_event, &context)?;
        assert!(backend.download_bytes(&empty_sum, &context)?.is_empty());

        drop((
            odd_left,
            odd_right,
            odd_copy,
            odd_sum,
            alias_sum,
            scalar_left,
            scalar_right,
            scalar_copy,
            scalar_sum,
            empty_left,
            empty_right,
            empty_copy,
            empty_sum,
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    }
    Ok(())
}

#[test]
fn transfers_preserve_contiguous_view_offsets_and_add_rejects_uncertified_views()
-> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (source, _) = backend.upload_bytes(
        offset_descriptor(vec![2], 2)?,
        &f32_bytes(&[7.0, 9.0]),
        &context,
    )?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&source, &context)?)?,
        vec![7.0, 9.0],
    );

    let (copy, _) = backend.copy(&source, offset_descriptor(vec![2], 1)?, &context)?;
    assert_eq!(
        bytes_f32(&backend.download_bytes(&copy, &context)?)?,
        vec![7.0, 9.0],
    );
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &source,
            &source,
            descriptor(vec![2])?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    Ok(())
}

#[test]
fn invalid_copy_sources_are_rejected_before_destination_or_native_effects()
-> Result<(), TensorError> {
    let runtime = test_runtime(1024);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch, &cancellation);

    let (base, _) = backend.upload_bytes(
        descriptor(vec![4])?,
        &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        &live_context,
    )?;
    let strided_npu_descriptor = TensorDescriptor::new_strided(
        vec![2],
        vec![2],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::new(DeviceKind::Npu, 0),
        StreamId::DEFAULT,
    )?;
    let strided_npu = base.view(strided_npu_descriptor, ViewAccess::ReadOnly)?;

    let strided_cpu_descriptor = TensorDescriptor::new_strided(
        vec![2],
        vec![2],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let strided_cpu = boundary_tensor(strided_cpu_descriptor, Some(vec![0; 12]))?;
    let inaccessible_cpu = boundary_tensor(
        TensorDescriptor::contiguous(vec![2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
        None,
    )?;
    let cross_stream_cpu = boundary_tensor(
        TensorDescriptor::contiguous(vec![2], DType::F32, DeviceId::CPU, StreamId::new(7))?,
        Some(f32_bytes(&[1.0, 2.0])),
    )?;
    let equal_bytes_wrong_shape = boundary_tensor(
        TensorDescriptor::contiguous(vec![2, 2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
        Some(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
    )?;
    let unsupported = boundary_tensor(
        TensorDescriptor::contiguous(
            vec![1],
            DType::F32,
            DeviceId::new(DeviceKind::Cuda, 0),
            StreamId::DEFAULT,
        )?,
        Some(f32_bytes(&[1.0])),
    )?;

    let (other, other_scratch) = test_backend()?;
    let other_context = context(other_scratch, &cancellation);
    let (foreign_npu, _) = other.upload_bytes(
        descriptor(vec![2])?,
        &f32_bytes(&[1.0, 2.0]),
        &other_context,
    )?;

    let allocation_calls = runtime.allocation_calls.load(Ordering::Acquire);
    let synchronization_calls = runtime.synchronization_calls.load(Ordering::Acquire);
    let memory = backend.memory_snapshot();
    assert!(matches!(
        backend.copy(&cross_stream_cpu, descriptor(vec![2])?, &live_context),
        Err(TensorError::StreamMismatch {
            expected,
            actual,
        }) if expected == StreamId::DEFAULT && actual == StreamId::new(7)
    ));
    assert!(matches!(
        backend.copy(
            &equal_bytes_wrong_shape,
            descriptor(vec![4])?,
            &live_context,
        ),
        Err(TensorError::Faulted { reason })
            if reason.contains("source shape [2, 2]")
                && reason.contains("destination shape [4]")
    ));
    for (source, destination) in [
        (&strided_npu, descriptor(vec![2])?),
        (&strided_cpu, descriptor(vec![2])?),
        (&inaccessible_cpu, descriptor(vec![2])?),
        (&unsupported, descriptor(vec![1])?),
        (&foreign_npu, descriptor(vec![2])?),
    ] {
        assert!(backend.copy(source, destination, &live_context).is_err());
        assert_eq!(
            runtime.allocation_calls.load(Ordering::Acquire),
            allocation_calls
        );
        assert_eq!(
            runtime.synchronization_calls.load(Ordering::Acquire),
            synchronization_calls
        );
        assert_eq!(
            backend.memory_snapshot().current_bytes,
            memory.current_bytes
        );
    }
    assert_eq!(
        runtime.allocation_calls.load(Ordering::Acquire),
        allocation_calls
    );
    assert_eq!(
        runtime.synchronization_calls.load(Ordering::Acquire),
        synchronization_calls
    );
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        memory.current_bytes
    );
    Ok(())
}

#[test]
fn cancellation_foreign_authority_and_foreign_storage_fail_closed() -> Result<(), TensorError> {
    let runtime = test_runtime(64);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let (tensor, _) =
        backend.upload_bytes(descriptor(vec![1])?, &f32_bytes(&[3.0]), &live_context)?;

    let (other, other_scratch) = test_backend()?;
    let foreign_context = context(other_scratch, &cancellation);
    assert!(matches!(
        backend.allocate(descriptor(vec![1])?, &foreign_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));
    assert!(matches!(
        other.download_bytes(&tensor, &live_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = context(scratch, &cancelled);
    let calls_before = runtime.allocation_calls.load(Ordering::Acquire);
    assert!(matches!(
        backend.allocate(typed_descriptor(vec![1], DType::F16)?, &cancelled_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(
        runtime.allocation_calls.load(Ordering::Acquire),
        calls_before
    );
    Ok(())
}

#[test]
fn post_synchronization_cancellation_retires_pending_event_once() -> Result<(), TensorError> {
    let runtime = test_runtime(1024);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch.clone(), &cancellation);
    let event = backend.record_event(&live_context)?;
    assert_eq!(backend.events.pending_len()?, 1);

    let cancelled_after_sync = CancellationToken::default();
    *runtime
        .cancel_after_next_synchronization
        .lock()
        .map_err(|_| TensorError::Faulted {
            reason: "NPU test synchronization cancellation lock is poisoned".to_owned(),
        })? = Some(cancelled_after_sync.clone());
    let cancelling_context = context(scratch.clone(), &cancelled_after_sync);
    assert!(matches!(
        backend.wait_event(event, &cancelling_context),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.events.pending_len()?, 0);

    let retry_cancellation = CancellationToken::default();
    let retry_context = context(scratch, &retry_cancellation);
    backend.wait_event(event, &retry_context)?;
    assert_eq!(backend.events.pending_len()?, 0);
    Ok(())
}

#[test]
fn oom_device_loss_and_workspace_failures_release_capacity() -> Result<(), TensorError> {
    let runtime = test_runtime(16);
    let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
    let cancellation = CancellationToken::default();
    let live_context = context(scratch, &cancellation);

    *runtime.failure.lock().map_err(|_| TensorError::Faulted {
        reason: "NPU test failure lock is poisoned".to_owned(),
    })? = Some(InjectedFailure::DeviceLost);
    assert!(matches!(
        backend.allocate(typed_descriptor(vec![1], DType::F16)?, &live_context),
        Err(TensorError::DeviceLost { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    *runtime.failure.lock().map_err(|_| TensorError::Faulted {
        reason: "NPU test failure lock is poisoned".to_owned(),
    })? = Some(InjectedFailure::OutOfMemory);
    assert!(matches!(
        backend.allocate(typed_descriptor(vec![1], DType::F16)?, &live_context),
        Err(TensorError::AllocationFailed { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let calls_before_capacity_rejection = runtime.allocation_calls.load(Ordering::Acquire);
    assert!(matches!(
        backend.allocate(typed_descriptor(vec![9], DType::F16)?, &live_context),
        Err(TensorError::AllocationFailed { .. })
    ));
    assert_eq!(
        runtime.allocation_calls.load(Ordering::Acquire),
        calls_before_capacity_rejection,
    );
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    assert!(matches!(
        backend.reserve_workspace(&live_context, 17),
        Err(TensorError::WorkspaceAuthorizationExceeded { .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn stream_and_event_registries_are_bounded() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    for stream in 0..MAX_NPU_STREAMS {
        backend.stream(StreamId::new(stream as u64), &cancellation)?;
    }
    assert!(matches!(
        backend.stream(StreamId::new(MAX_NPU_STREAMS as u64), &cancellation),
        Err(TensorError::ResourceLimitExceeded {
            resource: "NPU streams",
            limit: MAX_NPU_STREAMS,
        })
    ));

    let context = context(scratch, &cancellation);
    let mut events = Vec::with_capacity(MAX_NPU_PENDING_EVENTS);
    for _ in 0..MAX_NPU_PENDING_EVENTS {
        events.push(backend.record_event(&context)?);
    }
    assert!(matches!(
        backend.record_event(&context),
        Err(TensorError::ResourceLimitExceeded {
            resource: "NPU pending events",
            limit: MAX_NPU_PENDING_EVENTS,
        })
    ));
    let latest = events.last().copied().ok_or_else(|| TensorError::Faulted {
        reason: "NPU event fixture did not record events".to_owned(),
    })?;
    backend.wait_event(latest, &context)?;
    for event in events {
        backend.wait_event(event, &context)?;
    }
    assert_eq!(backend.events.pending_len()?, 0);
    Ok(())
}

#[test]
fn unsupported_operations_do_not_gain_capability_by_compilation() -> Result<(), TensorError> {
    let (backend, scratch) = test_backend()?;
    let cancellation = CancellationToken::default();
    let context = context(scratch, &cancellation);
    let (left, _) = backend.upload_bytes(descriptor(vec![1])?, &f32_bytes(&[2.0]), &context)?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Multiply,
            &left,
            &left,
            descriptor(vec![1])?,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    let rank_nine_descriptor = descriptor(vec![1, 1, 1, 1, 1, 1, 1, 1, 1])?;
    let (rank_nine, _) =
        backend.upload_bytes(rank_nine_descriptor.clone(), &f32_bytes(&[2.0]), &context)?;
    assert!(matches!(
        backend.binary(
            BinaryOperation::Add,
            &rank_nine,
            &rank_nine,
            rank_nine_descriptor,
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
    Ok(())
}
