use crate::{
    BackendCapabilityMatrix, BackendEventTracker, BackendMemoryReservation, BackendMemorySnapshot,
    BackendMemoryTracker, BackendResourceRegistry, BackendStorage, BackendWorkspaceAuthority,
    BackendWorkspaceLease, BinaryOperation, CachedAllocationOwner, CancellationToken,
    ConvolutionSpec, CustomKernelId, DType, DeviceId, EventFence, ExecutionContext, IndexSpec,
    Layout, LinearAlgebraOperation, NativeDeviceProperties, OperationSupport, PrimitiveOperation,
    ReductionSpec, ResizeSpec, Scalar, ScalarSide, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, TensorRole, UnaryOperation, ViewAccess, check_backend_context,
    check_backend_context_identity, required_storage_bytes, reserve_backend_workspace,
};
use sim_backend_directml::{
    DirectMlAllocation, DirectMlElementType, DirectMlEvent, DirectMlExecutionError,
    DirectMlExecutionSession, DirectMlStream, FILE_VERSION,
};
#[cfg(test)]
use sim_backend_directml::DirectMlTestControl;
use sim_types::DeviceKind;
use std::{fmt, sync::Arc};

const MAX_DIRECTML_STREAMS: usize = 1_024;
const MAX_DIRECTML_PENDING_EVENTS: usize = 4_096;

struct RuntimeAdapter(DirectMlExecutionSession);

impl RuntimeAdapter {
    fn allocate(
        &self,
        byte_length: u64,
        cancellation: &CancellationToken,
    ) -> Result<DirectMlAllocation, TensorError> {
        self.0
            .allocate(byte_length, &|| cancellation.is_cancelled())
            .map_err(|error| map_execution_error("zed.directml.allocate", error))
    }

    fn create_stream(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<DirectMlStream, TensorError> {
        self.0
            .create_stream(&|| cancellation.is_cancelled())
            .map_err(|error| map_execution_error("zed.directml.stream.create", error))
    }

    fn copy_from_host(
        &self,
        stream: &DirectMlStream,
        destination: &DirectMlAllocation,
        destination_offset: u64,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        self.0
            .copy_host_to_device(stream, destination, destination_offset, bytes, &|| {
                cancellation.is_cancelled()
            })
            .map_err(|error| map_execution_error("zed.directml.transfer.host-to-device", error))
    }

    fn copy_to_host(
        &self,
        stream: &DirectMlStream,
        source: &DirectMlAllocation,
        source_offset: u64,
        bytes: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        self.0
            .copy_device_to_host(stream, source, source_offset, bytes, &|| {
                cancellation.is_cancelled()
            })
            .map_err(|error| map_execution_error("zed.directml.transfer.device-to-host", error))
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        stream: &DirectMlStream,
        element_type: DirectMlElementType,
        left: &DirectMlAllocation,
        right: &DirectMlAllocation,
        output: &DirectMlAllocation,
        elements: u64,
        cancellation: &CancellationToken,
    ) -> Result<DirectMlEvent, TensorError> {
        self.0
            .dispatch_add(stream, element_type, left, right, output, elements, &|| {
                cancellation.is_cancelled()
            })
            .map_err(|error| map_execution_error("zed.directml.binary.add", error))
    }

    fn record_event(
        &self,
        stream: &DirectMlStream,
        cancellation: &CancellationToken,
    ) -> Result<DirectMlEvent, TensorError> {
        self.0
            .record_event(stream, &|| cancellation.is_cancelled())
            .map_err(|error| map_execution_error("zed.directml.event.record", error))
    }

    fn wait_event(
        &self,
        event: &DirectMlEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        self.0
            .wait_event(event, &|| cancellation.is_cancelled())
            .map_err(|error| map_execution_error("zed.directml.event.wait", error))
    }

    fn physical_memory_snapshot(&self, capacity: u64) -> BackendMemorySnapshot {
        BackendMemorySnapshot {
            limit_bytes: capacity,
            current_bytes: self.0.current_allocation_bytes(),
            peak_bytes: self.0.peak_allocation_bytes(),
        }
    }
}

struct DirectMlStorageInner {
    allocation: Option<DirectMlAllocation>,
    byte_length: u64,
    _memory: BackendMemoryReservation,
}

struct DirectMlStorage {
    backend_id: u64,
    device: DeviceId,
    inner: Arc<DirectMlStorageInner>,
}

enum PreparedCopySource {
    Cpu(Vec<u8>),
    DirectMl {
        storage: Arc<DirectMlStorageInner>,
        byte_offset: u64,
    },
}

impl fmt::Debug for DirectMlStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlStorage")
            .field("device", &self.device)
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for DirectMlStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceId {
        self.device
    }

    fn byte_len(&self) -> u64 {
        self.inner.byte_length
    }

    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError> {
        Err(TensorError::NonHostStorage)
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        None
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        None
    }
}

pub struct DirectMlTensorBackend {
    runtime: RuntimeAdapter,
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    logical_memory: Arc<BackendMemoryTracker>,
    physical_capacity_bytes: u64,
    streams: BackendResourceRegistry<DirectMlStream>,
    events: BackendEventTracker<DirectMlEvent>,
}

impl DirectMlTensorBackend {
    pub fn from_certified_session(
        session: DirectMlExecutionSession,
        requested_memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let vendor = session.properties();
        let physical_total = vendor
            .dedicated_memory_bytes()
            .checked_add(vendor.shared_memory_bytes())
            .ok_or(TensorError::ShapeOverflow)?;
        let physical_capacity_bytes = vendor.allocation_capacity_bytes();
        let effective_limit = requested_memory_limit_bytes.min(physical_capacity_bytes);
        let device = DeviceId::new(DeviceKind::DirectMl, 0);
        let properties = NativeDeviceProperties::new_with_allocation_limit(
            device,
            vendor.name(),
            physical_total,
            physical_capacity_bytes,
            u32::from(FILE_VERSION.major),
            u32::from(FILE_VERSION.minor),
            Some(format!("DXGI adapter LUID {:#018x}", vendor.adapter_luid())),
            vendor.has_fp16(),
        )?;
        Self::from_runtime(
            RuntimeAdapter(session),
            properties,
            effective_limit,
            physical_capacity_bytes,
            cancellation,
        )
    }

    fn from_runtime(
        runtime: RuntimeAdapter,
        properties: NativeDeviceProperties,
        effective_memory_limit_bytes: u64,
        physical_capacity_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let device = DeviceId::new(DeviceKind::DirectMl, 0);
        if properties.device() != device {
            return Err(TensorError::DeviceMismatch {
                expected: device,
                actual: properties.device(),
            });
        }
        if effective_memory_limit_bytes == 0 || physical_capacity_bytes == 0 {
            return Err(TensorError::AllocationFailed {
                requested: effective_memory_limit_bytes,
                reason: "DirectML logical and physical capacities must both be nonzero".to_owned(),
            });
        }
        if effective_memory_limit_bytes > physical_capacity_bytes {
            return Err(TensorError::AllocationFailed {
                requested: effective_memory_limit_bytes,
                reason: format!(
                    "DirectML effective logical limit exceeds physical allocation capacity {physical_capacity_bytes}"
                ),
            });
        }
        let capabilities = directml_capability_matrix(device, properties)?;
        let (backend_id, logical_memory, authority) =
            BackendWorkspaceAuthority::new(effective_memory_limit_bytes)?;
        Ok((
            Self {
                runtime,
                device,
                capabilities,
                backend_id,
                logical_memory,
                physical_capacity_bytes,
                streams: BackendResourceRegistry::new("DirectML streams", MAX_DIRECTML_STREAMS),
                events: BackendEventTracker::new(
                    "DirectML pending events",
                    MAX_DIRECTML_PENDING_EVENTS,
                ),
            },
            authority,
        ))
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.logical_memory.snapshot()
    }

    pub fn physical_memory_snapshot(&self) -> BackendMemorySnapshot {
        self.runtime
            .physical_memory_snapshot(self.physical_capacity_bytes)
    }

    pub fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        reserve_backend_workspace(
            self.backend_id,
            &self.logical_memory,
            context,
            requested,
            requested,
        )
    }

    pub fn upload_bytes(
        &self,
        descriptor: TensorDescriptor,
        bytes: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.directml.transfer.host-to-device",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        let expected = descriptor.byte_len()?;
        let actual = u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
        if actual != expected {
            return Err(TensorError::StorageLength { expected, actual });
        }
        let stream = self.stream(context.stream, context.cancellation)?;
        let memory = self.reserve_tensor_storage(&descriptor)?;
        let storage_bytes = required_storage_bytes(&descriptor)?;
        let destination_offset = tensor_byte_offset(&descriptor)?;
        let allocation = if storage_bytes == 0 {
            None
        } else {
            let allocation = self.runtime.allocate(storage_bytes, context.cancellation)?;
            if destination_offset != 0 {
                let storage_length =
                    usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
                let zeros = try_zeroed_bytes(storage_length, "DirectML upload gap initialization")?;
                self.runtime.copy_from_host(
                    &stream,
                    &allocation,
                    0,
                    &zeros,
                    context.cancellation,
                )?;
            }
            if !bytes.is_empty() {
                self.runtime.copy_from_host(
                    &stream,
                    &allocation,
                    destination_offset,
                    bytes,
                    context.cancellation,
                )?;
            }
            Some(allocation)
        };
        self.check_context(context)?;
        let tensor = self.tensor_from_allocation(descriptor, allocation, memory)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    pub fn download_bytes(
        &self,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<u8>, TensorError> {
        self.require_input("zed.directml.transfer.device-to-host", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = tensor.descriptor().byte_len()?;
        let staging = self.reserve_workspace(context, byte_length)?;
        let byte_length = usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(byte_length, "DirectML download staging")?;
        if let Some(allocation) = &storage.allocation {
            let stream = self.stream(context.stream, context.cancellation)?;
            self.runtime.copy_to_host(
                &stream,
                allocation,
                tensor_byte_offset(tensor.descriptor())?,
                &mut bytes,
                context.cancellation,
            )?;
        }
        self.check_context(context)?;
        drop(staging);
        Ok(bytes)
    }

    #[cfg(test)]
    fn download_storage_bytes(
        &self,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<u8>, TensorError> {
        self.require_input("zed.directml.test.download-storage", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = usize::try_from(storage.byte_length)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(byte_length, "DirectML test full-storage download")?;
        if let Some(allocation) = &storage.allocation {
            let stream = self.stream(context.stream, context.cancellation)?;
            self.runtime
                .copy_to_host(&stream, allocation, 0, &mut bytes, context.cancellation)?;
        }
        self.check_context(context)?;
        Ok(bytes)
    }

    fn check_context(&self, context: &ExecutionContext<'_>) -> Result<(), TensorError> {
        check_backend_context(self.backend_id, context)
    }

    fn stream(
        &self,
        stream: StreamId,
        cancellation: &CancellationToken,
    ) -> Result<DirectMlStream, TensorError> {
        self.streams
            .get_or_try_insert_with(stream, || self.runtime.create_stream(cancellation))
    }

    fn tensor_from_allocation(
        &self,
        descriptor: TensorDescriptor,
        allocation: Option<DirectMlAllocation>,
        memory: BackendMemoryReservation,
    ) -> Result<Tensor, TensorError> {
        let byte_length = required_storage_bytes(&descriptor)?;
        let actual = allocation
            .as_ref()
            .map_or(0, DirectMlAllocation::byte_length);
        if actual != byte_length {
            return Err(TensorError::StorageLength {
                expected: byte_length,
                actual,
            });
        }
        Tensor::from_backend_storage(
            descriptor,
            Box::new(DirectMlStorage {
                backend_id: self.backend_id,
                device: self.device,
                inner: Arc::new(DirectMlStorageInner {
                    allocation,
                    byte_length,
                    _memory: memory,
                }),
            }),
            ViewAccess::Writable,
        )
    }

    fn reserve_tensor_storage(
        &self,
        descriptor: &TensorDescriptor,
    ) -> Result<BackendMemoryReservation, TensorError> {
        self.logical_memory
            .reserve(required_storage_bytes(descriptor)?)
    }

    fn storage(&self, tensor: &Tensor) -> Result<Arc<DirectMlStorageInner>, TensorError> {
        tensor
            .backend_storage::<DirectMlStorage>()
            .filter(|storage| storage.backend_id == self.backend_id)
            .map(|storage| storage.inner.clone())
            .ok_or_else(|| TensorError::UnsupportedCapability {
                operation: "zed.directml.storage.lookup".to_owned(),
                device: tensor.descriptor().device(),
                reason: "tensor storage is not owned by this certified DirectML backend instance"
                    .to_owned(),
            })
    }

    fn require_descriptor(
        &self,
        operation: &str,
        primitive: PrimitiveOperation,
        role: TensorRole,
        descriptor: &TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.check_context(context)?;
        if descriptor.device() != self.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.device,
                actual: descriptor.device(),
            });
        }
        if descriptor.stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: descriptor.stream(),
            });
        }
        if !matches!(descriptor.dtype(), DType::F16 | DType::F32)
            || descriptor.layout() != Layout::Contiguous
        {
            return Err(self.unsupported(
                operation,
                "the reviewed DirectML baseline accepts contiguous f16/f32 tensors only",
            ));
        }
        if !descriptor.is_contiguous()? {
            return Err(self.unsupported(
                operation,
                "the descriptor does not have canonical contiguous strides",
            ));
        }
        self.capabilities.require(
            operation,
            OperationSupport::for_tensor(primitive, role, descriptor.dtype(), descriptor.layout())?,
        )
    }

    fn require_input(
        &self,
        operation: &str,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.require_descriptor(
            operation,
            PrimitiveOperation::Copy,
            TensorRole::Input,
            tensor.descriptor(),
            context,
        )?;
        self.storage(tensor)?;
        Ok(())
    }

    fn prepare_copy_source(
        &self,
        source: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<PreparedCopySource, TensorError> {
        if source.descriptor().stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: source.descriptor().stream(),
            });
        }
        if source.descriptor().device() == self.device {
            self.require_input("zed.directml.copy", source, context)?;
            return Ok(PreparedCopySource::DirectMl {
                storage: self.storage(source)?,
                byte_offset: tensor_byte_offset(source.descriptor())?,
            });
        }
        if source.descriptor().device().kind() == DeviceKind::Cpu {
            let bytes = source.contiguous_bytes()?;
            return try_copy_bytes(bytes, "DirectML CPU copy staging").map(PreparedCopySource::Cpu);
        }
        Err(self.unsupported(
            "zed.directml.copy",
            "source must be host-addressable contiguous CPU storage or this DirectML backend instance",
        ))
    }

    fn track_event(
        &self,
        context: &ExecutionContext<'_>,
        create: impl FnOnce() -> Result<DirectMlEvent, TensorError>,
    ) -> Result<EventFence, TensorError> {
        let sequence = self.events.record_with(context.stream, create)?;
        if let Err(error) = self.check_context(context) {
            drop(self.events.cancel(context.stream, sequence)?);
            return Err(error);
        }
        Ok(EventFence {
            backend_id: self.backend_id,
            device: self.device,
            stream: context.stream,
            sequence,
        })
    }

    fn unsupported(&self, operation: &str, reason: impl Into<String>) -> TensorError {
        TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: self.device,
            reason: reason.into(),
        }
    }

    fn unsupported_result<T>(
        &self,
        operation: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<T, TensorError> {
        self.check_context(context)?;
        Err(self.unsupported(operation, "no reviewed DirectML kernel is registered"))
    }
}

impl CachedAllocationOwner for DirectMlTensorBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn allocator_backend_name(&self) -> &'static str {
        "zed-native-directml-v1"
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for DirectMlTensorBackend {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn capabilities(&self) -> &BackendCapabilityMatrix {
        &self.capabilities
    }

    fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        DirectMlTensorBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.directml.allocate",
            PrimitiveOperation::Allocation,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        let stream = self.stream(context.stream, context.cancellation)?;
        let byte_length = required_storage_bytes(&descriptor)?;
        let memory = self.reserve_tensor_storage(&descriptor)?;
        let allocation = if byte_length == 0 {
            None
        } else {
            let allocation = self.runtime.allocate(byte_length, context.cancellation)?;
            let byte_length =
                usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
            let bytes = try_zeroed_bytes(byte_length, "DirectML zero initialization")?;
            self.runtime
                .copy_from_host(&stream, &allocation, 0, &bytes, context.cancellation)?;
            Some(allocation)
        };
        self.check_context(context)?;
        let tensor = self.tensor_from_allocation(descriptor, allocation, memory)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn copy(
        &self,
        source: &Tensor,
        destination: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.directml.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        if source.descriptor().shape() != destination.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "zed.directml.copy: source shape {:?} does not match destination shape {:?}",
                    source.descriptor().shape(),
                    destination.shape()
                ),
            });
        }
        if source.descriptor().dtype() != destination.dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: destination.dtype(),
                actual: source.descriptor().dtype(),
            });
        }
        let source = self.prepare_copy_source(source, context)?;
        self.check_context(context)?;
        let stream = self.stream(context.stream, context.cancellation)?;
        let byte_length = destination.byte_len()?;
        let storage_bytes = required_storage_bytes(&destination)?;
        let destination_offset = tensor_byte_offset(&destination)?;
        let memory = self.reserve_tensor_storage(&destination)?;
        let allocation = if storage_bytes == 0 {
            None
        } else {
            let allocation = self.runtime.allocate(storage_bytes, context.cancellation)?;
            let storage_length =
                usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
            let zeros = try_zeroed_bytes(storage_length, "DirectML copy gap initialization")?;
            self.runtime
                .copy_from_host(&stream, &allocation, 0, &zeros, context.cancellation)?;
            Some(allocation)
        };
        match source {
            PreparedCopySource::DirectMl {
                storage,
                byte_offset,
            } => {
                if let (Some(destination), Some(source)) = (&allocation, &storage.allocation) {
                    let staging = self.reserve_workspace(context, byte_length)?;
                    let byte_length =
                        usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
                    let mut bytes = try_zeroed_bytes(byte_length, "DirectML copy staging")?;
                    self.runtime.copy_to_host(
                        &stream,
                        source,
                        byte_offset,
                        &mut bytes,
                        context.cancellation,
                    )?;
                    self.runtime.copy_from_host(
                        &stream,
                        destination,
                        destination_offset,
                        &bytes,
                        context.cancellation,
                    )?;
                    drop(staging);
                }
            }
            PreparedCopySource::Cpu(bytes) => {
                if let Some(destination) = &allocation {
                    self.runtime.copy_from_host(
                        &stream,
                        destination,
                        destination_offset,
                        &bytes,
                        context.cancellation,
                    )?;
                }
            }
        }
        self.check_context(context)?;
        let tensor = self.tensor_from_allocation(destination, allocation, memory)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        self.check_context(context)?;
        self.capabilities.require(
            "zed.directml.event.record",
            OperationSupport::record_event(),
        )?;
        let stream = self.stream(context.stream, context.cancellation)?;
        self.track_event(context, || {
            self.runtime.record_event(&stream, context.cancellation)
        })
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        check_backend_context_identity(self.backend_id, context)?;
        self.capabilities
            .require("zed.directml.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "DirectML event belongs to a different backend instance".to_owned(),
            });
        }
        if event.device != self.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.device,
                actual: event.device,
            });
        }
        if event.stream != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: event.stream,
            });
        }
        let Some(native_event) = self.events.event_for_wait(event.stream, event.sequence)? else {
            return self.check_context(context);
        };
        let wait_result = self.runtime.wait_event(&native_event, context.cancellation);
        drop(self.events.complete(event.stream, event.sequence)?);
        wait_result?;
        self.check_context(context)
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.fill", context)
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.unary", context)
    }

    fn binary(
        &self,
        operation: BinaryOperation,
        left: &Tensor,
        right: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        if operation != BinaryOperation::Add {
            return self.unsupported_result("zed.directml.binary", context);
        }
        for input in [left, right] {
            self.require_descriptor(
                "zed.directml.binary.add",
                PrimitiveOperation::Binary(BinaryOperation::Add),
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
            self.storage(input)?;
        }
        self.require_descriptor(
            "zed.directml.binary.add",
            PrimitiveOperation::Binary(BinaryOperation::Add),
            TensorRole::Output,
            &output,
            context,
        )?;
        if left.descriptor().shape() != right.descriptor().shape()
            || left.descriptor().shape() != output.shape()
        {
            return Err(TensorError::Faulted {
                reason: "DirectML add requires identical input and output shapes".to_owned(),
            });
        }
        if left.descriptor().dtype() != output.dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: output.dtype(),
                actual: left.descriptor().dtype(),
            });
        }
        if right.descriptor().dtype() != output.dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: output.dtype(),
                actual: right.descriptor().dtype(),
            });
        }
        if left.descriptor().offset_elements() != 0
            || right.descriptor().offset_elements() != 0
            || output.offset_elements() != 0
        {
            return Err(self.unsupported(
                "zed.directml.binary.add",
                "the reviewed DirectML Add binding requires zero-offset whole buffers",
            ));
        }
        let left = self.storage(left)?;
        let right = self.storage(right)?;
        let memory = self.reserve_tensor_storage(&output)?;
        let elements = output.element_count()?;
        if elements == 0 {
            let tensor = self.tensor_from_allocation(output, None, memory)?;
            let event = self.record_event(context)?;
            return Ok((tensor, event));
        }
        let byte_length = output.byte_len()?;
        let output_allocation = self.runtime.allocate(byte_length, context.cancellation)?;
        let left_allocation = left
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty DirectML add left input has no allocation".to_owned(),
            })?;
        let right_allocation = right
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty DirectML add right input has no allocation".to_owned(),
            })?;
        let element_type = match output.dtype() {
            DType::F16 => DirectMlElementType::F16,
            DType::F32 => DirectMlElementType::F32,
            _ => {
                return Err(self.unsupported(
                    "zed.directml.binary.add",
                    "only reviewed f16/f32 DirectML Add kernels are available",
                ));
            }
        };
        let stream = self.stream(context.stream, context.cancellation)?;
        let event = self.track_event(context, || {
            self.runtime.add(
                &stream,
                element_type,
                left_allocation,
                right_allocation,
                &output_allocation,
                elements,
                context.cancellation,
            )
        })?;
        let tensor = self.tensor_from_allocation(output, Some(output_allocation), memory)?;
        Ok((tensor, event))
    }

    fn binary_scalar(
        &self,
        _operation: BinaryOperation,
        _input: &Tensor,
        _scalar: Scalar,
        _scalar_side: ScalarSide,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.binary-scalar", context)
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.reduction", context)
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.indexing", context)
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.resize", context)
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.convolution", context)
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.directml.linear-algebra", context)
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.unsupported_result("zed.directml.custom-kernel", context)
    }
}

fn directml_capability_matrix(
    device: DeviceId,
    properties: NativeDeviceProperties,
) -> Result<BackendCapabilityMatrix, TensorError> {
    if device.kind() != DeviceKind::DirectMl {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::new(DeviceKind::DirectMl, device.ordinal()),
            actual: device,
        });
    }
    let mut supported = Vec::with_capacity(if properties.has_fp16() { 12 } else { 7 });
    for dtype in [DType::F32]
        .into_iter()
        .chain(properties.has_fp16().then_some(DType::F16))
    {
        supported.extend([
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
            OperationSupport::binary_input(BinaryOperation::Add, dtype, Layout::Contiguous),
            OperationSupport::binary_output(BinaryOperation::Add, dtype, Layout::Contiguous),
        ]);
    }
    supported.extend([
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ]);
    BackendCapabilityMatrix::new_with_properties(
        device,
        supported.clone(),
        supported,
        Some(properties),
    )
}

fn tensor_byte_offset(descriptor: &TensorDescriptor) -> Result<u64, TensorError> {
    descriptor
        .offset_elements()
        .checked_mul(descriptor.dtype().byte_width())
        .ok_or(TensorError::ShapeOverflow)
}

fn map_execution_error(operation: &str, error: DirectMlExecutionError) -> TensorError {
    let device = DeviceId::new(DeviceKind::DirectMl, 0);
    match error {
        DirectMlExecutionError::UnsupportedTarget { target } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!(
                    "DirectML is unavailable on target {target}; a registry-certified Windows session is required"
                ),
            }
        }
        DirectMlExecutionError::InvalidCertifiedInputs { reason } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!("certified DirectML session inputs are invalid: {reason}"),
            }
        }
        DirectMlExecutionError::Cancelled => TensorError::Cancelled,
        DirectMlExecutionError::OutOfMemory {
            requested,
            capacity,
        } => TensorError::AllocationFailed {
            requested,
            reason: format!(
                "{operation}: DirectML physical allocation capacity is {capacity} bytes"
            ),
        },
        DirectMlExecutionError::DeviceLost { status } => TensorError::DeviceLost {
            reason: format!("{operation}: DirectML device lost with HRESULT {status:#x}"),
        },
        DirectMlExecutionError::UnsupportedElementType { element_type } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!("DirectML element type {element_type:?} is not supported"),
            }
        }
        DirectMlExecutionError::StreamLimit { limit }
        | DirectMlExecutionError::EventLimit { limit } => TensorError::ResourceLimitExceeded {
            resource: "DirectML physical resources",
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
        },
        other => TensorError::Faulted {
            reason: format!("{operation}: {other}"),
        },
    }
}

fn try_zeroed_bytes(byte_length: usize, purpose: &str) -> Result<Vec<u8>, TensorError> {
    let requested = u64::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_length)
        .map_err(|error| TensorError::AllocationFailed {
            requested,
            reason: format!("{purpose} allocation failed: {error}"),
        })?;
    bytes.resize(byte_length, 0);
    Ok(bytes)
}

fn try_copy_bytes(bytes: &[u8], purpose: &str) -> Result<Vec<u8>, TensorError> {
    let mut copy = try_zeroed_bytes(bytes.len(), purpose)?;
    copy.copy_from_slice(bytes);
    Ok(copy)
}

#[cfg(test)]
#[path = "../../tests/backends/directml_sim_model_0018.rs"]
mod tests;
