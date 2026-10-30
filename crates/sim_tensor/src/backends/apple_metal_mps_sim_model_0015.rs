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
use sim_backend_metal::{
    MetalAllocation, MetalElementType, MetalEvent, MetalExecutionError, MetalRuntime, MetalStream,
};
use sim_types::DeviceKind;
use std::{fmt, sync::Arc};
#[cfg(test)]
use std::sync::Mutex;

const MAX_METAL_STREAMS: usize = 1_024;
const MAX_METAL_PENDING_EVENTS: usize = 4_096;

struct MetalStorageInner {
    allocation: Option<MetalAllocation>,
    byte_length: u64,
    _memory: BackendMemoryReservation,
}

struct MetalStorage {
    backend_id: u64,
    device: DeviceId,
    inner: Arc<MetalStorageInner>,
}

impl fmt::Debug for MetalStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalStorage")
            .field("device", &self.device)
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for MetalStorage {
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

enum PreparedCopySource {
    Cpu {
        bytes: Vec<u8>,
        _staging: BackendWorkspaceLease,
    },
    Metal {
        storage: Arc<MetalStorageInner>,
        byte_offset: u64,
    },
}

pub struct MetalTensorBackend {
    runtime: MetalRuntime,
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    logical_memory: Arc<BackendMemoryTracker>,
    physical_capacity_bytes: u64,
    streams: BackendResourceRegistry<MetalStream>,
    events: BackendEventTracker<MetalEvent>,
    #[cfg(test)]
    post_native_event_cancellation: Mutex<Option<CancellationToken>>,
}

impl MetalTensorBackend {
    pub fn from_certified_runtime(
        runtime: MetalRuntime,
        host_physical_memory_bytes: u64,
        requested_memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let vendor = runtime.properties();
        let capacity = vendor.recommended_working_set_bytes();
        if capacity == 0 || host_physical_memory_bytes == 0 {
            return Err(TensorError::AllocationFailed {
                requested: requested_memory_limit_bytes,
                reason: "Metal host physical memory and recommended working set must be nonzero"
                    .to_owned(),
            });
        }
        let device = DeviceId::new(DeviceKind::Metal, 0);
        let properties = NativeDeviceProperties::new_with_allocation_limit(
            device,
            vendor.name(),
            host_physical_memory_bytes,
            capacity,
            3,
            0,
            Some(format!(
                "Metal registry {:#018x}, {:?} storage",
                vendor.registry_id(),
                vendor.storage_mode()
            )),
            true,
        )?;
        let effective_limit = requested_memory_limit_bytes.min(capacity);
        if effective_limit == 0 {
            return Err(TensorError::AllocationFailed {
                requested: requested_memory_limit_bytes,
                reason: "Metal effective memory limit must be nonzero".to_owned(),
            });
        }
        let capabilities = metal_capability_matrix(device, properties)?;
        let (backend_id, logical_memory, authority) =
            BackendWorkspaceAuthority::new(effective_limit)?;
        Ok((
            Self {
                runtime,
                device,
                capabilities,
                backend_id,
                logical_memory,
                physical_capacity_bytes: capacity,
                streams: BackendResourceRegistry::new("Metal streams", MAX_METAL_STREAMS),
                events: BackendEventTracker::new(
                    "Metal pending events",
                    MAX_METAL_PENDING_EVENTS,
                ),
                #[cfg(test)]
                post_native_event_cancellation: Mutex::new(None),
            },
            authority,
        ))
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.logical_memory.snapshot()
    }

    pub fn physical_memory_snapshot(&self) -> BackendMemorySnapshot {
        BackendMemorySnapshot {
            limit_bytes: self.physical_capacity_bytes,
            current_bytes: self.runtime.current_allocation_bytes(),
            peak_bytes: self.runtime.peak_allocation_bytes(),
        }
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
            "zed.metal.transfer.host-to-device",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        let expected = descriptor.byte_len()?;
        let actual = u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
        if expected != actual {
            return Err(TensorError::StorageLength { expected, actual });
        }
        let stream = self.stream(context.stream, context.cancellation)?;
        let memory = self.reserve_tensor_storage(&descriptor)?;
        let storage_bytes = required_storage_bytes(&descriptor)?;
        let offset = tensor_byte_offset(&descriptor)?;
        let allocation = if storage_bytes == 0 {
            None
        } else {
            let allocation = self.allocate_native(storage_bytes, context.cancellation)?;
            if offset != 0 {
                let length =
                    usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
                let zeros = try_zeroed_bytes(length, "Metal upload gap initialization")?;
                self.copy_from_host(&stream, &allocation, 0, &zeros, context.cancellation)?;
            }
            if !bytes.is_empty() {
                self.copy_from_host(&stream, &allocation, offset, bytes, context.cancellation)?;
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
        self.require_input("zed.metal.transfer.device-to-host", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = tensor.descriptor().byte_len()?;
        let staging = self.reserve_workspace(context, byte_length)?;
        let length = usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(length, "Metal download staging")?;
        if let Some(allocation) = &storage.allocation {
            let stream = self.stream(context.stream, context.cancellation)?;
            self.copy_to_host(
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

    fn check_context(&self, context: &ExecutionContext<'_>) -> Result<(), TensorError> {
        check_backend_context(self.backend_id, context)
    }

    fn allocate_native(
        &self,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<MetalAllocation, TensorError> {
        cancellation.check()?;
        let allocation = self
            .runtime
            .allocate(bytes)
            .map_err(|error| map_execution_error("zed.metal.allocate", error))?;
        cancellation.check()?;
        Ok(allocation)
    }

    fn stream(
        &self,
        stream: StreamId,
        cancellation: &CancellationToken,
    ) -> Result<MetalStream, TensorError> {
        cancellation.check()?;
        self.streams.get_or_try_insert_with(stream, || {
            self.runtime
                .create_stream()
                .map_err(|error| map_execution_error("zed.metal.stream.create", error))
        })
    }

    fn copy_from_host(
        &self,
        stream: &MetalStream,
        destination: &MetalAllocation,
        offset: u64,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        cancellation.check()?;
        self.runtime
            .copy_host_to_device(stream, destination, offset, bytes)
            .map_err(|error| map_execution_error("zed.metal.transfer.host-to-device", error))?;
        cancellation.check()?;
        Ok(())
    }

    fn copy_to_host(
        &self,
        stream: &MetalStream,
        source: &MetalAllocation,
        offset: u64,
        bytes: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        cancellation.check()?;
        self.runtime
            .copy_device_to_host(stream, source, offset, bytes)
            .map_err(|error| map_execution_error("zed.metal.transfer.device-to-host", error))?;
        cancellation.check()?;
        Ok(())
    }

    fn tensor_from_allocation(
        &self,
        descriptor: TensorDescriptor,
        allocation: Option<MetalAllocation>,
        memory: BackendMemoryReservation,
    ) -> Result<Tensor, TensorError> {
        let expected = required_storage_bytes(&descriptor)?;
        let actual = allocation
            .as_ref()
            .map_or(0, MetalAllocation::byte_length);
        if expected != actual {
            return Err(TensorError::StorageLength { expected, actual });
        }
        Tensor::from_backend_storage(
            descriptor,
            Box::new(MetalStorage {
                backend_id: self.backend_id,
                device: self.device,
                inner: Arc::new(MetalStorageInner {
                    allocation,
                    byte_length: expected,
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

    fn storage(&self, tensor: &Tensor) -> Result<Arc<MetalStorageInner>, TensorError> {
        tensor
            .backend_storage::<MetalStorage>()
            .filter(|storage| storage.backend_id == self.backend_id)
            .map(|storage| storage.inner.clone())
            .ok_or_else(|| self.unsupported(
                "zed.metal.storage.lookup",
                "tensor storage is not owned by this certified Metal backend instance",
            ))
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
            || !descriptor.is_contiguous()?
        {
            return Err(self.unsupported(
                operation,
                "the reviewed Metal ABI accepts canonical contiguous f16/f32 tensors only",
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
            self.require_input("zed.metal.copy", source, context)?;
            return Ok(PreparedCopySource::Metal {
                storage: self.storage(source)?,
                byte_offset: tensor_byte_offset(source.descriptor())?,
            });
        }
        if source.descriptor().device().kind() == DeviceKind::Cpu {
            if source.descriptor().layout() != Layout::Contiguous
                || !source.descriptor().is_contiguous()?
            {
                return Err(self.unsupported(
                    "zed.metal.copy",
                    "CPU copy source must use canonical contiguous layout",
                ));
            }
            let source_bytes = source.contiguous_bytes()?;
            let staging = self.reserve_workspace(context, source.descriptor().byte_len()?)?;
            let bytes = try_copy_bytes(source_bytes, "Metal CPU copy staging")?;
            return Ok(PreparedCopySource::Cpu {
                bytes,
                _staging: staging,
            });
        }
        Err(self.unsupported(
            "zed.metal.copy",
            "source must be contiguous CPU storage or this Metal backend instance",
        ))
    }

    fn record_native_event(
        &self,
        context: &ExecutionContext<'_>,
        create: impl FnOnce() -> Result<MetalEvent, TensorError>,
    ) -> Result<EventFence, TensorError> {
        let sequence = self.events.record_with(context.stream, create)?;
        #[cfg(test)]
        if let Some(cancellation) = self
            .post_native_event_cancellation
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "Metal post-native-event cancellation hook is poisoned".to_owned(),
            })?
            .take()
        {
            cancellation.cancel();
        }
        if let Err(error) = self.check_context(context) {
            if let Some(event) = self.events.event_for_wait(context.stream, sequence)? {
                let wait_result = self
                    .runtime
                    .wait_event(&event)
                    .map_err(|wait_error| map_execution_error("zed.metal.event.wait", wait_error));
                drop(self.events.complete(context.stream, sequence)?);
                match wait_result {
                    Ok(()) => {}
                    Err(cleanup_error) => {
                        if !matches!(error, TensorError::Cancelled) {
                            return Err(cleanup_error);
                        }
                        drop(cleanup_error);
                    }
                }
            }
            return Err(error);
        }
        Ok(EventFence {
            backend_id: self.backend_id,
            device: self.device,
            stream: context.stream,
            sequence,
        })
    }

    #[cfg(test)]
    fn cancel_after_next_native_event(
        &self,
        cancellation: CancellationToken,
    ) -> Result<(), TensorError> {
        *self
            .post_native_event_cancellation
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "Metal post-native-event cancellation hook is poisoned".to_owned(),
            })? = Some(cancellation);
        Ok(())
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
        Err(self.unsupported(operation, "no reviewed Metal kernel is registered"))
    }
}

impl CachedAllocationOwner for MetalTensorBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn allocator_backend_name(&self) -> &'static str {
        "zed-native-metal-v1"
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for MetalTensorBackend {
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
        MetalTensorBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.metal.allocate",
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
            let allocation = self.allocate_native(byte_length, context.cancellation)?;
            let length = usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
            let zeros = try_zeroed_bytes(length, "Metal zero initialization")?;
            self.copy_from_host(&stream, &allocation, 0, &zeros, context.cancellation)?;
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
            "zed.metal.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        if source.descriptor().shape() != destination.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "zed.metal.copy: source shape {:?} does not match destination shape {:?}",
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
            let allocation = self.allocate_native(storage_bytes, context.cancellation)?;
            let length = usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
            let zeros = try_zeroed_bytes(length, "Metal copy gap initialization")?;
            self.copy_from_host(&stream, &allocation, 0, &zeros, context.cancellation)?;
            Some(allocation)
        };
        match source {
            PreparedCopySource::Cpu {
                bytes,
                _staging: staging,
            } => {
                if let Some(destination) = &allocation {
                    self.copy_from_host(
                        &stream,
                        destination,
                        destination_offset,
                        &bytes,
                        context.cancellation,
                    )?;
                }
                drop(staging);
            }
            PreparedCopySource::Metal {
                storage,
                byte_offset,
            } => {
                if let (Some(source), Some(destination)) = (&storage.allocation, &allocation) {
                    let staging = self.reserve_workspace(context, byte_length)?;
                    let length =
                        usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
                    let mut bytes = try_zeroed_bytes(length, "Metal device copy staging")?;
                    self.copy_to_host(
                        &stream,
                        source,
                        byte_offset,
                        &mut bytes,
                        context.cancellation,
                    )?;
                    self.copy_from_host(
                        &stream,
                        destination,
                        destination_offset,
                        &bytes,
                        context.cancellation,
                    )?;
                    drop(staging);
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
        self.capabilities
            .require("zed.metal.event.record", OperationSupport::record_event())?;
        let stream = self.stream(context.stream, context.cancellation)?;
        self.record_native_event(context, || {
            self.runtime
                .record_event(&stream)
                .map_err(|error| map_execution_error("zed.metal.event.record", error))
        })
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        check_backend_context_identity(self.backend_id, context)?;
        self.capabilities
            .require("zed.metal.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "Metal event belongs to a different backend instance".to_owned(),
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
        let result = self
            .runtime
            .wait_event(&native_event)
            .map_err(|error| map_execution_error("zed.metal.event.wait", error));
        drop(self.events.complete(event.stream, event.sequence)?);
        if context.cancellation.is_cancelled() {
            return Err(TensorError::Cancelled);
        }
        result?;
        Ok(())
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.fill", context)
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.unary", context)
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
            return self.unsupported_result("zed.metal.binary", context);
        }
        for input in [left, right] {
            self.require_descriptor(
                "zed.metal.binary.add",
                PrimitiveOperation::Binary(BinaryOperation::Add),
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
            self.storage(input)?;
        }
        self.require_descriptor(
            "zed.metal.binary.add",
            PrimitiveOperation::Binary(BinaryOperation::Add),
            TensorRole::Output,
            &output,
            context,
        )?;
        if left.descriptor().shape() != right.descriptor().shape()
            || left.descriptor().shape() != output.shape()
        {
            return Err(TensorError::Faulted {
                reason: format!(
                    "Metal Add shape mismatch: left {:?}, right {:?}, output {:?}",
                    left.descriptor().shape(),
                    right.descriptor().shape(),
                    output.shape()
                ),
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
                "zed.metal.binary.add",
                "the reviewed Metal Add ABI requires zero-offset whole buffers",
            ));
        }
        let left = self.storage(left)?;
        let right = self.storage(right)?;
        let stream = self.stream(context.stream, context.cancellation)?;
        let memory = self.reserve_tensor_storage(&output)?;
        let elements = output.element_count()?;
        if elements == 0 {
            let tensor = self.tensor_from_allocation(output, None, memory)?;
            let event = self.record_event(context)?;
            return Ok((tensor, event));
        }
        let left = left.allocation.as_ref().ok_or_else(|| TensorError::Faulted {
            reason: "nonempty Metal Add left input has no allocation".to_owned(),
        })?;
        let right = right.allocation.as_ref().ok_or_else(|| TensorError::Faulted {
            reason: "nonempty Metal Add right input has no allocation".to_owned(),
        })?;
        let element_type = match output.dtype() {
            DType::F16 => MetalElementType::F16,
            DType::F32 => MetalElementType::F32,
            _ => return Err(self.unsupported("zed.metal.binary.add", "unsupported dtype")),
        };
        let output_byte_length = output.byte_len()?;
        let mut output_allocation = None;
        let event = self.record_native_event(context, || {
            let allocation =
                self.allocate_native(output_byte_length, context.cancellation)?;
            let event = self
                .runtime
                .dispatch_add(
                    &stream,
                    element_type,
                    left,
                    right,
                    &allocation,
                    elements,
                )
                .map_err(|error| map_execution_error("zed.metal.binary.add", error))?;
            output_allocation = Some(allocation);
            Ok(event)
        })?;
        let tensor = self.tensor_from_allocation(output, output_allocation, memory)?;
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
        self.unsupported_result("zed.metal.binary-scalar", context)
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.reduction", context)
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.indexing", context)
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.resize", context)
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.convolution", context)
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.metal.linear-algebra", context)
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.unsupported_result("zed.metal.custom-kernel", context)
    }
}

fn metal_capability_matrix(
    device: DeviceId,
    properties: NativeDeviceProperties,
) -> Result<BackendCapabilityMatrix, TensorError> {
    if device != DeviceId::new(DeviceKind::Metal, 0) {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::new(DeviceKind::Metal, 0),
            actual: device,
        });
    }
    let mut supported = Vec::with_capacity(12);
    for dtype in [DType::F16, DType::F32] {
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

fn map_execution_error(operation: &str, error: MetalExecutionError) -> TensorError {
    let device = DeviceId::new(DeviceKind::Metal, 0);
    match error {
        MetalExecutionError::UnsupportedTarget { target } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!("Metal is unavailable on target {target}"),
        },
        MetalExecutionError::NoSystemDevice => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: "MTLCreateSystemDefaultDevice returned no device".to_owned(),
        },
        MetalExecutionError::InvalidAbi { reason } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!("Metal execution ABI is unavailable: {reason}"),
        },
        MetalExecutionError::InvalidCertifiedInputs { reason } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!("certified Metal execution inputs are unavailable: {reason}"),
            }
        }
        MetalExecutionError::MissingFunction { function } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!("required Metal execution function is unavailable: {function}"),
        },
        MetalExecutionError::PipelineCreation { function, reason } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!("Metal execution pipeline {function} is unavailable: {reason}"),
            }
        }
        MetalExecutionError::OutOfMemory { requested } => TensorError::AllocationFailed {
            requested,
            reason: format!("{operation}: Metal allocation failed"),
        },
        MetalExecutionError::DeviceLost { code } => TensorError::DeviceLost {
            reason: format!("{operation}: Metal device lost with command error code {code}"),
        },
        MetalExecutionError::ForeignResource => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: "Metal resource belongs to a different certified runtime".to_owned(),
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
#[path = "../../tests/backends/apple_metal_mps_sim_model_0015.rs"]
mod tests;
