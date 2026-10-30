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
use sim_backend_cuda::{
    CudaAllocation, CudaElementType, CudaEvent, CudaExecutionError, CudaExecutionSession,
};
use sim_types::DeviceKind;
use std::{fmt, sync::Arc};

const MAX_CUDA_STREAMS: usize = 1_024;
const MAX_CUDA_PENDING_EVENTS: usize = 4_096;

struct RuntimeAdapter(CudaExecutionSession);

#[derive(Clone)]
enum CudaTrackedEvent {
    Native(CudaEvent),
    Synchronized,
}

impl RuntimeAdapter {
    fn allocate(
        &self,
        byte_length: u64,
        dtype: DType,
        cancellation: &CancellationToken,
    ) -> Result<CudaAllocation, TensorError> {
        let element_type = cuda_element_type(dtype)?;
        let byte_width = dtype.byte_width();
        let elements = byte_length
            .checked_div(byte_width)
            .filter(|elements| elements.checked_mul(byte_width) == Some(byte_length))
            .and_then(|elements| i64::try_from(elements).ok())
            .filter(|elements| *elements > 0)
            .ok_or(TensorError::ShapeOverflow)?;
        self.0
            .allocate(&[elements], element_type, cancellation)
            .map_err(|error| map_execution_error("zed.cuda.allocate", error))
    }

    fn create_stream(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        cancellation.check()?;
        Ok(())
    }

    fn copy_from_host(
        &self,
        _stream: &(),
        destination: &CudaAllocation,
        destination_offset: u64,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        let destination_offset =
            usize::try_from(destination_offset).map_err(|_| TensorError::ShapeOverflow)?;
        self.0
            .copy_from_host(destination, destination_offset, bytes, cancellation)
            .map_err(|error| map_execution_error("zed.cuda.transfer.host-to-device", error))
    }

    fn copy_to_host(
        &self,
        _stream: &(),
        source: &CudaAllocation,
        source_offset: u64,
        bytes: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        let source_offset =
            usize::try_from(source_offset).map_err(|_| TensorError::ShapeOverflow)?;
        self.0
            .copy_to_host(source, source_offset, bytes, cancellation)
            .map_err(|error| map_execution_error("zed.cuda.transfer.device-to-host", error))
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        _stream: &(),
        _element_type: CudaElementType,
        left: &CudaAllocation,
        right: &CudaAllocation,
        output: &CudaAllocation,
        _elements: u64,
        cancellation: &CancellationToken,
    ) -> Result<CudaTrackedEvent, TensorError> {
        self.0
            .add(left, right, output, cancellation)
            .map(CudaTrackedEvent::Native)
            .map_err(|error| map_execution_error("zed.cuda.binary.add", error))
    }

    fn record_event(
        &self,
        _stream: &(),
        cancellation: &CancellationToken,
    ) -> Result<CudaTrackedEvent, TensorError> {
        self.0
            .synchronize(cancellation)
            .map(|()| CudaTrackedEvent::Synchronized)
            .map_err(|error| map_execution_error("zed.cuda.event.record", error))
    }

    fn wait_event(
        &self,
        event: &CudaTrackedEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        match event {
            CudaTrackedEvent::Native(event) => self
                .0
                .wait_event(event, cancellation)
                .map_err(|error| map_execution_error("zed.cuda.event.wait", error)),
            CudaTrackedEvent::Synchronized => {
                cancellation.check()?;
                Ok(())
            }
        }
    }
}

struct CudaStorageInner {
    allocation: Option<CudaAllocation>,
    byte_length: u64,
    _memory: BackendMemoryReservation,
}

struct CudaStorage {
    backend_id: u64,
    device: DeviceId,
    inner: Arc<CudaStorageInner>,
}

enum PreparedCopySource {
    Cpu(Vec<u8>),
    Cuda {
        storage: Arc<CudaStorageInner>,
        byte_offset: u64,
    },
}

impl fmt::Debug for CudaStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaStorage")
            .field("device", &self.device)
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for CudaStorage {
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

pub struct CudaTensorBackend {
    runtime: RuntimeAdapter,
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    logical_memory: Arc<BackendMemoryTracker>,
    streams: BackendResourceRegistry<()>,
    events: BackendEventTracker<CudaTrackedEvent>,
}

impl CudaTensorBackend {
    pub fn from_certified_session(
        session: CudaExecutionSession,
        requested_memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let vendor = session.properties();
        let ordinal = u32::try_from(vendor.device_ordinal()).map_err(|_| TensorError::ShapeOverflow)?;
        let total_memory_bytes =
            u64::try_from(vendor.total_memory_bytes()).map_err(|_| TensorError::ShapeOverflow)?;
        let allocation_limit_bytes = u64::try_from(vendor.maximum_allocation_bytes())
            .map_err(|_| TensorError::ShapeOverflow)?;
        let effective_limit = requested_memory_limit_bytes
            .min(total_memory_bytes)
            .min(allocation_limit_bytes);
        let device = DeviceId::new(DeviceKind::Cuda, ordinal);
        let (major, minor) = vendor.nvrtc_version();
        let properties = NativeDeviceProperties::new_with_allocation_limit(
            device,
            vendor.name(),
            total_memory_bytes,
            allocation_limit_bytes,
            u32::try_from(major).map_err(|_| TensorError::ShapeOverflow)?,
            u32::try_from(minor).map_err(|_| TensorError::ShapeOverflow)?,
            Some(format!(
                "CUDA driver {}; NVRTC {}.{}; cuBLASLt {}; cuDNN {}",
                vendor.driver_version(),
                major,
                minor,
                vendor.cublaslt_version(),
                vendor.cudnn_version(),
            )),
            true,
        )?;
        Self::from_runtime(
            RuntimeAdapter(session),
            properties,
            effective_limit,
            cancellation,
        )
    }

    fn from_runtime(
        runtime: RuntimeAdapter,
        properties: NativeDeviceProperties,
        effective_memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let device = properties.device();
        if effective_memory_limit_bytes == 0 {
            return Err(TensorError::AllocationFailed {
                requested: effective_memory_limit_bytes,
                reason: "CUDA effective memory limit must be nonzero".to_owned(),
            });
        }
        let capabilities = cuda_capability_matrix(device, properties)?;
        let (backend_id, logical_memory, authority) =
            BackendWorkspaceAuthority::new(effective_memory_limit_bytes)?;
        Ok((
            Self {
                runtime,
                device,
                capabilities,
                backend_id,
                logical_memory,
                streams: BackendResourceRegistry::new("CUDA semantic streams", MAX_CUDA_STREAMS),
                events: BackendEventTracker::new(
                    "CUDA pending events",
                    MAX_CUDA_PENDING_EVENTS,
                ),
            },
            authority,
        ))
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.logical_memory.snapshot()
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
            "zed.cuda.transfer.host-to-device",
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
            let allocation =
                self.runtime
                    .allocate(storage_bytes, descriptor.dtype(), context.cancellation)?;
            if destination_offset != 0 {
                let storage_length =
                    usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
                let zeros = try_zeroed_bytes(storage_length, "CUDA upload gap initialization")?;
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
        self.require_input("zed.cuda.transfer.device-to-host", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = tensor.descriptor().byte_len()?;
        let staging = self.reserve_workspace(context, byte_length)?;
        let byte_length = usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(byte_length, "CUDA download staging")?;
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
        self.require_input("zed.cuda.test.download-storage", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = usize::try_from(storage.byte_length)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(byte_length, "CUDA test full-storage download")?;
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
    ) -> Result<(), TensorError> {
        self.streams
            .get_or_try_insert_with(stream, || self.runtime.create_stream(cancellation))
    }

    fn tensor_from_allocation(
        &self,
        descriptor: TensorDescriptor,
        allocation: Option<CudaAllocation>,
        memory: BackendMemoryReservation,
    ) -> Result<Tensor, TensorError> {
        let byte_length = required_storage_bytes(&descriptor)?;
        let actual = allocation
            .as_ref()
            .map(CudaAllocation::byte_length)
            .map(u64::try_from)
            .transpose()
            .map_err(|_| TensorError::ShapeOverflow)?
            .unwrap_or(0);
        if actual != byte_length {
            return Err(TensorError::StorageLength {
                expected: byte_length,
                actual,
            });
        }
        Tensor::from_backend_storage(
            descriptor,
            Box::new(CudaStorage {
                backend_id: self.backend_id,
                device: self.device,
                inner: Arc::new(CudaStorageInner {
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

    fn storage(&self, tensor: &Tensor) -> Result<Arc<CudaStorageInner>, TensorError> {
        tensor
            .backend_storage::<CudaStorage>()
            .filter(|storage| storage.backend_id == self.backend_id)
            .map(|storage| storage.inner.clone())
            .ok_or_else(|| TensorError::UnsupportedCapability {
                operation: "zed.cuda.storage.lookup".to_owned(),
                device: tensor.descriptor().device(),
                reason: "tensor storage is not owned by this certified CUDA backend instance"
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
                "the reviewed CUDA baseline accepts contiguous f16/f32 tensors only",
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
            self.require_input("zed.cuda.copy", source, context)?;
            return Ok(PreparedCopySource::Cuda {
                storage: self.storage(source)?,
                byte_offset: tensor_byte_offset(source.descriptor())?,
            });
        }
        if source.descriptor().device().kind() == DeviceKind::Cpu {
            let bytes = source.contiguous_bytes()?;
            return try_copy_bytes(bytes, "CUDA CPU copy staging").map(PreparedCopySource::Cpu);
        }
        Err(self.unsupported(
            "zed.cuda.copy",
            "source must be host-addressable contiguous CPU storage or this CUDA backend instance",
        ))
    }

    fn track_event(
        &self,
        context: &ExecutionContext<'_>,
        create: impl FnOnce() -> Result<CudaTrackedEvent, TensorError>,
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
        Err(self.unsupported(operation, "no reviewed CUDA kernel is registered"))
    }
}

impl CachedAllocationOwner for CudaTensorBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn allocator_backend_name(&self) -> &'static str {
        "zed-native-cuda-v1"
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for CudaTensorBackend {
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
        CudaTensorBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.cuda.allocate",
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
            let allocation =
                self.runtime
                    .allocate(byte_length, descriptor.dtype(), context.cancellation)?;
            let byte_length =
                usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
            let bytes = try_zeroed_bytes(byte_length, "CUDA zero initialization")?;
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
            "zed.cuda.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        if source.descriptor().shape() != destination.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "zed.cuda.copy: source shape {:?} does not match destination shape {:?}",
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
            let allocation =
                self.runtime
                    .allocate(storage_bytes, destination.dtype(), context.cancellation)?;
            let storage_length =
                usize::try_from(storage_bytes).map_err(|_| TensorError::ShapeOverflow)?;
            let zeros = try_zeroed_bytes(storage_length, "CUDA copy gap initialization")?;
            self.runtime
                .copy_from_host(&stream, &allocation, 0, &zeros, context.cancellation)?;
            Some(allocation)
        };
        match source {
            PreparedCopySource::Cuda {
                storage,
                byte_offset,
            } => {
                if let (Some(destination), Some(source)) = (&allocation, &storage.allocation) {
                    let staging = self.reserve_workspace(context, byte_length)?;
                    let byte_length =
                        usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
                    let mut bytes = try_zeroed_bytes(byte_length, "CUDA copy staging")?;
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
            "zed.cuda.event.record",
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
            .require("zed.cuda.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "CUDA event belongs to a different backend instance".to_owned(),
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
        self.unsupported_result("zed.cuda.fill", context)
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.unary", context)
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
            return self.unsupported_result("zed.cuda.binary", context);
        }
        for input in [left, right] {
            self.require_descriptor(
                "zed.cuda.binary.add",
                PrimitiveOperation::Binary(BinaryOperation::Add),
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
            self.storage(input)?;
        }
        self.require_descriptor(
            "zed.cuda.binary.add",
            PrimitiveOperation::Binary(BinaryOperation::Add),
            TensorRole::Output,
            &output,
            context,
        )?;
        if left.descriptor().shape() != right.descriptor().shape()
            || left.descriptor().shape() != output.shape()
        {
            return Err(TensorError::Faulted {
                reason: "CUDA add requires identical input and output shapes".to_owned(),
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
                "zed.cuda.binary.add",
                "the reviewed CUDA Add binding requires zero-offset whole buffers",
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
        let output_allocation =
            self.runtime
                .allocate(byte_length, output.dtype(), context.cancellation)?;
        let left_allocation = left
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty CUDA add left input has no allocation".to_owned(),
            })?;
        let right_allocation = right
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty CUDA add right input has no allocation".to_owned(),
            })?;
        let element_type = cuda_element_type(output.dtype())?;
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
        self.unsupported_result("zed.cuda.binary-scalar", context)
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.reduction", context)
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.indexing", context)
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.resize", context)
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.convolution", context)
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.linear-algebra", context)
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.unsupported_result("zed.cuda.custom-kernel", context)
    }
}

fn cuda_capability_matrix(
    device: DeviceId,
    properties: NativeDeviceProperties,
) -> Result<BackendCapabilityMatrix, TensorError> {
    if device.kind() != DeviceKind::Cuda {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::new(DeviceKind::Cuda, device.ordinal()),
            actual: device,
        });
    }
    let mut supported = Vec::with_capacity(if properties.has_fp16() { 10 } else { 7 });
    for dtype in [DType::F32]
        .into_iter()
        .chain(properties.has_fp16().then_some(DType::F16))
    {
        supported.extend([
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
        ]);
    }
    supported.extend([
        OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous),
        OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous),
    ]);
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

fn cuda_element_type(dtype: DType) -> Result<CudaElementType, TensorError> {
    match dtype {
        DType::F16 => Ok(CudaElementType::F16),
        DType::F32 => Ok(CudaElementType::F32),
        _ => Err(TensorError::UnsupportedCapability {
            operation: "zed.cuda.element-type".to_owned(),
            device: DeviceId::new(DeviceKind::Cuda, 0),
            reason: format!("CUDA element type for {dtype:?} is not reviewed"),
        }),
    }
}

fn map_execution_error(operation: &str, error: CudaExecutionError) -> TensorError {
    let device = DeviceId::new(DeviceKind::Cuda, 0);
    match error {
        CudaExecutionError::Load(error) => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!("a registry-certified CUDA execution session is unavailable: {error}"),
        },
        CudaExecutionError::InvalidCertifiedInputs { reason } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: format!("certified CUDA session inputs are invalid: {reason}"),
            }
        }
        CudaExecutionError::Cancelled => TensorError::Cancelled,
        CudaExecutionError::OutOfMemory { requested, limit } => TensorError::AllocationFailed {
            requested: u64::try_from(requested).unwrap_or(u64::MAX),
            reason: format!("{operation}: CUDA allocation limit is {limit} bytes"),
        },
        CudaExecutionError::DeviceLost {
            device,
            operation: native_operation,
        } => TensorError::DeviceLost {
            reason: format!(
                "{operation}: CUDA device {device} was lost during {native_operation}"
            ),
        },
        CudaExecutionError::IdentifierOverflow => TensorError::IdentifierOverflow,
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
#[path = "../../tests/backends/nvidia_cuda_sim_model_0022.rs"]
mod tests;
