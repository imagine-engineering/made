use crate::{
    BackendCapabilityMatrix, BackendEventTracker, BackendMemoryReservation, BackendMemorySnapshot,
    BackendMemoryTracker, BackendResourceRegistry, BackendStorage, BackendWorkspaceAuthority,
    BackendWorkspaceLease, BinaryOperation, CachedAllocationOwner, CancellationToken,
    ConvolutionSpec, CustomKernelId, DType, DeviceId, EventFence, ExecutionContext, IndexSpec,
    Layout, LinearAlgebraOperation, NativeDeviceProperties, OperationSupport, PrimitiveOperation,
    ReductionSpec, ResizeSpec, Scalar, ScalarSide, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, TensorRole, UnaryOperation, ViewAccess, check_backend_context,
    required_storage_bytes, reserve_backend_workspace,
};
use sim_backend_rocm::{
    RocmAllocation, RocmEvent, RocmExecutionError, RocmRuntime, RocmStream,
};
use sim_types::DeviceKind;
use std::{fmt, sync::Arc};
#[cfg(test)]
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

const MAX_ROCM_STREAMS: usize = 1_024;
const MAX_ROCM_PENDING_EVENTS: usize = 4_096;

#[derive(Clone)]
enum RuntimeAdapter {
    Native(RocmRuntime),
    #[cfg(test)]
    Test(Arc<TestRuntime>),
}

#[derive(Clone)]
enum AllocationAdapter {
    Native(Arc<RocmAllocation>),
    #[cfg(test)]
    Test(Arc<Mutex<Vec<u8>>>),
}

#[derive(Clone)]
enum StreamAdapter {
    Native(Arc<RocmStream>),
    #[cfg(test)]
    Test { device: u32 },
}

#[derive(Clone)]
enum EventAdapter {
    Native(Arc<RocmEvent>),
    #[cfg(test)]
    Test(Arc<TestEvent>),
}

#[cfg(test)]
struct TestEvent {
    device: u32,
    drop_probe: Option<TestEventDropProbe>,
}

#[cfg(test)]
impl Drop for TestEvent {
    fn drop(&mut self) {
        let Some(probe) = &self.drop_probe else {
            return;
        };
        probe.total_drops.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
#[derive(Clone)]
struct TestEventDropProbe {
    total_drops: Arc<AtomicU64>,
}

#[allow(unreachable_patterns)]
impl RuntimeAdapter {
    fn device_count(&self) -> u32 {
        match self {
            Self::Native(runtime) => runtime.device_count(),
            #[cfg(test)]
            Self::Test(runtime) => runtime.device_count,
        }
    }

    fn select_device(&self, device: u32) -> Result<(), RocmExecutionError> {
        match self {
            Self::Native(runtime) => runtime.select_device(device),
            #[cfg(test)]
            Self::Test(runtime) => runtime.select_device(device),
        }
    }

    fn device_properties(
        &self,
        device: DeviceId,
    ) -> Result<NativeDeviceProperties, TensorError> {
        match self {
            Self::Native(runtime) => {
                let properties = runtime
                    .device_properties(device.ordinal())
                    .map_err(|error| map_execution_error("zed.rocm.device-properties", error))?;
                NativeDeviceProperties::new(
                    device,
                    properties.name(),
                    properties.total_memory_bytes(),
                    properties.major(),
                    properties.minor(),
                    properties.architecture().map(str::to_owned),
                    properties.has_fp16(),
                )
            }
            #[cfg(test)]
            Self::Test(runtime) => {
                runtime.select_device(device.ordinal()).map_err(|error| {
                    map_execution_error("zed.rocm.device-properties", error)
                })?;
                NativeDeviceProperties::new(
                    device,
                    "test ROCm device",
                    runtime.memory_limit_bytes,
                    9,
                    0,
                    Some("test-gfx900".to_owned()),
                    false,
                )
            }
        }
    }

    fn allocate(
        &self,
        device: u32,
        byte_length: usize,
    ) -> Result<AllocationAdapter, RocmExecutionError> {
        match self {
            Self::Native(runtime) => runtime
                .allocate(device, byte_length)
                .map(|allocation| AllocationAdapter::Native(Arc::new(allocation))),
            #[cfg(test)]
            Self::Test(runtime) => runtime.allocate(device, byte_length),
        }
    }

    fn create_stream(&self, device: u32) -> Result<StreamAdapter, RocmExecutionError> {
        match self {
            Self::Native(runtime) => runtime
                .create_stream(device)
                .map(|stream| StreamAdapter::Native(Arc::new(stream))),
            #[cfg(test)]
            Self::Test(runtime) => runtime.create_stream(device),
        }
    }

    fn copy_host_to_device(
        &self,
        stream: &StreamAdapter,
        destination: &AllocationAdapter,
        destination_offset: usize,
        source: &[u8],
    ) -> Result<(), RocmExecutionError> {
        match (self, stream, destination) {
            (Self::Native(runtime), StreamAdapter::Native(stream), AllocationAdapter::Native(destination)) => {
                runtime.copy_host_to_device(stream, destination, destination_offset, source)
            }
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }, AllocationAdapter::Test(destination)) => {
                runtime.copy_host_to_device(*device, destination, destination_offset, source)
            }
            _ => Err(adapter_identity_error("stream and allocation runtime kinds differ")),
        }
    }

    fn copy_device_to_host(
        &self,
        stream: &StreamAdapter,
        destination: &mut [u8],
        source: &AllocationAdapter,
        source_offset: usize,
    ) -> Result<(), RocmExecutionError> {
        match (self, stream, source) {
            (Self::Native(runtime), StreamAdapter::Native(stream), AllocationAdapter::Native(source)) => {
                runtime.copy_device_to_host(stream, destination, source, source_offset)
            }
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }, AllocationAdapter::Test(source)) => {
                runtime.copy_device_to_host(*device, destination, source, source_offset)
            }
            _ => Err(adapter_identity_error("stream and allocation runtime kinds differ")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_device_to_device(
        &self,
        stream: &StreamAdapter,
        destination: &AllocationAdapter,
        destination_offset: usize,
        source: &AllocationAdapter,
        source_offset: usize,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        match (self, stream, destination, source) {
            (
                Self::Native(runtime),
                StreamAdapter::Native(stream),
                AllocationAdapter::Native(destination),
                AllocationAdapter::Native(source),
            ) => runtime.copy_device_to_device(
                stream,
                destination,
                destination_offset,
                source,
                source_offset,
                byte_length,
            ),
            #[cfg(test)]
            (
                Self::Test(runtime),
                StreamAdapter::Test { device },
                AllocationAdapter::Test(destination),
                AllocationAdapter::Test(source),
            ) => runtime.copy_device_to_device(
                *device,
                destination,
                destination_offset,
                source,
                source_offset,
                byte_length,
            ),
            _ => Err(adapter_identity_error("stream and allocation runtime kinds differ")),
        }
    }

    fn memset(
        &self,
        stream: &StreamAdapter,
        allocation: &AllocationAdapter,
        offset: usize,
        value: u8,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        match (self, stream, allocation) {
            (Self::Native(runtime), StreamAdapter::Native(stream), AllocationAdapter::Native(allocation)) => {
                runtime.memset(stream, allocation, offset, value, byte_length)
            }
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }, AllocationAdapter::Test(allocation)) => {
                runtime.memset(*device, allocation, offset, value, byte_length)
            }
            _ => Err(adapter_identity_error("stream and allocation runtime kinds differ")),
        }
    }

    fn record_event(&self, stream: &StreamAdapter) -> Result<EventAdapter, RocmExecutionError> {
        match (self, stream) {
            (Self::Native(runtime), StreamAdapter::Native(stream)) => runtime
                .record_event(stream)
                .map(|event| EventAdapter::Native(Arc::new(event))),
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }) => runtime.record_event(*device),
            _ => Err(adapter_identity_error("stream runtime kind differs")),
        }
    }

    fn synchronize_event(&self, event: &EventAdapter) -> Result<(), RocmExecutionError> {
        match (self, event) {
            (Self::Native(runtime), EventAdapter::Native(event)) => runtime.synchronize_event(event),
            #[cfg(test)]
            (Self::Test(runtime), EventAdapter::Test(event)) => {
                runtime.synchronize_event(event.device)
            }
            _ => Err(adapter_identity_error("event runtime kind differs")),
        }
    }

    fn synchronize_stream(&self, stream: &StreamAdapter) -> Result<(), RocmExecutionError> {
        match (self, stream) {
            (Self::Native(_), StreamAdapter::Native(stream)) => stream.synchronize(),
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }) => runtime.select_device(*device),
            _ => Err(adapter_identity_error("stream runtime kind differs")),
        }
    }

    fn sgemm_row_major_f32(
        &self,
        stream: &StreamAdapter,
        dimensions: [usize; 3],
        left: &AllocationAdapter,
        right: &AllocationAdapter,
        output: &AllocationAdapter,
    ) -> Result<(), RocmExecutionError> {
        match (self, stream, left, right, output) {
            (
                Self::Native(runtime),
                StreamAdapter::Native(stream),
                AllocationAdapter::Native(left),
                AllocationAdapter::Native(right),
                AllocationAdapter::Native(output),
            ) => {
                let [rows, columns, inner] = dimensions;
                runtime.sgemm_f32(stream, [columns, rows, inner], right, left, output)
            }
            #[cfg(test)]
            (
                Self::Test(runtime),
                StreamAdapter::Test { device },
                AllocationAdapter::Test(left),
                AllocationAdapter::Test(right),
                AllocationAdapter::Test(output),
            ) => runtime.sgemm_row_major_f32(*device, dimensions, left, right, output),
            _ => Err(adapter_identity_error("SGEMM resource runtime kinds differ")),
        }
    }
}

fn adapter_identity_error(reason: &str) -> RocmExecutionError {
    RocmExecutionError::InvalidArgument {
        reason: reason.to_owned(),
    }
}

struct RocmStorageInner {
    allocation: Option<AllocationAdapter>,
    byte_length: u64,
    _memory: BackendMemoryReservation,
}

struct RocmStorage {
    backend_id: u64,
    device: DeviceId,
    inner: Arc<RocmStorageInner>,
}

impl fmt::Debug for RocmStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocmStorage")
            .field("device", &self.device)
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for RocmStorage {
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

pub struct RocmTensorBackend {
    runtime: RuntimeAdapter,
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    memory: Arc<BackendMemoryTracker>,
    streams: BackendResourceRegistry<StreamAdapter>,
    events: BackendEventTracker<EventAdapter>,
}

impl RocmTensorBackend {
    pub fn from_certified_runtime(
        runtime: RocmRuntime,
        device_ordinal: u32,
        memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(RocmTensorBackend, BackendWorkspaceAuthority), TensorError> {
        Self::from_runtime(
            RuntimeAdapter::Native(runtime),
            device_ordinal,
            Some(memory_limit_bytes),
            cancellation,
        )
    }

    fn from_runtime(
        runtime: RuntimeAdapter,
        device_ordinal: u32,
        configured_memory_limit_bytes: Option<u64>,
        cancellation: &CancellationToken,
    ) -> Result<(RocmTensorBackend, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        runtime
            .select_device(device_ordinal)
            .map_err(|error| map_execution_error("zed.rocm.select-device", error))?;
        cancellation.check()?;
        let device = DeviceId::new(DeviceKind::Rocm, device_ordinal);
        let properties = runtime.device_properties(device)?;
        let capabilities = rocm_capability_matrix(device, properties.clone())?;
        let memory_limit_bytes = configured_memory_limit_bytes
            .map_or(properties.total_memory_bytes(), |configured| {
                configured.min(properties.total_memory_bytes())
            });
        let (backend_id, memory, authority) = BackendWorkspaceAuthority::new(memory_limit_bytes)?;
        Ok((
            Self {
                runtime,
                device,
                capabilities,
                backend_id,
                memory,
                streams: BackendResourceRegistry::new("ROCm streams", MAX_ROCM_STREAMS),
                events: BackendEventTracker::new(
                    "ROCm pending events",
                    MAX_ROCM_PENDING_EVENTS,
                ),
            },
            authority,
        ))
    }

    pub fn device_count(&self) -> u32 {
        self.runtime.device_count()
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.memory.snapshot()
    }

    pub fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        reserve_backend_workspace(
            self.backend_id,
            &self.memory,
            context,
            requested,
            requested,
        )
    }

    fn check_context(&self, context: &ExecutionContext<'_>) -> Result<(), TensorError> {
        check_backend_context(self.backend_id, context)
    }

    pub fn upload_bytes(
        &self,
        descriptor: TensorDescriptor,
        bytes: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.rocm.transfer.host-to-device",
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
        let stream = self.stream(context.stream)?;
        let tensor = self.allocate_tensor(descriptor, &stream, context)?;
        if !bytes.is_empty() {
            let storage = self.storage(&tensor)?;
            let allocation = storage.allocation.as_ref().ok_or_else(|| TensorError::Faulted {
                reason: "nonempty ROCm tensor has no device allocation".to_owned(),
            })?;
            let offset = tensor_byte_offset(tensor.descriptor())?;
            self.runtime
                .copy_host_to_device(&stream, allocation, offset, bytes)
                .map_err(|error| map_execution_error("zed.rocm.transfer.host-to-device", error))?;
            self.runtime
                .synchronize_stream(&stream)
                .map_err(|error| map_execution_error("zed.rocm.transfer.host-to-device", error))?;
        }
        self.check_context(context)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    pub fn download_bytes(
        &self,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<u8>, TensorError> {
        self.require_input("zed.rocm.transfer.device-to-host", tensor, context)?;
        let byte_length = usize::try_from(tensor.descriptor().byte_len()?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let staging_bytes =
            u64::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
        let staging = self.reserve_workspace(context, staging_bytes)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_length)
            .map_err(|error| TensorError::AllocationFailed {
                requested: byte_length as u64,
                reason: format!("host transfer allocation failed: {error}"),
            })?;
        bytes.resize(byte_length, 0);
        if byte_length != 0 {
            let stream = self.stream(context.stream)?;
            let storage = self.storage(tensor)?;
            let allocation = storage.allocation.as_ref().ok_or_else(|| TensorError::Faulted {
                reason: "nonempty ROCm tensor has no device allocation".to_owned(),
            })?;
            self.runtime
                .copy_device_to_host(
                    &stream,
                    &mut bytes,
                    allocation,
                    tensor_byte_offset(tensor.descriptor())?,
                )
                .map_err(|error| map_execution_error("zed.rocm.transfer.device-to-host", error))?;
            self.runtime
                .synchronize_stream(&stream)
                .map_err(|error| map_execution_error("zed.rocm.transfer.device-to-host", error))?;
        }
        self.check_context(context)?;
        drop(staging);
        Ok(bytes)
    }

    fn stream(&self, stream_id: StreamId) -> Result<StreamAdapter, TensorError> {
        self.streams.get_or_try_insert_with(stream_id, || {
            self.runtime
                .create_stream(self.device.ordinal())
                .map_err(|error| map_execution_error("zed.rocm.stream.create", error))
        })
    }

    fn allocate_tensor(
        &self,
        descriptor: TensorDescriptor,
        stream: &StreamAdapter,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, TensorError> {
        self.check_context(context)?;
        let byte_length = required_storage_bytes(&descriptor)?;
        let memory = self.memory.reserve(byte_length)?;
        let allocation = if byte_length == 0 {
            None
        } else {
            let byte_length_usize = usize::try_from(byte_length).map_err(|_| {
                TensorError::AllocationFailed {
                    requested: byte_length,
                    reason: "ROCm allocation exceeds the host ABI address range".to_owned(),
                }
            })?;
            let allocation = self
                .runtime
                .allocate(self.device.ordinal(), byte_length_usize)
                .map_err(|error| map_execution_error("zed.rocm.allocate", error))?;
            self.runtime
                .memset(stream, &allocation, 0, 0, byte_length_usize)
                .map_err(|error| map_execution_error("zed.rocm.allocate.zero", error))?;
            Some(allocation)
        };
        self.check_context(context)?;
        let inner = Arc::new(RocmStorageInner {
            allocation,
            byte_length,
            _memory: memory,
        });
        Tensor::from_backend_storage(
            descriptor,
            Box::new(RocmStorage {
                backend_id: self.backend_id,
                device: self.device,
                inner,
            }),
            ViewAccess::Writable,
        )
    }

    fn storage(&self, tensor: &Tensor) -> Result<Arc<RocmStorageInner>, TensorError> {
        tensor
            .backend_storage::<RocmStorage>()
            .filter(|storage| storage.backend_id == self.backend_id)
            .map(|storage| storage.inner.clone())
            .ok_or_else(|| TensorError::UnsupportedCapability {
                operation: "zed.rocm.storage.lookup".to_owned(),
                device: tensor.descriptor().device(),
                reason: "tensor storage is not owned by this certified ROCm backend instance"
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
        if descriptor.dtype() != DType::F32 || descriptor.layout() != Layout::Contiguous {
            return Err(self.unsupported(
                operation,
                "the certified ROCm baseline accepts contiguous f32 tensors only",
            ));
        }
        if !descriptor.is_contiguous()? {
            return Err(self.unsupported(
                operation,
                "the descriptor does not have canonical contiguous strides",
            ));
        }
        let support = OperationSupport::for_tensor(
            primitive,
            role,
            descriptor.dtype(),
            descriptor.layout(),
        )?;
        self.capabilities.require(operation, support)
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
        Err(self.unsupported(operation, "no certified ROCm kernel is registered"))
    }
}

impl CachedAllocationOwner for RocmTensorBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn allocator_backend_name(&self) -> &'static str {
        "zed-native-rocm-hip-v1"
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for RocmTensorBackend {
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
        RocmTensorBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.rocm.allocate",
            PrimitiveOperation::Allocation,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        let stream = self.stream(context.stream)?;
        let tensor = self.allocate_tensor(descriptor, &stream, context)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn copy(
        &self,
        source: &Tensor,
        destination: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.check_context(context)?;
        self.require_descriptor(
            "zed.rocm.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        if source.descriptor().shape() != destination.shape() {
            return Err(TensorError::StorageLength {
                expected: destination.byte_len()?,
                actual: source.descriptor().byte_len()?,
            });
        }
        if source.descriptor().dtype() != destination.dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: destination.dtype(),
                actual: source.descriptor().dtype(),
            });
        }
        if source.descriptor().layout() != Layout::Contiguous
            || !source.descriptor().is_contiguous()?
        {
            return Err(self.unsupported("zed.rocm.copy", "source tensor must be contiguous"));
        }
        let stream = self.stream(context.stream)?;
        let tensor = self.allocate_tensor(destination, &stream, context)?;
        let byte_length = usize::try_from(source.descriptor().byte_len()?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        if byte_length != 0 {
            let destination_storage = self.storage(&tensor)?;
            let destination_allocation = destination_storage.allocation.as_ref().ok_or_else(|| {
                TensorError::Faulted {
                    reason: "nonempty ROCm destination has no allocation".to_owned(),
                }
            })?;
            let destination_offset = tensor_byte_offset(tensor.descriptor())?;
            if source.descriptor().device() == self.device {
                if source.descriptor().stream() != context.stream {
                    return Err(TensorError::StreamMismatch {
                        expected: context.stream,
                        actual: source.descriptor().stream(),
                    });
                }
                self.capabilities.require(
                    "zed.rocm.copy",
                    OperationSupport::copy_input(DType::F32, Layout::Contiguous),
                )?;
                let source_storage = self.storage(source)?;
                let source_allocation = source_storage.allocation.as_ref().ok_or_else(|| {
                    TensorError::Faulted {
                        reason: "nonempty ROCm source has no allocation".to_owned(),
                    }
                })?;
                self.runtime
                    .copy_device_to_device(
                        &stream,
                        destination_allocation,
                        destination_offset,
                        source_allocation,
                        tensor_byte_offset(source.descriptor())?,
                        byte_length,
                    )
                    .map_err(|error| map_execution_error("zed.rocm.copy", error))?;
            } else if source.descriptor().device().kind() == DeviceKind::Cpu {
                let source_bytes = source.contiguous_bytes()?;
                self.runtime
                    .copy_host_to_device(
                        &stream,
                        destination_allocation,
                        destination_offset,
                        source_bytes,
                    )
                    .map_err(|error| map_execution_error("zed.rocm.copy", error))?;
                self.runtime
                    .synchronize_stream(&stream)
                    .map_err(|error| map_execution_error("zed.rocm.copy", error))?;
            } else {
                return Err(self.unsupported(
                    "zed.rocm.copy",
                    "source must be host-addressable CPU storage or this ROCm backend instance",
                ));
            }
        }
        self.check_context(context)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        self.check_context(context)?;
        self.capabilities
            .require("zed.rocm.event.record", OperationSupport::record_event())?;
        let stream = self.stream(context.stream)?;
        let sequence = self.events.record_with(context.stream, || {
            self.runtime
                .record_event(&stream)
                .map_err(|error| map_execution_error("zed.rocm.event.record", error))
        })?;
        if let Err(error) = self.check_context(context) {
            let removed = self.events.cancel(context.stream, sequence)?;
            drop(removed);
            return Err(error);
        }
        Ok(EventFence {
            backend_id: self.backend_id,
            device: self.device,
            stream: context.stream,
            sequence,
        })
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.check_context(context)?;
        self.capabilities
            .require("zed.rocm.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "ROCm event belongs to a different backend instance".to_owned(),
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
        let Some(native_event) = self
            .events
            .event_for_wait(event.stream, event.sequence)?
        else {
            return self.check_context(context);
        };
        self.runtime
            .synchronize_event(&native_event)
            .map_err(|error| map_execution_error("zed.rocm.event.wait", error))?;
        let retired = self.events.complete(event.stream, event.sequence)?;
        drop(retired);
        self.check_context(context)
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.fill", context)
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.unary", context)
    }

    fn binary(
        &self,
        _operation: BinaryOperation,
        _left: &Tensor,
        _right: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.binary", context)
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
        self.unsupported_result("zed.rocm.binary-scalar", context)
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.reduction", context)
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.indexing", context)
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.resize", context)
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.convolution", context)
    }

    fn linear_algebra(
        &self,
        operation: LinearAlgebraOperation,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.check_context(context)?;
        if operation != LinearAlgebraOperation::MatrixMultiply {
            return Err(self.unsupported(
                "zed.rocm.linear-algebra",
                "the certified ROCm baseline implements rank-two matrix multiplication only",
            ));
        }
        let [left, right] = inputs else {
            return Err(TensorError::Faulted {
                reason: format!("matrix multiplication requires two inputs, got {}", inputs.len()),
            });
        };
        for input in [left, right] {
            self.require_descriptor(
                "zed.rocm.linear-algebra.matmul",
                PrimitiveOperation::LinearAlgebra(LinearAlgebraOperation::MatrixMultiply),
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
            self.storage(input)?;
        }
        self.require_descriptor(
            "zed.rocm.linear-algebra.matmul",
            PrimitiveOperation::LinearAlgebra(LinearAlgebraOperation::MatrixMultiply),
            TensorRole::Output,
            &output,
            context,
        )?;
        let [rows, inner] = left.descriptor().shape() else {
            return Err(TensorError::Faulted {
                reason: "matrix multiplication left input must have rank two".to_owned(),
            });
        };
        let [right_inner, columns] = right.descriptor().shape() else {
            return Err(TensorError::Faulted {
                reason: "matrix multiplication right input must have rank two".to_owned(),
            });
        };
        if inner != right_inner || output.shape() != [*rows, *columns] {
            return Err(TensorError::Faulted {
                reason: "matrix multiplication dimensions are incompatible".to_owned(),
            });
        }
        let dimensions = [
            usize::try_from(*rows).map_err(|_| TensorError::ShapeOverflow)?,
            usize::try_from(*columns).map_err(|_| TensorError::ShapeOverflow)?,
            usize::try_from(*inner).map_err(|_| TensorError::ShapeOverflow)?,
        ];
        let stream = self.stream(context.stream)?;
        let tensor = self.allocate_tensor(output, &stream, context)?;
        if dimensions.iter().all(|dimension| *dimension != 0) {
            let left_storage = self.storage(left)?;
            let right_storage = self.storage(right)?;
            let output_storage = self.storage(&tensor)?;
            let left_allocation = left_storage.allocation.as_ref().ok_or_else(|| {
                TensorError::Faulted {
                    reason: "nonempty left matrix has no ROCm allocation".to_owned(),
                }
            })?;
            let right_allocation = right_storage.allocation.as_ref().ok_or_else(|| {
                TensorError::Faulted {
                    reason: "nonempty right matrix has no ROCm allocation".to_owned(),
                }
            })?;
            let output_allocation = output_storage.allocation.as_ref().ok_or_else(|| {
                TensorError::Faulted {
                    reason: "nonempty output matrix has no ROCm allocation".to_owned(),
                }
            })?;
            if left.descriptor().offset_elements() != 0
                || right.descriptor().offset_elements() != 0
                || tensor.descriptor().offset_elements() != 0
            {
                return Err(self.unsupported(
                    "zed.rocm.linear-algebra.matmul",
                    "rocBLAS baseline matrices must begin at storage offset zero",
                ));
            }
            self.runtime
                .sgemm_row_major_f32(
                    &stream,
                    dimensions,
                    left_allocation,
                    right_allocation,
                    output_allocation,
                )
                .map_err(|error| map_execution_error("zed.rocm.linear-algebra.matmul", error))?;
        }
        self.check_context(context)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.unsupported_result("zed.rocm.custom-kernel", context)
    }
}

fn rocm_capability_matrix(
    device: DeviceId,
    properties: NativeDeviceProperties,
) -> Result<BackendCapabilityMatrix, TensorError> {
    if device.kind() != DeviceKind::Rocm {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::new(DeviceKind::Rocm, device.ordinal()),
            actual: device,
        });
    }
    let supported = vec![
        OperationSupport::allocation(DType::F32, Layout::Contiguous),
        OperationSupport::copy_input(DType::F32, Layout::Contiguous),
        OperationSupport::copy_output(DType::F32, Layout::Contiguous),
        OperationSupport::linear_algebra_input(
            LinearAlgebraOperation::MatrixMultiply,
            DType::F32,
            Layout::Contiguous,
        ),
        OperationSupport::linear_algebra_output(
            LinearAlgebraOperation::MatrixMultiply,
            DType::F32,
            Layout::Contiguous,
        ),
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ];
    BackendCapabilityMatrix::new_with_properties(
        device,
        supported.clone(),
        supported,
        Some(properties),
    )
}

fn tensor_byte_offset(descriptor: &TensorDescriptor) -> Result<usize, TensorError> {
    descriptor
        .offset_elements()
        .checked_mul(descriptor.dtype().byte_width())
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(TensorError::ShapeOverflow)
}

fn map_execution_error(operation: &str, error: RocmExecutionError) -> TensorError {
    match error {
        RocmExecutionError::OutOfMemory { bytes, message, .. } => TensorError::AllocationFailed {
            requested: u64::try_from(bytes).unwrap_or(u64::MAX),
            reason: message,
        },
        RocmExecutionError::DeviceLost { message, .. } => TensorError::DeviceLost {
            reason: format!("{operation}: {message}"),
        },
        RocmExecutionError::InvalidDevice {
            device,
            device_count,
        } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: DeviceId::new(DeviceKind::Rocm, device),
            reason: format!("device ordinal is outside certified count {device_count}"),
        },
        RocmExecutionError::UnsupportedTarget { required, actual } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::new(DeviceKind::Rocm, 0),
                reason: format!("requires target {required}; current target is {actual}"),
            }
        }
        RocmExecutionError::InvalidArgument { reason } => TensorError::Faulted {
            reason: format!("{operation}: {reason}"),
        },
        RocmExecutionError::Status {
            operation: vendor_operation,
            status,
            message,
        } => TensorError::Faulted {
            reason: format!(
                "{operation}: ROCm {vendor_operation} failed with status {status}: {message}"
            ),
        },
    }
}

#[cfg(test)]
struct TestRuntime {
    device_count: u32,
    memory_limit_bytes: u64,
    allocation_failure: Mutex<Option<RocmExecutionError>>,
    allocation_calls: AtomicU64,
    device_to_host_calls: AtomicU64,
    event_drop_probe: Mutex<Option<TestEventDropProbe>>,
    cancel_after_event_record: Mutex<Option<CancellationToken>>,
    event_synchronize_barrier: Mutex<Option<Arc<std::sync::Barrier>>>,
}

#[cfg(test)]
impl TestRuntime {
    fn select_device(&self, device: u32) -> Result<(), RocmExecutionError> {
        if device < self.device_count {
            Ok(())
        } else {
            Err(RocmExecutionError::InvalidDevice {
                device,
                device_count: self.device_count,
            })
        }
    }

    fn allocate(
        &self,
        device: u32,
        byte_length: usize,
    ) -> Result<AllocationAdapter, RocmExecutionError> {
        self.select_device(device)?;
        self.allocation_calls.fetch_add(1, Ordering::AcqRel);
        if let Some(error) = self
            .allocation_failure
            .lock()
            .map_err(|_| adapter_identity_error("test allocation failure lock is poisoned"))?
            .take()
        {
            return Err(error);
        }
        Ok(AllocationAdapter::Test(Arc::new(Mutex::new(vec![
            0;
            byte_length
        ]))))
    }

    fn create_stream(&self, device: u32) -> Result<StreamAdapter, RocmExecutionError> {
        self.select_device(device)?;
        Ok(StreamAdapter::Test { device })
    }

    fn copy_host_to_device(
        &self,
        device: u32,
        destination: &Arc<Mutex<Vec<u8>>>,
        destination_offset: usize,
        source: &[u8],
    ) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        let mut destination = destination
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        checked_copy(&mut destination, destination_offset, source)
    }

    fn copy_device_to_host(
        &self,
        device: u32,
        destination: &mut [u8],
        source: &Arc<Mutex<Vec<u8>>>,
        source_offset: usize,
    ) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        self.device_to_host_calls.fetch_add(1, Ordering::AcqRel);
        let source = source
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        let end = source_offset
            .checked_add(destination.len())
            .ok_or_else(|| adapter_identity_error("test copy range overflows"))?;
        let source = source
            .get(source_offset..end)
            .ok_or_else(|| adapter_identity_error("test copy range exceeds allocation"))?;
        destination.copy_from_slice(source);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_device_to_device(
        &self,
        device: u32,
        destination: &Arc<Mutex<Vec<u8>>>,
        destination_offset: usize,
        source: &Arc<Mutex<Vec<u8>>>,
        source_offset: usize,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        let source_bytes = {
            let source = source
                .lock()
                .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
            let end = source_offset
                .checked_add(byte_length)
                .ok_or_else(|| adapter_identity_error("test copy range overflows"))?;
            source
                .get(source_offset..end)
                .ok_or_else(|| adapter_identity_error("test copy range exceeds allocation"))?
                .to_vec()
        };
        self.copy_host_to_device(device, destination, destination_offset, &source_bytes)
    }

    fn memset(
        &self,
        device: u32,
        allocation: &Arc<Mutex<Vec<u8>>>,
        offset: usize,
        value: u8,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        let mut allocation = allocation
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        let end = offset
            .checked_add(byte_length)
            .ok_or_else(|| adapter_identity_error("test memset range overflows"))?;
        allocation
            .get_mut(offset..end)
            .ok_or_else(|| adapter_identity_error("test memset range exceeds allocation"))?
            .fill(value);
        Ok(())
    }

    fn record_event(&self, device: u32) -> Result<EventAdapter, RocmExecutionError> {
        self.select_device(device)?;
        let drop_probe = self
            .event_drop_probe
            .lock()
            .map_err(|_| adapter_identity_error("test event drop probe lock is poisoned"))?
            .clone();
        if let Some(cancellation) = self
            .cancel_after_event_record
            .lock()
            .map_err(|_| adapter_identity_error("test event cancellation hook is poisoned"))?
            .take()
        {
            cancellation.cancel();
        }
        Ok(EventAdapter::Test(Arc::new(TestEvent {
            device,
            drop_probe,
        })))
    }

    fn synchronize_event(&self, device: u32) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        let barrier = self
            .event_synchronize_barrier
            .lock()
            .map_err(|_| adapter_identity_error("test event barrier lock is poisoned"))?
            .clone();
        if let Some(barrier) = barrier {
            barrier.wait();
        }
        Ok(())
    }

    fn sgemm_row_major_f32(
        &self,
        device: u32,
        [rows, columns, inner]: [usize; 3],
        left: &Arc<Mutex<Vec<u8>>>,
        right: &Arc<Mutex<Vec<u8>>>,
        output: &Arc<Mutex<Vec<u8>>>,
    ) -> Result<(), RocmExecutionError> {
        self.select_device(device)?;
        let left = left
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        let right = right
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        let mut result = vec![0_u8; rows * columns * std::mem::size_of::<f32>()];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0_f32;
                for contracted in 0..inner {
                    sum += test_read_f32(&left, row * inner + contracted)?
                        * test_read_f32(&right, contracted * columns + column)?;
                }
                let offset = (row * columns + column) * std::mem::size_of::<f32>();
                let target = result
                    .get_mut(offset..offset + std::mem::size_of::<f32>())
                    .ok_or_else(|| adapter_identity_error("test SGEMM output range is invalid"))?;
                target.copy_from_slice(&sum.to_ne_bytes());
            }
        }
        let mut output = output
            .lock()
            .map_err(|_| adapter_identity_error("test allocation lock is poisoned"))?;
        checked_copy(&mut output, 0, &result)
    }
}

#[cfg(test)]
fn checked_copy(
    destination: &mut [u8],
    destination_offset: usize,
    source: &[u8],
) -> Result<(), RocmExecutionError> {
    let end = destination_offset
        .checked_add(source.len())
        .ok_or_else(|| adapter_identity_error("test copy range overflows"))?;
    destination
        .get_mut(destination_offset..end)
        .ok_or_else(|| adapter_identity_error("test copy range exceeds allocation"))?
        .copy_from_slice(source);
    Ok(())
}

#[cfg(test)]
fn test_read_f32(bytes: &[u8], element: usize) -> Result<f32, RocmExecutionError> {
    let offset = element
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| adapter_identity_error("test SGEMM input offset overflows"))?;
    let value = bytes
        .get(offset..offset + std::mem::size_of::<f32>())
        .ok_or_else(|| adapter_identity_error("test SGEMM input range is invalid"))?;
    let value: [u8; 4] = value
        .try_into()
        .map_err(|_| adapter_identity_error("test SGEMM input width is invalid"))?;
    Ok(f32::from_ne_bytes(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchReservation;

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

    fn test_backend() -> Result<(RocmTensorBackend, ScratchReservation), TensorError> {
        test_backend_with_runtime(Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 1024 * 1024,
            allocation_failure: Mutex::new(None),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        }))
    }

    fn test_backend_with_runtime(
        runtime: Arc<TestRuntime>,
    ) -> Result<(RocmTensorBackend, ScratchReservation), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = RocmTensorBackend::from_runtime(
            RuntimeAdapter::Test(runtime),
            0,
            None,
            &cancellation,
        )?;
        let authorized = authority.memory_snapshot().limit_bytes;
        Ok((backend, authority.authorize_workspace(authorized)?))
    }

    #[test]
    fn configured_worker_limit_caps_the_certified_device_capacity() -> Result<(), TensorError> {
        let runtime = Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 64,
            allocation_failure: Mutex::new(None),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        });
        let cancellation = CancellationToken::default();
        let (backend, authority) = RocmTensorBackend::from_runtime(
            RuntimeAdapter::Test(runtime),
            0,
            Some(32),
            &cancellation,
        )?;
        assert_eq!(backend.memory_snapshot().limit_bytes, 32);
        let execution = context(authority.authorize_workspace(32)?, &cancellation);
        assert!(matches!(
            backend.allocate(descriptor(vec![9])?, &execution),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    fn descriptor(shape: Vec<u64>) -> Result<TensorDescriptor, TensorError> {
        TensorDescriptor::contiguous(
            shape,
            DType::F32,
            DeviceId::new(DeviceKind::Rocm, 0),
            StreamId::DEFAULT,
        )
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
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

    #[test]
    fn instance_matrix_advertises_only_backed_primitive_rows() -> Result<(), TensorError> {
        let (backend, _) = test_backend()?;
        assert_eq!(backend.capabilities().supported().len(), 7);
        assert!(backend.capabilities().supports(OperationSupport::allocation(
            DType::F32,
            Layout::Contiguous
        )));
        assert!(!backend
            .capabilities()
            .supports(OperationSupport::fill(DType::F32, Layout::Contiguous)));
        assert!(backend.capabilities().supports(
            OperationSupport::linear_algebra_input(
                LinearAlgebraOperation::MatrixMultiply,
                DType::F32,
                Layout::Contiguous,
            )
        ));
        assert!(backend.capabilities().supports(
            OperationSupport::linear_algebra_output(
                LinearAlgebraOperation::MatrixMultiply,
                DType::F32,
                Layout::Contiguous,
            )
        ));
        let properties = backend
            .capabilities()
            .device_properties()
            .ok_or_else(|| TensorError::Faulted {
                reason: "ROCm instance properties are absent".to_owned(),
            })?;
        assert_eq!(properties.device(), DeviceId::new(DeviceKind::Rocm, 0));
        assert_eq!(properties.name(), "test ROCm device");
        assert_eq!(properties.total_memory_bytes(), 1024 * 1024);
        assert_eq!(properties.major(), 9);
        assert_eq!(properties.minor(), 0);
        assert_eq!(properties.architecture(), Some("test-gfx900"));
        assert!(!properties.has_fp16());
        assert_eq!(backend.device_count(), 1);
        Ok(())
    }

    #[test]
    fn transfers_copy_events_and_row_major_sgemm_preserve_semantics() -> Result<(), TensorError> {
        let (backend, scratch) = test_backend()?;
        let cancellation = CancellationToken::default();
        let context = context(scratch, &cancellation);
        let (left, left_event) = backend.upload_bytes(
            descriptor(vec![2, 3])?,
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            &context,
        )?;
        backend.wait_event(left_event, &context)?;
        let (right, _) = backend.upload_bytes(
            descriptor(vec![3, 2])?,
            &f32_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            &context,
        )?;
        let (output, event) = backend.linear_algebra(
            LinearAlgebraOperation::MatrixMultiply,
            &[left.clone(), right],
            descriptor(vec![2, 2])?,
            &context,
        )?;
        backend.wait_event(event, &context)?;
        assert_eq!(
            bytes_f32(&backend.download_bytes(&output, &context)?)?,
            vec![58.0, 64.0, 139.0, 154.0]
        );
        let (copy, _) = backend.copy(&left, descriptor(vec![2, 3])?, &context)?;
        assert_eq!(
            bytes_f32(&backend.download_bytes(&copy, &context)?)?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn cancellation_and_cross_instance_storage_fail_closed() -> Result<(), TensorError> {
        let (backend, scratch) = test_backend()?;
        let cancellation = CancellationToken::default();
        let live_context = context(scratch.clone(), &cancellation);
        let (tensor, _) = backend.upload_bytes(
            descriptor(vec![1])?,
            &f32_bytes(&[1.0]),
            &live_context,
        )?;
        let (other, _) = test_backend()?;
        assert!(matches!(
            other.download_bytes(&tensor, &live_context),
            Err(TensorError::WorkspaceAuthorizationMismatch { .. })
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context = context(scratch, &cancelled);
        assert!(matches!(
            backend.allocate(descriptor(vec![1])?, &cancelled_context),
            Err(TensorError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn foreign_scratch_is_rejected_before_allocation_and_accounting_converges(
    ) -> Result<(), TensorError> {
        let runtime = Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 1024 * 1024,
            allocation_failure: Mutex::new(None),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        });
        let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
        let (foreign_backend, foreign_scratch) = test_backend()?;
        let cancellation = CancellationToken::default();
        let foreign_context = context(foreign_scratch, &cancellation);
        assert!(matches!(
            backend.allocate(descriptor(vec![1])?, &foreign_context),
            Err(TensorError::WorkspaceAuthorizationMismatch { .. })
        ));
        assert_eq!(runtime.allocation_calls.load(Ordering::Acquire), 0);
        assert_eq!(
            backend.memory_snapshot(),
            BackendMemorySnapshot {
                limit_bytes: 1024 * 1024,
                current_bytes: 0,
                peak_bytes: 0,
            }
        );
        assert_eq!(foreign_backend.memory_snapshot().current_bytes, 0);

        let live_context = context(scratch, &cancellation);
        let workspace = backend.reserve_workspace(&live_context, 12)?;
        assert_eq!(workspace.bytes(), 12);
        assert_eq!(backend.memory_snapshot().current_bytes, 12);
        drop(workspace);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        let (tensor, _) = backend.upload_bytes(
            descriptor(vec![1])?,
            &f32_bytes(&[3.0]),
            &live_context,
        )?;
        assert_eq!(backend.memory_snapshot().current_bytes, 4);
        assert_eq!(
            bytes_f32(&backend.download_bytes(&tensor, &live_context)?)?,
            vec![3.0]
        );
        assert_eq!(backend.memory_snapshot().current_bytes, 4);
        assert_eq!(backend.memory_snapshot().peak_bytes, 12);
        drop(tensor);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn oom_device_loss_and_cancellation_release_capacity() -> Result<(), TensorError> {
        let runtime = Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 16,
            allocation_failure: Mutex::new(Some(RocmExecutionError::DeviceLost {
                device: 0,
                operation: "hipMalloc",
                message: "injected loss".to_owned(),
            })),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        });
        let (backend, scratch) = test_backend_with_runtime(runtime)?;
        let cancellation = CancellationToken::default();
        let live_context = context(scratch.clone(), &cancellation);
        assert!(matches!(
            backend.allocate(descriptor(vec![1])?, &live_context),
            Err(TensorError::DeviceLost { .. })
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        assert!(matches!(
            backend.allocate(descriptor(vec![5])?, &live_context),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context = context(scratch, &cancelled);
        assert!(matches!(
            backend.allocate(descriptor(vec![1])?, &cancelled_context),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn download_staging_obeys_scratch_before_copy_or_output_allocation(
    ) -> Result<(), TensorError> {
        let runtime = Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 64,
            allocation_failure: Mutex::new(None),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        });
        let cancellation = CancellationToken::default();
        let (backend, authority) = RocmTensorBackend::from_runtime(
            RuntimeAdapter::Test(runtime.clone()),
            0,
            None,
            &cancellation,
        )?;
        let upload_context = context(authority.authorize_workspace(64)?, &cancellation);
        let (tensor, _) = backend.upload_bytes(
            descriptor(vec![1])?,
            &f32_bytes(&[7.0]),
            &upload_context,
        )?;
        let before = backend.memory_snapshot();
        let insufficient_context = context(authority.authorize_workspace(3)?, &cancellation);
        assert!(matches!(
            backend.download_bytes(&tensor, &insufficient_context),
            Err(TensorError::WorkspaceAuthorizationExceeded {
                requested: 4,
                authorized: 3,
                in_use: 0,
            })
        ));
        assert_eq!(runtime.device_to_host_calls.load(Ordering::Acquire), 0);
        assert_eq!(backend.memory_snapshot(), before);
        drop(tensor);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn stream_and_event_registries_are_bounded_and_completed_waits_repeat(
    ) -> Result<(), TensorError> {
        let runtime = Arc::new(TestRuntime {
            device_count: 1,
            memory_limit_bytes: 1024 * 1024,
            allocation_failure: Mutex::new(None),
            allocation_calls: AtomicU64::new(0),
            device_to_host_calls: AtomicU64::new(0),
            event_drop_probe: Mutex::new(None),
            cancel_after_event_record: Mutex::new(None),
            event_synchronize_barrier: Mutex::new(None),
        });
        let (backend, scratch) = test_backend_with_runtime(runtime.clone())?;
        let total_drops = Arc::new(AtomicU64::new(0));
        *runtime.event_drop_probe.lock().map_err(|_| TensorError::Faulted {
            reason: "test event drop probe lock is poisoned".to_owned(),
        })? = Some(TestEventDropProbe {
            total_drops: total_drops.clone(),
        });
        let cancellation = CancellationToken::default();
        for stream in 0..MAX_ROCM_STREAMS {
            backend.stream(StreamId::new(stream as u64))?;
        }
        assert!(matches!(
            backend.stream(StreamId::new(MAX_ROCM_STREAMS as u64)),
            Err(TensorError::ResourceLimitExceeded {
                resource: "ROCm streams",
                limit: MAX_ROCM_STREAMS,
            })
        ));

        let default_context = context(scratch, &cancellation);
        let mut fences = Vec::with_capacity(MAX_ROCM_PENDING_EVENTS);
        for _ in 0..MAX_ROCM_PENDING_EVENTS {
            fences.push(backend.record_event(&default_context)?);
        }
        assert!(matches!(
            backend.record_event(&default_context),
            Err(TensorError::ResourceLimitExceeded {
                resource: "ROCm pending events",
                limit: MAX_ROCM_PENDING_EVENTS,
            })
        ));
        let latest = fences.last().copied().ok_or_else(|| TensorError::Faulted {
            reason: "ROCm event stress fixture did not record events".to_owned(),
        })?;
        backend.wait_event(latest, &default_context)?;
        backend.wait_event(latest, &default_context)?;
        for fence in fences {
            backend.wait_event(fence, &default_context)?;
        }
        assert_eq!(total_drops.load(Ordering::Acquire), MAX_ROCM_PENDING_EVENTS as u64);
        assert_eq!(backend.events.pending_len()?, 0);
        assert!(backend.events.completed_stream_count()? <= MAX_ROCM_STREAMS);
        assert_eq!(backend.streams.len()?, MAX_ROCM_STREAMS);

        let concurrent_fence = backend.record_event(&default_context)?;
        *runtime
            .event_synchronize_barrier
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "test event barrier lock is poisoned".to_owned(),
            })? = Some(Arc::new(std::sync::Barrier::new(2)));
        std::thread::scope(|scope| -> Result<(), TensorError> {
            let first_scratch = default_context.scratch.clone();
            let second_scratch = default_context.scratch.clone();
            let first = scope.spawn(|| {
                let cancellation = CancellationToken::default();
                backend.wait_event(
                    concurrent_fence,
                    &context(first_scratch, &cancellation),
                )
            });
            let second = scope.spawn(|| {
                let cancellation = CancellationToken::default();
                backend.wait_event(
                    concurrent_fence,
                    &context(second_scratch, &cancellation),
                )
            });
            first.join().map_err(|_| TensorError::Faulted {
                reason: "first concurrent ROCm event waiter panicked".to_owned(),
            })??;
            second.join().map_err(|_| TensorError::Faulted {
                reason: "second concurrent ROCm event waiter panicked".to_owned(),
            })??;
            Ok(())
        })?;
        *runtime
            .event_synchronize_barrier
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "test event barrier lock is poisoned".to_owned(),
            })? = None;
        assert_eq!(
            total_drops.load(Ordering::Acquire),
            MAX_ROCM_PENDING_EVENTS as u64 + 1
        );

        let cancellation_during_record = CancellationToken::default();
        *runtime
            .cancel_after_event_record
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "test event cancellation hook is poisoned".to_owned(),
        })? = Some(cancellation_during_record.clone());
        let cancellation_context = context(
            default_context.scratch,
            &cancellation_during_record,
        );
        assert!(matches!(
            backend.record_event(&cancellation_context),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(
            total_drops.load(Ordering::Acquire),
            MAX_ROCM_PENDING_EVENTS as u64 + 2
        );
        Ok(())
    }

    #[test]
    fn execution_error_mapping_preserves_oom_and_device_loss() {
        assert!(matches!(
            map_execution_error(
                "test",
                RocmExecutionError::OutOfMemory {
                    device: 0,
                    bytes: 32,
                    message: "oom".to_owned(),
                }
            ),
            TensorError::AllocationFailed { requested: 32, .. }
        ));
        assert!(matches!(
            map_execution_error(
                "test",
                RocmExecutionError::DeviceLost {
                    device: 0,
                    operation: "hipMemcpy",
                    message: "lost".to_owned(),
                }
            ),
            TensorError::DeviceLost { .. }
        ));
    }
}
