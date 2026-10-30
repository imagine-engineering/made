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
use sim_backend_npu::{
    NpuAllocation, NpuElementType, NpuEvent, NpuExecutionError, NpuExecutionSession, NpuLoadError,
    NpuStream,
};
use sim_types::DeviceKind;
#[cfg(test)]
use std::sync::Mutex;
use std::{fmt, sync::Arc};

const MAX_NPU_STREAMS: usize = 1_024;
const MAX_NPU_PENDING_EVENTS: usize = 4_096;

enum RuntimeAdapter {
    Native(NpuExecutionSession),
    #[cfg(test)]
    Test(Arc<TestRuntime>),
}

#[derive(Clone)]
enum AllocationAdapter {
    Native(NpuAllocation),
    #[cfg(test)]
    Test {
        device: u32,
        bytes: Arc<Mutex<Vec<u8>>>,
    },
}

impl AllocationAdapter {
    fn byte_length(&self) -> Result<usize, TensorError> {
        match self {
            Self::Native(allocation) => Ok(allocation.byte_length()),
            #[cfg(test)]
            Self::Test { bytes, .. } => bytes
                .lock()
                .map_err(|_| TensorError::Faulted {
                    reason: "NPU test allocation lock is poisoned".to_owned(),
                })
                .map(|bytes| bytes.len()),
        }
    }
}

#[derive(Clone)]
enum StreamAdapter {
    Native(NpuStream),
    #[cfg(test)]
    Test {
        device: u32,
    },
}

#[derive(Clone)]
enum EventAdapter {
    Native(NpuEvent),
    #[cfg(test)]
    Test {
        device: u32,
    },
}

impl EventAdapter {
    fn is_synchronized(&self) -> bool {
        match self {
            Self::Native(event) => event.is_synchronized(),
            #[cfg(test)]
            Self::Test { .. } => true,
        }
    }
}

#[allow(unreachable_patterns)]
impl RuntimeAdapter {
    fn allocate(
        &self,
        device: u32,
        byte_length: usize,
        cancellation: &CancellationToken,
    ) -> Result<AllocationAdapter, TensorError> {
        match self {
            Self::Native(runtime) => runtime
                .allocate(byte_length, cancellation)
                .map(AllocationAdapter::Native)
                .map_err(|error| map_execution_error("zed.npu.allocate", device, error)),
            #[cfg(test)]
            Self::Test(runtime) => runtime
                .validate_allocation(
                    device,
                    &try_zeroed_bytes(byte_length, "NPU test allocation")?,
                )
                .map(|bytes| AllocationAdapter::Test {
                    device,
                    bytes: Arc::new(Mutex::new(bytes)),
                }),
        }
    }

    fn create_stream(
        &self,
        device: u32,
        cancellation: &CancellationToken,
    ) -> Result<StreamAdapter, TensorError> {
        match self {
            Self::Native(runtime) => runtime
                .create_stream(cancellation)
                .map(StreamAdapter::Native)
                .map_err(|error| map_execution_error("zed.npu.stream.create", device, error)),
            #[cfg(test)]
            Self::Test(runtime) => {
                runtime.synchronize(device)?;
                Ok(StreamAdapter::Test { device })
            }
        }
    }

    fn copy_from_host(
        &self,
        destination: &AllocationAdapter,
        destination_offset: usize,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        match (self, destination) {
            (Self::Native(runtime), AllocationAdapter::Native(destination)) => runtime
                .copy_from_host(destination, destination_offset, bytes, cancellation)
                .map_err(|error| {
                    map_execution_error(
                        "zed.npu.transfer.host-to-device",
                        runtime.properties().device_id(),
                        error,
                    )
                }),
            #[cfg(test)]
            (
                Self::Test(runtime),
                AllocationAdapter::Test {
                    device,
                    bytes: destination,
                },
            ) => {
                runtime.check(*device, bytes.len())?;
                let mut destination = destination.lock().map_err(|_| TensorError::Faulted {
                    reason: "NPU test allocation lock is poisoned".to_owned(),
                })?;
                let destination = allocation_range_mut(
                    &mut destination,
                    destination_offset,
                    bytes.len(),
                    "NPU test host-to-device destination",
                )?;
                destination.copy_from_slice(bytes);
                Ok(())
            }
            _ => Err(adapter_identity_error()),
        }
    }

    fn zero_allocation(
        &self,
        allocation: &AllocationAdapter,
        byte_length: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        const ZERO_CHUNK: [u8; 4_096] = [0; 4_096];
        let mut offset = 0_usize;
        while offset < byte_length {
            cancellation.check()?;
            let remaining = byte_length
                .checked_sub(offset)
                .ok_or(TensorError::ShapeOverflow)?;
            let chunk_length = remaining.min(ZERO_CHUNK.len());
            self.copy_from_host(
                allocation,
                offset,
                &ZERO_CHUNK[..chunk_length],
                cancellation,
            )?;
            offset = offset
                .checked_add(chunk_length)
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok(())
    }

    fn copy_to_host(
        &self,
        source: &AllocationAdapter,
        source_offset: usize,
        bytes: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        match (self, source) {
            (Self::Native(runtime), AllocationAdapter::Native(source)) => runtime
                .copy_to_host(source, source_offset, bytes, cancellation)
                .map_err(|error| {
                    map_execution_error(
                        "zed.npu.transfer.device-to-host",
                        runtime.properties().device_id(),
                        error,
                    )
                }),
            #[cfg(test)]
            (
                Self::Test(runtime),
                AllocationAdapter::Test {
                    device,
                    bytes: source,
                },
            ) => {
                runtime.check(*device, bytes.len())?;
                let source = source.lock().map_err(|_| TensorError::Faulted {
                    reason: "NPU test allocation lock is poisoned".to_owned(),
                })?;
                let source = allocation_range(
                    &source,
                    source_offset,
                    bytes.len(),
                    "NPU test device-to-host source",
                )?;
                bytes.copy_from_slice(source);
                Ok(())
            }
            _ => Err(adapter_identity_error()),
        }
    }

    fn copy_device_to_device(
        &self,
        destination: &AllocationAdapter,
        destination_offset: usize,
        source: &AllocationAdapter,
        source_offset: usize,
        byte_length: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        match (self, destination, source) {
            (
                Self::Native(runtime),
                AllocationAdapter::Native(destination),
                AllocationAdapter::Native(source),
            ) => runtime
                .copy_device_to_device(
                    destination,
                    destination_offset,
                    source,
                    source_offset,
                    byte_length,
                    cancellation,
                )
                .map_err(|error| {
                    map_execution_error("zed.npu.copy", runtime.properties().device_id(), error)
                }),
            #[cfg(test)]
            (
                Self::Test(runtime),
                AllocationAdapter::Test {
                    device: destination_device,
                    bytes: destination,
                },
                AllocationAdapter::Test {
                    device: source_device,
                    bytes: source,
                },
            ) => {
                if destination_device != source_device {
                    return Err(TensorError::DeviceMismatch {
                        expected: DeviceId::new(DeviceKind::Npu, *destination_device),
                        actual: DeviceId::new(DeviceKind::Npu, *source_device),
                    });
                }
                runtime.check(*destination_device, byte_length)?;
                let source = source.lock().map_err(|_| TensorError::Faulted {
                    reason: "NPU test source allocation lock is poisoned".to_owned(),
                })?;
                let copied = allocation_range(
                    &source,
                    source_offset,
                    byte_length,
                    "NPU test device-to-device source",
                )?
                .to_vec();
                drop(source);
                let mut destination = destination.lock().map_err(|_| TensorError::Faulted {
                    reason: "NPU test destination allocation lock is poisoned".to_owned(),
                })?;
                allocation_range_mut(
                    &mut destination,
                    destination_offset,
                    byte_length,
                    "NPU test device-to-device destination",
                )?
                .copy_from_slice(&copied);
                Ok(())
            }
            _ => Err(adapter_identity_error()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        stream: &StreamAdapter,
        element_type: NpuElementType,
        dimensions: &[i64],
        left: &AllocationAdapter,
        right: &AllocationAdapter,
        output: &AllocationAdapter,
        cancellation: &CancellationToken,
    ) -> Result<EventAdapter, TensorError> {
        match (self, stream, left, right, output) {
            (
                Self::Native(runtime),
                StreamAdapter::Native(stream),
                AllocationAdapter::Native(left),
                AllocationAdapter::Native(right),
                AllocationAdapter::Native(output),
            ) => runtime
                .add(
                    stream,
                    element_type,
                    dimensions,
                    left,
                    right,
                    output,
                    cancellation,
                )
                .map(EventAdapter::Native)
                .map_err(|error| {
                    map_execution_error(
                        "zed.npu.binary.add",
                        runtime.properties().device_id(),
                        error,
                    )
                }),
            #[cfg(test)]
            (
                Self::Test(runtime),
                StreamAdapter::Test { device },
                AllocationAdapter::Test { bytes: left, .. },
                AllocationAdapter::Test { bytes: right, .. },
                AllocationAdapter::Test { bytes: output, .. },
            ) => {
                let left_snapshot = left
                    .lock()
                    .map_err(|_| TensorError::Faulted {
                        reason: "NPU test left allocation lock is poisoned".to_owned(),
                    })?
                    .clone();
                let right_snapshot = if Arc::ptr_eq(left, right) {
                    left_snapshot.clone()
                } else {
                    right
                        .lock()
                        .map_err(|_| TensorError::Faulted {
                            reason: "NPU test right allocation lock is poisoned".to_owned(),
                        })?
                        .clone()
                };
                let bytes = runtime.add(
                    *device,
                    element_type,
                    dimensions,
                    &left_snapshot,
                    &right_snapshot,
                )?;
                *output.lock().map_err(|_| TensorError::Faulted {
                    reason: "NPU test output allocation lock is poisoned".to_owned(),
                })? = bytes;
                Ok(EventAdapter::Test { device: *device })
            }
            _ => Err(adapter_identity_error()),
        }
    }

    fn record_event(
        &self,
        stream: &StreamAdapter,
        cancellation: &CancellationToken,
    ) -> Result<EventAdapter, TensorError> {
        match (self, stream) {
            (Self::Native(runtime), StreamAdapter::Native(stream)) => runtime
                .record_event(stream, cancellation)
                .map(EventAdapter::Native)
                .map_err(|error| {
                    map_execution_error(
                        "zed.npu.event.record",
                        runtime.properties().device_id(),
                        error,
                    )
                }),
            #[cfg(test)]
            (Self::Test(runtime), StreamAdapter::Test { device }) => {
                runtime.synchronize(*device)?;
                Ok(EventAdapter::Test { device: *device })
            }
            _ => Err(adapter_identity_error()),
        }
    }

    fn wait_event(
        &self,
        event: &EventAdapter,
        cancellation: &CancellationToken,
    ) -> Result<(), TensorError> {
        match (self, event) {
            (Self::Native(runtime), EventAdapter::Native(event)) => runtime
                .wait_event(event, cancellation)
                .map_err(|error| {
                    map_execution_error(
                        "zed.npu.event.wait",
                        runtime.properties().device_id(),
                        error,
                    )
                }),
            #[cfg(test)]
            (Self::Test(runtime), EventAdapter::Test { device }) => {
                runtime.synchronize(*device)?;
                cancellation.check()?;
                Ok(())
            }
            _ => Err(adapter_identity_error()),
        }
    }
}

struct NpuStorageInner {
    allocation: Option<AllocationAdapter>,
    byte_length: u64,
    _memory: BackendMemoryReservation,
}

struct NpuStorage {
    backend_id: u64,
    device: DeviceId,
    inner: Arc<NpuStorageInner>,
}

impl fmt::Debug for NpuStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NpuStorage")
            .field("device", &self.device)
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl BackendStorage for NpuStorage {
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

pub struct NpuTensorBackend {
    runtime: RuntimeAdapter,
    device: DeviceId,
    device_count: u32,
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    memory: Arc<BackendMemoryTracker>,
    streams: BackendResourceRegistry<StreamAdapter>,
    events: BackendEventTracker<EventAdapter>,
}

impl NpuTensorBackend {
    pub fn from_certified_runtime(
        runtime: NpuExecutionSession,
        device_ordinal: u32,
        memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        let certified = runtime.properties();
        let device_count = certified.device_count();
        if device_ordinal >= device_count || device_ordinal != certified.device_id() {
            return Err(TensorError::UnsupportedCapability {
                operation: "zed.npu.select-device".to_owned(),
                device: DeviceId::new(DeviceKind::Npu, device_ordinal),
                reason: format!(
                    "device ordinal must match certified device {} within count {device_count}",
                    certified.device_id()
                ),
            });
        }
        let allocation_capacity = u64::try_from(certified.allocation_capacity_bytes())
            .map_err(|_| TensorError::ShapeOverflow)?;
        if memory_limit_bytes == 0 || memory_limit_bytes > allocation_capacity {
            return Err(TensorError::UnsupportedCapability {
                operation: "zed.npu.memory-limit".to_owned(),
                device: DeviceId::new(DeviceKind::Npu, device_ordinal),
                reason: format!(
                    "configured memory limit {memory_limit_bytes} must be within certified allocation capacity {allocation_capacity}"
                ),
            });
        }
        let device = DeviceId::new(DeviceKind::Npu, device_ordinal);
        let (major, minor, patch) = certified.api_version();
        let properties = NativeDeviceProperties::new_with_allocation_limit(
            device,
            certified.name(),
            u64::try_from(certified.physical_capacity_bytes())
                .map_err(|_| TensorError::ShapeOverflow)?,
            memory_limit_bytes,
            major,
            minor,
            Some(format!("AscendCL {major}.{minor}.{patch}")),
            true,
        )?;
        Self::from_runtime(
            RuntimeAdapter::Native(runtime),
            device,
            device_count,
            properties,
            memory_limit_bytes,
            cancellation,
        )
    }

    fn from_runtime(
        runtime: RuntimeAdapter,
        device: DeviceId,
        device_count: u32,
        properties: NativeDeviceProperties,
        memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Self, BackendWorkspaceAuthority), TensorError> {
        cancellation.check()?;
        if device.kind() != DeviceKind::Npu || device.ordinal() >= device_count {
            return Err(TensorError::UnsupportedCapability {
                operation: "zed.npu.select-device".to_owned(),
                device,
                reason: format!("device ordinal is outside certified count {device_count}"),
            });
        }
        if properties.device() != device {
            return Err(TensorError::DeviceMismatch {
                expected: device,
                actual: properties.device(),
            });
        }
        let capabilities = npu_capability_matrix(device, properties)?;
        let (backend_id, memory, authority) = BackendWorkspaceAuthority::new(memory_limit_bytes)?;
        Ok((
            Self {
                runtime,
                device,
                device_count,
                capabilities,
                backend_id,
                memory,
                streams: BackendResourceRegistry::new("NPU streams", MAX_NPU_STREAMS),
                events: BackendEventTracker::new("NPU pending events", MAX_NPU_PENDING_EVENTS),
            },
            authority,
        ))
    }

    pub const fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.memory.snapshot()
    }

    pub fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        reserve_backend_workspace(self.backend_id, &self.memory, context, requested, requested)
    }

    pub fn upload_bytes(
        &self,
        descriptor: TensorDescriptor,
        bytes: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.npu.transfer.host-to-device",
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
        self.stream(context.stream, context.cancellation)?;
        let storage_byte_length = required_storage_bytes(&descriptor)?;
        let memory = self.reserve_tensor_storage(&descriptor)?;
        let allocation = if storage_byte_length == 0 {
            None
        } else {
            let storage_byte_length =
                usize::try_from(storage_byte_length).map_err(|_| TensorError::ShapeOverflow)?;
            let allocation = self.runtime.allocate(
                self.device.ordinal(),
                storage_byte_length,
                context.cancellation,
            )?;
            self.runtime
                .zero_allocation(&allocation, storage_byte_length, context.cancellation)?;
            if !bytes.is_empty() {
                self.runtime.copy_from_host(
                    &allocation,
                    tensor_byte_offset(&descriptor)?,
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
        self.require_input("zed.npu.transfer.device-to-host", tensor, context)?;
        let storage = self.storage(tensor)?;
        let byte_length = tensor.descriptor().byte_len()?;
        let staging = self.reserve_workspace(context, byte_length)?;
        let byte_length = usize::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?;
        let mut bytes = try_zeroed_bytes(byte_length, "NPU download staging")?;
        if let Some(allocation) = &storage.allocation {
            self.runtime.copy_to_host(
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

    fn stream(
        &self,
        stream: StreamId,
        cancellation: &CancellationToken,
    ) -> Result<StreamAdapter, TensorError> {
        self.streams.get_or_try_insert_with(stream, || {
            self.runtime
                .create_stream(self.device.ordinal(), cancellation)
        })
    }

    fn tensor_from_allocation(
        &self,
        descriptor: TensorDescriptor,
        allocation: Option<AllocationAdapter>,
        memory: BackendMemoryReservation,
    ) -> Result<Tensor, TensorError> {
        let byte_length = required_storage_bytes(&descriptor)?;
        let actual = match &allocation {
            Some(AllocationAdapter::Native(allocation)) => {
                u64::try_from(allocation.byte_length()).map_err(|_| TensorError::ShapeOverflow)?
            }
            #[cfg(test)]
            Some(AllocationAdapter::Test {
                bytes: allocation, ..
            }) => u64::try_from(
                allocation
                    .lock()
                    .map_err(|_| TensorError::Faulted {
                        reason: "NPU test allocation lock is poisoned".to_owned(),
                    })?
                    .len(),
            )
            .map_err(|_| TensorError::ShapeOverflow)?,
            None => 0,
        };
        if actual != byte_length {
            return Err(TensorError::StorageLength {
                expected: byte_length,
                actual,
            });
        }
        Tensor::from_backend_storage(
            descriptor,
            Box::new(NpuStorage {
                backend_id: self.backend_id,
                device: self.device,
                inner: Arc::new(NpuStorageInner {
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
        self.memory.reserve(required_storage_bytes(descriptor)?)
    }

    fn storage(&self, tensor: &Tensor) -> Result<Arc<NpuStorageInner>, TensorError> {
        tensor
            .backend_storage::<NpuStorage>()
            .filter(|storage| storage.backend_id == self.backend_id)
            .map(|storage| storage.inner.clone())
            .ok_or_else(|| TensorError::UnsupportedCapability {
                operation: "zed.npu.storage.lookup".to_owned(),
                device: tensor.descriptor().device(),
                reason: "tensor storage is not owned by this certified NPU backend instance"
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
                "the certified NPU baseline accepts contiguous f16/f32 tensors only",
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

    fn track_event(
        &self,
        event: EventAdapter,
        context: &ExecutionContext<'_>,
    ) -> Result<EventFence, TensorError> {
        let sequence = self.events.record_with(context.stream, || Ok(event))?;
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
        Err(self.unsupported(operation, "no certified NPU kernel is registered"))
    }
}

impl CachedAllocationOwner for NpuTensorBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn allocator_backend_name(&self) -> &'static str {
        "zed-native-npu-cnrt-v1"
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for NpuTensorBackend {
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
        NpuTensorBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.npu.allocate",
            PrimitiveOperation::Allocation,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        self.stream(context.stream, context.cancellation)?;
        let byte_length = usize::try_from(required_storage_bytes(&descriptor)?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let memory = self.reserve_tensor_storage(&descriptor)?;
        let allocation = if byte_length == 0 {
            None
        } else {
            let allocation =
                self.runtime
                    .allocate(self.device.ordinal(), byte_length, context.cancellation)?;
            self.runtime
                .zero_allocation(&allocation, byte_length, context.cancellation)?;
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
            "zed.npu.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        if source.descriptor().shape() != destination.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "zed.npu.copy: source shape {:?} does not match destination shape {:?}",
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
        let byte_length =
            usize::try_from(destination.byte_len()?).map_err(|_| TensorError::ShapeOverflow)?;
        let storage_byte_length = usize::try_from(required_storage_bytes(&destination)?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let destination_offset = tensor_byte_offset(&destination)?;
        validate_adapter_range(storage_byte_length, destination_offset, byte_length)?;

        enum PreflightSource<'a> {
            Npu {
                allocation: Option<AllocationAdapter>,
                offset: usize,
            },
            Cpu(&'a [u8]),
        }

        let source = if source.descriptor().device() == self.device {
            self.require_input("zed.npu.copy", source, context)?;
            let source_offset = tensor_byte_offset(source.descriptor())?;
            let source_storage = self.storage(source)?;
            let allocation = source_storage.allocation.clone();
            match &allocation {
                Some(allocation) => {
                    validate_adapter_range(allocation.byte_length()?, source_offset, byte_length)?
                }
                None if byte_length == 0 => {}
                None => {
                    return Err(TensorError::Faulted {
                        reason: "nonempty NPU copy source has no allocation".to_owned(),
                    });
                }
            }
            PreflightSource::Npu {
                allocation,
                offset: source_offset,
            }
        } else if source.descriptor().device().kind() == DeviceKind::Cpu {
            if source.descriptor().stream() != context.stream {
                return Err(TensorError::StreamMismatch {
                    expected: context.stream,
                    actual: source.descriptor().stream(),
                });
            }
            if source.descriptor().layout() != Layout::Contiguous
                || !source.descriptor().is_contiguous()?
            {
                return Err(self.unsupported(
                    "zed.npu.copy",
                    "CPU source tensor must have canonical contiguous layout and strides",
                ));
            }
            let bytes = source.contiguous_bytes()?;
            if bytes.len() != byte_length {
                return Err(TensorError::StorageLength {
                    expected: u64::try_from(byte_length).map_err(|_| TensorError::ShapeOverflow)?,
                    actual: u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?,
                });
            }
            PreflightSource::Cpu(bytes)
        } else {
            return Err(self.unsupported(
                "zed.npu.copy",
                "source must be host-addressable CPU storage or this NPU backend instance",
            ));
        };

        self.stream(context.stream, context.cancellation)?;
        let memory = self.reserve_tensor_storage(&destination)?;
        let allocation = if storage_byte_length == 0 {
            None
        } else {
            Some(self.runtime.allocate(
                self.device.ordinal(),
                storage_byte_length,
                context.cancellation,
            )?)
        };
        if let Some(destination_allocation) = &allocation {
            self.runtime.zero_allocation(
                destination_allocation,
                storage_byte_length,
                context.cancellation,
            )?;
        }
        match (source, &allocation) {
            (
                PreflightSource::Npu {
                    allocation: Some(source),
                    offset: source_offset,
                },
                Some(destination),
            ) => {
                self.runtime.copy_device_to_device(
                    destination,
                    destination_offset,
                    &source,
                    source_offset,
                    byte_length,
                    context.cancellation,
                )?;
            }
            (PreflightSource::Cpu(source), Some(destination)) => {
                self.runtime.copy_from_host(
                    destination,
                    destination_offset,
                    source,
                    context.cancellation,
                )?;
            }
            (
                PreflightSource::Npu {
                    allocation: None, ..
                }
                | PreflightSource::Cpu(_),
                None,
            ) if byte_length == 0 => {}
            _ => {
                return Err(TensorError::Faulted {
                    reason: "NPU copy preflight and destination allocation disagree".to_owned(),
                });
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
            .require("zed.npu.event.record", OperationSupport::record_event())?;
        let stream = self.stream(context.stream, context.cancellation)?;
        let event = self.runtime.record_event(&stream, context.cancellation)?;
        self.track_event(event, context)
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.check_context(context)?;
        self.capabilities
            .require("zed.npu.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "NPU event belongs to a different backend instance".to_owned(),
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
        let completion_result = if wait_result.is_ok() || native_event.is_synchronized() {
            self.events.complete(event.stream, event.sequence).map(drop)
        } else {
            Ok(())
        };
        if let Err(wait_error) = wait_result {
            if let Err(completion_error) = completion_result {
                eprintln!(
                    "failed to retire synchronized NPU event after {wait_error}: {completion_error}"
                );
            }
            return Err(wait_error);
        }
        completion_result?;
        self.check_context(context)
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.fill", context)
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.unary", context)
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
            return self.unsupported_result("zed.npu.binary", context);
        }
        for input in [left, right] {
            self.require_descriptor(
                "zed.npu.binary.add",
                PrimitiveOperation::Binary(BinaryOperation::Add),
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
            self.storage(input)?;
        }
        self.require_descriptor(
            "zed.npu.binary.add",
            PrimitiveOperation::Binary(BinaryOperation::Add),
            TensorRole::Output,
            &output,
            context,
        )?;
        if left.descriptor().shape() != output.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "NPU add left shape {:?} does not match output shape {:?}",
                    left.descriptor().shape(),
                    output.shape()
                ),
            });
        }
        if right.descriptor().shape() != output.shape() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "NPU add right shape {:?} does not match output shape {:?}",
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
                "zed.npu.binary.add",
                "the certified AscendCL Add wrapper does not accept offset tensor views",
            ));
        }
        let elements = output.element_count()?;
        let dimensions = if output.shape().is_empty() {
            vec![1]
        } else {
            output
                .shape()
                .iter()
                .map(|dimension| i64::try_from(*dimension).map_err(|_| TensorError::ShapeOverflow))
                .collect::<Result<Vec<_>, _>>()?
        };
        if dimensions.len() > 8 {
            return Err(self.unsupported(
                "zed.npu.binary.add",
                "AscendCL array descriptors require rank 1 through 8 with positive dimensions",
            ));
        }
        let left = self.storage(left)?;
        let right = self.storage(right)?;
        let memory = self.reserve_tensor_storage(&output)?;
        if elements == 0 {
            let tensor = self.tensor_from_allocation(output, None, memory)?;
            let event = self.record_event(context)?;
            return Ok((tensor, event));
        }
        let byte_length =
            usize::try_from(output.byte_len()?).map_err(|_| TensorError::ShapeOverflow)?;
        let output_allocation =
            self.runtime
                .allocate(self.device.ordinal(), byte_length, context.cancellation)?;
        let left_allocation = left
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty NPU add left input has no allocation".to_owned(),
            })?;
        let right_allocation = right
            .allocation
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "nonempty NPU add right input has no allocation".to_owned(),
            })?;
        let element_type = match output.dtype() {
            DType::F16 => NpuElementType::F16,
            DType::F32 => NpuElementType::F32,
            _ => {
                return Err(self.unsupported(
                    "zed.npu.binary.add",
                    "only reviewed f16/f32 AscendCL Add kernels are available",
                ));
            }
        };
        let stream = self.stream(context.stream, context.cancellation)?;
        let native_event = self.runtime.add(
            &stream,
            element_type,
            &dimensions,
            left_allocation,
            right_allocation,
            &output_allocation,
            context.cancellation,
        )?;
        self.check_context(context)?;
        let tensor = self.tensor_from_allocation(output, Some(output_allocation), memory)?;
        let event = self.track_event(native_event, context)?;
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
        self.unsupported_result("zed.npu.binary-scalar", context)
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.reduction", context)
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.indexing", context)
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.resize", context)
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.convolution", context)
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.unsupported_result("zed.npu.linear-algebra", context)
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.unsupported_result("zed.npu.custom-kernel", context)
    }
}

fn npu_capability_matrix(
    device: DeviceId,
    properties: NativeDeviceProperties,
) -> Result<BackendCapabilityMatrix, TensorError> {
    if device.kind() != DeviceKind::Npu {
        return Err(TensorError::DeviceMismatch {
            expected: DeviceId::new(DeviceKind::Npu, device.ordinal()),
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

fn map_execution_error(operation: &str, device: u32, error: NpuExecutionError) -> TensorError {
    match error {
        NpuExecutionError::Load(NpuLoadError::UnsupportedTarget { target }) => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::new(DeviceKind::Npu, device),
                reason: format!("Huawei Ascend NPU is unsupported on target {target}"),
            }
        }
        NpuExecutionError::Load(load_error) => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: DeviceId::new(DeviceKind::Npu, device),
            reason: format!("certified AscendCL runtime is unavailable or invalid: {load_error}"),
        },
        NpuExecutionError::OutOfMemory {
            requested,
            capacity,
        } => TensorError::AllocationFailed {
            requested: u64::try_from(requested).unwrap_or(u64::MAX),
            reason: format!(
                "{operation}: certified NPU allocation exceeded capacity {capacity}"
            ),
        },
        NpuExecutionError::DeviceLost {
            operation: vendor_operation,
            ..
        } => TensorError::DeviceLost {
            reason: format!("{operation}: NPU device lost during {vendor_operation}"),
        },
        NpuExecutionError::InvalidCertifiedInputs { reason } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: DeviceId::new(DeviceKind::Npu, device),
            reason: format!("certified NPU execution inputs are invalid: {reason}"),
        },
        NpuExecutionError::Cancelled => TensorError::Cancelled,
        other => TensorError::Faulted {
            reason: format!("{operation}: {other}"),
        },
    }
}

fn adapter_identity_error() -> TensorError {
    TensorError::Faulted {
        reason: "NPU runtime and resource adapter kinds differ".to_owned(),
    }
}

fn tensor_byte_offset(descriptor: &TensorDescriptor) -> Result<usize, TensorError> {
    descriptor
        .offset_elements()
        .checked_mul(descriptor.dtype().byte_width())
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(TensorError::ShapeOverflow)
}

fn validate_adapter_range(
    available: usize,
    offset: usize,
    byte_length: usize,
) -> Result<(), TensorError> {
    let required = offset
        .checked_add(byte_length)
        .ok_or(TensorError::ShapeOverflow)?;
    if required > available {
        return Err(TensorError::StorageBounds {
            required: u64::try_from(required).map_err(|_| TensorError::ShapeOverflow)?,
            actual: u64::try_from(available).map_err(|_| TensorError::ShapeOverflow)?,
        });
    }
    Ok(())
}

#[cfg(test)]
fn allocation_range<'a>(
    bytes: &'a [u8],
    offset: usize,
    byte_length: usize,
    purpose: &str,
) -> Result<&'a [u8], TensorError> {
    let end = offset
        .checked_add(byte_length)
        .ok_or(TensorError::ShapeOverflow)?;
    bytes.get(offset..end).ok_or_else(|| TensorError::Faulted {
        reason: format!(
            "{purpose} range {offset}..{end} exceeds allocation length {}",
            bytes.len()
        ),
    })
}

#[cfg(test)]
fn allocation_range_mut<'a>(
    bytes: &'a mut [u8],
    offset: usize,
    byte_length: usize,
    purpose: &str,
) -> Result<&'a mut [u8], TensorError> {
    let allocation_length = bytes.len();
    let end = offset
        .checked_add(byte_length)
        .ok_or(TensorError::ShapeOverflow)?;
    bytes
        .get_mut(offset..end)
        .ok_or_else(|| TensorError::Faulted {
            reason: format!(
                "{purpose} range {offset}..{end} exceeds allocation length {allocation_length}"
            ),
        })
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

#[cfg(test)]
#[derive(Clone, Copy)]
enum InjectedFailure {
    DeviceLost,
    OutOfMemory,
}

#[cfg(test)]
struct TestRuntime {
    device_count: u32,
    memory_limit_bytes: u64,
    failure: Mutex<Option<InjectedFailure>>,
    cancel_after_next_synchronization: Mutex<Option<CancellationToken>>,
    allocation_calls: std::sync::atomic::AtomicU64,
    synchronization_calls: std::sync::atomic::AtomicU64,
}

#[cfg(test)]
impl TestRuntime {
    fn check(&self, device: u32, requested: usize) -> Result<(), TensorError> {
        if device >= self.device_count {
            return Err(TensorError::UnsupportedCapability {
                operation: "zed.npu.test.select-device".to_owned(),
                device: DeviceId::new(DeviceKind::Npu, device),
                reason: format!("device ordinal is outside test count {}", self.device_count),
            });
        }
        let mut failure = self.failure.lock().map_err(|_| TensorError::Faulted {
            reason: "NPU test failure lock is poisoned".to_owned(),
        })?;
        match failure.take() {
            Some(InjectedFailure::DeviceLost) => Err(TensorError::DeviceLost {
                reason: "injected NPU device loss".to_owned(),
            }),
            Some(InjectedFailure::OutOfMemory) => Err(TensorError::AllocationFailed {
                requested: u64::try_from(requested).unwrap_or(u64::MAX),
                reason: "injected NPU out of memory".to_owned(),
            }),
            None => Ok(()),
        }
    }

    fn validate_allocation(&self, device: u32, bytes: &[u8]) -> Result<Vec<u8>, TensorError> {
        self.check(device, bytes.len())?;
        self.allocation_calls
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(bytes.to_vec())
    }

    fn add(
        &self,
        device: u32,
        element_type: NpuElementType,
        _dimensions: &[i64],
        left: &[u8],
        right: &[u8],
    ) -> Result<Vec<u8>, TensorError> {
        self.check(device, left.len())?;
        if left.len() != right.len() {
            return Err(TensorError::Faulted {
                reason: "invalid NPU test add buffers".to_owned(),
            });
        }
        match element_type {
            NpuElementType::F16 => add_f16_bytes(left, right),
            NpuElementType::F32 => add_f32_bytes(left, right),
        }
    }

    fn synchronize(&self, device: u32) -> Result<(), TensorError> {
        self.check(device, 0)?;
        self.synchronization_calls
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if let Some(cancellation) = self
            .cancel_after_next_synchronization
            .lock()
            .map_err(|_| TensorError::Faulted {
                reason: "NPU test synchronization cancellation lock is poisoned".to_owned(),
            })?
            .take()
        {
            cancellation.cancel();
        }
        Ok(())
    }
}

#[cfg(test)]
fn add_f32_bytes(left: &[u8], right: &[u8]) -> Result<Vec<u8>, TensorError> {
    if !left.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(TensorError::Faulted {
            reason: "invalid NPU test f32 add buffers".to_owned(),
        });
    }
    left.chunks_exact(4).zip(right.chunks_exact(4)).try_fold(
        Vec::with_capacity(left.len()),
        |mut output, (left, right)| {
            let left: [u8; 4] = left.try_into().map_err(|_| TensorError::Faulted {
                reason: "invalid NPU test left f32 chunk".to_owned(),
            })?;
            let right: [u8; 4] = right.try_into().map_err(|_| TensorError::Faulted {
                reason: "invalid NPU test right f32 chunk".to_owned(),
            })?;
            output.extend_from_slice(
                &(f32::from_ne_bytes(left) + f32::from_ne_bytes(right)).to_ne_bytes(),
            );
            Ok(output)
        },
    )
}

#[cfg(test)]
fn add_f16_bytes(left: &[u8], right: &[u8]) -> Result<Vec<u8>, TensorError> {
    if !left.len().is_multiple_of(2) {
        return Err(TensorError::Faulted {
            reason: "invalid NPU test f16 add buffers".to_owned(),
        });
    }
    left.chunks_exact(2).zip(right.chunks_exact(2)).try_fold(
        Vec::with_capacity(left.len()),
        |mut output, (left, right)| {
            let left: [u8; 2] = left.try_into().map_err(|_| TensorError::Faulted {
                reason: "invalid NPU test left f16 chunk".to_owned(),
            })?;
            let right: [u8; 2] = right.try_into().map_err(|_| TensorError::Faulted {
                reason: "invalid NPU test right f16 chunk".to_owned(),
            })?;
            let value = half::f16::from_bits(u16::from_ne_bytes(left))
                + half::f16::from_bits(u16::from_ne_bytes(right));
            output.extend_from_slice(&value.to_bits().to_ne_bytes());
            Ok(output)
        },
    )
}

#[cfg(test)]
#[path = "../../tests/backends/huawei_ascend_npu_sim_model_0019.rs"]
mod tests;
