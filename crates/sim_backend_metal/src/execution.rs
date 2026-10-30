use crate::MetalExecutionAbi;
#[cfg(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use crate::{
    MAXIMUM_COMMAND_BUFFERS_PER_STREAM, METAL_ADD_F16_FUNCTION, METAL_ADD_F32_FUNCTION,
    READINESS_FUNCTION,
};
use std::{
    any::Any,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;

static NEXT_RUNTIME_IDENTITY: AtomicU64 = AtomicU64::new(1);
#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
const MAXIMUM_DEVICE_NAME_BYTES: usize = 256;
const MAXIMUM_DIAGNOSTIC_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDiagnostic(String);

impl MetalDiagnostic {
    fn bounded(mut value: String) -> Self {
        if value.len() > MAXIMUM_DIAGNOSTIC_BYTES {
            let mut boundary = MAXIMUM_DIAGNOSTIC_BYTES;
            while !value.is_char_boundary(boundary) {
                boundary -= 1;
            }
            value.truncate(boundary);
        }
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MetalDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<&str> for MetalDiagnostic {
    fn from(value: &str) -> Self {
        Self::bounded(value.to_owned())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalStorageMode {
    Shared,
    Managed,
}

impl MetalStorageMode {
    #[cfg(any(
        test,
        feature = "test-support",
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    ))]
    const fn requires_explicit_synchronization(self) -> bool {
        matches!(self, Self::Managed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalElementType {
    F16,
    F32,
}

impl MetalElementType {
    const fn byte_width(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceProperties {
    name: String,
    registry_id: u64,
    recommended_working_set_bytes: u64,
    unified_memory: bool,
}

impl MetalDeviceProperties {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn registry_id(&self) -> u64 {
        self.registry_id
    }

    pub const fn recommended_working_set_bytes(&self) -> u64 {
        self.recommended_working_set_bytes
    }

    pub const fn unified_memory(&self) -> bool {
        self.unified_memory
    }

    pub const fn storage_mode(&self) -> MetalStorageMode {
        if self.unified_memory {
            MetalStorageMode::Shared
        } else {
            MetalStorageMode::Managed
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MetalExecutionError {
    #[error("Metal execution target is unsupported: {target}")]
    UnsupportedTarget { target: MetalDiagnostic },
    #[error("MTLCreateSystemDefaultDevice returned no Metal device")]
    NoSystemDevice,
    #[error("Metal execution ABI is invalid: {reason}")]
    InvalidAbi { reason: MetalDiagnostic },
    #[error("certified Metal execution inputs are invalid: {reason}")]
    InvalidCertifiedInputs { reason: MetalDiagnostic },
    #[error("Metal execution function {function} is missing")]
    MissingFunction { function: MetalDiagnostic },
    #[error("failed to create Metal execution pipeline {function}: {reason}")]
    PipelineCreation {
        function: MetalDiagnostic,
        reason: MetalDiagnostic,
    },
    #[error("Metal allocation of {requested} bytes failed")]
    OutOfMemory { requested: u64 },
    #[error("Metal resource belongs to a different certified runtime")]
    ForeignResource,
    #[error("Metal resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: u64,
        length: u64,
        available: u64,
    },
    #[error("Metal element count {elements} exceeds the reviewed u32 dispatch ABI")]
    ElementCount { elements: u64 },
    #[error("Metal command failed with code {code}: {reason}")]
    CommandFailed { code: i64, reason: MetalDiagnostic },
    #[error("Metal device was lost with command error code {code}")]
    DeviceLost { code: i64 },
}

struct CapacityTracker {
    limit: u64,
    current: AtomicU64,
    peak: AtomicU64,
}

impl CapacityTracker {
    fn reserve(self: &Arc<Self>, bytes: u64) -> Result<CapacityReservation, MetalExecutionError> {
        let mut current = self.current.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .filter(|next| *next <= self.limit)
                .ok_or(MetalExecutionError::OutOfMemory { requested: bytes })?;
            match self.current.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(CapacityReservation {
                        bytes,
                        tracker: self.clone(),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

struct CapacityReservation {
    bytes: u64,
    tracker: Arc<CapacityTracker>,
}

impl Drop for CapacityReservation {
    fn drop(&mut self) {
        let previous = self.tracker.current.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes);
    }
}

struct CertifiedInputs {
    readiness_metallib: Arc<[u8]>,
    tensor_ops_metallib: Arc<[u8]>,
    _certification: Arc<dyn Any + Send + Sync>,
}

#[cfg(not(any(test, feature = "test-support")))]
type RuntimeImplementation = platform::Runtime;
#[cfg(not(any(test, feature = "test-support")))]
type AllocationImplementation = platform::Allocation;
#[cfg(not(any(test, feature = "test-support")))]
type StreamImplementation = platform::Stream;
#[cfg(not(any(test, feature = "test-support")))]
type EventImplementation = platform::Event;

#[cfg(any(test, feature = "test-support"))]
enum RuntimeImplementation {
    Platform(platform::Runtime),
    Fake(test_fake::Runtime),
}

#[cfg(any(test, feature = "test-support"))]
enum AllocationImplementation {
    Platform(platform::Allocation),
    Fake(test_fake::Allocation),
}

#[cfg(any(test, feature = "test-support"))]
enum StreamImplementation {
    Platform(platform::Stream),
    Fake(test_fake::Stream),
}

#[cfg(any(test, feature = "test-support"))]
enum EventImplementation {
    Platform(platform::Event),
    Fake(test_fake::Event),
}

#[cfg(any(test, feature = "test-support"))]
impl RuntimeImplementation {
    fn allocate(
        &self,
        byte_length: u64,
        storage_mode: MetalStorageMode,
    ) -> Result<AllocationImplementation, MetalExecutionError> {
        match self {
            Self::Platform(runtime) => runtime
                .allocate(byte_length, storage_mode)
                .map(AllocationImplementation::Platform),
            Self::Fake(runtime) => runtime
                .allocate(byte_length, storage_mode)
                .map(AllocationImplementation::Fake),
        }
    }

    fn create_stream(&self) -> Result<StreamImplementation, MetalExecutionError> {
        match self {
            Self::Platform(runtime) => runtime.create_stream().map(StreamImplementation::Platform),
            Self::Fake(runtime) => runtime.create_stream().map(StreamImplementation::Fake),
        }
    }

    fn copy_host_to_device(
        &self,
        stream: &StreamImplementation,
        destination: &AllocationImplementation,
        storage_mode: MetalStorageMode,
        destination_offset: u64,
        bytes: &[u8],
    ) -> Result<(), MetalExecutionError> {
        match (self, stream, destination) {
            (
                Self::Platform(runtime),
                StreamImplementation::Platform(stream),
                AllocationImplementation::Platform(destination),
            ) => runtime.copy_host_to_device(
                stream,
                destination,
                storage_mode,
                destination_offset,
                bytes,
            ),
            (
                Self::Fake(runtime),
                StreamImplementation::Fake(stream),
                AllocationImplementation::Fake(destination),
            ) => runtime.copy_host_to_device(
                stream,
                destination,
                storage_mode,
                destination_offset,
                bytes,
            ),
            _ => Err(MetalExecutionError::ForeignResource),
        }
    }

    fn copy_device_to_host(
        &self,
        stream: &StreamImplementation,
        source: &AllocationImplementation,
        storage_mode: MetalStorageMode,
        source_offset: u64,
        bytes: &mut [u8],
    ) -> Result<(), MetalExecutionError> {
        match (self, stream, source) {
            (
                Self::Platform(runtime),
                StreamImplementation::Platform(stream),
                AllocationImplementation::Platform(source),
            ) => runtime.copy_device_to_host(stream, source, storage_mode, source_offset, bytes),
            (
                Self::Fake(runtime),
                StreamImplementation::Fake(stream),
                AllocationImplementation::Fake(source),
            ) => runtime.copy_device_to_host(stream, source, storage_mode, source_offset, bytes),
            _ => Err(MetalExecutionError::ForeignResource),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_add(
        &self,
        stream: &StreamImplementation,
        element_type: MetalElementType,
        left: &AllocationImplementation,
        right: &AllocationImplementation,
        output: &AllocationImplementation,
        output_storage_mode: MetalStorageMode,
        elements: u32,
    ) -> Result<EventImplementation, MetalExecutionError> {
        match (self, stream, left, right, output) {
            (
                Self::Platform(runtime),
                StreamImplementation::Platform(stream),
                AllocationImplementation::Platform(left),
                AllocationImplementation::Platform(right),
                AllocationImplementation::Platform(output),
            ) => runtime
                .dispatch_add(
                    stream,
                    element_type,
                    left,
                    right,
                    output,
                    output_storage_mode,
                    elements,
                )
                .map(EventImplementation::Platform),
            (
                Self::Fake(runtime),
                StreamImplementation::Fake(stream),
                AllocationImplementation::Fake(left),
                AllocationImplementation::Fake(right),
                AllocationImplementation::Fake(output),
            ) => runtime
                .dispatch_add(
                    stream,
                    element_type,
                    left,
                    right,
                    output,
                    output_storage_mode,
                    elements,
                )
                .map(EventImplementation::Fake),
            _ => Err(MetalExecutionError::ForeignResource),
        }
    }

    fn record_event(
        &self,
        stream: &StreamImplementation,
    ) -> Result<EventImplementation, MetalExecutionError> {
        match (self, stream) {
            (Self::Platform(runtime), StreamImplementation::Platform(stream)) => runtime
                .record_event(stream)
                .map(EventImplementation::Platform),
            (Self::Fake(runtime), StreamImplementation::Fake(stream)) => {
                runtime.record_event(stream).map(EventImplementation::Fake)
            }
            _ => Err(MetalExecutionError::ForeignResource),
        }
    }

    fn wait_event(&self, event: &EventImplementation) -> Result<(), MetalExecutionError> {
        match (self, event) {
            (Self::Platform(runtime), EventImplementation::Platform(event)) => {
                runtime.wait_event(event)
            }
            (Self::Fake(runtime), EventImplementation::Fake(event)) => runtime.wait_event(event),
            _ => Err(MetalExecutionError::ForeignResource),
        }
    }
}

impl CertifiedInputs {
    fn new(
        readiness_metallib: Arc<[u8]>,
        tensor_ops_metallib: Arc<[u8]>,
        certification: Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<Self>, MetalExecutionError> {
        if readiness_metallib.is_empty()
            || readiness_metallib.len() > 4 * 1024 * 1024
            || tensor_ops_metallib.is_empty()
            || tensor_ops_metallib.len() > 4 * 1024 * 1024
        {
            return Err(MetalExecutionError::InvalidCertifiedInputs {
                reason: "each certified metallib must be nonempty and at most 4 MiB".into(),
            });
        }
        Ok(Arc::new(Self {
            readiness_metallib,
            tensor_ops_metallib,
            _certification: certification,
        }))
    }
}

fn next_runtime_identity() -> Result<u64, MetalExecutionError> {
    NEXT_RUNTIME_IDENTITY
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| MetalExecutionError::InvalidCertifiedInputs {
            reason: "runtime identity sequence overflowed".into(),
        })
}

#[cfg(not(any(test, feature = "test-support")))]
fn platform_runtime(runtime: platform::Runtime) -> RuntimeImplementation {
    runtime
}

#[cfg(any(test, feature = "test-support"))]
fn platform_runtime(runtime: platform::Runtime) -> RuntimeImplementation {
    RuntimeImplementation::Platform(runtime)
}

struct MetalRuntimeInner {
    identity: u64,
    properties: MetalDeviceProperties,
    capacity: Arc<CapacityTracker>,
    platform: RuntimeImplementation,
    _certified: Arc<CertifiedInputs>,
}

#[derive(Clone)]
pub struct MetalRuntime {
    inner: Arc<MetalRuntimeInner>,
}

impl fmt::Debug for MetalRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalRuntime")
            .field("identity", &self.inner.identity)
            .field("properties", &self.inner.properties)
            .finish_non_exhaustive()
    }
}

struct MetalAllocationInner {
    runtime_identity: u64,
    byte_length: u64,
    storage_mode: MetalStorageMode,
    platform: AllocationImplementation,
    _reservation: CapacityReservation,
    _runtime: Arc<MetalRuntimeInner>,
}

#[derive(Clone)]
pub struct MetalAllocation {
    inner: Arc<MetalAllocationInner>,
}

impl fmt::Debug for MetalAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalAllocation")
            .field("byte_length", &self.inner.byte_length)
            .field("storage_mode", &self.inner.storage_mode)
            .finish_non_exhaustive()
    }
}

impl MetalAllocation {
    pub fn byte_length(&self) -> u64 {
        self.inner.byte_length
    }

    pub fn storage_mode(&self) -> MetalStorageMode {
        self.inner.storage_mode
    }
}

struct MetalStreamInner {
    runtime_identity: u64,
    platform: StreamImplementation,
    _runtime: Arc<MetalRuntimeInner>,
}

#[derive(Clone)]
pub struct MetalStream {
    inner: Arc<MetalStreamInner>,
}

impl fmt::Debug for MetalStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalStream")
            .finish_non_exhaustive()
    }
}

struct MetalEventInner {
    runtime_identity: u64,
    platform: EventImplementation,
    _runtime: Arc<MetalRuntimeInner>,
}

#[derive(Clone)]
pub struct MetalEvent {
    inner: Arc<MetalEventInner>,
}

impl fmt::Debug for MetalEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("MetalEvent").finish_non_exhaustive()
    }
}

impl MetalRuntime {
    /// Constructs a Metal runtime only after the caller has independently certified the
    /// reviewed framework ABI and both signed Task 113 metallibs.
    ///
    /// # Safety
    ///
    /// Both metallibs must be the exact retained bytes certified by the canonical
    /// `NativeFfiRegistry` contracts for the current target. `certification` must retain every
    /// package, framework, and metallib certificate for at least the returned runtime's lifetime.
    pub unsafe fn from_certified_metallibs(
        readiness_metallib: Arc<[u8]>,
        tensor_ops_metallib: Arc<[u8]>,
        certification: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, MetalExecutionError> {
        let contract =
            MetalExecutionAbi::embedded().map_err(|error| MetalExecutionError::InvalidAbi {
                reason: MetalDiagnostic::bounded(error.to_string()),
            })?;
        let certified =
            CertifiedInputs::new(readiness_metallib, tensor_ops_metallib, certification)?;
        let identity = next_runtime_identity()?;
        let (platform, properties) = platform::Runtime::new(
            &contract,
            &certified.readiness_metallib,
            &certified.tensor_ops_metallib,
        )?;
        Ok(Self {
            inner: Arc::new(MetalRuntimeInner {
                identity,
                capacity: Arc::new(CapacityTracker {
                    limit: properties.recommended_working_set_bytes,
                    current: AtomicU64::new(0),
                    peak: AtomicU64::new(0),
                }),
                properties,
                platform: platform_runtime(platform),
                _certified: certified,
            }),
        })
    }

    #[cfg(test)]
    fn from_fake(
        config: test_fake::Config,
    ) -> Result<(Self, test_fake::Control), MetalExecutionError> {
        let certified = CertifiedInputs::new(
            Arc::<[u8]>::from([1_u8]),
            Arc::<[u8]>::from([1_u8]),
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
        )?;
        let identity = next_runtime_identity()?;
        let (platform, control) = test_fake::Runtime::new(config);
        let properties = MetalDeviceProperties {
            name: "Deterministic fake Metal device".to_owned(),
            registry_id: identity,
            recommended_working_set_bytes: config.capacity_bytes,
            unified_memory: config.storage_mode == MetalStorageMode::Shared,
        };
        Ok((
            Self {
                inner: Arc::new(MetalRuntimeInner {
                    identity,
                    capacity: Arc::new(CapacityTracker {
                        limit: properties.recommended_working_set_bytes,
                        current: AtomicU64::new(0),
                        peak: AtomicU64::new(0),
                    }),
                    properties,
                    platform: RuntimeImplementation::Fake(platform),
                    _certified: certified,
                }),
            },
            control,
        ))
    }

    #[cfg(feature = "test-support")]
    pub fn for_test_harness(
        capacity_bytes: u64,
        unified_memory: bool,
    ) -> Result<Self, MetalExecutionError> {
        if capacity_bytes == 0 {
            return Err(MetalExecutionError::InvalidCertifiedInputs {
                reason: "test harness capacity must be nonzero".into(),
            });
        }
        let certified = CertifiedInputs::new(
            Arc::<[u8]>::from([1_u8]),
            Arc::<[u8]>::from([1_u8]),
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
        )?;
        let identity = next_runtime_identity()?;
        let (platform, _) = test_fake::Runtime::new(test_fake::Config {
            storage_mode: if unified_memory {
                MetalStorageMode::Shared
            } else {
                MetalStorageMode::Managed
            },
            capacity_bytes,
        });
        Ok(Self {
            inner: Arc::new(MetalRuntimeInner {
                identity,
                capacity: Arc::new(CapacityTracker {
                    limit: capacity_bytes,
                    current: AtomicU64::new(0),
                    peak: AtomicU64::new(0),
                }),
                properties: MetalDeviceProperties {
                    name: "Injected deterministic Metal device".to_owned(),
                    registry_id: identity,
                    recommended_working_set_bytes: capacity_bytes,
                    unified_memory,
                },
                platform: RuntimeImplementation::Fake(platform),
                _certified: certified,
            }),
        })
    }

    #[cfg(feature = "test-support")]
    pub fn inject_test_command_failure(&self, code: i64) -> Result<(), MetalExecutionError> {
        match &self.inner.platform {
            RuntimeImplementation::Fake(runtime) => runtime.fail_next_command(code),
            RuntimeImplementation::Platform(_) => {
                Err(MetalExecutionError::InvalidCertifiedInputs {
                    reason: "failure injection is restricted to the deterministic test harness"
                        .into(),
                })
            }
        }
    }

    pub fn properties(&self) -> &MetalDeviceProperties {
        &self.inner.properties
    }

    pub fn current_allocation_bytes(&self) -> u64 {
        self.inner.capacity.current.load(Ordering::Acquire)
    }

    pub fn peak_allocation_bytes(&self) -> u64 {
        self.inner.capacity.peak.load(Ordering::Acquire)
    }

    pub fn allocate(&self, byte_length: u64) -> Result<MetalAllocation, MetalExecutionError> {
        let storage_mode = self.inner.properties.storage_mode();
        let reservation = self.inner.capacity.reserve(byte_length)?;
        let platform = self.inner.platform.allocate(byte_length, storage_mode)?;
        Ok(MetalAllocation {
            inner: Arc::new(MetalAllocationInner {
                runtime_identity: self.inner.identity,
                _runtime: self.inner.clone(),
                byte_length,
                storage_mode,
                platform,
                _reservation: reservation,
            }),
        })
    }

    pub fn create_stream(&self) -> Result<MetalStream, MetalExecutionError> {
        Ok(MetalStream {
            inner: Arc::new(MetalStreamInner {
                runtime_identity: self.inner.identity,
                _runtime: self.inner.clone(),
                platform: self.inner.platform.create_stream()?,
            }),
        })
    }

    pub fn copy_host_to_device(
        &self,
        stream: &MetalStream,
        destination: &MetalAllocation,
        destination_offset: u64,
        bytes: &[u8],
    ) -> Result<(), MetalExecutionError> {
        self.require_stream(stream)?;
        self.require_allocation(destination)?;
        let length =
            u64::try_from(bytes.len()).map_err(|_| MetalExecutionError::ResourceBounds {
                offset: destination_offset,
                length: u64::MAX,
                available: destination.byte_length(),
            })?;
        require_range(destination.byte_length(), destination_offset, length)?;
        self.inner.platform.copy_host_to_device(
            &stream.inner.platform,
            &destination.inner.platform,
            destination.inner.storage_mode,
            destination_offset,
            bytes,
        )
    }

    pub fn copy_device_to_host(
        &self,
        stream: &MetalStream,
        source: &MetalAllocation,
        source_offset: u64,
        bytes: &mut [u8],
    ) -> Result<(), MetalExecutionError> {
        self.require_stream(stream)?;
        self.require_allocation(source)?;
        let length =
            u64::try_from(bytes.len()).map_err(|_| MetalExecutionError::ResourceBounds {
                offset: source_offset,
                length: u64::MAX,
                available: source.byte_length(),
            })?;
        require_range(source.byte_length(), source_offset, length)?;
        self.inner.platform.copy_device_to_host(
            &stream.inner.platform,
            &source.inner.platform,
            source.inner.storage_mode,
            source_offset,
            bytes,
        )
    }

    pub fn dispatch_add(
        &self,
        stream: &MetalStream,
        element_type: MetalElementType,
        left: &MetalAllocation,
        right: &MetalAllocation,
        output: &MetalAllocation,
        elements: u64,
    ) -> Result<MetalEvent, MetalExecutionError> {
        self.require_stream(stream)?;
        for allocation in [left, right, output] {
            self.require_allocation(allocation)?;
            let required = elements
                .checked_mul(element_type.byte_width())
                .ok_or(MetalExecutionError::ElementCount { elements })?;
            require_range(allocation.byte_length(), 0, required)?;
        }
        let elements =
            u32::try_from(elements).map_err(|_| MetalExecutionError::ElementCount { elements })?;
        let platform = self.inner.platform.dispatch_add(
            &stream.inner.platform,
            element_type,
            &left.inner.platform,
            &right.inner.platform,
            &output.inner.platform,
            output.inner.storage_mode,
            elements,
        )?;
        Ok(MetalEvent {
            inner: Arc::new(MetalEventInner {
                runtime_identity: self.inner.identity,
                _runtime: self.inner.clone(),
                platform,
            }),
        })
    }

    pub fn record_event(&self, stream: &MetalStream) -> Result<MetalEvent, MetalExecutionError> {
        self.require_stream(stream)?;
        Ok(MetalEvent {
            inner: Arc::new(MetalEventInner {
                runtime_identity: self.inner.identity,
                _runtime: self.inner.clone(),
                platform: self.inner.platform.record_event(&stream.inner.platform)?,
            }),
        })
    }

    pub fn wait_event(&self, event: &MetalEvent) -> Result<(), MetalExecutionError> {
        require_identity(self.inner.identity, event.inner.runtime_identity)?;
        self.inner.platform.wait_event(&event.inner.platform)
    }

    fn require_stream(&self, stream: &MetalStream) -> Result<(), MetalExecutionError> {
        require_identity(self.inner.identity, stream.inner.runtime_identity)
    }

    fn require_allocation(&self, allocation: &MetalAllocation) -> Result<(), MetalExecutionError> {
        require_identity(self.inner.identity, allocation.inner.runtime_identity)
    }
}

fn require_identity(expected: u64, actual: u64) -> Result<(), MetalExecutionError> {
    if actual != expected {
        return Err(MetalExecutionError::ForeignResource);
    }
    Ok(())
}

fn require_range(available: u64, offset: u64, length: u64) -> Result<(), MetalExecutionError> {
    if offset.checked_add(length).is_none_or(|end| end > available) {
        return Err(MetalExecutionError::ResourceBounds {
            offset,
            length,
            available,
        });
    }
    Ok(())
}

#[cfg(any(
    test,
    feature = "test-support",
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn map_command_error(code: i64, reason: String) -> MetalExecutionError {
    match code {
        8 => MetalExecutionError::OutOfMemory { requested: 0 },
        3 | 4 | 11 => MetalExecutionError::DeviceLost { code },
        _ => MetalExecutionError::CommandFailed {
            code,
            reason: MetalDiagnostic::bounded(reason),
        },
    }
}

fn bind_execution_contract(contract: &MetalExecutionAbi) -> Result<(), MetalExecutionError> {
    contract
        .validate()
        .map_err(|error| MetalExecutionError::InvalidAbi {
            reason: MetalDiagnostic::bounded(error.to_string()),
        })
}

#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn bounded_device_name(name: String) -> Result<String, MetalExecutionError> {
    if name.len() > MAXIMUM_DEVICE_NAME_BYTES {
        return Err(MetalExecutionError::InvalidCertifiedInputs {
            reason: "Metal device name exceeds the reviewed 256-byte boundary".into(),
        });
    }
    Ok(name)
}

#[cfg(any(test, feature = "test-support"))]
mod test_fake {
    use super::*;
    use std::sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    #[derive(Clone, Copy)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) struct Config {
        pub(super) storage_mode: MetalStorageMode,
        pub(super) capacity_bytes: u64,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(super) enum CommandOutcome {
        Completed,
        Failed(i64),
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(super) enum TraceOperation {
        Allocate {
            allocation: u64,
            bytes: u64,
            storage_mode: MetalStorageMode,
        },
        RejectAllocation {
            bytes: u64,
        },
        CreateStream {
            stream: u64,
        },
        RecordEvent {
            stream: u64,
            event: u64,
        },
        HostWrite {
            allocation: u64,
            offset: u64,
            bytes: u64,
        },
        DidModify {
            allocation: u64,
            offset: u64,
            bytes: u64,
        },
        DispatchAdd {
            stream: u64,
            event: u64,
            elements: u32,
            element_type: MetalElementType,
        },
        SynchronizeResource {
            allocation: u64,
        },
        Commit {
            event: u64,
        },
        Wait {
            event: u64,
            outcome: CommandOutcome,
        },
        HostRead {
            allocation: u64,
            offset: u64,
            bytes: u64,
        },
    }

    struct State {
        capacity_bytes: u64,
        allocated_bytes: AtomicU64,
        next_allocation: AtomicU64,
        next_stream: AtomicU64,
        next_event: AtomicU64,
        next_command_error: Mutex<Option<i64>>,
        trace: Mutex<Vec<TraceOperation>>,
    }

    #[derive(Clone)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) struct Control {
        state: Arc<State>,
    }

    #[cfg_attr(not(test), allow(dead_code))]
    impl Control {
        pub(super) fn fail_next_command(&self, code: i64) -> Result<(), MetalExecutionError> {
            *lock(&self.state.next_command_error, "next command error")? = Some(code);
            Ok(())
        }

        pub(super) fn trace(&self) -> Result<Vec<TraceOperation>, MetalExecutionError> {
            Ok(lock(&self.state.trace, "execution trace")?.clone())
        }
    }

    pub(super) struct Runtime {
        state: Arc<State>,
    }

    pub(super) struct Allocation {
        identifier: u64,
        reserved_bytes: u64,
        bytes: Mutex<Vec<u8>>,
        state: Arc<State>,
    }

    pub(super) struct Stream {
        identifier: u64,
    }

    pub(super) struct Event {
        identifier: u64,
        error_code: Option<i64>,
        completed: AtomicBool,
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            self.state
                .allocated_bytes
                .fetch_sub(self.reserved_bytes, Ordering::AcqRel);
        }
    }

    impl Runtime {
        pub(super) fn new(config: Config) -> (Self, Control) {
            let state = Arc::new(State {
                capacity_bytes: config.capacity_bytes,
                allocated_bytes: AtomicU64::new(0),
                next_allocation: AtomicU64::new(1),
                next_stream: AtomicU64::new(1),
                next_event: AtomicU64::new(1),
                next_command_error: Mutex::new(None),
                trace: Mutex::new(Vec::new()),
            });
            (
                Self {
                    state: state.clone(),
                },
                Control { state },
            )
        }

        #[cfg(feature = "test-support")]
        pub(super) fn fail_next_command(&self, code: i64) -> Result<(), MetalExecutionError> {
            *lock(&self.state.next_command_error, "next command error")? = Some(code);
            Ok(())
        }

        pub(super) fn allocate(
            &self,
            byte_length: u64,
            storage_mode: MetalStorageMode,
        ) -> Result<Allocation, MetalExecutionError> {
            if usize::try_from(byte_length).is_err() || !self.reserve(byte_length) {
                self.trace(TraceOperation::RejectAllocation { bytes: byte_length })?;
                return Err(MetalExecutionError::OutOfMemory {
                    requested: byte_length,
                });
            }
            let length =
                usize::try_from(byte_length).map_err(|_| MetalExecutionError::OutOfMemory {
                    requested: byte_length,
                })?;
            let identifier = self.state.next_allocation.fetch_add(1, Ordering::AcqRel);
            self.trace(TraceOperation::Allocate {
                allocation: identifier,
                bytes: byte_length,
                storage_mode,
            })?;
            Ok(Allocation {
                identifier,
                reserved_bytes: byte_length,
                bytes: Mutex::new(vec![0; length]),
                state: self.state.clone(),
            })
        }

        pub(super) fn create_stream(&self) -> Result<Stream, MetalExecutionError> {
            let identifier = self.state.next_stream.fetch_add(1, Ordering::AcqRel);
            self.trace(TraceOperation::CreateStream { stream: identifier })?;
            Ok(Stream { identifier })
        }

        pub(super) fn copy_host_to_device(
            &self,
            stream: &Stream,
            destination: &Allocation,
            storage_mode: MetalStorageMode,
            destination_offset: u64,
            bytes: &[u8],
        ) -> Result<(), MetalExecutionError> {
            let event = self.record_event(stream)?;
            self.wait_event(&event)?;
            destination.write(destination_offset, bytes)?;
            let byte_length = u64::try_from(bytes.len())
                .map_err(|_| internal_error("host-to-device transfer length does not fit u64"))?;
            self.trace(TraceOperation::HostWrite {
                allocation: destination.identifier,
                offset: destination_offset,
                bytes: byte_length,
            })?;
            if storage_mode.requires_explicit_synchronization() {
                self.trace(TraceOperation::DidModify {
                    allocation: destination.identifier,
                    offset: destination_offset,
                    bytes: byte_length,
                })?;
            }
            Ok(())
        }

        pub(super) fn copy_device_to_host(
            &self,
            _stream: &Stream,
            source: &Allocation,
            storage_mode: MetalStorageMode,
            source_offset: u64,
            bytes: &mut [u8],
        ) -> Result<(), MetalExecutionError> {
            if storage_mode.requires_explicit_synchronization() {
                self.trace(TraceOperation::SynchronizeResource {
                    allocation: source.identifier,
                })?;
            }
            let event = self.command_event()?;
            self.wait_event(&event)?;
            source.read(source_offset, bytes)?;
            self.trace(TraceOperation::HostRead {
                allocation: source.identifier,
                offset: source_offset,
                bytes: u64::try_from(bytes.len()).map_err(|_| {
                    internal_error("device-to-host transfer length does not fit u64")
                })?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            stream: &Stream,
            element_type: MetalElementType,
            left: &Allocation,
            right: &Allocation,
            output: &Allocation,
            output_storage_mode: MetalStorageMode,
            elements: u32,
        ) -> Result<Event, MetalExecutionError> {
            let event = self.next_event()?;
            self.trace(TraceOperation::DispatchAdd {
                stream: stream.identifier,
                event: event.identifier,
                elements,
                element_type,
            })?;
            if event.error_code.is_none() {
                match element_type {
                    MetalElementType::F32 => add_f32(left, right, output, elements)?,
                    MetalElementType::F16 => add_f16(left, right, output, elements)?,
                }
            }
            if output_storage_mode.requires_explicit_synchronization() {
                self.trace(TraceOperation::SynchronizeResource {
                    allocation: output.identifier,
                })?;
            }
            self.trace(TraceOperation::Commit {
                event: event.identifier,
            })?;
            Ok(event)
        }

        pub(super) fn record_event(&self, stream: &Stream) -> Result<Event, MetalExecutionError> {
            let event = self.next_event()?;
            self.trace(TraceOperation::RecordEvent {
                stream: stream.identifier,
                event: event.identifier,
            })?;
            self.trace(TraceOperation::Commit {
                event: event.identifier,
            })?;
            Ok(event)
        }

        pub(super) fn wait_event(&self, event: &Event) -> Result<(), MetalExecutionError> {
            event.completed.store(true, Ordering::Release);
            let outcome = event
                .error_code
                .map_or(CommandOutcome::Completed, CommandOutcome::Failed);
            self.trace(TraceOperation::Wait {
                event: event.identifier,
                outcome,
            })?;
            event.error_code.map_or(Ok(()), |code| {
                Err(map_command_error(
                    code,
                    "injected deterministic Metal command failure".to_owned(),
                ))
            })
        }

        fn reserve(&self, requested: u64) -> bool {
            let mut current = self.state.allocated_bytes.load(Ordering::Acquire);
            loop {
                let Some(next) = current.checked_add(requested) else {
                    return false;
                };
                if next > self.state.capacity_bytes {
                    return false;
                }
                match self.state.allocated_bytes.compare_exchange_weak(
                    current,
                    next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return true,
                    Err(observed) => current = observed,
                }
            }
        }

        fn next_event(&self) -> Result<Event, MetalExecutionError> {
            Ok(Event {
                identifier: self.state.next_event.fetch_add(1, Ordering::AcqRel),
                error_code: lock(&self.state.next_command_error, "next command error")?.take(),
                completed: AtomicBool::new(false),
            })
        }

        fn command_event(&self) -> Result<Event, MetalExecutionError> {
            let event = self.next_event()?;
            self.trace(TraceOperation::Commit {
                event: event.identifier,
            })?;
            Ok(event)
        }

        fn trace(&self, operation: TraceOperation) -> Result<(), MetalExecutionError> {
            lock(&self.state.trace, "execution trace")?.push(operation);
            Ok(())
        }
    }

    impl Allocation {
        fn write(&self, offset: u64, source: &[u8]) -> Result<(), MetalExecutionError> {
            let start = usize::try_from(offset)
                .map_err(|_| internal_error("allocation offset does not fit usize"))?;
            let end = start
                .checked_add(source.len())
                .ok_or_else(|| internal_error("allocation write range overflowed"))?;
            let mut bytes = lock(&self.bytes, "allocation bytes")?;
            let destination = bytes
                .get_mut(start..end)
                .ok_or_else(|| internal_error("allocation write exceeded fake storage"))?;
            destination.copy_from_slice(source);
            Ok(())
        }

        fn read(&self, offset: u64, destination: &mut [u8]) -> Result<(), MetalExecutionError> {
            let start = usize::try_from(offset)
                .map_err(|_| internal_error("allocation offset does not fit usize"))?;
            let end = start
                .checked_add(destination.len())
                .ok_or_else(|| internal_error("allocation read range overflowed"))?;
            let bytes = lock(&self.bytes, "allocation bytes")?;
            let source = bytes
                .get(start..end)
                .ok_or_else(|| internal_error("allocation read exceeded fake storage"))?;
            destination.copy_from_slice(source);
            Ok(())
        }

        fn snapshot(&self, length: usize) -> Result<Vec<u8>, MetalExecutionError> {
            let bytes = lock(&self.bytes, "allocation bytes")?;
            Ok(bytes
                .get(..length)
                .ok_or_else(|| internal_error("Add input exceeded fake storage"))?
                .to_vec())
        }
    }

    fn add_f32(
        left: &Allocation,
        right: &Allocation,
        output: &Allocation,
        elements: u32,
    ) -> Result<(), MetalExecutionError> {
        let byte_length = usize::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| internal_error("F32 Add byte length overflowed"))?;
        let left = left.snapshot(byte_length)?;
        let right = right.snapshot(byte_length)?;
        let mut result = Vec::with_capacity(byte_length);
        for (left, right) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
            let left = <[u8; 4]>::try_from(left)
                .map(f32::from_le_bytes)
                .map_err(|_| internal_error("invalid left F32 lane"))?;
            let right = <[u8; 4]>::try_from(right)
                .map(f32::from_le_bytes)
                .map_err(|_| internal_error("invalid right F32 lane"))?;
            result.extend_from_slice(&(left + right).to_le_bytes());
        }
        output.write(0, &result)
    }

    fn add_f16(
        left: &Allocation,
        right: &Allocation,
        output: &Allocation,
        elements: u32,
    ) -> Result<(), MetalExecutionError> {
        let byte_length = usize::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| internal_error("F16 Add byte length overflowed"))?;
        let left = left.snapshot(byte_length)?;
        let right = right.snapshot(byte_length)?;
        let mut result = Vec::with_capacity(byte_length);
        for (left, right) in left.chunks_exact(2).zip(right.chunks_exact(2)) {
            let left = <[u8; 2]>::try_from(left)
                .map(u16::from_le_bytes)
                .map_err(|_| internal_error("invalid left F16 lane"))?;
            let right = <[u8; 2]>::try_from(right)
                .map(u16::from_le_bytes)
                .map_err(|_| internal_error("invalid right F16 lane"))?;
            let sum = f32_to_f16(f16_to_f32(left) + f16_to_f32(right));
            result.extend_from_slice(&sum.to_le_bytes());
        }
        output.write(0, &result)
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = (u32::from(bits & 0x8000)) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let fraction = u32::from(bits & 0x03ff);
        let converted = match exponent {
            0 if fraction == 0 => sign,
            0 => {
                let leading = fraction.leading_zeros() - 22;
                let normalized = fraction << (leading + 1);
                let exponent = 127_u32 - 15 - leading;
                sign | (exponent << 23) | ((normalized & 0x03ff) << 13)
            }
            0x1f => sign | 0x7f80_0000 | (fraction << 13),
            _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
        };
        f32::from_bits(converted)
    }

    fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let fraction = bits & 0x7f_ffff;
        if exponent == 0xff {
            return sign | 0x7c00 | if fraction == 0 { 0 } else { 0x0200 };
        }
        let half_exponent = exponent - 127 + 15;
        if half_exponent >= 0x1f {
            return sign | 0x7c00;
        }
        if half_exponent <= 0 {
            if half_exponent < -10 {
                return sign;
            }
            let mantissa = fraction | 0x80_0000;
            let shift = 14 - half_exponent;
            let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
            return sign | rounded as u16;
        }
        let rounded = fraction + 0x0fff + ((fraction >> 13) & 1);
        if rounded & 0x80_0000 != 0 {
            let exponent = half_exponent + 1;
            if exponent >= 0x1f {
                sign | 0x7c00
            } else {
                sign | ((exponent as u16) << 10)
            }
        } else {
            sign | ((half_exponent as u16) << 10) | ((rounded >> 13) as u16)
        }
    }

    fn lock<'a, Value>(
        mutex: &'a Mutex<Value>,
        subject: &'static str,
    ) -> Result<MutexGuard<'a, Value>, MetalExecutionError> {
        mutex
            .lock()
            .map_err(|_| internal_error(&format!("fake Metal {subject} lock was poisoned")))
    }

    fn internal_error(reason: &str) -> MetalExecutionError {
        MetalExecutionError::CommandFailed {
            code: -1,
            reason: MetalDiagnostic::bounded(reason.to_owned()),
        }
    }
}

#[cfg(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod platform {
    use super::*;
    use metal::{
        BlitCommandEncoderRef, Buffer, CommandBuffer, CommandBufferRef, CommandQueue,
        ComputeCommandEncoderRef, ComputePipelineState, Device, MTLBlitCommandEncoder, MTLBuffer,
        MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize, NSRange,
        foreign_types::{ForeignType, ForeignTypeRef},
    };
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    use std::ptr::NonNull;

    unsafe trait NullableMetalResourceCalls {
        unsafe fn new_buffer(
            device: *mut MTLDevice,
            byte_length: u64,
            options: MTLResourceOptions,
        ) -> *mut MTLBuffer;
        unsafe fn new_command_queue(
            device: *mut MTLDevice,
            maximum_command_buffers: u64,
        ) -> *mut MTLCommandQueue;
        unsafe fn command_buffer(queue: *mut MTLCommandQueue) -> *mut MTLCommandBuffer;
        unsafe fn blit_command_encoder(
            command_buffer: *mut MTLCommandBuffer,
        ) -> *mut MTLBlitCommandEncoder;
        unsafe fn compute_command_encoder(
            command_buffer: *mut MTLCommandBuffer,
        ) -> *mut MTLComputeCommandEncoder;
    }

    struct SystemNullableMetalResourceCalls;

    unsafe impl NullableMetalResourceCalls for SystemNullableMetalResourceCalls {
        unsafe fn new_buffer(
            device: *mut MTLDevice,
            byte_length: u64,
            options: MTLResourceOptions,
        ) -> *mut MTLBuffer {
            unsafe {
                msg_send![device, newBufferWithLength: byte_length
                                           options: options]
            }
        }

        unsafe fn new_command_queue(
            device: *mut MTLDevice,
            maximum_command_buffers: u64,
        ) -> *mut MTLCommandQueue {
            unsafe {
                msg_send![device, newCommandQueueWithMaxCommandBufferCount: maximum_command_buffers]
            }
        }

        unsafe fn command_buffer(queue: *mut MTLCommandQueue) -> *mut MTLCommandBuffer {
            unsafe { msg_send![queue, commandBuffer] }
        }

        unsafe fn blit_command_encoder(
            command_buffer: *mut MTLCommandBuffer,
        ) -> *mut MTLBlitCommandEncoder {
            unsafe { msg_send![command_buffer, blitCommandEncoder] }
        }

        unsafe fn compute_command_encoder(
            command_buffer: *mut MTLCommandBuffer,
        ) -> *mut MTLComputeCommandEncoder {
            unsafe { msg_send![command_buffer, computeCommandEncoder] }
        }
    }

    fn non_null_resource<Resource>(
        pointer: *mut Resource,
        requested: u64,
    ) -> Result<NonNull<Resource>, MetalExecutionError> {
        NonNull::new(pointer).ok_or(MetalExecutionError::OutOfMemory { requested })
    }

    fn new_buffer_with<Calls: NullableMetalResourceCalls>(
        device: *mut MTLDevice,
        byte_length: u64,
        options: MTLResourceOptions,
        requested: u64,
    ) -> Result<Buffer, MetalExecutionError> {
        let pointer = non_null_resource(
            unsafe { Calls::new_buffer(device, byte_length, options) },
            requested,
        )?;
        Ok(unsafe { Buffer::from_ptr(pointer.as_ptr()) })
    }

    fn new_command_queue_with<Calls: NullableMetalResourceCalls>(
        device: *mut MTLDevice,
        maximum_command_buffers: u64,
    ) -> Result<CommandQueue, MetalExecutionError> {
        let pointer = non_null_resource(
            unsafe { Calls::new_command_queue(device, maximum_command_buffers) },
            0,
        )?;
        Ok(unsafe { CommandQueue::from_ptr(pointer.as_ptr()) })
    }

    fn new_command_buffer_with<Calls: NullableMetalResourceCalls>(
        queue: *mut MTLCommandQueue,
    ) -> Result<CommandBuffer, MetalExecutionError> {
        let pointer = non_null_resource(unsafe { Calls::command_buffer(queue) }, 0)?;
        let command_buffer = unsafe { CommandBufferRef::from_ptr(pointer.as_ptr()) };
        Ok(command_buffer.to_owned())
    }

    fn blit_command_encoder_with<Calls: NullableMetalResourceCalls>(
        command_buffer: *mut MTLCommandBuffer,
    ) -> Result<NonNull<MTLBlitCommandEncoder>, MetalExecutionError> {
        non_null_resource(unsafe { Calls::blit_command_encoder(command_buffer) }, 0)
    }

    fn compute_command_encoder_with<Calls: NullableMetalResourceCalls>(
        command_buffer: *mut MTLCommandBuffer,
    ) -> Result<NonNull<MTLComputeCommandEncoder>, MetalExecutionError> {
        non_null_resource(unsafe { Calls::compute_command_encoder(command_buffer) }, 0)
    }

    pub(super) struct Runtime {
        device: Device,
        add_f16: ComputePipelineState,
        add_f32: ComputePipelineState,
    }

    pub(super) struct Allocation {
        buffer: Buffer,
    }

    pub(super) struct Stream {
        queue: CommandQueue,
    }

    pub(super) struct Event {
        command_buffer: CommandBuffer,
    }

    impl Runtime {
        pub(super) fn new(
            contract: &MetalExecutionAbi,
            readiness_metallib: &[u8],
            tensor_ops_metallib: &[u8],
        ) -> Result<(Self, MetalDeviceProperties), MetalExecutionError> {
            bind_execution_contract(contract)?;
            let probed = crate::probe_device().map_err(|error| match error {
                crate::MetalLoadError::NoSystemDevice => MetalExecutionError::NoSystemDevice,
                error => MetalExecutionError::InvalidCertifiedInputs {
                    reason: MetalDiagnostic::bounded(error.to_string()),
                },
            })?;
            let device = Device::system_default().ok_or(MetalExecutionError::NoSystemDevice)?;
            if device.registry_id() != probed.registry_id || device.name() != probed.name {
                return Err(MetalExecutionError::InvalidCertifiedInputs {
                    reason: "system device changed between certified probe and execution setup"
                        .into(),
                });
            }
            let readiness_library =
                device
                    .new_library_with_data(readiness_metallib)
                    .map_err(|reason| MetalExecutionError::InvalidCertifiedInputs {
                        reason: MetalDiagnostic::bounded(format!(
                            "readiness metallib load failed: {reason}"
                        )),
                    })?;
            let readiness = pipeline(&device, &readiness_library, READINESS_FUNCTION)?;
            run_readiness_kernel(&device, &readiness)?;
            let library = device
                .new_library_with_data(tensor_ops_metallib)
                .map_err(|reason| MetalExecutionError::InvalidCertifiedInputs {
                    reason: MetalDiagnostic::bounded(format!(
                        "tensor-operations metallib load failed: {reason}"
                    )),
                })?;
            let add_f16 = pipeline(&device, &library, METAL_ADD_F16_FUNCTION)?;
            let add_f32 = pipeline(&device, &library, METAL_ADD_F32_FUNCTION)?;
            let properties = MetalDeviceProperties {
                name: bounded_device_name(probed.name)?,
                registry_id: probed.registry_id,
                recommended_working_set_bytes: probed.recommended_working_set_bytes,
                unified_memory: probed.unified_memory,
            };
            Ok((
                Self {
                    device,
                    add_f16,
                    add_f32,
                },
                properties,
            ))
        }

        pub(super) fn allocate(
            &self,
            byte_length: u64,
            storage_mode: MetalStorageMode,
        ) -> Result<Allocation, MetalExecutionError> {
            let options = match storage_mode {
                MetalStorageMode::Shared => MTLResourceOptions::StorageModeShared,
                MetalStorageMode::Managed => MTLResourceOptions::StorageModeManaged,
            };
            let requested = byte_length.max(1);
            let buffer = new_buffer_with::<SystemNullableMetalResourceCalls>(
                self.device.as_ptr(),
                requested,
                options,
                byte_length,
            )?;
            Ok(Allocation { buffer })
        }

        pub(super) fn create_stream(&self) -> Result<Stream, MetalExecutionError> {
            let queue = new_command_queue_with::<SystemNullableMetalResourceCalls>(
                self.device.as_ptr(),
                MAXIMUM_COMMAND_BUFFERS_PER_STREAM as u64,
            )?;
            Ok(Stream { queue })
        }

        pub(super) fn copy_host_to_device(
            &self,
            stream: &Stream,
            destination: &Allocation,
            storage_mode: MetalStorageMode,
            destination_offset: u64,
            bytes: &[u8],
        ) -> Result<(), MetalExecutionError> {
            self.wait_event(&self.record_event(stream)?)?;
            if bytes.is_empty() {
                return Ok(());
            }
            let destination_pointer = destination.buffer.contents().cast::<u8>();
            if destination_pointer.is_null() {
                return Err(MetalExecutionError::CommandFailed {
                    code: -1,
                    reason: "Metal buffer is not host addressable".into(),
                });
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    destination_pointer.add(destination_offset as usize),
                    bytes.len(),
                );
            }
            if storage_mode.requires_explicit_synchronization() {
                destination.buffer.did_modify_range(NSRange {
                    location: destination_offset,
                    length: bytes.len() as u64,
                });
            }
            Ok(())
        }

        pub(super) fn copy_device_to_host(
            &self,
            stream: &Stream,
            source: &Allocation,
            storage_mode: MetalStorageMode,
            source_offset: u64,
            bytes: &mut [u8],
        ) -> Result<(), MetalExecutionError> {
            let command_buffer =
                new_command_buffer_with::<SystemNullableMetalResourceCalls>(stream.queue.as_ptr())?;
            if storage_mode.requires_explicit_synchronization() {
                let encoder = blit_command_encoder_with::<SystemNullableMetalResourceCalls>(
                    command_buffer.as_ptr(),
                )?;
                let encoder = unsafe { BlitCommandEncoderRef::from_ptr(encoder.as_ptr()) };
                encoder.synchronize_resource(&source.buffer);
                encoder.end_encoding();
            }
            command_buffer.commit();
            let event = Event { command_buffer };
            self.wait_event(&event)?;
            if bytes.is_empty() {
                return Ok(());
            }
            let source_pointer = source.buffer.contents().cast::<u8>();
            if source_pointer.is_null() {
                return Err(MetalExecutionError::CommandFailed {
                    code: -1,
                    reason: "Metal buffer is not host addressable".into(),
                });
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source_pointer.add(source_offset as usize),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            stream: &Stream,
            element_type: MetalElementType,
            left: &Allocation,
            right: &Allocation,
            output: &Allocation,
            output_storage_mode: MetalStorageMode,
            elements: u32,
        ) -> Result<Event, MetalExecutionError> {
            let command_buffer =
                new_command_buffer_with::<SystemNullableMetalResourceCalls>(stream.queue.as_ptr())?;
            if elements != 0 {
                let pipeline = match element_type {
                    MetalElementType::F16 => &self.add_f16,
                    MetalElementType::F32 => &self.add_f32,
                };
                let encoder = compute_command_encoder_with::<SystemNullableMetalResourceCalls>(
                    command_buffer.as_ptr(),
                )?;
                let encoder = unsafe { ComputeCommandEncoderRef::from_ptr(encoder.as_ptr()) };
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&left.buffer), 0);
                encoder.set_buffer(1, Some(&right.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    (&raw const elements).cast(),
                );
                let width = pipeline.thread_execution_width().max(1);
                let group_width = width.min(pipeline.max_total_threads_per_threadgroup().max(1));
                encoder.dispatch_threads(
                    MTLSize {
                        width: u64::from(elements),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: group_width,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
                if output_storage_mode.requires_explicit_synchronization() {
                    let blit = blit_command_encoder_with::<SystemNullableMetalResourceCalls>(
                        command_buffer.as_ptr(),
                    )?;
                    let blit = unsafe { BlitCommandEncoderRef::from_ptr(blit.as_ptr()) };
                    blit.synchronize_resource(&output.buffer);
                    blit.end_encoding();
                }
            }
            command_buffer.commit();
            Ok(Event { command_buffer })
        }

        pub(super) fn record_event(&self, stream: &Stream) -> Result<Event, MetalExecutionError> {
            let command_buffer =
                new_command_buffer_with::<SystemNullableMetalResourceCalls>(stream.queue.as_ptr())?;
            command_buffer.commit();
            Ok(Event { command_buffer })
        }

        pub(super) fn wait_event(&self, event: &Event) -> Result<(), MetalExecutionError> {
            event.command_buffer.wait_until_completed();
            match event.command_buffer.status() {
                MTLCommandBufferStatus::Completed => Ok(()),
                status => {
                    let code = command_error_code(&event.command_buffer).unwrap_or(-1);
                    Err(map_command_error(
                        code,
                        format!("command status {status:?}"),
                    ))
                }
            }
        }
    }

    fn pipeline(
        device: &Device,
        library: &metal::Library,
        function_name: &str,
    ) -> Result<ComputePipelineState, MetalExecutionError> {
        let function = library.get_function(function_name, None).map_err(|_| {
            MetalExecutionError::MissingFunction {
                function: MetalDiagnostic::bounded(function_name.to_owned()),
            }
        })?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|reason| MetalExecutionError::PipelineCreation {
                function: MetalDiagnostic::bounded(function_name.to_owned()),
                reason: MetalDiagnostic::bounded(reason),
            })
    }

    fn run_readiness_kernel(
        device: &Device,
        pipeline: &ComputePipelineState,
    ) -> Result<(), MetalExecutionError> {
        const INPUT: u32 = 0x0102_0304;
        const EXPECTED: u32 = INPUT ^ 0x5349_4d31;
        let options = MTLResourceOptions::StorageModeShared;
        let input = new_buffer_with::<SystemNullableMetalResourceCalls>(
            device.as_ptr(),
            std::mem::size_of::<u32>() as u64,
            options,
            std::mem::size_of::<u32>() as u64,
        )?;
        let output = new_buffer_with::<SystemNullableMetalResourceCalls>(
            device.as_ptr(),
            std::mem::size_of::<u32>() as u64,
            options,
            std::mem::size_of::<u32>() as u64,
        )?;
        let input_pointer = input.contents().cast::<u32>();
        let output_pointer = output.contents().cast::<u32>();
        if input_pointer.is_null() || output_pointer.is_null() {
            return Err(MetalExecutionError::InvalidCertifiedInputs {
                reason: "readiness buffers are not host addressable".into(),
            });
        }
        unsafe {
            input_pointer.write(INPUT);
            output_pointer.write(0);
        }
        let queue = new_command_queue_with::<SystemNullableMetalResourceCalls>(
            device.as_ptr(),
            MAXIMUM_COMMAND_BUFFERS_PER_STREAM as u64,
        )?;
        let command_buffer =
            new_command_buffer_with::<SystemNullableMetalResourceCalls>(queue.as_ptr())?;
        let encoder = compute_command_encoder_with::<SystemNullableMetalResourceCalls>(
            command_buffer.as_ptr(),
        )?;
        let encoder = unsafe { ComputeCommandEncoderRef::from_ptr(encoder.as_ptr()) };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            let code = command_error_code(&command_buffer).unwrap_or(-1);
            return Err(map_command_error(
                code,
                "certified readiness kernel command failed".to_owned(),
            ));
        }
        let observed = unsafe { output_pointer.read() };
        if observed != EXPECTED {
            return Err(MetalExecutionError::InvalidCertifiedInputs {
                reason: "certified readiness kernel returned an unexpected value".into(),
            });
        }
        Ok(())
    }

    fn command_error_code(command_buffer: &metal::CommandBufferRef) -> Option<i64> {
        unsafe {
            let error: *mut Object = msg_send![command_buffer, error];
            if error.is_null() {
                None
            } else {
                Some(msg_send![error, code])
            }
        }
    }

    #[cfg(test)]
    mod nullable_resource_tests {
        use super::*;

        struct NilResourceCalls;

        unsafe impl NullableMetalResourceCalls for NilResourceCalls {
            unsafe fn new_buffer(
                _device: *mut MTLDevice,
                _byte_length: u64,
                _options: MTLResourceOptions,
            ) -> *mut MTLBuffer {
                std::ptr::null_mut()
            }

            unsafe fn new_command_queue(
                _device: *mut MTLDevice,
                _maximum_command_buffers: u64,
            ) -> *mut MTLCommandQueue {
                std::ptr::null_mut()
            }

            unsafe fn command_buffer(_queue: *mut MTLCommandQueue) -> *mut MTLCommandBuffer {
                std::ptr::null_mut()
            }

            unsafe fn blit_command_encoder(
                _command_buffer: *mut MTLCommandBuffer,
            ) -> *mut MTLBlitCommandEncoder {
                std::ptr::null_mut()
            }

            unsafe fn compute_command_encoder(
                _command_buffer: *mut MTLCommandBuffer,
            ) -> *mut MTLComputeCommandEncoder {
                std::ptr::null_mut()
            }
        }

        #[test]
        fn nullable_sdk_resources_fail_before_foreign_type_construction() {
            let device = NonNull::<MTLDevice>::dangling().as_ptr();
            let queue = NonNull::<MTLCommandQueue>::dangling().as_ptr();
            let command_buffer = NonNull::<MTLCommandBuffer>::dangling().as_ptr();

            assert!(matches!(
                new_buffer_with::<NilResourceCalls>(
                    device,
                    64,
                    MTLResourceOptions::StorageModeShared,
                    64,
                ),
                Err(MetalExecutionError::OutOfMemory { requested: 64 })
            ));
            assert!(matches!(
                new_command_queue_with::<NilResourceCalls>(device, 64),
                Err(MetalExecutionError::OutOfMemory { requested: 0 })
            ));
            assert!(matches!(
                new_command_buffer_with::<NilResourceCalls>(queue),
                Err(MetalExecutionError::OutOfMemory { requested: 0 })
            ));
            assert!(matches!(
                blit_command_encoder_with::<NilResourceCalls>(command_buffer),
                Err(MetalExecutionError::OutOfMemory { requested: 0 })
            ));
            assert!(matches!(
                compute_command_encoder_with::<NilResourceCalls>(command_buffer),
                Err(MetalExecutionError::OutOfMemory { requested: 0 })
            ));
        }
    }
}

#[cfg(not(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod platform {
    use super::*;

    pub(super) struct Runtime;
    pub(super) struct Allocation;
    pub(super) struct Stream;
    pub(super) struct Event;

    fn unavailable() -> MetalExecutionError {
        MetalExecutionError::UnsupportedTarget {
            target: MetalDiagnostic::bounded(env!("SIM_METAL_TARGET").to_owned()),
        }
    }

    impl Runtime {
        pub(super) fn new(
            contract: &MetalExecutionAbi,
            _readiness_metallib: &[u8],
            _tensor_ops_metallib: &[u8],
        ) -> Result<(Self, MetalDeviceProperties), MetalExecutionError> {
            bind_execution_contract(contract)?;
            Err(unavailable())
        }

        pub(super) fn allocate(
            &self,
            _byte_length: u64,
            _storage_mode: MetalStorageMode,
        ) -> Result<Allocation, MetalExecutionError> {
            Err(unavailable())
        }

        pub(super) fn create_stream(&self) -> Result<Stream, MetalExecutionError> {
            Err(unavailable())
        }

        pub(super) fn copy_host_to_device(
            &self,
            _stream: &Stream,
            _destination: &Allocation,
            _storage_mode: MetalStorageMode,
            _destination_offset: u64,
            _bytes: &[u8],
        ) -> Result<(), MetalExecutionError> {
            Err(unavailable())
        }

        pub(super) fn copy_device_to_host(
            &self,
            _stream: &Stream,
            _source: &Allocation,
            _storage_mode: MetalStorageMode,
            _source_offset: u64,
            _bytes: &mut [u8],
        ) -> Result<(), MetalExecutionError> {
            Err(unavailable())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            _stream: &Stream,
            _element_type: MetalElementType,
            _left: &Allocation,
            _right: &Allocation,
            _output: &Allocation,
            _output_storage_mode: MetalStorageMode,
            _elements: u32,
        ) -> Result<Event, MetalExecutionError> {
            Err(unavailable())
        }

        pub(super) fn record_event(&self, _stream: &Stream) -> Result<Event, MetalExecutionError> {
            Err(unavailable())
        }

        pub(super) fn wait_event(&self, _event: &Event) -> Result<(), MetalExecutionError> {
            Err(unavailable())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, MetalExecutionError> {
        bytes
            .chunks_exact(4)
            .map(|lane| {
                <[u8; 4]>::try_from(lane)
                    .map(f32::from_le_bytes)
                    .map_err(|_| MetalExecutionError::CommandFailed {
                        code: -1,
                        reason: "test output contained an incomplete F32 lane".into(),
                    })
            })
            .collect()
    }

    fn execute_fake_f32_add(
        storage_mode: MetalStorageMode,
    ) -> Result<Vec<test_fake::TraceOperation>, MetalExecutionError> {
        let (runtime, control) = MetalRuntime::from_fake(test_fake::Config {
            storage_mode,
            capacity_bytes: 1_024,
        })?;
        assert_eq!(runtime.properties().storage_mode(), storage_mode);
        let left = runtime.allocate(12)?;
        let right = runtime.allocate(12)?;
        let output = runtime.allocate(12)?;
        let stream = runtime.create_stream()?;
        runtime.copy_host_to_device(&stream, &left, 0, &f32_bytes(&[1.5, -2.0, 8.25]))?;
        runtime.copy_host_to_device(&stream, &right, 0, &f32_bytes(&[2.25, 0.5, -3.0]))?;
        let event =
            runtime.dispatch_add(&stream, MetalElementType::F32, &left, &right, &output, 3)?;
        runtime.wait_event(&event)?;
        let mut result = [0_u8; 12];
        runtime.copy_device_to_host(&stream, &output, 0, &mut result)?;
        assert_eq!(decode_f32(&result)?, vec![3.75, -1.5, 5.25]);
        control.trace()
    }

    #[test]
    fn certified_inputs_retain_metallib_and_certification_without_self_authorizing() {
        let certification = Arc::new(()) as Arc<dyn Any + Send + Sync>;
        let readiness = Arc::<[u8]>::from([1_u8, 2, 3]);
        let tensor_ops = Arc::<[u8]>::from([4_u8, 5, 6]);
        let retained =
            CertifiedInputs::new(readiness.clone(), tensor_ops.clone(), certification.clone())
                .expect("bounded certified inputs");
        assert_eq!(Arc::strong_count(&readiness), 2);
        assert_eq!(Arc::strong_count(&tensor_ops), 2);
        assert_eq!(Arc::strong_count(&certification), 2);
        drop(retained);
        assert_eq!(Arc::strong_count(&readiness), 1);
        assert_eq!(Arc::strong_count(&tensor_ops), 1);
        assert_eq!(Arc::strong_count(&certification), 1);
    }

    #[test]
    fn escaped_resources_retain_the_certified_runtime_session() -> Result<(), MetalExecutionError> {
        let (runtime, _) = MetalRuntime::from_fake(test_fake::Config {
            storage_mode: MetalStorageMode::Shared,
            capacity_bytes: 64,
        })?;
        let runtime_lifetime = Arc::downgrade(&runtime.inner);
        let allocation = runtime.allocate(4)?;
        let stream = runtime.create_stream()?;
        let event = runtime.record_event(&stream)?;
        drop(runtime);
        assert!(runtime_lifetime.upgrade().is_some());
        drop(allocation);
        assert!(runtime_lifetime.upgrade().is_some());
        drop(stream);
        assert!(runtime_lifetime.upgrade().is_some());
        drop(event);
        assert!(runtime_lifetime.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn resource_bounds_and_command_error_mapping_are_exact() {
        assert!(require_range(8, 4, 4).is_ok());
        assert!(matches!(
            require_range(8, 5, 4),
            Err(MetalExecutionError::ResourceBounds { .. })
        ));
        assert_eq!(
            map_command_error(8, "oom".to_owned()),
            MetalExecutionError::OutOfMemory { requested: 0 }
        );
        assert_eq!(
            map_command_error(11, "removed".to_owned()),
            MetalExecutionError::DeviceLost { code: 11 }
        );
        assert!(matches!(
            map_command_error(9, "invalid resource".to_owned()),
            MetalExecutionError::CommandFailed { code: 9, .. }
        ));
        let long_reason = "é".repeat(MAXIMUM_DIAGNOSTIC_BYTES);
        let MetalExecutionError::CommandFailed { reason, .. } = map_command_error(9, long_reason)
        else {
            panic!("unexpected mapped error");
        };
        assert!(reason.as_str().len() <= MAXIMUM_DIAGNOSTIC_BYTES);
        assert!(reason.as_str().is_char_boundary(reason.as_str().len()));
    }

    #[test]
    fn device_names_and_public_diagnostics_are_bounded() {
        assert_eq!(
            bounded_device_name("M".repeat(MAXIMUM_DEVICE_NAME_BYTES))
                .expect("maximum-length name must remain valid")
                .len(),
            MAXIMUM_DEVICE_NAME_BYTES
        );
        assert!(matches!(
            bounded_device_name("M".repeat(MAXIMUM_DEVICE_NAME_BYTES + 1)),
            Err(MetalExecutionError::InvalidCertifiedInputs { .. })
        ));

        let diagnostic = MetalDiagnostic::bounded("d".repeat(MAXIMUM_DIAGNOSTIC_BYTES + 1));
        assert_eq!(diagnostic.as_str().len(), MAXIMUM_DIAGNOSTIC_BYTES);
        let diagnostic = MetalDiagnostic::from("e".repeat(MAXIMUM_DIAGNOSTIC_BYTES + 1).as_str());
        assert_eq!(diagnostic.as_str().len(), MAXIMUM_DIAGNOSTIC_BYTES);
    }

    #[test]
    fn opaque_identity_and_storage_synchronization_rules_are_platform_independent() {
        fn assert_send_sync<Type: Send + Sync>() {}

        assert_send_sync::<MetalRuntime>();
        assert_send_sync::<MetalAllocation>();
        assert_send_sync::<MetalStream>();
        assert_send_sync::<MetalEvent>();
        assert!(require_identity(7, 7).is_ok());
        assert_eq!(
            require_identity(7, 8),
            Err(MetalExecutionError::ForeignResource)
        );
        assert!(!MetalStorageMode::Shared.requires_explicit_synchronization());
        assert!(MetalStorageMode::Managed.requires_explicit_synchronization());
    }

    #[test]
    fn fake_runtime_exercises_shared_transfers_dispatch_and_completion_through_public_api()
    -> Result<(), MetalExecutionError> {
        use test_fake::{CommandOutcome, TraceOperation};

        assert_eq!(
            execute_fake_f32_add(MetalStorageMode::Shared)?,
            vec![
                TraceOperation::Allocate {
                    allocation: 1,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Shared,
                },
                TraceOperation::Allocate {
                    allocation: 2,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Shared,
                },
                TraceOperation::Allocate {
                    allocation: 3,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Shared,
                },
                TraceOperation::CreateStream { stream: 1 },
                TraceOperation::RecordEvent {
                    stream: 1,
                    event: 1,
                },
                TraceOperation::Commit { event: 1 },
                TraceOperation::Wait {
                    event: 1,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostWrite {
                    allocation: 1,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::RecordEvent {
                    stream: 1,
                    event: 2,
                },
                TraceOperation::Commit { event: 2 },
                TraceOperation::Wait {
                    event: 2,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostWrite {
                    allocation: 2,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::DispatchAdd {
                    stream: 1,
                    event: 3,
                    elements: 3,
                    element_type: MetalElementType::F32,
                },
                TraceOperation::Commit { event: 3 },
                TraceOperation::Wait {
                    event: 3,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::Commit { event: 4 },
                TraceOperation::Wait {
                    event: 4,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostRead {
                    allocation: 3,
                    offset: 0,
                    bytes: 12,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn fake_runtime_exercises_managed_transfer_synchronization_order_through_public_api()
    -> Result<(), MetalExecutionError> {
        use test_fake::{CommandOutcome, TraceOperation};

        assert_eq!(
            execute_fake_f32_add(MetalStorageMode::Managed)?,
            vec![
                TraceOperation::Allocate {
                    allocation: 1,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Managed,
                },
                TraceOperation::Allocate {
                    allocation: 2,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Managed,
                },
                TraceOperation::Allocate {
                    allocation: 3,
                    bytes: 12,
                    storage_mode: MetalStorageMode::Managed,
                },
                TraceOperation::CreateStream { stream: 1 },
                TraceOperation::RecordEvent {
                    stream: 1,
                    event: 1,
                },
                TraceOperation::Commit { event: 1 },
                TraceOperation::Wait {
                    event: 1,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostWrite {
                    allocation: 1,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::DidModify {
                    allocation: 1,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::RecordEvent {
                    stream: 1,
                    event: 2,
                },
                TraceOperation::Commit { event: 2 },
                TraceOperation::Wait {
                    event: 2,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostWrite {
                    allocation: 2,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::DidModify {
                    allocation: 2,
                    offset: 0,
                    bytes: 12,
                },
                TraceOperation::DispatchAdd {
                    stream: 1,
                    event: 3,
                    elements: 3,
                    element_type: MetalElementType::F32,
                },
                TraceOperation::SynchronizeResource { allocation: 3 },
                TraceOperation::Commit { event: 3 },
                TraceOperation::Wait {
                    event: 3,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::SynchronizeResource { allocation: 3 },
                TraceOperation::Commit { event: 4 },
                TraceOperation::Wait {
                    event: 4,
                    outcome: CommandOutcome::Completed,
                },
                TraceOperation::HostRead {
                    allocation: 3,
                    offset: 0,
                    bytes: 12,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn fake_runtime_injects_oom_and_device_loss_through_public_api()
    -> Result<(), MetalExecutionError> {
        use test_fake::{CommandOutcome, TraceOperation};

        let (runtime, control) = MetalRuntime::from_fake(test_fake::Config {
            storage_mode: MetalStorageMode::Shared,
            capacity_bytes: 8,
        })?;
        let allocation = runtime.allocate(8)?;
        assert_eq!(
            runtime
                .allocate(1)
                .expect_err("capacity must reject allocation"),
            MetalExecutionError::OutOfMemory { requested: 1 }
        );
        drop(allocation);
        assert_eq!(runtime.allocate(1)?.byte_length(), 1);
        assert!(
            !control
                .trace()?
                .contains(&TraceOperation::RejectAllocation { bytes: 1 })
        );

        let (runtime, control) = MetalRuntime::from_fake(test_fake::Config {
            storage_mode: MetalStorageMode::Shared,
            capacity_bytes: 32,
        })?;
        let left = runtime.allocate(4)?;
        let right = runtime.allocate(4)?;
        let output = runtime.allocate(4)?;
        let stream = runtime.create_stream()?;
        control.fail_next_command(11)?;
        let event =
            runtime.dispatch_add(&stream, MetalElementType::F32, &left, &right, &output, 1)?;
        assert_eq!(
            runtime.wait_event(&event),
            Err(MetalExecutionError::DeviceLost { code: 11 })
        );
        assert!(control.trace()?.contains(&TraceOperation::Wait {
            event: 1,
            outcome: CommandOutcome::Failed(11),
        }));
        Ok(())
    }

    #[test]
    fn unsupported_or_missing_device_never_becomes_available_from_metallib_bytes() {
        let result = unsafe {
            MetalRuntime::from_certified_metallibs(
                Arc::<[u8]>::from([1_u8]),
                Arc::<[u8]>::from([1_u8]),
                Arc::new(()) as Arc<dyn Any + Send + Sync>,
            )
        };
        assert!(matches!(
            result,
            Err(MetalExecutionError::NoSystemDevice)
                | Err(MetalExecutionError::UnsupportedTarget { .. })
                | Err(MetalExecutionError::InvalidCertifiedInputs { .. })
        ));
    }
}
