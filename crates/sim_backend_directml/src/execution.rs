use crate::{DirectMlLoadError, RetainedDirectMlLibraryHandles};
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(any(
    test,
    feature = "test-support",
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
use std::sync::{Mutex, MutexGuard};
use thiserror::Error;

const MAXIMUM_STREAMS: u64 = 1_024;
const MAXIMUM_EVENTS: u64 = 4_096;
#[cfg(any(
    test,
    feature = "test-support",
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
const MAXIMUM_DEVICE_NAME_BYTES: usize = 256;
const MAXIMUM_DIAGNOSTIC_BYTES: usize = 1_024;
const DXGI_ERROR_DEVICE_REMOVED: u32 = 0x887a_0005;
const DXGI_ERROR_DEVICE_HUNG: u32 = 0x887a_0006;
const DXGI_ERROR_DEVICE_RESET: u32 = 0x887a_0007;
const DXGI_ERROR_DRIVER_INTERNAL_ERROR: u32 = 0x887a_0020;
const E_OUTOFMEMORY: u32 = 0x8007_000e;
#[cfg(feature = "test-support")]
const E_FAIL: u32 = 0x8000_4005;
static NEXT_SESSION_IDENTITY: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectMlElementType {
    F16,
    F32,
}

impl DirectMlElementType {
    const fn byte_width(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlDeviceProperties {
    name: String,
    adapter_luid: u64,
    dedicated_memory_bytes: u64,
    shared_memory_bytes: u64,
    allocation_capacity_bytes: u64,
    has_fp16: bool,
}

impl DirectMlDeviceProperties {
    #[cfg(any(
        test,
        feature = "test-support",
        all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    ))]
    fn checked(
        name: String,
        adapter_luid: u64,
        dedicated_memory_bytes: u64,
        shared_memory_bytes: u64,
        allocation_capacity_bytes: u64,
        has_fp16: bool,
    ) -> Result<Self, DirectMlExecutionError> {
        if name.is_empty() || name.len() > MAXIMUM_DEVICE_NAME_BYTES || name.contains('\0') {
            return Err(DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("device name must contain 1..=256 non-NUL bytes"),
            });
        }
        if allocation_capacity_bytes == 0 {
            return Err(DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("allocation capacity must be nonzero"),
            });
        }
        let total_memory_bytes = dedicated_memory_bytes
            .checked_add(shared_memory_bytes)
            .ok_or_else(|| DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("dedicated and shared memory total overflowed"),
            })?;
        if total_memory_bytes == 0 || allocation_capacity_bytes > total_memory_bytes {
            return Err(DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic(
                    "allocation capacity must not exceed nonzero dedicated plus shared memory",
                ),
            });
        }
        Ok(Self {
            name,
            adapter_luid,
            dedicated_memory_bytes,
            shared_memory_bytes,
            allocation_capacity_bytes,
            has_fp16,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn adapter_luid(&self) -> u64 {
        self.adapter_luid
    }

    pub const fn dedicated_memory_bytes(&self) -> u64 {
        self.dedicated_memory_bytes
    }

    pub const fn shared_memory_bytes(&self) -> u64 {
        self.shared_memory_bytes
    }

    pub const fn allocation_capacity_bytes(&self) -> u64 {
        self.allocation_capacity_bytes
    }

    pub const fn has_fp16(&self) -> bool {
        self.has_fp16
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DirectMlExecutionError {
    #[error("DirectML execution target is unsupported: {target}")]
    UnsupportedTarget { target: String },
    #[error("certified DirectML execution inputs are invalid: {reason}")]
    InvalidCertifiedInputs { reason: String },
    #[error("DirectML execution was cancelled")]
    Cancelled,
    #[error("DirectML allocation of {requested} bytes exceeded capacity {capacity}")]
    OutOfMemory { requested: u64, capacity: u64 },
    #[error("DirectML resource belongs to a different certified session")]
    ForeignResource,
    #[error("DirectML resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: u64,
        length: u64,
        available: u64,
    },
    #[error("DirectML element count {elements} exceeds the reviewed tensor ABI")]
    ElementCount { elements: u64 },
    #[error("DirectML element type {element_type:?} is unsupported by this device")]
    UnsupportedElementType { element_type: DirectMlElementType },
    #[error("DirectML stream limit {limit} was reached")]
    StreamLimit { limit: u64 },
    #[error("DirectML pending-event limit {limit} was reached")]
    EventLimit { limit: u64 },
    #[error("DirectML device was lost with HRESULT {status:#x}")]
    DeviceLost { status: i32 },
    #[error("DirectML operation {operation} failed with HRESULT {status:#x}: {reason}")]
    CommandFailed {
        operation: &'static str,
        status: i32,
        reason: String,
    },
}

impl From<DirectMlLoadError> for DirectMlExecutionError {
    fn from(error: DirectMlLoadError) -> Self {
        match error {
            DirectMlLoadError::UnsupportedTarget { target } => Self::UnsupportedTarget { target },
            DirectMlLoadError::ComCall { operation, status } => {
                map_hresult(operation, status, "certified COM call failed")
            }
            other => Self::InvalidCertifiedInputs {
                reason: diagnostic(&other.to_string()),
            },
        }
    }
}

struct CapacityTracker {
    limit: u64,
    current: AtomicU64,
    peak: AtomicU64,
}

impl CapacityTracker {
    fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<CapacityReservation, DirectMlExecutionError> {
        let mut current = self.current.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .filter(|next| *next <= self.limit)
                .ok_or(DirectMlExecutionError::OutOfMemory {
                    requested: bytes,
                    capacity: self.limit,
                })?;
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

struct BoundedResourceToken {
    active: Arc<AtomicU64>,
}

impl Drop for BoundedResourceToken {
    fn drop(&mut self) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}

fn acquire_bounded(
    active: &Arc<AtomicU64>,
    limit: u64,
    event: bool,
) -> Result<BoundedResourceToken, DirectMlExecutionError> {
    let mut current = active.load(Ordering::Acquire);
    loop {
        if current >= limit {
            return Err(if event {
                DirectMlExecutionError::EventLimit { limit }
            } else {
                DirectMlExecutionError::StreamLimit { limit }
            });
        }
        match active.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                return Ok(BoundedResourceToken {
                    active: active.clone(),
                });
            }
            Err(observed) => current = observed,
        }
    }
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

struct SessionInner {
    identity: u64,
    properties: DirectMlDeviceProperties,
    capacity: Arc<CapacityTracker>,
    active_streams: Arc<AtomicU64>,
    active_events: Arc<AtomicU64>,
    next_event: AtomicU64,
    _certification: Arc<dyn std::any::Any + Send + Sync>,
    platform: RuntimeImplementation,
}

#[derive(Clone)]
pub struct DirectMlExecutionSession {
    inner: Arc<SessionInner>,
}

#[cfg(feature = "test-support")]
#[derive(Clone)]
pub struct DirectMlTestControl {
    inner: test_fake::Control,
}

#[cfg(feature = "test-support")]
impl DirectMlTestControl {
    pub fn fail_next_event_with_device_loss(&self) -> Result<(), DirectMlExecutionError> {
        self.inner.fail_next(DXGI_ERROR_DEVICE_REMOVED as i32)
    }

    pub fn fail_next_event_with_command_failure(&self) -> Result<(), DirectMlExecutionError> {
        self.inner.fail_next(E_FAIL as i32)
    }

    pub fn cancel_after_next_wait(&self) -> Result<(), DirectMlExecutionError> {
        self.inner.cancel_after_next_wait()
    }
}

impl fmt::Debug for DirectMlExecutionSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlExecutionSession")
            .field("identity", &self.inner.identity)
            .field("properties", &self.inner.properties)
            .finish_non_exhaustive()
    }
}

struct AllocationInner {
    session_identity: u64,
    byte_length: u64,
    platform: AllocationImplementation,
    _reservation: CapacityReservation,
    _session: Arc<SessionInner>,
}

#[derive(Clone)]
pub struct DirectMlAllocation {
    inner: Arc<AllocationInner>,
}

impl fmt::Debug for DirectMlAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlAllocation")
            .field("byte_length", &self.inner.byte_length)
            .finish_non_exhaustive()
    }
}

impl DirectMlAllocation {
    pub fn byte_length(&self) -> u64 {
        self.inner.byte_length
    }
}

struct StreamInner {
    session_identity: u64,
    platform: StreamImplementation,
    _token: BoundedResourceToken,
    _session: Arc<SessionInner>,
}

#[derive(Clone)]
pub struct DirectMlStream {
    inner: Arc<StreamInner>,
}

impl fmt::Debug for DirectMlStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlStream")
            .finish_non_exhaustive()
    }
}

struct EventInner {
    session_identity: u64,
    sequence: u64,
    platform: EventImplementation,
    _token: BoundedResourceToken,
    _session: Arc<SessionInner>,
}

#[derive(Clone)]
pub struct DirectMlEvent {
    inner: Arc<EventInner>,
}

impl fmt::Debug for DirectMlEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlEvent")
            .field("sequence", &self.inner.sequence)
            .finish_non_exhaustive()
    }
}

impl DirectMlEvent {
    pub fn sequence(&self) -> u64 {
        self.inner.sequence
    }
}

impl DirectMlExecutionSession {
    pub fn from_registry_certified_handles(
        handles: RetainedDirectMlLibraryHandles,
    ) -> Result<Self, DirectMlExecutionError> {
        let certification = handles.certification_retention();
        let inputs = handles.into_execution_inputs()?;
        let identity = next_identity()?;
        let (platform, properties) = platform::Runtime::new(inputs)?;
        Self::from_parts(
            identity,
            platform_runtime(platform),
            properties,
            certification,
        )
    }

    #[cfg(feature = "test-support")]
    pub fn for_test_harness(
        capacity: u64,
        has_fp16: bool,
    ) -> Result<DirectMlExecutionSession, DirectMlExecutionError> {
        let (session, _) = Self::for_test_harness_with_control(capacity, has_fp16)?;
        Ok(session)
    }

    #[cfg(feature = "test-support")]
    pub fn for_test_harness_with_control(
        capacity: u64,
        has_fp16: bool,
    ) -> Result<(DirectMlExecutionSession, DirectMlTestControl), DirectMlExecutionError> {
        Self::test_harness_with_properties(capacity, 0, capacity, has_fp16)
    }

    #[cfg(feature = "test-support")]
    pub fn for_test_harness_with_memory_properties(
        dedicated_memory_bytes: u64,
        shared_memory_bytes: u64,
        allocation_capacity_bytes: u64,
        has_fp16: bool,
    ) -> Result<DirectMlExecutionSession, DirectMlExecutionError> {
        let (session, _) = Self::test_harness_with_properties(
            dedicated_memory_bytes,
            shared_memory_bytes,
            allocation_capacity_bytes,
            has_fp16,
        )?;
        Ok(session)
    }

    #[cfg(feature = "test-support")]
    fn test_harness_with_properties(
        dedicated_memory_bytes: u64,
        shared_memory_bytes: u64,
        allocation_capacity_bytes: u64,
        has_fp16: bool,
    ) -> Result<(DirectMlExecutionSession, DirectMlTestControl), DirectMlExecutionError> {
        let (runtime, control) = test_fake::Runtime::new();
        let properties = DirectMlDeviceProperties::checked(
            "Injected DirectML adapter".to_owned(),
            0x1122_3344_5566_7788,
            dedicated_memory_bytes,
            shared_memory_bytes,
            allocation_capacity_bytes,
            has_fp16,
        )?;
        let identity = next_identity()?;
        let session = DirectMlExecutionSession::from_parts(
            identity,
            RuntimeImplementation::Fake(runtime),
            properties,
            Arc::new(()),
        )?;
        Ok((session, DirectMlTestControl { inner: control }))
    }

    fn from_parts(
        identity: u64,
        platform: RuntimeImplementation,
        properties: DirectMlDeviceProperties,
        certification: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Self, DirectMlExecutionError> {
        let capacity = properties.allocation_capacity_bytes();
        Ok(Self {
            inner: Arc::new(SessionInner {
                identity,
                properties,
                capacity: Arc::new(CapacityTracker {
                    limit: capacity,
                    current: AtomicU64::new(0),
                    peak: AtomicU64::new(0),
                }),
                active_streams: Arc::new(AtomicU64::new(0)),
                active_events: Arc::new(AtomicU64::new(0)),
                next_event: AtomicU64::new(1),
                _certification: certification,
                platform,
            }),
        })
    }

    pub fn properties(&self) -> &DirectMlDeviceProperties {
        &self.inner.properties
    }

    pub fn current_allocation_bytes(&self) -> u64 {
        self.inner.capacity.current.load(Ordering::Acquire)
    }

    pub fn peak_allocation_bytes(&self) -> u64 {
        self.inner.capacity.peak.load(Ordering::Acquire)
    }

    pub fn allocate(
        &self,
        byte_length: u64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DirectMlAllocation, DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        let physical_byte_length =
            dword_rounded(byte_length).ok_or(DirectMlExecutionError::OutOfMemory {
                requested: byte_length,
                capacity: self.inner.capacity.limit,
            })?;
        let reservation = self.inner.capacity.reserve(physical_byte_length)?;
        let platform = runtime_allocate(&self.inner.platform, physical_byte_length)?;
        check_cancelled(cancelled)?;
        Ok(DirectMlAllocation {
            inner: Arc::new(AllocationInner {
                session_identity: self.inner.identity,
                byte_length,
                platform,
                _reservation: reservation,
                _session: self.inner.clone(),
            }),
        })
    }

    pub fn create_stream(
        &self,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DirectMlStream, DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        let token = acquire_bounded(&self.inner.active_streams, MAXIMUM_STREAMS, false)?;
        let platform = runtime_create_stream(&self.inner.platform)?;
        Ok(DirectMlStream {
            inner: Arc::new(StreamInner {
                session_identity: self.inner.identity,
                platform,
                _token: token,
                _session: self.inner.clone(),
            }),
        })
    }

    pub fn copy_host_to_device(
        &self,
        stream: &DirectMlStream,
        destination: &DirectMlAllocation,
        destination_offset: u64,
        bytes: &[u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        self.require_stream(stream)?;
        self.require_allocation(destination)?;
        let length =
            u64::try_from(bytes.len()).map_err(|_| DirectMlExecutionError::ResourceBounds {
                offset: destination_offset,
                length: u64::MAX,
                available: destination.byte_length(),
            })?;
        require_range(destination.byte_length(), destination_offset, length)?;
        let _upload_reservation = self.inner.capacity.reserve(length)?;
        let physical_length = dword_rounded(length).ok_or(DirectMlExecutionError::OutOfMemory {
            requested: length,
            capacity: self.inner.capacity.limit,
        })?;
        let _commit_staging_reservation = self.inner.capacity.reserve(physical_length)?;
        runtime_copy_host_to_device(
            &self.inner.platform,
            &stream.inner.platform,
            &destination.inner.platform,
            destination_offset,
            bytes,
            cancelled,
        )
    }

    pub fn copy_device_to_host(
        &self,
        stream: &DirectMlStream,
        source: &DirectMlAllocation,
        source_offset: u64,
        bytes: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        self.require_stream(stream)?;
        self.require_allocation(source)?;
        let length =
            u64::try_from(bytes.len()).map_err(|_| DirectMlExecutionError::ResourceBounds {
                offset: source_offset,
                length: u64::MAX,
                available: source.byte_length(),
            })?;
        require_range(source.byte_length(), source_offset, length)?;
        let _staging_reservation = self.inner.capacity.reserve(length)?;
        runtime_copy_device_to_host(
            &self.inner.platform,
            &stream.inner.platform,
            &source.inner.platform,
            source_offset,
            bytes,
            cancelled,
        )
    }

    pub fn dispatch_add(
        &self,
        stream: &DirectMlStream,
        element_type: DirectMlElementType,
        left: &DirectMlAllocation,
        right: &DirectMlAllocation,
        output: &DirectMlAllocation,
        elements: u64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DirectMlEvent, DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        self.require_stream(stream)?;
        if element_type == DirectMlElementType::F16 && !self.inner.properties.has_fp16() {
            return Err(DirectMlExecutionError::UnsupportedElementType { element_type });
        }
        let required = elements
            .checked_mul(element_type.byte_width())
            .ok_or(DirectMlExecutionError::ElementCount { elements })?;
        for allocation in [left, right, output] {
            self.require_allocation(allocation)?;
            require_range(allocation.byte_length(), 0, required)?;
        }
        let elements = u32::try_from(elements)
            .map_err(|_| DirectMlExecutionError::ElementCount { elements })?;
        let physical_required =
            dword_rounded(required).ok_or(DirectMlExecutionError::ElementCount {
                elements: u64::from(elements),
            })?;
        let _result_reservation = self.inner.capacity.reserve(physical_required)?;
        let sequence = self.next_event_sequence()?;
        let token = acquire_bounded(&self.inner.active_events, MAXIMUM_EVENTS, true)?;
        let platform = runtime_dispatch_add(
            &self.inner.platform,
            &stream.inner.platform,
            element_type,
            &left.inner.platform,
            &right.inner.platform,
            &output.inner.platform,
            elements,
            sequence,
            &self.inner.capacity,
            cancelled,
        )?;
        Ok(self.event(sequence, platform, token))
    }

    pub fn record_event(
        &self,
        stream: &DirectMlStream,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DirectMlEvent, DirectMlExecutionError> {
        check_cancelled(cancelled)?;
        self.require_stream(stream)?;
        let sequence = self.next_event_sequence()?;
        let token = acquire_bounded(&self.inner.active_events, MAXIMUM_EVENTS, true)?;
        let platform =
            runtime_record_event(&self.inner.platform, &stream.inner.platform, sequence)?;
        Ok(self.event(sequence, platform, token))
    }

    pub fn wait_event(
        &self,
        event: &DirectMlEvent,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), DirectMlExecutionError> {
        require_identity(self.inner.identity, event.inner.session_identity)?;
        runtime_wait_event(&self.inner.platform, &event.inner.platform, cancelled)
    }

    fn event(
        &self,
        sequence: u64,
        platform: EventImplementation,
        token: BoundedResourceToken,
    ) -> DirectMlEvent {
        DirectMlEvent {
            inner: Arc::new(EventInner {
                session_identity: self.inner.identity,
                sequence,
                platform,
                _token: token,
                _session: self.inner.clone(),
            }),
        }
    }

    fn next_event_sequence(&self) -> Result<u64, DirectMlExecutionError> {
        self.inner
            .next_event
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("event sequence overflowed"),
            })
    }

    fn require_stream(&self, stream: &DirectMlStream) -> Result<(), DirectMlExecutionError> {
        require_identity(self.inner.identity, stream.inner.session_identity)
    }

    fn require_allocation(
        &self,
        allocation: &DirectMlAllocation,
    ) -> Result<(), DirectMlExecutionError> {
        require_identity(self.inner.identity, allocation.inner.session_identity)
    }
}

fn next_identity() -> Result<u64, DirectMlExecutionError> {
    NEXT_SESSION_IDENTITY
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| DirectMlExecutionError::InvalidCertifiedInputs {
            reason: diagnostic("session identity sequence overflowed"),
        })
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), DirectMlExecutionError> {
    if cancelled() {
        Err(DirectMlExecutionError::Cancelled)
    } else {
        Ok(())
    }
}

fn require_identity(expected: u64, actual: u64) -> Result<(), DirectMlExecutionError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DirectMlExecutionError::ForeignResource)
    }
}

fn require_range(available: u64, offset: u64, length: u64) -> Result<(), DirectMlExecutionError> {
    if offset.checked_add(length).is_none_or(|end| end > available) {
        Err(DirectMlExecutionError::ResourceBounds {
            offset,
            length,
            available,
        })
    } else {
        Ok(())
    }
}

fn dword_rounded(logical_bytes: u64) -> Option<u64> {
    logical_bytes.checked_add(3).map(|bytes| bytes & !3)
}

#[cfg(any(
    test,
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PersistentResourcePhase {
    InitializerOutput,
    InitializerBarrier,
    ExecutionBinding,
}

#[cfg(any(
    test,
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn planned_persistent_resource<Resource>(
    phase: PersistentResourcePhase,
    persistent_bytes: u64,
    resource: Option<&Resource>,
) -> Result<Option<&Resource>, DirectMlExecutionError> {
    match (persistent_bytes == 0, resource) {
        (true, None) => Ok(None),
        (false, Some(resource)) => Ok(Some(resource)),
        _ => Err(DirectMlExecutionError::CommandFailed {
            operation: "DirectML persistent-resource plan",
            status: -1,
            reason: diagnostic(&format!(
                "{phase:?} resource presence disagrees with the compiled operator binding properties"
            )),
        }),
    }
}

fn diagnostic(value: &str) -> String {
    if value.len() <= MAXIMUM_DIAGNOSTIC_BYTES {
        return value.to_owned();
    }
    let mut boundary = MAXIMUM_DIAGNOSTIC_BYTES;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value[..boundary].to_owned()
}

fn map_hresult(operation: &'static str, status: i32, reason: &str) -> DirectMlExecutionError {
    let code = status as u32;
    if [
        DXGI_ERROR_DEVICE_REMOVED,
        DXGI_ERROR_DEVICE_HUNG,
        DXGI_ERROR_DEVICE_RESET,
        DXGI_ERROR_DRIVER_INTERNAL_ERROR,
    ]
    .contains(&code)
    {
        DirectMlExecutionError::DeviceLost { status }
    } else {
        let reason = if code == E_OUTOFMEMORY {
            format!(
                "native allocation failed before DirectML reported a request size or capacity: {reason}"
            )
        } else {
            reason.to_owned()
        };
        DirectMlExecutionError::CommandFailed {
            operation,
            status,
            reason: diagnostic(&reason),
        }
    }
}

#[cfg(any(
    test,
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn map_allocation_hresult(
    operation: &'static str,
    status: i32,
    reason: &str,
    requested: u64,
    capacity: u64,
) -> DirectMlExecutionError {
    if status as u32 == E_OUTOFMEMORY {
        DirectMlExecutionError::OutOfMemory {
            requested,
            capacity,
        }
    } else {
        map_hresult(operation, status, reason)
    }
}

#[cfg(any(
    test,
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn available_allocation_capacity(
    local_budget: u64,
    local_usage: u64,
    non_local_budget: u64,
    non_local_usage: u64,
) -> Option<u64> {
    local_budget
        .saturating_sub(local_usage)
        .checked_add(non_local_budget.saturating_sub(non_local_usage))
        .filter(|capacity| *capacity > 0)
}

#[cfg(any(
    test,
    feature = "test-support",
    all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn lock<'a, T>(
    mutex: &'a Mutex<T>,
    name: &'static str,
) -> Result<MutexGuard<'a, T>, DirectMlExecutionError> {
    mutex
        .lock()
        .map_err(|_| DirectMlExecutionError::CommandFailed {
            operation: "DirectML execution synchronization",
            status: -1,
            reason: diagnostic(&format!("{name} lock is poisoned")),
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

#[cfg(not(any(test, feature = "test-support")))]
fn runtime_allocate(
    runtime: &RuntimeImplementation,
    bytes: u64,
) -> Result<AllocationImplementation, DirectMlExecutionError> {
    runtime.allocate(bytes)
}

#[cfg(any(test, feature = "test-support"))]
fn runtime_allocate(
    runtime: &RuntimeImplementation,
    bytes: u64,
) -> Result<AllocationImplementation, DirectMlExecutionError> {
    match runtime {
        RuntimeImplementation::Platform(runtime) => runtime
            .allocate(bytes)
            .map(AllocationImplementation::Platform),
        RuntimeImplementation::Fake(runtime) => {
            runtime.allocate(bytes).map(AllocationImplementation::Fake)
        }
    }
}

#[cfg(not(any(test, feature = "test-support")))]
fn runtime_create_stream(
    runtime: &RuntimeImplementation,
) -> Result<StreamImplementation, DirectMlExecutionError> {
    runtime.create_stream()
}

#[cfg(any(test, feature = "test-support"))]
fn runtime_create_stream(
    runtime: &RuntimeImplementation,
) -> Result<StreamImplementation, DirectMlExecutionError> {
    match runtime {
        RuntimeImplementation::Platform(runtime) => {
            runtime.create_stream().map(StreamImplementation::Platform)
        }
        RuntimeImplementation::Fake(runtime) => {
            runtime.create_stream().map(StreamImplementation::Fake)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_copy_host_to_device(
    runtime: &RuntimeImplementation,
    stream: &StreamImplementation,
    destination: &AllocationImplementation,
    offset: u64,
    bytes: &[u8],
    cancelled: &dyn Fn() -> bool,
) -> Result<(), DirectMlExecutionError> {
    #[cfg(not(any(test, feature = "test-support")))]
    return runtime.copy_host_to_device(stream, destination, offset, bytes, cancelled);
    #[cfg(any(test, feature = "test-support"))]
    match (runtime, stream, destination) {
        (
            RuntimeImplementation::Platform(runtime),
            StreamImplementation::Platform(stream),
            AllocationImplementation::Platform(destination),
        ) => runtime.copy_host_to_device(stream, destination, offset, bytes, cancelled),
        (
            RuntimeImplementation::Fake(runtime),
            StreamImplementation::Fake(stream),
            AllocationImplementation::Fake(destination),
        ) => runtime.copy_host_to_device(stream, destination, offset, bytes, cancelled),
        _ => Err(DirectMlExecutionError::ForeignResource),
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_copy_device_to_host(
    runtime: &RuntimeImplementation,
    stream: &StreamImplementation,
    source: &AllocationImplementation,
    offset: u64,
    bytes: &mut [u8],
    cancelled: &dyn Fn() -> bool,
) -> Result<(), DirectMlExecutionError> {
    #[cfg(not(any(test, feature = "test-support")))]
    return runtime.copy_device_to_host(stream, source, offset, bytes, cancelled);
    #[cfg(any(test, feature = "test-support"))]
    match (runtime, stream, source) {
        (
            RuntimeImplementation::Platform(runtime),
            StreamImplementation::Platform(stream),
            AllocationImplementation::Platform(source),
        ) => runtime.copy_device_to_host(stream, source, offset, bytes, cancelled),
        (
            RuntimeImplementation::Fake(runtime),
            StreamImplementation::Fake(stream),
            AllocationImplementation::Fake(source),
        ) => runtime.copy_device_to_host(stream, source, offset, bytes, cancelled),
        _ => Err(DirectMlExecutionError::ForeignResource),
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_dispatch_add(
    runtime: &RuntimeImplementation,
    stream: &StreamImplementation,
    element_type: DirectMlElementType,
    left: &AllocationImplementation,
    right: &AllocationImplementation,
    output: &AllocationImplementation,
    elements: u32,
    sequence: u64,
    capacity: &Arc<CapacityTracker>,
    cancelled: &dyn Fn() -> bool,
) -> Result<EventImplementation, DirectMlExecutionError> {
    #[cfg(not(any(test, feature = "test-support")))]
    return runtime.dispatch_add(
        stream,
        element_type,
        left,
        right,
        output,
        elements,
        sequence,
        capacity,
        cancelled,
    );
    #[cfg(any(test, feature = "test-support"))]
    match (runtime, stream, left, right, output) {
        (
            RuntimeImplementation::Platform(runtime),
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
                elements,
                sequence,
                capacity,
                cancelled,
            )
            .map(EventImplementation::Platform),
        (
            RuntimeImplementation::Fake(runtime),
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
                elements,
                sequence,
                capacity,
                cancelled,
            )
            .map(EventImplementation::Fake),
        _ => Err(DirectMlExecutionError::ForeignResource),
    }
}

fn runtime_record_event(
    runtime: &RuntimeImplementation,
    stream: &StreamImplementation,
    sequence: u64,
) -> Result<EventImplementation, DirectMlExecutionError> {
    #[cfg(not(any(test, feature = "test-support")))]
    return runtime.record_event(stream, sequence);
    #[cfg(any(test, feature = "test-support"))]
    match (runtime, stream) {
        (RuntimeImplementation::Platform(runtime), StreamImplementation::Platform(stream)) => {
            runtime
                .record_event(stream, sequence)
                .map(EventImplementation::Platform)
        }
        (RuntimeImplementation::Fake(runtime), StreamImplementation::Fake(stream)) => runtime
            .record_event(stream, sequence)
            .map(EventImplementation::Fake),
        _ => Err(DirectMlExecutionError::ForeignResource),
    }
}

fn runtime_wait_event(
    runtime: &RuntimeImplementation,
    event: &EventImplementation,
    cancelled: &dyn Fn() -> bool,
) -> Result<(), DirectMlExecutionError> {
    #[cfg(not(any(test, feature = "test-support")))]
    return runtime.wait_event(event, cancelled);
    #[cfg(any(test, feature = "test-support"))]
    match (runtime, event) {
        (RuntimeImplementation::Platform(runtime), EventImplementation::Platform(event)) => {
            runtime.wait_event(event, cancelled)
        }
        (RuntimeImplementation::Fake(runtime), EventImplementation::Fake(event)) => {
            runtime.wait_event(event, cancelled)
        }
        _ => Err(DirectMlExecutionError::ForeignResource),
    }
}

#[cfg(any(test, feature = "test-support"))]
mod test_fake {
    use super::*;
    use std::sync::Weak;
    use std::sync::atomic::AtomicBool;

    struct State {
        next_stream: AtomicU64,
        next_fault: Mutex<Option<i32>>,
        cancel_after_wait: AtomicBool,
        alive: Arc<AtomicBool>,
    }

    pub(super) struct Runtime {
        state: Arc<State>,
    }

    pub(super) struct Allocation {
        bytes: Mutex<Vec<u8>>,
    }

    pub(super) struct Stream {
        _identifier: u64,
    }

    pub(super) struct Event {
        _sequence: u64,
        fault: Option<i32>,
    }

    #[derive(Clone)]
    pub(super) struct Control {
        state: Weak<State>,
        #[cfg(test)]
        alive: Arc<AtomicBool>,
    }

    impl Control {
        pub(super) fn fail_next(&self, status: i32) -> Result<(), DirectMlExecutionError> {
            let state = self.state.upgrade().ok_or_else(|| {
                DirectMlExecutionError::InvalidCertifiedInputs {
                    reason: diagnostic("fake runtime was already torn down"),
                }
            })?;
            *lock(&state.next_fault, "fake fault")? = Some(status);
            Ok(())
        }

        #[cfg(test)]
        pub(super) fn is_alive(&self) -> bool {
            self.alive.load(Ordering::Acquire)
        }

        #[cfg(feature = "test-support")]
        pub(super) fn cancel_after_next_wait(&self) -> Result<(), DirectMlExecutionError> {
            let state = self.state.upgrade().ok_or_else(|| {
                DirectMlExecutionError::InvalidCertifiedInputs {
                    reason: diagnostic("fake runtime was already torn down"),
                }
            })?;
            state.cancel_after_wait.store(true, Ordering::Release);
            Ok(())
        }
    }

    impl Drop for Runtime {
        fn drop(&mut self) {
            self.state.alive.store(false, Ordering::Release);
        }
    }

    impl Runtime {
        pub(super) fn new() -> (Self, Control) {
            let alive = Arc::new(AtomicBool::new(true));
            #[cfg(test)]
            let state_alive = alive.clone();
            #[cfg(not(test))]
            let state_alive = alive;
            let state = Arc::new(State {
                next_stream: AtomicU64::new(1),
                next_fault: Mutex::new(None),
                cancel_after_wait: AtomicBool::new(false),
                alive: state_alive,
            });
            (
                Self {
                    state: state.clone(),
                },
                Control {
                    state: Arc::downgrade(&state),
                    #[cfg(test)]
                    alive,
                },
            )
        }

        pub(super) fn allocate(&self, bytes: u64) -> Result<Allocation, DirectMlExecutionError> {
            let bytes =
                usize::try_from(bytes).map_err(|_| DirectMlExecutionError::OutOfMemory {
                    requested: bytes,
                    capacity: usize::MAX as u64,
                })?;
            Ok(Allocation {
                bytes: Mutex::new(vec![0; bytes]),
            })
        }

        pub(super) fn create_stream(&self) -> Result<Stream, DirectMlExecutionError> {
            let identifier = self.state.next_stream.fetch_add(1, Ordering::AcqRel);
            Ok(Stream {
                _identifier: identifier,
            })
        }

        pub(super) fn copy_host_to_device(
            &self,
            _stream: &Stream,
            destination: &Allocation,
            offset: u64,
            bytes: &[u8],
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            let offset =
                usize::try_from(offset).map_err(|_| DirectMlExecutionError::ResourceBounds {
                    offset,
                    length: bytes.len() as u64,
                    available: 0,
                })?;
            let mut staged = bytes.to_vec();
            check_cancelled(cancelled)?;
            let mut destination = lock(&destination.bytes, "fake allocation")?;
            let end =
                offset
                    .checked_add(bytes.len())
                    .ok_or(DirectMlExecutionError::ResourceBounds {
                        offset: offset as u64,
                        length: bytes.len() as u64,
                        available: destination.len() as u64,
                    })?;
            destination[offset..end].swap_with_slice(&mut staged);
            Ok(())
        }

        pub(super) fn copy_device_to_host(
            &self,
            _stream: &Stream,
            source: &Allocation,
            offset: u64,
            bytes: &mut [u8],
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            let offset =
                usize::try_from(offset).map_err(|_| DirectMlExecutionError::ResourceBounds {
                    offset,
                    length: bytes.len() as u64,
                    available: 0,
                })?;
            let source = lock(&source.bytes, "fake allocation")?;
            let end =
                offset
                    .checked_add(bytes.len())
                    .ok_or(DirectMlExecutionError::ResourceBounds {
                        offset: offset as u64,
                        length: bytes.len() as u64,
                        available: source.len() as u64,
                    })?;
            let staged = source[offset..end].to_vec();
            drop(source);
            check_cancelled(cancelled)?;
            bytes.copy_from_slice(&staged);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            _stream: &Stream,
            element_type: DirectMlElementType,
            left: &Allocation,
            right: &Allocation,
            output: &Allocation,
            elements: u32,
            sequence: u64,
            _capacity: &Arc<CapacityTracker>,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<Event, DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            if elements == 0 {
                let fault = lock(&self.state.next_fault, "fake fault")?.take();
                return Ok(Event {
                    _sequence: sequence,
                    fault,
                });
            }
            let output_length = usize::try_from(u64::from(elements) * element_type.byte_width())
                .map_err(|_| DirectMlExecutionError::ElementCount {
                    elements: u64::from(elements),
                })?;
            let (left, right) = if std::ptr::eq(left, right) {
                let snapshot =
                    lock(&left.bytes, "fake aliased input allocation")?[..output_length].to_vec();
                (snapshot.clone(), snapshot)
            } else {
                let left = lock(&left.bytes, "fake left allocation")?[..output_length].to_vec();
                let right = lock(&right.bytes, "fake right allocation")?[..output_length].to_vec();
                (left, right)
            };
            let mut staged_output = vec![0_u8; output_length];
            match element_type {
                DirectMlElementType::F32 => {
                    for index in 0..elements as usize {
                        check_cancelled(cancelled)?;
                        let offset = index * 4;
                        let left_value =
                            f32::from_le_bytes(left[offset..offset + 4].try_into().map_err(
                                |_| DirectMlExecutionError::ElementCount {
                                    elements: u64::from(elements),
                                },
                            )?);
                        let right_value =
                            f32::from_le_bytes(right[offset..offset + 4].try_into().map_err(
                                |_| DirectMlExecutionError::ElementCount {
                                    elements: u64::from(elements),
                                },
                            )?);
                        staged_output[offset..offset + 4]
                            .copy_from_slice(&(left_value + right_value).to_le_bytes());
                    }
                }
                DirectMlElementType::F16 => {
                    for index in 0..elements as usize {
                        check_cancelled(cancelled)?;
                        let offset = index * 2;
                        let left_bits =
                            u16::from_le_bytes(left[offset..offset + 2].try_into().map_err(
                                |_| DirectMlExecutionError::ElementCount {
                                    elements: u64::from(elements),
                                },
                            )?);
                        let right_bits =
                            u16::from_le_bytes(right[offset..offset + 2].try_into().map_err(
                                |_| DirectMlExecutionError::ElementCount {
                                    elements: u64::from(elements),
                                },
                            )?);
                        let result = f32_to_f16(f16_to_f32(left_bits) + f16_to_f32(right_bits));
                        staged_output[offset..offset + 2].copy_from_slice(&result.to_le_bytes());
                    }
                }
            }
            check_cancelled(cancelled)?;
            let fault = lock(&self.state.next_fault, "fake fault")?.take();
            if fault.is_none() {
                let mut output = lock(&output.bytes, "fake output allocation")?;
                output[..output_length].copy_from_slice(&staged_output);
            }
            Ok(Event {
                _sequence: sequence,
                fault,
            })
        }

        pub(super) fn record_event(
            &self,
            _stream: &Stream,
            sequence: u64,
        ) -> Result<Event, DirectMlExecutionError> {
            let fault = lock(&self.state.next_fault, "fake fault")?.take();
            Ok(Event {
                _sequence: sequence,
                fault,
            })
        }

        pub(super) fn wait_event(
            &self,
            event: &Event,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            if self.state.cancel_after_wait.swap(false, Ordering::AcqRel) {
                return Err(DirectMlExecutionError::Cancelled);
            }
            match event.fault {
                Some(status) => Err(map_hresult("fake DirectML event", status, "injected fault")),
                None => Ok(()),
            }
        }
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
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod platform {
    use super::*;
    use crate::loader::CertifiedDirectMlExecutionInputs;

    pub(super) struct Runtime;
    pub(super) struct Allocation;
    pub(super) struct Stream;
    pub(super) struct Event;

    impl Runtime {
        pub(super) fn new(
            _inputs: CertifiedDirectMlExecutionInputs,
        ) -> Result<(Self, DirectMlDeviceProperties), DirectMlExecutionError> {
            Err(DirectMlExecutionError::UnsupportedTarget {
                target: env!("SIM_DIRECTML_TARGET").to_owned(),
            })
        }

        pub(super) fn allocate(&self, _bytes: u64) -> Result<Allocation, DirectMlExecutionError> {
            Err(unavailable())
        }

        pub(super) fn create_stream(&self) -> Result<Stream, DirectMlExecutionError> {
            Err(unavailable())
        }

        pub(super) fn copy_host_to_device(
            &self,
            _stream: &Stream,
            _destination: &Allocation,
            _offset: u64,
            _bytes: &[u8],
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            Err(unavailable())
        }

        pub(super) fn copy_device_to_host(
            &self,
            _stream: &Stream,
            _source: &Allocation,
            _offset: u64,
            _bytes: &mut [u8],
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            Err(unavailable())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            _stream: &Stream,
            _element_type: DirectMlElementType,
            _left: &Allocation,
            _right: &Allocation,
            _output: &Allocation,
            _elements: u32,
            _sequence: u64,
            _capacity: &Arc<CapacityTracker>,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<Event, DirectMlExecutionError> {
            Err(unavailable())
        }

        pub(super) fn record_event(
            &self,
            _stream: &Stream,
            _sequence: u64,
        ) -> Result<Event, DirectMlExecutionError> {
            Err(unavailable())
        }

        pub(super) fn wait_event(
            &self,
            _event: &Event,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            Err(unavailable())
        }
    }

    fn unavailable() -> DirectMlExecutionError {
        DirectMlExecutionError::UnsupportedTarget {
            target: env!("SIM_DIRECTML_TARGET").to_owned(),
        }
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod platform {
    use super::*;
    use crate::{
        DML_EXECUTION_FLAG_NONE, DML_OPERATOR_ELEMENT_WISE_ADD, DML_TENSOR_DATA_TYPE_FLOAT16,
        DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAG_NONE, DML_TENSOR_TYPE_BUFFER,
        DmlBufferTensorDesc, DmlElementWiseAddOperatorDesc, DmlOperatorDesc, DmlTensorDesc,
        MINIMUM_FEATURE_LEVEL,
        loader::{
            CertifiedDirectMlExecutionInputs, DirectMlBinding, DirectMlCommandRecorder,
            DirectMlDevice, DirectMlDispatchable,
        },
    };
    use std::{ffi::c_void, mem::ManuallyDrop, ptr};
    use windows::{
        Win32::{
            Foundation::{CloseHandle, HANDLE, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT},
            Graphics::{
                Direct3D12::{
                    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DESCRIPTOR_HEAP_DESC,
                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_FENCE_FLAG_NONE,
                    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN,
                    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER,
                    D3D12_RESOURCE_UAV_BARRIER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
                    ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
                    ID3D12Resource,
                },
                Dxgi::{
                    Common::DXGI_SAMPLE_DESC, DXGI_ADAPTER_DESC3, DXGI_ADAPTER_FLAG3_SOFTWARE,
                    DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                    DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3, IDXGIAdapter4, IDXGIFactory6,
                },
            },
            System::Threading::{CreateEventW, WaitForSingleObject},
        },
        core::{Interface, PCWSTR, Result as WindowsResult},
    };

    const FENCE_CANCELLATION_POLL_MILLISECONDS: u32 = 10;

    pub(super) struct Runtime {
        _factory: IDXGIFactory6,
        _adapter: IDXGIAdapter4,
        device: ID3D12Device,
        directml: DirectMlDevice,
        recorder: DirectMlCommandRecorder,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        allocation_capacity_bytes: u64,
        next_fence_value: AtomicU64,
        submission: Mutex<()>,
        // Retained modules must drop after every COM object whose vtable points into them.
        _inputs: CertifiedDirectMlExecutionInputs,
    }

    pub(super) struct Allocation {
        resource: Option<ID3D12Resource>,
        byte_length: u64,
    }

    pub(super) struct Stream {
        _private: (),
    }

    pub(super) struct Event {
        fence: ID3D12Fence,
        value: u64,
        completion_event: CompletionEvent,
        _retained: Vec<ID3D12Resource>,
        _descriptor_heap: Option<ID3D12DescriptorHeap>,
    }

    struct CompletionEvent {
        raw: usize,
    }

    struct SelectedExecutionRuntime {
        adapter: IDXGIAdapter4,
        device: ID3D12Device,
        directml: DirectMlDevice,
        recorder: DirectMlCommandRecorder,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
    }

    // Win32 event handles are process-owned synchronization objects that support waits from any
    // thread; this wrapper uniquely owns and closes the handle.
    unsafe impl Send for CompletionEvent {}
    unsafe impl Sync for CompletionEvent {}

    impl CompletionEvent {
        fn new() -> Result<Self, DirectMlExecutionError> {
            let handle = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
                .map_err(map_windows("CreateEventW"))?;
            Ok(Self {
                raw: handle.0 as usize,
            })
        }

        fn handle(&self) -> HANDLE {
            HANDLE(self.raw as *mut c_void)
        }
    }

    impl Drop for CompletionEvent {
        fn drop(&mut self) {
            if let Err(error) = unsafe { CloseHandle(self.handle()) } {
                eprintln!("failed to close DirectML completion event: {error}");
            }
        }
    }

    impl Runtime {
        pub(super) fn new(
            inputs: CertifiedDirectMlExecutionInputs,
        ) -> Result<(Self, DirectMlDeviceProperties), DirectMlExecutionError> {
            let factory = inputs.create_dxgi_factory6()?;
            let (selected, properties) = select_execution_runtime(&inputs, &factory)?;
            Ok((
                Self {
                    _factory: factory,
                    _adapter: selected.adapter,
                    device: selected.device,
                    directml: selected.directml,
                    recorder: selected.recorder,
                    queue: selected.queue,
                    fence: selected.fence,
                    allocation_capacity_bytes: properties.allocation_capacity_bytes(),
                    next_fence_value: AtomicU64::new(1),
                    submission: Mutex::new(()),
                    _inputs: inputs,
                },
                properties,
            ))
        }

        pub(super) fn allocate(&self, bytes: u64) -> Result<Allocation, DirectMlExecutionError> {
            if bytes == 0 {
                return Ok(Allocation {
                    resource: None,
                    byte_length: 0,
                });
            }
            let resource = create_buffer(
                &self.device,
                bytes,
                self.allocation_capacity_bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            Ok(Allocation {
                resource: Some(resource),
                byte_length: bytes,
            })
        }

        pub(super) fn create_stream(&self) -> Result<Stream, DirectMlExecutionError> {
            self.check_device()?;
            Ok(Stream { _private: () })
        }

        pub(super) fn copy_host_to_device(
            &self,
            _stream: &Stream,
            destination: &Allocation,
            offset: u64,
            bytes: &[u8],
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            if bytes.is_empty() {
                return Ok(());
            }
            let destination = required_resource(destination)?;
            let staging = create_buffer(
                &self.device,
                bytes.len() as u64,
                self.allocation_capacity_bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            unsafe {
                let mut mapped = ptr::null_mut();
                staging
                    .Map(0, None, Some(&mut mapped))
                    .map_err(map_windows("ID3D12Resource::Map"))?;
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
                staging.Unmap(0, None);
            }
            let physical_length =
                dword_rounded(bytes.len() as u64).ok_or(DirectMlExecutionError::OutOfMemory {
                    requested: bytes.len() as u64,
                    capacity: self.allocation_capacity_bytes,
                })?;
            let commit_staging = create_buffer(
                &self.device,
                physical_length,
                self.allocation_capacity_bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            let (stage_allocator, stage_list) = command_list(&self.device)?;
            check_cancelled(cancelled)?;
            unsafe {
                transition_resource(
                    &stage_list,
                    &commit_staging,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                stage_list.CopyBufferRegion(&commit_staging, 0, &staging, 0, bytes.len() as u64);
                transition_resource(
                    &stage_list,
                    &commit_staging,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                stage_list
                    .Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?;
            }
            let staged = self.submit(stage_list, vec![staging, commit_staging.clone()], None)?;
            self.wait_event(&staged, cancelled)?;
            check_cancelled(cancelled)?;

            // Submission remains cancellable until this commit list is queued; after that
            // linearization point the complete copy is waited and reported as successful.
            let (commit_allocator, commit_list) = command_list(&self.device)?;
            unsafe {
                transition_resource(
                    &commit_list,
                    &commit_staging,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                transition_resource(
                    &commit_list,
                    destination,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                commit_list.CopyBufferRegion(
                    destination,
                    offset,
                    &commit_staging,
                    0,
                    bytes.len() as u64,
                );
                transition_resource(
                    &commit_list,
                    destination,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                transition_resource(
                    &commit_list,
                    &commit_staging,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                commit_list
                    .Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?;
            }
            let committed = self.submit(commit_list, vec![commit_staging], None)?;
            self.wait_event(&committed, &|| false)?;
            drop(stage_allocator);
            drop(commit_allocator);
            Ok(())
        }

        pub(super) fn copy_device_to_host(
            &self,
            _stream: &Stream,
            source: &Allocation,
            offset: u64,
            bytes: &mut [u8],
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            if bytes.is_empty() {
                return Ok(());
            }
            let source = required_resource(source)?;
            let staging = create_buffer(
                &self.device,
                bytes.len() as u64,
                self.allocation_capacity_bytes,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let (allocator, list) = command_list(&self.device)?;
            check_cancelled(cancelled)?;
            unsafe {
                transition_resource(
                    &list,
                    source,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                list.CopyBufferRegion(&staging, 0, source, offset, bytes.len() as u64);
                transition_resource(
                    &list,
                    source,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                list.Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?;
            }
            let event = self.submit(list, vec![staging.clone()], None)?;
            self.wait_event(&event, cancelled)?;
            check_cancelled(cancelled)?;
            // The caller buffer is the publication boundary and is untouched on cancellation.
            unsafe {
                let mut mapped = ptr::null_mut();
                staging
                    .Map(0, None, Some(&mut mapped))
                    .map_err(map_windows("ID3D12Resource::Map"))?;
                ptr::copy_nonoverlapping(mapped.cast(), bytes.as_mut_ptr(), bytes.len());
                staging.Unmap(0, None);
            }
            drop(allocator);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn dispatch_add(
            &self,
            _stream: &Stream,
            element_type: DirectMlElementType,
            left: &Allocation,
            right: &Allocation,
            output: &Allocation,
            elements: u32,
            sequence: u64,
            capacity: &Arc<CapacityTracker>,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<Event, DirectMlExecutionError> {
            check_cancelled(cancelled)?;
            if elements == 0 {
                return self.record_event(_stream, sequence);
            }
            let left = required_resource(left)?;
            let right = required_resource(right)?;
            let output = required_resource(output)?;
            let logical_byte_length = u64::from(elements) * element_type.byte_width();
            let binding_byte_length =
                dword_rounded(logical_byte_length).ok_or(DirectMlExecutionError::ElementCount {
                    elements: u64::from(elements),
                })?;
            let data_type = match element_type {
                DirectMlElementType::F16 => DML_TENSOR_DATA_TYPE_FLOAT16,
                DirectMlElementType::F32 => DML_TENSOR_DATA_TYPE_FLOAT32,
            };
            let sizes = [elements];
            let buffer = DmlBufferTensorDesc {
                data_type,
                flags: DML_TENSOR_FLAG_NONE,
                dimension_count: 1,
                sizes: sizes.as_ptr(),
                strides: ptr::null(),
                total_tensor_size_in_bytes: binding_byte_length,
                guaranteed_base_offset_alignment: 0,
            };
            let tensor = DmlTensorDesc {
                tensor_type: DML_TENSOR_TYPE_BUFFER,
                desc: (&buffer as *const DmlBufferTensorDesc).cast(),
            };
            let add = DmlElementWiseAddOperatorDesc {
                a_tensor: &tensor,
                b_tensor: &tensor,
                output_tensor: &tensor,
            };
            let operator_descriptor = DmlOperatorDesc {
                operator_type: DML_OPERATOR_ELEMENT_WISE_ADD,
                desc: (&add as *const DmlElementWiseAddOperatorDesc).cast(),
            };
            let operator = self.directml.create_operator(&operator_descriptor)?;
            let compiled = self
                .directml
                .compile_operator(&operator, DML_EXECUTION_FLAG_NONE)?;
            let initializer = self.directml.create_operator_initializer(&[&compiled])?;
            initializer.reset(&[&compiled])?;
            let initialize_properties = initializer.binding_properties();
            let execute_properties = compiled.binding_properties();
            let descriptor_count = initialize_properties
                .required_descriptor_count
                .max(execute_properties.required_descriptor_count)
                .max(1);
            let descriptor_heap = create_descriptor_heap(&self.device, descriptor_count)?;
            let cpu = unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() };
            let gpu = unsafe { descriptor_heap.GetGPUDescriptorHandleForHeapStart() };
            let temporary_bytes = initialize_properties
                .temporary_resource_size
                .max(execute_properties.temporary_resource_size);
            let persistent_bytes = execute_properties.persistent_resource_size;
            let _temporary_reservation = capacity.reserve(temporary_bytes)?;
            let _persistent_reservation = capacity.reserve(persistent_bytes)?;
            let temporary = optional_default_buffer(
                &self.device,
                temporary_bytes,
                self.allocation_capacity_bytes,
            )?;
            let persistent = optional_default_buffer(
                &self.device,
                persistent_bytes,
                self.allocation_capacity_bytes,
            )?;
            let result = create_buffer(
                &self.device,
                binding_byte_length,
                self.allocation_capacity_bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;

            let initialize_table = self.directml.create_binding_table(
                DirectMlDispatchable::Initializer(&initializer),
                cpu,
                gpu,
                descriptor_count,
            )?;
            bind_temporary_resource(&initialize_table, temporary.as_ref())?;
            match planned_persistent_resource(
                PersistentResourcePhase::InitializerOutput,
                persistent_bytes,
                persistent.as_ref(),
            )? {
                Some(resource) => initialize_table.bind_outputs(&[DirectMlBinding::Buffer {
                    resource,
                    offset: 0,
                    size_in_bytes: persistent_bytes,
                }])?,
                None => initialize_table.bind_outputs(&[DirectMlBinding::None])?,
            }
            let (initialize_allocator, initialize_list) = command_list(&self.device)?;
            unsafe { initialize_list.SetDescriptorHeaps(&[Some(descriptor_heap.clone())]) };
            self.recorder.record_dispatch(
                &initialize_list,
                DirectMlDispatchable::Initializer(&initializer),
                &initialize_table,
            );
            unsafe {
                if let Some(resource) = planned_persistent_resource(
                    PersistentResourcePhase::InitializerBarrier,
                    persistent_bytes,
                    persistent.as_ref(),
                )? {
                    uav_barrier(&initialize_list, resource);
                }
                initialize_list
                    .Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?
            };
            let mut retained = Vec::new();
            if let Some(resource) = temporary.clone() {
                retained.push(resource);
            }
            if let Some(resource) = persistent.clone() {
                retained.push(resource);
            }
            let initialize_event = self.submit(
                initialize_list,
                retained.clone(),
                Some(descriptor_heap.clone()),
            )?;
            self.wait_event(&initialize_event, cancelled)?;
            drop(initialize_allocator);

            let execute_table = self.directml.create_binding_table(
                DirectMlDispatchable::Compiled(&compiled),
                cpu,
                gpu,
                descriptor_count,
            )?;
            execute_table.bind_inputs(&[
                DirectMlBinding::Buffer {
                    resource: left,
                    offset: 0,
                    size_in_bytes: binding_byte_length,
                },
                DirectMlBinding::Buffer {
                    resource: right,
                    offset: 0,
                    size_in_bytes: binding_byte_length,
                },
            ])?;
            execute_table.bind_outputs(&[DirectMlBinding::Buffer {
                resource: &result,
                offset: 0,
                size_in_bytes: binding_byte_length,
            }])?;
            bind_temporary_resource(&execute_table, temporary.as_ref())?;
            if let Some(resource) = planned_persistent_resource(
                PersistentResourcePhase::ExecutionBinding,
                persistent_bytes,
                persistent.as_ref(),
            )? {
                execute_table.bind_persistent_resource(&DirectMlBinding::Buffer {
                    resource,
                    offset: 0,
                    size_in_bytes: persistent_bytes,
                })?;
            }
            let (_allocator, list) = command_list(&self.device)?;
            unsafe { list.SetDescriptorHeaps(&[Some(descriptor_heap.clone())]) };
            self.recorder.record_dispatch(
                &list,
                DirectMlDispatchable::Compiled(&compiled),
                &execute_table,
            );
            unsafe {
                list.Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?
            };
            retained.push(result.clone());
            let event = self.submit(list, retained, Some(descriptor_heap))?;
            self.wait_event(&event, cancelled)?;
            check_cancelled(cancelled)?;
            // The public output remains unchanged until this copy is submitted. Cancellation is
            // observed before that commit boundary; an accepted commit is completed successfully.
            let (_copy_allocator, copy_list) = command_list(&self.device)?;
            unsafe {
                transition_resource(
                    &copy_list,
                    &result,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                transition_resource(
                    &copy_list,
                    output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                copy_list.CopyBufferRegion(output, 0, &result, 0, logical_byte_length);
                transition_resource(
                    &copy_list,
                    output,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                transition_resource(
                    &copy_list,
                    &result,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                copy_list
                    .Close()
                    .map_err(map_windows("ID3D12GraphicsCommandList::Close"))?;
            }
            let commit = self.submit(copy_list, vec![result], None)?;
            self.wait_event(&commit, &|| false)?;
            drop(commit);
            self.record_event(_stream, sequence)
        }

        pub(super) fn record_event(
            &self,
            _stream: &Stream,
            _sequence: u64,
        ) -> Result<Event, DirectMlExecutionError> {
            let _guard = lock(&self.submission, "DirectML submission")?;
            let completion_event = CompletionEvent::new()?;
            let value = self.next_fence_value()?;
            unsafe { self.queue.Signal(&self.fence, value) }
                .map_err(map_windows("ID3D12CommandQueue::Signal"))?;
            Ok(Event {
                fence: self.fence.clone(),
                value,
                completion_event,
                _retained: Vec::new(),
                _descriptor_heap: None,
            })
        }

        pub(super) fn wait_event(
            &self,
            event: &Event,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<(), DirectMlExecutionError> {
            let mut observed_cancel = false;
            if unsafe { event.fence.GetCompletedValue() } < event.value {
                unsafe {
                    event
                        .fence
                        .SetEventOnCompletion(event.value, event.completion_event.handle())
                }
                .map_err(map_windows("ID3D12Fence::SetEventOnCompletion"))?;
                loop {
                    let wait = unsafe {
                        WaitForSingleObject(
                            event.completion_event.handle(),
                            FENCE_CANCELLATION_POLL_MILLISECONDS,
                        )
                    };
                    if wait == WAIT_OBJECT_0 {
                        break;
                    }
                    if wait == WAIT_FAILED {
                        return Err(map_windows("WaitForSingleObject")(
                            windows::core::Error::from_win32(),
                        ));
                    }
                    if wait != WAIT_TIMEOUT {
                        return Err(DirectMlExecutionError::CommandFailed {
                            operation: "WaitForSingleObject",
                            status: wait.0 as i32,
                            reason: diagnostic(
                                "completion event returned an unexpected wait status",
                            ),
                        });
                    }
                    if unsafe { event.fence.GetCompletedValue() } >= event.value {
                        break;
                    }
                    observed_cancel |= cancelled();
                    self.check_device()?;
                }
            }
            self.check_device()?;
            if observed_cancel || cancelled() {
                Err(DirectMlExecutionError::Cancelled)
            } else {
                Ok(())
            }
        }

        fn submit(
            &self,
            list: ID3D12GraphicsCommandList,
            retained: Vec<ID3D12Resource>,
            descriptor_heap: Option<ID3D12DescriptorHeap>,
        ) -> Result<Event, DirectMlExecutionError> {
            let _guard = lock(&self.submission, "DirectML submission")?;
            let completion_event = CompletionEvent::new()?;
            let value = self.next_fence_value()?;
            let command_list: ID3D12CommandList = list
                .cast()
                .map_err(map_windows("ID3D12GraphicsCommandList::cast"))?;
            unsafe {
                self.queue.ExecuteCommandLists(&[Some(command_list)]);
                self.queue
                    .Signal(&self.fence, value)
                    .map_err(map_windows("ID3D12CommandQueue::Signal"))?;
            }
            Ok(Event {
                fence: self.fence.clone(),
                value,
                completion_event,
                _retained: retained,
                _descriptor_heap: descriptor_heap,
            })
        }

        fn next_fence_value(&self) -> Result<u64, DirectMlExecutionError> {
            self.next_fence_value
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| DirectMlExecutionError::CommandFailed {
                    operation: "DirectML fence allocation",
                    status: -1,
                    reason: diagnostic("fence sequence exhausted"),
                })
        }

        fn check_device(&self) -> Result<(), DirectMlExecutionError> {
            self.directml.removed_reason().map_err(Into::into)
        }
    }

    fn select_execution_runtime(
        inputs: &CertifiedDirectMlExecutionInputs,
        factory: &IDXGIFactory6,
    ) -> Result<(SelectedExecutionRuntime, DirectMlDeviceProperties), DirectMlExecutionError> {
        let mut rejections = Vec::new();
        for ordinal in 0..64_u32 {
            let adapter: WindowsResult<IDXGIAdapter4> = unsafe {
                factory.EnumAdapterByGpuPreference(ordinal, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            };
            let adapter = match adapter {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => {
                    return Err(map_windows("IDXGIFactory6::EnumAdapterByGpuPreference")(
                        error,
                    ));
                }
            };
            let descriptor = match unsafe { adapter.GetDesc3() } {
                Ok(descriptor) => descriptor,
                Err(error) => {
                    rejections.push(format!(
                        "adapter ordinal {ordinal} descriptor query failed: {error}"
                    ));
                    continue;
                }
            };
            if descriptor.Flags.0 & DXGI_ADAPTER_FLAG3_SOFTWARE.0 != 0 {
                continue;
            }
            match initialize_execution_runtime(inputs, adapter, descriptor) {
                Ok(selected) => return Ok(selected),
                Err(DirectMlExecutionError::DeviceLost { status }) => {
                    return Err(DirectMlExecutionError::DeviceLost { status });
                }
                Err(error) => {
                    rejections.push(format!("adapter ordinal {ordinal} was rejected: {error}"));
                }
            }
        }
        let reason = if rejections.is_empty() {
            "no non-software DXGI adapter was enumerated".to_owned()
        } else {
            format!(
                "no non-software DXGI adapter supports the reviewed DirectML path: {}",
                rejections.join("; ")
            )
        };
        Err(DirectMlExecutionError::InvalidCertifiedInputs {
            reason: diagnostic(&reason),
        })
    }

    fn initialize_execution_runtime(
        inputs: &CertifiedDirectMlExecutionInputs,
        adapter: IDXGIAdapter4,
        descriptor: DXGI_ADAPTER_DESC3,
    ) -> Result<(SelectedExecutionRuntime, DirectMlDeviceProperties), DirectMlExecutionError> {
        let device = inputs.create_d3d12_device(&adapter)?;
        let directml = inputs.create_directml_device(&device, false)?;
        let feature_level = directml.maximum_supported_feature_level(&[MINIMUM_FEATURE_LEVEL.0])?;
        if feature_level < MINIMUM_FEATURE_LEVEL.0 {
            return Err(DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("DirectML device does not support the required feature level"),
            });
        }
        let has_fp16 = directml.tensor_data_type_supported(DML_TENSOR_DATA_TYPE_FLOAT16)?;
        let recorder = directml.create_command_recorder()?;
        let queue = create_queue(&device)?;
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(map_windows("ID3D12Device::CreateFence"))?;
        let budget_adapter: IDXGIAdapter3 = adapter
            .cast()
            .map_err(map_windows("IDXGIAdapter4::cast<IDXGIAdapter3>"))?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            budget_adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
        }
        .map_err(map_windows("IDXGIAdapter3::QueryVideoMemoryInfo(local)"))?;
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            budget_adapter.QueryVideoMemoryInfo(
                0,
                DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                &mut non_local,
            )
        }
        .map_err(map_windows(
            "IDXGIAdapter3::QueryVideoMemoryInfo(non-local)",
        ))?;
        let available_capacity = available_allocation_capacity(
            local.Budget,
            local.CurrentUsage,
            non_local.Budget,
            non_local.CurrentUsage,
        )
        .ok_or_else(|| DirectMlExecutionError::InvalidCertifiedInputs {
            reason: diagnostic("DXGI adapter reports no available video-memory budget"),
        })?;
        let name = adapter_name(&descriptor)?;
        let luid = (u64::from(descriptor.AdapterLuid.HighPart as u32) << 32)
            | u64::from(descriptor.AdapterLuid.LowPart);
        let dedicated = descriptor.DedicatedVideoMemory as u64;
        let shared = descriptor.SharedSystemMemory as u64;
        let physical_capacity = dedicated
            .checked_add(shared)
            .filter(|capacity| *capacity > 0)
            .ok_or_else(|| DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("DXGI adapter reports zero physical memory capacity"),
            })?;
        let capacity = available_capacity.min(physical_capacity);
        let properties =
            DirectMlDeviceProperties::checked(name, luid, dedicated, shared, capacity, has_fp16)?;
        Ok((
            SelectedExecutionRuntime {
                adapter,
                device,
                directml,
                recorder,
                queue,
                fence,
            },
            properties,
        ))
    }

    fn create_queue(device: &ID3D12Device) -> Result<ID3D12CommandQueue, DirectMlExecutionError> {
        let descriptor = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        unsafe { device.CreateCommandQueue(&descriptor) }
            .map_err(map_windows("ID3D12Device::CreateCommandQueue"))
    }

    fn create_buffer(
        device: &ID3D12Device,
        bytes: u64,
        allocation_capacity_bytes: u64,
        heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
        flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
        state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, DirectMlExecutionError> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let descriptor = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
        };
        let mut resource = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &descriptor,
                state,
                None,
                &mut resource,
            )
        }
        .map_err(map_windows_allocation(
            "ID3D12Device::CreateCommittedResource",
            bytes,
            allocation_capacity_bytes,
        ))?;
        resource.ok_or_else(|| DirectMlExecutionError::CommandFailed {
            operation: "ID3D12Device::CreateCommittedResource",
            status: -1,
            reason: diagnostic("resource creation returned null"),
        })
    }

    fn optional_default_buffer(
        device: &ID3D12Device,
        bytes: u64,
        allocation_capacity_bytes: u64,
    ) -> Result<Option<ID3D12Resource>, DirectMlExecutionError> {
        if bytes == 0 {
            Ok(None)
        } else {
            create_buffer(
                device,
                bytes,
                allocation_capacity_bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )
            .map(Some)
        }
    }

    fn command_list(
        device: &ID3D12Device,
    ) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), DirectMlExecutionError> {
        let allocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(map_windows("ID3D12Device::CreateCommandAllocator"))?;
        let list = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
        }
        .map_err(map_windows("ID3D12Device::CreateCommandList"))?;
        Ok((allocator, list))
    }

    unsafe fn transition_resource(
        list: &ID3D12GraphicsCommandList,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        unsafe { list.ResourceBarrier(std::slice::from_ref(&barrier)) };
        unsafe {
            let transition = &mut *barrier.Anonymous.Transition;
            ManuallyDrop::drop(&mut transition.pResource);
        }
    }

    unsafe fn uav_barrier(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                }),
            },
        };
        unsafe { list.ResourceBarrier(std::slice::from_ref(&barrier)) };
        unsafe {
            let uav = &mut *barrier.Anonymous.UAV;
            ManuallyDrop::drop(&mut uav.pResource);
        }
    }

    fn create_descriptor_heap(
        device: &ID3D12Device,
        count: u32,
    ) -> Result<ID3D12DescriptorHeap, DirectMlExecutionError> {
        let descriptor = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: count,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        unsafe { device.CreateDescriptorHeap(&descriptor) }
            .map_err(map_windows("ID3D12Device::CreateDescriptorHeap"))
    }

    fn bind_temporary_resource(
        table: &crate::loader::DirectMlBindingTable<'_>,
        temporary: Option<&ID3D12Resource>,
    ) -> Result<(), DirectMlExecutionError> {
        if let Some(resource) = temporary {
            table.bind_temporary_resource(&DirectMlBinding::Buffer {
                resource,
                offset: 0,
                size_in_bytes: unsafe { resource.GetDesc() }.Width,
            })?;
        }
        Ok(())
    }

    fn required_resource(
        allocation: &Allocation,
    ) -> Result<&ID3D12Resource, DirectMlExecutionError> {
        allocation
            .resource
            .as_ref()
            .ok_or(DirectMlExecutionError::ResourceBounds {
                offset: 0,
                length: 1,
                available: allocation.byte_length,
            })
    }

    fn adapter_name(descriptor: &DXGI_ADAPTER_DESC3) -> Result<String, DirectMlExecutionError> {
        let length = descriptor
            .Description
            .iter()
            .position(|unit| *unit == 0)
            .unwrap_or(descriptor.Description.len());
        String::from_utf16(&descriptor.Description[..length]).map_err(|error| {
            DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic(&format!("DXGI adapter name is invalid UTF-16: {error}")),
            }
        })
    }

    fn map_windows(
        operation: &'static str,
    ) -> impl FnOnce(windows::core::Error) -> DirectMlExecutionError {
        move |error| map_hresult(operation, error.code().0, &error.to_string())
    }

    fn map_windows_allocation(
        operation: &'static str,
        requested: u64,
        capacity: u64,
    ) -> impl FnOnce(windows::core::Error) -> DirectMlExecutionError {
        move |error| {
            map_allocation_hresult(
                operation,
                error.code().0,
                &error.to_string(),
                requested,
                capacity,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn active() -> bool {
        false
    }

    fn fake_session(
        capacity: u64,
        has_fp16: bool,
    ) -> Result<(DirectMlExecutionSession, test_fake::Control), DirectMlExecutionError> {
        let (runtime, control) = test_fake::Runtime::new();
        let properties = DirectMlDeviceProperties::checked(
            "Injected DirectML adapter".to_owned(),
            0x1122_3344_5566_7788,
            capacity,
            0,
            capacity,
            has_fp16,
        )?;
        let identity = next_identity()?;
        let session = DirectMlExecutionSession::from_parts(
            identity,
            RuntimeImplementation::Fake(runtime),
            properties,
            Arc::new(()),
        )?;
        Ok((session, control))
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_resources_are_opaque_send_sync_and_retain_the_session()
    -> Result<(), DirectMlExecutionError> {
        assert_send_sync::<DirectMlExecutionSession>();
        assert_send_sync::<DirectMlAllocation>();
        assert_send_sync::<DirectMlStream>();
        assert_send_sync::<DirectMlEvent>();
        let (session, control) = fake_session(64, true)?;
        let allocation = session.allocate(4, &active)?;
        let stream = session.create_stream(&active)?;
        let event = session.record_event(&stream, &active)?;
        drop(session);
        assert!(control.is_alive());
        drop(event);
        drop(stream);
        drop(allocation);
        assert!(!control.is_alive());
        Ok(())
    }

    #[test]
    fn child_resources_retain_the_owned_certification_bundle() -> Result<(), DirectMlExecutionError>
    {
        struct DropMarker(Arc<AtomicBool>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let (runtime, _) = test_fake::Runtime::new();
        let properties = DirectMlDeviceProperties::checked(
            "Injected DirectML adapter".to_owned(),
            0x1122_3344_5566_7788,
            64,
            0,
            64,
            true,
        )?;
        let session = DirectMlExecutionSession::from_parts(
            next_identity()?,
            RuntimeImplementation::Fake(runtime),
            properties,
            Arc::new(DropMarker(dropped.clone())),
        )?;
        let allocation = session.allocate(4, &active)?;
        drop(session);
        assert!(!dropped.load(Ordering::Acquire));
        drop(allocation);
        assert!(dropped.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn exact_properties_capacity_bounds_and_oom_are_enforced() -> Result<(), DirectMlExecutionError>
    {
        let (session, _control) = fake_session(16, true)?;
        assert_eq!(session.properties().name(), "Injected DirectML adapter");
        assert_eq!(session.properties().adapter_luid(), 0x1122_3344_5566_7788);
        assert_eq!(session.properties().allocation_capacity_bytes(), 16);
        assert!(session.properties().has_fp16());
        let allocation = session.allocate(16, &active)?;
        assert_eq!(session.current_allocation_bytes(), 16);
        assert!(matches!(
            session.allocate(1, &active),
            Err(DirectMlExecutionError::OutOfMemory { .. })
        ));
        let stream = session.create_stream(&active)?;
        assert!(matches!(
            session.copy_host_to_device(&stream, &allocation, 0, &[1], &active),
            Err(DirectMlExecutionError::OutOfMemory { .. })
        ));
        drop(allocation);
        assert_eq!(session.current_allocation_bytes(), 0);
        assert_eq!(session.peak_allocation_bytes(), 16);

        let (without_fp16, _control) = fake_session(8, false)?;
        let stream = without_fp16.create_stream(&active)?;
        let scalar = without_fp16.allocate(2, &active)?;
        assert!(matches!(
            without_fp16.dispatch_add(
                &stream,
                DirectMlElementType::F16,
                &scalar,
                &scalar,
                &scalar,
                1,
                &active,
            ),
            Err(DirectMlExecutionError::UnsupportedElementType {
                element_type: DirectMlElementType::F16
            })
        ));
        Ok(())
    }

    #[test]
    fn cancellation_and_device_failure_do_not_publish_partial_results()
    -> Result<(), DirectMlExecutionError> {
        let (session, control) = fake_session(256, true)?;
        let stream = session.create_stream(&active)?;
        let left = session.allocate(16, &active)?;
        let right = session.allocate(16, &active)?;
        let output = session.allocate(16, &active)?;
        session.copy_host_to_device(&stream, &left, 0, &[1; 16], &active)?;
        session.copy_host_to_device(&stream, &right, 0, &[2; 16], &active)?;
        session.copy_host_to_device(&stream, &output, 0, &[0x7f; 16], &active)?;

        let copy_calls = AtomicU64::new(0);
        assert!(matches!(
            session.copy_host_to_device(&stream, &output, 0, &[9; 16], &|| {
                copy_calls.fetch_add(1, Ordering::AcqRel) >= 2
            }),
            Err(DirectMlExecutionError::Cancelled)
        ));
        let mut observed = [0; 16];
        session.copy_device_to_host(&stream, &output, 0, &mut observed, &active)?;
        assert_eq!(observed, [0x7f; 16]);

        let readback_calls = AtomicU64::new(0);
        let mut unpublished = [0xaa; 16];
        assert!(matches!(
            session.copy_device_to_host(&stream, &output, 0, &mut unpublished, &|| {
                readback_calls.fetch_add(1, Ordering::AcqRel) >= 2
            }),
            Err(DirectMlExecutionError::Cancelled)
        ));
        assert_eq!(unpublished, [0xaa; 16]);

        let dispatch_calls = AtomicU64::new(0);
        assert!(matches!(
            session.dispatch_add(
                &stream,
                DirectMlElementType::F32,
                &left,
                &right,
                &output,
                4,
                &|| dispatch_calls.fetch_add(1, Ordering::AcqRel) >= 3,
            ),
            Err(DirectMlExecutionError::Cancelled)
        ));
        session.copy_device_to_host(&stream, &output, 0, &mut observed, &active)?;
        assert_eq!(observed, [0x7f; 16]);

        control.fail_next(0x887a0005_u32 as i32)?;
        let event = session.dispatch_add(
            &stream,
            DirectMlElementType::F32,
            &left,
            &right,
            &output,
            4,
            &active,
        )?;
        assert!(matches!(
            session.wait_event(&event, &active),
            Err(DirectMlExecutionError::DeviceLost { .. })
        ));
        session.copy_device_to_host(&stream, &output, 0, &mut observed, &active)?;
        assert_eq!(observed, [0x7f; 16]);
        Ok(())
    }

    #[test]
    fn stream_and_pending_event_ownership_is_bounded_and_reusable()
    -> Result<(), DirectMlExecutionError> {
        let (session, _control) = fake_session(1, true)?;
        let mut streams = Vec::new();
        for _ in 0..MAXIMUM_STREAMS {
            streams.push(session.create_stream(&active)?);
        }
        assert!(matches!(
            session.create_stream(&active),
            Err(DirectMlExecutionError::StreamLimit {
                limit: MAXIMUM_STREAMS
            })
        ));
        let stream =
            streams
                .pop()
                .ok_or_else(|| DirectMlExecutionError::InvalidCertifiedInputs {
                    reason: diagnostic("stream fixture was unexpectedly empty"),
                })?;
        drop(stream);
        let stream = session.create_stream(&active)?;

        let mut events = Vec::new();
        for _ in 0..MAXIMUM_EVENTS {
            events.push(session.record_event(&stream, &active)?);
        }
        assert!(matches!(
            session.record_event(&stream, &active),
            Err(DirectMlExecutionError::EventLimit {
                limit: MAXIMUM_EVENTS
            })
        ));
        let event = events
            .pop()
            .ok_or_else(|| DirectMlExecutionError::InvalidCertifiedInputs {
                reason: diagnostic("event fixture was unexpectedly empty"),
            })?;
        drop(event);
        session.record_event(&stream, &active)?;
        Ok(())
    }

    #[test]
    fn f32_and_f16_add_cover_scalar_empty_transfer_and_events() -> Result<(), DirectMlExecutionError>
    {
        let (session, _control) = fake_session(128, true)?;
        let stream = session.create_stream(&active)?;
        let left = session.allocate(8, &active)?;
        let right = session.allocate(8, &active)?;
        let output = session.allocate(8, &active)?;
        session.copy_host_to_device(&stream, &left, 0, &[0, 0, 128, 63, 0, 0, 0, 192], &active)?;
        session.copy_host_to_device(&stream, &right, 0, &[0, 0, 0, 64, 0, 0, 64, 64], &active)?;
        let event = session.dispatch_add(
            &stream,
            DirectMlElementType::F32,
            &left,
            &right,
            &output,
            2,
            &active,
        )?;
        session.wait_event(&event, &active)?;
        let mut values = [0; 8];
        session.copy_device_to_host(&stream, &output, 0, &mut values, &active)?;
        assert_eq!(values, [0, 0, 64, 64, 0, 0, 128, 63]);

        let odd_left = session.allocate(12, &active)?;
        let odd_right = session.allocate(12, &active)?;
        let odd_output = session.allocate(12, &active)?;
        let mut odd_left_bytes = Vec::new();
        let mut odd_right_bytes = Vec::new();
        for value in [1.0_f32, -2.0, 0.5] {
            odd_left_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [3.0_f32, 4.0, -0.25] {
            odd_right_bytes.extend_from_slice(&value.to_le_bytes());
        }
        session.copy_host_to_device(&stream, &odd_left, 0, &odd_left_bytes, &active)?;
        session.copy_host_to_device(&stream, &odd_right, 0, &odd_right_bytes, &active)?;
        let odd_event = session.dispatch_add(
            &stream,
            DirectMlElementType::F32,
            &odd_left,
            &odd_right,
            &odd_output,
            3,
            &active,
        )?;
        session.wait_event(&odd_event, &active)?;
        let mut odd_bytes = [0_u8; 12];
        session.copy_device_to_host(&stream, &odd_output, 0, &mut odd_bytes, &active)?;
        let mut odd_values = Vec::new();
        for bytes in odd_bytes.chunks_exact(4) {
            odd_values.push(f32::from_le_bytes(bytes.try_into().map_err(|_| {
                DirectMlExecutionError::InvalidCertifiedInputs {
                    reason: diagnostic("odd-element result chunk had the wrong width"),
                }
            })?));
        }
        assert_eq!(odd_values, [4.0, 2.0, 0.25]);
        drop(odd_event);
        drop(odd_output);
        drop(odd_right);
        drop(odd_left);

        let half_left = session.allocate(2, &active)?;
        let half_right = session.allocate(2, &active)?;
        let half_output = session.allocate(2, &active)?;
        assert_eq!(half_left.byte_length(), 2);
        assert_eq!(
            session.current_allocation_bytes(),
            8 * 3 + 4 * 3,
            "logical F16 scalars use reviewed DWORD-sized resources"
        );
        session.copy_host_to_device(&stream, &half_left, 0, &0x3c00_u16.to_le_bytes(), &active)?;
        session.copy_host_to_device(&stream, &half_right, 0, &0x4000_u16.to_le_bytes(), &active)?;
        let half_event = session.dispatch_add(
            &stream,
            DirectMlElementType::F16,
            &half_left,
            &half_right,
            &half_output,
            1,
            &active,
        )?;
        session.wait_event(&half_event, &active)?;
        let mut half = [0; 2];
        session.copy_device_to_host(&stream, &half_output, 0, &mut half, &active)?;
        assert_eq!(u16::from_le_bytes(half), 0x4200);

        let allocation_before_odd_half = session.current_allocation_bytes();
        let odd_half_left = session.allocate(6, &active)?;
        let odd_half_right = session.allocate(6, &active)?;
        let odd_half_output = session.allocate(6, &active)?;
        assert_eq!(
            session.current_allocation_bytes(),
            allocation_before_odd_half + 8 * 3,
            "logical F16[3] buffers use reviewed eight-byte physical resources"
        );
        let mut odd_half_left_bytes = Vec::new();
        let mut odd_half_right_bytes = Vec::new();
        for value in [0x3c00_u16, 0x4000, 0x4200] {
            odd_half_left_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0x3c00_u16, 0xbc00, 0x3800] {
            odd_half_right_bytes.extend_from_slice(&value.to_le_bytes());
        }
        session.copy_host_to_device(&stream, &odd_half_left, 0, &odd_half_left_bytes, &active)?;
        session.copy_host_to_device(&stream, &odd_half_right, 0, &odd_half_right_bytes, &active)?;
        let odd_half_event = session.dispatch_add(
            &stream,
            DirectMlElementType::F16,
            &odd_half_left,
            &odd_half_right,
            &odd_half_output,
            3,
            &active,
        )?;
        session.wait_event(&odd_half_event, &active)?;
        let mut odd_half_bytes = [0_u8; 6];
        session.copy_device_to_host(&stream, &odd_half_output, 0, &mut odd_half_bytes, &active)?;
        let mut odd_half_values = Vec::new();
        for bytes in odd_half_bytes.chunks_exact(2) {
            odd_half_values.push(u16::from_le_bytes(bytes.try_into().map_err(|_| {
                DirectMlExecutionError::InvalidCertifiedInputs {
                    reason: diagnostic("odd F16 result chunk had the wrong width"),
                }
            })?));
        }
        assert_eq!(odd_half_values, [0x4000, 0x3c00, 0x4300]);

        let empty = session.allocate(0, &active)?;
        let empty_event = session.dispatch_add(
            &stream,
            DirectMlElementType::F32,
            &empty,
            &empty,
            &empty,
            0,
            &active,
        )?;
        session.wait_event(&empty_event, &active)?;
        Ok(())
    }

    #[test]
    fn foreign_bounds_cancellation_device_loss_and_teardown_are_typed()
    -> Result<(), DirectMlExecutionError> {
        let (session, control) = fake_session(64, true)?;
        let (foreign, _foreign_control) = fake_session(64, true)?;
        let stream = session.create_stream(&active)?;
        let allocation = session.allocate(4, &active)?;
        let foreign_allocation = foreign.allocate(4, &active)?;
        assert!(matches!(
            session.copy_host_to_device(&stream, &foreign_allocation, 0, &[1], &active),
            Err(DirectMlExecutionError::ForeignResource)
        ));
        assert!(matches!(
            session.copy_host_to_device(&stream, &allocation, 4, &[1], &active),
            Err(DirectMlExecutionError::ResourceBounds { .. })
        ));
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            session.allocate(1, &|| cancelled.load(Ordering::Acquire)),
            Err(DirectMlExecutionError::Cancelled)
        ));
        control.fail_next(0x887a0005_u32 as i32)?;
        let event = session.record_event(&stream, &active)?;
        assert!(matches!(
            session.wait_event(&event, &active),
            Err(DirectMlExecutionError::DeviceLost { .. })
        ));
        Ok(())
    }

    #[test]
    fn error_diagnostics_and_device_names_are_bounded() {
        assert!(DirectMlDeviceProperties::checked("x".repeat(257), 0, 1, 0, 1, false).is_err());
        let error = map_hresult("fixture", -1, &"x".repeat(2_000));
        match error {
            DirectMlExecutionError::CommandFailed { reason, .. } => {
                assert_eq!(reason.len(), MAXIMUM_DIAGNOSTIC_BYTES);
            }
            other => panic!("unexpected error: {other}"),
        }
        for status in [
            DXGI_ERROR_DEVICE_REMOVED,
            DXGI_ERROR_DEVICE_HUNG,
            DXGI_ERROR_DEVICE_RESET,
            DXGI_ERROR_DRIVER_INTERNAL_ERROR,
        ] {
            assert!(matches!(
                map_hresult("fixture", status as i32, "removed"),
                DirectMlExecutionError::DeviceLost { .. }
            ));
        }
        assert!(matches!(
            map_hresult("fixture", 0x887a_0008_u32 as i32, "not reviewed"),
            DirectMlExecutionError::CommandFailed { .. }
        ));
        assert!(matches!(
            map_hresult("compile operator", E_OUTOFMEMORY as i32, "driver allocation failed"),
            DirectMlExecutionError::CommandFailed {
                operation: "compile operator",
                status,
                reason,
            } if status as u32 == E_OUTOFMEMORY
                && reason.contains("before DirectML reported a request size or capacity")
        ));
        assert_eq!(
            map_allocation_hresult(
                "create staging allocation",
                E_OUTOFMEMORY as i32,
                "driver allocation failed",
                12,
                64,
            ),
            DirectMlExecutionError::OutOfMemory {
                requested: 12,
                capacity: 64,
            }
        );
    }

    #[test]
    fn physical_memory_properties_are_distinct_from_available_budget_capacity()
    -> Result<(), DirectMlExecutionError> {
        let properties = DirectMlDeviceProperties::checked(
            "Budgeted DirectML adapter".to_owned(),
            7,
            256,
            512,
            96,
            true,
        )?;
        assert_eq!(properties.dedicated_memory_bytes(), 256);
        assert_eq!(properties.shared_memory_bytes(), 512);
        assert_eq!(properties.allocation_capacity_bytes(), 96);
        assert_eq!(available_allocation_capacity(128, 64, 64, 32), Some(96));
        assert_eq!(available_allocation_capacity(32, 64, 0, 0), None);
        assert_eq!(
            available_allocation_capacity(u64::MAX, 0, 1, 0),
            None,
            "overflow may not overstate an allocation ceiling"
        );
        Ok(())
    }

    #[test]
    fn directml_buffer_resources_are_dword_rounded_without_expanding_logical_ranges() {
        assert_eq!(dword_rounded(0), Some(0));
        assert_eq!(dword_rounded(1), Some(4));
        assert_eq!(dword_rounded(2), Some(4));
        assert_eq!(dword_rounded(3), Some(4));
        assert_eq!(dword_rounded(4), Some(4));
        assert_eq!(dword_rounded(5), Some(8));
        assert_eq!(dword_rounded(u64::MAX), None);
    }

    #[test]
    fn persistent_resource_plan_records_initializer_barrier_and_execution_order()
    -> Result<(), DirectMlExecutionError> {
        let resource = 7_u8;
        let mut calls = Vec::new();
        if planned_persistent_resource(
            PersistentResourcePhase::InitializerOutput,
            4,
            Some(&resource),
        )?
        .is_some()
        {
            calls.push("initializer-output");
        }
        calls.push("initializer-dispatch");
        if planned_persistent_resource(
            PersistentResourcePhase::InitializerBarrier,
            4,
            Some(&resource),
        )?
        .is_some()
        {
            calls.push("initializer-uav-barrier");
        }
        if planned_persistent_resource(
            PersistentResourcePhase::ExecutionBinding,
            4,
            Some(&resource),
        )?
        .is_some()
        {
            calls.push("execution-persistent-binding");
        }
        calls.push("execution-dispatch");
        assert_eq!(
            calls,
            [
                "initializer-output",
                "initializer-dispatch",
                "initializer-uav-barrier",
                "execution-persistent-binding",
                "execution-dispatch",
            ]
        );
        assert_eq!(
            planned_persistent_resource::<u8>(PersistentResourcePhase::InitializerOutput, 0, None,)?,
            None
        );
        assert!(
            planned_persistent_resource(
                PersistentResourcePhase::ExecutionBinding,
                0,
                Some(&resource),
            )
            .is_err()
        );
        assert!(planned_persistent_resource::<u8>(
            PersistentResourcePhase::InitializerBarrier,
            4,
            None,
        )
        .is_err());
        Ok(())
    }
}
