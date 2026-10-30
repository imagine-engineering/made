use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

#[cfg(any(test, feature = "test-support"))]
use std::collections::BTreeMap;

use sim_types::CancellationToken;
use thiserror::Error;

use crate::{
    NpuLoadError,
    abi::AclDataType,
    loader::{NativeNpuProbe, OwnedNpuCore, RegistryCertifiedNpuImages},
};

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NpuElementType {
    F16,
    F32,
}

impl NpuElementType {
    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    const fn acl(self) -> AclDataType {
        match self {
            Self::F16 => AclDataType::Float16,
            Self::F32 => AclDataType::Float,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NpuDeviceProperties {
    device_id: u32,
    device_count: u32,
    name: String,
    api_version: (u32, u32, u32),
    physical_capacity_bytes: usize,
    allocation_capacity_bytes: usize,
}

impl NpuDeviceProperties {
    fn from_probe(probe: &NativeNpuProbe) -> Result<Self, NpuExecutionError> {
        Self::checked(
            probe.device_id,
            probe.device_count,
            probe.device_name.clone(),
            probe.api_version,
            probe.total_memory_bytes,
            probe.free_memory_bytes,
        )
    }

    fn checked(
        device_id: u32,
        device_count: u32,
        name: String,
        api_version: (u32, u32, u32),
        physical_capacity_bytes: usize,
        allocation_capacity_bytes: usize,
    ) -> Result<Self, NpuExecutionError> {
        if device_count == 0 || device_id >= device_count {
            return Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "selected device must be within a nonzero device count",
            });
        }
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "device name must contain 1..=256 non-NUL bytes",
            });
        }
        if api_version < (8, 0, 3) {
            return Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "AscendCL API version is below CANN 8.0.RC3",
            });
        }
        if physical_capacity_bytes == 0
            || allocation_capacity_bytes == 0
            || allocation_capacity_bytes > physical_capacity_bytes
        {
            return Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "allocation capacity must be nonzero and not exceed physical capacity",
            });
        }
        Ok(Self {
            device_id,
            device_count,
            name,
            api_version,
            physical_capacity_bytes,
            allocation_capacity_bytes,
        })
    }

    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    pub const fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn api_version(&self) -> (u32, u32, u32) {
        self.api_version
    }

    pub const fn physical_capacity_bytes(&self) -> usize {
        self.physical_capacity_bytes
    }

    pub const fn allocation_capacity_bytes(&self) -> usize {
        self.allocation_capacity_bytes
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NpuExecutionError {
    #[error(transparent)]
    Load(#[from] NpuLoadError),
    #[error("certified NPU execution inputs are invalid: {reason}")]
    InvalidCertifiedInputs { reason: &'static str },
    #[error("NPU execution session lock is poisoned")]
    Poisoned,
    #[error("NPU execution was cancelled")]
    Cancelled,
    #[error("NPU allocation of {requested} bytes exceeded capacity {capacity}")]
    OutOfMemory { requested: usize, capacity: usize },
    #[error("NPU resource belongs to another certified session")]
    ForeignResource,
    #[error("NPU resource is closed")]
    ClosedResource,
    #[error("NPU resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("NPU tensor dimensions are invalid or overflow the reviewed ABI")]
    InvalidDimensions,
    #[error("NPU device {device} was lost during {operation}")]
    DeviceLost {
        device: u32,
        operation: &'static str,
    },
    #[error("NPU resource identifier space is exhausted")]
    IdentifierOverflow,
}

struct CapacityTracker {
    limit: usize,
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl CapacityTracker {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<CapacityReservation, NpuExecutionError> {
        let mut current = self.current.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .filter(|next| *next <= self.limit)
                .ok_or(NpuExecutionError::OutOfMemory {
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
                        tracker: self.clone(),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

struct CapacityReservation {
    tracker: Arc<CapacityTracker>,
    bytes: usize,
}

impl Drop for CapacityReservation {
    fn drop(&mut self) {
        let previous = self.tracker.current.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes);
    }
}

struct Session {
    id: u64,
    properties: NpuDeviceProperties,
    capacity: Arc<CapacityTracker>,
    next_resource_id: AtomicU64,
    state: Mutex<RuntimeState>,
}

enum RuntimeState {
    Native(OwnedNpuCore),
    #[cfg(any(test, feature = "test-support"))]
    Fake(FakeCore),
}

impl RuntimeState {
    fn allocate(&mut self, id: u64, bytes: usize) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.allocate(id, bytes).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.allocate(id, bytes),
        }
    }

    fn release_allocation(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.release_allocation(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.release_allocation(id),
        }
    }

    fn create_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.create_stream(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.create_stream(id),
        }
    }

    fn synchronize_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.synchronize_stream(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.synchronize_stream(id),
        }
    }

    fn release_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.release_stream(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.release_stream(id),
        }
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        source: &[u8],
    ) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.copy_from_host(id, offset, source).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.copy_from_host(id, offset, source),
        }
    }

    fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core
                .copy_to_host(id, offset, destination)
                .map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.copy_to_host(id, offset, destination),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_device_to_device(
        &mut self,
        destination: u64,
        destination_offset: usize,
        source: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core
                .copy_device_to_device(
                    destination,
                    destination_offset,
                    source,
                    source_offset,
                    bytes,
                )
                .map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.copy_device_to_device(
                destination,
                destination_offset,
                source,
                source_offset,
                bytes,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        stream: u64,
        left: u64,
        right: u64,
        output: u64,
        dimensions: &[i64],
        element_type: NpuElementType,
        required_bytes: usize,
    ) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core
                .add(
                    stream,
                    left,
                    right,
                    output,
                    dimensions,
                    element_type.acl(),
                    required_bytes,
                )
                .map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.add(
                stream,
                left,
                right,
                output,
                dimensions,
                element_type,
                required_bytes,
            ),
        }
    }

    fn create_event(&mut self, id: u64, stream: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.create_event(id, stream).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.create_event(id, stream),
        }
    }

    fn synchronize_event(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.synchronize_event(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.synchronize_event(id),
        }
    }

    fn release_event(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        match self {
            Self::Native(core) => core.release_event(id).map_err(Into::into),
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.release_event(id),
        }
    }
}

#[derive(Clone)]
pub struct NpuExecutionSession {
    session: Arc<Session>,
}

impl NpuExecutionSession {
    pub fn from_registry_certified_images(
        images: RegistryCertifiedNpuImages,
        device_id: u32,
    ) -> Result<Self, NpuExecutionError> {
        let core = unsafe { OwnedNpuCore::load_certified(images, device_id) }?;
        let properties = NpuDeviceProperties::from_probe(core.probe())?;
        Self::from_state(properties, RuntimeState::Native(core))
    }

    #[cfg(feature = "test-support")]
    #[allow(clippy::too_many_arguments)]
    pub fn for_test_harness(
        device_id: u32,
        device_count: u32,
        name: impl Into<String>,
        api_version: (u32, u32, u32),
        physical_capacity_bytes: usize,
        allocation_capacity_bytes: usize,
    ) -> Result<Self, NpuExecutionError> {
        let properties = NpuDeviceProperties::checked(
            device_id,
            device_count,
            name.into(),
            api_version,
            physical_capacity_bytes,
            allocation_capacity_bytes,
        )?;
        Self::from_state(
            properties,
            RuntimeState::Fake(FakeCore {
                allocations: BTreeMap::new(),
                streams: std::collections::BTreeSet::new(),
                events: BTreeMap::new(),
                failure: None,
                cancel_after_call: None,
                drop_log: Arc::new(Mutex::new(Vec::new())),
            }),
        )
    }

    #[cfg(feature = "test-support")]
    pub fn fail_next_test_call_with_oom(&self) -> Result<(), NpuExecutionError> {
        self.with_state(|state| match state {
            RuntimeState::Fake(core) => {
                core.failure = Some(FakeFailure::OutOfMemory);
                Ok(())
            }
            RuntimeState::Native(_) => Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "test failure injection requires the NPU test harness",
            }),
        })
    }

    #[cfg(feature = "test-support")]
    pub fn fail_next_test_call_with_device_loss(&self) -> Result<(), NpuExecutionError> {
        self.with_state(|state| match state {
            RuntimeState::Fake(core) => {
                core.failure = Some(FakeFailure::DeviceLost);
                Ok(())
            }
            RuntimeState::Native(_) => Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "test failure injection requires the NPU test harness",
            }),
        })
    }

    #[cfg(feature = "test-support")]
    pub fn cancel_after_next_test_call(
        &self,
        cancellation: CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        self.with_state(|state| match state {
            RuntimeState::Fake(core) => {
                core.cancel_after_call = Some(cancellation);
                Ok(())
            }
            RuntimeState::Native(_) => Err(NpuExecutionError::InvalidCertifiedInputs {
                reason: "test cancellation injection requires the NPU test harness",
            }),
        })
    }

    fn from_state(
        properties: NpuDeviceProperties,
        state: RuntimeState,
    ) -> Result<Self, NpuExecutionError> {
        let id = NEXT_SESSION_ID
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| NpuExecutionError::IdentifierOverflow)?;
        let capacity = Arc::new(CapacityTracker {
            limit: properties.allocation_capacity_bytes,
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        Ok(Self {
            session: Arc::new(Session {
                id,
                properties,
                capacity,
                next_resource_id: AtomicU64::new(1),
                state: Mutex::new(state),
            }),
        })
    }

    pub fn properties(&self) -> &NpuDeviceProperties {
        &self.session.properties
    }

    pub fn current_allocation_bytes(&self) -> usize {
        self.session.capacity.current.load(Ordering::Acquire)
    }

    pub fn peak_allocation_bytes(&self) -> usize {
        self.session.capacity.peak.load(Ordering::Acquire)
    }

    pub fn allocate(
        &self,
        bytes: usize,
        cancellation: &CancellationToken,
    ) -> Result<NpuAllocation, NpuExecutionError> {
        check_cancellation(cancellation)?;
        if bytes == 0 {
            return Err(NpuExecutionError::ResourceBounds {
                offset: 0,
                length: 0,
                available: 0,
            });
        }
        let reservation = self.session.capacity.reserve(bytes)?;
        let id = self.next_resource_id()?;
        self.with_state(|state| state.allocate(id, bytes))?;
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release_allocation(id))?;
            return Err(error);
        }
        Ok(NpuAllocation {
            lease: Arc::new(AllocationLease {
                session: self.session.clone(),
                id,
                bytes,
                _reservation: reservation,
            }),
        })
    }

    pub fn create_stream(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NpuStream, NpuExecutionError> {
        check_cancellation(cancellation)?;
        let id = self.next_resource_id()?;
        self.with_state(|state| state.create_stream(id))?;
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release_stream(id))?;
            return Err(error);
        }
        Ok(NpuStream {
            lease: Arc::new(StreamLease {
                session: self.session.clone(),
                id,
            }),
        })
    }

    pub fn copy_from_host(
        &self,
        destination: &NpuAllocation,
        destination_offset: usize,
        source: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(destination)?;
        validate_range(destination.byte_length(), destination_offset, source.len())?;
        self.with_state(|state| {
            state.copy_from_host(destination.lease.id, destination_offset, source)
        })?;
        check_cancellation(cancellation)
    }

    pub fn copy_to_host(
        &self,
        source: &NpuAllocation,
        source_offset: usize,
        destination: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(source)?;
        validate_range(source.byte_length(), source_offset, destination.len())?;
        self.with_state(|state| state.copy_to_host(source.lease.id, source_offset, destination))?;
        check_cancellation(cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy_device_to_device(
        &self,
        destination: &NpuAllocation,
        destination_offset: usize,
        source: &NpuAllocation,
        source_offset: usize,
        bytes: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(destination)?;
        self.validate_allocation(source)?;
        validate_range(destination.byte_length(), destination_offset, bytes)?;
        validate_range(source.byte_length(), source_offset, bytes)?;
        self.with_state(|state| {
            state.copy_device_to_device(
                destination.lease.id,
                destination_offset,
                source.lease.id,
                source_offset,
                bytes,
            )
        })?;
        check_cancellation(cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &self,
        stream: &NpuStream,
        element_type: NpuElementType,
        dimensions: &[i64],
        left: &NpuAllocation,
        right: &NpuAllocation,
        output: &NpuAllocation,
        cancellation: &CancellationToken,
    ) -> Result<NpuEvent, NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_stream(stream)?;
        for allocation in [left, right, output] {
            self.validate_allocation(allocation)?;
        }
        let required_bytes = required_tensor_bytes(dimensions, element_type)?;
        for allocation in [left, right, output] {
            validate_range(allocation.byte_length(), 0, required_bytes)?;
        }
        self.with_state(|state| {
            state.add(
                stream.lease.id,
                left.lease.id,
                right.lease.id,
                output.lease.id,
                dimensions,
                element_type,
                required_bytes,
            )
        })?;
        check_cancellation(cancellation)?;
        self.record_event(stream, cancellation)
    }

    pub fn record_event(
        &self,
        stream: &NpuStream,
        cancellation: &CancellationToken,
    ) -> Result<NpuEvent, NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_stream(stream)?;
        let id = self.next_resource_id()?;
        self.with_state(|state| state.create_event(id, stream.lease.id))?;
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release_event(id))?;
            return Err(error);
        }
        Ok(NpuEvent {
            lease: Arc::new(EventLease {
                session: self.session.clone(),
                id,
                stream: stream.clone(),
                synchronized: AtomicBool::new(false),
            }),
        })
    }

    pub fn wait_event(
        &self,
        event: &NpuEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_event(event)?;
        self.with_state(|state| state.synchronize_event(event.lease.id))?;
        event.lease.synchronized.store(true, Ordering::Release);
        check_cancellation(cancellation)
    }

    pub fn synchronize_stream(
        &self,
        stream: &NpuStream,
        cancellation: &CancellationToken,
    ) -> Result<(), NpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_stream(stream)?;
        self.with_state(|state| state.synchronize_stream(stream.lease.id))?;
        check_cancellation(cancellation)
    }

    fn next_resource_id(&self) -> Result<u64, NpuExecutionError> {
        self.session
            .next_resource_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| NpuExecutionError::IdentifierOverflow)
    }

    fn validate_allocation(&self, allocation: &NpuAllocation) -> Result<(), NpuExecutionError> {
        if !Arc::ptr_eq(&self.session, &allocation.lease.session) {
            return Err(NpuExecutionError::ForeignResource);
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &NpuStream) -> Result<(), NpuExecutionError> {
        if !Arc::ptr_eq(&self.session, &stream.lease.session) {
            return Err(NpuExecutionError::ForeignResource);
        }
        Ok(())
    }

    fn validate_event(&self, event: &NpuEvent) -> Result<(), NpuExecutionError> {
        if !Arc::ptr_eq(&self.session, &event.lease.session)
            || event.lease.stream.lease.session.id != self.session.id
        {
            return Err(NpuExecutionError::ForeignResource);
        }
        Ok(())
    }

    fn with_state<Output>(
        &self,
        operation: impl FnOnce(&mut RuntimeState) -> Result<Output, NpuExecutionError>,
    ) -> Result<Output, NpuExecutionError> {
        let mut state = self
            .session
            .state
            .lock()
            .map_err(|_| NpuExecutionError::Poisoned)?;
        operation(&mut state)
    }
}

struct AllocationLease {
    session: Arc<Session>,
    id: u64,
    bytes: usize,
    _reservation: CapacityReservation,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        let result = self
            .session
            .state
            .lock()
            .map_err(|_| NpuExecutionError::Poisoned)
            .and_then(|mut state| state.release_allocation(self.id));
        if let Err(error) = result {
            eprintln!("failed to release owned NPU allocation: {error}");
        }
    }
}

#[derive(Clone)]
pub struct NpuAllocation {
    lease: Arc<AllocationLease>,
}

impl NpuAllocation {
    pub fn byte_length(&self) -> usize {
        self.lease.bytes
    }
}

struct StreamLease {
    session: Arc<Session>,
    id: u64,
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        let result = self
            .session
            .state
            .lock()
            .map_err(|_| NpuExecutionError::Poisoned)
            .and_then(|mut state| state.release_stream(self.id));
        if let Err(error) = result {
            eprintln!("failed to release owned NPU stream: {error}");
        }
    }
}

#[derive(Clone)]
pub struct NpuStream {
    lease: Arc<StreamLease>,
}

struct EventLease {
    session: Arc<Session>,
    id: u64,
    stream: NpuStream,
    synchronized: AtomicBool,
}

impl Drop for EventLease {
    fn drop(&mut self) {
        let result = self
            .session
            .state
            .lock()
            .map_err(|_| NpuExecutionError::Poisoned)
            .and_then(|mut state| state.release_event(self.id));
        if let Err(error) = result {
            eprintln!("failed to release owned NPU event: {error}");
        }
    }
}

#[derive(Clone)]
pub struct NpuEvent {
    lease: Arc<EventLease>,
}

impl NpuEvent {
    pub fn is_synchronized(&self) -> bool {
        self.lease.synchronized.load(Ordering::Acquire)
    }
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), NpuExecutionError> {
    cancellation
        .check()
        .map_err(|_| NpuExecutionError::Cancelled)
}

fn validate_range(available: usize, offset: usize, length: usize) -> Result<(), NpuExecutionError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > available) {
        return Err(NpuExecutionError::ResourceBounds {
            offset,
            length,
            available,
        });
    }
    Ok(())
}

fn required_tensor_bytes(
    dimensions: &[i64],
    element_type: NpuElementType,
) -> Result<usize, NpuExecutionError> {
    if dimensions.is_empty() || dimensions.len() > 8 {
        return Err(NpuExecutionError::InvalidDimensions);
    }
    dimensions
        .iter()
        .try_fold(element_type.byte_width(), |bytes, dimension| {
            usize::try_from(*dimension)
                .ok()
                .filter(|dimension| *dimension != 0)
                .and_then(|dimension| bytes.checked_mul(dimension))
        })
        .ok_or(NpuExecutionError::InvalidDimensions)
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy)]
enum FakeFailure {
    OutOfMemory,
    DeviceLost,
}

#[cfg(any(test, feature = "test-support"))]
struct FakeCore {
    allocations: BTreeMap<u64, Vec<u8>>,
    streams: std::collections::BTreeSet<u64>,
    events: BTreeMap<u64, u64>,
    failure: Option<FakeFailure>,
    cancel_after_call: Option<CancellationToken>,
    drop_log: Arc<Mutex<Vec<&'static str>>>,
}

#[cfg(any(test, feature = "test-support"))]
impl FakeCore {
    fn before_call(&mut self, requested: usize) -> Result<(), NpuExecutionError> {
        if let Some(cancellation) = self.cancel_after_call.take() {
            cancellation.cancel();
        }
        match self.failure.take() {
            Some(FakeFailure::OutOfMemory) => Err(NpuExecutionError::OutOfMemory {
                requested,
                capacity: requested.saturating_sub(1),
            }),
            Some(FakeFailure::DeviceLost) => Err(NpuExecutionError::DeviceLost {
                device: 0,
                operation: "fake",
            }),
            None => Ok(()),
        }
    }

    fn allocate(&mut self, id: u64, bytes: usize) -> Result<(), NpuExecutionError> {
        self.before_call(bytes)?;
        self.allocations.insert(id, vec![0; bytes]);
        Ok(())
    }

    fn release_allocation(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        self.allocations
            .remove(&id)
            .ok_or(NpuExecutionError::ClosedResource)?;
        self.log("allocation")
    }

    fn create_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        self.before_call(0)?;
        self.streams.insert(id);
        Ok(())
    }

    fn synchronize_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        self.before_call(0)?;
        if !self.streams.contains(&id) {
            return Err(NpuExecutionError::ClosedResource);
        }
        Ok(())
    }

    fn release_stream(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        if !self.streams.remove(&id) {
            return Err(NpuExecutionError::ClosedResource);
        }
        self.log("stream")
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        source: &[u8],
    ) -> Result<(), NpuExecutionError> {
        self.before_call(source.len())?;
        let allocation = self
            .allocations
            .get_mut(&id)
            .ok_or(NpuExecutionError::ClosedResource)?;
        allocation[offset..offset + source.len()].copy_from_slice(source);
        Ok(())
    }

    fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), NpuExecutionError> {
        self.before_call(destination.len())?;
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(NpuExecutionError::ClosedResource)?;
        destination.copy_from_slice(&allocation[offset..offset + destination.len()]);
        Ok(())
    }

    fn copy_device_to_device(
        &mut self,
        destination: u64,
        destination_offset: usize,
        source: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), NpuExecutionError> {
        self.before_call(bytes)?;
        let source_bytes = self
            .allocations
            .get(&source)
            .ok_or(NpuExecutionError::ClosedResource)?[source_offset..source_offset + bytes]
            .to_vec();
        let destination = self
            .allocations
            .get_mut(&destination)
            .ok_or(NpuExecutionError::ClosedResource)?;
        destination[destination_offset..destination_offset + bytes].copy_from_slice(&source_bytes);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        stream: u64,
        left: u64,
        right: u64,
        output: u64,
        _dimensions: &[i64],
        element_type: NpuElementType,
        required_bytes: usize,
    ) -> Result<(), NpuExecutionError> {
        self.before_call(required_bytes)?;
        if !self.streams.contains(&stream) {
            return Err(NpuExecutionError::ClosedResource);
        }
        let left = self
            .allocations
            .get(&left)
            .ok_or(NpuExecutionError::ClosedResource)?[..required_bytes]
            .to_vec();
        let right = self
            .allocations
            .get(&right)
            .ok_or(NpuExecutionError::ClosedResource)?[..required_bytes]
            .to_vec();
        let output = self
            .allocations
            .get_mut(&output)
            .ok_or(NpuExecutionError::ClosedResource)?;
        match element_type {
            NpuElementType::F32 => {
                for index in (0..required_bytes).step_by(4) {
                    let left_value = f32::from_le_bytes(
                        left[index..index + 4]
                            .try_into()
                            .map_err(|_| NpuExecutionError::InvalidDimensions)?,
                    );
                    let right_value = f32::from_le_bytes(
                        right[index..index + 4]
                            .try_into()
                            .map_err(|_| NpuExecutionError::InvalidDimensions)?,
                    );
                    output[index..index + 4]
                        .copy_from_slice(&(left_value + right_value).to_le_bytes());
                }
            }
            NpuElementType::F16 => {
                for index in (0..required_bytes).step_by(2) {
                    let left_bits = u16::from_le_bytes(
                        left[index..index + 2]
                            .try_into()
                            .map_err(|_| NpuExecutionError::InvalidDimensions)?,
                    );
                    let right_bits = u16::from_le_bytes(
                        right[index..index + 2]
                            .try_into()
                            .map_err(|_| NpuExecutionError::InvalidDimensions)?,
                    );
                    let result = encode_f16(decode_f16(left_bits) + decode_f16(right_bits));
                    output[index..index + 2].copy_from_slice(&result.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    fn create_event(&mut self, id: u64, stream: u64) -> Result<(), NpuExecutionError> {
        self.before_call(0)?;
        if !self.streams.contains(&stream) {
            return Err(NpuExecutionError::ClosedResource);
        }
        self.events.insert(id, stream);
        Ok(())
    }

    fn synchronize_event(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        self.before_call(0)?;
        self.events
            .get(&id)
            .ok_or(NpuExecutionError::ClosedResource)?;
        Ok(())
    }

    fn release_event(&mut self, id: u64) -> Result<(), NpuExecutionError> {
        self.events
            .remove(&id)
            .ok_or(NpuExecutionError::ClosedResource)?;
        self.log("event")
    }

    fn log(&self, entry: &'static str) -> Result<(), NpuExecutionError> {
        self.drop_log
            .lock()
            .map_err(|_| NpuExecutionError::Poisoned)?
            .push(entry);
        Ok(())
    }
}

#[cfg(any(test, feature = "test-support"))]
fn decode_f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = u32::from(fraction);
            let mut exponent = 113_u32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent = exponent.saturating_sub(1);
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13),
    };
    f32::from_bits(value)
}

#[cfg(any(test, feature = "test-support"))]
fn encode_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x007f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = fraction | 0x0080_0000;
        let Ok(shift) = u32::try_from(14 - exponent) else {
            return sign;
        };
        let rounded = (mantissa + (1_u32 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    let rounded = fraction + 0x0000_1000;
    if rounded & 0x0080_0000 != 0 {
        let exponent = exponent + 1;
        if exponent >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((exponent as u16) << 10);
    }
    sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harness(
        capacity: usize,
    ) -> Result<(NpuExecutionSession, Arc<Mutex<Vec<&'static str>>>), NpuExecutionError> {
        let drop_log = Arc::new(Mutex::new(Vec::new()));
        let properties = NpuDeviceProperties::checked(
            0,
            1,
            "Ascend910B".to_owned(),
            (8, 0, 3),
            capacity * 2,
            capacity,
        )?;
        let session = NpuExecutionSession::from_state(
            properties,
            RuntimeState::Fake(FakeCore {
                allocations: BTreeMap::new(),
                streams: std::collections::BTreeSet::new(),
                events: BTreeMap::new(),
                failure: None,
                cancel_after_call: None,
                drop_log: drop_log.clone(),
            }),
        )?;
        Ok((session, drop_log))
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn owned_session_executes_f16_f32_events_and_teardown() -> Result<(), NpuExecutionError> {
        assert_send_sync::<NpuExecutionSession>();
        assert_send_sync::<NpuAllocation>();
        assert_send_sync::<NpuStream>();
        assert_send_sync::<NpuEvent>();

        let cancellation = CancellationToken::default();
        let (session, drop_log) = harness(256)?;
        assert_eq!(session.properties().device_id(), 0);
        assert_eq!(session.properties().device_count(), 1);
        assert_eq!(session.properties().name(), "Ascend910B");
        assert_eq!(session.properties().api_version(), (8, 0, 3));
        assert_eq!(session.properties().physical_capacity_bytes(), 512);
        assert_eq!(session.properties().allocation_capacity_bytes(), 256);
        let stream = session.create_stream(&cancellation)?;
        let left = session.allocate(16, &cancellation)?;
        let right = session.allocate(16, &cancellation)?;
        let output = session.allocate(16, &cancellation)?;
        session.copy_from_host(&left, 0, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]), &cancellation)?;
        session.copy_from_host(&right, 0, &f32_bytes(&[0.5, 1.0, 1.5, 2.0]), &cancellation)?;
        let event = session.add(
            &stream,
            NpuElementType::F32,
            &[4],
            &left,
            &right,
            &output,
            &cancellation,
        )?;
        session.wait_event(&event, &cancellation)?;
        let mut output_bytes = vec![0; 16];
        session.copy_to_host(&output, 0, &mut output_bytes, &cancellation)?;
        assert_eq!(output_bytes, f32_bytes(&[1.5, 3.0, 4.5, 6.0]));

        let half_left = session.allocate(4, &cancellation)?;
        let half_right = session.allocate(4, &cancellation)?;
        let half_output = session.allocate(4, &cancellation)?;
        let half_values = [encode_f16(1.0), encode_f16(2.0)];
        let half_right_values = [encode_f16(0.5), encode_f16(1.0)];
        let half_left_bytes = half_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let half_right_bytes = half_right_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        session.copy_from_host(&half_left, 0, &half_left_bytes, &cancellation)?;
        session.copy_from_host(&half_right, 0, &half_right_bytes, &cancellation)?;
        let half_event = session.add(
            &stream,
            NpuElementType::F16,
            &[2],
            &half_left,
            &half_right,
            &half_output,
            &cancellation,
        )?;
        session.wait_event(&half_event, &cancellation)?;
        let mut half_result = vec![0; 4];
        session.copy_to_host(&half_output, 0, &mut half_result, &cancellation)?;
        assert_eq!(
            u16::from_le_bytes([half_result[0], half_result[1]]),
            encode_f16(1.5)
        );
        assert_eq!(
            u16::from_le_bytes([half_result[2], half_result[3]]),
            encode_f16(3.0)
        );
        assert_eq!(session.current_allocation_bytes(), 60);
        assert_eq!(session.peak_allocation_bytes(), 60);

        drop(half_event);
        drop(event);
        drop(half_output);
        drop(half_right);
        drop(half_left);
        drop(output);
        drop(right);
        drop(left);
        drop(stream);
        assert_eq!(session.current_allocation_bytes(), 0);
        let log = drop_log.lock().map_err(|_| NpuExecutionError::Poisoned)?;
        assert_eq!(log.iter().filter(|entry| **entry == "event").count(), 2);
        assert_eq!(
            log.iter().filter(|entry| **entry == "allocation").count(),
            6
        );
        assert_eq!(log.last(), Some(&"stream"));
        Ok(())
    }

    #[test]
    fn bounds_cancellation_capacity_and_foreign_resources_fail_typed()
    -> Result<(), NpuExecutionError> {
        let cancellation = CancellationToken::default();
        let (session, _) = harness(32)?;
        let (other, _) = harness(32)?;
        let allocation = session.allocate(16, &cancellation)?;
        assert!(matches!(
            session.allocate(17, &cancellation),
            Err(NpuExecutionError::OutOfMemory { .. })
        ));
        let destination = session.allocate(16, &cancellation)?;
        let input = [1_u8, 2, 3, 4];
        session.copy_from_host(&allocation, 4, &input, &cancellation)?;
        session.copy_device_to_device(
            &destination,
            8,
            &allocation,
            4,
            input.len(),
            &cancellation,
        )?;
        let mut copied = [0_u8; 4];
        session.copy_to_host(&destination, 8, &mut copied, &cancellation)?;
        assert_eq!(copied, input);
        assert!(matches!(
            session.copy_from_host(&allocation, 15, &[1, 2], &cancellation),
            Err(NpuExecutionError::ResourceBounds { .. })
        ));
        let stream = session.create_stream(&cancellation)?;
        assert_eq!(
            session
                .add(
                    &stream,
                    NpuElementType::F32,
                    &[0],
                    &allocation,
                    &destination,
                    &allocation,
                    &cancellation,
                )
                .err(),
            Some(NpuExecutionError::InvalidDimensions)
        );
        let foreign = other.allocate(4, &cancellation)?;
        assert_eq!(
            session.copy_from_host(&foreign, 0, &[1], &cancellation),
            Err(NpuExecutionError::ForeignResource)
        );
        cancellation.cancel();
        assert_eq!(
            session.create_stream(&cancellation).err(),
            Some(NpuExecutionError::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn injected_oom_device_loss_and_post_call_cancellation_are_atomic()
    -> Result<(), NpuExecutionError> {
        let cancellation = CancellationToken::default();
        let (session, _) = harness(64)?;
        session.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(NpuExecutionError::InvalidCertifiedInputs {
                    reason: "test harness state is not fake",
                });
            };
            core.failure = Some(FakeFailure::OutOfMemory);
            Ok(())
        })?;
        assert!(matches!(
            session.allocate(8, &cancellation),
            Err(NpuExecutionError::OutOfMemory { .. })
        ));
        assert_eq!(session.current_allocation_bytes(), 0);
        session.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(NpuExecutionError::InvalidCertifiedInputs {
                    reason: "test harness state is not fake",
                });
            };
            core.failure = Some(FakeFailure::DeviceLost);
            Ok(())
        })?;
        assert!(matches!(
            session.create_stream(&cancellation),
            Err(NpuExecutionError::DeviceLost { .. })
        ));

        let post_call_cancellation = CancellationToken::default();
        session.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(NpuExecutionError::InvalidCertifiedInputs {
                    reason: "test harness state is not fake",
                });
            };
            core.cancel_after_call = Some(post_call_cancellation.clone());
            Ok(())
        })?;
        assert_eq!(
            session.allocate(8, &post_call_cancellation).err(),
            Some(NpuExecutionError::Cancelled)
        );
        assert_eq!(session.current_allocation_bytes(), 0);
        Ok(())
    }
}
