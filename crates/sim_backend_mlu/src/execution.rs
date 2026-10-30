use crate::{
    CnnlStatus, CnrtStatus, MluAbiProbe, MluLoadError, RegistryCertifiedImage,
    loader::{CertifiedMluImages, MluRuntime, SerializedMluCore},
};
use sim_types::CancellationToken;
#[cfg(test)]
use std::collections::BTreeMap;
use std::{
    any::Any,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use thiserror::Error;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MluElementType {
    F16,
    F32,
}

impl MluElementType {
    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MluExecutionError {
    #[error(transparent)]
    Load(#[from] MluLoadError),
    #[error("MLU execution runtime lock is poisoned")]
    Poisoned,
    #[error("MLU execution resource belongs to another certified runtime")]
    ForeignResource,
    #[error("MLU execution resource is closed")]
    ClosedResource,
    #[error("MLU resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("MLU device {device} is outside certified count {device_count}")]
    InvalidDevice { device: u32, device_count: u32 },
    #[error("MLU allocation of {requested} bytes failed")]
    OutOfMemory { requested: usize },
    #[error("MLU device {device} was lost during {operation}")]
    DeviceLost {
        device: u32,
        operation: &'static str,
    },
    #[error("MLU vendor operation {operation} failed with status {status}")]
    VendorCallFailed {
        operation: &'static str,
        status: i32,
    },
    #[error("invalid MLU execution argument: {reason}")]
    InvalidArgument { reason: &'static str },
    #[error("MLU execution resource identifier space is exhausted")]
    IdentifierOverflow,
    #[error("MLU execution was cancelled")]
    Cancelled,
}

struct Session {
    runtime_id: u64,
    probe: MluAbiProbe,
    device_count: u32,
    next_resource_id: AtomicU64,
    state: Mutex<RuntimeState>,
    _certification: Arc<dyn Any + Send + Sync>,
}

enum RuntimeState {
    Native(SerializedMluCore),
    #[cfg(test)]
    Fake(FakeCore),
}

impl RuntimeState {
    fn allocate(&mut self, id: u64, device: u32, bytes: usize) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .allocate(id, device, bytes)
                .map_err(|error| map_load_error(error, bytes, device)),
            #[cfg(test)]
            Self::Fake(core) => core.allocate(id, device, bytes),
        }
    }

    fn release_allocation(&mut self, id: u64, device: u32) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .release_allocation(id)
                .map_err(|error| map_load_error(error, 0, device)),
            #[cfg(test)]
            Self::Fake(core) => core.release_allocation(id),
        }
    }

    fn create_stream(&mut self, id: u64, device: u32) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .create_queue(id, device)
                .map_err(|error| map_load_error(error, 0, device)),
            #[cfg(test)]
            Self::Fake(core) => core.create_stream(id, device),
        }
    }

    fn synchronize_stream(&mut self, id: u64, device: u32) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .synchronize_queue(id)
                .map_err(|error| map_load_error(error, 0, device)),
            #[cfg(test)]
            Self::Fake(core) => core.synchronize_stream(id),
        }
    }

    fn release_stream(&mut self, id: u64, device: u32) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .release_queue(id)
                .map_err(|error| map_load_error(error, 0, device)),
            #[cfg(test)]
            Self::Fake(core) => core.release_stream(id),
        }
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        device: u32,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .copy_from_host(id, offset, bytes)
                .map_err(|error| map_load_error(error, bytes.len(), device)),
            #[cfg(test)]
            Self::Fake(core) => core.copy_from_host(id, offset, bytes),
        }
    }

    fn copy_to_host(
        &mut self,
        id: u64,
        device: u32,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .copy_to_host(id, offset, bytes)
                .map_err(|error| map_load_error(error, bytes.len(), device)),
            #[cfg(test)]
            Self::Fake(core) => core.copy_to_host(id, offset, bytes),
        }
    }

    fn copy_device_to_device(
        &mut self,
        device: u32,
        destination: u64,
        destination_offset: usize,
        source: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .copy_device_to_device(
                    destination,
                    destination_offset,
                    source,
                    source_offset,
                    bytes,
                )
                .map_err(|error| map_load_error(error, bytes, device)),
            #[cfg(test)]
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
        device: u32,
        stream: u64,
        left: u64,
        right: u64,
        output: u64,
        dimensions: &[i32],
        element_type: MluElementType,
        requested: usize,
    ) -> Result<(), MluExecutionError> {
        match self {
            Self::Native(core) => core
                .add(
                    stream,
                    left,
                    right,
                    output,
                    dimensions,
                    match element_type {
                        MluElementType::F16 => crate::CnnlDataType::Half,
                        MluElementType::F32 => crate::CnnlDataType::Float,
                    },
                    element_type.byte_width(),
                )
                .map_err(|error| map_load_error(error, requested, device)),
            #[cfg(test)]
            Self::Fake(core) => core.add(stream, left, right, output, dimensions, element_type),
        }
    }
}

#[derive(Clone)]
pub struct MluExecutionRuntime {
    session: Arc<Session>,
}

impl MluExecutionRuntime {
    /// Loads one owned, serialized MLU execution session from exact registry-certified images.
    ///
    /// # Safety
    ///
    /// `images` must be direct projections of live `sim_runtime::NativeFfiRegistry`
    /// certificates retained by `certification`. Each path must name the exact immutable sealed
    /// image covered by its certificate for the complete lifetime of the returned runtime.
    pub unsafe fn load_certified(
        certification: Arc<dyn Any + Send + Sync>,
        images: impl IntoIterator<Item = RegistryCertifiedImage>,
    ) -> Result<Self, MluExecutionError> {
        let certified = unsafe {
            CertifiedMluImages::from_registry_certificates(certification.as_ref(), images)
        }?;
        let runtime = MluRuntime::load(&certified)?;
        let core = runtime.into_serialized_core()?;
        let probe = core.probe()?;
        let device_count = core.device_count();
        Self::from_state(
            probe,
            device_count,
            RuntimeState::Native(core),
            certification,
        )
    }

    fn from_state(
        probe: MluAbiProbe,
        device_count: u32,
        state: RuntimeState,
        certification: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, MluExecutionError> {
        if device_count == 0 {
            return Err(MluExecutionError::InvalidArgument {
                reason: "certified MLU runtime reports zero devices",
            });
        }
        let runtime_id = NEXT_RUNTIME_ID
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MluExecutionError::IdentifierOverflow)?;
        Ok(Self {
            session: Arc::new(Session {
                runtime_id,
                probe,
                device_count,
                next_resource_id: AtomicU64::new(1),
                state: Mutex::new(state),
                _certification: certification,
            }),
        })
    }

    pub fn probe(&self) -> &MluAbiProbe {
        &self.session.probe
    }

    pub fn device_count(&self) -> u32 {
        self.session.device_count
    }

    pub fn allocate(
        &self,
        device: u32,
        byte_length: usize,
        cancellation: &CancellationToken,
    ) -> Result<MluExecutionAllocation, MluExecutionError> {
        check_cancellation(cancellation)?;
        self.check_device(device)?;
        if byte_length == 0 {
            return Err(MluExecutionError::InvalidArgument {
                reason: "MLU allocation size must be nonzero",
            });
        }
        let id = self.next_resource_id()?;
        self.with_state(|state| state.allocate(id, device, byte_length))?;
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release_allocation(id, device))?;
            return Err(error);
        }
        Ok(MluExecutionAllocation {
            lease: Arc::new(AllocationLease {
                session: self.session.clone(),
                id,
                device,
                byte_length,
            }),
        })
    }

    pub fn create_stream(
        &self,
        device: u32,
        cancellation: &CancellationToken,
    ) -> Result<MluExecutionStream, MluExecutionError> {
        check_cancellation(cancellation)?;
        self.check_device(device)?;
        let id = self.next_resource_id()?;
        self.with_state(|state| state.create_stream(id, device))?;
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release_stream(id, device))?;
            return Err(error);
        }
        Ok(MluExecutionStream {
            lease: Arc::new(StreamLease {
                session: self.session.clone(),
                id,
                device,
            }),
        })
    }

    pub fn copy_from_host(
        &self,
        destination: &MluExecutionAllocation,
        destination_offset: usize,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), MluExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(destination)?;
        validate_range(destination.byte_length(), destination_offset, bytes.len())?;
        self.with_state(|state| {
            state.copy_from_host(
                destination.lease.id,
                destination.device(),
                destination_offset,
                bytes,
            )
        })?;
        check_cancellation(cancellation)
    }

    pub fn copy_to_host(
        &self,
        source: &MluExecutionAllocation,
        source_offset: usize,
        bytes: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), MluExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(source)?;
        validate_range(source.byte_length(), source_offset, bytes.len())?;
        self.with_state(|state| {
            state.copy_to_host(source.lease.id, source.device(), source_offset, bytes)
        })?;
        check_cancellation(cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy_device_to_device(
        &self,
        destination: &MluExecutionAllocation,
        destination_offset: usize,
        source: &MluExecutionAllocation,
        source_offset: usize,
        byte_length: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), MluExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(destination)?;
        self.validate_allocation(source)?;
        if destination.device() != source.device() {
            return Err(MluExecutionError::ForeignResource);
        }
        validate_range(destination.byte_length(), destination_offset, byte_length)?;
        validate_range(source.byte_length(), source_offset, byte_length)?;
        self.with_state(|state| {
            state.copy_device_to_device(
                destination.device(),
                destination.lease.id,
                destination_offset,
                source.lease.id,
                source_offset,
                byte_length,
            )
        })?;
        check_cancellation(cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &self,
        stream: &MluExecutionStream,
        element_type: MluElementType,
        dimensions: &[i32],
        left: &MluExecutionAllocation,
        right: &MluExecutionAllocation,
        output: &MluExecutionAllocation,
        cancellation: &CancellationToken,
    ) -> Result<MluExecutionEvent, MluExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_stream(stream)?;
        for allocation in [left, right, output] {
            self.validate_allocation(allocation)?;
            if allocation.device() != stream.device() {
                return Err(MluExecutionError::ForeignResource);
            }
        }
        let required = required_tensor_bytes(dimensions, element_type)?;
        for allocation in [left, right, output] {
            validate_range(allocation.byte_length(), 0, required)?;
        }
        self.with_state(|state| {
            state.add(
                stream.device(),
                stream.lease.id,
                left.lease.id,
                right.lease.id,
                output.lease.id,
                dimensions,
                element_type,
                required,
            )
        })?;
        check_cancellation(cancellation)?;
        self.record_event(stream, cancellation)
    }

    pub fn record_event(
        &self,
        stream: &MluExecutionStream,
        cancellation: &CancellationToken,
    ) -> Result<MluExecutionEvent, MluExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_stream(stream)?;
        self.with_state(|state| state.synchronize_stream(stream.lease.id, stream.device()))?;
        check_cancellation(cancellation)?;
        Ok(MluExecutionEvent {
            runtime_id: self.session.runtime_id,
            stream: stream.clone(),
            synchronized: Arc::new(AtomicBool::new(true)),
        })
    }

    pub fn wait_event(
        &self,
        event: &MluExecutionEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), MluExecutionError> {
        check_cancellation(cancellation)?;
        if event.runtime_id != self.session.runtime_id {
            return Err(MluExecutionError::ForeignResource);
        }
        self.validate_stream(&event.stream)?;
        self.with_state(|state| {
            state.synchronize_stream(event.stream.lease.id, event.stream.device())
        })?;
        event.synchronized.store(true, Ordering::Release);
        check_cancellation(cancellation)
    }

    fn next_resource_id(&self) -> Result<u64, MluExecutionError> {
        self.session
            .next_resource_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MluExecutionError::IdentifierOverflow)
    }

    fn check_device(&self, device: u32) -> Result<(), MluExecutionError> {
        if device >= self.session.device_count {
            Err(MluExecutionError::InvalidDevice {
                device,
                device_count: self.session.device_count,
            })
        } else {
            Ok(())
        }
    }

    fn validate_allocation(
        &self,
        allocation: &MluExecutionAllocation,
    ) -> Result<(), MluExecutionError> {
        if !Arc::ptr_eq(&allocation.lease.session, &self.session) {
            return Err(MluExecutionError::ForeignResource);
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &MluExecutionStream) -> Result<(), MluExecutionError> {
        if !Arc::ptr_eq(&stream.lease.session, &self.session) {
            return Err(MluExecutionError::ForeignResource);
        }
        Ok(())
    }

    fn with_state<Output>(
        &self,
        operation: impl FnOnce(&mut RuntimeState) -> Result<Output, MluExecutionError>,
    ) -> Result<Output, MluExecutionError> {
        let mut state = self
            .session
            .state
            .lock()
            .map_err(|_| MluExecutionError::Poisoned)?;
        operation(&mut state)
    }
}

struct AllocationLease {
    session: Arc<Session>,
    id: u64,
    device: u32,
    byte_length: usize,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        let result = self
            .session
            .state
            .lock()
            .map_err(|_| MluExecutionError::Poisoned)
            .and_then(|mut state| state.release_allocation(self.id, self.device));
        if let Err(error) = result {
            eprintln!("failed to release owned MLU allocation: {error}");
        }
    }
}

#[derive(Clone)]
pub struct MluExecutionAllocation {
    lease: Arc<AllocationLease>,
}

impl MluExecutionAllocation {
    pub fn device(&self) -> u32 {
        self.lease.device
    }

    pub fn byte_length(&self) -> usize {
        self.lease.byte_length
    }
}

struct StreamLease {
    session: Arc<Session>,
    id: u64,
    device: u32,
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        let result = self
            .session
            .state
            .lock()
            .map_err(|_| MluExecutionError::Poisoned)
            .and_then(|mut state| state.release_stream(self.id, self.device));
        if let Err(error) = result {
            eprintln!("failed to release owned MLU stream: {error}");
        }
    }
}

#[derive(Clone)]
pub struct MluExecutionStream {
    lease: Arc<StreamLease>,
}

impl MluExecutionStream {
    pub fn device(&self) -> u32 {
        self.lease.device
    }
}

#[derive(Clone)]
pub struct MluExecutionEvent {
    runtime_id: u64,
    stream: MluExecutionStream,
    synchronized: Arc<AtomicBool>,
}

impl MluExecutionEvent {
    pub fn device(&self) -> u32 {
        self.stream.device()
    }

    pub fn is_synchronized(&self) -> bool {
        self.synchronized.load(Ordering::Acquire)
    }
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), MluExecutionError> {
    cancellation
        .check()
        .map_err(|_| MluExecutionError::Cancelled)
}

fn validate_range(available: usize, offset: usize, length: usize) -> Result<(), MluExecutionError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > available) {
        return Err(MluExecutionError::ResourceBounds {
            offset,
            length,
            available,
        });
    }
    Ok(())
}

fn required_tensor_bytes(
    dimensions: &[i32],
    element_type: MluElementType,
) -> Result<usize, MluExecutionError> {
    if dimensions.is_empty() || dimensions.len() > 8 {
        return Err(MluExecutionError::InvalidArgument {
            reason: "MLU Add rank must be 1 through 8",
        });
    }
    dimensions
        .iter()
        .try_fold(element_type.byte_width(), |bytes, dimension| {
            usize::try_from(*dimension)
                .ok()
                .filter(|dimension| *dimension != 0)
                .and_then(|dimension| bytes.checked_mul(dimension))
        })
        .ok_or(MluExecutionError::InvalidArgument {
            reason: "MLU Add dimensions must be positive and fit the host ABI",
        })
}

fn map_load_error(error: MluLoadError, requested: usize, device: u32) -> MluExecutionError {
    match error {
        MluLoadError::CallFailed { operation, status }
            if operation.starts_with("cnrt") && status == CnrtStatus::NO_MEMORY.0 =>
        {
            MluExecutionError::OutOfMemory { requested }
        }
        MluLoadError::CallFailed { operation, status }
            if operation.starts_with("cnrt") && status == CnrtStatus::NO_DEVICE.0 =>
        {
            MluExecutionError::DeviceLost { device, operation }
        }
        MluLoadError::CallFailed { operation, status }
            if operation.starts_with("cnnl") && status == CnnlStatus::ALLOCATION_FAILED.0 =>
        {
            MluExecutionError::OutOfMemory { requested }
        }
        MluLoadError::CallFailed { operation, status } => {
            MluExecutionError::VendorCallFailed { operation, status }
        }
        other => MluExecutionError::Load(other),
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum FakeFailure {
    OutOfMemory,
    DeviceLost,
}

#[cfg(test)]
struct FakeCore {
    device_count: u32,
    allocations: BTreeMap<u64, (u32, Vec<u8>)>,
    streams: BTreeMap<u64, u32>,
    failure: Option<FakeFailure>,
    cancel_after_next_call: Option<CancellationToken>,
    selected_devices: Vec<u32>,
    drop_log: Arc<Mutex<Vec<&'static str>>>,
}

#[cfg(test)]
impl FakeCore {
    fn check(&mut self, device: u32, requested: usize) -> Result<(), MluExecutionError> {
        if device >= self.device_count {
            return Err(MluExecutionError::InvalidDevice {
                device,
                device_count: self.device_count,
            });
        }
        self.selected_devices.push(device);
        if let Some(cancellation) = self.cancel_after_next_call.take() {
            cancellation.cancel();
        }
        match self.failure.take() {
            Some(FakeFailure::OutOfMemory) => Err(MluExecutionError::OutOfMemory { requested }),
            Some(FakeFailure::DeviceLost) => Err(MluExecutionError::DeviceLost {
                device,
                operation: "fake",
            }),
            None => Ok(()),
        }
    }

    fn allocate(&mut self, id: u64, device: u32, bytes: usize) -> Result<(), MluExecutionError> {
        self.check(device, bytes)?;
        self.allocations.insert(id, (device, vec![0; bytes]));
        Ok(())
    }

    fn release_allocation(&mut self, id: u64) -> Result<(), MluExecutionError> {
        self.allocations
            .remove(&id)
            .ok_or(MluExecutionError::ClosedResource)?;
        self.drop_log
            .lock()
            .map_err(|_| MluExecutionError::Poisoned)?
            .push("allocation");
        Ok(())
    }

    fn create_stream(&mut self, id: u64, device: u32) -> Result<(), MluExecutionError> {
        self.check(device, 0)?;
        self.streams.insert(id, device);
        Ok(())
    }

    fn synchronize_stream(&mut self, id: u64) -> Result<(), MluExecutionError> {
        let device = *self
            .streams
            .get(&id)
            .ok_or(MluExecutionError::ClosedResource)?;
        self.check(device, 0)
    }

    fn release_stream(&mut self, id: u64) -> Result<(), MluExecutionError> {
        self.streams
            .remove(&id)
            .ok_or(MluExecutionError::ClosedResource)?;
        self.drop_log
            .lock()
            .map_err(|_| MluExecutionError::Poisoned)?
            .push("stream");
        Ok(())
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), MluExecutionError> {
        let device = self
            .allocations
            .get(&id)
            .ok_or(MluExecutionError::ClosedResource)?
            .0;
        self.check(device, bytes.len())?;
        let allocation = self
            .allocations
            .get_mut(&id)
            .ok_or(MluExecutionError::ClosedResource)?;
        allocation.1[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), MluExecutionError> {
        let device = self
            .allocations
            .get(&id)
            .ok_or(MluExecutionError::ClosedResource)?
            .0;
        self.check(device, bytes.len())?;
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(MluExecutionError::ClosedResource)?;
        bytes.copy_from_slice(&allocation.1[offset..offset + bytes.len()]);
        Ok(())
    }

    fn copy_device_to_device(
        &mut self,
        destination: u64,
        destination_offset: usize,
        source: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), MluExecutionError> {
        let source_allocation = self
            .allocations
            .get(&source)
            .ok_or(MluExecutionError::ClosedResource)?;
        let device = source_allocation.0;
        let staging = source_allocation.1[source_offset..source_offset + bytes].to_vec();
        self.check(device, bytes)?;
        let destination = self
            .allocations
            .get_mut(&destination)
            .ok_or(MluExecutionError::ClosedResource)?;
        destination.1[destination_offset..destination_offset + bytes].copy_from_slice(&staging);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        stream: u64,
        left: u64,
        right: u64,
        output: u64,
        dimensions: &[i32],
        element_type: MluElementType,
    ) -> Result<(), MluExecutionError> {
        let bytes = required_tensor_bytes(dimensions, element_type)?;
        let device = *self
            .streams
            .get(&stream)
            .ok_or(MluExecutionError::ClosedResource)?;
        self.check(device, bytes)?;
        let left = self
            .allocations
            .get(&left)
            .ok_or(MluExecutionError::ClosedResource)?
            .1[..bytes]
            .to_vec();
        let right = self
            .allocations
            .get(&right)
            .ok_or(MluExecutionError::ClosedResource)?
            .1[..bytes]
            .to_vec();
        let output = self
            .allocations
            .get_mut(&output)
            .ok_or(MluExecutionError::ClosedResource)?;
        match element_type {
            MluElementType::F32 => add_f32_bytes(&left, &right, &mut output.1[..bytes])?,
            MluElementType::F16 => add_f16_bytes(&left, &right, &mut output.1[..bytes])?,
        }
        Ok(())
    }
}

#[cfg(test)]
fn add_f32_bytes(left: &[u8], right: &[u8], output: &mut [u8]) -> Result<(), MluExecutionError> {
    for ((left, right), output) in left
        .chunks_exact(4)
        .zip(right.chunks_exact(4))
        .zip(output.chunks_exact_mut(4))
    {
        let left: [u8; 4] = left
            .try_into()
            .map_err(|_| MluExecutionError::InvalidArgument {
                reason: "invalid f32 input",
            })?;
        let right: [u8; 4] = right
            .try_into()
            .map_err(|_| MluExecutionError::InvalidArgument {
                reason: "invalid f32 input",
            })?;
        output
            .copy_from_slice(&(f32::from_ne_bytes(left) + f32::from_ne_bytes(right)).to_ne_bytes());
    }
    Ok(())
}

#[cfg(test)]
fn add_f16_bytes(left: &[u8], right: &[u8], output: &mut [u8]) -> Result<(), MluExecutionError> {
    for ((left, right), output) in left
        .chunks_exact(2)
        .zip(right.chunks_exact(2))
        .zip(output.chunks_exact_mut(2))
    {
        let left: [u8; 2] = left
            .try_into()
            .map_err(|_| MluExecutionError::InvalidArgument {
                reason: "invalid f16 input",
            })?;
        let right: [u8; 2] = right
            .try_into()
            .map_err(|_| MluExecutionError::InvalidArgument {
                reason: "invalid f16 input",
            })?;
        let value = f16_to_f32(u16::from_ne_bytes(left)) + f16_to_f32(u16::from_ne_bytes(right));
        output.copy_from_slice(&f32_to_f16(value).to_ne_bytes());
    }
    Ok(())
}

#[cfg(test)]
fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = u32::from(value & 0x03ff);
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction;
            let mut exponent = 113_u32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::LibraryVersion;
    use std::thread;

    fn fake_runtime(
        drop_log: Arc<Mutex<Vec<&'static str>>>,
    ) -> Result<MluExecutionRuntime, MluExecutionError> {
        MluExecutionRuntime::from_state(
            MluAbiProbe {
                target: "x86_64-unknown-linux-gnu".to_owned(),
                abi_floor: "Neuware 1.20".to_owned(),
                cnrt_version: LibraryVersion {
                    major: 6,
                    minor: 10,
                    patch: 1,
                },
                cnnl_version: LibraryVersion {
                    major: 1,
                    minor: 20,
                    patch: 0,
                },
                symbol_count: 20,
            },
            2,
            RuntimeState::Fake(FakeCore {
                device_count: 2,
                allocations: BTreeMap::new(),
                streams: BTreeMap::new(),
                failure: None,
                cancel_after_next_call: None,
                selected_devices: Vec::new(),
                drop_log,
            }),
            Arc::new(()),
        )
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn owned_runtime_and_opaque_resources_are_send_sync() {
        assert_send_sync::<MluExecutionRuntime>();
        assert_send_sync::<MluExecutionAllocation>();
        assert_send_sync::<MluExecutionStream>();
        assert_send_sync::<MluExecutionEvent>();
    }

    #[test]
    fn versions_device_count_offset_copies_and_events_are_exact() -> Result<(), MluExecutionError> {
        let runtime = fake_runtime(Arc::new(Mutex::new(Vec::new())))?;
        assert_eq!(runtime.device_count(), 2);
        assert_eq!(runtime.probe().cnnl_version.minor, 20);
        let cancellation = CancellationToken::default();
        let source = runtime.allocate(1, 12, &cancellation)?;
        let destination = runtime.allocate(1, 12, &cancellation)?;
        runtime.copy_from_host(&source, 2, &[1, 2, 3, 4], &cancellation)?;
        runtime.copy_device_to_device(&destination, 5, &source, 2, 4, &cancellation)?;
        let mut bytes = [0_u8; 4];
        runtime.copy_to_host(&destination, 5, &mut bytes, &cancellation)?;
        assert_eq!(bytes, [1, 2, 3, 4]);
        let stream = runtime.create_stream(1, &cancellation)?;
        let event = runtime.record_event(&stream, &cancellation)?;
        runtime.wait_event(&event, &cancellation)?;
        assert_eq!(event.device(), 1);
        assert!(event.is_synchronized());
        Ok(())
    }

    #[test]
    fn wait_event_marks_post_sync_cancellation_as_synchronized() -> Result<(), MluExecutionError> {
        let runtime = fake_runtime(Arc::new(Mutex::new(Vec::new())))?;
        let cancellation = CancellationToken::default();
        let stream = runtime.create_stream(0, &cancellation)?;
        let event = runtime.record_event(&stream, &cancellation)?;
        event.synchronized.store(false, Ordering::Release);

        let cancelled_after_sync = CancellationToken::default();
        runtime.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(MluExecutionError::InvalidArgument {
                    reason: "expected fake",
                });
            };
            core.cancel_after_next_call = Some(cancelled_after_sync.clone());
            Ok(())
        })?;
        assert_eq!(
            runtime.wait_event(&event, &cancelled_after_sync),
            Err(MluExecutionError::Cancelled)
        );
        assert!(event.is_synchronized());
        Ok(())
    }

    #[test]
    fn unreviewed_vendor_status_is_not_promoted_to_oom_or_device_loss() {
        assert_eq!(
            map_load_error(
                MluLoadError::CallFailed {
                    operation: "cnrtMalloc",
                    status: 42,
                },
                4096,
                0,
            ),
            MluExecutionError::VendorCallFailed {
                operation: "cnrtMalloc",
                status: 42,
            }
        );
    }

    #[test]
    fn reviewed_native_status_values_map_without_cross_library_aliasing() {
        assert_eq!(
            map_load_error(
                MluLoadError::CallFailed {
                    operation: "cnrtMalloc",
                    status: CnrtStatus::NO_MEMORY.0,
                },
                4096,
                1,
            ),
            MluExecutionError::OutOfMemory { requested: 4096 }
        );
        assert_eq!(
            map_load_error(
                MluLoadError::CallFailed {
                    operation: "cnnlCreateTensorDescriptor",
                    status: CnnlStatus::ALLOCATION_FAILED.0,
                },
                64,
                1,
            ),
            MluExecutionError::OutOfMemory { requested: 64 }
        );
        assert_eq!(
            map_load_error(
                MluLoadError::CallFailed {
                    operation: "cnrtSetDevice",
                    status: CnrtStatus::NO_DEVICE.0,
                },
                0,
                1,
            ),
            MluExecutionError::DeviceLost {
                device: 1,
                operation: "cnrtSetDevice",
            }
        );
        assert_eq!(
            map_load_error(
                MluLoadError::CallFailed {
                    operation: "cnrtMalloc",
                    status: CnnlStatus::ALLOCATION_FAILED.0,
                },
                64,
                1,
            ),
            MluExecutionError::VendorCallFailed {
                operation: "cnrtMalloc",
                status: CnnlStatus::ALLOCATION_FAILED.0,
            }
        );
    }

    #[test]
    fn f16_and_f32_add_use_the_reviewed_typed_surface() -> Result<(), MluExecutionError> {
        let runtime = fake_runtime(Arc::new(Mutex::new(Vec::new())))?;
        let cancellation = CancellationToken::default();
        let stream = runtime.create_stream(0, &cancellation)?;
        let left = runtime.allocate(0, 8, &cancellation)?;
        let right = runtime.allocate(0, 8, &cancellation)?;
        let output = runtime.allocate(0, 8, &cancellation)?;
        let left_f32 = [1.0_f32, 2.0_f32]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let right_f32 = [10.0_f32, 20.0_f32]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        runtime.copy_from_host(&left, 0, &left_f32, &cancellation)?;
        runtime.copy_from_host(&right, 0, &right_f32, &cancellation)?;
        runtime.add(
            &stream,
            MluElementType::F32,
            &[2],
            &left,
            &right,
            &output,
            &cancellation,
        )?;
        let mut bytes = [0_u8; 8];
        runtime.copy_to_host(&output, 0, &mut bytes, &cancellation)?;
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let chunk: [u8; 4] =
                    chunk
                        .try_into()
                        .map_err(|_| MluExecutionError::InvalidArgument {
                            reason: "invalid output",
                        })?;
                Ok(f32::from_ne_bytes(chunk))
            })
            .collect::<Result<Vec<_>, MluExecutionError>>()?;
        assert_eq!(values, [11.0, 22.0]);

        let left_f16 = [f32_to_f16(1.5), f32_to_f16(2.0)]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        let right_f16 = [f32_to_f16(0.5), f32_to_f16(3.0)]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        runtime.copy_from_host(&left, 0, &left_f16, &cancellation)?;
        runtime.copy_from_host(&right, 0, &right_f16, &cancellation)?;
        runtime.add(
            &stream,
            MluElementType::F16,
            &[2],
            &left,
            &right,
            &output,
            &cancellation,
        )?;
        let mut half_bytes = [0_u8; 4];
        runtime.copy_to_host(&output, 0, &mut half_bytes, &cancellation)?;
        let values = half_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let chunk: [u8; 2] =
                    chunk
                        .try_into()
                        .map_err(|_| MluExecutionError::InvalidArgument {
                            reason: "invalid output",
                        })?;
                Ok(f16_to_f32(u16::from_ne_bytes(chunk)))
            })
            .collect::<Result<Vec<_>, MluExecutionError>>()?;
        assert_eq!(values, [2.0, 5.0]);
        Ok(())
    }

    #[test]
    fn bounds_foreign_resources_cancellation_and_failures_are_typed()
    -> Result<(), MluExecutionError> {
        let runtime = fake_runtime(Arc::new(Mutex::new(Vec::new())))?;
        let other = fake_runtime(Arc::new(Mutex::new(Vec::new())))?;
        let cancellation = CancellationToken::default();
        let allocation = runtime.allocate(0, 4, &cancellation)?;
        assert!(matches!(
            runtime.copy_from_host(&allocation, 3, &[1, 2], &cancellation),
            Err(MluExecutionError::ResourceBounds { .. })
        ));
        assert!(matches!(
            other.copy_from_host(&allocation, 0, &[1], &cancellation),
            Err(MluExecutionError::ForeignResource)
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert_eq!(
            runtime.allocate(0, 4, &cancelled).err(),
            Some(MluExecutionError::Cancelled)
        );

        let cancelled_after_call = CancellationToken::default();
        runtime.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(MluExecutionError::InvalidArgument {
                    reason: "expected fake",
                });
            };
            core.cancel_after_next_call = Some(cancelled_after_call.clone());
            Ok(())
        })?;
        assert_eq!(
            runtime.allocate(0, 4, &cancelled_after_call).err(),
            Some(MluExecutionError::Cancelled)
        );

        runtime.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(MluExecutionError::InvalidArgument {
                    reason: "expected fake",
                });
            };
            core.failure = Some(FakeFailure::OutOfMemory);
            Ok(())
        })?;
        assert_eq!(
            runtime.allocate(0, 9, &cancellation).err(),
            Some(MluExecutionError::OutOfMemory { requested: 9 })
        );
        runtime.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(MluExecutionError::InvalidArgument {
                    reason: "expected fake",
                });
            };
            core.failure = Some(FakeFailure::DeviceLost);
            Ok(())
        })?;
        assert!(matches!(
            runtime.create_stream(0, &cancellation),
            Err(MluExecutionError::DeviceLost { .. })
        ));
        Ok(())
    }

    #[test]
    fn concurrent_calls_are_serialized_and_resources_drop_once() -> Result<(), MluExecutionError> {
        let drop_log = Arc::new(Mutex::new(Vec::new()));
        let runtime = fake_runtime(drop_log.clone())?;
        let cancellation = CancellationToken::default();
        let allocation = runtime.allocate(0, 4, &cancellation)?;
        let stream = runtime.create_stream(0, &cancellation)?;
        let mut threads = Vec::new();
        for value in 0_u8..8 {
            let runtime = runtime.clone();
            let allocation = allocation.clone();
            threads.push(thread::spawn(move || {
                runtime.copy_from_host(&allocation, 0, &[value], &CancellationToken::default())
            }));
        }
        for thread in threads {
            thread
                .join()
                .map_err(|_| MluExecutionError::InvalidArgument {
                    reason: "thread panicked",
                })??;
        }
        runtime.with_state(|state| {
            let RuntimeState::Fake(core) = state else {
                return Err(MluExecutionError::InvalidArgument {
                    reason: "expected fake",
                });
            };
            assert!(core.selected_devices.len() >= 10);
            assert!(core.selected_devices.iter().all(|device| *device == 0));
            Ok(())
        })?;
        drop(allocation);
        drop(stream);
        let log = drop_log.lock().map_err(|_| MluExecutionError::Poisoned)?;
        assert_eq!(log.as_slice(), ["allocation", "stream"]);
        Ok(())
    }
}
