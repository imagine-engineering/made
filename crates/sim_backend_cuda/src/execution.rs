use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

#[cfg(any(test, feature = "test-support"))]
use std::collections::BTreeMap;

use sim_types::CancellationToken;
use thiserror::Error;

use crate::abi::{
    CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_DEVICE_UNAVAILABLE, CUDA_ERROR_INVALID_CONTEXT,
    CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_OUT_OF_MEMORY,
};
use crate::loader::{
    CudaAbiProbe, CudaLoadError, NativeCudaDeviceFacts, NativeCudaElementType, OwnedCudaCore,
    RegistryCertifiedCudaImages,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaElementType {
    F16,
    F32,
}

impl CudaElementType {
    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    const fn native(self) -> NativeCudaElementType {
        match self {
            Self::F16 => NativeCudaElementType::F16,
            Self::F32 => NativeCudaElementType::F32,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceProperties {
    device_ordinal: usize,
    name: String,
    driver_version: i32,
    nvrtc_version: (i32, i32),
    cublaslt_version: usize,
    cudnn_version: usize,
    total_memory_bytes: usize,
    maximum_allocation_bytes: usize,
}

impl CudaDeviceProperties {
    fn from_native(
        facts: &NativeCudaDeviceFacts,
        probe: &CudaAbiProbe,
    ) -> Result<Self, CudaExecutionError> {
        Self::checked(
            facts.device_ordinal,
            facts.name.clone(),
            probe.versions.cuda_driver,
            (probe.versions.nvrtc_major, probe.versions.nvrtc_minor),
            probe.versions.cublaslt,
            probe.versions.cudnn,
            usize::try_from(facts.total_memory_bytes).map_err(|_| {
                CudaExecutionError::InvalidCertifiedInputs {
                    reason: "total memory does not fit the host address space",
                }
            })?,
            usize::try_from(facts.maximum_allocation_bytes).map_err(|_| {
                CudaExecutionError::InvalidCertifiedInputs {
                    reason: "maximum allocation does not fit the host address space",
                }
            })?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked(
        device_ordinal: usize,
        name: String,
        driver_version: i32,
        nvrtc_version: (i32, i32),
        cublaslt_version: usize,
        cudnn_version: usize,
        total_memory_bytes: usize,
        maximum_allocation_bytes: usize,
    ) -> Result<Self, CudaExecutionError> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(CudaExecutionError::InvalidCertifiedInputs {
                reason: "device name must contain 1..=256 non-NUL bytes",
            });
        }
        if driver_version < 12_020
            || nvrtc_version.0 != 12
            || nvrtc_version.1 < 2
            || cublaslt_version < 120_205
            || cudnn_version < 90_000
            || cudnn_version / 10_000 != 9
        {
            return Err(CudaExecutionError::InvalidCertifiedInputs {
                reason: "runtime versions are below the reviewed CUDA ABI floor",
            });
        }
        if total_memory_bytes == 0
            || maximum_allocation_bytes == 0
            || maximum_allocation_bytes > total_memory_bytes
        {
            return Err(CudaExecutionError::InvalidCertifiedInputs {
                reason: "maximum allocation must be nonzero and not exceed total memory",
            });
        }
        Ok(Self {
            device_ordinal,
            name,
            driver_version,
            nvrtc_version,
            cublaslt_version,
            cudnn_version,
            total_memory_bytes,
            maximum_allocation_bytes,
        })
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn driver_version(&self) -> i32 {
        self.driver_version
    }

    pub const fn nvrtc_version(&self) -> (i32, i32) {
        self.nvrtc_version
    }

    pub const fn cublaslt_version(&self) -> usize {
        self.cublaslt_version
    }

    pub const fn cudnn_version(&self) -> usize {
        self.cudnn_version
    }

    pub const fn total_memory_bytes(&self) -> usize {
        self.total_memory_bytes
    }

    pub const fn maximum_allocation_bytes(&self) -> usize {
        self.maximum_allocation_bytes
    }
}

#[derive(Debug, Error)]
pub enum CudaExecutionError {
    #[error(transparent)]
    Load(#[from] CudaLoadError),
    #[error("certified CUDA execution inputs are invalid: {reason}")]
    InvalidCertifiedInputs { reason: &'static str },
    #[error("CUDA execution session lock is poisoned")]
    Poisoned,
    #[error("CUDA execution was cancelled")]
    Cancelled,
    #[error("CUDA allocation of {requested} bytes exceeds the device limit {limit}")]
    OutOfMemory { requested: usize, limit: usize },
    #[error("CUDA resource belongs to another certified session")]
    ForeignResource,
    #[error("CUDA resource is closed")]
    ClosedResource,
    #[error("CUDA resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("CUDA tensor dimensions are invalid or overflow the reviewed ABI")]
    InvalidDimensions,
    #[error("CUDA device {device} was lost during {operation}")]
    DeviceLost {
        device: usize,
        operation: &'static str,
    },
    #[error("CUDA resource identifier space is exhausted")]
    IdentifierOverflow,
}

struct Session {
    properties: CudaDeviceProperties,
    next_resource_id: AtomicU64,
    state: Mutex<RuntimeState>,
}

enum RuntimeState {
    Native(OwnedCudaCore),
    #[cfg(any(test, feature = "test-support"))]
    Fake(FakeCore),
}

impl RuntimeState {
    fn allocate(
        &mut self,
        id: u64,
        dimensions: &[i64],
        element_type: CudaElementType,
    ) -> Result<usize, CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.allocate(id, dimensions, element_type.native())
                    .map_err(|error| map_load_error(error, 0, device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.allocate(id, dimensions, element_type),
        }
    }

    fn release(&mut self, id: u64) -> Result<(), CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.release_allocation(id)
                    .map_err(|error| map_load_error(error, 0, device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.release(id),
        }
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.copy_from_host(id, offset, bytes)
                    .map_err(|error| map_load_error(error, bytes.len(), device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.copy_from_host(id, offset, bytes),
        }
    }

    fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.copy_to_host(id, offset, bytes)
                    .map_err(|error| map_load_error(error, bytes.len(), device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.copy_to_host(id, offset, bytes),
        }
    }

    fn add(&mut self, left: u64, right: u64, output: u64) -> Result<(), CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.add(left, right, output)
                    .map_err(|error| map_load_error(error, 0, device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(core) => core.add(left, right, output),
        }
    }

    fn synchronize(&mut self) -> Result<(), CudaExecutionError> {
        match self {
            Self::Native(core) => {
                let (device, limit) = native_error_context(core);
                core.synchronize()
                    .map_err(|error| map_load_error(error, 0, device, limit))
            }
            #[cfg(any(test, feature = "test-support"))]
            Self::Fake(_) => Ok(()),
        }
    }
}

#[derive(Clone)]
pub struct CudaExecutionSession {
    session: Arc<Session>,
}

impl CudaExecutionSession {
    pub fn from_registry_certified_images(
        images: RegistryCertifiedCudaImages,
        device_ordinal: usize,
    ) -> Result<Self, CudaExecutionError> {
        let core = OwnedCudaCore::load_certified(images, device_ordinal)?;
        let properties = CudaDeviceProperties::from_native(core.device_facts(), core.probe())?;
        Self::from_state(properties, RuntimeState::Native(core))
    }

    #[cfg(feature = "test-support")]
    #[allow(clippy::too_many_arguments)]
    pub fn for_test_harness(
        device_ordinal: usize,
        name: impl Into<String>,
        driver_version: i32,
        nvrtc_version: (i32, i32),
        cublaslt_version: usize,
        cudnn_version: usize,
        total_memory_bytes: usize,
        maximum_allocation_bytes: usize,
    ) -> Result<Self, CudaExecutionError> {
        let properties = CudaDeviceProperties::checked(
            device_ordinal,
            name.into(),
            driver_version,
            nvrtc_version,
            cublaslt_version,
            cudnn_version,
            total_memory_bytes,
            maximum_allocation_bytes,
        )?;
        Self::from_state(
            properties,
            RuntimeState::Fake(FakeCore {
                allocations: BTreeMap::new(),
            }),
        )
    }

    fn from_state(
        properties: CudaDeviceProperties,
        state: RuntimeState,
    ) -> Result<Self, CudaExecutionError> {
        Ok(Self {
            session: Arc::new(Session {
                properties,
                next_resource_id: AtomicU64::new(1),
                state: Mutex::new(state),
            }),
        })
    }

    pub fn properties(&self) -> &CudaDeviceProperties {
        &self.session.properties
    }

    pub fn allocate(
        &self,
        dimensions: &[i64],
        element_type: CudaElementType,
        cancellation: &CancellationToken,
    ) -> Result<CudaAllocation, CudaExecutionError> {
        check_cancellation(cancellation)?;
        let bytes = required_tensor_bytes(dimensions, element_type)?;
        if bytes > self.session.properties.maximum_allocation_bytes {
            return Err(CudaExecutionError::OutOfMemory {
                requested: bytes,
                limit: self.session.properties.maximum_allocation_bytes,
            });
        }
        let id = self.next_identifier(&self.session.next_resource_id)?;
        let native_bytes = self.with_state(|state| state.allocate(id, dimensions, element_type))?;
        if native_bytes != bytes {
            self.with_state(|state| state.release(id))?;
            return Err(CudaExecutionError::InvalidCertifiedInputs {
                reason: "native allocation layout differs from the safe tensor layout",
            });
        }
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release(id))?;
            return Err(error);
        }
        Ok(CudaAllocation {
            lease: Arc::new(AllocationLease {
                session: self.session.clone(),
                id,
                bytes,
                dimensions: dimensions.to_vec(),
                element_type,
                closed: AtomicBool::new(false),
            }),
        })
    }

    pub fn copy_from_host(
        &self,
        destination: &CudaAllocation,
        destination_offset: usize,
        source: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), CudaExecutionError> {
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
        source: &CudaAllocation,
        source_offset: usize,
        destination: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), CudaExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(source)?;
        validate_range(source.byte_length(), source_offset, destination.len())?;
        self.with_state(|state| state.copy_to_host(source.lease.id, source_offset, destination))?;
        check_cancellation(cancellation)
    }

    pub fn add(
        &self,
        left: &CudaAllocation,
        right: &CudaAllocation,
        output: &CudaAllocation,
        cancellation: &CancellationToken,
    ) -> Result<CudaEvent, CudaExecutionError> {
        check_cancellation(cancellation)?;
        for allocation in [left, right, output] {
            self.validate_allocation(allocation)?;
        }
        if left.dimensions() != right.dimensions()
            || left.dimensions() != output.dimensions()
            || left.element_type() != right.element_type()
            || left.element_type() != output.element_type()
        {
            return Err(CudaExecutionError::InvalidDimensions);
        }
        self.with_state(|state| state.add(left.lease.id, right.lease.id, output.lease.id))?;
        check_cancellation(cancellation)?;
        Ok(CudaEvent {
            lease: Arc::new(EventLease {
                session: self.session.clone(),
                synchronized: AtomicBool::new(false),
            }),
        })
    }

    pub fn wait_event(
        &self,
        event: &CudaEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), CudaExecutionError> {
        check_cancellation(cancellation)?;
        if !Arc::ptr_eq(&self.session, &event.lease.session) {
            return Err(CudaExecutionError::ForeignResource);
        }
        if !event.lease.synchronized.load(Ordering::Acquire) {
            self.with_state(RuntimeState::synchronize)?;
            event.lease.synchronized.store(true, Ordering::Release);
        }
        check_cancellation(cancellation)
    }

    pub fn synchronize(&self, cancellation: &CancellationToken) -> Result<(), CudaExecutionError> {
        check_cancellation(cancellation)?;
        self.with_state(RuntimeState::synchronize)?;
        check_cancellation(cancellation)
    }

    fn validate_allocation(&self, allocation: &CudaAllocation) -> Result<(), CudaExecutionError> {
        if !Arc::ptr_eq(&self.session, &allocation.lease.session) {
            return Err(CudaExecutionError::ForeignResource);
        }
        if allocation.lease.closed.load(Ordering::Acquire) {
            return Err(CudaExecutionError::ClosedResource);
        }
        Ok(())
    }

    fn next_identifier(&self, counter: &AtomicU64) -> Result<u64, CudaExecutionError> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| CudaExecutionError::IdentifierOverflow)
    }

    fn with_state<T>(
        &self,
        operation: impl FnOnce(&mut RuntimeState) -> Result<T, CudaExecutionError>,
    ) -> Result<T, CudaExecutionError> {
        let mut state = self
            .session
            .state
            .lock()
            .map_err(|_| CudaExecutionError::Poisoned)?;
        operation(&mut state)
    }
}

#[derive(Clone)]
pub struct CudaAllocation {
    lease: Arc<AllocationLease>,
}

impl CudaAllocation {
    pub fn byte_length(&self) -> usize {
        self.lease.bytes
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.lease.dimensions
    }

    pub fn element_type(&self) -> CudaElementType {
        self.lease.element_type
    }
}

struct AllocationLease {
    session: Arc<Session>,
    id: u64,
    bytes: usize,
    dimensions: Vec<i64>,
    element_type: CudaElementType,
    closed: AtomicBool,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        match self.session.state.lock() {
            Ok(mut state) => {
                if let Err(error) = state.release(self.id) {
                    eprintln!("sim_backend_cuda: allocation release failed: {error}");
                }
            }
            Err(_) => eprintln!("sim_backend_cuda: allocation release lock is poisoned"),
        }
    }
}

#[derive(Clone)]
pub struct CudaEvent {
    lease: Arc<EventLease>,
}

impl CudaEvent {
    pub fn is_synchronized(&self) -> bool {
        self.lease.synchronized.load(Ordering::Acquire)
    }
}

struct EventLease {
    session: Arc<Session>,
    synchronized: AtomicBool,
}

fn required_tensor_bytes(
    dimensions: &[i64],
    element_type: CudaElementType,
) -> Result<usize, CudaExecutionError> {
    if dimensions.is_empty() || dimensions.len() > 12 || dimensions.iter().any(|value| *value <= 0)
    {
        return Err(CudaExecutionError::InvalidDimensions);
    }
    dimensions
        .iter()
        .try_fold(element_type.byte_width(), |bytes, dimension| {
            let dimension =
                usize::try_from(*dimension).map_err(|_| CudaExecutionError::InvalidDimensions)?;
            bytes
                .checked_mul(dimension)
                .ok_or(CudaExecutionError::InvalidDimensions)
        })
}

fn validate_range(
    available: usize,
    offset: usize,
    length: usize,
) -> Result<(), CudaExecutionError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > available) {
        Err(CudaExecutionError::ResourceBounds {
            offset,
            length,
            available,
        })
    } else {
        Ok(())
    }
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), CudaExecutionError> {
    cancellation
        .check()
        .map_err(|_| CudaExecutionError::Cancelled)
}

fn native_error_context(core: &OwnedCudaCore) -> (usize, usize) {
    (
        core.device_facts().device_ordinal,
        usize::try_from(core.device_facts().maximum_allocation_bytes).unwrap_or(usize::MAX),
    )
}

fn map_load_error(
    error: CudaLoadError,
    requested: usize,
    device: usize,
    limit: usize,
) -> CudaExecutionError {
    match error {
        CudaLoadError::CallFailed {
            status: CUDA_ERROR_OUT_OF_MEMORY,
            ..
        } => CudaExecutionError::OutOfMemory { requested, limit },
        CudaLoadError::CallFailed {
            operation,
            status:
                CUDA_ERROR_DEVICE_UNAVAILABLE
                | CUDA_ERROR_INVALID_CONTEXT
                | CUDA_ERROR_CONTEXT_IS_DESTROYED
                | CUDA_ERROR_LAUNCH_FAILED,
        } => CudaExecutionError::DeviceLost { device, operation },
        error => CudaExecutionError::Load(error),
    }
}

#[cfg(any(test, feature = "test-support"))]
struct FakeAllocation {
    dimensions: Vec<i64>,
    element_type: CudaElementType,
    bytes: Vec<u8>,
}

#[cfg(any(test, feature = "test-support"))]
struct FakeCore {
    allocations: BTreeMap<u64, FakeAllocation>,
}

#[cfg(any(test, feature = "test-support"))]
impl FakeCore {
    fn allocate(
        &mut self,
        id: u64,
        dimensions: &[i64],
        element_type: CudaElementType,
    ) -> Result<usize, CudaExecutionError> {
        let bytes = required_tensor_bytes(dimensions, element_type)?;
        if self
            .allocations
            .insert(
                id,
                FakeAllocation {
                    dimensions: dimensions.to_vec(),
                    element_type,
                    bytes: vec![0; bytes],
                },
            )
            .is_some()
        {
            return Err(CudaExecutionError::InvalidCertifiedInputs {
                reason: "duplicate fake allocation identifier",
            });
        }
        Ok(bytes)
    }

    fn release(&mut self, id: u64) -> Result<(), CudaExecutionError> {
        self.allocations
            .remove(&id)
            .map(|_| ())
            .ok_or(CudaExecutionError::ClosedResource)
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CudaExecutionError> {
        let allocation = self
            .allocations
            .get_mut(&id)
            .ok_or(CudaExecutionError::ClosedResource)?;
        validate_range(allocation.bytes.len(), offset, bytes.len())?;
        allocation.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn copy_to_host(
        &self,
        id: u64,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaExecutionError> {
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(CudaExecutionError::ClosedResource)?;
        validate_range(allocation.bytes.len(), offset, bytes.len())?;
        bytes.copy_from_slice(&allocation.bytes[offset..offset + bytes.len()]);
        Ok(())
    }

    fn add(&mut self, left: u64, right: u64, output: u64) -> Result<(), CudaExecutionError> {
        let left = self
            .allocations
            .get(&left)
            .ok_or(CudaExecutionError::ClosedResource)?;
        let right = self
            .allocations
            .get(&right)
            .ok_or(CudaExecutionError::ClosedResource)?;
        if left.dimensions != right.dimensions || left.element_type != right.element_type {
            return Err(CudaExecutionError::InvalidDimensions);
        }
        let dimensions = left.dimensions.clone();
        let element_type = left.element_type;
        let left = left.bytes.clone();
        let right = right.bytes.clone();
        let output = self
            .allocations
            .get_mut(&output)
            .ok_or(CudaExecutionError::ClosedResource)?;
        if output.dimensions != dimensions || output.element_type != element_type {
            return Err(CudaExecutionError::InvalidDimensions);
        }
        match element_type {
            CudaElementType::F32 => add_f32(&left, &right, &mut output.bytes),
            CudaElementType::F16 => Err(CudaExecutionError::InvalidDimensions),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
fn add_f32(left: &[u8], right: &[u8], output: &mut [u8]) -> Result<(), CudaExecutionError> {
    for ((left, right), output) in left
        .chunks_exact(4)
        .zip(right.chunks_exact(4))
        .zip(output.chunks_exact_mut(4))
    {
        let left = f32::from_ne_bytes(
            left.try_into()
                .map_err(|_| CudaExecutionError::InvalidDimensions)?,
        );
        let right = f32::from_ne_bytes(
            right
                .try_into()
                .map_err(|_| CudaExecutionError::InvalidDimensions)?,
        );
        output.copy_from_slice(&(left + right).to_ne_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Result<CudaExecutionSession, CudaExecutionError> {
        let properties = CudaDeviceProperties::checked(
            0,
            "NVIDIA test CUDA".to_owned(),
            12_020,
            (12, 2),
            120_205,
            90_000,
            1 << 20,
            1 << 18,
        )?;
        CudaExecutionSession::from_state(
            properties,
            RuntimeState::Fake(FakeCore {
                allocations: BTreeMap::new(),
            }),
        )
    }

    #[test]
    fn deterministic_f32_add_and_event_completion() -> Result<(), CudaExecutionError> {
        let session = session()?;
        let cancellation = CancellationToken::default();
        let left = session.allocate(&[3], CudaElementType::F32, &cancellation)?;
        let right = session.allocate(&[3], CudaElementType::F32, &cancellation)?;
        let output = session.allocate(&[3], CudaElementType::F32, &cancellation)?;
        let left_bytes = [1.0_f32, 2.0, -3.5]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let right_bytes = [0.5_f32, -2.0, 4.5]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        session.copy_from_host(&left, 0, &left_bytes, &cancellation)?;
        session.copy_from_host(&right, 0, &right_bytes, &cancellation)?;
        let event = session.add(&left, &right, &output, &cancellation)?;
        session.wait_event(&event, &cancellation)?;
        let mut bytes = vec![0; output.byte_length()];
        session.copy_to_host(&output, 0, &mut bytes, &cancellation)?;
        let values = bytes
            .chunks_exact(4)
            .map(|value| {
                let value: [u8; 4] = value
                    .try_into()
                    .map_err(|_| CudaExecutionError::InvalidDimensions)?;
                Ok(f32::from_ne_bytes(value))
            })
            .collect::<Result<Vec<_>, CudaExecutionError>>()?;
        assert_eq!(values, vec![1.5, 0.0, 1.0]);
        assert!(event.is_synchronized());
        Ok(())
    }

    #[test]
    fn cancellation_bounds_and_foreign_resources_fail_closed() -> Result<(), CudaExecutionError> {
        let session = session()?;
        let other = self::session()?;
        let cancellation = CancellationToken::default();
        let allocation = session.allocate(&[2], CudaElementType::F16, &cancellation)?;
        assert!(matches!(
            other.copy_from_host(&allocation, 0, &[0; 4], &cancellation),
            Err(CudaExecutionError::ForeignResource)
        ));
        assert!(matches!(
            session.copy_from_host(&allocation, 3, &[0; 2], &cancellation),
            Err(CudaExecutionError::ResourceBounds { .. })
        ));
        cancellation.cancel();
        assert!(matches!(
            session.synchronize(&cancellation),
            Err(CudaExecutionError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn unsupported_f16_add_fails_closed() -> Result<(), CudaExecutionError> {
        let session = session()?;
        let cancellation = CancellationToken::default();
        let left = session.allocate(&[2], CudaElementType::F16, &cancellation)?;
        let right = session.allocate(&[2], CudaElementType::F16, &cancellation)?;
        let output = session.allocate(&[2], CudaElementType::F16, &cancellation)?;
        assert!(matches!(
            session.add(&left, &right, &output, &cancellation),
            Err(CudaExecutionError::InvalidDimensions)
        ));
        Ok(())
    }

    #[test]
    fn reviewed_oom_and_device_loss_statuses_map_to_typed_errors() {
        assert!(matches!(
            map_load_error(
                CudaLoadError::CallFailed {
                    operation: "cuMemAlloc_v2",
                    status: CUDA_ERROR_OUT_OF_MEMORY,
                },
                4096,
                3,
                8192,
            ),
            CudaExecutionError::OutOfMemory {
                requested: 4096,
                limit: 8192,
            }
        ));
        assert!(matches!(
            map_load_error(
                CudaLoadError::CallFailed {
                    operation: "cuStreamSynchronize",
                    status: CUDA_ERROR_LAUNCH_FAILED,
                },
                0,
                3,
                8192,
            ),
            CudaExecutionError::DeviceLost {
                device: 3,
                operation: "cuStreamSynchronize",
            }
        ));
    }

    #[test]
    fn safe_owner_is_send_and_sync_and_exposes_no_vendor_pointer() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaExecutionSession>();
        let source = include_str!("execution.rs");
        assert!(!source.contains(&["pub fn ", "raw"].concat()));
        assert!(!source.contains(&["pub fn ", "handle"].concat()));
        assert!(!source.contains(&["unsafe impl Send for ", "CudaExecutionSession"].concat()));
        assert!(source.contains("Mutex<RuntimeState>"));
    }
}
