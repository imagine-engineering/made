use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

#[cfg(any(test, feature = "test-support"))]
use std::collections::BTreeMap;

use sim_types::CancellationToken;
use thiserror::Error;

use crate::loader::{
    NativeXpuDeviceFacts, NativeXpuElementType, OwnedXpuCore, RegistryCertifiedXpuImages,
    XpuAbiProbe, XpuLoadError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XpuElementType {
    F16,
    F32,
}

impl XpuElementType {
    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    const fn native(self) -> NativeXpuElementType {
        match self {
            Self::F16 => NativeXpuElementType::F16,
            Self::F32 => NativeXpuElementType::F32,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XpuDeviceProperties {
    device_ordinal: usize,
    name: String,
    vendor_id: u32,
    device_id: u32,
    level_zero_api_versions: Vec<(u16, u16)>,
    onednn_version: (i32, i32, i32),
    total_memory_bytes: usize,
    maximum_allocation_bytes: usize,
}

impl XpuDeviceProperties {
    fn from_native(
        facts: &NativeXpuDeviceFacts,
        probe: &XpuAbiProbe,
    ) -> Result<Self, XpuExecutionError> {
        Self::checked(
            facts.device_ordinal,
            facts.name.clone(),
            facts.vendor_id,
            facts.device_id,
            probe
                .driver_api_versions
                .iter()
                .map(|version| (version.major(), version.minor()))
                .collect(),
            probe.onednn_version,
            usize::try_from(facts.total_memory_bytes).map_err(|_| {
                XpuExecutionError::InvalidCertifiedInputs {
                    reason: "total memory does not fit the host address space",
                }
            })?,
            usize::try_from(facts.maximum_allocation_bytes).map_err(|_| {
                XpuExecutionError::InvalidCertifiedInputs {
                    reason: "maximum allocation does not fit the host address space",
                }
            })?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked(
        device_ordinal: usize,
        name: String,
        vendor_id: u32,
        device_id: u32,
        level_zero_api_versions: Vec<(u16, u16)>,
        onednn_version: (i32, i32, i32),
        total_memory_bytes: usize,
        maximum_allocation_bytes: usize,
    ) -> Result<Self, XpuExecutionError> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(XpuExecutionError::InvalidCertifiedInputs {
                reason: "device name must contain 1..=256 non-NUL bytes",
            });
        }
        if level_zero_api_versions.is_empty()
            || level_zero_api_versions
                .iter()
                .any(|version| *version < (1, 6))
            || onednn_version.0 != 3
            || onednn_version.1 < 5
        {
            return Err(XpuExecutionError::InvalidCertifiedInputs {
                reason: "runtime versions are below the reviewed XPU ABI floor",
            });
        }
        if total_memory_bytes == 0
            || maximum_allocation_bytes == 0
            || maximum_allocation_bytes > total_memory_bytes
        {
            return Err(XpuExecutionError::InvalidCertifiedInputs {
                reason: "maximum allocation must be nonzero and not exceed total memory",
            });
        }
        Ok(Self {
            device_ordinal,
            name,
            vendor_id,
            device_id,
            level_zero_api_versions,
            onednn_version,
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

    pub const fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn level_zero_api_versions(&self) -> &[(u16, u16)] {
        &self.level_zero_api_versions
    }

    pub const fn onednn_version(&self) -> (i32, i32, i32) {
        self.onednn_version
    }

    pub const fn total_memory_bytes(&self) -> usize {
        self.total_memory_bytes
    }

    pub const fn maximum_allocation_bytes(&self) -> usize {
        self.maximum_allocation_bytes
    }
}

#[derive(Debug, Error)]
pub enum XpuExecutionError {
    #[error(transparent)]
    Load(#[from] XpuLoadError),
    #[error("certified XPU execution inputs are invalid: {reason}")]
    InvalidCertifiedInputs { reason: &'static str },
    #[error("XPU execution session lock is poisoned")]
    Poisoned,
    #[error("XPU execution was cancelled")]
    Cancelled,
    #[error("XPU allocation of {requested} bytes exceeds the device limit {limit}")]
    OutOfMemory { requested: usize, limit: usize },
    #[error("XPU resource belongs to another certified session")]
    ForeignResource,
    #[error("XPU resource is closed")]
    ClosedResource,
    #[error("XPU resource range offset {offset} length {length} exceeds {available} bytes")]
    ResourceBounds {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("XPU tensor dimensions are invalid or overflow the reviewed ABI")]
    InvalidDimensions,
    #[error("XPU device {device} was lost during {operation}")]
    DeviceLost {
        device: usize,
        operation: &'static str,
    },
    #[error("XPU resource identifier space is exhausted")]
    IdentifierOverflow,
}

struct Session {
    properties: XpuDeviceProperties,
    next_resource_id: AtomicU64,
    state: Mutex<RuntimeState>,
}

enum RuntimeState {
    Native(OwnedXpuCore),
    #[cfg(any(test, feature = "test-support"))]
    Fake(FakeCore),
}

impl RuntimeState {
    fn allocate(
        &mut self,
        id: u64,
        dimensions: &[i64],
        element_type: XpuElementType,
    ) -> Result<usize, XpuExecutionError> {
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

    fn release(&mut self, id: u64) -> Result<(), XpuExecutionError> {
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
    ) -> Result<(), XpuExecutionError> {
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
    ) -> Result<(), XpuExecutionError> {
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

    fn add(&mut self, left: u64, right: u64, output: u64) -> Result<(), XpuExecutionError> {
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

    fn synchronize(&mut self) -> Result<(), XpuExecutionError> {
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
pub struct XpuExecutionSession {
    session: Arc<Session>,
}

impl XpuExecutionSession {
    pub fn from_registry_certified_images(
        images: RegistryCertifiedXpuImages,
        device_ordinal: usize,
    ) -> Result<Self, XpuExecutionError> {
        let core = OwnedXpuCore::load_certified(images, device_ordinal)?;
        let properties = XpuDeviceProperties::from_native(core.device_facts(), core.probe())?;
        Self::from_state(properties, RuntimeState::Native(core))
    }

    #[cfg(feature = "test-support")]
    #[allow(clippy::too_many_arguments)]
    pub fn for_test_harness(
        device_ordinal: usize,
        name: impl Into<String>,
        vendor_id: u32,
        device_id: u32,
        level_zero_api_versions: Vec<(u16, u16)>,
        onednn_version: (i32, i32, i32),
        total_memory_bytes: usize,
        maximum_allocation_bytes: usize,
    ) -> Result<Self, XpuExecutionError> {
        let properties = XpuDeviceProperties::checked(
            device_ordinal,
            name.into(),
            vendor_id,
            device_id,
            level_zero_api_versions,
            onednn_version,
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
        properties: XpuDeviceProperties,
        state: RuntimeState,
    ) -> Result<Self, XpuExecutionError> {
        Ok(Self {
            session: Arc::new(Session {
                properties,
                next_resource_id: AtomicU64::new(1),
                state: Mutex::new(state),
            }),
        })
    }

    pub fn properties(&self) -> &XpuDeviceProperties {
        &self.session.properties
    }

    pub fn allocate(
        &self,
        dimensions: &[i64],
        element_type: XpuElementType,
        cancellation: &CancellationToken,
    ) -> Result<XpuAllocation, XpuExecutionError> {
        check_cancellation(cancellation)?;
        let bytes = required_tensor_bytes(dimensions, element_type)?;
        if bytes > self.session.properties.maximum_allocation_bytes {
            return Err(XpuExecutionError::OutOfMemory {
                requested: bytes,
                limit: self.session.properties.maximum_allocation_bytes,
            });
        }
        let id = self.next_identifier(&self.session.next_resource_id)?;
        let native_bytes = self.with_state(|state| state.allocate(id, dimensions, element_type))?;
        if native_bytes != bytes {
            self.with_state(|state| state.release(id))?;
            return Err(XpuExecutionError::InvalidCertifiedInputs {
                reason: "native allocation layout differs from the safe tensor layout",
            });
        }
        if let Err(error) = check_cancellation(cancellation) {
            self.with_state(|state| state.release(id))?;
            return Err(error);
        }
        Ok(XpuAllocation {
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
        destination: &XpuAllocation,
        destination_offset: usize,
        source: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<(), XpuExecutionError> {
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
        source: &XpuAllocation,
        source_offset: usize,
        destination: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), XpuExecutionError> {
        check_cancellation(cancellation)?;
        self.validate_allocation(source)?;
        validate_range(source.byte_length(), source_offset, destination.len())?;
        self.with_state(|state| state.copy_to_host(source.lease.id, source_offset, destination))?;
        check_cancellation(cancellation)
    }

    pub fn add(
        &self,
        left: &XpuAllocation,
        right: &XpuAllocation,
        output: &XpuAllocation,
        cancellation: &CancellationToken,
    ) -> Result<XpuEvent, XpuExecutionError> {
        check_cancellation(cancellation)?;
        for allocation in [left, right, output] {
            self.validate_allocation(allocation)?;
        }
        if left.dimensions() != right.dimensions()
            || left.dimensions() != output.dimensions()
            || left.element_type() != right.element_type()
            || left.element_type() != output.element_type()
        {
            return Err(XpuExecutionError::InvalidDimensions);
        }
        self.with_state(|state| state.add(left.lease.id, right.lease.id, output.lease.id))?;
        check_cancellation(cancellation)?;
        Ok(XpuEvent {
            lease: Arc::new(EventLease {
                session: self.session.clone(),
                synchronized: AtomicBool::new(false),
            }),
        })
    }

    pub fn wait_event(
        &self,
        event: &XpuEvent,
        cancellation: &CancellationToken,
    ) -> Result<(), XpuExecutionError> {
        check_cancellation(cancellation)?;
        if !Arc::ptr_eq(&self.session, &event.lease.session) {
            return Err(XpuExecutionError::ForeignResource);
        }
        if !event.lease.synchronized.load(Ordering::Acquire) {
            self.with_state(RuntimeState::synchronize)?;
            event.lease.synchronized.store(true, Ordering::Release);
        }
        check_cancellation(cancellation)
    }

    pub fn synchronize(&self, cancellation: &CancellationToken) -> Result<(), XpuExecutionError> {
        check_cancellation(cancellation)?;
        self.with_state(RuntimeState::synchronize)?;
        check_cancellation(cancellation)
    }

    fn validate_allocation(&self, allocation: &XpuAllocation) -> Result<(), XpuExecutionError> {
        if !Arc::ptr_eq(&self.session, &allocation.lease.session) {
            return Err(XpuExecutionError::ForeignResource);
        }
        if allocation.lease.closed.load(Ordering::Acquire) {
            return Err(XpuExecutionError::ClosedResource);
        }
        Ok(())
    }

    fn next_identifier(&self, counter: &AtomicU64) -> Result<u64, XpuExecutionError> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| XpuExecutionError::IdentifierOverflow)
    }

    fn with_state<T>(
        &self,
        operation: impl FnOnce(&mut RuntimeState) -> Result<T, XpuExecutionError>,
    ) -> Result<T, XpuExecutionError> {
        let mut state = self
            .session
            .state
            .lock()
            .map_err(|_| XpuExecutionError::Poisoned)?;
        operation(&mut state)
    }
}

#[derive(Clone)]
pub struct XpuAllocation {
    lease: Arc<AllocationLease>,
}

impl XpuAllocation {
    pub fn byte_length(&self) -> usize {
        self.lease.bytes
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.lease.dimensions
    }

    pub fn element_type(&self) -> XpuElementType {
        self.lease.element_type
    }
}

struct AllocationLease {
    session: Arc<Session>,
    id: u64,
    bytes: usize,
    dimensions: Vec<i64>,
    element_type: XpuElementType,
    closed: AtomicBool,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        match self.session.state.lock() {
            Ok(mut state) => {
                if let Err(error) = state.release(self.id) {
                    eprintln!("sim_backend_xpu: allocation release failed: {error}");
                }
            }
            Err(_) => eprintln!("sim_backend_xpu: allocation release lock is poisoned"),
        }
    }
}

#[derive(Clone)]
pub struct XpuEvent {
    lease: Arc<EventLease>,
}

impl XpuEvent {
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
    element_type: XpuElementType,
) -> Result<usize, XpuExecutionError> {
    if dimensions.is_empty() || dimensions.len() > 12 || dimensions.iter().any(|value| *value <= 0)
    {
        return Err(XpuExecutionError::InvalidDimensions);
    }
    dimensions
        .iter()
        .try_fold(element_type.byte_width(), |bytes, dimension| {
            let dimension =
                usize::try_from(*dimension).map_err(|_| XpuExecutionError::InvalidDimensions)?;
            bytes
                .checked_mul(dimension)
                .ok_or(XpuExecutionError::InvalidDimensions)
        })
}

fn validate_range(available: usize, offset: usize, length: usize) -> Result<(), XpuExecutionError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > available) {
        Err(XpuExecutionError::ResourceBounds {
            offset,
            length,
            available,
        })
    } else {
        Ok(())
    }
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), XpuExecutionError> {
    cancellation
        .check()
        .map_err(|_| XpuExecutionError::Cancelled)
}

fn native_error_context(core: &OwnedXpuCore) -> (usize, usize) {
    (
        core.device_facts().device_ordinal,
        usize::try_from(core.device_facts().maximum_allocation_bytes).unwrap_or(usize::MAX),
    )
}

fn map_load_error(
    error: XpuLoadError,
    requested: usize,
    device: usize,
    limit: usize,
) -> XpuExecutionError {
    match error {
        XpuLoadError::VendorCall {
            library: "onednn",
            status: 1,
            ..
        }
        | XpuLoadError::VendorCall {
            library: "level_zero",
            status: 0x7000_0003,
            ..
        } => XpuExecutionError::OutOfMemory { requested, limit },
        XpuLoadError::VendorCall {
            symbol,
            status: 0x7000_0001,
            ..
        } => XpuExecutionError::DeviceLost {
            device,
            operation: symbol,
        },
        error => XpuExecutionError::Load(error),
    }
}

#[cfg(any(test, feature = "test-support"))]
struct FakeAllocation {
    dimensions: Vec<i64>,
    element_type: XpuElementType,
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
        element_type: XpuElementType,
    ) -> Result<usize, XpuExecutionError> {
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
            return Err(XpuExecutionError::InvalidCertifiedInputs {
                reason: "duplicate fake allocation identifier",
            });
        }
        Ok(bytes)
    }

    fn release(&mut self, id: u64) -> Result<(), XpuExecutionError> {
        self.allocations
            .remove(&id)
            .map(|_| ())
            .ok_or(XpuExecutionError::ClosedResource)
    }

    fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), XpuExecutionError> {
        let allocation = self
            .allocations
            .get_mut(&id)
            .ok_or(XpuExecutionError::ClosedResource)?;
        validate_range(allocation.bytes.len(), offset, bytes.len())?;
        allocation.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn copy_to_host(
        &self,
        id: u64,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), XpuExecutionError> {
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(XpuExecutionError::ClosedResource)?;
        validate_range(allocation.bytes.len(), offset, bytes.len())?;
        bytes.copy_from_slice(&allocation.bytes[offset..offset + bytes.len()]);
        Ok(())
    }

    fn add(&mut self, left: u64, right: u64, output: u64) -> Result<(), XpuExecutionError> {
        let left = self
            .allocations
            .get(&left)
            .ok_or(XpuExecutionError::ClosedResource)?;
        let right = self
            .allocations
            .get(&right)
            .ok_or(XpuExecutionError::ClosedResource)?;
        if left.dimensions != right.dimensions || left.element_type != right.element_type {
            return Err(XpuExecutionError::InvalidDimensions);
        }
        let dimensions = left.dimensions.clone();
        let element_type = left.element_type;
        let left = left.bytes.clone();
        let right = right.bytes.clone();
        let output = self
            .allocations
            .get_mut(&output)
            .ok_or(XpuExecutionError::ClosedResource)?;
        if output.dimensions != dimensions || output.element_type != element_type {
            return Err(XpuExecutionError::InvalidDimensions);
        }
        match element_type {
            XpuElementType::F32 => add_f32(&left, &right, &mut output.bytes),
            XpuElementType::F16 => add_f16(&left, &right, &mut output.bytes),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
fn add_f32(left: &[u8], right: &[u8], output: &mut [u8]) -> Result<(), XpuExecutionError> {
    for ((left, right), output) in left
        .chunks_exact(4)
        .zip(right.chunks_exact(4))
        .zip(output.chunks_exact_mut(4))
    {
        let left = f32::from_ne_bytes(
            left.try_into()
                .map_err(|_| XpuExecutionError::InvalidDimensions)?,
        );
        let right = f32::from_ne_bytes(
            right
                .try_into()
                .map_err(|_| XpuExecutionError::InvalidDimensions)?,
        );
        output.copy_from_slice(&(left + right).to_ne_bytes());
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
fn add_f16(left: &[u8], right: &[u8], output: &mut [u8]) -> Result<(), XpuExecutionError> {
    for ((left, right), output) in left
        .chunks_exact(2)
        .zip(right.chunks_exact(2))
        .zip(output.chunks_exact_mut(2))
    {
        let left = u16::from_ne_bytes(
            left.try_into()
                .map_err(|_| XpuExecutionError::InvalidDimensions)?,
        );
        let right = u16::from_ne_bytes(
            right
                .try_into()
                .map_err(|_| XpuExecutionError::InvalidDimensions)?,
        );
        output.copy_from_slice(&f32_to_f16(f16_to_f32(left) + f16_to_f32(right)).to_ne_bytes());
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
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

    fn session() -> Result<XpuExecutionSession, XpuExecutionError> {
        let properties = XpuDeviceProperties::checked(
            0,
            "Intel test XPU".to_owned(),
            0x8086,
            1,
            vec![(1, 6)],
            (3, 5, 0),
            1 << 20,
            1 << 18,
        )?;
        XpuExecutionSession::from_state(
            properties,
            RuntimeState::Fake(FakeCore {
                allocations: BTreeMap::new(),
            }),
        )
    }

    #[test]
    fn deterministic_f32_add_and_event_completion() -> Result<(), XpuExecutionError> {
        let session = session()?;
        let cancellation = CancellationToken::default();
        let left = session.allocate(&[3], XpuElementType::F32, &cancellation)?;
        let right = session.allocate(&[3], XpuElementType::F32, &cancellation)?;
        let output = session.allocate(&[3], XpuElementType::F32, &cancellation)?;
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
                    .map_err(|_| XpuExecutionError::InvalidDimensions)?;
                Ok(f32::from_ne_bytes(value))
            })
            .collect::<Result<Vec<_>, XpuExecutionError>>()?;
        assert_eq!(values, vec![1.5, 0.0, 1.0]);
        assert!(event.is_synchronized());
        Ok(())
    }

    #[test]
    fn cancellation_bounds_and_foreign_resources_fail_closed() -> Result<(), XpuExecutionError> {
        let session = session()?;
        let other = self::session()?;
        let cancellation = CancellationToken::default();
        let allocation = session.allocate(&[2], XpuElementType::F16, &cancellation)?;
        assert!(matches!(
            other.copy_from_host(&allocation, 0, &[0; 4], &cancellation),
            Err(XpuExecutionError::ForeignResource)
        ));
        assert!(matches!(
            session.copy_from_host(&allocation, 3, &[0; 2], &cancellation),
            Err(XpuExecutionError::ResourceBounds { .. })
        ));
        cancellation.cancel();
        assert!(matches!(
            session.synchronize(&cancellation),
            Err(XpuExecutionError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn deterministic_f16_add_uses_the_same_safe_surface() -> Result<(), XpuExecutionError> {
        let session = session()?;
        let cancellation = CancellationToken::default();
        let left = session.allocate(&[2], XpuElementType::F16, &cancellation)?;
        let right = session.allocate(&[2], XpuElementType::F16, &cancellation)?;
        let output = session.allocate(&[2], XpuElementType::F16, &cancellation)?;
        let left_bytes = [f32_to_f16(1.5), f32_to_f16(-2.0)]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        let right_bytes = [f32_to_f16(0.5), f32_to_f16(3.0)]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        session.copy_from_host(&left, 0, &left_bytes, &cancellation)?;
        session.copy_from_host(&right, 0, &right_bytes, &cancellation)?;
        let event = session.add(&left, &right, &output, &cancellation)?;
        session.wait_event(&event, &cancellation)?;
        let mut bytes = vec![0; output.byte_length()];
        session.copy_to_host(&output, 0, &mut bytes, &cancellation)?;
        assert_eq!(
            bytes,
            [f32_to_f16(2.0), f32_to_f16(1.0)]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn reviewed_oom_and_device_loss_statuses_map_to_typed_errors() {
        assert!(matches!(
            map_load_error(
                XpuLoadError::VendorCall {
                    library: "onednn",
                    symbol: "dnnl_memory_create",
                    status: 1,
                },
                4096,
                3,
                8192,
            ),
            XpuExecutionError::OutOfMemory {
                requested: 4096,
                limit: 8192,
            }
        ));
        assert!(matches!(
            map_load_error(
                XpuLoadError::VendorCall {
                    library: "level_zero",
                    symbol: "zeCommandQueueSynchronize",
                    status: 0x7000_0001,
                },
                0,
                3,
                8192,
            ),
            XpuExecutionError::DeviceLost {
                device: 3,
                operation: "zeCommandQueueSynchronize",
            }
        ));
    }

    #[test]
    fn safe_owner_is_send_and_sync_and_exposes_no_vendor_pointer() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<XpuExecutionSession>();
        let source = include_str!("execution.rs");
        assert!(!source.contains(&["pub fn ", "raw"].concat()));
        assert!(!source.contains(&["pub fn ", "handle"].concat()));
        assert!(!source.contains(&["unsafe impl Send for ", "XpuExecutionSession"].concat()));
        assert!(source.contains("Mutex<RuntimeState>"));
    }
}
