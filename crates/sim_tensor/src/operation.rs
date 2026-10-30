use crate::{
    DType, DecodedScalar, DeviceId, Layout, NumericClass, RngStreamAddress, StreamId, Tensor,
    TensorDescriptor, TensorError,
};
use serde::{Deserialize, Serialize};
pub use sim_types::CancellationToken;
use sim_types::{
    BackendUnavailable, DeviceKind, NativeBackendBindingStatus, WorkerBackendCapabilities,
    WorkerBinaryOperationV1, WorkerDType, WorkerLayout, WorkerLinearAlgebraOperationV1,
    WorkerNativeDeviceProperties, WorkerOperationCategory, WorkerOperationSupport,
    WorkerPrimitiveOperationV2, WorkerReductionOperationV1, WorkerResizeModeV1, WorkerTensorRoleV1,
    WorkerUnaryOperationV1,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_BACKEND_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WORKSPACE_AUTHORITY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendMemorySnapshot {
    pub limit_bytes: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct BackendMemoryTracker {
    limit: u64,
    current: AtomicU64,
    peak: AtomicU64,
}

impl BackendMemoryTracker {
    pub(crate) fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            current: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        })
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        requested: u64,
    ) -> Result<BackendMemoryReservation, TensorError> {
        let mut current = self.current.load(Ordering::Acquire);
        loop {
            let next =
                current
                    .checked_add(requested)
                    .ok_or_else(|| TensorError::AllocationFailed {
                        requested,
                        reason: format!(
                            "backend limit is {} bytes with {current} bytes already reserved",
                            self.limit
                        ),
                    })?;
            if next > self.limit {
                return Err(TensorError::AllocationFailed {
                    requested,
                    reason: format!(
                        "backend limit is {} bytes with {current} bytes already reserved",
                        self.limit
                    ),
                });
            }
            match self.current.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(BackendMemoryReservation {
                        bytes: requested,
                        tracker: self.clone(),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn limit(&self) -> u64 {
        self.limit
    }

    pub(crate) fn current(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }

    pub(crate) fn snapshot(&self) -> BackendMemorySnapshot {
        BackendMemorySnapshot {
            limit_bytes: self.limit,
            current_bytes: self.current(),
            peak_bytes: self.peak.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BackendMemoryReservation {
    bytes: u64,
    tracker: Arc<BackendMemoryTracker>,
}

impl BackendMemoryReservation {
    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn tracker(&self) -> &Arc<BackendMemoryTracker> {
        &self.tracker
    }
}

impl Drop for BackendMemoryReservation {
    fn drop(&mut self) {
        let previous =
            self.tracker
                .current
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_sub(self.bytes)
                });
        assert!(previous.is_ok(), "backend memory accounting underflowed");
    }
}

#[derive(Debug)]
pub struct BackendWorkspaceAuthority {
    backend_id: u64,
    memory: Arc<BackendMemoryTracker>,
}

impl BackendWorkspaceAuthority {
    pub(crate) fn new(
        memory_limit_bytes: u64,
    ) -> Result<(u64, Arc<BackendMemoryTracker>, Self), TensorError> {
        let backend_id = NEXT_BACKEND_INSTANCE_ID
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| TensorError::IdentifierOverflow)?;
        let memory = BackendMemoryTracker::new(memory_limit_bytes);
        Ok((backend_id, memory.clone(), Self { backend_id, memory }))
    }

    pub fn authorize_workspace(
        &self,
        authorized_bytes: u64,
    ) -> Result<ScratchReservation, TensorError> {
        if authorized_bytes > self.memory.limit() {
            return Err(TensorError::WorkspaceAuthorizationExceeded {
                requested: authorized_bytes,
                authorized: self.memory.limit(),
                in_use: self.memory.current(),
            });
        }
        let authority_id = NEXT_WORKSPACE_AUTHORITY_ID
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| TensorError::IdentifierOverflow)?;
        Ok(ScratchReservation::bound(
            authorized_bytes,
            self.backend_id,
            authority_id,
        ))
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.memory.snapshot()
    }
}

pub(crate) fn check_backend_context(
    backend_id: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), TensorError> {
    context.check()?;
    check_backend_context_identity(backend_id, context)
}

pub(crate) fn check_backend_context_identity(
    backend_id: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), TensorError> {
    if context.scratch.backend_id() != backend_id || context.scratch.authority_id() == 0 {
        return Err(TensorError::WorkspaceAuthorizationMismatch {
            expected_backend: backend_id,
            expected_authority: 0,
            actual_backend: context.scratch.backend_id(),
            actual_authority: context.scratch.authority_id(),
        });
    }
    Ok(())
}

pub(crate) fn reserve_backend_workspace(
    backend_id: u64,
    memory: &Arc<BackendMemoryTracker>,
    context: &ExecutionContext<'_>,
    logical_bytes: u64,
    accounted_bytes: u64,
) -> Result<BackendWorkspaceLease, TensorError> {
    check_backend_context(backend_id, context)?;
    let authorization = context.scratch.try_acquire(logical_bytes)?;
    let memory = memory.reserve(accounted_bytes)?;
    Ok(BackendWorkspaceLease {
        logical_bytes,
        _authorization: authorization,
        _memory: memory,
    })
}

#[derive(Debug)]
pub struct BackendWorkspaceLease {
    logical_bytes: u64,
    _authorization: ScratchLease,
    _memory: BackendMemoryReservation,
}

impl BackendWorkspaceLease {
    pub const fn bytes(&self) -> u64 {
        self.logical_bytes
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "OperationContractIdWire")]
pub struct OperationContractId(String);

#[derive(Deserialize)]
struct OperationContractIdWire(String);

impl TryFrom<OperationContractIdWire> for OperationContractId {
    type Error = TensorError;

    fn try_from(value: OperationContractIdWire) -> Result<Self, Self::Error> {
        Self::from_wire_value(value.0)
    }
}

impl OperationContractId {
    pub fn new(value: impl Into<String>) -> Result<Self, TensorError> {
        Self::cataloged(value)
    }

    pub fn cataloged(value: impl Into<String>) -> Result<Self, TensorError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(TensorError::Faulted {
                reason: "operation contract identifier is empty".to_owned(),
            });
        }
        if crate::operation_contracts::compiled_resolution_by_identifier(&value).is_none() {
            return Err(TensorError::Faulted {
                reason: format!(
                    "operation contract identifier has no valid compiled resolution: {value}"
                ),
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_wire_value(value: String) -> Result<Self, TensorError> {
        Self::cataloged(value)
    }
}

#[derive(Clone, Debug)]
pub struct ScratchReservation {
    bytes: u64,
    authority: Arc<ScratchAuthorization>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScratchBindingIdentity {
    pub(crate) backend_id: u64,
    pub(crate) authority_id: u64,
}

#[derive(Debug)]
pub(crate) struct ScratchAuthorization {
    backend_id: u64,
    authority_id: u64,
    ceiling_bytes: u64,
    in_use_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

#[derive(Debug)]
pub(crate) struct ScratchLease {
    authority: Arc<ScratchAuthorization>,
    bytes: u64,
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        let previous = self.authority.in_use_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| current.checked_sub(self.bytes),
        );
        assert!(
            previous.is_ok(),
            "workspace authorization lease accounting underflowed"
        );
    }
}

impl ScratchReservation {
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn in_use_bytes(&self) -> u64 {
        self.authority.in_use_bytes.load(Ordering::Acquire)
    }

    pub fn peak_bytes(&self) -> u64 {
        self.authority.peak_bytes.load(Ordering::Acquire)
    }

    pub fn binding_identity(&self) -> ScratchBindingIdentity {
        ScratchBindingIdentity {
            backend_id: self.authority.backend_id,
            authority_id: self.authority.authority_id,
        }
    }

    pub(crate) fn bound(bytes: u64, backend_id: u64, authority_id: u64) -> Self {
        Self {
            bytes,
            authority: Arc::new(ScratchAuthorization {
                backend_id,
                authority_id,
                ceiling_bytes: bytes,
                in_use_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn backend_id(&self) -> u64 {
        self.authority.backend_id
    }

    pub(crate) fn authority_id(&self) -> u64 {
        self.authority.authority_id
    }

    pub(crate) fn try_acquire(&self, requested: u64) -> Result<ScratchLease, TensorError> {
        let authority = &self.authority;
        let mut current = authority.in_use_bytes.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(requested).ok_or(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested,
                    authorized: authority.ceiling_bytes,
                    in_use: current,
                },
            )?;
            if next > authority.ceiling_bytes {
                return Err(TensorError::WorkspaceAuthorizationExceeded {
                    requested,
                    authorized: authority.ceiling_bytes,
                    in_use: current,
                });
            }
            match authority.in_use_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    authority.peak_bytes.fetch_max(next, Ordering::AcqRel);
                    return Ok(ScratchLease {
                        authority: authority.clone(),
                        bytes: requested,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl PartialEq for ScratchReservation {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.backend_id() == other.backend_id()
            && self.authority_id() == other.authority_id()
    }
}

impl Eq for ScratchReservation {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EventFence {
    pub(crate) backend_id: u64,
    pub(crate) device: DeviceId,
    pub(crate) stream: StreamId,
    pub(crate) sequence: u64,
}

impl EventFence {
    pub const fn device(self) -> DeviceId {
        self.device
    }

    pub const fn stream(self) -> StreamId {
        self.stream
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationCategory {
    Allocation,
    Copy,
    Event,
    Scalar,
    Unary,
    Binary,
    Reduction,
    Indexing,
    Resize,
    Convolution,
    LinearAlgebra,
    CustomKernel,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveOperation {
    Allocation,
    Copy,
    Fill,
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    BinaryScalar(BinaryOperation),
    Reduction(ReductionOperation),
    Select,
    Narrow,
    Resize(ResizeMode),
    RecordEvent,
    WaitEvent,
    LinearAlgebra(LinearAlgebraOperation),
    Gather,
    Scatter,
    MaskedSelect,
    Convolution,
    CustomKernel,
}

impl PrimitiveOperation {
    pub const fn category(self) -> OperationCategory {
        match self {
            Self::Allocation => OperationCategory::Allocation,
            Self::Copy => OperationCategory::Copy,
            Self::Fill => OperationCategory::Scalar,
            Self::Unary(_) => OperationCategory::Unary,
            Self::Binary(_) | Self::BinaryScalar(_) => OperationCategory::Binary,
            Self::Reduction(_) => OperationCategory::Reduction,
            Self::Select | Self::Narrow | Self::Gather | Self::Scatter | Self::MaskedSelect => {
                OperationCategory::Indexing
            }
            Self::Resize(_) => OperationCategory::Resize,
            Self::Convolution => OperationCategory::Convolution,
            Self::LinearAlgebra(_) => OperationCategory::LinearAlgebra,
            Self::CustomKernel => OperationCategory::CustomKernel,
            Self::RecordEvent | Self::WaitEvent => OperationCategory::Event,
        }
    }

    pub const fn is_event(self) -> bool {
        matches!(self, Self::RecordEvent | Self::WaitEvent)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(into = "OperationSupportWire")]
pub struct OperationSupport {
    primitive: PrimitiveOperation,
    role: Option<TensorRole>,
    dtype: Option<DType>,
    layout: Option<Layout>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OperationSupportWire {
    primitive: PrimitiveOperation,
    role: Option<TensorRole>,
    dtype: Option<DType>,
    layout: Option<Layout>,
}

impl OperationSupport {
    const fn tensor(
        primitive: PrimitiveOperation,
        role: TensorRole,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self {
            primitive,
            role: Some(role),
            dtype: Some(dtype),
            layout: Some(layout),
        }
    }

    pub const fn allocation(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Allocation,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn copy_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Copy, TensorRole::Input, dtype, layout)
    }

    pub const fn copy_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Copy, TensorRole::Output, dtype, layout)
    }

    pub const fn fill(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Fill, TensorRole::Output, dtype, layout)
    }

    pub const fn unary_input(operation: UnaryOperation, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Unary(operation),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn unary_output(operation: UnaryOperation, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Unary(operation),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn binary_input(operation: BinaryOperation, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Binary(operation),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn binary_output(operation: BinaryOperation, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Binary(operation),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn binary_scalar_input(
        operation: BinaryOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::BinaryScalar(operation),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn binary_scalar_output(
        operation: BinaryOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::BinaryScalar(operation),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn reduction_input(
        operation: ReductionOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::Reduction(operation),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn reduction_output(
        operation: ReductionOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::Reduction(operation),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn select_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Select, TensorRole::Input, dtype, layout)
    }

    pub const fn select_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Select,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn narrow_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Narrow, TensorRole::Input, dtype, layout)
    }

    pub const fn narrow_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Narrow,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn gather_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(PrimitiveOperation::Gather, TensorRole::Input, dtype, layout)
    }

    pub const fn gather_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Gather,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn scatter_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Scatter,
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn scatter_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Scatter,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn masked_select_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::MaskedSelect,
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn masked_select_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::MaskedSelect,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn resize_input(mode: ResizeMode, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Resize(mode),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn resize_output(mode: ResizeMode, dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Resize(mode),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn linear_algebra_input(
        operation: LinearAlgebraOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::LinearAlgebra(operation),
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn linear_algebra_output(
        operation: LinearAlgebraOperation,
        dtype: DType,
        layout: Layout,
    ) -> Self {
        Self::tensor(
            PrimitiveOperation::LinearAlgebra(operation),
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn convolution_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Convolution,
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn convolution_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::Convolution,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn custom_kernel_input(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::CustomKernel,
            TensorRole::Input,
            dtype,
            layout,
        )
    }

    pub const fn custom_kernel_output(dtype: DType, layout: Layout) -> Self {
        Self::tensor(
            PrimitiveOperation::CustomKernel,
            TensorRole::Output,
            dtype,
            layout,
        )
    }

    pub const fn record_event() -> Self {
        Self {
            primitive: PrimitiveOperation::RecordEvent,
            role: None,
            dtype: None,
            layout: None,
        }
    }

    pub const fn wait_event() -> Self {
        Self {
            primitive: PrimitiveOperation::WaitEvent,
            role: None,
            dtype: None,
            layout: None,
        }
    }

    pub const fn primitive(self) -> PrimitiveOperation {
        self.primitive
    }

    pub const fn category(self) -> OperationCategory {
        self.primitive.category()
    }

    pub const fn role(self) -> Option<TensorRole> {
        self.role
    }

    pub const fn dtype(self) -> Option<DType> {
        self.dtype
    }

    pub const fn layout(self) -> Option<Layout> {
        self.layout
    }

    pub(crate) fn for_tensor(
        primitive: PrimitiveOperation,
        role: TensorRole,
        dtype: DType,
        layout: Layout,
    ) -> Result<Self, TensorError> {
        if primitive.is_event() {
            Err(TensorError::Faulted {
                reason: "event primitive capability cannot declare a role, dtype, or layout"
                    .to_owned(),
            })
        } else {
            Ok(Self::tensor(primitive, role, dtype, layout))
        }
    }

    fn for_event(primitive: PrimitiveOperation) -> Result<Self, TensorError> {
        if primitive.is_event() {
            Ok(Self {
                primitive,
                role: None,
                dtype: None,
                layout: None,
            })
        } else {
            Err(TensorError::Faulted {
                reason: "tensor primitive capability requires role, dtype, and layout".to_owned(),
            })
        }
    }
}

impl From<OperationSupport> for OperationSupportWire {
    fn from(value: OperationSupport) -> Self {
        Self {
            primitive: value.primitive,
            role: value.role,
            dtype: value.dtype,
            layout: value.layout,
        }
    }
}

impl TryFrom<OperationSupportWire> for OperationSupport {
    type Error = TensorError;

    fn try_from(value: OperationSupportWire) -> Result<Self, Self::Error> {
        match (
            value.primitive.is_event(),
            value.role,
            value.dtype,
            value.layout,
        ) {
            (true, None, None, None) => Self::for_event(value.primitive),
            (true, _, _, _) => Err(TensorError::Faulted {
                reason: "event primitive capability cannot declare a role, dtype, or layout"
                    .to_owned(),
            }),
            (false, Some(role), Some(dtype), Some(layout)) => {
                Self::for_tensor(value.primitive, role, dtype, layout)
            }
            (false, _, _, _) => Err(TensorError::Faulted {
                reason: "tensor primitive capability requires role, dtype, and layout".to_owned(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for OperationSupport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        OperationSupportWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeDevicePropertiesWire",
    into = "NativeDevicePropertiesWire"
)]
pub struct NativeDeviceProperties {
    device: DeviceId,
    name: String,
    total_memory_bytes: u64,
    allocation_limit_bytes: u64,
    major: u32,
    minor: u32,
    architecture: Option<String>,
    has_fp16: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NativeDevicePropertiesWire {
    device: DeviceId,
    name: String,
    total_memory_bytes: u64,
    major: u32,
    minor: u32,
    architecture: Option<String>,
    has_fp16: bool,
    allocation_limit_bytes: u64,
}

impl From<NativeDeviceProperties> for NativeDevicePropertiesWire {
    fn from(properties: NativeDeviceProperties) -> Self {
        Self {
            device: properties.device,
            name: properties.name,
            total_memory_bytes: properties.total_memory_bytes,
            major: properties.major,
            minor: properties.minor,
            architecture: properties.architecture,
            has_fp16: properties.has_fp16,
            allocation_limit_bytes: properties.allocation_limit_bytes,
        }
    }
}

impl TryFrom<NativeDevicePropertiesWire> for NativeDeviceProperties {
    type Error = TensorError;

    fn try_from(properties: NativeDevicePropertiesWire) -> Result<Self, Self::Error> {
        Self::new_with_allocation_limit(
            properties.device,
            properties.name,
            properties.total_memory_bytes,
            properties.allocation_limit_bytes,
            properties.major,
            properties.minor,
            properties.architecture,
            properties.has_fp16,
        )
    }
}

impl NativeDeviceProperties {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: DeviceId,
        name: impl Into<String>,
        total_memory_bytes: u64,
        major: u32,
        minor: u32,
        architecture: Option<String>,
        has_fp16: bool,
    ) -> Result<Self, TensorError> {
        Self::new_with_allocation_limit(
            device,
            name,
            total_memory_bytes,
            total_memory_bytes,
            major,
            minor,
            architecture,
            has_fp16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_allocation_limit(
        device: DeviceId,
        name: impl Into<String>,
        total_memory_bytes: u64,
        allocation_limit_bytes: u64,
        major: u32,
        minor: u32,
        architecture: Option<String>,
        has_fp16: bool,
    ) -> Result<Self, TensorError> {
        let name = name.into();
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(TensorError::Faulted {
                reason: "native device name must contain 1..=256 non-NUL bytes".to_owned(),
            });
        }
        if total_memory_bytes == 0 {
            return Err(TensorError::Faulted {
                reason: "native device total memory must be nonzero".to_owned(),
            });
        }
        if allocation_limit_bytes == 0 || allocation_limit_bytes > total_memory_bytes {
            return Err(TensorError::Faulted {
                reason:
                    "native device allocation limit must be nonzero and no larger than total memory"
                        .to_owned(),
            });
        }
        if architecture
            .as_ref()
            .is_some_and(|value| value.is_empty() || value.len() > 256 || value.contains('\0'))
        {
            return Err(TensorError::Faulted {
                reason: "native device architecture must contain 1..=256 non-NUL bytes".to_owned(),
            });
        }
        Ok(Self {
            device,
            name,
            total_memory_bytes,
            allocation_limit_bytes,
            major,
            minor,
            architecture,
            has_fp16,
        })
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    pub const fn allocation_limit_bytes(&self) -> u64 {
        self.allocation_limit_bytes
    }

    pub const fn major(&self) -> u32 {
        self.major
    }

    pub const fn minor(&self) -> u32 {
        self.minor
    }

    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    pub const fn has_fp16(&self) -> bool {
        self.has_fp16
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "BackendCapabilityMatrixWire",
    into = "BackendCapabilityMatrixWire"
)]
pub struct BackendCapabilityMatrix {
    device: DeviceId,
    supported: Vec<OperationSupport>,
    deterministic: Vec<OperationSupport>,
    properties: Option<NativeDeviceProperties>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BackendCapabilityMatrixWire {
    device: DeviceId,
    supported: Vec<OperationSupport>,
    deterministic: Vec<OperationSupport>,
    #[serde(default)]
    properties: Option<NativeDeviceProperties>,
}

impl BackendCapabilityMatrix {
    pub fn worker_readiness_requirements(device: DeviceId) -> Result<Self, TensorError> {
        let add_input =
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        let add_output =
            OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        Self::new(
            device,
            vec![
                OperationSupport::allocation(DType::F32, Layout::Contiguous),
                OperationSupport::copy_input(DType::F32, Layout::Contiguous),
                OperationSupport::copy_output(DType::F32, Layout::Contiguous),
                add_input,
                add_output,
                OperationSupport::record_event(),
                OperationSupport::wait_event(),
            ],
            vec![add_input, add_output],
        )
    }

    pub fn for_native_device(device: DeviceId) -> Result<Self, BackendUnavailable> {
        #[cfg(feature = "cpu")]
        if device == DeviceId::CPU {
            return Ok(crate::CpuBackend::capability_matrix());
        }

        let binding = native_backend_binding_status(device.kind());
        if binding.device() != device.kind() {
            return Err(BackendUnavailable::new(
                device.kind(),
                "native backend binding reported a different device kind",
            ));
        }
        match binding {
            NativeBackendBindingStatus::Unbound { reason, .. } => {
                Err(BackendUnavailable::new(device.kind(), reason))
            }
            NativeBackendBindingStatus::Bound { .. } => Err(BackendUnavailable::new(
                device.kind(),
                format!(
                    "device ordinal {} has no registered canonical capability matrix",
                    device.ordinal()
                ),
            )),
        }
    }

    pub fn new(
        device: DeviceId,
        supported: Vec<OperationSupport>,
        deterministic: Vec<OperationSupport>,
    ) -> Result<Self, TensorError> {
        Self::new_with_properties(device, supported, deterministic, None)
    }

    pub fn new_with_properties(
        device: DeviceId,
        supported: Vec<OperationSupport>,
        deterministic: Vec<OperationSupport>,
        properties: Option<NativeDeviceProperties>,
    ) -> Result<Self, TensorError> {
        let supported = canonicalize_declared_support(supported)?;
        let deterministic = canonicalize_declared_support(deterministic)?;
        if deterministic.iter().any(|entry| {
            supported
                .binary_search_by_key(&operation_support_order(*entry), |candidate| {
                    operation_support_order(*candidate)
                })
                .is_err()
        }) {
            return Err(TensorError::Faulted {
                reason: "deterministic backend capabilities must also be supported".to_owned(),
            });
        }
        if let Some(properties) = properties.as_ref()
            && properties.device() != device
        {
            return Err(TensorError::DeviceMismatch {
                expected: device,
                actual: properties.device(),
            });
        }
        Ok(Self {
            device,
            supported,
            deterministic,
            properties,
        })
    }

    pub(crate) fn all_deterministic(device: DeviceId, supported: Vec<OperationSupport>) -> Self {
        let supported = canonicalize_internal_support(supported);
        Self {
            device,
            deterministic: supported.clone(),
            supported,
            properties: None,
        }
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn supports(&self, support: OperationSupport) -> bool {
        self.supported
            .binary_search_by_key(&operation_support_order(support), |candidate| {
                operation_support_order(*candidate)
            })
            .is_ok()
    }

    pub fn supports_primitive(&self, primitive: PrimitiveOperation) -> bool {
        self.supported
            .iter()
            .any(|support| support.primitive() == primitive)
    }

    pub fn is_deterministic(&self, support: OperationSupport) -> bool {
        self.deterministic
            .binary_search_by_key(&operation_support_order(support), |candidate| {
                operation_support_order(*candidate)
            })
            .is_ok()
    }

    pub fn supported(&self) -> &[OperationSupport] {
        &self.supported
    }

    pub fn deterministic(&self) -> &[OperationSupport] {
        &self.deterministic
    }

    pub fn device_properties(&self) -> Option<&NativeDeviceProperties> {
        self.properties.as_ref()
    }

    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported
            .iter()
            .any(|support| support.dtype == Some(dtype))
    }

    pub fn negotiate(&self, requested: &Self) -> Result<Self, TensorError> {
        if self.device != requested.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.device,
                actual: requested.device,
            });
        }
        let supported = self
            .supported
            .iter()
            .copied()
            .filter(|entry| requested.supports(*entry))
            .collect();
        let deterministic = self
            .deterministic
            .iter()
            .copied()
            .filter(|entry| requested.is_deterministic(*entry))
            .collect();
        Self::new_with_properties(
            self.device,
            supported,
            deterministic,
            self.properties.clone(),
        )
    }

    pub fn is_subset_of(&self, available: &Self) -> bool {
        self.device == available.device
            && self
                .supported
                .iter()
                .all(|entry| available.supports(*entry))
            && self
                .deterministic
                .iter()
                .all(|entry| available.is_deterministic(*entry))
    }

    pub fn to_worker_capabilities(&self) -> Result<WorkerBackendCapabilities, TensorError> {
        let properties = self
            .properties
            .as_ref()
            .map(|properties| {
                WorkerNativeDeviceProperties::new_with_allocation_limit(
                    properties.name(),
                    properties.total_memory_bytes(),
                    properties.allocation_limit_bytes(),
                    properties.major(),
                    properties.minor(),
                    properties.architecture().map(str::to_owned),
                    properties.has_fp16(),
                )
            })
            .transpose()
            .map_err(|error| TensorError::Faulted {
                reason: format!("worker native device property projection failed: {error}"),
            })?;
        WorkerBackendCapabilities::new_with_properties(
            self.device.kind(),
            self.device.ordinal(),
            self.supported
                .iter()
                .copied()
                .map(WorkerOperationSupport::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| TensorError::Faulted {
                    reason: format!("worker operation-support projection failed: {error}"),
                })?,
            self.deterministic
                .iter()
                .copied()
                .map(WorkerOperationSupport::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| TensorError::Faulted {
                    reason: format!("worker operation-support projection failed: {error}"),
                })?,
            properties,
        )
        .map_err(|error| TensorError::Faulted {
            reason: format!("worker backend capability projection failed: {error}"),
        })
    }

    pub fn require(&self, operation: &str, support: OperationSupport) -> Result<(), TensorError> {
        if self.supports(support) {
            Ok(())
        } else {
            Err(TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: self.device,
                reason: format!(
                    "primitive {:?}, role {:?}, dtype {:?}, layout {:?}",
                    support.primitive(),
                    support.role(),
                    support.dtype(),
                    support.layout()
                ),
            })
        }
    }

    pub fn require_primitive(
        &self,
        operation: &str,
        primitive: PrimitiveOperation,
    ) -> Result<(), TensorError> {
        if self.supports_primitive(primitive) {
            Ok(())
        } else {
            Err(TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: self.device,
                reason: format!("primitive {primitive:?} has no advertised tensor signature"),
            })
        }
    }
}

fn canonicalize_declared_support(
    mut support: Vec<OperationSupport>,
) -> Result<Vec<OperationSupport>, TensorError> {
    support.sort_unstable_by_key(|entry| operation_support_order(*entry));
    if support.windows(2).any(|entries| entries[0] == entries[1]) {
        return Err(TensorError::Faulted {
            reason: "backend capability declaration contains a duplicate primitive signature"
                .to_owned(),
        });
    }
    Ok(support)
}

fn canonicalize_internal_support(mut support: Vec<OperationSupport>) -> Vec<OperationSupport> {
    support.sort_unstable_by_key(|entry| operation_support_order(*entry));
    support.dedup();
    support
}

pub fn native_backend_binding_status(device: DeviceKind) -> NativeBackendBindingStatus {
    match device {
        DeviceKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                NativeBackendBindingStatus::bound(device)
            }
            #[cfg(not(feature = "cpu"))]
            {
                feature_disabled_binding(device, "cpu")
            }
        }
        DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_cuda::CudaBackend)
            }
            #[cfg(not(feature = "cuda"))]
            {
                feature_disabled_binding(device, "cuda")
            }
        }
        DeviceKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_rocm::RocmBackend)
            }
            #[cfg(not(feature = "rocm"))]
            {
                feature_disabled_binding(device, "rocm")
            }
        }
        DeviceKind::Metal => {
            #[cfg(feature = "metal")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_metal::MetalBackend)
            }
            #[cfg(not(feature = "metal"))]
            {
                feature_disabled_binding(device, "metal")
            }
        }
        DeviceKind::DirectMl => {
            #[cfg(feature = "directml")]
            {
                sim_types::NativeBackendBinding::binding_status(
                    &sim_backend_directml::DirectMlBackend,
                )
            }
            #[cfg(not(feature = "directml"))]
            {
                feature_disabled_binding(device, "directml")
            }
        }
        DeviceKind::Xpu => {
            #[cfg(feature = "xpu")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_xpu::XpuBackend)
            }
            #[cfg(not(feature = "xpu"))]
            {
                feature_disabled_binding(device, "xpu")
            }
        }
        DeviceKind::Npu => {
            #[cfg(feature = "npu")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_npu::NpuBackend)
            }
            #[cfg(not(feature = "npu"))]
            {
                feature_disabled_binding(device, "npu")
            }
        }
        DeviceKind::Mlu => {
            #[cfg(feature = "mlu")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_mlu::MluBackend)
            }
            #[cfg(not(feature = "mlu"))]
            {
                feature_disabled_binding(device, "mlu")
            }
        }
        DeviceKind::CoreX => {
            #[cfg(feature = "corex")]
            {
                sim_types::NativeBackendBinding::binding_status(&sim_backend_corex::CoreXBackend)
            }
            #[cfg(not(feature = "corex"))]
            {
                feature_disabled_binding(device, "corex")
            }
        }
    }
}

#[cfg(any(
    not(feature = "cpu"),
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "metal"),
    not(feature = "directml"),
    not(feature = "xpu"),
    not(feature = "npu"),
    not(feature = "mlu"),
    not(feature = "corex")
))]
fn feature_disabled_binding(device: DeviceKind, feature: &str) -> NativeBackendBindingStatus {
    NativeBackendBindingStatus::unbound(
        device,
        format!("the {feature} native backend binding is disabled in this build"),
    )
}

fn operation_support_order(support: OperationSupport) -> (u16, u8, u8, u8) {
    (
        primitive_operation_order(support.primitive()),
        match support.role() {
            Some(TensorRole::Input) => 0,
            Some(TensorRole::Output) => 1,
            None => u8::MAX,
        },
        support.dtype().map_or(u8::MAX, dtype_order),
        support.layout().map_or(u8::MAX, layout_order),
    )
}

fn primitive_operation_order(operation: PrimitiveOperation) -> u16 {
    match operation {
        PrimitiveOperation::Allocation => 0,
        PrimitiveOperation::Copy => 1,
        PrimitiveOperation::Fill => 2,
        PrimitiveOperation::Unary(operation) => 10 + unary_operation_order(operation),
        PrimitiveOperation::Binary(operation) => 100 + binary_operation_order(operation),
        PrimitiveOperation::BinaryScalar(operation) => 200 + binary_operation_order(operation),
        PrimitiveOperation::Reduction(operation) => 250 + reduction_operation_order(operation),
        PrimitiveOperation::Select => 300,
        PrimitiveOperation::Narrow => 301,
        PrimitiveOperation::Gather => 302,
        PrimitiveOperation::Scatter => 303,
        PrimitiveOperation::MaskedSelect => 304,
        PrimitiveOperation::Resize(mode) => 400 + resize_mode_order(mode),
        PrimitiveOperation::Convolution => 430,
        PrimitiveOperation::LinearAlgebra(operation) => {
            450 + linear_algebra_operation_order(operation)
        }
        PrimitiveOperation::CustomKernel => 480,
        PrimitiveOperation::RecordEvent => 500,
        PrimitiveOperation::WaitEvent => 501,
    }
}

fn linear_algebra_operation_order(operation: LinearAlgebraOperation) -> u16 {
    match operation {
        LinearAlgebraOperation::MatrixMultiply => 0,
        LinearAlgebraOperation::BatchMatrixMultiply => 1,
        LinearAlgebraOperation::MatrixVectorMultiply => 2,
        LinearAlgebraOperation::Dot => 3,
        LinearAlgebraOperation::Outer => 4,
        LinearAlgebraOperation::Solve => 5,
        LinearAlgebraOperation::SingularValueDecomposition => 6,
    }
}

fn reduction_operation_order(operation: ReductionOperation) -> u16 {
    match operation {
        ReductionOperation::Sum => 0,
        ReductionOperation::Product => 1,
        ReductionOperation::Mean => 2,
        ReductionOperation::Minimum => 3,
        ReductionOperation::Maximum => 4,
        ReductionOperation::ArgMinimum => 5,
        ReductionOperation::ArgMaximum => 6,
        ReductionOperation::All => 7,
        ReductionOperation::Any => 8,
        ReductionOperation::Variance => 9,
        ReductionOperation::StandardDeviation => 10,
    }
}

fn unary_operation_order(operation: UnaryOperation) -> u16 {
    match operation {
        UnaryOperation::Absolute => 0,
        UnaryOperation::Negate => 1,
        UnaryOperation::Exponential => 2,
        UnaryOperation::NaturalLogarithm => 3,
        UnaryOperation::SquareRoot => 4,
        UnaryOperation::Reciprocal => 5,
        UnaryOperation::Sine => 6,
        UnaryOperation::Cosine => 7,
        UnaryOperation::HyperbolicTangent => 8,
        UnaryOperation::Sigmoid => 9,
        UnaryOperation::Relu => 10,
        UnaryOperation::IsFinite => 11,
        UnaryOperation::InvertUnitInterval => 12,
        UnaryOperation::Round => 13,
        UnaryOperation::Sinc => 14,
        UnaryOperation::Log1p => 15,
        UnaryOperation::ReciprocalSquareRoot => 16,
        UnaryOperation::LogarithmBaseTwo => 17,
        UnaryOperation::Signum => 18,
        UnaryOperation::Tangent => 19,
        UnaryOperation::ArcTangent => 20,
        UnaryOperation::ArcHyperbolicTangent => 21,
    }
}

fn binary_operation_order(operation: BinaryOperation) -> u16 {
    match operation {
        BinaryOperation::Add => 0,
        BinaryOperation::Subtract => 1,
        BinaryOperation::Multiply => 2,
        BinaryOperation::Divide => 3,
        BinaryOperation::Remainder => 4,
        BinaryOperation::Power => 5,
        BinaryOperation::Minimum => 6,
        BinaryOperation::Maximum => 7,
        BinaryOperation::Equal => 8,
        BinaryOperation::Less => 9,
        BinaryOperation::LessEqual => 10,
        BinaryOperation::Greater => 11,
        BinaryOperation::GreaterEqual => 12,
        BinaryOperation::LogicalAnd => 13,
        BinaryOperation::LogicalOr => 14,
        BinaryOperation::FloatingRemainder => 15,
        BinaryOperation::Atan2 => 16,
        BinaryOperation::LogAddExp => 17,
    }
}

fn resize_mode_order(mode: ResizeMode) -> u16 {
    match mode {
        ResizeMode::NearestExact => 0,
        ResizeMode::Bilinear => 1,
        ResizeMode::Area => 2,
        ResizeMode::Bicubic => 3,
        ResizeMode::Lanczos => 4,
    }
}

fn dtype_order(dtype: DType) -> u8 {
    match dtype {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::F16 => 2,
        DType::Bf16 => 3,
        DType::I64 => 4,
        DType::I32 => 5,
        DType::I16 => 6,
        DType::I8 => 7,
        DType::U64 => 8,
        DType::U32 => 9,
        DType::U16 => 10,
        DType::U8 => 11,
        DType::Bool => 12,
        DType::Complex64 => 13,
        DType::Complex128 => 14,
        DType::Float8E4m3Fn => 15,
        DType::Float8E5m2 => 16,
        DType::Float8E4m3Fnuz => 17,
        DType::Float8E5m2Fnuz => 18,
        DType::Float8E8m0Fnu => 19,
    }
}

fn layout_order(layout: Layout) -> u8 {
    match layout {
        Layout::Contiguous => 0,
        Layout::ChannelsLast => 1,
        Layout::ChannelsLast3d => 2,
        Layout::Strided => 3,
    }
}

impl From<BackendCapabilityMatrix> for BackendCapabilityMatrixWire {
    fn from(value: BackendCapabilityMatrix) -> Self {
        Self {
            device: value.device,
            supported: value.supported,
            deterministic: value.deterministic,
            properties: value.properties,
        }
    }
}

impl TryFrom<BackendCapabilityMatrixWire> for BackendCapabilityMatrix {
    type Error = TensorError;

    fn try_from(value: BackendCapabilityMatrixWire) -> Result<Self, Self::Error> {
        Self::new_with_properties(
            value.device,
            value.supported,
            value.deterministic,
            value.properties,
        )
    }
}

impl TryFrom<WorkerBackendCapabilities> for BackendCapabilityMatrix {
    type Error = TensorError;

    fn try_from(value: WorkerBackendCapabilities) -> Result<Self, Self::Error> {
        let device = DeviceId::new(value.device(), value.ordinal());
        let properties = value
            .properties()
            .map(|properties| {
                NativeDeviceProperties::new_with_allocation_limit(
                    device,
                    properties.name(),
                    properties.total_memory_bytes(),
                    properties.allocation_limit_bytes(),
                    properties.major(),
                    properties.minor(),
                    properties.architecture().map(str::to_owned),
                    properties.has_fp16(),
                )
            })
            .transpose()?;
        Self::new_with_properties(
            device,
            value
                .supported()
                .iter()
                .copied()
                .map(OperationSupport::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            value
                .deterministic()
                .iter()
                .copied()
                .map(OperationSupport::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            properties,
        )
    }
}

impl TryFrom<OperationSupport> for WorkerOperationSupport {
    type Error = sim_types::WorkerOperationSupportError;

    fn try_from(value: OperationSupport) -> Result<Self, Self::Error> {
        let operation = WorkerPrimitiveOperationV2::from(value.primitive());
        match (value.role(), value.dtype(), value.layout()) {
            (Some(role), Some(dtype), Some(layout)) => {
                Self::for_tensor_v2(operation, role.into(), dtype.into(), layout.into())
            }
            (None, None, None) => Self::for_event_v2(operation),
            _ => Err(sim_types::WorkerOperationSupportError::MissingTensorSignature),
        }
    }
}

impl TryFrom<WorkerOperationSupport> for OperationSupport {
    type Error = TensorError;

    fn try_from(value: WorkerOperationSupport) -> Result<Self, Self::Error> {
        let primitive = PrimitiveOperation::from(value.operation());
        match (value.role(), value.dtype(), value.layout()) {
            (Some(role), Some(dtype), Some(layout)) => {
                Self::for_tensor(primitive, role.into(), dtype.into(), layout.into())
            }
            (None, None, None) => match primitive {
                PrimitiveOperation::RecordEvent => Ok(Self::record_event()),
                PrimitiveOperation::WaitEvent => Ok(Self::wait_event()),
                _ => Err(TensorError::Faulted {
                    reason: "worker tensor primitive capability has no role, dtype, or layout"
                        .to_owned(),
                }),
            },
            _ => Err(TensorError::Faulted {
                reason: "worker primitive capability contains a partial tensor signature"
                    .to_owned(),
            }),
        }
    }
}

macro_rules! exhaustive_enum_mapping {
    ($source:ty => $target:ty { $($variant:ident),+ $(,)? }) => {
        impl From<$source> for $target {
            fn from(value: $source) -> Self {
                match value {
                    $(<$source>::$variant => <$target>::$variant,)+
                }
            }
        }

        impl From<$target> for $source {
            fn from(value: $target) -> Self {
                match value {
                    $(<$target>::$variant => <$source>::$variant,)+
                }
            }
        }
    };
}

exhaustive_enum_mapping!(OperationCategory => WorkerOperationCategory {
    Allocation,
    Copy,
    Event,
    Scalar,
    Unary,
    Binary,
    Reduction,
    Indexing,
    Resize,
    Convolution,
    LinearAlgebra,
    CustomKernel,
});

exhaustive_enum_mapping!(DType => WorkerDType {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Complex64,
    Complex128,
    Float8E4m3Fn,
    Float8E5m2,
    Float8E4m3Fnuz,
    Float8E5m2Fnuz,
    Float8E8m0Fnu,
});

exhaustive_enum_mapping!(Layout => WorkerLayout {
    Contiguous,
    ChannelsLast,
    ChannelsLast3d,
    Strided,
});

exhaustive_enum_mapping!(TensorRole => WorkerTensorRoleV1 {
    Input,
    Output,
});

exhaustive_enum_mapping!(UnaryOperation => WorkerUnaryOperationV1 {
    Absolute,
    Negate,
    Exponential,
    NaturalLogarithm,
    SquareRoot,
    Reciprocal,
    Sine,
    Cosine,
    HyperbolicTangent,
    Sigmoid,
    Round,
    Sinc,
    Log1p,
    ReciprocalSquareRoot,
    Relu,
    IsFinite,
    InvertUnitInterval,
    LogarithmBaseTwo,
    Signum,
    Tangent,
    ArcTangent,
    ArcHyperbolicTangent,
});

exhaustive_enum_mapping!(BinaryOperation => WorkerBinaryOperationV1 {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Minimum,
    Maximum,
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    FloatingRemainder,
    Atan2,
    LogAddExp,
});

exhaustive_enum_mapping!(ResizeMode => WorkerResizeModeV1 {
    NearestExact,
    Bilinear,
    Area,
    Bicubic,
    Lanczos,
});

exhaustive_enum_mapping!(ReductionOperation => WorkerReductionOperationV1 {
    Sum,
    Product,
    Mean,
    Minimum,
    Maximum,
    ArgMinimum,
    ArgMaximum,
    All,
    Any,
    Variance,
    StandardDeviation,
});

exhaustive_enum_mapping!(LinearAlgebraOperation => WorkerLinearAlgebraOperationV1 {
    MatrixMultiply,
    BatchMatrixMultiply,
    MatrixVectorMultiply,
    Dot,
    Outer,
    Solve,
    SingularValueDecomposition,
});

impl From<PrimitiveOperation> for WorkerPrimitiveOperationV2 {
    fn from(value: PrimitiveOperation) -> Self {
        match value {
            PrimitiveOperation::Allocation => Self::Allocation,
            PrimitiveOperation::Copy => Self::Copy,
            PrimitiveOperation::Fill => Self::Fill,
            PrimitiveOperation::Unary(operation) => Self::Unary(operation.into()),
            PrimitiveOperation::Binary(operation) => Self::Binary(operation.into()),
            PrimitiveOperation::BinaryScalar(operation) => Self::BinaryScalar(operation.into()),
            PrimitiveOperation::Reduction(operation) => Self::Reduction(operation.into()),
            PrimitiveOperation::Select => Self::Select,
            PrimitiveOperation::Narrow => Self::Narrow,
            PrimitiveOperation::Resize(mode) => Self::Resize(mode.into()),
            PrimitiveOperation::LinearAlgebra(operation) => Self::LinearAlgebra(operation.into()),
            PrimitiveOperation::RecordEvent => Self::RecordEvent,
            PrimitiveOperation::WaitEvent => Self::WaitEvent,
            PrimitiveOperation::Gather => Self::Gather,
            PrimitiveOperation::Scatter => Self::Scatter,
            PrimitiveOperation::MaskedSelect => Self::MaskedSelect,
            PrimitiveOperation::Convolution => Self::Convolution,
            PrimitiveOperation::CustomKernel => Self::CustomKernel,
        }
    }
}

impl From<WorkerPrimitiveOperationV2> for PrimitiveOperation {
    fn from(value: WorkerPrimitiveOperationV2) -> Self {
        match value {
            WorkerPrimitiveOperationV2::Allocation => Self::Allocation,
            WorkerPrimitiveOperationV2::Copy => Self::Copy,
            WorkerPrimitiveOperationV2::Fill => Self::Fill,
            WorkerPrimitiveOperationV2::Unary(operation) => Self::Unary(operation.into()),
            WorkerPrimitiveOperationV2::Binary(operation) => Self::Binary(operation.into()),
            WorkerPrimitiveOperationV2::BinaryScalar(operation) => {
                Self::BinaryScalar(operation.into())
            }
            WorkerPrimitiveOperationV2::Reduction(operation) => Self::Reduction(operation.into()),
            WorkerPrimitiveOperationV2::Select => Self::Select,
            WorkerPrimitiveOperationV2::Narrow => Self::Narrow,
            WorkerPrimitiveOperationV2::Resize(mode) => Self::Resize(mode.into()),
            WorkerPrimitiveOperationV2::LinearAlgebra(operation) => {
                Self::LinearAlgebra(operation.into())
            }
            WorkerPrimitiveOperationV2::RecordEvent => Self::RecordEvent,
            WorkerPrimitiveOperationV2::WaitEvent => Self::WaitEvent,
            WorkerPrimitiveOperationV2::Gather => Self::Gather,
            WorkerPrimitiveOperationV2::Scatter => Self::Scatter,
            WorkerPrimitiveOperationV2::MaskedSelect => Self::MaskedSelect,
            WorkerPrimitiveOperationV2::Convolution => Self::Convolution,
            WorkerPrimitiveOperationV2::CustomKernel => Self::CustomKernel,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperation {
    Absolute,
    Negate,
    Exponential,
    NaturalLogarithm,
    SquareRoot,
    Reciprocal,
    Sine,
    Cosine,
    HyperbolicTangent,
    Sigmoid,
    Round,
    Sinc,
    Log1p,
    ReciprocalSquareRoot,
    Relu,
    IsFinite,
    InvertUnitInterval,
    LogarithmBaseTwo,
    Signum,
    Tangent,
    ArcTangent,
    ArcHyperbolicTangent,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Minimum,
    Maximum,
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    FloatingRemainder,
    Atan2,
    LogAddExp,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionOperation {
    Sum,
    Product,
    Mean,
    Minimum,
    Maximum,
    ArgMinimum,
    ArgMaximum,
    All,
    Any,
    Variance,
    StandardDeviation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReductionSpec {
    pub operation: ReductionOperation,
    pub dimensions: Vec<u64>,
    pub keep_dimensions: bool,
    pub accumulation_dtype: Option<DType>,
    #[serde(default)]
    pub correction: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum IndexSpec {
    Select {
        dimension: u64,
        index: i64,
    },
    Narrow {
        dimension: u64,
        start: i64,
        length: u64,
    },
    Gather {
        dimension: u64,
    },
    Scatter {
        dimension: u64,
    },
    MaskedSelect,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeMode {
    NearestExact,
    Bilinear,
    Area,
    Bicubic,
    Lanczos,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeCrop {
    Disabled,
    Center,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResizeSpec {
    pub width: u64,
    pub height: u64,
    pub mode: ResizeMode,
    pub crop: ResizeCrop,
    pub antialias: bool,
    #[serde(default)]
    pub align_corners: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConvolutionSpec {
    pub stride: Vec<u64>,
    pub padding: Vec<u64>,
    pub dilation: Vec<u64>,
    pub groups: u64,
    pub transposed: bool,
    pub output_padding: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearAlgebraOperation {
    MatrixMultiply,
    BatchMatrixMultiply,
    MatrixVectorMultiply,
    Dot,
    Outer,
    Solve,
    SingularValueDecomposition,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CustomKernelIdWire")]
pub struct CustomKernelId(String);

#[derive(Deserialize)]
struct CustomKernelIdWire(String);

impl TryFrom<CustomKernelIdWire> for CustomKernelId {
    type Error = TensorError;

    fn try_from(value: CustomKernelIdWire) -> Result<Self, Self::Error> {
        Self::new(value.0)
    }
}

impl CustomKernelId {
    pub fn new(value: impl Into<String>) -> Result<Self, TensorError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(TensorError::Faulted {
                reason: "custom kernel identifier is empty".to_owned(),
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub struct ExecutionContext<'a> {
    pub stream: StreamId,
    pub scratch: ScratchReservation,
    pub rng_phase: Option<&'a RngStreamAddress>,
    pub cancellation: &'a CancellationToken,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeStream {
    id: StreamId,
    device: DeviceId,
    priority: i32,
}

impl NativeStream {
    pub const fn id(self) -> StreamId {
        self.id
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }

    pub const fn priority(self) -> i32 {
        self.priority
    }
}

#[derive(Debug)]
pub struct NativeStreamRegistry {
    next_id: AtomicU64,
}

impl Default for NativeStreamRegistry {
    fn default() -> Self {
        Self {
            next_id: AtomicU64::new(1),
        }
    }
}

impl NativeStreamRegistry {
    pub fn create(
        &self,
        capabilities: &BackendCapabilityMatrix,
        device: DeviceId,
        priority: i32,
        cancellation: &CancellationToken,
    ) -> Result<NativeStream, TensorError> {
        cancellation.check()?;
        if capabilities.device() != device {
            return Err(TensorError::DeviceMismatch {
                expected: capabilities.device(),
                actual: device,
            });
        }
        let id = self
            .next_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| TensorError::ShapeOverflow)?;
        if id == StreamId::DEFAULT.get() {
            return Err(TensorError::Faulted {
                reason: "native stream registry attempted to allocate the default stream"
                    .to_owned(),
            });
        }
        Ok(NativeStream {
            id: StreamId::new(id),
            device,
            priority,
        })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BackendResourceRegistry<Resource> {
    resource_name: &'static str,
    limit: usize,
    resources: Mutex<BTreeMap<u64, Resource>>,
}

#[allow(dead_code)]
impl<Resource> BackendResourceRegistry<Resource>
where
    Resource: Clone,
{
    pub(crate) fn new(resource_name: &'static str, limit: usize) -> Self {
        Self {
            resource_name,
            limit,
            resources: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn get_or_try_insert_with(
        &self,
        stream: StreamId,
        create: impl FnOnce() -> Result<Resource, TensorError>,
    ) -> Result<Resource, TensorError> {
        let mut resources = self.resources.lock().map_err(|_| TensorError::Faulted {
            reason: format!("{} registry lock is poisoned", self.resource_name),
        })?;
        if let Some(resource) = resources.get(&stream.get()) {
            return Ok(resource.clone());
        }
        if resources.len() >= self.limit {
            return Err(TensorError::ResourceLimitExceeded {
                resource: self.resource_name,
                limit: self.limit,
            });
        }
        let resource = create()?;
        resources.insert(stream.get(), resource.clone());
        Ok(resource)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> Result<usize, TensorError> {
        self.resources
            .lock()
            .map(|resources| resources.len())
            .map_err(|_| TensorError::Faulted {
                reason: format!("{} registry lock is poisoned", self.resource_name),
            })
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct PendingBackendEvent<Event> {
    stream: StreamId,
    event: Event,
}

#[derive(Debug)]
#[allow(dead_code)]
struct BackendEventCursor {
    slot: u16,
    next: u64,
    completed: u64,
}

#[derive(Debug)]
#[allow(dead_code)]
struct BackendEventState<Event> {
    pending: BTreeMap<u64, PendingBackendEvent<Event>>,
    streams: BTreeMap<u64, BackendEventCursor>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BackendEventTracker<Event> {
    resource_name: &'static str,
    limit: usize,
    state: Arc<Mutex<BackendEventState<Event>>>,
}

const BACKEND_EVENT_COUNTER_BITS: u32 = 48;
const BACKEND_EVENT_COUNTER_MASK: u64 = (1_u64 << BACKEND_EVENT_COUNTER_BITS) - 1;

fn encode_backend_event_sequence(slot: u16, counter: u64) -> Result<u64, TensorError> {
    if slot == 0 || counter == 0 || counter > BACKEND_EVENT_COUNTER_MASK {
        return Err(TensorError::IdentifierOverflow);
    }
    Ok((u64::from(slot) << BACKEND_EVENT_COUNTER_BITS) | counter)
}

fn decode_backend_event_sequence(sequence: u64) -> Option<(u16, u64)> {
    let slot = u16::try_from(sequence >> BACKEND_EVENT_COUNTER_BITS).ok()?;
    let counter = sequence & BACKEND_EVENT_COUNTER_MASK;
    (slot != 0 && counter != 0).then_some((slot, counter))
}

#[allow(dead_code)]
impl<Event> BackendEventTracker<Event>
where
    Event: Clone,
{
    pub(crate) fn new(resource_name: &'static str, limit: usize) -> Self {
        Self {
            resource_name,
            limit,
            state: Arc::new(Mutex::new(BackendEventState {
                pending: BTreeMap::new(),
                streams: BTreeMap::new(),
            })),
        }
    }

    pub(crate) fn record_with(
        &self,
        stream: StreamId,
        create: impl FnOnce() -> Result<Event, TensorError>,
    ) -> Result<u64, TensorError> {
        let mut state = self.state.lock().map_err(|_| TensorError::Faulted {
            reason: format!("{} registry lock is poisoned", self.resource_name),
        })?;
        if state.pending.len() >= self.limit {
            return Err(TensorError::ResourceLimitExceeded {
                resource: self.resource_name,
                limit: self.limit,
            });
        }
        let stream_key = stream.get();
        let (slot, previous) = if let Some(cursor) = state.streams.get(&stream_key) {
            (cursor.slot, cursor.next)
        } else {
            if state.streams.len() >= self.limit {
                return Err(TensorError::ResourceLimitExceeded {
                    resource: self.resource_name,
                    limit: self.limit,
                });
            }
            let slot = state
                .streams
                .len()
                .checked_add(1)
                .and_then(|slot| u16::try_from(slot).ok())
                .filter(|slot| *slot != 0)
                .ok_or(TensorError::IdentifierOverflow)?;
            (slot, 0)
        };
        let counter = previous
            .checked_add(1)
            .filter(|counter| *counter <= BACKEND_EVENT_COUNTER_MASK)
            .ok_or(TensorError::IdentifierOverflow)?;
        let sequence = encode_backend_event_sequence(slot, counter)?;
        let event = create()?;
        state
            .streams
            .entry(stream_key)
            .and_modify(|cursor| cursor.next = counter)
            .or_insert(BackendEventCursor {
                slot,
                next: counter,
                completed: 0,
            });
        state
            .pending
            .insert(sequence, PendingBackendEvent { stream, event });
        Ok(sequence)
    }

    pub(crate) fn cancel(
        &self,
        stream: StreamId,
        sequence: u64,
    ) -> Result<Option<Event>, TensorError> {
        let event = {
            let mut state = self.state.lock().map_err(|_| TensorError::Faulted {
                reason: format!("{} registry lock is poisoned", self.resource_name),
            })?;
            self.validate_sequence(&state, stream, sequence)?;
            if !state
                .pending
                .get(&sequence)
                .is_some_and(|pending| pending.stream == stream)
            {
                return Ok(None);
            }
            state.pending.remove(&sequence).map(|pending| pending.event)
        };
        Ok(event)
    }

    pub(crate) fn event_for_wait(
        &self,
        stream: StreamId,
        sequence: u64,
    ) -> Result<Option<Event>, TensorError> {
        let state = self.state.lock().map_err(|_| TensorError::Faulted {
            reason: format!("{} registry lock is poisoned", self.resource_name),
        })?;
        let counter = self.validate_sequence(&state, stream, sequence)?;
        let completed = state
            .streams
            .get(&stream.get())
            .map(|cursor| cursor.completed)
            .unwrap_or(0);
        if counter <= completed {
            return Ok(None);
        }
        state
            .pending
            .get(&sequence)
            .filter(|pending| pending.stream == stream)
            .map(|pending| Some(pending.event.clone()))
            .ok_or_else(|| TensorError::Faulted {
                reason: format!(
                    "{} sequence {sequence} was not recorded for stream {}",
                    self.resource_name,
                    stream.get()
                ),
            })
    }

    pub(crate) fn complete(
        &self,
        stream: StreamId,
        sequence: u64,
    ) -> Result<Vec<Event>, TensorError> {
        let retired = {
            let mut state = self.state.lock().map_err(|_| TensorError::Faulted {
                reason: format!("{} registry lock is poisoned", self.resource_name),
            })?;
            let counter = self.validate_sequence(&state, stream, sequence)?;
            let completed = state
                .streams
                .get(&stream.get())
                .map(|cursor| cursor.completed)
                .unwrap_or(0);
            if counter <= completed {
                return Ok(Vec::new());
            }
            if !state
                .pending
                .get(&sequence)
                .is_some_and(|pending| pending.stream == stream)
            {
                return Err(TensorError::Faulted {
                    reason: format!(
                        "{} sequence {sequence} was not recorded for stream {}",
                        self.resource_name,
                        stream.get()
                    ),
                });
            }
            let sequences = state
                .pending
                .iter()
                .filter_map(|(pending_sequence, pending_event)| {
                    let pending_counter = decode_backend_event_sequence(*pending_sequence)
                        .map(|(_, counter)| counter);
                    (pending_event.stream == stream
                        && pending_counter.is_some_and(|pending| pending <= counter))
                    .then_some(*pending_sequence)
                })
                .collect::<Vec<_>>();
            state
                .streams
                .get_mut(&stream.get())
                .ok_or_else(|| TensorError::Faulted {
                    reason: format!(
                        "{} sequence {sequence} has no stream owner",
                        self.resource_name
                    ),
                })?
                .completed = counter;
            sequences
                .into_iter()
                .filter_map(|pending_sequence| {
                    state
                        .pending
                        .remove(&pending_sequence)
                        .map(|pending| pending.event)
                })
                .collect::<Vec<_>>()
        };
        Ok(retired)
    }

    fn validate_sequence(
        &self,
        state: &BackendEventState<Event>,
        stream: StreamId,
        sequence: u64,
    ) -> Result<u64, TensorError> {
        let (slot, counter) =
            decode_backend_event_sequence(sequence).ok_or_else(|| TensorError::Faulted {
                reason: format!(
                    "{} sequence {sequence} has no bounded stream provenance",
                    self.resource_name
                ),
            })?;
        let valid = state
            .streams
            .get(&stream.get())
            .is_some_and(|cursor| cursor.slot == slot && counter <= cursor.next);
        if !valid {
            return Err(TensorError::Faulted {
                reason: format!(
                    "{} sequence {sequence} was not recorded for stream {}",
                    self.resource_name,
                    stream.get()
                ),
            });
        }
        Ok(counter)
    }

    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> Result<usize, TensorError> {
        self.state
            .lock()
            .map(|state| state.pending.len())
            .map_err(|_| TensorError::Faulted {
                reason: format!("{} registry lock is poisoned", self.resource_name),
            })
    }

    #[cfg(test)]
    pub(crate) fn completed_stream_count(&self) -> Result<usize, TensorError> {
        self.state
            .lock()
            .map(|state| {
                state
                    .streams
                    .values()
                    .filter(|cursor| cursor.completed != 0)
                    .count()
            })
            .map_err(|_| TensorError::Faulted {
                reason: format!(
                    "{} completion registry lock is poisoned",
                    self.resource_name
                ),
            })
    }
}

pub fn native_device_name_exact(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    expected_kind: DeviceKind,
    operation: &str,
    cancellation: &CancellationToken,
) -> Result<String, TensorError> {
    native_device_name_exact_for_kinds(
        capabilities,
        device,
        &[expected_kind],
        operation,
        cancellation,
    )
}

pub fn native_device_name_exact_for_kinds(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    expected_kinds: &[DeviceKind],
    operation: &str,
    cancellation: &CancellationToken,
) -> Result<String, TensorError> {
    cancellation.check()?;
    if !expected_kinds.contains(&device.kind()) || capabilities.device() != device {
        return Err(TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!(
                "expected an exact {expected_kinds:?} capability matrix with native device properties"
            ),
        });
    }
    let properties =
        capabilities
            .device_properties()
            .ok_or_else(|| TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "the canonical capability matrix has no certified device properties"
                    .to_owned(),
            })?;
    let name = properties.name().to_owned();
    cancellation.check()?;
    Ok(name)
}

pub fn native_select_device_exact<'a>(
    available: &'a [BackendCapabilityMatrix],
    device: DeviceId,
    expected_kinds: &[DeviceKind],
    operation: &str,
    cancellation: &CancellationToken,
) -> Result<&'a BackendCapabilityMatrix, TensorError> {
    cancellation.check()?;
    if !expected_kinds.contains(&device.kind()) {
        return Err(TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: format!("expected one of the native device kinds {expected_kinds:?}"),
        });
    }
    let mut matching = available.iter().filter(|matrix| matrix.device() == device);
    let selected = matching
        .next()
        .ok_or_else(|| TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: "the selected device has no canonical capability matrix".to_owned(),
        })?;
    if matching.next().is_some() {
        return Err(TensorError::Faulted {
            reason: format!(
                "operation {operation} found duplicate capability matrices for {device:?}"
            ),
        });
    }
    cancellation.check()?;
    Ok(selected)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AutocastPolicy {
    enabled: bool,
    dtype: DType,
    cache_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeterministicAlgorithmsPolicy {
    enabled: bool,
    warn_only: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeterministicOperationDisposition {
    Allowed,
    Warn,
    Reject,
}

impl DeterministicAlgorithmsPolicy {
    pub const fn new(enabled: bool, warn_only: bool) -> Self {
        Self { enabled, warn_only }
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn warn_only(self) -> bool {
        self.warn_only
    }

    pub fn disposition(
        self,
        capabilities: &BackendCapabilityMatrix,
        operation: OperationSupport,
    ) -> DeterministicOperationDisposition {
        if !self.enabled || capabilities.is_deterministic(operation) {
            DeterministicOperationDisposition::Allowed
        } else if self.warn_only {
            DeterministicOperationDisposition::Warn
        } else {
            DeterministicOperationDisposition::Reject
        }
    }
}

impl AutocastPolicy {
    pub fn new(enabled: bool, dtype: DType, cache_enabled: bool) -> Result<Self, TensorError> {
        if !matches!(dtype.class(), NumericClass::FloatingPoint) {
            return Err(TensorError::InvalidNumeric {
                reason: format!("autocast requires a floating-point dtype, got {dtype:?}"),
            });
        }
        Ok(Self {
            enabled,
            dtype,
            cache_enabled,
        })
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn cache_enabled(self) -> bool {
        self.cache_enabled
    }
}

impl ExecutionContext<'_> {
    pub fn check(&self) -> Result<(), TensorError> {
        self.cancellation.check()?;
        Ok(())
    }
}

pub trait CachedAllocationOwner: Send + Sync {
    fn cache_device(&self) -> DeviceId;

    fn allocator_backend_name(&self) -> &'static str {
        match self.cache_device().kind() {
            DeviceKind::Cpu => "zed-native-cpu-host-v1",
            DeviceKind::Cuda | DeviceKind::Rocm => "zed-native-cuda-caching-v1",
            DeviceKind::Metal => "zed-native-metal-heap-v1",
            DeviceKind::Xpu => "zed-native-xpu-caching-v1",
            DeviceKind::DirectMl => "zed-native-directml-residency-v1",
            DeviceKind::Npu => "zed-native-npu-caching-v1",
            DeviceKind::Mlu => "zed-native-mlu-caching-v1",
            DeviceKind::CoreX => "zed-native-corex-caching-v1",
        }
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError>;
}

pub trait TensorBackend: CachedAllocationOwner + Send + Sync {
    #[cfg(feature = "cpu")]
    fn cpu_backend(&self) -> Option<&crate::CpuBackend> {
        None
    }

    fn device(&self) -> DeviceId;
    fn capabilities(&self) -> &BackendCapabilityMatrix;

    fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "zed.tensor.workspace.reserve".to_owned(),
            device: self.device(),
            reason: format!(
                "the selected backend does not expose workspace reservation for {requested} bytes"
            ),
        })
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn copy(
        &self,
        source: &Tensor,
        destination: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError>;

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError>;

    fn fill(
        &self,
        value: Scalar,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn unary(
        &self,
        operation: UnaryOperation,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn binary(
        &self,
        operation: BinaryOperation,
        left: &Tensor,
        right: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    #[allow(clippy::too_many_arguments)]
    fn binary_scalar(
        &self,
        operation: BinaryOperation,
        input: &Tensor,
        scalar: Scalar,
        scalar_side: ScalarSide,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn reduction(
        &self,
        operation: &ReductionSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn indexing(
        &self,
        operation: &IndexSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn resize(
        &self,
        operation: ResizeSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn convolution(
        &self,
        operation: &ConvolutionSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn linear_algebra(
        &self,
        operation: LinearAlgebraOperation,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError>;

    fn matrix_inverse(
        &self,
        _input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-7DD46810B2C2".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical matrix inverse".to_owned(),
        })
    }

    fn kronecker_product(
        &self,
        _left: &Tensor,
        _right: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-F122D7D4E807".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical Kronecker product".to_owned(),
        })
    }

    fn constant_pad(
        &self,
        _input: &Tensor,
        _padding: &[i64],
        _value: Option<DecodedScalar>,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-E867958E2F71".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical constant padding".to_owned(),
        })
    }

    fn vector_norm(
        &self,
        _input: &Tensor,
        _order: f64,
        _dimensions: &[i64],
        _keep_dimension: bool,
        _dtype: Option<DType>,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-8E3FD7459720".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical vector norm".to_owned(),
        })
    }

    fn eye(
        &self,
        _rows: u64,
        _columns: Option<u64>,
        _dtype: DType,
        _layout: Layout,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-CA2E738EA0EF".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical identity-matrix creation"
                .to_owned(),
        })
    }

    fn replace_rectangular_slice(
        &self,
        _input: &Tensor,
        _source: &Tensor,
        _offsets: &[u64],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "zed.tensor.indexing.rectangular-slice-replacement.v1".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose exact rectangular slice replacement"
                .to_owned(),
        })
    }

    fn validate_finite(
        &self,
        _input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-2C5A78E85B7F".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical finite-value validation"
                .to_owned(),
        })
    }

    fn upload_f32_payload(
        &self,
        _shape: &[u64],
        _values: &[f32],
        _dtype: DType,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-56B106D5BEE7".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose checked F32 payload upload".to_owned(),
        })
    }

    fn cast_tensor(
        &self,
        _input: &Tensor,
        _dtype: DType,
        _non_blocking: bool,
        _copy: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        Err(TensorError::UnsupportedCapability {
            operation: "COMFY-TENSOR-OP-56B106D5BEE7".to_owned(),
            device: self.device(),
            reason: "the selected backend does not expose canonical dtype cast or contiguous copy"
                .to_owned(),
        })
    }

    fn custom_kernel(
        &self,
        kernel: &CustomKernelId,
        inputs: &[Tensor],
        outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError>;
}

pub fn synchronize_device_exact_native(
    backend: &dyn TensorBackend,
    capabilities: &BackendCapabilityMatrix,
    expected_kinds: &[DeviceKind],
    operation: &str,
    execution: &ExecutionContext<'_>,
) -> Result<(), TensorError> {
    execution.check()?;
    let device = capabilities.device();
    if !expected_kinds.contains(&device.kind()) || backend.device() != device {
        return Err(TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device,
            reason: "backend and capability matrix must identify the same allowed device"
                .to_owned(),
        });
    }
    capabilities.require(operation, OperationSupport::record_event())?;
    capabilities.require(operation, OperationSupport::wait_event())?;
    let event = backend.record_event(execution)?;
    backend.wait_event(event, execution)?;
    execution.check()?;
    Ok(())
}

pub fn validate_inputs(
    backend: &dyn TensorBackend,
    operation: &str,
    primitive: PrimitiveOperation,
    inputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<(), TensorError> {
    context.check()?;
    for input in inputs {
        if input.descriptor().device() != backend.device() {
            return Err(TensorError::DeviceMismatch {
                expected: backend.device(),
                actual: input.descriptor().device(),
            });
        }
        if input.descriptor().stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: input.descriptor().stream(),
            });
        }
        backend.capabilities().require(
            operation,
            OperationSupport::for_tensor(
                primitive,
                TensorRole::Input,
                input.descriptor().dtype(),
                input.descriptor().layout(),
            )?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DropCheckedEvent {
        state: std::sync::Weak<Mutex<BackendEventState<DropCheckedEvent>>>,
        drops: Arc<AtomicU64>,
        drops_while_locked: Arc<AtomicU64>,
    }

    impl Drop for DropCheckedEvent {
        fn drop(&mut self) {
            if self
                .state
                .upgrade()
                .is_some_and(|pending| pending.try_lock().is_err())
            {
                self.drops_while_locked.fetch_add(1, Ordering::AcqRel);
            }
            self.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn backend_resource_registries_own_bounds_completion_and_drop_transitions()
    -> Result<(), TensorError> {
        let streams = BackendResourceRegistry::new("test streams", 2);
        let creates = AtomicU64::new(0);
        assert_eq!(
            streams.get_or_try_insert_with(StreamId::DEFAULT, || {
                creates.fetch_add(1, Ordering::AcqRel);
                Ok(11_u64)
            })?,
            11
        );
        assert_eq!(
            streams.get_or_try_insert_with(StreamId::DEFAULT, || {
                creates.fetch_add(1, Ordering::AcqRel);
                Ok(99_u64)
            })?,
            11
        );
        assert_eq!(
            streams.get_or_try_insert_with(StreamId::new(1), || Ok(12_u64))?,
            12
        );
        assert!(matches!(
            streams.get_or_try_insert_with(StreamId::new(2), || Ok(13_u64)),
            Err(TensorError::ResourceLimitExceeded {
                resource: "test streams",
                limit: 2,
            })
        ));
        assert_eq!(creates.load(Ordering::Acquire), 1);

        let events = BackendEventTracker::new("test pending events", 2);
        let drops = Arc::new(AtomicU64::new(0));
        let drops_while_locked = Arc::new(AtomicU64::new(0));
        let event = || DropCheckedEvent {
            state: Arc::downgrade(&events.state),
            drops: drops.clone(),
            drops_while_locked: drops_while_locked.clone(),
        };
        let first = events.record_with(StreamId::DEFAULT, || Ok(event()))?;
        let second = events.record_with(StreamId::DEFAULT, || Ok(event()))?;
        assert!(matches!(
            events.record_with(StreamId::DEFAULT, || Ok(event())),
            Err(TensorError::ResourceLimitExceeded {
                resource: "test pending events",
                limit: 2,
            })
        ));
        assert!(events.event_for_wait(StreamId::DEFAULT, first)?.is_some());
        assert!(matches!(
            events.complete(StreamId::new(1), first),
            Err(TensorError::Faulted { .. })
        ));
        assert!(matches!(
            events.complete(StreamId::DEFAULT, 999),
            Err(TensorError::Faulted { .. })
        ));
        let retired = events.complete(StreamId::DEFAULT, second)?;
        assert_eq!(retired.len(), 2);
        drop(retired);
        assert!(events.event_for_wait(StreamId::DEFAULT, first)?.is_none());
        assert!(matches!(
            events.event_for_wait(StreamId::new(1), first),
            Err(TensorError::Faulted { .. })
        ));
        assert!(matches!(
            events.complete(StreamId::new(1), first),
            Err(TensorError::Faulted { .. })
        ));
        assert!(matches!(
            events.event_for_wait(StreamId::DEFAULT, 0),
            Err(TensorError::Faulted { .. })
        ));
        assert!(matches!(
            events.complete(StreamId::DEFAULT, 0),
            Err(TensorError::Faulted { .. })
        ));
        assert_eq!(events.pending_len()?, 0);
        assert_eq!(events.completed_stream_count()?, 1);

        let third = events.record_with(StreamId::new(1), || Ok(event()))?;
        let cancelled = events.cancel(StreamId::new(1), third)?;
        drop(cancelled);
        assert_eq!(drops.load(Ordering::Acquire), 4);
        assert_eq!(drops_while_locked.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn backend_event_watermark_preserves_old_fences_with_bounded_stream_provenance()
    -> Result<(), TensorError> {
        let events = BackendEventTracker::new("test event watermark", 2);
        let first = events.record_with(StreamId::DEFAULT, || Ok(1_u64))?;
        let second = events.record_with(StreamId::DEFAULT, || Ok(2_u64))?;
        drop(events.complete(StreamId::DEFAULT, second)?);

        let third = events.record_with(StreamId::DEFAULT, || Ok(3_u64))?;
        let fourth = events.record_with(StreamId::DEFAULT, || Ok(4_u64))?;
        drop(events.complete(StreamId::DEFAULT, fourth)?);

        assert!(events.event_for_wait(StreamId::DEFAULT, first)?.is_none());
        assert!(events.complete(StreamId::DEFAULT, first)?.is_empty());
        assert!(matches!(
            events.event_for_wait(StreamId::new(1), first),
            Err(TensorError::Faulted { .. })
        ));
        assert!(matches!(
            events.event_for_wait(StreamId::DEFAULT, 0),
            Err(TensorError::Faulted { .. })
        ));
        assert!(third < fourth);
        Ok(())
    }

    #[test]
    fn workspace_authority_is_single_owner_and_not_cloneable() {
        let operation_source = include_str!("operation.rs");
        let authority_declaration = ["pub struct ", "BackendWorkspaceAuthority"].concat();
        let declaration = operation_source
            .split(&authority_declaration)
            .next()
            .and_then(|prefix| prefix.rsplit("#[derive(").next())
            .expect("workspace authority declaration has a derive attribute");
        assert!(!declaration.contains("Clone"));
        let forbidden_clone_impl = ["impl Clone for ", "BackendWorkspaceAuthority"].concat();
        assert!(!operation_source.contains(&forbidden_clone_impl));

        let cpu_source = include_str!("cpu_backend.rs");
        assert!(cpu_source.contains("pub type CpuWorkspaceAuthority = BackendWorkspaceAuthority;"));
        let forbidden_cpu_struct = ["pub struct ", "CpuWorkspaceAuthority"].concat();
        assert!(!cpu_source.contains(&forbidden_cpu_struct));
        let forbidden_unpaired_constructor = ["pub fn ", "new(memory_limit_bytes"].concat();
        assert!(!cpu_source.contains(&forbidden_unpaired_constructor));
    }

    #[test]
    fn val_cancel_001_tensor_adapter_maps_before_dispatch() {
        let token = CancellationToken::default();
        let clone = token.clone();
        assert!(token.check().is_ok());
        clone.cancel();
        assert_eq!(
            token.check().map_err(TensorError::from),
            Err(TensorError::Cancelled)
        );
    }

    #[test]
    fn identifier_wire_adapters_revalidate_invariants() {
        assert!(OperationContractId::try_from(OperationContractIdWire("  ".to_owned())).is_err());
        assert!(
            OperationContractId::try_from(OperationContractIdWire(
                "zed.native-internal.operation.v1".to_owned()
            ))
            .is_err()
        );
        assert!(OperationContractId::cataloged("unknown.operation").is_err());
        let resolved_callable =
            crate::operation_contracts::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                .iter()
                .flat_map(|slice| slice.iter())
                .next()
                .expect("the native tensor ledger has compiled callable resolutions");
        assert!(matches!(
            OperationContractId::cataloged(resolved_callable.operation_id),
            Ok(identifier) if identifier.as_str() == resolved_callable.operation_id
        ));
        assert!(matches!(
            OperationContractId::cataloged(resolved_callable.overload_id),
            Ok(identifier) if identifier.as_str() == resolved_callable.overload_id
        ));
        let reference = crate::operation_contracts::OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.typed_reference().is_some())
            .expect("the Task 7 ledger has typed references");
        assert!(OperationContractId::cataloged(reference.operation_id).is_err());
        assert!(CustomKernelId::try_from(CustomKernelIdWire(String::new())).is_err());
        assert!(matches!(
            CustomKernelId::try_from(CustomKernelIdWire("kernel.v1".to_owned())),
            Ok(identifier) if identifier.as_str() == "kernel.v1"
        ));
    }

    #[test]
    fn deterministic_capabilities_must_be_supported() {
        let support =
            OperationSupport::unary_input(UnaryOperation::Absolute, DType::F32, Layout::Contiguous);
        assert!(BackendCapabilityMatrix::new(DeviceId::CPU, vec![], vec![support]).is_err());
        assert!(
            BackendCapabilityMatrix::new(DeviceId::CPU, vec![support, support], vec![]).is_err()
        );
        let matrix = BackendCapabilityMatrix::new(DeviceId::CPU, vec![support], vec![support]);
        assert!(matches!(matrix, Ok(value) if value.is_deterministic(support)));
    }

    #[test]
    fn backend_capability_wire_adapter_preserves_canonical_semantics() {
        let unary =
            OperationSupport::unary_input(UnaryOperation::Absolute, DType::F32, Layout::Contiguous);
        let binary =
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        let record_event = OperationSupport::record_event();
        let matrix = BackendCapabilityMatrix::new(
            DeviceId::CPU,
            vec![unary, binary, record_event],
            vec![unary],
        )
        .expect("valid canonical matrix");
        let wire = matrix
            .to_worker_capabilities()
            .expect("matrix projects to bounded worker DTO");
        let round_trip = BackendCapabilityMatrix::try_from(wire).expect("worker DTO maps back");
        assert_eq!(round_trip, matrix);
        assert_eq!(round_trip.supported(), [unary, binary, record_event]);
        assert!(round_trip.supports(binary));
        assert!(round_trip.supports(record_event));
        assert_eq!(record_event.dtype(), None);
        assert_eq!(record_event.layout(), None);
        assert!(round_trip.is_deterministic(unary));
        assert!(!round_trip.is_deterministic(binary));

        let semantically_invalid_wire = WorkerBackendCapabilities::new(
            sim_types::DeviceKind::Cpu,
            0,
            vec![WorkerOperationSupport::try_from(unary).expect("valid unary projection")],
            vec![WorkerOperationSupport::try_from(binary).expect("valid binary projection")],
        )
        .expect("boundary DTO is structurally valid");
        assert!(BackendCapabilityMatrix::try_from(semantically_invalid_wire).is_err());
    }

    #[test]
    fn native_device_properties_revalidate_and_remain_matrix_owned() {
        let device = DeviceId::new(sim_types::DeviceKind::Cuda, 2);
        let properties = NativeDeviceProperties::new_with_allocation_limit(
            device,
            "fixture accelerator",
            16 * 1024 * 1024,
            12 * 1024 * 1024,
            9,
            0,
            Some("sm_90".to_owned()),
            true,
        )
        .expect("valid properties");
        let support = OperationSupport::allocation(DType::Bf16, Layout::Contiguous);
        let matrix = BackendCapabilityMatrix::new_with_properties(
            device,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("matching matrix properties");
        assert_eq!(matrix.device_properties(), Some(&properties));
        assert_eq!(properties.total_memory_bytes(), 16 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 12 * 1024 * 1024);
        let worker = matrix
            .to_worker_capabilities()
            .expect("canonical properties map to the bounded worker DTO");
        assert_eq!(worker.device(), sim_types::DeviceKind::Cuda);
        assert_eq!(worker.ordinal(), 2);
        let round_trip = BackendCapabilityMatrix::try_from(worker)
            .expect("bounded worker properties map through the canonical validator");
        assert_eq!(round_trip, matrix);
        assert_eq!(
            round_trip
                .device_properties()
                .expect("round-trip properties")
                .allocation_limit_bytes(),
            12 * 1024 * 1024
        );
        assert!(matrix.supports_dtype(DType::Bf16));
        assert!(!matrix.supports_dtype(DType::F16));
        let serialized = serde_json::to_string(&matrix).expect("serialize matrix");
        let round_trip: BackendCapabilityMatrix =
            serde_json::from_str(&serialized).expect("revalidate matrix");
        assert_eq!(round_trip, matrix);

        let mut invalid = serde_json::to_value(&properties).expect("serialize properties");
        invalid["name"] = serde_json::Value::String(String::new());
        invalid["total_memory_bytes"] = serde_json::Value::from(0);
        assert!(serde_json::from_value::<NativeDeviceProperties>(invalid).is_err());
        assert!(
            NativeDeviceProperties::new_with_allocation_limit(
                device,
                "fixture accelerator",
                16,
                0,
                9,
                0,
                None,
                true,
            )
            .is_err()
        );
        assert!(
            NativeDeviceProperties::new_with_allocation_limit(
                device,
                "fixture accelerator",
                16,
                17,
                9,
                0,
                None,
                true,
            )
            .is_err()
        );

        let other = DeviceId::new(sim_types::DeviceKind::Cuda, 3);
        assert!(matches!(
            BackendCapabilityMatrix::new_with_properties(
                other,
                vec![support],
                vec![support],
                Some(properties),
            ),
            Err(TensorError::DeviceMismatch { .. })
        ));

        let invalid_properties =
            WorkerNativeDeviceProperties::new("fixture", 1, 0, 0, Some(String::new()), false)
                .expect("wire DTO owns bounds, not canonical device semantics");
        let invalid_worker = WorkerBackendCapabilities::new_with_properties(
            sim_types::DeviceKind::Cuda,
            2,
            vec![WorkerOperationSupport::try_from(support).expect("valid support projection")],
            vec![],
            Some(invalid_properties),
        )
        .expect("wire declaration is structurally bounded");
        assert!(BackendCapabilityMatrix::try_from(invalid_worker).is_err());
    }

    #[test]
    fn every_v1_primitive_support_round_trips_exactly() {
        let unary_operations = [
            UnaryOperation::Absolute,
            UnaryOperation::Negate,
            UnaryOperation::Exponential,
            UnaryOperation::NaturalLogarithm,
            UnaryOperation::SquareRoot,
            UnaryOperation::Reciprocal,
            UnaryOperation::Sine,
            UnaryOperation::Cosine,
            UnaryOperation::HyperbolicTangent,
            UnaryOperation::Sigmoid,
            UnaryOperation::Relu,
            UnaryOperation::IsFinite,
            UnaryOperation::InvertUnitInterval,
        ];
        let binary_operations = [
            BinaryOperation::Add,
            BinaryOperation::Subtract,
            BinaryOperation::Multiply,
            BinaryOperation::Divide,
            BinaryOperation::Remainder,
            BinaryOperation::Power,
            BinaryOperation::Minimum,
            BinaryOperation::Maximum,
            BinaryOperation::Equal,
            BinaryOperation::Less,
            BinaryOperation::LessEqual,
            BinaryOperation::Greater,
            BinaryOperation::GreaterEqual,
            BinaryOperation::LogicalAnd,
            BinaryOperation::LogicalOr,
        ];
        let resize_modes = [
            ResizeMode::NearestExact,
            ResizeMode::Bilinear,
            ResizeMode::Area,
            ResizeMode::Bicubic,
            ResizeMode::Lanczos,
        ];
        let mut supports = vec![
            OperationSupport::allocation(DType::F32, Layout::Contiguous),
            OperationSupport::copy_input(DType::F32, Layout::Contiguous),
            OperationSupport::copy_output(DType::F32, Layout::Contiguous),
            OperationSupport::fill(DType::F32, Layout::Contiguous),
            OperationSupport::select_input(DType::F32, Layout::Contiguous),
            OperationSupport::select_output(DType::F32, Layout::Contiguous),
            OperationSupport::narrow_input(DType::F32, Layout::Contiguous),
            OperationSupport::narrow_output(DType::F32, Layout::Contiguous),
            OperationSupport::gather_input(DType::F32, Layout::Contiguous),
            OperationSupport::gather_output(DType::F32, Layout::Contiguous),
            OperationSupport::scatter_input(DType::F32, Layout::Contiguous),
            OperationSupport::scatter_output(DType::F32, Layout::Contiguous),
            OperationSupport::masked_select_input(DType::F32, Layout::Contiguous),
            OperationSupport::masked_select_output(DType::F32, Layout::Contiguous),
            OperationSupport::convolution_input(DType::F32, Layout::Contiguous),
            OperationSupport::convolution_output(DType::F32, Layout::Contiguous),
            OperationSupport::custom_kernel_input(DType::F32, Layout::Contiguous),
            OperationSupport::custom_kernel_output(DType::F32, Layout::Contiguous),
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
        for operation in unary_operations {
            supports.push(OperationSupport::unary_input(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
            supports.push(OperationSupport::unary_output(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
        }
        for operation in binary_operations {
            supports.push(OperationSupport::binary_input(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
            supports.push(OperationSupport::binary_output(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
            supports.push(OperationSupport::binary_scalar_input(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
            supports.push(OperationSupport::binary_scalar_output(
                operation,
                DType::F32,
                Layout::Contiguous,
            ));
        }
        for mode in resize_modes {
            supports.push(OperationSupport::resize_input(
                mode,
                DType::F32,
                Layout::Contiguous,
            ));
            supports.push(OperationSupport::resize_output(
                mode,
                DType::F32,
                Layout::Contiguous,
            ));
        }

        for support in supports {
            let worker = WorkerOperationSupport::try_from(support)
                .expect("valid domain support maps to worker v1");
            assert_eq!(
                worker.version(),
                sim_types::WORKER_OPERATION_SUPPORT_VERSION
            );
            assert_eq!(
                worker.category(),
                WorkerOperationCategory::from(support.category())
            );
            assert_eq!(worker.role().map(TensorRole::from), support.role());
            assert_eq!(OperationSupport::try_from(worker), Ok(support));
        }
    }

    #[test]
    fn operation_support_wire_rejects_tensor_event_shape_confusion() {
        assert!(
            OperationSupport::try_from(OperationSupportWire {
                primitive: PrimitiveOperation::RecordEvent,
                role: Some(TensorRole::Output),
                dtype: Some(DType::F32),
                layout: Some(Layout::Contiguous),
            })
            .is_err()
        );
        assert!(
            OperationSupport::try_from(OperationSupportWire {
                primitive: PrimitiveOperation::Allocation,
                role: None,
                dtype: None,
                layout: None,
            })
            .is_err()
        );
    }

    #[test]
    fn canonical_matrix_owns_deterministic_backend_negotiation() {
        let unary =
            OperationSupport::unary_input(UnaryOperation::Absolute, DType::F32, Layout::Contiguous);
        let binary =
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        let available =
            BackendCapabilityMatrix::new(DeviceId::CPU, vec![unary, binary], vec![unary, binary])
                .expect("valid available matrix");
        let requested = BackendCapabilityMatrix::new(DeviceId::CPU, vec![unary], vec![])
            .expect("valid requested matrix");
        let negotiated = available
            .negotiate(&requested)
            .expect("same-device matrices negotiate");
        assert_eq!(negotiated.supported(), [unary]);
        assert!(negotiated.deterministic().is_empty());
        assert!(negotiated.is_subset_of(&available));
        assert!(negotiated.is_subset_of(&requested));
        assert!(
            available
                .negotiate(
                    &BackendCapabilityMatrix::new(
                        DeviceId::new(sim_types::DeviceKind::Cuda, 0),
                        vec![unary],
                        vec![],
                    )
                    .expect("valid CUDA request")
                )
                .is_err()
        );
    }

    #[test]
    fn canonical_matrix_is_the_only_native_backend_readiness_owner() {
        for device_kind in DeviceKind::ALL {
            let result = BackendCapabilityMatrix::for_native_device(DeviceId::new(device_kind, 0));
            if device_kind == DeviceKind::Cpu && cfg!(feature = "cpu") {
                let matrix = result.expect("the compiled CPU backend has a canonical matrix");
                assert_eq!(matrix.device(), DeviceId::CPU);
                assert!(!matrix.supported().is_empty());
            } else {
                let unavailable = result.expect_err("unsupported devices cannot advertise support");
                assert_eq!(unavailable.device(), device_kind);
                assert!(!unavailable.reason().is_empty());
            }
        }

        let nonzero_cpu =
            BackendCapabilityMatrix::for_native_device(DeviceId::new(DeviceKind::Cpu, 1))
                .expect_err("an unregistered CPU ordinal is unavailable");
        assert_eq!(nonzero_cpu.device(), DeviceKind::Cpu);
        assert!(nonzero_cpu.reason().contains("canonical capability matrix"));
    }

    #[test]
    fn worker_readiness_matrix_requires_the_executed_deterministic_add() {
        let device = DeviceId::new(DeviceKind::DirectMl, 0);
        let matrix = BackendCapabilityMatrix::worker_readiness_requirements(device)
            .expect("readiness requirements are valid");
        let add_input =
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        let add_output =
            OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous);
        assert_eq!(matrix.device(), device);
        assert_eq!(matrix.supported().len(), 7);
        assert_eq!(matrix.deterministic(), [add_input, add_output]);
        assert!(matrix.supports(add_input));
        assert!(matrix.supports(add_output));
    }

    #[test]
    fn optional_backend_adapters_report_binding_status_without_support_claims() {
        for device_kind in DeviceKind::ALL
            .into_iter()
            .filter(|kind| *kind != DeviceKind::Cpu)
        {
            let status = native_backend_binding_status(device_kind);
            assert_eq!(status.device(), device_kind);
            assert!(matches!(status, NativeBackendBindingStatus::Unbound { .. }));
        }
    }

    #[test]
    fn val_device_001() -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest as _, Sha256};
        use std::{collections::BTreeMap, fs, path::Path};

        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let mut cases = BTreeMap::new();
        for device_kind in DeviceKind::ALL {
            let status = native_backend_binding_status(device_kind);
            if status.device() != device_kind {
                return Err(format!(
                    "{device_kind:?} binding reported mismatched device {:?}",
                    status.device()
                )
                .into());
            }
            let feature_enabled = device_feature_enabled(device_kind);
            let capability =
                BackendCapabilityMatrix::for_native_device(DeviceId::new(device_kind, 0));
            let (binding, reason, passed) = match (device_kind, feature_enabled, status, capability)
            {
                (DeviceKind::Cpu, true, NativeBackendBindingStatus::Bound { .. }, Ok(matrix)) => (
                    "bound",
                    None,
                    matrix.device() == DeviceId::CPU && !matrix.supported().is_empty(),
                ),
                (
                    DeviceKind::Cpu,
                    false,
                    NativeBackendBindingStatus::Unbound { reason, .. },
                    Err(unavailable),
                ) => (
                    "feature_disabled",
                    Some(reason),
                    unavailable.device() == DeviceKind::Cpu,
                ),
                (_, _, NativeBackendBindingStatus::Unbound { reason, .. }, Err(unavailable)) => (
                    if feature_enabled {
                        "adapter_unbound"
                    } else {
                        "feature_disabled"
                    },
                    Some(reason),
                    unavailable.device() == device_kind,
                ),
                (_, _, binding, capability) => {
                    return Err(format!(
                        "{device_kind:?} exposed inconsistent binding {binding:?} and capability {capability:?}"
                    )
                    .into());
                }
            };
            if !passed {
                return Err(format!("{device_kind:?} device validation failed").into());
            }
            cases.insert(
                format!("{:?}", device_kind).to_ascii_lowercase(),
                serde_json::json!({
                    "passed": true,
                    "feature_enabled": feature_enabled,
                    "binding": binding,
                    "reason": reason,
                }),
            );
        }

        let ledger_path = workspace
            .join("crates/sim_test_support/parity/catalogs/native-backend-dependencies.json");
        let ledger_bytes = fs::read(&ledger_path)?;
        let lock_bytes = fs::read(workspace.join("Cargo.lock"))?;
        let mut fixture_digests = BTreeMap::from([
            (
                "native_backend_dependencies".to_owned(),
                format!("{:x}", Sha256::digest(&ledger_bytes)),
            ),
            (
                "cargo_lock".to_owned(),
                format!("{:x}", Sha256::digest(&lock_bytes)),
            ),
        ]);
        if cfg!(feature = "rocm") {
            for relative in [
                "crates/sim_backend_rocm/abi/symbols-v1.json",
                "crates/sim_backend_rocm/abi/reviewed-bindings-v1.txt",
                "crates/sim_backend_rocm/abi/verify-completion-evidence.sh",
                "crates/sim_backend_rocm/build.rs",
                "crates/sim_backend_rocm/src/loader.rs",
                "crates/sim_backend_rocm/src/sim_backend_rocm.rs",
                "crates/sim_tensor/src/backends/amd_rocm_sim_model_0014.rs",
                "crates/sim_tensor/src/ops/backend_amd_rocm_sim_model_0014.rs",
                "crates/sim_runtime/src/native_ffi_rocm.rs",
                "crates/sim_runtime/src/trust.rs",
                "script/package-sim-backend-rocm",
                "nix/sim-backends/rocm/package-policy.json",
            ] {
                fixture_digests.insert(
                    relative.to_owned(),
                    format!("{:x}", Sha256::digest(fs::read(workspace.join(relative))?)),
                );
            }
        }
        if cfg!(feature = "metal") {
            for relative in [
                "crates/sim_backend_metal/abi/symbols-v1.json",
                "crates/sim_backend_metal/abi/reviewed-bindings-v1.txt",
                "crates/sim_backend_metal/build.rs",
                "crates/sim_backend_metal/src/abi.rs",
                "crates/sim_backend_metal/src/loader.rs",
                "crates/sim_backend_metal/src/sim_backend_metal.rs",
                "crates/sim_backend_metal/src/execution.rs",
                "crates/sim_backend_metal/src/execution_abi.rs",
                "crates/sim_backend_metal/kernels/readiness.metal",
                "crates/sim_backend_metal/kernels/tensor_ops.metal",
                "crates/sim_backend_metal/abi/execution-v1.json",
                "crates/sim_backend_metal/abi/reviewed-execution-bindings-v1.txt",
                "crates/sim_backend_metal/LICENSES",
                "crates/sim_runtime/src/native_ffi_metal.rs",
                "crates/sim_runtime/src/trust.rs",
                "script/package-sim-backend-metal",
                "script/package-sim-backend-metal-execution",
                "nix/sim-backends/metal/package-policy.json",
                "nix/sim-backends/metal/execution-policy.json",
                "nix/sim-backends/metal/ffi-contracts-v1.schema.json",
                "nix/sim-backends/metal/default.nix",
                "crates/sim_test_support/parity/catalogs/native-backend-abi/metal.json",
                "crates/sim_test_support/parity/catalogs/native-backend-abi/metal-execution.json",
            ] {
                fixture_digests.insert(
                    relative.to_owned(),
                    format!("{:x}", Sha256::digest(fs::read(workspace.join(relative))?)),
                );
            }
        }
        let artifact = serde_json::json!({
            "validation_id": "VAL-DEVICE-001",
            "validation": "VAL-DEVICE-001",
            "scope": "canonical-native-device-binding-and-capability-foundation",
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "backend": "native-rust",
            },
            "fixture_digests": fixture_digests,
            "cases": cases,
            "summary": {
                "passed": DeviceKind::ALL.len(),
                "failed": 0,
                "skipped": 0,
            },
            "skipped": [],
            "foundation_closure": {
                "rocm": {
                    "claimed": cfg!(feature = "rocm"),
                    "stage": if cfg!(feature = "rocm") {
                        "certified_abi_and_native_adapter"
                    } else {
                        "feature_disabled"
                    },
                    "release_ready": false,
                    "remaining_gate": "production selection integration and hardware certification",
                },
                "metal": {
                    "claimed": cfg!(feature = "metal"),
                    "stage": if cfg!(feature = "metal") {
                        "certified_abi_and_native_semantic_adapter_unbound"
                    } else {
                        "feature_disabled"
                    },
                    "release_ready": false,
                    "remaining_gate": "signed runtime certification, production selection integration, and hardware certification",
                },
            },
            "execution_resource_closure": {
                "metal": {
                    "claimed": cfg!(feature = "metal"),
                    "stage": if cfg!(feature = "metal") {
                        "opaque_runtime_and_precompiled_arithmetic_metallib_unbound"
                    } else {
                        "feature_disabled"
                    },
                    "runtime_compilation": false,
                    "global_availability": false,
                    "planned_adapter_capability_rows": if cfg!(feature = "metal") { 12 } else { 0 },
                    "remaining_gate": "NativeFfiRegistry-certified signed package mapping and production worker selection",
                },
            },
            "adapter_closure": {
                "rocm": {
                    "claimed": cfg!(feature = "rocm"),
                    "stage": if cfg!(feature = "rocm") {
                        "implemented_but_not_globally_bound"
                    } else {
                        "feature_disabled"
                    },
                    "instance_derived": true,
                    "advertised_primitive_rows": if cfg!(feature = "rocm") { 7 } else { 0 },
                    "production_integration_pending": true,
                    "hardware_certification_pending": true,
                },
                "metal": {
                    "claimed": cfg!(feature = "metal"),
                    "stage": if cfg!(feature = "metal") {
                        "implemented_but_not_globally_bound"
                    } else {
                        "feature_disabled"
                    },
                    "instance_derived": true,
                    "advertised_primitive_rows": if cfg!(feature = "metal") { 12 } else { 0 },
                    "production_integration_pending": true,
                    "hardware_certification_pending": true,
                },
            },
            "release_closure": {
                "claimed": false,
                "stage": "vendor_dependency_stub_foundation",
                "reason": "Vendor ABI binding, certified backend implementation, and hardware certification remain assigned to their pending executable tasks.",
                "remaining_gates": [
                    "native device ABI binding and packaging tasks",
                    "native device adapter implementation tasks",
                    "hardware certification tasks",
                ],
            },
        });
        let artifact_directory = workspace.join("target/sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        let artifact_path = artifact_directory.join("val-device-001.json");
        let temporary_path = artifact_directory.join("val-device-001.json.tmp");
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        fs::write(&temporary_path, bytes)?;
        fs::rename(temporary_path, artifact_path)?;
        Ok(())
    }

    const fn device_feature_enabled(device: DeviceKind) -> bool {
        match device {
            DeviceKind::Cpu => cfg!(feature = "cpu"),
            DeviceKind::Cuda => cfg!(feature = "cuda"),
            DeviceKind::Rocm => cfg!(feature = "rocm"),
            DeviceKind::Metal => cfg!(feature = "metal"),
            DeviceKind::DirectMl => cfg!(feature = "directml"),
            DeviceKind::Xpu => cfg!(feature = "xpu"),
            DeviceKind::Npu => cfg!(feature = "npu"),
            DeviceKind::Mlu => cfg!(feature = "mlu"),
            DeviceKind::CoreX => cfg!(feature = "corex"),
        }
    }
}
