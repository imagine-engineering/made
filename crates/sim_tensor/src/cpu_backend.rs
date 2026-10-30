use crate::generated_elementwise_or_runtime_operation_04::{
    ElementwiseRuntimePartFourError, isfinite_with_context_exact_native,
};
use crate::generated_elementwise_or_runtime_operation_21::{
    ElementwiseRuntimePartTwentyOneError, kron_with_context_exact_native,
};
use crate::generated_linear_algebra_01::{
    LinearAlgebraPartOneError, inverse_with_context_exact_native,
    vector_norm_with_context_exact_native,
};
use crate::generated_shape_layout_transform_03::{
    FunctionalPadMode, ShapeLayoutTransformPartThreeError, functional_pad_with_context_exact_native,
};
use crate::generated_sim_operator_indirection_01::{
    ConvolutionGeometry, OperatorIndirectionError, cast_to_with_backend_exact_native,
    convolution_into_with_context_exact_native, tensor_from_f32_with_backend_exact_native,
};
use crate::generated_tensor_creation_01::{
    TensorCreationPartOneError, eye_with_context_exact_native,
};
use crate::{
    BackendCapabilityMatrix, BackendMemoryReservation, BackendMemorySnapshot, BackendMemoryTracker,
    BackendStorage, BackendWorkspaceAuthority, BackendWorkspaceLease, BinaryOperation,
    CachedAllocationOwner, CancellationToken, ConvolutionSpec, CpuStorage, CustomKernelId, DType,
    DecodedScalar, DeviceId, EventFence, ExecutionContext, IndexSpec, Layout,
    LinearAlgebraOperation, NativeDeviceProperties, OperationSupport, PrimitiveOperation,
    ReductionOperation, ReductionSpec, ResizeCrop, ResizeMode, ResizeSpec, Scalar, ScalarSide,
    ScratchReservation, Tensor, TensorBackend, TensorDescriptor, TensorError, TensorRole,
    TensorWrite, UnaryOperation, ViewAccess, check_backend_context, normalize_narrow_range,
    required_storage_bytes, reserve_backend_workspace, validate_inputs,
};
use std::{
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

const DTYPES: [DType; 20] = [
    DType::F64,
    DType::F32,
    DType::F16,
    DType::Bf16,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
    DType::Complex64,
    DType::Complex128,
    DType::Float8E4m3Fn,
    DType::Float8E5m2,
    DType::Float8E4m3Fnuz,
    DType::Float8E5m2Fnuz,
    DType::Float8E8m0Fnu,
];
const LAYOUTS: [Layout; 4] = [
    Layout::Contiguous,
    Layout::ChannelsLast,
    Layout::ChannelsLast3d,
    Layout::Strided,
];
const UNARY_OPERATIONS: [UnaryOperation; 22] = [
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
    UnaryOperation::Round,
    UnaryOperation::Sinc,
    UnaryOperation::Log1p,
    UnaryOperation::ReciprocalSquareRoot,
    UnaryOperation::Relu,
    UnaryOperation::IsFinite,
    UnaryOperation::InvertUnitInterval,
    UnaryOperation::LogarithmBaseTwo,
    UnaryOperation::Signum,
    UnaryOperation::Tangent,
    UnaryOperation::ArcTangent,
    UnaryOperation::ArcHyperbolicTangent,
];
const BINARY_OPERATIONS: [BinaryOperation; 18] = [
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
    BinaryOperation::FloatingRemainder,
    BinaryOperation::Atan2,
    BinaryOperation::LogAddExp,
];
const REDUCTION_OPERATIONS: [ReductionOperation; 11] = [
    ReductionOperation::Sum,
    ReductionOperation::Product,
    ReductionOperation::Mean,
    ReductionOperation::Minimum,
    ReductionOperation::Maximum,
    ReductionOperation::ArgMinimum,
    ReductionOperation::ArgMaximum,
    ReductionOperation::All,
    ReductionOperation::Any,
    ReductionOperation::Variance,
    ReductionOperation::StandardDeviation,
];
const RESIZE_MODES: [ResizeMode; 5] = [
    ResizeMode::NearestExact,
    ResizeMode::Bilinear,
    ResizeMode::Area,
    ResizeMode::Bicubic,
    ResizeMode::Lanczos,
];

pub type CpuMemorySnapshot = BackendMemorySnapshot;
pub type CpuWorkspaceAuthority = BackendWorkspaceAuthority;
pub type CpuWorkspaceLease = BackendWorkspaceLease;

#[derive(Debug)]
pub struct CpuWorkspaceVec<T> {
    values: Vec<T>,
    element_limit: usize,
    _lease: CpuWorkspaceLease,
}

impl<T> Deref for CpuWorkspaceVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.values.as_slice()
    }
}

impl<T> DerefMut for CpuWorkspaceVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.as_mut_slice()
    }
}

impl<T> CpuWorkspaceVec<T> {
    pub fn capacity(&self) -> usize {
        self.element_limit
    }

    pub fn try_push(&mut self, value: T) -> Result<(), TensorError> {
        if self.values.len() == self.element_limit {
            let element_bytes =
                u64::try_from(size_of::<T>()).map_err(|_| TensorError::ShapeOverflow)?;
            return Err(TensorError::WorkspaceAuthorizationExceeded {
                requested: element_bytes,
                authorized: self._lease.bytes(),
                in_use: self._lease.bytes(),
            });
        }
        self.values.push(value);
        Ok(())
    }
}

#[derive(Debug)]
struct TrackedCpuStorage {
    storage: CpuStorage,
    logical_bytes: u64,
    reservation: BackendMemoryReservation,
}

impl TrackedCpuStorage {
    fn zeroed(tracker: Arc<BackendMemoryTracker>, bytes: u64) -> Result<Self, TensorError> {
        let reserved = aligned_allocation_bytes(bytes)?;
        let reservation = tracker.reserve(reserved)?;
        let length = match usize::try_from(bytes) {
            Ok(length) => length,
            Err(_) => return Err(TensorError::ShapeOverflow),
        };
        match CpuStorage::zeroed(length) {
            Ok(storage) => Ok(Self {
                storage,
                logical_bytes: bytes,
                reservation,
            }),
            Err(error) => Err(error),
        }
    }
}

impl BackendStorage for TrackedCpuStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn byte_len(&self) -> u64 {
        self.logical_bytes
    }

    fn clone_for_write(&self) -> Result<Box<dyn BackendStorage>, TensorError> {
        let reservation = self
            .reservation
            .tracker()
            .reserve(self.reservation.bytes())?;
        match self.storage.try_clone() {
            Ok(storage) => Ok(Box::new(Self {
                storage,
                logical_bytes: self.logical_bytes,
                reservation,
            })),
            Err(error) => Err(error),
        }
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        self.storage.host_bytes()
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.storage.host_bytes_mut()
    }
}

fn aligned_allocation_bytes(logical_bytes: u64) -> Result<u64, TensorError> {
    const ALIGNMENT: u64 = 16;
    if logical_bytes == 0 {
        return Ok(0);
    }
    logical_bytes
        .checked_add(ALIGNMENT - 1)
        .map(|bytes| bytes / ALIGNMENT * ALIGNMENT)
        .ok_or(TensorError::ShapeOverflow)
}

#[derive(Debug)]
pub struct CpuBackend {
    capabilities: BackendCapabilityMatrix,
    backend_id: u64,
    event_sequence: AtomicU64,
    memory: Arc<BackendMemoryTracker>,
}

impl CpuBackend {
    pub fn validate_scratch_reservation(
        &self,
        scratch: &ScratchReservation,
    ) -> Result<(), TensorError> {
        let binding = scratch.binding_identity();
        if binding.backend_id != self.backend_id || binding.authority_id == 0 {
            return Err(TensorError::WorkspaceAuthorizationMismatch {
                expected_backend: self.backend_id,
                expected_authority: 0,
                actual_backend: binding.backend_id,
                actual_authority: binding.authority_id,
            });
        }
        Ok(())
    }

    pub fn capability_matrix() -> BackendCapabilityMatrix {
        BackendCapabilityMatrix::all_deterministic(DeviceId::CPU, Self::supported_capabilities())
    }

    fn instance_capability_matrix(
        memory_limit_bytes: u64,
    ) -> Result<BackendCapabilityMatrix, TensorError> {
        let properties = NativeDeviceProperties::new(
            DeviceId::CPU,
            "Zed native Rust CPU",
            memory_limit_bytes,
            0,
            0,
            Some(std::env::consts::ARCH.to_owned()),
            false,
        )?;
        let supported = Self::supported_capabilities();
        BackendCapabilityMatrix::new_with_properties(
            DeviceId::CPU,
            supported.clone(),
            supported,
            Some(properties),
        )
    }

    fn supported_capabilities() -> Vec<OperationSupport> {
        let mut supported = Vec::new();
        for dtype in DTYPES {
            for layout in LAYOUTS {
                supported.push(OperationSupport::allocation(dtype, layout));
                supported.push(OperationSupport::copy_input(dtype, layout));
                supported.push(OperationSupport::copy_output(dtype, layout));
                supported.push(OperationSupport::select_input(dtype, layout));
                supported.push(OperationSupport::select_output(dtype, layout));
                supported.push(OperationSupport::narrow_input(dtype, layout));
                supported.push(OperationSupport::narrow_output(dtype, layout));
                if dtype != DType::Float8E8m0Fnu {
                    supported.push(OperationSupport::fill(dtype, layout));
                }
            }
        }
        for layout in LAYOUTS {
            for operation in UNARY_OPERATIONS {
                let input_dtypes: &[DType] = if operation == UnaryOperation::HyperbolicTangent {
                    &[DType::F32, DType::F16, DType::Bf16]
                } else {
                    &[DType::F32]
                };
                for dtype in input_dtypes {
                    supported.push(OperationSupport::unary_input(operation, *dtype, layout));
                    supported.push(OperationSupport::unary_output(
                        operation,
                        if operation == UnaryOperation::IsFinite {
                            DType::Bool
                        } else {
                            *dtype
                        },
                        layout,
                    ));
                }
            }
            for operation in BINARY_OPERATIONS {
                let output_dtype = binary_output_dtype(operation);
                supported.push(OperationSupport::binary_input(
                    operation,
                    DType::F32,
                    layout,
                ));
                supported.push(OperationSupport::binary_output(
                    operation,
                    output_dtype,
                    layout,
                ));
                supported.push(OperationSupport::binary_scalar_input(
                    operation,
                    DType::F32,
                    layout,
                ));
                supported.push(OperationSupport::binary_scalar_output(
                    operation,
                    output_dtype,
                    layout,
                ));
            }
            for operation in REDUCTION_OPERATIONS {
                for dtype in DTYPES {
                    if reduction_supports_input(operation, dtype) {
                        supported.push(OperationSupport::reduction_input(operation, dtype, layout));
                    }
                    if reduction_supports_output(operation, dtype) {
                        supported
                            .push(OperationSupport::reduction_output(operation, dtype, layout));
                    }
                }
            }
            if layout != Layout::ChannelsLast3d {
                for mode in RESIZE_MODES {
                    supported.push(OperationSupport::resize_input(mode, DType::F32, layout));
                    supported.push(OperationSupport::resize_output(mode, DType::F32, layout));
                }
            }
            for dtype in [DType::F32, DType::F16, DType::Bf16] {
                supported.push(OperationSupport::convolution_input(dtype, layout));
                supported.push(OperationSupport::convolution_output(dtype, layout));
            }
        }
        for layout in [Layout::Contiguous, Layout::Strided] {
            for dtype in [DType::F32, DType::F16, DType::Bf16] {
                supported.push(OperationSupport::linear_algebra_input(
                    LinearAlgebraOperation::BatchMatrixMultiply,
                    dtype,
                    layout,
                ));
                supported.push(OperationSupport::linear_algebra_output(
                    LinearAlgebraOperation::BatchMatrixMultiply,
                    dtype,
                    layout,
                ));
            }
        }
        supported.push(OperationSupport::record_event());
        supported.push(OperationSupport::wait_event());
        supported
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.memory.snapshot()
    }

    pub fn execution_context<'a>(
        &self,
        stream: crate::StreamId,
        scratch: crate::ScratchReservation,
        cancellation: &'a CancellationToken,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            stream,
            scratch,
            rng_phase: None,
            cancellation,
        }
    }

    pub fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<CpuWorkspaceLease, TensorError> {
        let reserved_bytes = match aligned_allocation_bytes(requested) {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        reserve_backend_workspace(
            self.backend_id,
            &self.memory,
            context,
            requested,
            reserved_bytes,
        )
    }

    pub fn workspace_vec<T>(
        &self,
        context: &ExecutionContext<'_>,
        capacity: usize,
    ) -> Result<CpuWorkspaceVec<T>, TensorError> {
        let requested = u64::try_from(capacity)
            .ok()
            .and_then(|capacity| {
                u64::try_from(size_of::<T>())
                    .ok()
                    .and_then(|width| capacity.checked_mul(width))
            })
            .ok_or(TensorError::ShapeOverflow)?;
        let lease = self.reserve_workspace(context, requested)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|error| TensorError::AllocationFailed {
                requested,
                reason: format!("CPU workspace allocation failed: {error}"),
            })?;
        context.check()?;
        Ok(CpuWorkspaceVec {
            values,
            element_limit: capacity,
            _lease: lease,
        })
    }

    pub fn upload_bytes(
        &self,
        descriptor: TensorDescriptor,
        bytes: &[u8],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let expected = required_storage_bytes(&descriptor)?;
        let actual = u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
        if actual != expected {
            return Err(TensorError::StorageLength { expected, actual });
        }
        let mut tensor = self.allocate_tensor("zed.cpu.upload", descriptor, context)?;
        {
            let mut write = tensor.write()?;
            let destination = write.storage_bytes_mut()?;
            for (destination, source) in destination
                .chunks_mut(64 * 1024)
                .zip(bytes.chunks(64 * 1024))
            {
                context.check()?;
                destination.copy_from_slice(source);
            }
        }
        self.finish(tensor, context)
    }

    pub fn upload_f32(
        &self,
        descriptor: TensorDescriptor,
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        if descriptor.dtype() != DType::F32 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                actual: descriptor.dtype(),
            });
        }
        let expected =
            usize::try_from(descriptor.element_count()?).map_err(|_| TensorError::ShapeOverflow)?;
        if values.len() != expected {
            return Err(TensorError::StorageLength {
                expected: descriptor.byte_len()?,
                actual: u64::try_from(values.len())
                    .ok()
                    .and_then(|length| length.checked_mul(4))
                    .ok_or(TensorError::ShapeOverflow)?,
            });
        }
        let mut tensor = self.allocate_tensor("zed.cpu.upload-f32", descriptor, context)?;
        {
            let mut write = tensor.write()?;
            for (destination, value) in write
                .bytes_mut()?
                .chunks_exact_mut(std::mem::size_of::<f32>())
                .zip(values)
            {
                context.check()?;
                destination.copy_from_slice(&value.to_ne_bytes());
            }
        }
        self.finish(tensor, context)
    }

    pub fn resize_vjp(
        &self,
        operation: ResizeSpec,
        input: &Tensor,
        output_gradient: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, TensorError> {
        context.check()?;
        require_dtype(DType::F32, input.descriptor().dtype())?;
        require_dtype(DType::F32, output_gradient.descriptor().dtype())?;
        if operation.antialias
            || !matches!(
                operation.mode,
                ResizeMode::NearestExact | ResizeMode::Bilinear
            )
        {
            return Err(self.unsupported(
                "zed.cpu.resize-vjp",
                "resize VJP is certified for non-antialiased nearest-exact and bilinear modes",
            ));
        }
        ResizeGeometry::new(input.descriptor(), output_gradient.descriptor(), operation)?;
        let input_count = usize::try_from(input.descriptor().element_count()?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let mut input_gradient = Vec::new();
        let requested = u64::try_from(input_count)
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or(TensorError::ShapeOverflow)?;
        input_gradient
            .try_reserve_exact(input_count)
            .map_err(|error| TensorError::AllocationFailed {
                requested,
                reason: format!("resize VJP allocation failed: {error}"),
            })?;
        input_gradient.resize(input_count, 0.0_f32);
        let [batch_count, channel_count, input_height, input_width] = input.descriptor().shape()
        else {
            return Err(TensorError::Faulted {
                reason: "resize VJP input rank changed after validation".to_owned(),
            });
        };
        let shape = output_gradient.descriptor().shape().to_vec();
        for_each_index(&shape, context.cancellation, |indices| {
            let [batch, channel, output_y, output_x] = indices else {
                return Err(TensorError::Faulted {
                    reason: "resize VJP output rank changed after validation".to_owned(),
                });
            };
            let gradient = read_f32(output_gradient, indices)?;
            let mut accumulate = |input_y: u64, input_x: u64, weight: f32| {
                let index = batch
                    .checked_mul(*channel_count)
                    .and_then(|value| value.checked_add(*channel))
                    .and_then(|value| value.checked_mul(*input_height))
                    .and_then(|value| value.checked_add(input_y))
                    .and_then(|value| value.checked_mul(*input_width))
                    .and_then(|value| value.checked_add(input_x))
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(TensorError::ShapeOverflow)?;
                let destination = input_gradient
                    .get_mut(index)
                    .ok_or(TensorError::ShapeOverflow)?;
                *destination = gradient.mul_add(weight, *destination);
                Ok::<_, TensorError>(())
            };
            match operation.mode {
                ResizeMode::NearestExact => {
                    let input_y = nearest_exact(*output_y, *input_height, operation.height)?;
                    let input_x = nearest_exact(*output_x, *input_width, operation.width)?;
                    accumulate(input_y, input_x, 1.0)?;
                }
                ResizeMode::Bilinear => {
                    let (y0, y1, y_weight) = linear_coordinates(
                        *output_y,
                        *input_height,
                        operation.height,
                        operation.align_corners,
                    )?;
                    let (x0, x1, x_weight) = linear_coordinates(
                        *output_x,
                        *input_width,
                        operation.width,
                        operation.align_corners,
                    )?;
                    accumulate(y0, x0, (1.0 - y_weight) * (1.0 - x_weight))?;
                    accumulate(y0, x1, (1.0 - y_weight) * x_weight)?;
                    accumulate(y1, x0, y_weight * (1.0 - x_weight))?;
                    accumulate(y1, x1, y_weight * x_weight)?;
                }
                ResizeMode::Area | ResizeMode::Bicubic | ResizeMode::Lanczos => {
                    return Err(TensorError::Faulted {
                        reason: "resize VJP mode changed after validation".to_owned(),
                    });
                }
            }
            Ok(())
        })?;
        context.check()?;
        let descriptor = TensorDescriptor::contiguous(
            vec![*batch_count, *channel_count, *input_height, *input_width],
            DType::F32,
            input.descriptor().device(),
            input.descriptor().stream(),
        )?;
        self.upload_f32(descriptor, &input_gradient, context)
            .map(|(tensor, _)| tensor)
    }

    fn require_descriptor(
        &self,
        operation: &str,
        primitive: PrimitiveOperation,
        role: TensorRole,
        descriptor: &TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        context.check()?;
        if descriptor.device() != DeviceId::CPU {
            return Err(TensorError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual: descriptor.device(),
            });
        }
        if descriptor.stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: descriptor.stream(),
            });
        }
        self.capabilities.require(
            operation,
            OperationSupport::for_tensor(primitive, role, descriptor.dtype(), descriptor.layout())?,
        )
    }

    fn allocate_tensor(
        &self,
        operation: &str,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, TensorError> {
        self.require_descriptor(
            operation,
            PrimitiveOperation::Allocation,
            TensorRole::Output,
            &descriptor,
            context,
        )?;
        let bytes = required_storage_bytes(&descriptor)?;
        check_backend_context(self.backend_id, context)?;
        let storage = TrackedCpuStorage::zeroed(self.memory.clone(), bytes)?;
        Tensor::from_backend_storage(descriptor, Box::new(storage), ViewAccess::Writable)
    }

    fn unsupported(&self, operation: &str, reason: impl Into<String>) -> TensorError {
        TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: DeviceId::CPU,
            reason: reason.into(),
        }
    }

    fn finish(
        &self,
        tensor: Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }
}

impl BackendWorkspaceAuthority {
    pub fn create_backend(
        memory_limit_bytes: u64,
    ) -> Result<(CpuBackend, BackendWorkspaceAuthority), TensorError> {
        let capabilities = CpuBackend::instance_capability_matrix(memory_limit_bytes)?;
        let (backend_id, memory, authority) = Self::new(memory_limit_bytes)?;
        Ok((
            CpuBackend {
                capabilities,
                backend_id,
                event_sequence: AtomicU64::new(0),
                memory,
            },
            authority,
        ))
    }
}

impl CachedAllocationOwner for CpuBackend {
    fn cache_device(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for CpuBackend {
    fn cpu_backend(&self) -> Option<&CpuBackend> {
        Some(self)
    }

    fn device(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn capabilities(&self) -> &BackendCapabilityMatrix {
        &self.capabilities
    }

    fn reserve_workspace(
        &self,
        context: &ExecutionContext<'_>,
        requested: u64,
    ) -> Result<BackendWorkspaceLease, TensorError> {
        CpuBackend::reserve_workspace(self, context, requested)
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let tensor = self.allocate_tensor("zed.cpu.allocate", descriptor, context)?;
        self.finish(tensor, context)
    }

    fn copy(
        &self,
        source: &Tensor,
        destination: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        validate_inputs(
            self,
            "zed.cpu.copy",
            PrimitiveOperation::Copy,
            std::slice::from_ref(source),
            context,
        )?;
        self.require_descriptor(
            "zed.cpu.copy",
            PrimitiveOperation::Copy,
            TensorRole::Output,
            &destination,
            context,
        )?;
        require_same_shape(source.descriptor().shape(), destination.shape())?;
        require_dtype(source.descriptor().dtype(), destination.dtype())?;
        let mut tensor = self.allocate_tensor("zed.cpu.copy", destination, context)?;
        copy_logical(source, &mut tensor, context.cancellation)?;
        self.finish(tensor, context)
    }

    fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        context.check()?;
        self.capabilities
            .require("zed.cpu.event.record", OperationSupport::record_event())?;
        let previous = self
            .event_sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| TensorError::IdentifierOverflow)?;
        let sequence = previous
            .checked_add(1)
            .ok_or(TensorError::IdentifierOverflow)?;
        Ok(EventFence {
            backend_id: self.backend_id,
            device: DeviceId::CPU,
            stream: context.stream,
            sequence,
        })
    }

    fn wait_event(
        &self,
        event: EventFence,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        context.check()?;
        self.capabilities
            .require("zed.cpu.event.wait", OperationSupport::wait_event())?;
        if event.backend_id != self.backend_id {
            return Err(TensorError::Faulted {
                reason: "CPU event belongs to a different backend instance".to_owned(),
            });
        }
        if event.device != DeviceId::CPU {
            return Err(TensorError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual: event.device,
            });
        }
        if event.stream != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: event.stream,
            });
        }
        if event.sequence == 0 {
            return Err(TensorError::Faulted {
                reason: "CPU event sequence zero is invalid".to_owned(),
            });
        }
        let recorded = self.event_sequence.load(Ordering::Acquire);
        if event.sequence > recorded {
            return Err(TensorError::Faulted {
                reason: format!(
                    "CPU event {} has not been recorded; latest sequence is {recorded}",
                    event.sequence
                ),
            });
        }
        Ok(())
    }

    fn fill(
        &self,
        value: Scalar,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.require_descriptor(
            "zed.cpu.fill",
            PrimitiveOperation::Fill,
            TensorRole::Output,
            &output,
            context,
        )?;
        let encoded = output
            .dtype()
            .encode_scalar(value, "zed.cpu.fill", DeviceId::CPU)
            .map_err(|error| match error {
                TensorError::UnsupportedCapability { reason, .. } => {
                    self.unsupported("zed.cpu.fill", reason)
                }
                other => other,
            })?;
        let shape = output.shape().to_vec();
        let mut tensor = self.allocate_tensor("zed.cpu.fill", output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            write_element(&mut write, indices, &encoded)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn unary(
        &self,
        operation: UnaryOperation,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let operation_label = unary_operation_label(operation);
        validate_inputs(
            self,
            operation_label,
            PrimitiveOperation::Unary(operation),
            std::slice::from_ref(input),
            context,
        )?;
        self.require_descriptor(
            operation_label,
            PrimitiveOperation::Unary(operation),
            TensorRole::Output,
            &output,
            context,
        )?;
        let input_dtype = input.descriptor().dtype();
        let supports_low_precision = operation == UnaryOperation::HyperbolicTangent
            && matches!(input_dtype, DType::F16 | DType::Bf16);
        if input_dtype != DType::F32 && !supports_low_precision {
            return Err(self.unsupported(
                operation_label,
                "the reference unary kernel accepts f32 input, plus f16/bf16 for tanh",
            ));
        }
        require_same_shape(input.descriptor().shape(), output.shape())?;
        let expected_dtype = if operation == UnaryOperation::IsFinite {
            DType::Bool
        } else {
            input_dtype
        };
        require_dtype(expected_dtype, output.dtype())?;
        let shape = output.shape().to_vec();
        let output_dtype = output.dtype();
        let mut tensor = self.allocate_tensor(operation_label, output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let value = read_real_f32(input, indices)?;
            if operation == UnaryOperation::IsFinite {
                write_element(&mut write, indices, &[u8::from(value.is_finite())])
            } else {
                let result = apply_unary_scalar(operation, value);
                let encoded = output_dtype.encode_scalar(
                    Scalar::Float(f64::from(result)),
                    operation_label,
                    DeviceId::CPU,
                )?;
                write_element(&mut write, indices, &encoded)
            }
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn binary(
        &self,
        operation: BinaryOperation,
        left: &Tensor,
        right: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let operation_label = binary_operation_label(operation, false);
        validate_inputs(
            self,
            operation_label,
            PrimitiveOperation::Binary(operation),
            &[left.clone(), right.clone()],
            context,
        )?;
        self.require_descriptor(
            operation_label,
            PrimitiveOperation::Binary(operation),
            TensorRole::Output,
            &output,
            context,
        )?;
        require_dtype(DType::F32, left.descriptor().dtype())?;
        require_dtype(DType::F32, right.descriptor().dtype())?;
        let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
        require_same_shape(&shape, output.shape())?;
        require_dtype(binary_output_dtype(operation), output.dtype())?;
        let mut tensor = self.allocate_tensor(operation_label, output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let left_indices = broadcast_indices(indices, left.descriptor().shape())?;
            let right_indices = broadcast_indices(indices, right.descriptor().shape())?;
            let left_value = read_f32(left, &left_indices)?;
            let right_value = read_f32(right, &right_indices)?;
            write_binary(&mut write, indices, operation, left_value, right_value)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn binary_scalar(
        &self,
        operation: BinaryOperation,
        input: &Tensor,
        scalar: Scalar,
        scalar_side: ScalarSide,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let operation_label = binary_operation_label(operation, true);
        validate_inputs(
            self,
            operation_label,
            PrimitiveOperation::BinaryScalar(operation),
            std::slice::from_ref(input),
            context,
        )?;
        self.require_descriptor(
            operation_label,
            PrimitiveOperation::BinaryScalar(operation),
            TensorRole::Output,
            &output,
            context,
        )?;
        require_dtype(DType::F32, input.descriptor().dtype())?;
        require_same_shape(input.descriptor().shape(), output.shape())?;
        require_dtype(binary_output_dtype(operation), output.dtype())?;
        let scalar = scalar_to_f64(scalar)? as f32;
        let shape = output.shape().to_vec();
        let mut tensor = self.allocate_tensor(operation_label, output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let input = read_f32(input, indices)?;
            let (left, right) = match scalar_side {
                ScalarSide::Left => (scalar, input),
                ScalarSide::Right => (input, scalar),
            };
            write_binary(&mut write, indices, operation, left, right)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn reduction(
        &self,
        operation: &ReductionSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        context.check()?;
        let label = reduction_operation_label(operation.operation);
        self.require_descriptor(
            label,
            PrimitiveOperation::Reduction(operation.operation),
            TensorRole::Input,
            input.descriptor(),
            context,
        )?;
        self.require_descriptor(
            label,
            PrimitiveOperation::Reduction(operation.operation),
            TensorRole::Output,
            &output,
            context,
        )?;
        validate_reduction_contract(operation, input.descriptor(), &output)?;
        let dimensions = canonical_reduction_dimensions(operation, input.descriptor().rank())?;
        let reduction = compute_reduction(operation, input, &output, &dimensions, context)?;
        context.check()?;
        let output_shape = output.shape().to_vec();
        let output_dtype = output.dtype();
        let mut tensor = self.allocate_tensor(label, output, context)?;
        let mut write = tensor.write()?;
        for (linear_index, value) in reduction.into_iter().enumerate() {
            context.cancellation.check()?;
            let linear_index =
                u64::try_from(linear_index).map_err(|_| TensorError::ShapeOverflow)?;
            let indices = linear_to_indices(linear_index, &output_shape)?;
            let encoded = output_dtype.encode_decoded_scalar(value, label, DeviceId::CPU)?;
            write_element(&mut write, &indices, &encoded)?;
        }
        drop(write);
        self.finish(tensor, context)
    }

    fn indexing(
        &self,
        operation: &IndexSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let operation_label = index_operation_label(operation);
        let primitive = match operation {
            IndexSpec::Select { .. } => PrimitiveOperation::Select,
            IndexSpec::Narrow { .. } => PrimitiveOperation::Narrow,
            IndexSpec::Gather { .. } => PrimitiveOperation::Gather,
            IndexSpec::Scatter { .. } => PrimitiveOperation::Scatter,
            IndexSpec::MaskedSelect => PrimitiveOperation::MaskedSelect,
        };
        validate_inputs(self, operation_label, primitive, inputs, context)?;
        self.require_descriptor(
            operation_label,
            primitive,
            TensorRole::Output,
            &output,
            context,
        )?;
        let input = match inputs {
            [input] => input,
            _ => {
                return Err(TensorError::Faulted {
                    reason: format!("index operation requires one input, got {}", inputs.len()),
                });
            }
        };
        require_dtype(input.descriptor().dtype(), output.dtype())?;
        let mapping = IndexMapping::new(operation, input.descriptor(), &output).map_err(
            |error| match error {
                IndexMappingError::Invalid(reason) => TensorError::Faulted { reason },
                IndexMappingError::Unsupported(reason) => self.unsupported(operation_label, reason),
            },
        )?;
        let shape = output.shape().to_vec();
        let mut tensor = self.allocate_tensor(operation_label, output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let source_indices = mapping.source_indices(indices)?;
            let source = input.element_bytes(&source_indices)?;
            write_element(&mut write, indices, source)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn resize(
        &self,
        operation: ResizeSpec,
        input: &Tensor,
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let operation_label = resize_operation_label(operation.mode);
        validate_inputs(
            self,
            operation_label,
            PrimitiveOperation::Resize(operation.mode),
            std::slice::from_ref(input),
            context,
        )?;
        self.require_descriptor(
            operation_label,
            PrimitiveOperation::Resize(operation.mode),
            TensorRole::Output,
            &output,
            context,
        )?;
        require_dtype(DType::F32, input.descriptor().dtype())?;
        require_dtype(DType::F32, output.dtype())?;
        let geometry = ResizeGeometry::new(input.descriptor(), &output, operation)?;
        let shape = output.shape().to_vec();
        let mut tensor = self.allocate_tensor(operation_label, output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let [batch, channel, output_y, output_x] = indices else {
                return Err(TensorError::Faulted {
                    reason: "resize output rank changed after validation".to_owned(),
                });
            };
            let value = geometry.sample(
                input,
                *batch,
                *channel,
                *output_y,
                *output_x,
                context.cancellation,
            )?;
            write_element(&mut write, indices, &value.to_ne_bytes())
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn convolution(
        &self,
        operation: &ConvolutionSpec,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.capabilities
            .require_primitive("zed.cpu.convolution", PrimitiveOperation::Convolution)?;
        validate_inputs(
            self,
            "zed.cpu.convolution",
            PrimitiveOperation::Convolution,
            inputs,
            context,
        )?;
        self.require_descriptor(
            "zed.cpu.convolution",
            PrimitiveOperation::Convolution,
            TensorRole::Output,
            &output,
            context,
        )?;
        let (input, weight, bias) = match inputs {
            [input, weight] => (input, weight, None),
            [input, weight, bias] => (input, weight, Some(bias)),
            _ => {
                return Err(TensorError::Faulted {
                    reason: format!(
                        "convolution requires input, weight, and optional bias tensors, got {} inputs",
                        inputs.len()
                    ),
                });
            }
        };
        let dtype = input.descriptor().dtype();
        let mismatched_dtype = if weight.descriptor().dtype() != dtype {
            Some(weight.descriptor().dtype())
        } else if let Some(bias) = bias.filter(|bias| bias.descriptor().dtype() != dtype) {
            Some(bias.descriptor().dtype())
        } else if output.dtype() != dtype {
            Some(output.dtype())
        } else {
            None
        };
        if let Some(actual) = mismatched_dtype {
            return Err(TensorError::DTypeMismatch {
                expected: dtype,
                actual,
            });
        }
        let spatial_dimensions =
            input
                .descriptor()
                .shape()
                .len()
                .checked_sub(2)
                .ok_or_else(|| TensorError::Faulted {
                    reason: "convolution input must include batch and channel dimensions"
                        .to_owned(),
                })?;
        let geometry = ConvolutionGeometry::new(
            spatial_dimensions,
            convolution_dimensions_to_usize(&operation.stride)?,
            convolution_dimensions_to_usize(&operation.padding)?,
            convolution_dimensions_to_usize(&operation.dilation)?,
            usize::try_from(operation.groups).map_err(|_| TensorError::ShapeOverflow)?,
            operation.transposed,
            convolution_dimensions_to_usize(&operation.output_padding)?,
        )
        .map_err(map_operator_indirection_error)?;
        let expected_output_shape = geometry
            .checked_output_shape(
                input.descriptor().shape(),
                weight.descriptor().shape(),
                bias.map(|bias| bias.descriptor().shape()),
            )
            .map_err(map_operator_indirection_error)?;
        if output.shape() != expected_output_shape {
            return Err(TensorError::Faulted {
                reason: format!(
                    "convolution output shape mismatch: expected {expected_output_shape:?}, got {:?}",
                    output.shape()
                ),
            });
        }

        let input_values = tensor_to_f32_workspace(self, input, context)?;
        let weight_values = tensor_to_f32_workspace(self, weight, context)?;
        let bias_values = bias
            .map(|bias| tensor_to_f32_workspace(self, bias, context))
            .transpose()?;
        let output_count =
            usize::try_from(output.element_count()?).map_err(|_| TensorError::ShapeOverflow)?;
        let mut output_values = self.workspace_vec::<f32>(context, output_count)?;
        for _ in 0..output_count {
            output_values.try_push(0.0)?;
        }
        let input_shape = convolution_dimensions_to_usize(input.descriptor().shape())?;
        let weight_shape = convolution_dimensions_to_usize(weight.descriptor().shape())?;
        let computed_shape = convolution_into_with_context_exact_native(
            &input_values,
            &input_shape,
            &weight_values,
            &weight_shape,
            bias_values.as_deref(),
            &geometry,
            DeviceId::CPU,
            &mut output_values,
            context,
        )
        .map_err(map_operator_indirection_error)?;
        if computed_shape != convolution_dimensions_to_usize(&expected_output_shape)? {
            return Err(TensorError::Faulted {
                reason: "canonical convolution returned an unexpected output shape".to_owned(),
            });
        }

        let output_shape = output.shape().to_vec();
        let mut tensor = self.allocate_tensor("zed.cpu.convolution", output, context)?;
        let mut write = tensor.write()?;
        let mut value_index = 0usize;
        for_each_index(&output_shape, context.cancellation, |indices| {
            let value = output_values
                .get(value_index)
                .copied()
                .ok_or(TensorError::ShapeOverflow)?;
            value_index = value_index
                .checked_add(1)
                .ok_or(TensorError::ShapeOverflow)?;
            let encoded = dtype.encode_decoded_scalar(
                DecodedScalar::Real(f64::from(value)),
                "zed.cpu.convolution",
                DeviceId::CPU,
            )?;
            write_element(&mut write, indices, &encoded)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn linear_algebra(
        &self,
        operation: LinearAlgebraOperation,
        inputs: &[Tensor],
        output: TensorDescriptor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let primitive = PrimitiveOperation::LinearAlgebra(operation);
        self.capabilities
            .require_primitive("zed.cpu.linear-algebra", primitive)?;
        if operation != LinearAlgebraOperation::BatchMatrixMultiply {
            return Err(self.unsupported(
                "zed.cpu.linear-algebra",
                "this native profile currently certifies batch matrix multiplication only",
            ));
        }
        let [left, right] = inputs else {
            return Err(TensorError::Faulted {
                reason: format!(
                    "batch matrix multiplication requires two inputs, got {}",
                    inputs.len()
                ),
            });
        };
        for input in [left, right] {
            self.require_descriptor(
                "zed.cpu.linear-algebra.bmm",
                primitive,
                TensorRole::Input,
                input.descriptor(),
                context,
            )?;
        }
        let dtype = left.descriptor().dtype();
        if right.descriptor().dtype() != dtype {
            return Err(TensorError::DTypeMismatch {
                expected: dtype,
                actual: right.descriptor().dtype(),
            });
        }
        if output.dtype() != dtype {
            return Err(TensorError::DTypeMismatch {
                expected: dtype,
                actual: output.dtype(),
            });
        }
        let [batch, rows, contracted] = left.descriptor().shape() else {
            return Err(TensorError::Faulted {
                reason: "batch matrix multiplication requires rank-three inputs".to_owned(),
            });
        };
        let [right_batch, right_contracted, columns] = right.descriptor().shape() else {
            return Err(TensorError::Faulted {
                reason: "batch matrix multiplication requires rank-three inputs".to_owned(),
            });
        };
        if batch != right_batch || contracted != right_contracted {
            return Err(TensorError::Faulted {
                reason: "batch matrix multiplication dimensions are incompatible".to_owned(),
            });
        }
        let shape = vec![*batch, *rows, *columns];
        require_same_shape(&shape, output.shape())?;
        self.require_descriptor(
            "zed.cpu.linear-algebra.bmm",
            primitive,
            TensorRole::Output,
            &output,
            context,
        )?;
        let mut tensor = self.allocate_tensor("zed.cpu.linear-algebra.bmm", output, context)?;
        let mut write = tensor.write()?;
        for_each_index(&shape, context.cancellation, |indices| {
            let [batch, row, column] = indices else {
                return Err(TensorError::Faulted {
                    reason: "batch matrix output rank changed after validation".to_owned(),
                });
            };
            let mut sum = 0.0_f32;
            for inner in 0..*contracted {
                context.check()?;
                sum += read_real_f32(left, &[*batch, *row, inner])?
                    * read_real_f32(right, &[*batch, inner, *column])?;
            }
            let encoded = dtype.encode_decoded_scalar(
                DecodedScalar::Real(f64::from(sum)),
                "zed.cpu.linear-algebra.bmm",
                DeviceId::CPU,
            )?;
            write_element(&mut write, indices, &encoded)
        })?;
        drop(write);
        self.finish(tensor, context)
    }

    fn matrix_inverse(
        &self,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let inverse = inverse_with_context_exact_native(self, input, context)
            .map_err(map_linear_algebra_part_one_error)?;
        let event = self.record_event(context)?;
        Ok((inverse, event))
    }

    fn kronecker_product(
        &self,
        left: &Tensor,
        right: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let product = kron_with_context_exact_native(self, left, right, context)
            .map_err(map_elementwise_part_twenty_one_error)?;
        let event = self.record_event(context)?;
        Ok((product, event))
    }

    fn constant_pad(
        &self,
        input: &Tensor,
        padding: &[i64],
        value: Option<DecodedScalar>,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let padded = functional_pad_with_context_exact_native(
            self,
            input,
            padding,
            FunctionalPadMode::Constant,
            value,
            context,
        )
        .map_err(map_shape_layout_transform_part_three_error)?;
        let event = self.record_event(context)?;
        Ok((padded, event))
    }

    fn vector_norm(
        &self,
        input: &Tensor,
        order: f64,
        dimensions: &[i64],
        keep_dimension: bool,
        dtype: Option<DType>,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let norm = vector_norm_with_context_exact_native(
            self,
            input,
            order,
            dimensions,
            keep_dimension,
            dtype,
            context,
        )
        .map_err(map_linear_algebra_part_one_error)?;
        let event = self.record_event(context)?;
        Ok((norm, event))
    }

    fn eye(
        &self,
        rows: u64,
        columns: Option<u64>,
        dtype: DType,
        layout: Layout,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        if layout != Layout::Contiguous {
            return Err(TensorError::UnsupportedCapability {
                operation: "COMFY-TENSOR-OP-CA2E738EA0EF".to_owned(),
                device: DeviceId::CPU,
                reason: format!(
                    "identity-matrix output requires contiguous layout, got {layout:?}"
                ),
            });
        }
        let identity = eye_with_context_exact_native(
            self,
            rows,
            columns,
            dtype,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )
        .map_err(map_tensor_creation_part_one_error)?;
        let event = self.record_event(context)?;
        Ok((identity, event))
    }

    fn replace_rectangular_slice(
        &self,
        input: &Tensor,
        source: &Tensor,
        offsets: &[u64],
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        const OPERATION: &str = "zed.tensor.indexing.rectangular-slice-replacement.v1";
        context.check()?;
        for tensor in [input, source] {
            if tensor.descriptor().device() != DeviceId::CPU {
                return Err(TensorError::DeviceMismatch {
                    expected: DeviceId::CPU,
                    actual: tensor.descriptor().device(),
                });
            }
            if tensor.descriptor().stream() != context.stream {
                return Err(TensorError::StreamMismatch {
                    expected: context.stream,
                    actual: tensor.descriptor().stream(),
                });
            }
        }
        require_dtype(input.descriptor().dtype(), source.descriptor().dtype())?;
        let rank = input.descriptor().rank();
        if source.descriptor().rank() != rank || offsets.len() != rank {
            return Err(TensorError::Faulted {
                reason: format!(
                    "{OPERATION} requires equal input/source rank and one offset per dimension"
                ),
            });
        }
        for (dimension, ((&offset, &source_size), &input_size)) in offsets
            .iter()
            .zip(source.descriptor().shape())
            .zip(input.descriptor().shape())
            .enumerate()
        {
            let end = offset
                .checked_add(source_size)
                .ok_or(TensorError::ShapeOverflow)?;
            if end > input_size {
                return Err(TensorError::IndexOutOfBounds {
                    dimension,
                    index: end,
                    size: input_size,
                });
            }
        }

        let mut destination_indices = self.workspace_vec::<u64>(context, rank)?;
        for _ in 0..rank {
            destination_indices.try_push(0)?;
        }
        let (mut output, _) = self.copy(input, input.descriptor().clone(), context)?;
        {
            let mut write = output.write()?;
            for_each_index(
                source.descriptor().shape(),
                context.cancellation,
                |indices| {
                    for ((destination, source_index), offset) in
                        destination_indices.iter_mut().zip(indices).zip(offsets)
                    {
                        *destination = source_index
                            .checked_add(*offset)
                            .ok_or(TensorError::ShapeOverflow)?;
                    }
                    write_element(
                        &mut write,
                        &destination_indices,
                        source.element_bytes(indices)?,
                    )
                },
            )?;
        }
        self.finish(output, context)
    }

    fn validate_finite(
        &self,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        let finite = isfinite_with_context_exact_native(self, input, context)
            .map_err(map_elementwise_part_four_error)?;
        let dimensions = (0..finite.descriptor().rank())
            .map(|dimension| u64::try_from(dimension).map_err(|_| TensorError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let descriptor =
            TensorDescriptor::contiguous(Vec::new(), DType::Bool, DeviceId::CPU, context.stream)?;
        let (all_finite, _) = self.reduction(
            &ReductionSpec {
                operation: ReductionOperation::All,
                dimensions,
                keep_dimensions: false,
                accumulation_dtype: None,
                correction: 0,
            },
            &finite,
            descriptor,
            context,
        )?;
        context.check()?;
        if all_finite
            .descriptor()
            .dtype()
            .decode_scalar(all_finite.element_bytes(&[])?)?
            != DecodedScalar::Boolean(true)
        {
            return Err(TensorError::InvalidNumeric {
                reason: "tensor contains a non-finite value".to_owned(),
            });
        }
        context.check()?;
        Ok(())
    }

    fn upload_f32_payload(
        &self,
        shape: &[u64],
        values: &[f32],
        dtype: DType,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let tensor = tensor_from_f32_with_backend_exact_native(
            self,
            shape,
            values,
            dtype,
            DeviceId::CPU,
            context,
        )
        .map_err(map_operator_indirection_error)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn cast_tensor(
        &self,
        input: &Tensor,
        dtype: DType,
        non_blocking: bool,
        copy: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        let tensor = cast_to_with_backend_exact_native(
            self,
            input,
            dtype,
            DeviceId::CPU,
            non_blocking,
            copy,
            context,
        )
        .map_err(map_operator_indirection_error)?;
        let event = self.record_event(context)?;
        Ok((tensor, event))
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        inputs: &[Tensor],
        outputs: &[TensorDescriptor],
        context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.capabilities
            .require_primitive("zed.cpu.custom-kernel", PrimitiveOperation::CustomKernel)?;
        validate_inputs(
            self,
            "zed.cpu.custom-kernel",
            PrimitiveOperation::CustomKernel,
            inputs,
            context,
        )?;
        for output in outputs {
            self.require_descriptor(
                "zed.cpu.custom-kernel",
                PrimitiveOperation::CustomKernel,
                TensorRole::Output,
                output,
                context,
            )?;
        }
        Err(self.unsupported(
            "zed.cpu.custom-kernel",
            "no custom CPU kernel is registered for this foundation backend",
        ))
    }
}

fn map_linear_algebra_part_one_error(error: LinearAlgebraPartOneError) -> TensorError {
    match error {
        LinearAlgebraPartOneError::Tensor(error) => error,
        LinearAlgebraPartOneError::Cancelled => TensorError::Cancelled,
        LinearAlgebraPartOneError::UnsupportedDevice { operation, device } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "canonical linear algebra does not support the selected device".to_owned(),
            }
        }
        LinearAlgebraPartOneError::UnsupportedDType { operation, dtype } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::CPU,
                reason: format!("canonical linear algebra does not support {dtype:?}"),
            }
        }
        LinearAlgebraPartOneError::ShapeOverflow(_) => TensorError::ShapeOverflow,
        LinearAlgebraPartOneError::Singular { operation } => TensorError::InvalidNumeric {
            reason: format!("operation {operation} received a singular matrix"),
        },
        LinearAlgebraPartOneError::DidNotConverge { operation } => TensorError::InvalidNumeric {
            reason: format!("operation {operation} did not converge"),
        },
        LinearAlgebraPartOneError::Invalid { operation, reason } => TensorError::Faulted {
            reason: format!("operation {operation} received invalid input: {reason}"),
        },
        LinearAlgebraPartOneError::Cross(error) => TensorError::Faulted {
            reason: error.to_string(),
        },
    }
}

fn map_elementwise_part_twenty_one_error(
    error: ElementwiseRuntimePartTwentyOneError,
) -> TensorError {
    match error {
        ElementwiseRuntimePartTwentyOneError::Tensor(error) => error,
        ElementwiseRuntimePartTwentyOneError::Cancelled => TensorError::Cancelled,
        other => TensorError::Faulted {
            reason: other.to_string(),
        },
    }
}

fn map_operator_indirection_error(error: OperatorIndirectionError) -> TensorError {
    match error {
        OperatorIndirectionError::Tensor(error) => error,
        OperatorIndirectionError::Cancelled => TensorError::Cancelled,
        OperatorIndirectionError::ShapeOverflow(_) => TensorError::ShapeOverflow,
        OperatorIndirectionError::UnsupportedDevice { operation, device } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "the canonical tensor operator does not support the selected device"
                    .to_owned(),
            }
        }
        OperatorIndirectionError::ValueCount {
            expected, actual, ..
        } => TensorError::StorageLength {
            expected: u64::try_from(expected)
                .ok()
                .and_then(|count| count.checked_mul(4))
                .unwrap_or(u64::MAX),
            actual: u64::try_from(actual)
                .ok()
                .and_then(|count| count.checked_mul(4))
                .unwrap_or(u64::MAX),
        },
        OperatorIndirectionError::Invalid(reason) => TensorError::Faulted {
            reason: format!("the canonical tensor operator received invalid input: {reason}"),
        },
        OperatorIndirectionError::Attention(error) => TensorError::Faulted {
            reason: error.to_string(),
        },
    }
}

fn map_shape_layout_transform_part_three_error(
    error: ShapeLayoutTransformPartThreeError,
) -> TensorError {
    match error {
        ShapeLayoutTransformPartThreeError::Tensor(error) => error,
        ShapeLayoutTransformPartThreeError::Cancelled => TensorError::Cancelled,
        ShapeLayoutTransformPartThreeError::UnsupportedDevice { operation, device } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "canonical padding does not support the selected device".to_owned(),
            }
        }
        ShapeLayoutTransformPartThreeError::UnsupportedDType { operation, dtype } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::CPU,
                reason: format!("canonical padding does not support {dtype:?}"),
            }
        }
        ShapeLayoutTransformPartThreeError::ShapeOverflow { .. } => TensorError::ShapeOverflow,
        ShapeLayoutTransformPartThreeError::Invalid { operation, reason }
        | ShapeLayoutTransformPartThreeError::CanonicalOwner { operation, reason } => {
            TensorError::Faulted {
                reason: format!("operation {operation} failed: {reason}"),
            }
        }
    }
}

fn map_tensor_creation_part_one_error(error: TensorCreationPartOneError) -> TensorError {
    match error {
        TensorCreationPartOneError::Tensor { source, .. } => source,
        TensorCreationPartOneError::Cancelled { .. } => TensorError::Cancelled,
        TensorCreationPartOneError::UnsupportedDevice { operation, device } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "canonical tensor creation does not support the selected device".to_owned(),
            }
        }
        TensorCreationPartOneError::UnsupportedLayout { operation, layout } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::CPU,
                reason: format!("canonical tensor creation does not support {layout:?}"),
            }
        }
        TensorCreationPartOneError::UnsupportedDType {
            operation,
            dtype,
            reason,
        } => TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: DeviceId::CPU,
            reason: format!("canonical tensor creation does not support {dtype:?}: {reason}"),
        },
        TensorCreationPartOneError::Invalid { operation, reason } => TensorError::Faulted {
            reason: format!("operation {operation} received invalid input: {reason}"),
        },
        TensorCreationPartOneError::Cast { operation, source } => TensorError::Faulted {
            reason: format!("operation {operation} failed in canonical cast: {source}"),
        },
        TensorCreationPartOneError::Autograd { operation, source } => TensorError::Faulted {
            reason: format!("operation {operation} failed in canonical autograd: {source}"),
        },
    }
}

fn map_elementwise_part_four_error(error: ElementwiseRuntimePartFourError) -> TensorError {
    match error {
        ElementwiseRuntimePartFourError::Tensor(error) => error,
        ElementwiseRuntimePartFourError::Cancelled => TensorError::Cancelled,
        ElementwiseRuntimePartFourError::UnsupportedDevice { operation, device } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device,
                reason: "canonical finite validation does not support the selected device"
                    .to_owned(),
            }
        }
        ElementwiseRuntimePartFourError::UnsupportedDType { operation, dtype } => {
            TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: DeviceId::CPU,
                reason: format!("canonical finite validation does not support {dtype:?}"),
            }
        }
        ElementwiseRuntimePartFourError::ShapeOverflow(_) => TensorError::ShapeOverflow,
        other => TensorError::Faulted {
            reason: other.to_string(),
        },
    }
}

fn unary_operation_label(operation: UnaryOperation) -> &'static str {
    match operation {
        UnaryOperation::Absolute => "zed.cpu.unary.absolute",
        UnaryOperation::Negate => "zed.cpu.unary.negate",
        UnaryOperation::Exponential => "zed.cpu.unary.exponential",
        UnaryOperation::NaturalLogarithm => "zed.cpu.unary.natural-logarithm",
        UnaryOperation::SquareRoot => "zed.cpu.unary.square-root",
        UnaryOperation::Reciprocal => "zed.cpu.unary.reciprocal",
        UnaryOperation::Sine => "zed.cpu.unary.sine",
        UnaryOperation::Cosine => "zed.cpu.unary.cosine",
        UnaryOperation::HyperbolicTangent => "zed.cpu.unary.hyperbolic-tangent",
        UnaryOperation::Sigmoid => "zed.cpu.unary.sigmoid",
        UnaryOperation::Round => "zed.cpu.unary.round",
        UnaryOperation::Sinc => "zed.cpu.unary.sinc",
        UnaryOperation::Log1p => "zed.cpu.unary.log1p",
        UnaryOperation::ReciprocalSquareRoot => "zed.cpu.unary.rsqrt",
        UnaryOperation::Relu => "zed.cpu.unary.relu",
        UnaryOperation::IsFinite => "zed.cpu.unary.is-finite",
        UnaryOperation::InvertUnitInterval => "zed.cpu.unary.invert-unit-interval",
        UnaryOperation::LogarithmBaseTwo => "zed.cpu.unary.log2",
        UnaryOperation::Signum => "zed.cpu.unary.signum",
        UnaryOperation::Tangent => "zed.cpu.unary.tangent",
        UnaryOperation::ArcTangent => "zed.cpu.unary.arc-tangent",
        UnaryOperation::ArcHyperbolicTangent => "zed.cpu.unary.arc-hyperbolic-tangent",
    }
}

fn binary_operation_label(operation: BinaryOperation, scalar: bool) -> &'static str {
    match (scalar, operation) {
        (false, BinaryOperation::Add) => "zed.cpu.binary.add",
        (false, BinaryOperation::Subtract) => "zed.cpu.binary.subtract",
        (false, BinaryOperation::Multiply) => "zed.cpu.binary.multiply",
        (false, BinaryOperation::Divide) => "zed.cpu.binary.divide",
        (false, BinaryOperation::Remainder) => "zed.cpu.binary.remainder",
        (false, BinaryOperation::Power) => "zed.cpu.binary.power",
        (false, BinaryOperation::Minimum) => "zed.cpu.binary.minimum",
        (false, BinaryOperation::Maximum) => "zed.cpu.binary.maximum",
        (false, BinaryOperation::Equal) => "zed.cpu.binary.equal",
        (false, BinaryOperation::Less) => "zed.cpu.binary.less",
        (false, BinaryOperation::LessEqual) => "zed.cpu.binary.less-equal",
        (false, BinaryOperation::Greater) => "zed.cpu.binary.greater",
        (false, BinaryOperation::GreaterEqual) => "zed.cpu.binary.greater-equal",
        (false, BinaryOperation::LogicalAnd) => "zed.cpu.binary.logical-and",
        (false, BinaryOperation::LogicalOr) => "zed.cpu.binary.logical-or",
        (false, BinaryOperation::FloatingRemainder) => "zed.cpu.binary.fmod",
        (false, BinaryOperation::Atan2) => "zed.cpu.binary.atan2",
        (false, BinaryOperation::LogAddExp) => "zed.cpu.binary.logaddexp",
        (true, BinaryOperation::Add) => "zed.cpu.binary-scalar.add",
        (true, BinaryOperation::Subtract) => "zed.cpu.binary-scalar.subtract",
        (true, BinaryOperation::Multiply) => "zed.cpu.binary-scalar.multiply",
        (true, BinaryOperation::Divide) => "zed.cpu.binary-scalar.divide",
        (true, BinaryOperation::Remainder) => "zed.cpu.binary-scalar.remainder",
        (true, BinaryOperation::Power) => "zed.cpu.binary-scalar.power",
        (true, BinaryOperation::Minimum) => "zed.cpu.binary-scalar.minimum",
        (true, BinaryOperation::Maximum) => "zed.cpu.binary-scalar.maximum",
        (true, BinaryOperation::Equal) => "zed.cpu.binary-scalar.equal",
        (true, BinaryOperation::Less) => "zed.cpu.binary-scalar.less",
        (true, BinaryOperation::LessEqual) => "zed.cpu.binary-scalar.less-equal",
        (true, BinaryOperation::Greater) => "zed.cpu.binary-scalar.greater",
        (true, BinaryOperation::GreaterEqual) => "zed.cpu.binary-scalar.greater-equal",
        (true, BinaryOperation::LogicalAnd) => "zed.cpu.binary-scalar.logical-and",
        (true, BinaryOperation::LogicalOr) => "zed.cpu.binary-scalar.logical-or",
        (true, BinaryOperation::FloatingRemainder) => "zed.cpu.binary-scalar.fmod",
        (true, BinaryOperation::Atan2) => "zed.cpu.binary-scalar.atan2",
        (true, BinaryOperation::LogAddExp) => "zed.cpu.binary-scalar.logaddexp",
    }
}

fn index_operation_label(operation: &IndexSpec) -> &'static str {
    match operation {
        IndexSpec::Select { .. } => "zed.cpu.index.select",
        IndexSpec::Narrow { .. } => "zed.cpu.index.narrow",
        IndexSpec::Gather { .. } => "zed.cpu.index.gather",
        IndexSpec::Scatter { .. } => "zed.cpu.index.scatter",
        IndexSpec::MaskedSelect => "zed.cpu.index.masked-select",
    }
}

fn resize_operation_label(mode: ResizeMode) -> &'static str {
    match mode {
        ResizeMode::NearestExact => "zed.cpu.resize.nearest-exact",
        ResizeMode::Bilinear => "zed.cpu.resize.bilinear",
        ResizeMode::Area => "zed.cpu.resize.area",
        ResizeMode::Bicubic => "zed.cpu.resize.bicubic",
        ResizeMode::Lanczos => "zed.cpu.resize.lanczos",
    }
}

fn require_same_shape(expected: &[u64], actual: &[u64]) -> Result<(), TensorError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TensorError::Faulted {
            reason: format!("tensor shape mismatch: expected {expected:?}, got {actual:?}"),
        })
    }
}

fn require_dtype(expected: DType, actual: DType) -> Result<(), TensorError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TensorError::DTypeMismatch { expected, actual })
    }
}

fn reduction_operation_label(operation: ReductionOperation) -> &'static str {
    match operation {
        ReductionOperation::Sum => "zed.cpu.reduction.sum",
        ReductionOperation::Product => "zed.cpu.reduction.product",
        ReductionOperation::Mean => "zed.cpu.reduction.mean",
        ReductionOperation::Minimum => "zed.cpu.reduction.minimum",
        ReductionOperation::Maximum => "zed.cpu.reduction.maximum",
        ReductionOperation::ArgMinimum => "zed.cpu.reduction.arg-minimum",
        ReductionOperation::ArgMaximum => "zed.cpu.reduction.arg-maximum",
        ReductionOperation::All => "zed.cpu.reduction.all",
        ReductionOperation::Any => "zed.cpu.reduction.any",
        ReductionOperation::Variance => "zed.cpu.reduction.variance",
        ReductionOperation::StandardDeviation => "zed.cpu.reduction.standard-deviation",
    }
}

fn reduction_supports_input(operation: ReductionOperation, dtype: DType) -> bool {
    match operation {
        ReductionOperation::All | ReductionOperation::Any => true,
        ReductionOperation::Minimum
        | ReductionOperation::Maximum
        | ReductionOperation::ArgMinimum
        | ReductionOperation::ArgMaximum => !matches!(dtype, DType::Complex64 | DType::Complex128),
        ReductionOperation::Sum => matches!(
            dtype,
            DType::F64
                | DType::F32
                | DType::F16
                | DType::Bf16
                | DType::I64
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U64
                | DType::U32
                | DType::U16
                | DType::U8
                | DType::Bool
        ),
        ReductionOperation::Product
        | ReductionOperation::Mean
        | ReductionOperation::Variance
        | ReductionOperation::StandardDeviation => {
            matches!(dtype, DType::F64 | DType::F32 | DType::F16 | DType::Bf16)
        }
    }
}

fn reduction_supports_output(operation: ReductionOperation, dtype: DType) -> bool {
    match operation {
        ReductionOperation::All | ReductionOperation::Any => dtype == DType::Bool,
        ReductionOperation::ArgMinimum | ReductionOperation::ArgMaximum => dtype == DType::I64,
        ReductionOperation::Minimum | ReductionOperation::Maximum => {
            !matches!(dtype, DType::Complex64 | DType::Complex128)
        }
        ReductionOperation::Sum => matches!(
            dtype,
            DType::I64 | DType::F64 | DType::F32 | DType::F16 | DType::Bf16
        ),
        ReductionOperation::Product
        | ReductionOperation::Mean
        | ReductionOperation::Variance
        | ReductionOperation::StandardDeviation => {
            matches!(dtype, DType::F64 | DType::F32 | DType::F16 | DType::Bf16)
        }
    }
}

fn canonical_reduction_dimensions(
    operation: &ReductionSpec,
    rank: usize,
) -> Result<Vec<usize>, TensorError> {
    let dimensions = if operation.dimensions.is_empty() {
        (0..rank).collect::<Vec<_>>()
    } else {
        operation
            .dimensions
            .iter()
            .map(|dimension| usize::try_from(*dimension).map_err(|_| TensorError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut seen = vec![false; rank];
    for &dimension in &dimensions {
        let rank_u64 = u64::try_from(rank).map_err(|_| TensorError::ShapeOverflow)?;
        let dimension_u64 = u64::try_from(dimension).map_err(|_| TensorError::ShapeOverflow)?;
        let entry = seen
            .get_mut(dimension)
            .ok_or(TensorError::IndexOutOfBounds {
                dimension,
                index: dimension_u64,
                size: rank_u64,
            })?;
        if *entry {
            return Err(TensorError::InvalidNumeric {
                reason: format!("reduction dimension {dimension} was specified more than once"),
            });
        }
        *entry = true;
    }
    Ok(dimensions)
}

fn validate_reduction_contract(
    operation: &ReductionSpec,
    input: &TensorDescriptor,
    output: &TensorDescriptor,
) -> Result<(), TensorError> {
    let dimensions = canonical_reduction_dimensions(operation, input.rank())?;
    let mut reduced = vec![false; input.rank()];
    for dimension in dimensions {
        *reduced
            .get_mut(dimension)
            .ok_or(TensorError::ShapeOverflow)? = true;
    }
    let expected_shape = input
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| {
            if reduced.get(axis).copied().unwrap_or(false) {
                operation.keep_dimensions.then_some(1)
            } else {
                Some(*size)
            }
        })
        .collect::<Vec<_>>();
    if output.shape() != expected_shape {
        return Err(TensorError::InvalidNumeric {
            reason: format!(
                "reduction output shape {:?} does not match expected {expected_shape:?}",
                output.shape()
            ),
        });
    }
    let expected_dtype = match operation.operation {
        ReductionOperation::All | ReductionOperation::Any => DType::Bool,
        ReductionOperation::ArgMinimum | ReductionOperation::ArgMaximum => DType::I64,
        ReductionOperation::Minimum | ReductionOperation::Maximum => input.dtype(),
        ReductionOperation::Sum
        | ReductionOperation::Product
        | ReductionOperation::Mean
        | ReductionOperation::Variance
        | ReductionOperation::StandardDeviation => {
            operation.accumulation_dtype.unwrap_or(input.dtype())
        }
    };
    if expected_dtype != output.dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: expected_dtype,
            actual: output.dtype(),
        });
    }
    if !reduction_supports_input(operation.operation, input.dtype())
        || !reduction_supports_output(operation.operation, output.dtype())
    {
        return Err(TensorError::UnsupportedCapability {
            operation: reduction_operation_label(operation.operation).to_owned(),
            device: DeviceId::CPU,
            reason: format!(
                "the {:?} to {:?} dtype pair is not implemented by the canonical CPU reduction owner",
                input.dtype(),
                output.dtype()
            ),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ReductionAccumulator {
    count: u64,
    sum: f64,
    integer_sum: i64,
    product: f64,
    mean: f64,
    squared_deviation: f64,
    extremum: Option<(DecodedScalar, u64)>,
    all: bool,
    any: bool,
}

impl Default for ReductionAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            integer_sum: 0,
            product: 1.0,
            mean: 0.0,
            squared_deviation: 0.0,
            extremum: None,
            all: true,
            any: false,
        }
    }
}

fn compute_reduction(
    operation: &ReductionSpec,
    input: &Tensor,
    output: &TensorDescriptor,
    dimensions: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<Vec<DecodedScalar>, TensorError> {
    let output_count =
        usize::try_from(output.element_count()?).map_err(|_| TensorError::ShapeOverflow)?;
    let mut accumulators = vec![ReductionAccumulator::default(); output_count];
    let mut reduced = vec![false; input.descriptor().rank()];
    for &dimension in dimensions {
        *reduced
            .get_mut(dimension)
            .ok_or(TensorError::ShapeOverflow)? = true;
    }
    for linear_index in 0..input.descriptor().element_count()? {
        context.cancellation.check()?;
        let input_indices = linear_to_indices(linear_index, input.descriptor().shape())?;
        let output_indices = input_indices
            .iter()
            .enumerate()
            .filter_map(|(axis, index)| {
                if reduced.get(axis).copied().unwrap_or(false) {
                    operation.keep_dimensions.then_some(0)
                } else {
                    Some(*index)
                }
            })
            .collect::<Vec<_>>();
        let output_index = usize::try_from(indices_to_linear(&output_indices, output.shape())?)
            .map_err(|_| TensorError::ShapeOverflow)?;
        let accumulator = accumulators
            .get_mut(output_index)
            .ok_or(TensorError::ShapeOverflow)?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.linear_element_bytes(linear_index)?)?;
        let reduced_index =
            reduced_linear_index(&input_indices, input.descriptor().shape(), &reduced)?;
        update_reduction_accumulator(accumulator, operation.operation, value, reduced_index)?;
    }
    accumulators
        .into_iter()
        .map(|accumulator| finalize_reduction_accumulator(accumulator, operation))
        .collect()
}

fn update_reduction_accumulator(
    accumulator: &mut ReductionAccumulator,
    operation: ReductionOperation,
    value: DecodedScalar,
    reduced_index: u64,
) -> Result<(), TensorError> {
    accumulator.all &= value.is_nonzero();
    accumulator.any |= value.is_nonzero();
    match operation {
        ReductionOperation::Sum => accumulate_sum(accumulator, value)?,
        ReductionOperation::Product => accumulator.product *= decoded_real(value)?,
        ReductionOperation::Mean => accumulator.sum += decoded_real(value)?,
        ReductionOperation::Variance | ReductionOperation::StandardDeviation => {
            let value = decoded_real(value)?;
            let next_count = accumulator
                .count
                .checked_add(1)
                .ok_or(TensorError::ShapeOverflow)?;
            let delta = value - accumulator.mean;
            accumulator.mean += delta / next_count as f64;
            accumulator.squared_deviation += delta * (value - accumulator.mean);
            accumulator.count = next_count;
        }
        ReductionOperation::Minimum | ReductionOperation::ArgMinimum => {
            if accumulator
                .extremum
                .is_none_or(|(current, _)| decoded_is_better(value, current, true))
            {
                accumulator.extremum = Some((value, reduced_index));
            }
        }
        ReductionOperation::Maximum | ReductionOperation::ArgMaximum => {
            if accumulator
                .extremum
                .is_none_or(|(current, _)| decoded_is_better(value, current, false))
            {
                accumulator.extremum = Some((value, reduced_index));
            }
        }
        ReductionOperation::All | ReductionOperation::Any => {}
    }
    if !matches!(
        operation,
        ReductionOperation::Variance | ReductionOperation::StandardDeviation
    ) {
        accumulator.count = accumulator
            .count
            .checked_add(1)
            .ok_or(TensorError::ShapeOverflow)?;
    }
    Ok(())
}

fn finalize_reduction_accumulator(
    accumulator: ReductionAccumulator,
    operation: &ReductionSpec,
) -> Result<DecodedScalar, TensorError> {
    Ok(match operation.operation {
        ReductionOperation::Sum => {
            if operation.accumulation_dtype == Some(DType::I64) {
                DecodedScalar::Signed(accumulator.integer_sum)
            } else {
                DecodedScalar::Real(accumulator.sum)
            }
        }
        ReductionOperation::Product => DecodedScalar::Real(accumulator.product),
        ReductionOperation::Mean => DecodedScalar::Real(if accumulator.count == 0 {
            f64::NAN
        } else {
            accumulator.sum / accumulator.count as f64
        }),
        ReductionOperation::Variance | ReductionOperation::StandardDeviation => {
            let variance = accumulator
                .count
                .checked_sub(operation.correction)
                .filter(|denominator| *denominator > 0)
                .map_or(f64::NAN, |denominator| {
                    accumulator.squared_deviation / denominator as f64
                });
            DecodedScalar::Real(
                if operation.operation == ReductionOperation::StandardDeviation {
                    variance.sqrt()
                } else {
                    variance
                },
            )
        }
        ReductionOperation::Minimum | ReductionOperation::Maximum => accumulator
            .extremum
            .map(|(value, _)| value)
            .ok_or_else(|| TensorError::InvalidNumeric {
                reason: "minimum and maximum reductions require a nonempty reduced domain"
                    .to_owned(),
            })?,
        ReductionOperation::ArgMinimum | ReductionOperation::ArgMaximum => {
            let index = accumulator
                .extremum
                .map(|(_, index)| index)
                .ok_or_else(|| TensorError::InvalidNumeric {
                    reason: "arg reductions require a nonempty reduced domain".to_owned(),
                })?;
            DecodedScalar::Signed(i64::try_from(index).map_err(|_| TensorError::ShapeOverflow)?)
        }
        ReductionOperation::All => DecodedScalar::Boolean(accumulator.all),
        ReductionOperation::Any => DecodedScalar::Boolean(accumulator.any),
    })
}

fn decoded_real(value: DecodedScalar) -> Result<f64, TensorError> {
    match value {
        DecodedScalar::Real(value) => Ok(value),
        _ => Err(TensorError::InvalidNumeric {
            reason: "floating reduction received a non-floating scalar".to_owned(),
        }),
    }
}

fn accumulate_sum(
    accumulator: &mut ReductionAccumulator,
    value: DecodedScalar,
) -> Result<(), TensorError> {
    match value {
        DecodedScalar::Boolean(value) => {
            let value = i64::from(value);
            accumulator.integer_sum = accumulator.integer_sum.wrapping_add(value);
            accumulator.sum += value as f64;
        }
        DecodedScalar::Signed(value) => {
            accumulator.integer_sum = accumulator.integer_sum.wrapping_add(value);
            accumulator.sum += value as f64;
        }
        DecodedScalar::Unsigned(value) => {
            accumulator.integer_sum = accumulator.integer_sum.wrapping_add(value as i64);
            accumulator.sum += value as f64;
        }
        DecodedScalar::Real(value) => {
            accumulator.integer_sum = accumulator.integer_sum.wrapping_add(value as i64);
            accumulator.sum += value;
        }
        DecodedScalar::Complex { .. } => {
            return Err(TensorError::InvalidNumeric {
                reason: "real reduction received a complex scalar".to_owned(),
            });
        }
    }
    Ok(())
}

fn decoded_is_better(candidate: DecodedScalar, current: DecodedScalar, minimum: bool) -> bool {
    match (candidate, current) {
        (DecodedScalar::Boolean(candidate), DecodedScalar::Boolean(current)) => {
            if minimum {
                !candidate && current
            } else {
                candidate && !current
            }
        }
        (DecodedScalar::Signed(candidate), DecodedScalar::Signed(current)) => {
            if minimum {
                candidate < current
            } else {
                candidate > current
            }
        }
        (DecodedScalar::Unsigned(candidate), DecodedScalar::Unsigned(current)) => {
            if minimum {
                candidate < current
            } else {
                candidate > current
            }
        }
        (DecodedScalar::Real(candidate), DecodedScalar::Real(current)) => {
            if candidate.is_nan() {
                !current.is_nan()
            } else if current.is_nan() {
                false
            } else if minimum {
                candidate < current
            } else {
                candidate > current
            }
        }
        _ => false,
    }
}

fn reduced_linear_index(
    indices: &[u64],
    shape: &[u64],
    reduced: &[bool],
) -> Result<u64, TensorError> {
    let mut linear = 0_u64;
    for (axis, (&index, &size)) in indices.iter().zip(shape).enumerate() {
        if reduced.get(axis).copied().unwrap_or(false) {
            linear = linear
                .checked_mul(size)
                .and_then(|value| value.checked_add(index))
                .ok_or(TensorError::ShapeOverflow)?;
        }
    }
    Ok(linear)
}

fn linear_to_indices(mut linear: u64, shape: &[u64]) -> Result<Vec<u64>, TensorError> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let size = *shape.get(axis).ok_or(TensorError::ShapeOverflow)?;
        if size == 0 {
            return Err(TensorError::ShapeOverflow);
        }
        *indices.get_mut(axis).ok_or(TensorError::ShapeOverflow)? = linear % size;
        linear /= size;
    }
    Ok(indices)
}

fn indices_to_linear(indices: &[u64], shape: &[u64]) -> Result<u64, TensorError> {
    if indices.len() != shape.len() {
        return Err(TensorError::IndexRankMismatch {
            rank: shape.len(),
            indices: indices.len(),
        });
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_u64, |linear, (&index, &size)| {
            linear
                .checked_mul(size)
                .and_then(|value| value.checked_add(index))
                .ok_or(TensorError::ShapeOverflow)
        })
}

fn for_each_index(
    shape: &[u64],
    cancellation: &CancellationToken,
    mut callback: impl FnMut(&[u64]) -> Result<(), TensorError>,
) -> Result<(), TensorError> {
    if shape.contains(&0) {
        return Ok(());
    }
    if shape.is_empty() {
        cancellation.check()?;
        return callback(&[]);
    }
    let mut indices = vec![0_u64; shape.len()];
    loop {
        cancellation.check()?;
        callback(&indices)?;
        let mut dimension = shape.len();
        loop {
            dimension = dimension.checked_sub(1).ok_or(TensorError::ShapeOverflow)?;
            let index = indices
                .get_mut(dimension)
                .ok_or(TensorError::ShapeOverflow)?;
            *index = index.checked_add(1).ok_or(TensorError::ShapeOverflow)?;
            let size = shape.get(dimension).ok_or(TensorError::ShapeOverflow)?;
            if *index < *size {
                break;
            }
            *index = 0;
            if dimension == 0 {
                return Ok(());
            }
        }
    }
}

fn copy_logical(
    source: &Tensor,
    destination: &mut Tensor,
    cancellation: &CancellationToken,
) -> Result<(), TensorError> {
    let shape = source.descriptor().shape().to_vec();
    let mut write = destination.write()?;
    for_each_index(&shape, cancellation, |indices| {
        write_element(&mut write, indices, source.element_bytes(indices)?)
    })
}

fn write_element(
    write: &mut TensorWrite<'_>,
    indices: &[u64],
    value: &[u8],
) -> Result<(), TensorError> {
    let destination = write.element_bytes_mut(indices)?;
    if destination.len() != value.len() {
        return Err(TensorError::StorageLength {
            expected: u64::try_from(destination.len()).map_err(|_| TensorError::ShapeOverflow)?,
            actual: u64::try_from(value.len()).map_err(|_| TensorError::ShapeOverflow)?,
        });
    }
    destination.copy_from_slice(value);
    Ok(())
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, TensorError> {
    let bytes: [u8; 4] =
        tensor
            .element_bytes(indices)?
            .try_into()
            .map_err(|_| TensorError::StorageLength {
                expected: 4,
                actual: tensor.descriptor().dtype().byte_width(),
            })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn read_real_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, TensorError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: tensor.descriptor().dtype(),
        }),
    }
}

fn convolution_dimensions_to_usize(values: &[u64]) -> Result<Vec<usize>, TensorError> {
    values
        .iter()
        .map(|value| usize::try_from(*value).map_err(|_| TensorError::ShapeOverflow))
        .collect()
}

fn tensor_to_f32_workspace(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, TensorError> {
    let element_count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let mut values = backend.workspace_vec(context, element_count)?;
    let shape = tensor.descriptor().shape().to_vec();
    for_each_index(&shape, context.cancellation, |indices| {
        let decoded = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.element_bytes(indices)?)?;
        let DecodedScalar::Real(value) = decoded else {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                actual: tensor.descriptor().dtype(),
            });
        };
        values.try_push(value as f32)
    })?;
    Ok(values)
}

fn scalar_to_f64(value: Scalar) -> Result<f64, TensorError> {
    Ok(match value {
        Scalar::Boolean(value) => f64::from(u8::from(value)),
        Scalar::Signed(value) => value as f64,
        Scalar::Unsigned(value) => value as f64,
        Scalar::Float(value) => value,
    })
}

pub(crate) fn apply_unary_scalar(operation: UnaryOperation, value: f32) -> f32 {
    match operation {
        UnaryOperation::Absolute => value.abs(),
        UnaryOperation::Negate => -value,
        UnaryOperation::Exponential => value.exp(),
        UnaryOperation::NaturalLogarithm => value.ln(),
        UnaryOperation::SquareRoot => value.sqrt(),
        UnaryOperation::Reciprocal => value.recip(),
        UnaryOperation::Sine => value.sin(),
        UnaryOperation::Cosine => value.cos(),
        UnaryOperation::HyperbolicTangent => value.tanh(),
        UnaryOperation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        UnaryOperation::Round => value.round_ties_even(),
        UnaryOperation::Sinc if value == 0.0 => 1.0,
        UnaryOperation::Sinc => {
            let argument = std::f32::consts::PI * value;
            argument.sin() / argument
        }
        UnaryOperation::Log1p => value.ln_1p(),
        UnaryOperation::ReciprocalSquareRoot => value.sqrt().recip(),
        UnaryOperation::Relu if value.is_nan() => f32::NAN,
        UnaryOperation::Relu => value.max(0.0),
        UnaryOperation::InvertUnitInterval => 1.0 - value,
        UnaryOperation::IsFinite => value,
        UnaryOperation::LogarithmBaseTwo => value.log2(),
        UnaryOperation::Signum if value == 0.0 => value,
        UnaryOperation::Signum => value.signum(),
        // Platform tanf differs by one ULP in source-pinned field-of-view observations.
        UnaryOperation::Tangent => libm::tanf(value),
        UnaryOperation::ArcTangent => value.atan(),
        UnaryOperation::ArcHyperbolicTangent => value.atanh(),
    }
}

pub fn binary_broadcast_shape(left: &[u64], right: &[u64]) -> Result<Vec<u64>, TensorError> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| TensorError::Faulted {
            reason: "binary broadcast shape allocation failed".to_owned(),
        })?;
    for offset in 0..rank {
        let left_dimension = left
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| left.get(index))
            .copied()
            .unwrap_or(1);
        let right_dimension = right
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| right.get(index))
            .copied()
            .unwrap_or(1);
        let dimension = if left_dimension == right_dimension {
            left_dimension
        } else if left_dimension == 1 {
            right_dimension
        } else if right_dimension == 1 {
            left_dimension
        } else {
            return Err(TensorError::Faulted {
                reason: format!(
                    "tensor shapes are not broadcast-compatible: {left:?} and {right:?}"
                ),
            });
        };
        shape.push(dimension);
    }
    shape.reverse();
    Ok(shape)
}

pub(crate) fn broadcast_indices(
    output: &[u64],
    input_shape: &[u64],
) -> Result<Vec<u64>, TensorError> {
    if input_shape.len() > output.len() {
        return Err(TensorError::ShapeOverflow);
    }
    let offset = output
        .len()
        .checked_sub(input_shape.len())
        .ok_or(TensorError::ShapeOverflow)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(input_shape.len())
        .map_err(|_| TensorError::Faulted {
            reason: "binary broadcast index allocation failed".to_owned(),
        })?;
    for (dimension_index, dimension) in input_shape.iter().enumerate() {
        let output_index = output
            .get(offset + dimension_index)
            .copied()
            .ok_or(TensorError::ShapeOverflow)?;
        indices.push(if *dimension == 1 { 0 } else { output_index });
    }
    Ok(indices)
}

fn binary_output_dtype(operation: BinaryOperation) -> DType {
    match operation {
        BinaryOperation::Equal
        | BinaryOperation::Less
        | BinaryOperation::LessEqual
        | BinaryOperation::Greater
        | BinaryOperation::GreaterEqual
        | BinaryOperation::LogicalAnd
        | BinaryOperation::LogicalOr => DType::Bool,
        _ => DType::F32,
    }
}

fn write_binary(
    write: &mut TensorWrite<'_>,
    indices: &[u64],
    operation: BinaryOperation,
    left: f32,
    right: f32,
) -> Result<(), TensorError> {
    if binary_output_dtype(operation) == DType::Bool {
        let value = match operation {
            BinaryOperation::Equal => left == right,
            BinaryOperation::Less => left < right,
            BinaryOperation::LessEqual => left <= right,
            BinaryOperation::Greater => left > right,
            BinaryOperation::GreaterEqual => left >= right,
            BinaryOperation::LogicalAnd => left != 0.0 && right != 0.0,
            BinaryOperation::LogicalOr => left != 0.0 || right != 0.0,
            _ => {
                return Err(TensorError::Faulted {
                    reason: "binary output classification changed during dispatch".to_owned(),
                });
            }
        };
        return write_element(write, indices, &[u8::from(value)]);
    }
    let value = match operation {
        BinaryOperation::Add => left + right,
        BinaryOperation::Subtract => left - right,
        BinaryOperation::Multiply => left * right,
        BinaryOperation::Divide => left / right,
        BinaryOperation::Remainder => left - (left / right).floor() * right,
        BinaryOperation::FloatingRemainder => left % right,
        BinaryOperation::Atan2 => left.atan2(right),
        BinaryOperation::LogAddExp if left.is_nan() || right.is_nan() => f32::NAN,
        BinaryOperation::LogAddExp if left == right && left.is_infinite() => left,
        BinaryOperation::LogAddExp => left.max(right) + (-(left - right).abs()).exp().ln_1p(),
        BinaryOperation::Power => left.powf(right),
        BinaryOperation::Minimum if left.is_nan() || right.is_nan() => f32::NAN,
        BinaryOperation::Minimum => left.min(right),
        BinaryOperation::Maximum if left.is_nan() || right.is_nan() => f32::NAN,
        BinaryOperation::Maximum => left.max(right),
        _ => {
            return Err(TensorError::Faulted {
                reason: "binary output classification changed during dispatch".to_owned(),
            });
        }
    };
    write_element(write, indices, &value.to_ne_bytes())
}

enum IndexMappingError {
    Invalid(String),
    Unsupported(String),
}

enum IndexMapping {
    Select { dimension: usize, index: u64 },
    Narrow { dimension: usize, start: u64 },
}

impl IndexMapping {
    fn new(
        operation: &IndexSpec,
        input: &TensorDescriptor,
        output: &TensorDescriptor,
    ) -> Result<Self, IndexMappingError> {
        match operation {
            IndexSpec::Select { dimension, index } => {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    IndexMappingError::Invalid(format!("select dimension {dimension} is too large"))
                })?;
                let size = input.shape().get(dimension).copied().ok_or_else(|| {
                    IndexMappingError::Invalid(format!(
                        "select dimension {dimension} is outside rank {}",
                        input.rank()
                    ))
                })?;
                let index = normalize_index(*index, size)?;
                let mut expected = input.shape().to_vec();
                expected.remove(dimension);
                if expected != output.shape() {
                    return Err(IndexMappingError::Invalid(format!(
                        "select output shape must be {expected:?}, got {:?}",
                        output.shape()
                    )));
                }
                Ok(Self::Select { dimension, index })
            }
            IndexSpec::Narrow {
                dimension,
                start,
                length,
            } => {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    IndexMappingError::Invalid(format!("narrow dimension {dimension} is too large"))
                })?;
                let expected = input
                    .narrowed_view(dimension, *start, *length)
                    .map_err(|error| IndexMappingError::Invalid(error.to_string()))?;
                if expected.shape() != output.shape() {
                    return Err(IndexMappingError::Invalid(format!(
                        "narrow output shape must be {:?}, got {:?}",
                        expected.shape(),
                        output.shape()
                    )));
                }
                let size = input.shape().get(dimension).copied().ok_or_else(|| {
                    IndexMappingError::Invalid(format!(
                        "narrow dimension {dimension} is outside rank {}",
                        input.rank()
                    ))
                })?;
                let start = normalize_narrow_range(size, *start, *length)
                    .map_err(|error| IndexMappingError::Invalid(error.to_string()))?
                    .start;
                Ok(Self::Narrow { dimension, start })
            }
            IndexSpec::Gather { .. } | IndexSpec::Scatter { .. } | IndexSpec::MaskedSelect => {
                Err(IndexMappingError::Unsupported(
                    "gather, scatter, and masked-select are owned by tensor-operation breadth tasks"
                        .to_owned(),
                ))
            }
        }
    }

    fn source_indices(&self, output: &[u64]) -> Result<Vec<u64>, TensorError> {
        match self {
            Self::Select { dimension, index } => {
                let mut source = output.to_vec();
                if *dimension > source.len() {
                    return Err(TensorError::ShapeOverflow);
                }
                source.insert(*dimension, *index);
                Ok(source)
            }
            Self::Narrow { dimension, start } => {
                let mut source = output.to_vec();
                let index = source
                    .get_mut(*dimension)
                    .ok_or(TensorError::ShapeOverflow)?;
                *index = index
                    .checked_add(*start)
                    .ok_or(TensorError::ShapeOverflow)?;
                Ok(source)
            }
        }
    }
}

fn normalize_index(index: i64, size: u64) -> Result<u64, IndexMappingError> {
    let size = i128::from(size);
    let index = if index < 0 {
        size.checked_add(i128::from(index))
    } else {
        Some(i128::from(index))
    }
    .ok_or_else(|| IndexMappingError::Invalid("index normalization overflowed".to_owned()))?;
    if index < 0 || index >= size {
        return Err(IndexMappingError::Invalid(format!(
            "index {index} is outside dimension size {size}"
        )));
    }
    u64::try_from(index)
        .map_err(|_| IndexMappingError::Invalid("normalized index is too large".to_owned()))
}

struct ResizeGeometry {
    input_height: u64,
    input_width: u64,
    crop_y: u64,
    crop_x: u64,
    crop_height: u64,
    crop_width: u64,
    output_height: u64,
    output_width: u64,
    mode: ResizeMode,
    antialias: bool,
    align_corners: bool,
}

impl ResizeGeometry {
    fn new(
        input: &TensorDescriptor,
        output: &TensorDescriptor,
        operation: ResizeSpec,
    ) -> Result<Self, TensorError> {
        let [input_batch, input_channels, input_height, input_width] = input.shape() else {
            return Err(TensorError::Faulted {
                reason: format!(
                    "resize input must be NCHW rank four, got {:?}",
                    input.shape()
                ),
            });
        };
        let [output_batch, output_channels, output_height, output_width] = output.shape() else {
            return Err(TensorError::Faulted {
                reason: format!(
                    "resize output must be NCHW rank four, got {:?}",
                    output.shape()
                ),
            });
        };
        if input_batch != output_batch || input_channels != output_channels {
            return Err(TensorError::Faulted {
                reason: "resize preserves batch and channel dimensions".to_owned(),
            });
        }
        if operation.width == 0 || operation.height == 0 {
            return Err(TensorError::Faulted {
                reason: "resize width and height must be non-zero".to_owned(),
            });
        }
        if *output_width != operation.width || *output_height != operation.height {
            return Err(TensorError::Faulted {
                reason: format!(
                    "resize output shape is {}x{}, but operation requests {}x{}",
                    output_width, output_height, operation.width, operation.height
                ),
            });
        }
        if *input_width == 0 || *input_height == 0 {
            return Err(TensorError::Faulted {
                reason: "resize input spatial dimensions must be non-zero".to_owned(),
            });
        }
        let (crop_x, crop_y) = match operation.crop {
            ResizeCrop::Disabled => (0, 0),
            ResizeCrop::Center => center_crop(
                *input_width,
                *input_height,
                operation.width,
                operation.height,
            )?,
        };
        let crop_width = input_width
            .checked_sub(crop_x.saturating_mul(2))
            .ok_or(TensorError::ShapeOverflow)?;
        let crop_height = input_height
            .checked_sub(crop_y.saturating_mul(2))
            .ok_or(TensorError::ShapeOverflow)?;
        if crop_width == 0 || crop_height == 0 {
            return Err(TensorError::Faulted {
                reason: "center crop produced an empty image".to_owned(),
            });
        }
        Ok(Self {
            input_height: *input_height,
            input_width: *input_width,
            crop_y,
            crop_x,
            crop_height,
            crop_width,
            output_height: operation.height,
            output_width: operation.width,
            mode: operation.mode,
            antialias: operation.antialias,
            align_corners: operation.align_corners,
        })
    }

    fn sample(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
        cancellation: &CancellationToken,
    ) -> Result<f32, TensorError> {
        cancellation.check()?;
        match self.mode {
            ResizeMode::NearestExact => {
                let x = nearest_exact(output_x, self.crop_width, self.output_width)?;
                let y = nearest_exact(output_y, self.crop_height, self.output_height)?;
                self.read(input, batch, channel, y, x)
            }
            ResizeMode::Bilinear if self.antialias => {
                self.antialiased(input, batch, channel, output_y, output_x, false)
            }
            ResizeMode::Bilinear => self.bilinear(input, batch, channel, output_y, output_x),
            ResizeMode::Area => self.area(input, batch, channel, output_y, output_x, cancellation),
            ResizeMode::Bicubic if self.antialias => {
                self.antialiased(input, batch, channel, output_y, output_x, true)
            }
            ResizeMode::Bicubic => self.bicubic(input, batch, channel, output_y, output_x),
            ResizeMode::Lanczos => {
                self.pillow_lanczos(input, batch, channel, output_y, output_x, cancellation)
            }
        }
    }

    fn read(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        y: u64,
        x: u64,
    ) -> Result<f32, TensorError> {
        let y = self
            .crop_y
            .checked_add(y)
            .ok_or(TensorError::ShapeOverflow)?;
        let x = self
            .crop_x
            .checked_add(x)
            .ok_or(TensorError::ShapeOverflow)?;
        if y >= self.input_height || x >= self.input_width {
            return Err(TensorError::IndexOutOfBounds {
                dimension: if y >= self.input_height { 2 } else { 3 },
                index: if y >= self.input_height { y } else { x },
                size: if y >= self.input_height {
                    self.input_height
                } else {
                    self.input_width
                },
            });
        }
        read_f32(input, &[batch, channel, y, x])
    }

    fn bilinear(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
    ) -> Result<f32, TensorError> {
        let (y0, y1, y_weight) = linear_coordinates(
            output_y,
            self.crop_height,
            self.output_height,
            self.align_corners,
        )?;
        let (x0, x1, x_weight) = linear_coordinates(
            output_x,
            self.crop_width,
            self.output_width,
            self.align_corners,
        )?;
        let top = lerp(
            self.read(input, batch, channel, y0, x0)?,
            self.read(input, batch, channel, y0, x1)?,
            x_weight,
        );
        let bottom = lerp(
            self.read(input, batch, channel, y1, x0)?,
            self.read(input, batch, channel, y1, x1)?,
            x_weight,
        );
        Ok(lerp(top, bottom, y_weight))
    }

    fn antialiased(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
        cubic: bool,
    ) -> Result<f32, TensorError> {
        let y_weights =
            antialias_axis_weights(output_y, self.crop_height, self.output_height, cubic)?;
        let x_weights =
            antialias_axis_weights(output_x, self.crop_width, self.output_width, cubic)?;
        // Torch's antialiased CPU resize rounds the horizontal pass before the vertical pass.
        let horizontal = |y| -> Result<f32, TensorError> {
            let ((first_x, first_weight), remaining) =
                x_weights
                    .split_first()
                    .ok_or_else(|| TensorError::Faulted {
                        reason: "resize antialias horizontal filter is empty".to_owned(),
                    })?;
            let mut value = self.read(input, batch, channel, y, *first_x)? * first_weight;
            for (x, weight) in remaining {
                value += self.read(input, batch, channel, y, *x)? * weight;
            }
            Ok(value)
        };
        let ((first_y, first_weight), remaining) =
            y_weights
                .split_first()
                .ok_or_else(|| TensorError::Faulted {
                    reason: "resize antialias vertical filter is empty".to_owned(),
                })?;
        let mut value = horizontal(*first_y)? * first_weight;
        for (y, weight) in remaining {
            value += horizontal(*y)? * weight;
        }
        Ok(value)
    }

    fn area(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
        cancellation: &CancellationToken,
    ) -> Result<f32, TensorError> {
        let y_start = output_y
            .checked_mul(self.crop_height)
            .ok_or(TensorError::ShapeOverflow)?
            / self.output_height;
        let y_end = div_ceil(
            output_y
                .checked_add(1)
                .and_then(|value| value.checked_mul(self.crop_height))
                .ok_or(TensorError::ShapeOverflow)?,
            self.output_height,
        )?;
        let x_start = output_x
            .checked_mul(self.crop_width)
            .ok_or(TensorError::ShapeOverflow)?
            / self.output_width;
        let x_end = div_ceil(
            output_x
                .checked_add(1)
                .and_then(|value| value.checked_mul(self.crop_width))
                .ok_or(TensorError::ShapeOverflow)?,
            self.output_width,
        )?;
        let mut sum = 0_f64;
        let mut count = 0_u64;
        for y in y_start..y_end {
            cancellation.check()?;
            for x in x_start..x_end {
                cancellation.check()?;
                sum += f64::from(self.read(input, batch, channel, y, x)?);
                count = count.checked_add(1).ok_or(TensorError::ShapeOverflow)?;
            }
        }
        if count == 0 {
            return Err(TensorError::Faulted {
                reason: "area resize selected no source pixels".to_owned(),
            });
        }
        Ok((sum / count as f64) as f32)
    }

    fn bicubic(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
    ) -> Result<f32, TensorError> {
        let source_y = source_coordinate(output_y, self.crop_height, self.output_height)?;
        let source_x = source_coordinate(output_x, self.crop_width, self.output_width)?;
        let y_base = source_y.floor() as i64;
        let x_base = source_x.floor() as i64;
        let mut value = 0_f64;
        for y_offset in -1_i64..=2 {
            let y = clamp_coordinate(y_base + y_offset, self.crop_height)?;
            let y_weight = cubic_weight(source_y - (y_base + y_offset) as f64);
            for x_offset in -1_i64..=2 {
                let x = clamp_coordinate(x_base + x_offset, self.crop_width)?;
                let x_weight = cubic_weight(source_x - (x_base + x_offset) as f64);
                value += f64::from(self.read(input, batch, channel, y, x)?) * y_weight * x_weight;
            }
        }
        Ok(value as f32)
    }

    fn pillow_lanczos(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
        cancellation: &CancellationToken,
    ) -> Result<f32, TensorError> {
        let mut value =
            self.pillow_lanczos_byte(input, batch, channel, output_y, output_x, cancellation)?;
        if input.descriptor().shape().get(1) == Some(&4) && channel < 3 {
            let alpha =
                self.pillow_lanczos_byte(input, batch, 3, output_y, output_x, cancellation)?;
            if alpha != 0 && alpha != 255 {
                value = ((255 * u32::from(value)) / u32::from(alpha)).min(255) as u8;
            }
        }
        Ok(f32::from(value) / 255.0)
    }

    fn pillow_lanczos_byte(
        &self,
        input: &Tensor,
        batch: u64,
        channel: u64,
        output_y: u64,
        output_x: u64,
        cancellation: &CancellationToken,
    ) -> Result<u8, TensorError> {
        let horizontal =
            PillowCoefficients::new(output_x, self.crop_width, self.output_width, cancellation)?;
        let vertical =
            PillowCoefficients::new(output_y, self.crop_height, self.output_height, cancellation)?;
        let mut vertical_sum = PILLOW_ROUNDING_BIAS;
        for source_y in vertical.indices() {
            cancellation.check()?;
            let mut horizontal_sum = PILLOW_ROUNDING_BIAS;
            for source_x in horizontal.indices() {
                cancellation.check()?;
                let mut source =
                    pillow_input_byte(self.read(input, batch, channel, source_y, source_x)?);
                if input.descriptor().shape().get(1) == Some(&4) && channel < 3 {
                    // Pillow filters RGBA in premultiplied byte space, including
                    // the rounded conversion before either separable pass.
                    let alpha = pillow_input_byte(self.read(input, batch, 3, source_y, source_x)?);
                    let product = u32::from(source) * u32::from(alpha) + 128;
                    source = (((product >> 8) + product) >> 8) as u8;
                }
                let weighted = i128::from(source)
                    .checked_mul(i128::from(horizontal.quantized_weight(source_x)?))
                    .ok_or(TensorError::ShapeOverflow)?;
                horizontal_sum = horizontal_sum
                    .checked_add(weighted)
                    .ok_or(TensorError::ShapeOverflow)?;
            }
            let horizontal_byte = pillow_clip(horizontal_sum);
            let weighted = i128::from(horizontal_byte)
                .checked_mul(i128::from(vertical.quantized_weight(source_y)?))
                .ok_or(TensorError::ShapeOverflow)?;
            vertical_sum = vertical_sum
                .checked_add(weighted)
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok(pillow_clip(vertical_sum))
    }
}

const PILLOW_PRECISION_BITS: u32 = 22;
const PILLOW_ROUNDING_BIAS: i128 = 1_i128 << (PILLOW_PRECISION_BITS - 1);

struct PillowCoefficients {
    start: u64,
    end: u64,
    center: f64,
    filter_scale: f64,
    normalization: f64,
}

impl PillowCoefficients {
    fn new(
        output: u64,
        input_size: u64,
        output_size: u64,
        cancellation: &CancellationToken,
    ) -> Result<Self, TensorError> {
        if output >= output_size || input_size == 0 || output_size == 0 {
            return Err(TensorError::ShapeOverflow);
        }
        let scale = input_size as f64 / output_size as f64;
        let filter_scale = scale.max(1.0);
        let support = 3.0 * filter_scale;
        let center = (output as f64 + 0.5) * scale;
        if !scale.is_finite() || !support.is_finite() || !center.is_finite() {
            return Err(TensorError::ShapeOverflow);
        }
        let start = (center - support + 0.5).trunc().max(0.0) as u64;
        let end = (center + support + 0.5).trunc().min(input_size as f64) as u64;
        if start >= end {
            return Err(TensorError::Faulted {
                reason: "Pillow Lanczos resize selected no source pixels".to_owned(),
            });
        }
        let mut normalization = 0.0;
        for source in start..end {
            cancellation.check()?;
            normalization += pillow_lanczos_weight((source as f64 - center + 0.5) / filter_scale);
        }
        if normalization == 0.0 || !normalization.is_finite() {
            return Err(TensorError::Faulted {
                reason: "Pillow Lanczos resize produced an invalid weight sum".to_owned(),
            });
        }
        Ok(Self {
            start,
            end,
            center,
            filter_scale,
            normalization,
        })
    }

    fn indices(&self) -> std::ops::Range<u64> {
        self.start..self.end
    }

    fn quantized_weight(&self, source: u64) -> Result<i64, TensorError> {
        if source < self.start || source >= self.end {
            return Err(TensorError::IndexOutOfBounds {
                dimension: 0,
                index: source,
                size: self.end,
            });
        }
        let normalized =
            pillow_lanczos_weight((source as f64 - self.center + 0.5) / self.filter_scale)
                / self.normalization;
        let scaled = normalized * (1_u64 << PILLOW_PRECISION_BITS) as f64 + 0.5;
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err(TensorError::ShapeOverflow);
        }
        Ok(scaled as i64)
    }
}

fn pillow_input_byte(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0) as u8
}

fn pillow_clip(sum: i128) -> u8 {
    (sum >> PILLOW_PRECISION_BITS).clamp(0, i128::from(u8::MAX)) as u8
}

fn center_crop(
    input_width: u64,
    input_height: u64,
    output_width: u64,
    output_height: u64,
) -> Result<(u64, u64), TensorError> {
    let old_aspect = input_width as f64 / input_height as f64;
    let new_aspect = output_width as f64 / output_height as f64;
    if old_aspect > new_aspect {
        let removed = input_width as f64 - input_width as f64 * (new_aspect / old_aspect);
        Ok((round_ties_even_nonnegative(removed / 2.0)?, 0))
    } else if old_aspect < new_aspect {
        let removed = input_height as f64 - input_height as f64 * (old_aspect / new_aspect);
        Ok((0, round_ties_even_nonnegative(removed / 2.0)?))
    } else {
        Ok((0, 0))
    }
}

fn round_ties_even_nonnegative(value: f64) -> Result<u64, TensorError> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(TensorError::ShapeOverflow);
    }
    Ok(value.round_ties_even() as u64)
}

fn nearest_exact(output: u64, input_size: u64, output_size: u64) -> Result<u64, TensorError> {
    let numerator = output
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_mul(input_size))
        .ok_or(TensorError::ShapeOverflow)?;
    let denominator = output_size
        .checked_mul(2)
        .ok_or(TensorError::ShapeOverflow)?;
    Ok((numerator / denominator).min(input_size.saturating_sub(1)))
}

fn source_coordinate(output: u64, input_size: u64, output_size: u64) -> Result<f64, TensorError> {
    if output >= output_size || input_size == 0 || output_size == 0 {
        return Err(TensorError::ShapeOverflow);
    }
    Ok((output as f64 + 0.5) * input_size as f64 / output_size as f64 - 0.5)
}

fn linear_coordinates(
    output: u64,
    input_size: u64,
    output_size: u64,
    align_corners: bool,
) -> Result<(u64, u64, f32), TensorError> {
    let coordinate = if align_corners {
        if output >= output_size || input_size == 0 || output_size == 0 {
            return Err(TensorError::ShapeOverflow);
        }
        if output_size == 1 {
            0.0
        } else {
            output as f64 * input_size.saturating_sub(1) as f64
                / output_size.saturating_sub(1) as f64
        }
    } else {
        source_coordinate(output, input_size, output_size)?.max(0.0)
    };
    let lower = (coordinate.floor() as u64).min(input_size.saturating_sub(1));
    let upper = lower.saturating_add(1).min(input_size.saturating_sub(1));
    Ok((lower, upper, (coordinate - lower as f64) as f32))
}

fn antialias_axis_weights(
    output: u64,
    input_size: u64,
    output_size: u64,
    cubic: bool,
) -> Result<Vec<(u64, f32)>, TensorError> {
    if input_size == 0 || output_size == 0 {
        return Err(TensorError::ShapeOverflow);
    }
    let scale = (input_size as f64 / output_size as f64) as f32;
    let center = scale * (output as f32 + 0.5);
    let filter_scale = scale.max(1.0);
    let base_support = if cubic { 2.0 } else { 1.0 };
    let support = base_support * filter_scale;
    let input_end = i64::try_from(input_size).map_err(|_| TensorError::ShapeOverflow)?;
    // Antialias clips the filter window and normalizes its surviving weights; it does not repeat border samples.
    let start = ((center - support + 0.5) as i64).max(0);
    let end = ((center + support + 0.5) as i64).min(input_end);
    let mut weights = Vec::new();
    let capacity =
        usize::try_from(end.saturating_sub(start)).map_err(|_| TensorError::ShapeOverflow)?;
    weights
        .try_reserve_exact(capacity)
        .map_err(|error| TensorError::Faulted {
            reason: format!("resize antialias weight allocation failed: {error}"),
        })?;
    let mut normalization = 0.0_f32;
    for source in start..end {
        let distance = (source as f32 - center + 0.5) / filter_scale;
        let weight = if cubic {
            antialias_cubic_weight(distance)
        } else {
            (1.0 - distance.abs()).max(0.0)
        };
        if weight == 0.0 {
            continue;
        }
        weights.push((
            u64::try_from(source).map_err(|_| TensorError::ShapeOverflow)?,
            weight,
        ));
        normalization += weight;
    }
    if normalization == 0.0 || !normalization.is_finite() {
        return Err(TensorError::Faulted {
            reason: "resize antialias filter produced an invalid weight sum".to_owned(),
        });
    }
    Ok(weights
        .into_iter()
        .map(|(source, weight)| (source, weight / normalization))
        .collect())
}

fn antialias_cubic_weight(value: f32) -> f32 {
    let absolute = value.abs();
    // Torch uses Keys' coefficient for antialiasing and -0.75 for ordinary bicubic interpolation.
    const COEFFICIENT: f32 = -0.5;
    if absolute < 1.0 {
        ((COEFFICIENT + 2.0) * absolute - (COEFFICIENT + 3.0)) * absolute * absolute + 1.0
    } else if absolute < 2.0 {
        (((absolute - 5.0) * absolute + 8.0) * absolute - 4.0) * COEFFICIENT
    } else {
        0.0
    }
}

fn lerp(left: f32, right: f32, weight: f32) -> f32 {
    left + (right - left) * weight
}

fn div_ceil(numerator: u64, denominator: u64) -> Result<u64, TensorError> {
    if denominator == 0 {
        return Err(TensorError::ShapeOverflow);
    }
    Ok(numerator / denominator + u64::from(!numerator.is_multiple_of(denominator)))
}

fn clamp_coordinate(coordinate: i64, size: u64) -> Result<u64, TensorError> {
    let maximum = i64::try_from(size.saturating_sub(1)).map_err(|_| TensorError::ShapeOverflow)?;
    u64::try_from(coordinate.clamp(0, maximum)).map_err(|_| TensorError::ShapeOverflow)
}

fn cubic_weight(value: f64) -> f64 {
    let absolute = value.abs();
    const A: f64 = -0.75;
    if absolute <= 1.0 {
        (A + 2.0) * absolute.powi(3) - (A + 3.0) * absolute.powi(2) + 1.0
    } else if absolute < 2.0 {
        A * absolute.powi(3) - 5.0 * A * absolute.powi(2) + 8.0 * A * absolute - 4.0 * A
    } else {
        0.0
    }
}

fn pillow_lanczos_weight(value: f64) -> f64 {
    let absolute = value.abs();
    if absolute == 0.0 {
        1.0
    } else if absolute >= 3.0 {
        0.0
    } else {
        let pi_value = std::f64::consts::PI * value;
        (pi_value.sin() / pi_value) * ((pi_value / 3.0).sin() / (pi_value / 3.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamId;
    use std::sync::atomic::AtomicUsize;

    fn descriptor(shape: Vec<u64>) -> TensorDescriptor {
        TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)
            .expect("test descriptor")
    }

    fn typed_descriptor(shape: Vec<u64>, dtype: DType) -> TensorDescriptor {
        TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)
            .expect("typed test descriptor")
    }

    struct UnsupportedTensorBackend {
        capabilities: BackendCapabilityMatrix,
        primitive_calls: AtomicUsize,
    }

    impl UnsupportedTensorBackend {
        fn new() -> Result<Self, TensorError> {
            Ok(Self {
                capabilities: BackendCapabilityMatrix::new(
                    DeviceId::new(sim_types::DeviceKind::Metal, 0),
                    Vec::new(),
                    Vec::new(),
                )?,
                primitive_calls: AtomicUsize::new(0),
            })
        }

        fn unsupported<T>(&self) -> Result<T, TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            Err(TensorError::UnsupportedCapability {
                operation: "test.unexpected-primitive".to_owned(),
                device: self.device(),
                reason: "a default adapter attempted a primitive fallback".to_owned(),
            })
        }
    }

    impl CachedAllocationOwner for UnsupportedTensorBackend {
        fn cache_device(&self) -> DeviceId {
            self.device()
        }

        fn release_cached_allocations(
            &self,
            cancellation: &CancellationToken,
        ) -> Result<u64, TensorError> {
            cancellation.check()?;
            Ok(0)
        }
    }

    impl TensorBackend for UnsupportedTensorBackend {
        fn device(&self) -> DeviceId {
            DeviceId::new(sim_types::DeviceKind::Metal, 0)
        }

        fn capabilities(&self) -> &BackendCapabilityMatrix {
            &self.capabilities
        }

        fn allocate(
            &self,
            _descriptor: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn copy(
            &self,
            _source: &Tensor,
            _destination: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn record_event(&self, _context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
            self.unsupported()
        }

        fn wait_event(
            &self,
            _event: EventFence,
            _context: &ExecutionContext<'_>,
        ) -> Result<(), TensorError> {
            self.unsupported()
        }

        fn fill(
            &self,
            _value: Scalar,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn unary(
            &self,
            _operation: UnaryOperation,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn binary(
            &self,
            _operation: BinaryOperation,
            _left: &Tensor,
            _right: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn binary_scalar(
            &self,
            _operation: BinaryOperation,
            _input: &Tensor,
            _scalar: Scalar,
            _scalar_side: ScalarSide,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn reduction(
            &self,
            _operation: &ReductionSpec,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn indexing(
            &self,
            _operation: &IndexSpec,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn resize(
            &self,
            _operation: ResizeSpec,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn convolution(
            &self,
            _operation: &ConvolutionSpec,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn linear_algebra(
            &self,
            _operation: LinearAlgebraOperation,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unsupported()
        }

        fn custom_kernel(
            &self,
            _kernel: &CustomKernelId,
            _inputs: &[Tensor],
            _outputs: &[TensorDescriptor],
            _context: &ExecutionContext<'_>,
        ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
            self.unsupported()
        }
    }

    fn shape_for_layout(layout: Layout) -> Vec<u64> {
        match layout {
            Layout::Contiguous | Layout::Strided => vec![1],
            Layout::ChannelsLast => vec![1, 1, 1, 1],
            Layout::ChannelsLast3d => vec![1, 1, 1, 1, 1],
        }
    }

    fn formatted_descriptor(
        shape: Vec<u64>,
        dtype: DType,
        layout: Layout,
    ) -> Result<TensorDescriptor, TensorError> {
        match layout {
            Layout::Contiguous => {
                TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)
            }
            Layout::ChannelsLast => {
                TensorDescriptor::channels_last(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)
            }
            Layout::ChannelsLast3d => {
                TensorDescriptor::channels_last_3d(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)
            }
            Layout::Strided => {
                let strides = TensorDescriptor::contiguous(
                    shape.clone(),
                    dtype,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?
                .strides()
                .to_vec();
                TensorDescriptor::new_strided(
                    shape,
                    strides,
                    0,
                    dtype,
                    Layout::Strided,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )
            }
        }
    }

    fn zero_tensor(descriptor: TensorDescriptor) -> Result<Tensor, TensorError> {
        let bytes =
            usize::try_from(descriptor.byte_len()?).map_err(|_| TensorError::ShapeOverflow)?;
        Tensor::from_bytes(descriptor, vec![0; bytes])
    }

    fn output_dtype(operation: PrimitiveOperation) -> DType {
        match operation {
            PrimitiveOperation::Unary(UnaryOperation::IsFinite) => DType::Bool,
            PrimitiveOperation::Binary(operation) | PrimitiveOperation::BinaryScalar(operation) => {
                binary_output_dtype(operation)
            }
            _ => DType::F32,
        }
    }

    fn execute_advertised_support(
        backend: &CpuBackend,
        support: OperationSupport,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        let primitive = support.primitive();
        if primitive == PrimitiveOperation::RecordEvent {
            backend.record_event(context)?;
            return Ok(());
        }
        if primitive == PrimitiveOperation::WaitEvent {
            let event = backend.record_event(context)?;
            backend.wait_event(event, context)?;
            return Ok(());
        }
        let role = support.role().ok_or_else(|| TensorError::Faulted {
            reason: "tensor capability omitted its role".to_owned(),
        })?;
        let dtype = support.dtype().ok_or_else(|| TensorError::Faulted {
            reason: "tensor capability omitted its dtype".to_owned(),
        })?;
        let layout = support.layout().ok_or_else(|| TensorError::Faulted {
            reason: "tensor capability omitted its layout".to_owned(),
        })?;
        let advertised_shape = match primitive {
            PrimitiveOperation::LinearAlgebra(LinearAlgebraOperation::BatchMatrixMultiply) => {
                vec![1, 1, 1]
            }
            PrimitiveOperation::Convolution if layout == Layout::ChannelsLast3d => {
                vec![1, 1, 1, 1, 1]
            }
            PrimitiveOperation::Convolution => vec![1, 1, 1, 1],
            _ => shape_for_layout(layout),
        };
        let advertised_descriptor = formatted_descriptor(advertised_shape.clone(), dtype, layout)?;

        match primitive {
            PrimitiveOperation::Allocation => {
                backend.allocate(advertised_descriptor, context)?;
            }
            PrimitiveOperation::Fill => {
                backend.fill(Scalar::Float(0.0), advertised_descriptor, context)?;
            }
            PrimitiveOperation::Copy => {
                let (input, output) = match role {
                    TensorRole::Input => (
                        zero_tensor(advertised_descriptor)?,
                        TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                    ),
                    TensorRole::Output => (
                        zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        advertised_descriptor,
                    ),
                };
                backend.copy(&input, output, context)?;
            }
            PrimitiveOperation::Unary(operation) => {
                let operation_output_dtype = if operation == UnaryOperation::IsFinite {
                    DType::Bool
                } else {
                    dtype
                };
                let (input, output) = match role {
                    TensorRole::Input => (
                        zero_tensor(advertised_descriptor)?,
                        TensorDescriptor::contiguous(
                            advertised_shape,
                            operation_output_dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                    ),
                    TensorRole::Output => (
                        zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            if operation == UnaryOperation::IsFinite {
                                DType::F32
                            } else {
                                dtype
                            },
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        advertised_descriptor,
                    ),
                };
                backend.unary(operation, &input, output, context)?;
            }
            PrimitiveOperation::Binary(operation) => {
                let operation_output_dtype = output_dtype(primitive);
                let (left, right, output) = match role {
                    TensorRole::Input => {
                        let left = zero_tensor(advertised_descriptor)?;
                        let right = left.clone();
                        let output = TensorDescriptor::contiguous(
                            advertised_shape,
                            operation_output_dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?;
                        (left, right, output)
                    }
                    TensorRole::Output => {
                        let left = zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            DType::F32,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?;
                        let right = left.clone();
                        (left, right, advertised_descriptor)
                    }
                };
                backend.binary(operation, &left, &right, output, context)?;
            }
            PrimitiveOperation::BinaryScalar(operation) => {
                let operation_output_dtype = output_dtype(primitive);
                let (input, output) = match role {
                    TensorRole::Input => (
                        zero_tensor(advertised_descriptor)?,
                        TensorDescriptor::contiguous(
                            advertised_shape,
                            operation_output_dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                    ),
                    TensorRole::Output => (
                        zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            DType::F32,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        advertised_descriptor,
                    ),
                };
                backend.binary_scalar(
                    operation,
                    &input,
                    Scalar::Float(1.0),
                    ScalarSide::Right,
                    output,
                    context,
                )?;
            }
            PrimitiveOperation::Reduction(operation) => {
                let (input, output, dimensions, keep_dimensions) = match role {
                    TensorRole::Input => {
                        let input = zero_tensor(advertised_descriptor)?;
                        let dimension = input
                            .descriptor()
                            .rank()
                            .checked_sub(1)
                            .ok_or(TensorError::ShapeOverflow)?;
                        let mut output_shape = advertised_shape;
                        output_shape.remove(dimension);
                        let output_dtype = match operation {
                            ReductionOperation::All | ReductionOperation::Any => DType::Bool,
                            ReductionOperation::ArgMinimum | ReductionOperation::ArgMaximum => {
                                DType::I64
                            }
                            ReductionOperation::Sum
                                if matches!(
                                    dtype,
                                    DType::I64
                                        | DType::I32
                                        | DType::I16
                                        | DType::I8
                                        | DType::U64
                                        | DType::U32
                                        | DType::U16
                                        | DType::U8
                                        | DType::Bool
                                ) =>
                            {
                                DType::I64
                            }
                            _ => dtype,
                        };
                        (
                            input,
                            TensorDescriptor::contiguous(
                                output_shape,
                                output_dtype,
                                DeviceId::CPU,
                                StreamId::DEFAULT,
                            )?,
                            vec![u64::try_from(dimension).map_err(|_| TensorError::ShapeOverflow)?],
                            false,
                        )
                    }
                    TensorRole::Output => {
                        let mut input_shape = advertised_shape;
                        input_shape.push(2);
                        let input_dtype = match operation {
                            ReductionOperation::All | ReductionOperation::Any => DType::F32,
                            ReductionOperation::ArgMinimum | ReductionOperation::ArgMaximum => {
                                DType::F32
                            }
                            _ => dtype,
                        };
                        let dimension = input_shape
                            .len()
                            .checked_sub(1)
                            .ok_or(TensorError::ShapeOverflow)?;
                        (
                            zero_tensor(TensorDescriptor::contiguous(
                                input_shape,
                                input_dtype,
                                DeviceId::CPU,
                                StreamId::DEFAULT,
                            )?)?,
                            advertised_descriptor,
                            vec![u64::try_from(dimension).map_err(|_| TensorError::ShapeOverflow)?],
                            false,
                        )
                    }
                };
                backend.reduction(
                    &ReductionSpec {
                        operation,
                        dimensions,
                        keep_dimensions,
                        accumulation_dtype: matches!(
                            operation,
                            ReductionOperation::Sum
                                | ReductionOperation::Product
                                | ReductionOperation::Mean
                                | ReductionOperation::Variance
                                | ReductionOperation::StandardDeviation
                        )
                        .then_some(output.dtype()),
                        correction: 0,
                    },
                    &input,
                    output,
                    context,
                )?;
            }
            PrimitiveOperation::Select | PrimitiveOperation::Narrow => {
                let (input, output, operation) = match (primitive, role) {
                    (PrimitiveOperation::Select, TensorRole::Input) => {
                        let mut output_shape = advertised_shape;
                        output_shape.remove(0);
                        (
                            zero_tensor(advertised_descriptor)?,
                            TensorDescriptor::contiguous(
                                output_shape,
                                dtype,
                                DeviceId::CPU,
                                StreamId::DEFAULT,
                            )?,
                            IndexSpec::Select {
                                dimension: 0,
                                index: 0,
                            },
                        )
                    }
                    (PrimitiveOperation::Select, TensorRole::Output) => {
                        let mut input_shape = vec![1];
                        input_shape.extend_from_slice(&advertised_shape);
                        (
                            zero_tensor(TensorDescriptor::contiguous(
                                input_shape,
                                dtype,
                                DeviceId::CPU,
                                StreamId::DEFAULT,
                            )?)?,
                            advertised_descriptor,
                            IndexSpec::Select {
                                dimension: 0,
                                index: 0,
                            },
                        )
                    }
                    (PrimitiveOperation::Narrow, TensorRole::Input) => (
                        zero_tensor(advertised_descriptor)?,
                        TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                        IndexSpec::Narrow {
                            dimension: 0,
                            start: 0,
                            length: 1,
                        },
                    ),
                    (PrimitiveOperation::Narrow, TensorRole::Output) => (
                        zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        advertised_descriptor,
                        IndexSpec::Narrow {
                            dimension: 0,
                            start: 0,
                            length: 1,
                        },
                    ),
                    _ => return Err(TensorError::ShapeOverflow),
                };
                backend.indexing(&operation, &[input], output, context)?;
            }
            PrimitiveOperation::Resize(mode) => {
                let shape = vec![1, 1, 1, 1];
                let (input, output) = match role {
                    TensorRole::Input => (
                        zero_tensor(formatted_descriptor(shape.clone(), DType::F32, layout)?)?,
                        TensorDescriptor::contiguous(
                            shape,
                            DType::F32,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                    ),
                    TensorRole::Output => (
                        zero_tensor(TensorDescriptor::contiguous(
                            shape.clone(),
                            DType::F32,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        formatted_descriptor(shape, DType::F32, layout)?,
                    ),
                };
                backend.resize(
                    ResizeSpec {
                        width: 1,
                        height: 1,
                        mode,
                        crop: ResizeCrop::Disabled,
                        antialias: false,
                        align_corners: false,
                    },
                    &input,
                    output,
                    context,
                )?;
            }
            PrimitiveOperation::LinearAlgebra(LinearAlgebraOperation::BatchMatrixMultiply) => {
                let (left, right, output) = match role {
                    TensorRole::Input => {
                        let left = zero_tensor(advertised_descriptor)?;
                        let right =
                            zero_tensor(formatted_descriptor(vec![1, 1, 1], dtype, layout)?)?;
                        let output = TensorDescriptor::contiguous(
                            vec![1, 1, 1],
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?;
                        (left, right, output)
                    }
                    TensorRole::Output => {
                        let left = zero_tensor(TensorDescriptor::contiguous(
                            vec![1, 1, 1],
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?;
                        let right = left.clone();
                        (left, right, advertised_descriptor)
                    }
                };
                backend.linear_algebra(
                    LinearAlgebraOperation::BatchMatrixMultiply,
                    &[left, right],
                    output,
                    context,
                )?;
            }
            PrimitiveOperation::Convolution => {
                let spatial_dimensions = advertised_shape
                    .len()
                    .checked_sub(2)
                    .ok_or(TensorError::ShapeOverflow)?;
                let weight_shape = vec![1; advertised_shape.len()];
                let (input, output) = match role {
                    TensorRole::Input => (
                        zero_tensor(advertised_descriptor)?,
                        TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?,
                    ),
                    TensorRole::Output => (
                        zero_tensor(TensorDescriptor::contiguous(
                            advertised_shape,
                            dtype,
                            DeviceId::CPU,
                            StreamId::DEFAULT,
                        )?)?,
                        advertised_descriptor,
                    ),
                };
                let weight = zero_tensor(formatted_descriptor(weight_shape, dtype, layout)?)?;
                backend.convolution(
                    &ConvolutionSpec {
                        stride: vec![1; spatial_dimensions],
                        padding: vec![0; spatial_dimensions],
                        dilation: vec![1; spatial_dimensions],
                        groups: 1,
                        transposed: false,
                        output_padding: vec![0; spatial_dimensions],
                    },
                    &[input, weight],
                    output,
                    context,
                )?;
            }
            PrimitiveOperation::LinearAlgebra(_)
            | PrimitiveOperation::Gather
            | PrimitiveOperation::Scatter
            | PrimitiveOperation::MaskedSelect
            | PrimitiveOperation::CustomKernel => {
                return Err(TensorError::Faulted {
                    reason: "unadvertised primitive reached CPU capability execution".to_owned(),
                });
            }
            PrimitiveOperation::RecordEvent | PrimitiveOperation::WaitEvent => {
                return Err(TensorError::Faulted {
                    reason: "event capability reached tensor dispatch".to_owned(),
                });
            }
        }
        Ok(())
    }

    #[test]
    fn area_and_lanczos_sampling_check_cancellation_inside_source_loops() {
        let input = Tensor::from_bytes(descriptor(vec![1, 1, 64, 64]), vec![0; 64 * 64 * 4])
            .expect("test input");
        let output = descriptor(vec![1, 1, 1, 1]);
        let cancellation = CancellationToken::default();
        cancellation.cancel();

        for mode in [ResizeMode::Area, ResizeMode::Lanczos] {
            let geometry = ResizeGeometry::new(
                input.descriptor(),
                &output,
                ResizeSpec {
                    width: 1,
                    height: 1,
                    mode,
                    crop: ResizeCrop::Disabled,
                    antialias: false,
                    align_corners: false,
                },
            )
            .expect("resize geometry");
            let result = match mode {
                ResizeMode::Area => geometry.area(&input, 0, 0, 0, 0, &cancellation),
                ResizeMode::Lanczos => geometry.pillow_lanczos(&input, 0, 0, 0, 0, &cancellation),
                _ => unreachable!("test enumerates the two unbounded sampling modes"),
            };
            assert!(matches!(result, Err(TensorError::Cancelled)));
        }
    }

    #[test]
    fn every_advertised_cpu_primitive_signature_executes() -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for support in backend.capabilities().supported() {
            execute_advertised_support(&backend, *support, &context).map_err(|error| {
                TensorError::Faulted {
                    reason: format!("advertised capability {support:?} failed: {error}"),
                }
            })?;
        }
        Ok(())
    }

    #[test]
    fn backend_adapters_delegate_inverse_and_kronecker_to_canonical_owners()
    -> Result<(), Box<dyn std::error::Error>> {
        fn tensor(shape: Vec<u64>, values: &[f32]) -> Result<Tensor, TensorError> {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Tensor::from_bytes(descriptor(shape), bytes)
        }

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let matrix = tensor(vec![2, 2], &[2.0, 0.0, 0.0, 4.0])?;
        let direct_inverse = inverse_with_context_exact_native(&backend, &matrix, &context)?;
        let (adapted_inverse, inverse_event) = backend.matrix_inverse(&matrix, &context)?;
        backend.wait_event(inverse_event, &context)?;
        assert_eq!(
            direct_inverse.contiguous_bytes()?,
            adapted_inverse.contiguous_bytes()?
        );

        let left = tensor(vec![2], &[1.0, 2.0])?;
        let right = tensor(vec![2], &[3.0, 4.0])?;
        let direct_product = kron_with_context_exact_native(&backend, &left, &right, &context)?;
        let (adapted_product, product_event) =
            backend.kronecker_product(&left, &right, &context)?;
        backend.wait_event(product_event, &context)?;
        assert_eq!(
            direct_product.contiguous_bytes()?,
            adapted_product.contiguous_bytes()?
        );
        Ok(())
    }

    #[test]
    fn constructed_cpu_matrix_derives_properties_from_the_effective_budget()
    -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(257)?;
        let properties =
            backend
                .capabilities()
                .device_properties()
                .ok_or_else(|| TensorError::Faulted {
                    reason: "constructed CPU backend has no native properties".to_owned(),
                })?;
        assert_eq!(properties.device(), DeviceId::CPU);
        assert_eq!(properties.name(), "Zed native Rust CPU");
        assert_eq!(properties.total_memory_bytes(), 257);
        assert_eq!(properties.major(), 0);
        assert_eq!(properties.minor(), 0);
        assert_eq!(properties.architecture(), Some(std::env::consts::ARCH));
        assert!(!properties.has_fp16());
        assert_eq!(
            properties.total_memory_bytes(),
            authority.memory_snapshot().limit_bytes
        );

        let compatibility = CpuBackend::capability_matrix();
        assert_eq!(
            backend.capabilities().supported(),
            compatibility.supported()
        );
        assert_eq!(
            backend.capabilities().deterministic(),
            compatibility.deterministic()
        );
        assert!(compatibility.device_properties().is_none());
        assert!(matches!(
            CpuWorkspaceAuthority::create_backend(0),
            Err(TensorError::Faulted { .. })
        ));
        Ok(())
    }

    #[test]
    fn payload_upload_and_cast_adapters_match_canonical_cpu_bytes() -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let values = [1.5_f32, -2.25, 0.125, 32.0];
        for dtype in [DType::F16, DType::Bf16] {
            let (direct, event) = backend.upload_f32_payload(&[2, 2], &values, dtype, &context)?;
            backend.wait_event(event, &context)?;
            let canonical = tensor_from_f32_with_backend_exact_native(
                &backend,
                &[2, 2],
                &values,
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(map_operator_indirection_error)?;
            assert_eq!(direct.descriptor(), canonical.descriptor());
            assert_eq!(direct.contiguous_bytes()?, canonical.contiguous_bytes()?);
            drop(direct);
            drop(canonical);
        }

        let strided_descriptor = TensorDescriptor::new_strided(
            vec![2, 2],
            vec![1, 2],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let input = Tensor::from_bytes(
            strided_descriptor,
            [1.0_f32, 3.0, 2.0, 4.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
        )?;
        for (dtype, copy) in [(DType::F16, false), (DType::F32, true)] {
            let (direct, event) = backend.cast_tensor(&input, dtype, false, copy, &context)?;
            backend.wait_event(event, &context)?;
            let canonical = cast_to_with_backend_exact_native(
                &backend,
                &input,
                dtype,
                DeviceId::CPU,
                false,
                copy,
                &context,
            )
            .map_err(map_operator_indirection_error)?;
            assert_eq!(direct.descriptor(), canonical.descriptor());
            assert_eq!(direct.contiguous_bytes()?, canonical.contiguous_bytes()?);
            drop(direct);
            drop(canonical);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn payload_upload_and_cast_release_resources_on_cancel_and_oom() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(16)?,
            &cancellation,
        );
        let (output, event) =
            backend.upload_f32_payload(&[2], &[1.0, -2.0], DType::F16, &context)?;
        backend.wait_event(event, &context)?;
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 16);
        drop(output);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(16)?;
        let oom_context = oom_backend.execution_context(
            StreamId::DEFAULT,
            oom_authority.authorize_workspace(16)?,
            &cancellation,
        );
        assert!(matches!(
            oom_backend.upload_f32_payload(&[2], &[1.0, 2.0], DType::F16, &oom_context),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);
        assert_eq!(oom_backend.memory_snapshot().current_bytes, 0);

        let input = Tensor::from_bytes(descriptor(vec![2]), vec![0; 8])?;
        assert!(matches!(
            oom_backend.cast_tensor(&input, DType::F16, false, false, &oom_context),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);
        assert_eq!(oom_backend.memory_snapshot().current_bytes, 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(16)?,
            &cancelled,
        );
        assert!(matches!(
            backend.upload_f32_payload(&[1], &[1.0], DType::F32, &cancelled_context),
            Err(TensorError::Cancelled)
        ));
        assert!(matches!(
            backend.cast_tensor(&input, DType::F16, false, false, &cancelled_context),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        assert!(matches!(
            backend.upload_f32_payload(&[2], &[1.0], DType::F32, &context),
            Err(TensorError::StorageLength {
                expected: 8,
                actual: 4
            })
        ));
        Ok(())
    }

    #[test]
    fn patch_graph_adapters_match_canonical_cpu_owners() -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let input = Tensor::from_bytes(
            descriptor(vec![2, 2]),
            [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
        )?;

        let direct_pad = backend
            .constant_pad(
                &input,
                &[1, 0, 0, 1],
                Some(DecodedScalar::Real(-2.0)),
                &context,
            )?
            .0;
        let canonical_pad = functional_pad_with_context_exact_native(
            &backend,
            &input,
            &[1, 0, 0, 1],
            FunctionalPadMode::Constant,
            Some(DecodedScalar::Real(-2.0)),
            &context,
        )
        .map_err(map_shape_layout_transform_part_three_error)?;
        assert_eq!(direct_pad.descriptor(), canonical_pad.descriptor());
        assert_eq!(
            direct_pad.contiguous_bytes()?,
            canonical_pad.contiguous_bytes()?
        );
        drop(direct_pad);
        drop(canonical_pad);

        let direct_norm = backend
            .vector_norm(&input, 2.0, &[0, 1], false, Some(DType::F32), &context)?
            .0;
        let canonical_norm = vector_norm_with_context_exact_native(
            &backend,
            &input,
            2.0,
            &[0, 1],
            false,
            Some(DType::F32),
            &context,
        )
        .map_err(map_linear_algebra_part_one_error)?;
        assert_eq!(direct_norm.descriptor(), canonical_norm.descriptor());
        assert_eq!(
            direct_norm.contiguous_bytes()?,
            canonical_norm.contiguous_bytes()?
        );
        drop(direct_norm);
        drop(canonical_norm);

        let direct_eye = backend
            .eye(3, Some(2), DType::F32, Layout::Contiguous, &context)?
            .0;
        let canonical_eye = eye_with_context_exact_native(
            &backend,
            3,
            Some(2),
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            &context,
        )
        .map_err(map_tensor_creation_part_one_error)?;
        assert_eq!(direct_eye.descriptor(), canonical_eye.descriptor());
        assert_eq!(
            direct_eye.contiguous_bytes()?,
            canonical_eye.contiguous_bytes()?
        );
        Ok(())
    }

    #[test]
    fn rectangular_slice_replacement_preserves_strided_low_precision_bytes_and_resources()
    -> Result<(), TensorError> {
        for dtype in [DType::F16, DType::Bf16] {
            let input_descriptor = TensorDescriptor::new_strided(
                vec![2, 3],
                vec![1, 2],
                0,
                dtype,
                Layout::Strided,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?;
            let input = Tensor::from_bytes(input_descriptor, (0_u8..12).collect())?;
            let source = Tensor::from_bytes(
                typed_descriptor(vec![1, 2], dtype),
                vec![0xa1, 0xb2, 0xc3, 0xd4],
            )?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(128)?;
            let cancellation = CancellationToken::default();
            let context = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(16)?,
                &cancellation,
            );
            let output = backend
                .replace_rectangular_slice(&input, &source, &[1, 1], &context)?
                .0;
            assert_eq!(output.descriptor(), input.descriptor());
            assert_eq!(
                output.element_bytes(&[0, 0])?,
                input.element_bytes(&[0, 0])?
            );
            assert_eq!(
                output.element_bytes(&[0, 2])?,
                input.element_bytes(&[0, 2])?
            );
            assert_eq!(
                output.element_bytes(&[1, 0])?,
                input.element_bytes(&[1, 0])?
            );
            assert_eq!(output.element_bytes(&[1, 1])?, &[0xa1, 0xb2]);
            assert_eq!(output.element_bytes(&[1, 2])?, &[0xc3, 0xd4]);
            assert_eq!(input.element_bytes(&[1, 1])?, &[6, 7]);
            assert_eq!(context.scratch.in_use_bytes(), 0);
            assert_eq!(backend.memory_snapshot().current_bytes, 16);
            drop(output);
            assert_eq!(backend.memory_snapshot().current_bytes, 0);
        }

        let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(16)?;
        let cancellation = CancellationToken::default();
        let oom_context = oom_backend.execution_context(
            StreamId::DEFAULT,
            oom_authority.authorize_workspace(16)?,
            &cancellation,
        );
        let input = Tensor::from_bytes(typed_descriptor(vec![2, 2], DType::F16), vec![0; 8])?;
        let source = Tensor::from_bytes(typed_descriptor(vec![1, 1], DType::F16), vec![1, 2])?;
        assert!(matches!(
            oom_backend.replace_rectangular_slice(&input, &source, &[1, 1], &oom_context),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);
        assert_eq!(oom_backend.memory_snapshot().current_bytes, 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let (cancel_backend, cancel_authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let cancel_context = cancel_backend.execution_context(
            StreamId::DEFAULT,
            cancel_authority.authorize_workspace(16)?,
            &cancelled,
        );
        assert!(matches!(
            cancel_backend.replace_rectangular_slice(&input, &source, &[1, 1], &cancel_context),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(cancel_context.scratch.in_use_bytes(), 0);
        assert_eq!(cancel_backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn finite_validation_supports_half_types_and_fails_closed() -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancellation,
        );
        for dtype in [DType::F16, DType::Bf16] {
            let finite = Tensor::from_bytes(
                typed_descriptor(vec![2], dtype),
                if dtype == DType::F16 {
                    vec![0x00, 0x3c, 0x00, 0xc0]
                } else {
                    vec![0x80, 0x3f, 0x00, 0xc0]
                },
            )?;
            backend.validate_finite(&finite, &context)?;

            let non_finite_values = if dtype == DType::F16 {
                [[0x00, 0x7c], [0x00, 0x7e]]
            } else {
                [[0x80, 0x7f], [0xc0, 0x7f]]
            };
            for bytes in non_finite_values {
                let non_finite =
                    Tensor::from_bytes(typed_descriptor(vec![1], dtype), bytes.to_vec())?;
                assert!(matches!(
                    backend.validate_finite(&non_finite, &context),
                    Err(TensorError::InvalidNumeric { .. })
                ));
            }
        }

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancelled,
        );
        let finite = Tensor::from_bytes(descriptor(vec![1]), 1.0_f32.to_ne_bytes().to_vec())?;
        assert!(matches!(
            backend.validate_finite(&finite, &cancelled_context),
            Err(TensorError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn default_patch_graph_adapters_are_typed_unsupported_without_fallback()
    -> Result<(), TensorError> {
        let backend = UnsupportedTensorBackend::new()?;
        let (_cpu, authority) = CpuWorkspaceAuthority::create_backend(1)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor = Tensor::from_bytes(descriptor(vec![1]), 1.0_f32.to_ne_bytes().to_vec())?;
        let cases = [
            backend
                .constant_pad(&tensor, &[0, 0], None, &context)
                .map(|_| ()),
            backend
                .vector_norm(&tensor, 2.0, &[0], false, Some(DType::F32), &context)
                .map(|_| ()),
            backend
                .eye(1, None, DType::F32, Layout::Contiguous, &context)
                .map(|_| ()),
            backend
                .replace_rectangular_slice(&tensor, &tensor, &[0], &context)
                .map(|_| ()),
            backend.validate_finite(&tensor, &context),
            backend
                .upload_f32_payload(&[1], &[1.0], DType::F32, &context)
                .map(|_| ()),
            backend
                .cast_tensor(&tensor, DType::F16, false, false, &context)
                .map(|_| ()),
        ];
        for result in cases {
            assert!(matches!(
                result,
                Err(TensorError::UnsupportedCapability { device, .. })
                    if device == backend.device()
            ));
        }
        assert_eq!(backend.primitive_calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn cpu_capability_matrix_owns_bmm_convolution_and_unsupported_breadth()
    -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancellation,
        );
        for layout in [Layout::Contiguous, Layout::Strided] {
            for dtype in [DType::F32, DType::F16, DType::Bf16] {
                assert!(
                    backend
                        .capabilities()
                        .supports(OperationSupport::linear_algebra_input(
                            LinearAlgebraOperation::BatchMatrixMultiply,
                            dtype,
                            layout,
                        ))
                );
                assert!(
                    backend
                        .capabilities()
                        .supports(OperationSupport::linear_algebra_output(
                            LinearAlgebraOperation::BatchMatrixMultiply,
                            dtype,
                            layout,
                        ))
                );
            }
            for dtype in [DType::F32, DType::F16, DType::Bf16] {
                assert!(
                    backend
                        .capabilities()
                        .supports(OperationSupport::convolution_input(dtype, layout))
                );
                assert!(
                    backend
                        .capabilities()
                        .supports(OperationSupport::convolution_output(dtype, layout))
                );
            }
        }

        let input = zero_tensor(descriptor(vec![1]))?;
        let output = descriptor(vec![1]);
        for operation in [
            IndexSpec::Gather { dimension: 0 },
            IndexSpec::Scatter { dimension: 0 },
            IndexSpec::MaskedSelect,
        ] {
            assert!(matches!(
                backend.indexing(
                    &operation,
                    std::slice::from_ref(&input),
                    output.clone(),
                    &context,
                ),
                Err(TensorError::UnsupportedCapability { .. })
            ));
        }
        assert!(matches!(
            backend.custom_kernel(
                &CustomKernelId::new("fixture.custom")?,
                std::slice::from_ref(&input),
                &[output],
                &context,
            ),
            Err(TensorError::UnsupportedCapability { .. })
        ));
        Ok(())
    }

    #[test]
    fn cpu_low_precision_tanh_executes_exact_values_and_fails_atomically() -> Result<(), TensorError>
    {
        fn encoded_tensor(values: &[f32], dtype: DType) -> Result<Tensor, TensorError> {
            let bytes = values
                .iter()
                .map(|value| {
                    dtype.encode_decoded_scalar(
                        DecodedScalar::Real(f64::from(*value)),
                        "zed.cpu.unary.hyperbolic-tangent.test",
                        DeviceId::CPU,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            Tensor::from_bytes(typed_descriptor(vec![values.len() as u64], dtype), bytes)
        }

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancellation,
        );
        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            let input = encoded_tensor(&[-1.0, 0.0, 0.5], dtype)?;
            let (actual, event) = backend.unary(
                UnaryOperation::HyperbolicTangent,
                &input,
                typed_descriptor(vec![3], dtype),
                &context,
            )?;
            backend.wait_event(event, &context)?;
            for index in 0..3 {
                let decoded_source = read_real_f32(&input, &[index as u64])?;
                let expected_bytes = dtype.encode_decoded_scalar(
                    DecodedScalar::Real(f64::from(decoded_source.tanh())),
                    "zed.cpu.unary.hyperbolic-tangent.test",
                    DeviceId::CPU,
                )?;
                let expected = match dtype.decode_scalar(&expected_bytes)? {
                    DecodedScalar::Real(value) => value as f32,
                    _ => unreachable!("floating tanh dtype decoded as a non-real scalar"),
                };
                assert_eq!(read_real_f32(&actual, &[index as u64])?, expected);
            }
        }
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let input = encoded_tensor(&[0.5], DType::F16)?;
        let before = backend.memory_snapshot();
        assert!(matches!(
            backend.unary(
                UnaryOperation::HyperbolicTangent,
                &input,
                typed_descriptor(vec![1], DType::F32),
                &context,
            ),
            Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: DType::F32,
            })
        ));
        assert_eq!(backend.memory_snapshot(), before);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancelled,
        );
        assert!(matches!(
            backend.unary(
                UnaryOperation::HyperbolicTangent,
                &input,
                typed_descriptor(vec![1], DType::F16),
                &cancelled_context,
            ),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot(), before);

        let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(1)?;
        let oom_context = oom_backend.execution_context(
            StreamId::DEFAULT,
            oom_authority.authorize_workspace(0)?,
            &cancellation,
        );
        assert!(matches!(
            oom_backend.unary(
                UnaryOperation::HyperbolicTangent,
                &input,
                typed_descriptor(vec![1], DType::F16),
                &oom_context,
            ),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(oom_backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn cpu_low_precision_bmm_executes_exact_values_and_fails_atomically() -> Result<(), TensorError>
    {
        fn encoded_tensor(
            shape: Vec<u64>,
            values: &[f32],
            dtype: DType,
        ) -> Result<Tensor, TensorError> {
            let bytes = values
                .iter()
                .map(|value| {
                    dtype.encode_decoded_scalar(
                        DecodedScalar::Real(f64::from(*value)),
                        "zed.cpu.linear-algebra.bmm.test",
                        DeviceId::CPU,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            Tensor::from_bytes(typed_descriptor(shape, dtype), bytes)
        }

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancellation,
        );
        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            let left = encoded_tensor(vec![1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?;
            let right = encoded_tensor(vec![1, 3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype)?;
            let output = typed_descriptor(vec![1, 2, 2], dtype);
            let (actual, event) = backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[left, right],
                output,
                &context,
            )?;
            backend.wait_event(event, &context)?;
            let actual = tensor_to_f32_workspace(&backend, &actual, &context)?;
            assert_eq!(&*actual, &[58.0, 64.0, 139.0, 154.0]);

            let rounded_left = encoded_tensor(vec![1, 1, 1], &[0.1], dtype)?;
            let rounded_right = encoded_tensor(vec![1, 1, 1], &[0.2], dtype)?;
            let decoded_product = read_real_f32(&rounded_left, &[0, 0, 0])?
                * read_real_f32(&rounded_right, &[0, 0, 0])?;
            let expected_bytes = dtype.encode_decoded_scalar(
                DecodedScalar::Real(f64::from(decoded_product)),
                "zed.cpu.linear-algebra.bmm.test",
                DeviceId::CPU,
            )?;
            let expected = match dtype.decode_scalar(&expected_bytes)? {
                DecodedScalar::Real(value) => value as f32,
                _ => unreachable!("floating BMM dtype decoded as a non-real scalar"),
            };
            let (rounded, event) = backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[rounded_left, rounded_right],
                typed_descriptor(vec![1, 1, 1], dtype),
                &context,
            )?;
            backend.wait_event(event, &context)?;
            assert_eq!(read_real_f32(&rounded, &[0, 0, 0])?, expected);
        }
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let f16_left = encoded_tensor(vec![1, 1, 1], &[2.0], DType::F16)?;
        let bf16_right = encoded_tensor(vec![1, 1, 1], &[3.0], DType::Bf16)?;
        let before = backend.memory_snapshot();
        assert!(matches!(
            backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[f16_left.clone(), bf16_right],
                typed_descriptor(vec![1, 1, 1], DType::F16),
                &context,
            ),
            Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: DType::Bf16,
            })
        ));
        assert_eq!(backend.memory_snapshot(), before);
        assert!(matches!(
            backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[f16_left.clone(), f16_left.clone()],
                typed_descriptor(vec![1, 1, 1], DType::F32),
                &context,
            ),
            Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: DType::F32,
            })
        ));
        assert_eq!(backend.memory_snapshot(), before);

        let malformed_right = encoded_tensor(vec![1, 2, 1], &[3.0, 4.0], DType::F16)?;
        assert!(matches!(
            backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[f16_left.clone(), malformed_right],
                typed_descriptor(vec![1, 1, 1], DType::F16),
                &context,
            ),
            Err(TensorError::Faulted { reason })
                if reason == "batch matrix multiplication dimensions are incompatible"
        ));
        assert_eq!(backend.memory_snapshot(), before);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancelled,
        );
        assert!(matches!(
            backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[f16_left.clone(), f16_left],
                typed_descriptor(vec![1, 1, 1], DType::F16),
                &cancelled_context,
            ),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot(), before);

        let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(1)?;
        let oom_context = oom_backend.execution_context(
            StreamId::DEFAULT,
            oom_authority.authorize_workspace(0)?,
            &cancellation,
        );
        let left = encoded_tensor(vec![1, 1, 1], &[2.0], DType::Bf16)?;
        assert!(matches!(
            oom_backend.linear_algebra(
                LinearAlgebraOperation::BatchMatrixMultiply,
                &[left.clone(), left],
                typed_descriptor(vec![1, 1, 1], DType::Bf16),
                &oom_context,
            ),
            Err(TensorError::AllocationFailed { .. })
        ));
        assert_eq!(oom_backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn convolution_adapter_delegates_grouped_strided_dilated_mixed_precision_execution()
    -> Result<(), TensorError> {
        fn cast_for_test(
            backend: &CpuBackend,
            tensor: &Tensor,
            dtype: DType,
            context: &ExecutionContext<'_>,
        ) -> Result<Tensor, TensorError> {
            if dtype == DType::F32 {
                return Ok(tensor.clone());
            }
            let (tensor, event) = backend.cast_tensor(tensor, dtype, false, true, context)?;
            backend.wait_event(event, context)?;
            Ok(tensor)
        }

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let input_source = (0..50)
            .map(|index| (index as f32 - 17.0) / 8.0)
            .collect::<Vec<_>>();
        let weight_source = [0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.375, -0.125];
        let bias_source = [0.25, -0.5];
        let (input_source, input_event) =
            backend.upload_f32(descriptor(vec![1, 2, 5, 5]), &input_source, &context)?;
        backend.wait_event(input_event, &context)?;
        let (weight_source, weight_event) =
            backend.upload_f32(descriptor(vec![2, 1, 2, 2]), &weight_source, &context)?;
        backend.wait_event(weight_event, &context)?;
        let (bias_source, bias_event) =
            backend.upload_f32(descriptor(vec![2]), &bias_source, &context)?;
        backend.wait_event(bias_event, &context)?;
        let operation = ConvolutionSpec {
            stride: vec![2, 2],
            padding: vec![1, 1],
            dilation: vec![2, 2],
            groups: 2,
            transposed: false,
            output_padding: vec![0, 0],
        };
        let geometry =
            ConvolutionGeometry::new(2, vec![2, 2], vec![1, 1], vec![2, 2], 2, false, vec![0, 0])
                .map_err(map_operator_indirection_error)?;

        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            let input = cast_for_test(&backend, &input_source, dtype, &context)?;
            let weight = cast_for_test(&backend, &weight_source, dtype, &context)?;
            let bias = cast_for_test(&backend, &bias_source, dtype, &context)?;
            let rounded_input = tensor_to_f32_workspace(&backend, &input, &context)?;
            let rounded_weight = tensor_to_f32_workspace(&backend, &weight, &context)?;
            let rounded_bias = tensor_to_f32_workspace(&backend, &bias, &context)?;
            let expected = crate::generated_sim_operator_indirection_01::convolution_with_context_exact_native(
                &rounded_input,
                &[1, 2, 5, 5],
                &rounded_weight,
                &[2, 1, 2, 2],
                Some(&rounded_bias),
                &geometry,
                DeviceId::CPU,
                &context,
            )
            .map_err(map_operator_indirection_error)?;
            let output_descriptor = TensorDescriptor::contiguous(
                vec![1, 2, 3, 3],
                dtype,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?;
            let (actual, event) = backend.convolution(
                &operation,
                &[input, weight, bias],
                output_descriptor,
                &context,
            )?;
            backend.wait_event(event, &context)?;
            let actual_values = tensor_to_f32_workspace(&backend, &actual, &context)?;
            let expected_values = expected
                .values
                .iter()
                .map(|value| {
                    let encoded = dtype.encode_decoded_scalar(
                        DecodedScalar::Real(f64::from(*value)),
                        "zed.cpu.convolution",
                        DeviceId::CPU,
                    )?;
                    match dtype.decode_scalar(&encoded)? {
                        DecodedScalar::Real(value) => Ok(value as f32),
                        _ => Err(TensorError::DTypeMismatch {
                            expected: DType::F32,
                            actual: dtype,
                        }),
                    }
                })
                .collect::<Result<Vec<_>, TensorError>>()?;
            assert_eq!(&*actual_values, expected_values.as_slice());
        }
        Ok(())
    }

    #[test]
    fn convolution_workspace_failure_and_cancellation_publish_nothing() -> Result<(), TensorError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let setup_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let (input, input_event) =
            backend.upload_f32(descriptor(vec![1, 1, 3, 3]), &[1.0; 9], &setup_context)?;
        backend.wait_event(input_event, &setup_context)?;
        let (weight, weight_event) =
            backend.upload_f32(descriptor(vec![1, 1, 2, 2]), &[1.0; 4], &setup_context)?;
        backend.wait_event(weight_event, &setup_context)?;
        let operation = ConvolutionSpec {
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
            transposed: false,
            output_padding: vec![0, 0],
        };
        let output = descriptor(vec![1, 1, 2, 2]);
        let before = backend.memory_snapshot().current_bytes;
        let underauthorized_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(67)?,
            &cancellation,
        );
        assert!(matches!(
            backend.convolution(
                &operation,
                &[input.clone(), weight.clone()],
                output.clone(),
                &underauthorized_context,
            ),
            Err(TensorError::WorkspaceAuthorizationExceeded { .. })
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(68)?,
            &cancelled,
        );
        assert!(matches!(
            backend.convolution(
                &operation,
                &[input.clone(), weight.clone()],
                output,
                &cancelled_context,
            ),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        assert_eq!(input.descriptor().shape(), &[1, 1, 3, 3]);
        assert_eq!(weight.descriptor().shape(), &[1, 1, 2, 2]);
        Ok(())
    }

    #[test]
    fn memory_foundation_preserves_capacity_and_failure_atomicity() -> Result<(), TensorError> {
        let token = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &token,
        };
        let byte_descriptor =
            TensorDescriptor::contiguous(vec![17], DType::U8, DeviceId::CPU, StreamId::DEFAULT)?;
        let (tensor, _) = backend.allocate(byte_descriptor, &context)?;
        let aligned_capacity_accounting = backend.memory_snapshot().current_bytes == 32;
        let second_descriptor =
            TensorDescriptor::contiguous(vec![1], DType::U8, DeviceId::CPU, StreamId::DEFAULT)?;
        let oom_is_atomic = matches!(
            backend.allocate(second_descriptor, &context),
            Err(TensorError::AllocationFailed { requested: 16, .. })
        ) && backend.memory_snapshot().current_bytes == 32;
        drop(tensor);
        let aligned_storage_released = backend.memory_snapshot().current_bytes == 0;

        let (cow_backend, cow_authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let cow_context = cow_backend.execution_context(
            StreamId::DEFAULT,
            cow_authority.authorize_workspace(0)?,
            &token,
        );
        let (mut writable, _) = cow_backend.allocate(descriptor(vec![1]), &cow_context)?;
        let shared = writable.clone();
        writable.write()?.bytes_mut()?[0] = 1;
        let copy_on_write_is_charged = cow_backend.memory_snapshot().current_bytes == 32
            && writable.storage_id() != shared.storage_id();
        drop(shared);
        drop(writable);
        let copy_on_write_is_released = cow_backend.memory_snapshot().current_bytes == 0;

        let (upload_backend, upload_authority) = CpuWorkspaceAuthority::create_backend(16)?;
        let upload_context = upload_backend.execution_context(
            StreamId::DEFAULT,
            upload_authority.authorize_workspace(0)?,
            &token,
        );
        let (uploaded, _) =
            upload_backend.upload_f32(descriptor(vec![1]), &[1.0], &upload_context)?;
        let upload_was_charged = upload_backend.memory_snapshot().current_bytes == 16;
        drop(uploaded);
        let canonical_upload_is_charged_and_released =
            upload_was_charged && upload_backend.memory_snapshot().current_bytes == 0;

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let (cancellation_backend, cancellation_authority) =
            CpuWorkspaceAuthority::create_backend(16)?;
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: cancellation_authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        let cancellation_releases_uncommitted_storage =
            matches!(
                cancellation_backend.allocate(descriptor(vec![1]), &cancelled_context),
                Err(TensorError::Cancelled)
            ) && cancellation_backend.memory_snapshot().current_bytes == 0;

        let input = Tensor::from_bytes(descriptor(vec![1, 1, 64, 64]), vec![0; 64 * 64 * 4])?;
        let output = descriptor(vec![1, 1, 1, 1]);
        let geometry = ResizeGeometry::new(
            input.descriptor(),
            &output,
            ResizeSpec {
                width: 1,
                height: 1,
                mode: ResizeMode::Area,
                crop: ResizeCrop::Disabled,
                antialias: false,
                align_corners: false,
            },
        )?;
        let inner_resize_sampling_is_cancellable = matches!(
            geometry.area(&input, 0, 0, 0, 0, &cancelled),
            Err(TensorError::Cancelled)
        );

        let writable_tensor = Tensor::from_bytes(descriptor(vec![1]), vec![0; 4])?;
        let read_only =
            writable_tensor.view(writable_tensor.descriptor().clone(), ViewAccess::ReadOnly)?;
        let read_only_views_cannot_escalate = matches!(
            read_only.view(read_only.descriptor().clone(), ViewAccess::Writable),
            Err(TensorError::ReadOnlyView)
        );

        let cross_stream_descriptor =
            TensorDescriptor::contiguous(vec![1], DType::F32, DeviceId::CPU, StreamId::new(7))?;
        let cross_stream_input = Tensor::from_bytes(cross_stream_descriptor, vec![0; 4])?;
        let (cross_stream_backend, cross_stream_authority) =
            CpuWorkspaceAuthority::create_backend(16)?;
        let cross_stream_context = cross_stream_backend.execution_context(
            StreamId::DEFAULT,
            cross_stream_authority.authorize_workspace(0)?,
            &token,
        );
        let cross_stream_inputs_fail_before_allocation =
            matches!(
                cross_stream_backend.copy(
                    &cross_stream_input,
                    descriptor(vec![1]),
                    &cross_stream_context,
                ),
                Err(TensorError::StreamMismatch { .. })
            ) && cross_stream_backend.memory_snapshot().current_bytes == 0;

        let (event_backend, event_authority) = CpuWorkspaceAuthority::create_backend(1)?;
        let event_authorization = event_authority.authorize_workspace(0)?;
        let mut last_sequence = 0;
        for ordinal in 0..10_000 {
            let event_context = ExecutionContext {
                stream: StreamId::new(ordinal),
                scratch: event_authorization.clone(),
                rng_phase: None,
                cancellation: &token,
            };
            last_sequence = event_backend.record_event(&event_context)?.sequence();
        }
        let event_tracking_is_constant_space =
            last_sequence == 10_000 && event_backend.memory_snapshot().current_bytes == 0;

        assert!(aligned_capacity_accounting);
        assert!(aligned_storage_released);
        assert!(oom_is_atomic);
        assert!(copy_on_write_is_charged);
        assert!(copy_on_write_is_released);
        assert!(canonical_upload_is_charged_and_released);
        assert!(cancellation_releases_uncommitted_storage);
        assert!(inner_resize_sampling_is_cancellable);
        assert!(read_only_views_cannot_escalate);
        assert!(cross_stream_inputs_fail_before_allocation);
        assert!(event_tracking_is_constant_space);
        Ok(())
    }
}
