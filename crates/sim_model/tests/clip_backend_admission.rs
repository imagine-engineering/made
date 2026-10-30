use sim_model::{NativeExecutionRequirements, NativeOpsError};
use sim_tensor::{
    BackendCapabilityMatrix, CachedAllocationOwner, CancellationToken, ConvolutionSpec,
    CpuWorkspaceAuthority, CustomKernelId, DType, DeviceId, EventFence, ExecutionContext,
    IndexSpec, Layout, LinearAlgebraOperation, OperationSupport, ReductionSpec, ResizeSpec, Scalar,
    ScalarSide, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
};
use sim_types::DeviceKind;
use std::error::Error;

const MEMORY_LIMIT: u64 = 1 << 20;

struct AdmissionProbeBackend {
    device: DeviceId,
    capabilities: BackendCapabilityMatrix,
}

impl AdmissionProbeBackend {
    fn unavailable(&self) -> TensorError {
        TensorError::UnsupportedCapability {
            operation: "test.clip.backend-admission.probe".to_owned(),
            device: self.device,
            reason: "admission must not execute backend operations".to_owned(),
        }
    }
}

impl CachedAllocationOwner for AdmissionProbeBackend {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

impl TensorBackend for AdmissionProbeBackend {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn capabilities(&self) -> &BackendCapabilityMatrix {
        &self.capabilities
    }

    fn allocate(
        &self,
        _descriptor: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn copy(
        &self,
        _source: &Tensor,
        _destination: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn record_event(&self, _context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        Err(self.unavailable())
    }

    fn wait_event(
        &self,
        _event: EventFence,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        Err(self.unavailable())
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn unary(
        &self,
        _operation: UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn binary(
        &self,
        _operation: sim_tensor::BinaryOperation,
        _left: &Tensor,
        _right: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn binary_scalar(
        &self,
        _operation: sim_tensor::BinaryOperation,
        _input: &Tensor,
        _scalar: Scalar,
        _scalar_side: ScalarSide,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        Err(self.unavailable())
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        _context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        Err(self.unavailable())
    }
}

fn requirements(dtype: DType, layout: Layout) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.extend([
        OperationSupport::allocation(dtype, layout),
        OperationSupport::copy_input(dtype, layout),
        OperationSupport::copy_output(dtype, layout),
        OperationSupport::unary_input(UnaryOperation::Exponential, dtype, layout),
        OperationSupport::unary_output(UnaryOperation::Exponential, dtype, layout),
    ]);
    requirements
}

fn make_context<'a>(
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(MEMORY_LIMIT)?,
        rng_phase: None,
        cancellation,
    })
}

#[test]
fn cpu_f32_admission_uses_the_selected_backend_without_effects() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let cancellation = CancellationToken::default();
    let context = make_context(&authority, &cancellation)?;
    let memory_before = backend.memory_snapshot();
    let scratch_before = context.scratch.in_use_bytes();

    requirements(DType::F32, Layout::Contiguous).admit_backend_target(
        &backend,
        DeviceId::CPU,
        DType::F32,
        Layout::Contiguous,
        StreamId::DEFAULT,
        &context,
    )?;

    assert_eq!(backend.memory_snapshot(), memory_before);
    assert_eq!(context.scratch.in_use_bytes(), scratch_before);
    Ok(())
}

#[test]
fn unsupported_clip_targets_fail_before_backend_or_workspace_effects() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let cancellation = CancellationToken::default();
    let context = make_context(&authority, &cancellation)?;
    let memory_before = backend.memory_snapshot();
    let scratch_before = context.scratch.in_use_bytes();

    for dtype in [DType::F16, DType::Bf16, DType::F64] {
        assert!(matches!(
            requirements(dtype, Layout::Contiguous).admit_backend_target(
                &backend,
                DeviceId::CPU,
                dtype,
                Layout::Contiguous,
                StreamId::DEFAULT,
                &context,
            ),
            Err(NativeOpsError::Workspace(
                TensorError::UnsupportedCapability { .. }
            ))
        ));
    }
    for kind in DeviceKind::ALL
        .into_iter()
        .filter(|kind| *kind != DeviceKind::Cpu)
    {
        let device = DeviceId::new(kind, 0);
        assert_eq!(
            requirements(DType::F32, Layout::Contiguous).admit_backend_target(
                &backend,
                device,
                DType::F32,
                Layout::Contiguous,
                StreamId::DEFAULT,
                &context,
            ),
            Err(NativeOpsError::BackendTargetMismatch {
                requested: device,
                backend: DeviceId::CPU,
            })
        );
    }

    assert_eq!(backend.memory_snapshot(), memory_before);
    assert_eq!(context.scratch.in_use_bytes(), scratch_before);
    Ok(())
}

#[test]
fn target_shape_stream_and_stale_capabilities_are_checked_atomically() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let cancellation = CancellationToken::default();
    let context = make_context(&authority, &cancellation)?;
    let requirements = requirements(DType::F32, Layout::Contiguous);

    assert_eq!(
        requirements.admit_backend_target(
            &backend,
            DeviceId::CPU,
            DType::F16,
            Layout::Contiguous,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::ExecutionDTypeMismatch {
            requested: DType::F16,
            requirement: DType::F32,
        })
    );
    assert_eq!(
        requirements.admit_backend_target(
            &backend,
            DeviceId::CPU,
            DType::F32,
            Layout::Strided,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::ExecutionLayoutMismatch {
            requested: Layout::Strided,
            requirement: Layout::Contiguous,
        })
    );
    assert_eq!(
        requirements.admit_backend_target(
            &backend,
            DeviceId::CPU,
            DType::F32,
            Layout::Contiguous,
            StreamId::new(7),
            &context,
        ),
        Err(NativeOpsError::ExecutionStreamMismatch {
            requested: StreamId::new(7),
            context: StreamId::DEFAULT,
        })
    );

    let stale_device = DeviceId::new(DeviceKind::Metal, 0);
    let stale = AdmissionProbeBackend {
        device: DeviceId::CPU,
        capabilities: BackendCapabilityMatrix::new(stale_device, Vec::new(), Vec::new())?,
    };
    assert_eq!(
        requirements.admit_backend_target(
            &stale,
            DeviceId::CPU,
            DType::F32,
            Layout::Contiguous,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::StaleBackendCapabilities {
            backend: DeviceId::CPU,
            capabilities: stale_device,
        })
    );
    Ok(())
}

#[test]
fn missing_operation_or_event_and_cancellation_fail_without_execution() -> Result<(), Box<dyn Error>>
{
    let (_backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let cancellation = CancellationToken::default();
    let context = make_context(&authority, &cancellation)?;
    let requirements = requirements(DType::F32, Layout::Contiguous);
    let support_without_unary = vec![
        OperationSupport::allocation(DType::F32, Layout::Contiguous),
        OperationSupport::copy_input(DType::F32, Layout::Contiguous),
        OperationSupport::copy_output(DType::F32, Layout::Contiguous),
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ];
    let incomplete = AdmissionProbeBackend {
        device: DeviceId::CPU,
        capabilities: BackendCapabilityMatrix::new(
            DeviceId::CPU,
            support_without_unary.clone(),
            support_without_unary,
        )?,
    };
    assert!(matches!(
        requirements.admit_backend_target(
            &incomplete,
            DeviceId::CPU,
            DType::F32,
            Layout::Contiguous,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::Workspace(
            TensorError::UnsupportedCapability { .. }
        ))
    ));

    let mut complete_without_events = requirements.iter().collect::<Vec<_>>();
    complete_without_events.sort_unstable_by_key(|support| format!("{support:?}"));
    complete_without_events.dedup();
    let missing_events = AdmissionProbeBackend {
        device: DeviceId::CPU,
        capabilities: BackendCapabilityMatrix::new(
            DeviceId::CPU,
            complete_without_events.clone(),
            complete_without_events,
        )?,
    };
    assert!(matches!(
        requirements.admit_backend_target(
            &missing_events,
            DeviceId::CPU,
            DType::F32,
            Layout::Contiguous,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::Workspace(
            TensorError::UnsupportedCapability { .. }
        ))
    ));

    let mut complete = requirements.iter().collect::<Vec<_>>();
    complete.extend([
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ]);
    let nondeterministic = AdmissionProbeBackend {
        device: DeviceId::CPU,
        capabilities: BackendCapabilityMatrix::new(
            DeviceId::CPU,
            complete,
            vec![
                OperationSupport::record_event(),
                OperationSupport::wait_event(),
            ],
        )?,
    };
    assert!(matches!(
        requirements.admit_backend_target(
            &nondeterministic,
            DeviceId::CPU,
            DType::F32,
            Layout::Contiguous,
            StreamId::DEFAULT,
            &context,
        ),
        Err(NativeOpsError::Workspace(
            TensorError::UnsupportedCapability { .. }
        ))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = make_context(&authority, &cancelled)?;
    assert_eq!(
        requirements.admit_backend_target(
            &missing_events,
            DeviceId::new(DeviceKind::Metal, 0),
            DType::F16,
            Layout::Strided,
            StreamId::new(9),
            &cancelled_context,
        ),
        Err(NativeOpsError::Cancelled)
    );
    Ok(())
}

#[test]
fn production_clip_call_sites_use_only_the_canonical_admission_adapter()
-> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let clip = std::fs::read_to_string(root.join("src/clip.rs"))?;
    let text = std::fs::read_to_string(root.join("src/clip_text.rs"))?;
    let t5_bidirectional = std::fs::read_to_string(root.join("src/clip_text_encoder_t5.rs"))?;
    let decoder = std::fs::read_to_string(root.join("src/clip_text_encoder_decoder.rs"))?;
    let vision = std::fs::read_to_string(root.join("src/clip_vision.rs"))?;
    let native_ops = std::fs::read_to_string(root.join("src/native_ops.rs"))?;

    assert!(clip.matches(".admit_backend_target(").count() >= 2);
    assert_eq!(text.matches(".admit_backend_target(").count(), 1);
    assert_eq!(
        t5_bidirectional.matches(".admit_backend_target(").count(),
        1
    );
    assert_eq!(decoder.matches(".admit_backend_target(").count(), 1);
    assert!(vision.matches(".admit_backend_target(").count() >= 4);
    assert!(native_ops.matches("pub fn admit_backend_target(").count() >= 2);
    for source in [&clip, &text, &t5_bidirectional, &decoder, &vision] {
        assert!(!source.contains("require_supported(backend.capabilities())"));
        assert!(!source.contains("require_supported(capabilities)"));
    }
    Ok(())
}
