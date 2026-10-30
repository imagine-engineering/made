use sim_model::{
    NativeModelExecutionRole, auraflow_recipe_fixture_state_schema,
    clip::native_clip_transform_fixture,
    conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningSet, ConditioningValue,
    },
    model_family::{
        NativeFamilyDenoiserContext, NativeFamilyDenoiserInvocation, NativeFamilyModelResource,
        NativeModelRecipeSourceForm,
    },
    vae::{NativeVaeRecipeResource, load_native_vae_recipe},
};
use sim_tensor::{
    BackendCapabilityMatrix, CachedAllocationOwner, CancellationToken, ConvolutionSpec, CpuBackend,
    CpuWorkspaceAuthority, CustomKernelId, DType, DeviceId, EventFence, ExecutionContext,
    IndexSpec, Layout, LinearAlgebraOperation, OperationSupport, ReductionSpec, ResizeSpec, Scalar,
    ScalarSide, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::DeviceKind;
use std::sync::atomic::{AtomicUsize, Ordering};

const MEMORY_LIMIT: u64 = 512 * 1024 * 1024;

struct PartialMetalBackend {
    capabilities: BackendCapabilityMatrix,
    mutations: AtomicUsize,
}

impl PartialMetalBackend {
    fn new() -> Result<Self, TensorError> {
        let device = DeviceId::new(DeviceKind::Metal, 0);
        let supports = vec![
            OperationSupport::allocation(DType::F32, Layout::Contiguous),
            OperationSupport::copy_input(DType::F32, Layout::Contiguous),
            OperationSupport::copy_output(DType::F32, Layout::Contiguous),
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ];
        Ok(Self {
            capabilities: BackendCapabilityMatrix::new(device, supports.clone(), supports)?,
            mutations: AtomicUsize::new(0),
        })
    }

    fn rejected<T>(&self, operation: &str) -> Result<T, TensorError> {
        self.mutations.fetch_add(1, Ordering::AcqRel);
        Err(TensorError::UnsupportedCapability {
            operation: operation.to_owned(),
            device: self.device(),
            reason: "partial Metal fixture must fail during model pre-admission".to_owned(),
        })
    }
}

impl CachedAllocationOwner for PartialMetalBackend {
    fn cache_device(&self) -> DeviceId {
        self.device()
    }

    fn release_cached_allocations(
        &self,
        _cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        self.rejected("release_cached_allocations")
    }
}

impl TensorBackend for PartialMetalBackend {
    fn device(&self) -> DeviceId {
        self.capabilities.device()
    }

    fn capabilities(&self) -> &BackendCapabilityMatrix {
        &self.capabilities
    }

    fn reserve_workspace(
        &self,
        _context: &ExecutionContext<'_>,
        _requested: u64,
    ) -> Result<sim_tensor::BackendWorkspaceLease, TensorError> {
        self.rejected("reserve_workspace")
    }

    fn allocate(
        &self,
        _descriptor: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("allocate")
    }

    fn copy(
        &self,
        _source: &Tensor,
        _destination: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("copy")
    }

    fn record_event(&self, _context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
        self.rejected("record_event")
    }

    fn wait_event(
        &self,
        _event: EventFence,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), TensorError> {
        self.rejected("wait_event")
    }

    fn fill(
        &self,
        _value: Scalar,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("fill")
    }

    fn unary(
        &self,
        _operation: sim_tensor::UnaryOperation,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("unary")
    }

    fn binary(
        &self,
        _operation: sim_tensor::BinaryOperation,
        _left: &Tensor,
        _right: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("binary")
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
        self.rejected("binary_scalar")
    }

    fn reduction(
        &self,
        _operation: &ReductionSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("reduction")
    }

    fn indexing(
        &self,
        _operation: &IndexSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("indexing")
    }

    fn resize(
        &self,
        _operation: ResizeSpec,
        _input: &Tensor,
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("resize")
    }

    fn convolution(
        &self,
        _operation: &ConvolutionSpec,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("convolution")
    }

    fn linear_algebra(
        &self,
        _operation: LinearAlgebraOperation,
        _inputs: &[Tensor],
        _output: TensorDescriptor,
        _context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, EventFence), TensorError> {
        self.rejected("linear_algebra")
    }

    fn custom_kernel(
        &self,
        _kernel: &CustomKernelId,
        _inputs: &[Tensor],
        _outputs: &[TensorDescriptor],
        _context: &ExecutionContext<'_>,
    ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
        self.rejected("custom_kernel")
    }
}

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        cancellation,
    ))
}

#[test]
fn task409_model_clip_and_vae_execute_only_after_exact_backend_admission()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = context(&backend, &authority, &cancellation)?;

    let clip = native_clip_transform_fixture(&backend, &context)?;
    let clip_admission = clip.admit_execution_target(&backend, &context)?;
    assert_eq!(clip_admission.role(), NativeModelExecutionRole::Clip);
    assert_eq!(clip_admission.target_device(), DeviceId::CPU);
    assert_eq!(clip_admission.dtype(), DType::F32);
    assert_eq!(clip_admission.transfer_bytes(), 0);
    assert_eq!(clip_admission.cast_bytes(), 0);
    assert_eq!(clip_admission.resident_bytes(), clip.resident_bytes()?);
    assert_eq!(clip_admission.execution_profile(), "clip.sd1");
    assert_eq!(
        clip_admission.resident_bytes(),
        clip.resident_bytes()?,
        "public CLIP admission must deduplicate the aliased base/patched retained maps"
    );
    assert_eq!(clip_admission.semantic_digest_sha256().len(), 64);
    let clip_output = clip.execute(&backend, "accelerator admission", &context)?;
    assert_eq!(clip_output.len(), 1);
    assert_eq!(
        clip_output[0].conditioning().descriptor().shape(),
        &[1, 77, 2]
    );
    let constrained_workspace = authority.authorize_workspace(1)?;
    let constrained_context = backend.execution_context(
        StreamId::DEFAULT,
        constrained_workspace.clone(),
        &cancellation,
    );
    let memory_before_oom = backend.memory_snapshot().current_bytes;
    let oom = clip.execute(&backend, "accelerator admission", &constrained_context);
    assert!(
        oom.is_err(),
        "CLIP execution unexpectedly published output with one byte of workspace"
    );
    assert_eq!(constrained_workspace.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, memory_before_oom);

    let marker = tensor_from_f32(&backend, &[], &[1.0], &context)?;
    let vae = match load_native_vae_recipe(
        NativeModelRecipeSourceForm::Standalone,
        &"b".repeat(64),
        vec![("pixel_space_vae".to_owned(), marker)],
        None,
        &backend,
        &context,
    )? {
        NativeVaeRecipeResource::Native(vae) => vae,
        NativeVaeRecipeResource::Structured(_) => {
            return Err("pixel-space VAE was structured".into());
        }
    };
    let vae_admission = vae.admit_execution_target(&backend, &context)?;
    assert_eq!(vae_admission.role(), NativeModelExecutionRole::Vae);
    assert_eq!(vae_admission.resident_bytes(), vae.resident_bytes()?);
    assert_eq!(vae_admission.transfer_bytes(), 0);
    assert_eq!(vae_admission.cast_bytes(), 0);
    let pixels = tensor_from_f32(&backend, &[1, 3, 1, 1], &[0.0, 0.5, 1.0], &context)?;
    let latent = vae.encode(&backend, &pixels, &context)?;
    assert_eq!(
        &*tensor_to_f32(&backend, &latent, &context)?,
        &[-1.0, 0.0, 1.0]
    );
    let decoded = vae.decode(&backend, &latent, &context)?;
    assert_eq!(decoded.contiguous_bytes()?, pixels.contiguous_bytes()?);
    let vae_memory_before_oom = backend.memory_snapshot().current_bytes;
    let vae_oom = vae.encode(&backend, &pixels, &constrained_context);
    assert!(
        vae_oom.is_err(),
        "VAE execution unexpectedly published output with one byte of workspace"
    );
    assert_eq!(constrained_workspace.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        vae_memory_before_oom
    );

    let state = auraflow_recipe_fixture_state_schema(NativeModelRecipeSourceForm::Standalone)?
        .into_iter()
        .map(|(name, shape)| {
            let elements = usize::try_from(shape.iter().product::<u64>())?;
            let values = (0..elements)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.000_5)
                .collect::<Vec<_>>();
            Ok((name, tensor_from_f32(&backend, &shape, &values, &context)?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let model = NativeFamilyModelResource::from_reduced_recipe_state(
        NativeModelRecipeSourceForm::Standalone,
        "a".repeat(64),
        state,
        MEMORY_LIMIT,
        &backend,
        &context,
    )?;
    let model_admission = model.admit_execution_target(&backend, &context)?;
    assert_eq!(model_admission.role(), NativeModelExecutionRole::Model);
    assert_eq!(model_admission.resident_bytes(), model.resident_bytes()?);
    assert_eq!(model_admission.transfer_bytes(), 0);
    assert_eq!(model_admission.cast_bytes(), 0);
    assert!(
        model.keyed_state().len() * 2 > model.resident_tensor_allocations()?.len(),
        "public MODEL admission fixture must contain aliased unpatched/patched state"
    );
    assert_ne!(
        model_admission.execution_profile(),
        vae_admission.execution_profile()
    );
    let scaled_latent =
        tensor_from_f32(&backend, &[1, 4, 1, 1], &[0.1, -0.2, 0.3, -0.4], &context)?;
    let conditioning_tensor =
        tensor_from_f32(&backend, &[1, 1, 2_048], &vec![0.001; 2_048], &context)?;
    let conditioning = ConditioningSet::checked(
        model.conditioning_identity().clone(),
        vec![ConditioningEntry::checked(
            "positive",
            ConditioningValue::cross_attention(conditioning_tensor)?,
            ConditioningEntryOptions::default(),
        )?],
        &cancellation,
    )?;
    let resolved = conditioning.resolve(scaled_latent.descriptor(), &backend, &context)?;
    let resolved = resolved.first().ok_or("conditioning did not resolve")?;
    let model_time = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
    let family_context =
        NativeFamilyDenoiserContext::checked(model.conditioning_identity(), &context)?;
    let prediction = model.invoke_denoiser(
        &backend,
        NativeFamilyDenoiserInvocation {
            scaled_latent: &scaled_latent,
            model_time: &model_time,
            conditioning: resolved,
            attention_mask: None,
            reference_latents: &[],
            additional_timestep_condition: None,
        },
        &family_context,
    )?;
    assert_eq!(prediction.descriptor(), scaled_latent.descriptor());
    assert!(
        tensor_to_f32(&backend, &prediction, &context)?
            .iter()
            .all(|value| value.is_finite())
    );
    let model_memory_before_oom = backend.memory_snapshot().current_bytes;
    let model_oom = model.invoke_denoiser(
        &backend,
        NativeFamilyDenoiserInvocation {
            scaled_latent: &scaled_latent,
            model_time: &model_time,
            conditioning: resolved,
            attention_mask: None,
            reference_latents: &[],
            additional_timestep_condition: None,
        },
        &NativeFamilyDenoiserContext::checked(model.conditioning_identity(), &constrained_context)?,
    );
    assert!(
        model_oom.is_err(),
        "MODEL execution unexpectedly published output with one byte of workspace"
    );
    assert_eq!(constrained_workspace.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        model_memory_before_oom
    );

    let cancelled = CancellationToken::default();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancelled,
    );
    let cancelled_family_context =
        NativeFamilyDenoiserContext::checked(model.conditioning_identity(), &cancelled_context)?;
    cancelled.cancel();
    assert!(
        clip.execute(&backend, "cancelled CLIP", &cancelled_context)
            .is_err()
    );
    assert!(vae.encode(&backend, &pixels, &cancelled_context).is_err());
    assert!(
        model
            .invoke_denoiser(
                &backend,
                NativeFamilyDenoiserInvocation {
                    scaled_latent: &scaled_latent,
                    model_time: &model_time,
                    conditioning: resolved,
                    attention_mask: None,
                    reference_latents: &[],
                    additional_timestep_condition: None,
                },
                &cancelled_family_context,
            )
            .is_err()
    );
    let partial = PartialMetalBackend::new()?;
    assert!(
        clip.execute(&partial, "partial Metal", &context).is_err(),
        "public CLIP execution admitted a partial accelerator"
    );
    assert!(
        vae.encode(&partial, &pixels, &context).is_err(),
        "public VAE execution admitted a partial accelerator"
    );
    assert!(
        model
            .invoke_denoiser(
                &partial,
                NativeFamilyDenoiserInvocation {
                    scaled_latent: &scaled_latent,
                    model_time: &model_time,
                    conditioning: resolved,
                    attention_mask: None,
                    reference_latents: &[],
                    additional_timestep_condition: None,
                },
                &family_context,
            )
            .is_err(),
        "public MODEL execution admitted a partial accelerator"
    );
    assert_eq!(partial.mutations.load(Ordering::Acquire), 0);
    Ok(())
}
