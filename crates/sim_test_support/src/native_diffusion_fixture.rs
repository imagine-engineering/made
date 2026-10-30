use sha2::{Digest, Sha256};
use sim_model::{
    ArtifactIndex, ArtifactIndexError, ArtifactKey, ArtifactRecord, ArtifactRoot, AttentionError,
    ModelStore, ModelStoreError, ParserLimits, PatchGraph,
    clip::NativeTokenizer,
    generated_native_diffusion::{
        NativeDiffusionModelError, Sd1Tokenizer, Sd15DetectorProjection, Sd15TinyModel,
        admit_reduced_fixture, bind_sd15_clip_execution, bind_sd15_empty_patch_execution,
        bind_sd15_vae_execution, load_sd15_clip_execution, load_sd15_tokenizer,
        load_sd15_vae_execution,
    },
};
use sim_runtime::{
    CanonicalClipCacheIdentities, CanonicalNativeDiffusionCacheIdentities,
    CanonicalVaeCacheIdentities, NativeConditioningExecution, NativeDiffusionBundle,
    NativeDiffusionProvider, NativeImageRuntimeError, PreboundControlExecution,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
    generated_activation_normalization_functional_01::FunctionalError,
    generated_native_diffusion::NativeDiffusionTensorError,
    generated_sim_operator_indirection_01::OperatorIndirectionError,
};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

// The checked-in fixture pins the digest produced by its materialized canonical VAE so identity
// discovery never privately materializes a second execution owner.
const SD15_TINY_VAE_EXECUTION_DIGEST: &str =
    "a7cd5a39f5a1cd2dbfc4ade4fe7743683b1d22e621b042f0bcd61ff33fe98d95";
const SD15_TINY_MODEL_DIGEST: &str =
    "063b21b57d085ff0e93fa5d1bb219f8954016cd24a1218063645bbcfdfa6300f";

#[derive(Clone)]
pub struct NativeDiffusionFixture {
    root: PathBuf,
    prebound_conditioning: Option<Arc<FixtureConditioningPlan>>,
    #[cfg(test)]
    model_load_probe: Option<Arc<AtomicUsize>>,
}

#[derive(Clone)]
struct FixtureConditioningPlan {
    patch_graph: Arc<PatchGraph>,
    control: Option<PreboundControlExecution>,
    expected_model_execution_digest: String,
}

impl std::fmt::Debug for NativeDiffusionFixture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeDiffusionFixture")
            .field("root", &self.root)
            .field(
                "has_prebound_conditioning",
                &self.prebound_conditioning.is_some(),
            )
            .finish()
    }
}

impl NativeDiffusionFixture {
    pub fn checked_in() -> Self {
        Self {
            root: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/sd15-tiny-v1"),
            prebound_conditioning: None,
            #[cfg(test)]
            model_load_probe: None,
        }
    }

    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            prebound_conditioning: None,
            #[cfg(test)]
            model_load_probe: None,
        }
    }

    pub fn with_prebound_conditioning(
        mut self,
        patch_graph: Arc<PatchGraph>,
        control: Option<PreboundControlExecution>,
        expected_model_execution_digest: impl Into<String>,
    ) -> Result<Self, NativeImageRuntimeError> {
        patch_graph
            .identity()
            .validate_for_base(SD15_TINY_MODEL_DIGEST)
            .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        let expected_model_execution_digest = expected_model_execution_digest.into();
        if !is_lowercase_sha256(&expected_model_execution_digest) {
            return Err(NativeImageRuntimeError::Registry(
                "prebound fixture model execution identity is invalid".to_owned(),
            ));
        }
        self.prebound_conditioning = Some(Arc::new(FixtureConditioningPlan {
            patch_graph,
            control,
            expected_model_execution_digest,
        }));
        Ok(self)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn source_sampler_reference(&self) -> anyhow::Result<serde_json::Value> {
        self.source_sampler_reference_for("euler")
    }

    pub fn source_void_sampler_reference(&self) -> anyhow::Result<serde_json::Value> {
        self.source_sampler_reference_for("void_ddim")
    }

    fn source_sampler_reference_for(&self, sampler: &str) -> anyhow::Result<serde_json::Value> {
        use sim_model::generated_native_diffusion::empty_sd15_latent;
        use sim_runtime::Sd15GuidanceAdapter;
        use sim_sampler::{
            NoiseRequest,
            generated_native_diffusion::{
                checked_native_diffusion_plan, normal_noise, normal_sigmas, sample_euler,
                scale_initial_noise, scale_model_input, sd15_interpret_prediction,
            },
        };
        use sim_tensor::{DeviceId, generated_native_diffusion::tensor_to_f32};

        const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
        let oracle: serde_json::Value = serde_json::from_str(match sampler {
            "euler" => include_str!("../fixtures/native_diffusion/source-equations.json"),
            "void_ddim" => include_str!("../fixtures/native_diffusion/void-source-equations.json"),
            _ => anyhow::bail!("unsupported source oracle sampler: {sampler}"),
        })?;
        let base_generator = format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../fixtures/native_diffusion/generate_source_equations.py"
            ))
        );
        if sampler == "void_ddim" {
            anyhow::ensure!(
                oracle["base_generator_sha256"].as_str() == Some(&base_generator),
                "VOID base source generator digest changed"
            );
            let generator = format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../fixtures/native_diffusion/generate_void_source_oracle.py"
                ))
            );
            anyhow::ensure!(
                oracle["generator_sha256"].as_str() == Some(&generator),
                "VOID source generator digest changed"
            );
        } else {
            anyhow::ensure!(
                oracle["generator_sha256"].as_str() == Some(&base_generator),
                "source generator digest changed"
            );
        }
        for name in [
            "model.safetensors",
            "tokens.json",
            "initial-noise.safetensors",
            "normal-sigmas.f64le",
        ] {
            let actual = format!("{:x}", Sha256::digest(self.read(name)?));
            anyhow::ensure!(
                oracle["inputs"][name].as_str() == Some(&actual),
                "source oracle input digest mismatch: {name}"
            );
        }
        anyhow::ensure!(
            oracle["tolerance"]["absolute"].as_f64() == Some(2e-6)
                && oracle["tolerance"]["relative"].as_f64() == Some(2e-5),
            "source oracle tolerance policy changed"
        );
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
        let backend = Arc::new(backend);
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(LIMIT)?,
            &cancellation,
        );
        let bundle = self.load_bundle_with_context(backend.clone(), &context)?;
        let (_, positive) = bundle.encode_text("a test", &context)?;
        let (_, negative) = bundle.encode_text("", &context)?;
        compare_source_checkpoint(
            &oracle["conditioning"]["positive"],
            &positive,
            &backend,
            &context,
            false,
        )?;
        compare_source_checkpoint(
            &oracle["conditioning"]["negative"],
            &negative,
            &backend,
            &context,
            false,
        )?;
        let sigmas = normal_sigmas(&backend, &context, 4, 1.0)?;
        let expected_sigmas = self.read("normal-sigmas.f64le")?;
        let sigma_bytes = sigmas
            .iter()
            .flat_map(|sigma| f64::from(*sigma).to_le_bytes())
            .collect::<Vec<_>>();
        anyhow::ensure!(sigma_bytes == expected_sigmas, "CPU sigma fixture changed");
        let stream = NoiseRequest::native_diffusion("53494d00-0000-0000-0000-000000003702", "5")?
            .stream(0x0123_4567_89ab_cdef, DeviceId::CPU)?;
        let noise = normal_noise(&backend, &[1, 4, 4, 4], &stream, &context)?;
        compare_source_checkpoint(&oracle["noise"], &noise.noise, &backend, &context, true)?;
        let empty = empty_sd15_latent(&backend, 1, 32, 32, &context)?;
        let initial = if sampler == "void_ddim" {
            noise.noise.clone()
        } else {
            scale_initial_noise(&backend, &noise.noise, &empty, sigmas[0], &context)?
        };
        let fixture_plan =
            checked_native_diffusion_plan("euler", "normal", 0x0123_4567_89ab_cdef, 4, 7.0, 1.0)?;
        let plan = if sampler == "void_ddim" {
            sim_sampler::SamplingPlan::new(
                sampler,
                "normal",
                fixture_plan.profile().clone(),
                0x0123_4567_89ab_cdef,
                4,
                7.0,
                1.0,
            )?
        } else {
            fixture_plan
        };
        let mut guidance =
            Sd15GuidanceAdapter::checked(bundle.model(), &positive, &negative, &context)?;
        let mut denoiser = |latent: &sim_tensor::Tensor, sigma, _step| {
            let input = scale_model_input(&backend, latent, sigma, &context)
                .map_err(|error| error.to_string())?;
            let prediction = guidance
                .execute(&backend, &input, sigma, &plan, &context)
                .map_err(|error| error.to_string())?;
            sd15_interpret_prediction(&backend, prediction.guided(), latent, sigma, &context)
                .map_err(|error| error.to_string())
        };
        let trace = if sampler == "void_ddim" {
            sim_sampler::generated_void_ddim::sample_void_ddim(
                &backend,
                plan.clone(),
                initial,
                &sigmas,
                &context,
                &mut denoiser,
                |_, _, _| Ok::<(), String>(()),
            )?
        } else {
            sample_euler(&backend, initial, &sigmas, &context, &mut denoiser)?
        };
        let denoised = oracle["denoiser_evaluations"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("source denoiser checkpoints missing"))?;
        let latents = oracle["latents"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("source latent checkpoints missing"))?;
        anyhow::ensure!(
            trace.denoiser_evaluations.len() == denoised.len()
                && trace.latents.len() == latents.len(),
            "source sampler checkpoint count changed"
        );
        for (actual, expected) in trace
            .denoiser_evaluations
            .iter()
            .zip(denoised)
            .chain(trace.latents.iter().zip(latents))
        {
            compare_source_checkpoint(expected, actual, &backend, &context, false)?;
        }
        let decoded = bundle.vae().decode(
            backend.as_ref(),
            trace
                .latents
                .last()
                .ok_or_else(|| anyhow::anyhow!("source sampler final latent missing"))?,
            &context,
        )?;
        compare_source_checkpoint(&oracle["decoded"], &decoded, &backend, &context, false)?;
        let hash = |tensor: &sim_tensor::Tensor| -> anyhow::Result<String> {
            let values = tensor_to_f32(&backend, tensor, &context)?;
            Ok(format!(
                "{:x}",
                Sha256::digest(
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>()
                )
            ))
        };
        Ok(serde_json::json!({
            "noise_sha256": hash(&noise.noise)?,
            "denoiser_sha256": trace.denoiser_evaluations.iter().map(hash).collect::<anyhow::Result<Vec<_>>>()?,
            "latent_sha256": trace.latents.iter().map(hash).collect::<anyhow::Result<Vec<_>>>()?,
        }))
    }

    #[cfg(test)]
    fn with_model_load_probe(mut self, probe: Arc<AtomicUsize>) -> Self {
        self.model_load_probe = Some(probe);
        self
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, NativeDiffusionFixtureError> {
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name == "."
            || name == ".."
        {
            return Err(NativeDiffusionFixtureError::UnsafeName(name.to_owned()));
        }
        Ok(fs::read(self.root.join(name))?)
    }

    pub fn tokenizer(&self) -> Result<Sd1Tokenizer, NativeDiffusionFixtureError> {
        self.tokenizer_with_cancellation(&CancellationToken::default())
    }

    fn tokenizer_with_cancellation(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Sd1Tokenizer, NativeDiffusionFixtureError> {
        check_fixture_cancellation(cancellation)?;
        let vocabulary = String::from_utf8(self.read("vocab.json")?)?;
        check_fixture_cancellation(cancellation)?;
        let merges = String::from_utf8(self.read("merges.txt")?)?;
        check_fixture_cancellation(cancellation)?;
        let tokenizer = load_sd15_tokenizer(&vocabulary, &merges)?;
        check_fixture_cancellation(cancellation)?;
        Ok(tokenizer)
    }

    pub fn detector_projection(
        &self,
    ) -> Result<Sd15DetectorProjection, NativeDiffusionFixtureError> {
        Ok(serde_json::from_slice(
            &self.read("sd15-detector-projection.json")?,
        )?)
    }

    pub fn load_model(
        &self,
        memory_limit_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Sd15TinyModel, NativeDiffusionFixtureError> {
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(memory_limit_bytes)?;
        let backend = Arc::new(backend);
        let scratch = workspace_authority.authorize_workspace(memory_limit_bytes)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, cancellation);
        self.load_model_with_context(backend, &context)
    }

    pub fn load_model_with_context(
        &self,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Sd15TinyModel, NativeDiffusionFixtureError> {
        let (store, index, loaded, _artifact, _projection, admission) =
            self.load_checkpoint(context.cancellation)?;
        Ok(Sd15TinyModel::load_reduced_fixture(
            &store, &index, &loaded, &admission, backend, context,
        )?)
    }

    pub fn load_bundle_with_context(
        &self,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDiffusionBundle, NativeDiffusionFixtureError> {
        let tokenizer = Arc::new(self.tokenizer_with_cancellation(context.cancellation)?);
        let (store, index, loaded, artifact, projection, admission) =
            self.load_checkpoint(context.cancellation)?;
        let model = Arc::new(match &self.prebound_conditioning {
            Some(plan) => Sd15TinyModel::load_reduced_fixture_with_patch_graph(
                &store,
                &index,
                &loaded,
                &admission,
                &plan.patch_graph,
                backend.clone(),
                context,
            )?,
            None => Sd15TinyModel::load_reduced_fixture(
                &store,
                &index,
                &loaded,
                &admission,
                backend.clone(),
                context,
            )?,
        });
        if let Some(plan) = &self.prebound_conditioning
            && model.patch_execution_digest() != plan.expected_model_execution_digest
        {
            return Err(NativeDiffusionFixtureError::Runtime(
                NativeImageRuntimeError::Registry(
                    "prebound fixture model execution identity does not match the loaded model"
                        .to_owned(),
                ),
            ));
        }
        let clip = Arc::new(load_sd15_clip_execution(
            &store,
            &index,
            &loaded,
            &projection,
            tokenizer.identity().clone(),
            backend.clone(),
            context,
        )?);
        let vae = Arc::new(load_sd15_vae_execution(
            &store,
            &index,
            loaded.clone(),
            &artifact,
            &projection,
            &backend,
            context,
        )?);
        let vae_execution_digest = vae.execution_digest();
        require_exact_fixture_vae_execution(&vae_execution_digest, SD15_TINY_VAE_EXECUTION_DIGEST)?;
        match &self.prebound_conditioning {
            Some(plan) => {
                let conditioning = Arc::new(
                    NativeConditioningExecution::checked(
                        loaded.identity(),
                        &model,
                        plan.patch_graph.clone(),
                        plan.control.clone(),
                    )
                    .map_err(NativeDiffusionFixtureError::from)?,
                );
                NativeDiffusionBundle::new_prebound(
                    "sd15-tiny-v1",
                    loaded.identity(),
                    model,
                    tokenizer,
                    clip,
                    vae,
                    conditioning,
                )
            }
            None => NativeDiffusionBundle::new_with_vae(
                "sd15-tiny-v1",
                loaded.identity(),
                model,
                tokenizer,
                clip,
                vae,
            ),
        }
        .map_err(NativeDiffusionFixtureError::from)
    }

    pub fn load_clip_with_context(
        &self,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<sim_model::clip::LoadedSd1Clip, NativeDiffusionFixtureError> {
        let tokenizer = self.tokenizer()?;
        let (store, index, loaded, _artifact, projection, _admission) =
            self.load_checkpoint(context.cancellation)?;
        Ok(load_sd15_clip_execution(
            &store,
            &index,
            &loaded,
            &projection,
            tokenizer.identity().clone(),
            backend,
            context,
        )?)
    }

    fn load_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            ModelStore,
            ArtifactIndex,
            Arc<sim_model::LoadedModel>,
            ArtifactRecord,
            Sd15DetectorProjection,
            sim_model::generated_native_diffusion::ReducedFixtureAdmission,
        ),
        NativeDiffusionFixtureError,
    > {
        let (index, artifact, projection, admission) =
            self.checkpoint_identity_snapshot(cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        #[cfg(test)]
        if let Some(probe) = &self.model_load_probe {
            probe.fetch_add(1, Ordering::SeqCst);
        }
        let loaded = store.load(&index, &artifact.key, cancellation)?;
        Ok((store, index, loaded, artifact, projection, admission))
    }

    fn checkpoint_identity_snapshot(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            ArtifactIndex,
            ArtifactRecord,
            Sd15DetectorProjection,
            sim_model::generated_native_diffusion::ReducedFixtureAdmission,
        ),
        NativeDiffusionFixtureError,
    > {
        check_fixture_cancellation(cancellation)?;
        let detector_bytes = self.read("sd15-detector-projection.json")?;
        check_fixture_cancellation(cancellation)?;
        let projection: Sd15DetectorProjection = serde_json::from_slice(&detector_bytes)?;
        projection.detect()?;
        let digest = format!("{:x}", Sha256::digest(&detector_bytes));
        let admission = admit_reduced_fixture("sd15-tiny-v1", &digest)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "sd15-tiny-fixture",
            "checkpoint",
            &self.root,
            ["safetensors"],
        )?)?;
        index.refresh(cancellation)?;
        let key = ArtifactKey::new("sd15-tiny-fixture", "model.safetensors")?;
        let artifact = index
            .record(&key)
            .cloned()
            .ok_or_else(|| sim_model::ArtifactIndexError::Missing(key.clone()))?;
        if artifact.sha256 != SD15_TINY_MODEL_DIGEST {
            return Err(NativeDiffusionFixtureError::ModelDigestMismatch {
                expected: SD15_TINY_MODEL_DIGEST,
                actual: artifact.sha256,
            });
        }
        check_fixture_cancellation(cancellation)?;
        Ok((index, artifact, projection, admission))
    }
}

impl NativeDiffusionProvider for NativeDiffusionFixture {
    fn cache_identities(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<CanonicalNativeDiffusionCacheIdentities, NativeImageRuntimeError> {
        let tokenizer = self
            .tokenizer_with_cancellation(cancellation)
            .map_err(map_fixture_load_error)?;
        let (_index, artifact, projection, _admission) = self
            .checkpoint_identity_snapshot(cancellation)
            .map_err(map_fixture_load_error)?;
        let model_digest = artifact.sha256.clone();
        let (_, binding) =
            bind_sd15_clip_execution(&projection, &model_digest, tokenizer.identity().clone())
                .map_err(model_identity_runtime_error)?;
        let clip = CanonicalClipCacheIdentities::checked(
            tokenizer.identity().digest(),
            binding.architecture().digest(),
            binding.plan().artifact_identity().as_str(),
            binding.plan().model_identity().as_str(),
            binding.plan().patch_identity().as_str(),
            binding.plan().digest(),
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        let descriptor = bind_sd15_vae_execution(&projection, &artifact)
            .map_err(model_identity_runtime_error)?;
        let identity = descriptor.identity();
        let vae = CanonicalVaeCacheIdentities::checked(
            identity.digest(),
            identity.artifact_sha256(),
            &identity.patch().ordered_digest,
            SD15_TINY_VAE_EXECUTION_DIGEST,
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        let conditioning = if let Some(plan) = &self.prebound_conditioning {
            plan.patch_graph
                .identity()
                .validate_for_base(&model_digest)
                .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
            NativeConditioningExecution::cache_identities_for_sd15(
                &plan.patch_graph.identity(),
                &plan.expected_model_execution_digest,
                plan.control
                    .as_ref()
                    .map(PreboundControlExecution::execution_digest),
            )?
        } else {
            let (patch_graph, model_execution_digest) =
                bind_sd15_empty_patch_execution(&model_digest)
                    .map_err(model_identity_runtime_error)?;
            NativeConditioningExecution::cache_identities_for_sd15(
                &patch_graph.identity(),
                &model_execution_digest,
                None,
            )?
        };
        check_fixture_cancellation(cancellation).map_err(map_fixture_load_error)?;
        CanonicalNativeDiffusionCacheIdentities::checked(
            model_digest,
            tokenizer.identity().digest(),
            clip,
            vae,
            conditioning,
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))
    }

    fn load(
        &self,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDiffusionBundle, NativeImageRuntimeError> {
        self.load_bundle_with_context(backend, context)
            .map_err(map_fixture_load_error)
    }
}

fn compare_source_checkpoint(
    expected: &serde_json::Value,
    actual: &sim_tensor::Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    exact: bool,
) -> anyhow::Result<()> {
    let shape = serde_json::to_value(actual.descriptor().shape())?;
    anyhow::ensure!(
        expected["shape"] == shape,
        "source checkpoint shape mismatch"
    );
    let values = sim_tensor::generated_native_diffusion::tensor_to_f32(backend, actual, context)?;
    let expected = expected["values"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("source checkpoint values missing"))?;
    anyhow::ensure!(
        values.len() == expected.len(),
        "source checkpoint value count mismatch"
    );
    for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
        let expected = expected
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("source checkpoint scalar missing"))?;
        let error = (f64::from(*actual) - expected).abs();
        let matches = if exact {
            actual.to_bits() == (expected as f32).to_bits()
        } else {
            error <= 2e-6 + 2e-5 * expected.abs()
        };
        anyhow::ensure!(
            actual.is_finite() && expected.is_finite() && matches,
            "source checkpoint[{index}]: native {actual}, source {expected}, error {error}"
        );
    }
    Ok(())
}

fn tensor_runtime_error(error: TensorError) -> NativeImageRuntimeError {
    match error {
        TensorError::Cancelled => NativeImageRuntimeError::Cancelled,
        TensorError::AllocationFailed { .. }
        | TensorError::ResourceLimitExceeded { .. }
        | TensorError::WorkspaceAuthorizationExceeded { .. } => {
            NativeImageRuntimeError::ResourceExhausted(error.to_string())
        }
        error => NativeImageRuntimeError::Execution(error.to_string()),
    }
}

fn functional_runtime_error(error: FunctionalError) -> NativeImageRuntimeError {
    match error {
        FunctionalError::Cancelled => NativeImageRuntimeError::Cancelled,
        FunctionalError::AllocationFailed { .. } => {
            NativeImageRuntimeError::ResourceExhausted(error.to_string())
        }
        FunctionalError::Tensor(error) => tensor_runtime_error(error),
        error => NativeImageRuntimeError::Execution(error.to_string()),
    }
}

fn operator_runtime_error(error: OperatorIndirectionError) -> NativeImageRuntimeError {
    match error {
        OperatorIndirectionError::Cancelled => NativeImageRuntimeError::Cancelled,
        OperatorIndirectionError::Tensor(error) => tensor_runtime_error(error),
        error => NativeImageRuntimeError::Execution(error.to_string()),
    }
}

fn native_tensor_runtime_error(error: NativeDiffusionTensorError) -> NativeImageRuntimeError {
    match error {
        NativeDiffusionTensorError::Tensor(error) => tensor_runtime_error(error),
        NativeDiffusionTensorError::Functional(error) => functional_runtime_error(error),
        NativeDiffusionTensorError::Operator(error) => operator_runtime_error(error),
        error => NativeImageRuntimeError::Execution(error.to_string()),
    }
}

fn attention_runtime_error(error: AttentionError) -> NativeImageRuntimeError {
    match error {
        AttentionError::Cancelled => NativeImageRuntimeError::Cancelled,
        AttentionError::Tensor(error) => tensor_runtime_error(error),
        AttentionError::AllocationFailed { .. } | AttentionError::WorkspaceTooSmall { .. } => {
            NativeImageRuntimeError::ResourceExhausted(error.to_string())
        }
        error => NativeImageRuntimeError::Execution(error.to_string()),
    }
}

fn artifact_index_runtime_error(error: ArtifactIndexError) -> NativeImageRuntimeError {
    match error {
        ArtifactIndexError::Cancelled => NativeImageRuntimeError::Cancelled,
        ArtifactIndexError::AllocationFailed(_) => {
            NativeImageRuntimeError::ResourceExhausted(error.to_string())
        }
        error => NativeImageRuntimeError::Asset(error.to_string()),
    }
}

fn model_store_runtime_error(error: ModelStoreError) -> NativeImageRuntimeError {
    match error {
        ModelStoreError::Cancelled => NativeImageRuntimeError::Cancelled,
        ModelStoreError::AllocationFailed { .. } => {
            NativeImageRuntimeError::ResourceExhausted(error.to_string())
        }
        ModelStoreError::Index(error) => artifact_index_runtime_error(error),
        error => NativeImageRuntimeError::Asset(error.to_string()),
    }
}

fn model_runtime_error(error: NativeDiffusionModelError) -> NativeImageRuntimeError {
    match error {
        NativeDiffusionModelError::Cancelled => NativeImageRuntimeError::Cancelled,
        NativeDiffusionModelError::ResourceExhausted(message) => {
            NativeImageRuntimeError::ResourceExhausted(message)
        }
        NativeDiffusionModelError::TensorBackend(error) => tensor_runtime_error(error),
        NativeDiffusionModelError::Tensor(error) => native_tensor_runtime_error(error),
        NativeDiffusionModelError::Attention(error) => attention_runtime_error(error),
        NativeDiffusionModelError::Store(error) => model_store_runtime_error(error),
        error @ (NativeDiffusionModelError::UnsupportedFamily
        | NativeDiffusionModelError::Tokenizer(_)) => {
            NativeImageRuntimeError::Asset(error.to_string())
        }
        error @ (NativeDiffusionModelError::InvalidFixtureAdmission
        | NativeDiffusionModelError::DuplicateWeight
        | NativeDiffusionModelError::MissingWeight(_)
        | NativeDiffusionModelError::WeightKeys { .. }
        | NativeDiffusionModelError::WeightShape { .. }
        | NativeDiffusionModelError::WeightBytes(_)
        | NativeDiffusionModelError::Clip(_)
        | NativeDiffusionModelError::Vae(_)
        | NativeDiffusionModelError::Patch(_)
        | NativeDiffusionModelError::LatentAdapter(_)) => {
            NativeImageRuntimeError::Registry(error.to_string())
        }
        error @ (NativeDiffusionModelError::ExcessControl(_)
        | NativeDiffusionModelError::InputShape { .. }
        | NativeDiffusionModelError::InvalidModelTime
        | NativeDiffusionModelError::InvalidLatentDimensions { .. }
        | NativeDiffusionModelError::Overflow(_)) => {
            NativeImageRuntimeError::Execution(error.to_string())
        }
    }
}

fn model_identity_runtime_error(error: NativeDiffusionModelError) -> NativeImageRuntimeError {
    let message = error.to_string();
    match model_runtime_error(error) {
        NativeImageRuntimeError::Cancelled => NativeImageRuntimeError::Cancelled,
        NativeImageRuntimeError::ResourceExhausted(message) => {
            NativeImageRuntimeError::ResourceExhausted(message)
        }
        _ => NativeImageRuntimeError::Registry(message),
    }
}

fn map_fixture_load_error(error: NativeDiffusionFixtureError) -> NativeImageRuntimeError {
    match error {
        NativeDiffusionFixtureError::Model(error) => model_runtime_error(error),
        NativeDiffusionFixtureError::Index(error) => artifact_index_runtime_error(error),
        NativeDiffusionFixtureError::Store(error) => model_store_runtime_error(error),
        NativeDiffusionFixtureError::Tensor(error) => tensor_runtime_error(error),
        NativeDiffusionFixtureError::Runtime(error) => error,
        NativeDiffusionFixtureError::Io(_)
        | NativeDiffusionFixtureError::Utf8(_)
        | NativeDiffusionFixtureError::Json(_)
        | NativeDiffusionFixtureError::ModelDigestMismatch { .. }
        | NativeDiffusionFixtureError::UnsafeName(_) => {
            NativeImageRuntimeError::Asset(error.to_string())
        }
    }
}

fn check_fixture_cancellation(
    cancellation: &CancellationToken,
) -> Result<(), NativeDiffusionFixtureError> {
    cancellation
        .check()
        .map_err(TensorError::from)
        .map_err(NativeDiffusionFixtureError::from)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_exact_fixture_vae_execution(
    actual: &str,
    expected: &str,
) -> Result<(), NativeImageRuntimeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NativeImageRuntimeError::Registry(format!(
            "checked-in fixture VAE execution identity does not match its pinned commitment: expected {expected}, found {actual}"
        )))
    }
}

#[derive(Debug, Error)]
pub enum NativeDiffusionFixtureError {
    #[error("unsafe native diffusion fixture filename {0:?}")]
    UnsafeName(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Model(#[from] NativeDiffusionModelError),
    #[error(transparent)]
    Index(#[from] sim_model::ArtifactIndexError),
    #[error(transparent)]
    Store(#[from] sim_model::ModelStoreError),
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error("native diffusion fixture model digest mismatch: expected {expected}, found {actual}")]
    ModelDigestMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error(transparent)]
    Runtime(#[from] NativeImageRuntimeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_load_preserves_cancelled_and_capacity_failures_and_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
        let model_load_probe = Arc::new(AtomicUsize::new(0));
        let fixture =
            NativeDiffusionFixture::checked_in().with_model_load_probe(model_load_probe.clone());
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
        let backend = Arc::new(backend);

        let active = CancellationToken::default();
        let snapshot = NativeDiffusionProvider::cache_identities(&fixture, &active)?;
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 0);
        assert_eq!(snapshot.vae().execution(), SD15_TINY_VAE_EXECUTION_DIGEST);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            NativeDiffusionProvider::cache_identities(&fixture, &cancelled),
            Err(NativeImageRuntimeError::Cancelled)
        ));
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 0);
        let cancelled_scratch = authority.authorize_workspace(LIMIT)?;
        let cancelled_context =
            backend.execution_context(StreamId::DEFAULT, cancelled_scratch.clone(), &cancelled);
        assert!(matches!(
            NativeDiffusionProvider::load(&fixture, backend.clone(), &cancelled_context),
            Err(NativeImageRuntimeError::Cancelled)
        ));
        assert_eq!(cancelled_scratch.in_use_bytes(), 0);
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 0);

        let constrained_scratch = authority.authorize_workspace(64)?;
        let constrained_context =
            backend.execution_context(StreamId::DEFAULT, constrained_scratch.clone(), &active);
        assert!(matches!(
            NativeDiffusionProvider::load(&fixture, backend.clone(), &constrained_context),
            Err(NativeImageRuntimeError::ResourceExhausted(_))
        ));
        assert_eq!(constrained_scratch.in_use_bytes(), 0);
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 1);

        let recovery_scratch = authority.authorize_workspace(LIMIT)?;
        let recovery_context =
            backend.execution_context(StreamId::DEFAULT, recovery_scratch.clone(), &active);
        let bundle = NativeDiffusionProvider::load(&fixture, backend.clone(), &recovery_context)?;
        assert_eq!(recovery_scratch.in_use_bytes(), 0);
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 2);
        snapshot.require_exact_match(bundle.cache_identities())?;
        assert_eq!(snapshot.vae().execution(), bundle.vae().execution_digest());
        let actual_vae_execution = bundle.vae().execution_digest();
        let wrong_vae_execution = if actual_vae_execution.starts_with('f') {
            "e".repeat(64)
        } else {
            "f".repeat(64)
        };
        assert!(matches!(
            require_exact_fixture_vae_execution(
                &actual_vae_execution,
                &wrong_vae_execution,
            ),
            Err(NativeImageRuntimeError::Registry(message))
                if message.starts_with(
                    "checked-in fixture VAE execution identity does not match its pinned commitment"
                )
        ));
        assert_eq!(recovery_scratch.in_use_bytes(), 0);
        assert_eq!(model_load_probe.load(Ordering::SeqCst), 2);

        let wrong_model_execution = if bundle.model().patch_execution_digest().starts_with('f') {
            "e".repeat(64)
        } else {
            "f".repeat(64)
        };
        let wrong_commitment_probe = Arc::new(AtomicUsize::new(0));
        let wrong_commitment_fixture = NativeDiffusionFixture::checked_in()
            .with_prebound_conditioning(
                Arc::new(PatchGraph::checked_semantic(
                    SD15_TINY_MODEL_DIGEST,
                    Vec::new(),
                )?),
                None,
                wrong_model_execution,
            )?
            .with_model_load_probe(wrong_commitment_probe.clone());
        let wrong_commitment_scratch = authority.authorize_workspace(LIMIT)?;
        let wrong_commitment_context =
            backend.execution_context(StreamId::DEFAULT, wrong_commitment_scratch.clone(), &active);
        assert!(matches!(
            NativeDiffusionProvider::load(
                &wrong_commitment_fixture,
                backend.clone(),
                &wrong_commitment_context,
            ),
            Err(NativeImageRuntimeError::Registry(message))
                if message == "prebound fixture model execution identity does not match the loaded model"
        ));
        assert_eq!(wrong_commitment_scratch.in_use_bytes(), 0);
        assert_eq!(wrong_commitment_probe.load(Ordering::SeqCst), 1);

        let stale_root = tempfile::tempdir()?;
        for filename in ["vocab.json", "merges.txt", "sd15-detector-projection.json"] {
            fs::copy(
                fixture.root().join(filename),
                stale_root.path().join(filename),
            )?;
        }
        fs::write(stale_root.path().join("model.safetensors"), b"not a model")?;
        let stale_probe = Arc::new(AtomicUsize::new(0));
        let stale_fixture = NativeDiffusionFixture::at(stale_root.path())
            .with_model_load_probe(stale_probe.clone());
        assert!(matches!(
            NativeDiffusionProvider::cache_identities(&stale_fixture, &active),
            Err(NativeImageRuntimeError::Asset(_))
        ));
        assert_eq!(stale_probe.load(Ordering::SeqCst), 0);
        Ok(())
    }
}
