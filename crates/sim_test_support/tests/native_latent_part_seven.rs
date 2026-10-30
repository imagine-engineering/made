use serde_json::{Value, json};
use sim_nodes::{
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativePrimitive,
    NativeResolvedPayload, NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, ImageTensor, NativeLatentBundle,
    NativeLatentMetadata, NativeTensorPayload, NativeTensorRole, StreamId, Tensor, TensorBackend,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use uuid::Uuid;
type TestResult = Result<(), Box<dyn Error>>;
fn fixture() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0714/source-contract.json"
    ))?)
}
fn context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("latent-part-seven".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x63303), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x63301)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}
fn canonical_source_image_vae(
    context: &NativeNodeContext,
    checkpoint: &std::path::Path,
    profile: &sim_model::VaeKernelProfile,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    use sim_model::vae_image::load_image_vae_from_model_store_with_context;
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, NativeModelPayload, ParserLimits,
        PatchGraph, VaeArchitectureRegistry, VaeBoundary, VaeDescriptor,
    };
    let directory = checkpoint.parent().ok_or("Checkpoint parent absent")?;
    let filename = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("UTF8 checkpoint filename required")?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "task633-source-vae",
        "checkpoint",
        directory,
        ["safetensors"],
    )?)?;
    index.refresh(&context.cancellation)?;
    let key = ArtifactKey::new("task633-source-vae", filename)?;
    let artifact = index
        .record(&key)
        .cloned()
        .ok_or("Actual checkpoint record absent")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &artifact.key, &context.cancellation)?;
    let registry = VaeArchitectureRegistry::checked()?;
    let (families, latents) = VaeArchitectureRegistry::canonical_targets()?;
    let probe = store.family_probe(&loaded, &context.cancellation)?;
    let selection = registry.select(&probe, &context.cancellation)?;
    let target =
        registry.intended_target(&selection, &families, &latents, &context.cancellation)?;
    let definition = latents
        .get(target.latent_format())
        .ok_or("Latent format absent")?;
    let patch = PatchGraph::checked_semantic(&artifact.sha256, Vec::new())?.identity();
    let descriptor = VaeDescriptor::checked_selection(
        &artifact,
        &selection,
        &target,
        &families,
        &latents,
        patch,
        VaeBoundary::image(3)?,
        [0.0, 1.0],
        &context.cancellation,
    )?;
    assert_eq!(descriptor.identity().profile(), profile);
    if profile == &sim_model::VaeKernelProfile::StableCascadeStageCEncoderV1 {
        assert_eq!(descriptor.source_image_encode_downscale_ratio()?, 32);
    }
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let vae = Arc::new(load_image_vae_from_model_store_with_context(
        compute.backend(),
        &store,
        &index,
        loaded,
        descriptor,
        definition,
        &execution,
    )?);
    let model = Arc::new(NativeModelPayload::native_vae(vae)?);
    let diffusion = Arc::new(sim_sampler::NativeDiffusionPayload::vae(model)?);
    Ok(context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(
            sim_nodes::NativeStoredModelPayload::native_diffusion(diffusion)?,
        )),
        &context.cancellation,
    )?)
}

fn source_checkpoint(
    source: &Value,
    directory: &std::path::Path,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::from_value::<Vec<u8>>(source["artifact"]["bytes"].clone())?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        source["artifact"]["sha256"]
    );
    let checkpoint = directory.join("original-pixel-space.safetensors");
    std::fs::write(&checkpoint, bytes)?;
    Ok(checkpoint)
}
fn publish_input(
    context: &NativeNodeContext,
    case: &Value,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let shape = serde_json::from_value::<Vec<u64>>(case["input"]["shape"].clone())?;
    let values = serde_json::from_value::<Vec<f32>>(case["input"]["values"].clone())?;
    let tensor = tensor_from_f32(compute.backend(), &shape, &values, &execution)?;
    let payload = if case["node"] == "VAEDecodeTiled" {
        NativeStoredPayload::Latent(Arc::new(NativeLatentBundle::single(
            tensor,
            None,
            None,
            NativeLatentMetadata::default(),
            &execution,
        )?))
    } else {
        NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            ImageTensor::from_tensor(tensor)?,
        )?))
    };
    Ok(context
        .handle_store()
        .publish(payload, &context.cancellation)?)
}
fn inputs(
    context: &NativeNodeContext,
    case: &Value,
    vae: &NativeOpaqueHandle,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let name = if case["node"] == "VAEDecodeTiled" {
        "samples"
    } else {
        "pixels"
    };
    let mut values = BTreeMap::from([
        (
            name.to_owned(),
            NativeValue::Handle {
                value: publish_input(context, case)?,
            },
        ),
        ("vae".to_owned(), NativeValue::Handle { value: vae.clone() }),
    ]);
    if case["node"] == "VAEEncodeForInpaint" {
        let compute = context.compute_session()?;
        let execution = compute.execution_context(context)?;
        let shape = serde_json::from_value::<Vec<u64>>(case["mask"]["shape"].clone())?;
        let mask = serde_json::from_value::<Vec<f32>>(case["mask"]["values"].clone())?;
        let mask = tensor_from_f32(compute.backend(), &shape, &mask, &execution)?;
        let mask = context.handle_store().publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_tensor(
                NativeTensorRole::Mask,
                mask,
            )?)),
            &context.cancellation,
        )?;
        values.insert("mask".to_owned(), NativeValue::Handle { value: mask });
        values.insert(
            "grow_mask_by".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(
                    case["options"]["grow_mask_by"]
                        .as_i64()
                        .ok_or("Source grow option")?,
                ),
            },
        );
    } else if case["node"] != "VAEEncode" {
        for (name, default) in [
            ("tile_size", 512),
            ("overlap", 64),
            ("temporal_size", 64),
            ("temporal_overlap", 8),
        ] {
            values.insert(
                name.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        case["options"]
                            .get(name)
                            .and_then(Value::as_i64)
                            .unwrap_or(default),
                    ),
                },
            );
        }
    }
    Ok(values)
}
fn numeric(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Value, Box<dyn Error>> {
    let payload =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let tensor = match payload.as_ref() {
        NativeStoredPayload::Tensor(value) => value.tensor(),
        NativeStoredPayload::Latent(value) => {
            assert!(value.noise_mask().is_none());
            assert!(value.batch_indices().is_none());
            assert_eq!(value.metadata(), &NativeLatentMetadata::default());
            let projection: sim_tensor::NativeLatentBundleProjection =
                serde_json::from_slice(&serde_json::to_vec(value.projection())?)?;
            assert_eq!(&projection, value.projection());
            value.samples().tensor().ok_or("Tensor LATENT required")?
        }
        _ => return Err("Canonical numerical payload required".into()),
    };
    let compute = context.compute_session()?;
    let values = tensor_to_f32(
        compute.backend(),
        tensor,
        &compute.execution_context(context)?,
    )?;
    Ok(json!({"shape":tensor.descriptor().shape(), "dtype":"torch.float32", "values":&*values}))
}
fn exact_bits(actual: &Value, expected: &Value) -> TestResult {
    assert_eq!(actual["shape"], expected["shape"]);
    assert_eq!(actual["dtype"], expected["dtype"]);
    let actual = serde_json::from_value::<Vec<f32>>(actual["values"].clone())?;
    let expected = serde_json::from_value::<Vec<f32>>(expected["values"].clone())?;
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}
#[test]
fn task633_exact_registry_schema_and_original_pixel_vae_cases() -> TestResult {
    let source = fixture()?;
    let directory = tempfile::tempdir()?;
    let checkpoint = source_checkpoint(&source, directory.path())?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x633101))),
        CancellationToken::default(),
        16 * 1024 * 1024,
    )?;
    let vae = canonical_source_image_vae(
        &context,
        &checkpoint,
        &sim_model::VaeKernelProfile::PixelSpaceV1,
    )?;
    let bindings = sim_nodes::generated_model_latent_07::NATIVE_NODE_BINDINGS()?;
    assert_eq!(
        bindings
            .iter()
            .map(|binding| binding.feature_id())
            .collect::<Vec<_>>(),
        [
            "COMFY-NODE-0714",
            "COMFY-NODE-0716",
            "COMFY-NODE-0718",
            "COMFY-NODE-0719",
            "COMFY-NODE-0745"
        ]
    );
    for binding in &bindings {
        binding.descriptor().validate()?;
        assert!(binding.descriptor().source_schema.is_some());
        assert_eq!(binding.descriptor().outputs.len(), 1);
        assert_eq!(
            binding.descriptor().effect,
            sim_nodes::NativeEffectClass::Pure
        );
        assert_eq!(
            binding.descriptor().cache,
            sim_nodes::NativeCachePolicy::InputIdentity
        );
        if binding.descriptor().class_type == "VAEEncode" {
            assert_eq!(
                binding.presentation().search_aliases,
                ["encode", "encode image", "image to latent"]
            );
        }
    }
    let registry = generated_native_node_registry_projection(None)?;
    let mut successes = 0;
    let mut failures = 0;
    let mut gaps = 0;
    for case in source["cases"].as_array().ok_or("Cases absent")? {
        let name = case["node"].as_str().ok_or("Node name")?;
        let node = registry
            .node(name)
            .ok_or("Registry implementation absent")?;
        let before = generation.len();
        let values = match inputs(&context, case, &vae) {
            Ok(values) => values,
            Err(error) => {
                let shape = serde_json::from_value::<Vec<u64>>(case["input"]["shape"].clone())?;
                assert!(
                    shape.contains(&0) || shape.last() == Some(&1),
                    "Unexpected input rejection: {error}"
                );
                assert_eq!(generation.len(), before);
                gaps += 1;
                continue;
            }
        };
        let count = generation.len();
        let input_name = if name == "VAEDecodeTiled" {
            "samples"
        } else {
            "pixels"
        };
        let NativeValue::Handle {
            value: input_handle,
        } = values.get(input_name).ok_or("Input absent")?
        else {
            return Err("Input handle required".into());
        };
        let original = numeric(&context, input_handle)?;
        let dependencies = node.cache_dependencies(&context, &values)?;
        assert_eq!(dependencies.artifact_digests.len(), 2);
        let mapped: BTreeMap<_, _> = values
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    NativeValue::List {
                        values: vec![value.clone()],
                    },
                )
            })
            .collect();
        assert!(node.demanded_lazy_inputs(&context, &mapped)?.is_empty());
        let outcome = smol::block_on(node.execute(context.clone(), values.clone()));
        if case.get("error_type").is_some() {
            assert!(outcome.is_err(), "Source error accepted: {}", case["label"]);
            assert_eq!(generation.len(), count);
            failures += 1;
            continue;
        }
        let lower_source_channels = name != "VAEDecodeTiled"
            && case["input"]["shape"]
                .as_array()
                .and_then(|shape| shape.last())
                .and_then(Value::as_u64)
                == Some(1);
        let source_nonfinite = case["output"]["values"]
            .as_array()
            .is_some_and(|values| values.iter().any(Value::is_object));
        if lower_source_channels
            || source_nonfinite
            || case["output"].is_null()
            || case["output"]["shape"]
                .as_array()
                .is_some_and(|shape| shape.iter().any(|extent| extent == 0))
        {
            assert!(outcome.is_err());
            assert_eq!(generation.len(), count);
            gaps += 1;
            continue;
        }
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome.map_err(|error| format!("{name} {}: {error:?}", case["label"]))?
        else {
            return Err("Values outcome required".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        assert_eq!(outputs.len(), 1);
        let Some(NativeValue::Handle { value: output }) = outputs.first() else {
            return Err("Output handle absent".into());
        };
        assert_eq!(generation.len(), count + 1);
        exact_bits(&numeric(&context, output)?, &case["output"])?;
        exact_bits(&numeric(&context, input_handle)?, &original)?;
        assert_eq!(case["input_unchanged"], true);
        assert_eq!(node.cache_dependencies(&context, &values)?, dependencies);
        successes += 1;
    }
    assert!(successes > 25);
    assert!(failures > 0);
    assert!(gaps > 0);
    Ok(())
}
fn tensor_numeric(context: &NativeNodeContext, tensor: &Tensor) -> Result<Value, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let values = tensor_to_f32(
        compute.backend(),
        tensor,
        &compute.execution_context(context)?,
    )?;
    Ok(json!({"shape":tensor.descriptor().shape(), "dtype":"torch.float32", "values":&*values}))
}
#[test]
fn task633_inpaint_real_source_resource_and_independent_mask_batch() -> TestResult {
    let source = fixture()?;
    let directory = tempfile::tempdir()?;
    let checkpoint = source_checkpoint(&source, directory.path())?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x633301))),
        CancellationToken::default(),
        16 * 1024 * 1024,
    )?;
    let vae = canonical_source_image_vae(
        &context,
        &checkpoint,
        &sim_model::VaeKernelProfile::PixelSpaceV1,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("VAEEncodeForInpaint")
        .ok_or("Inpaint registry node absent")?;
    let mut successes = 0;
    let mut errors = 0;
    let mut representation_gaps = 0;
    for case in source["inpaint_cases"]
        .as_array()
        .ok_or("Inpaint source cases absent")?
    {
        let values = match inputs(&context, case, &vae) {
            Ok(values) => values,
            Err(error) => {
                assert!(
                    case["mask"]["shape"]
                        .as_array()
                        .is_some_and(|shape| shape.len() == 2)
                        || case["input"]["shape"]
                            .as_array()
                            .and_then(|shape| shape.last())
                            == Some(&json!(1)),
                    "Unexpected canonical representation rejection: {error}"
                );
                representation_gaps += 1;
                continue;
            }
        };
        let count = generation.len();
        let dependencies = node.cache_dependencies(&context, &values)?;
        assert_eq!(dependencies.artifact_digests.len(), 3);
        let mapped = values
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    NativeValue::List {
                        values: vec![value.clone()],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert!(node.demanded_lazy_inputs(&context, &mapped)?.is_empty());
        assert_eq!(node.cache_dependencies(&context, &mapped)?, dependencies);
        let outcome = smol::block_on(node.execute(context.clone(), values.clone()));
        if case.get("error_type").is_some() {
            assert!(
                outcome.is_err(),
                "Source failure was accepted: {}",
                case["label"]
            );
            assert_eq!(generation.len(), count);
            errors += 1;
            continue;
        }
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome.map_err(|error| format!("Inpaint {}: {error:?}", case["label"]))?
        else {
            return Err("Inpaint values outcome required".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let [NativeValue::Handle { value: output }] = outputs.as_slice() else {
            return Err("One inpaint LATENT output required".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(output, output.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
            return Err("Inpaint LATENT required".into());
        };
        exact_bits(
            &tensor_numeric(&context, bundle.samples().tensor().ok_or("Tensor samples")?)?,
            &case["output"],
        )?;
        let sim_tensor::NativeLatentNoiseMask::Tensor(mask) =
            bundle.noise_mask().ok_or("Source noise mask")?
        else {
            return Err("Source tensor mask required".into());
        };
        exact_bits(&tensor_numeric(&context, mask)?, &case["noise_mask"])?;
        assert!(bundle.batch_indices().is_none());
        assert_eq!(bundle.metadata(), &NativeLatentMetadata::default());
        let restored: sim_tensor::NativeLatentBundleProjection =
            serde_json::from_slice(&serde_json::to_vec(bundle.projection())?)?;
        assert_eq!(&restored, bundle.projection());
        for (name, expected) in [("pixels", &case["input"]), ("mask", &case["mask"])] {
            let NativeValue::Handle { value } = values.get(name).ok_or("Input handle")? else {
                return Err("Input handle required".into());
            };
            exact_bits(&numeric(&context, value)?, expected)?;
        }
        assert_eq!(case["input_unchanged"], true);
        assert_eq!(case["mask_unchanged"], true);
        assert_eq!(generation.len(), count + 1);
        assert_eq!(node.cache_dependencies(&context, &values)?, dependencies);
        successes += 1;
    }
    assert_eq!(successes, 12);
    assert_eq!(errors, 6);
    assert_eq!(representation_gaps, 2);
    let case = source["inpaint_cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("First source inpaint case")?;
    let values = inputs(&context, case, &vae)?;
    let count = generation.len();
    for grow in [-1, 65] {
        let mut invalid_values = values.clone();
        invalid_values.insert(
            "grow_mask_by".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(grow),
            },
        );
        assert!(
            node.demanded_lazy_inputs(&context, &invalid_values)
                .is_err()
        );
        assert!(smol::block_on(node.execute(context.clone(), invalid_values)).is_err());
        assert_eq!(generation.len(), count);
    }
    let mut invalid_values = values.clone();
    invalid_values.insert("mask".to_owned(), NativeValue::Handle { value: vae });
    assert!(smol::block_on(node.execute(context.clone(), invalid_values)).is_err());
    assert_eq!(generation.len(), count);
    let mut empty_mapped = values;
    empty_mapped.insert("mask".to_owned(), NativeValue::List { values: Vec::new() });
    assert!(node.demanded_lazy_inputs(&context, &empty_mapped).is_err());
    assert_eq!(generation.len(), count);
    Ok(())
}
#[test]
fn task633_inpaint_original_body_preparation_crop_and_even_grow() -> TestResult {
    let source = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x633302))),
        CancellationToken::default(),
        4 * 1024 * 1024,
    )?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let cases = source["inpaint_preparation_cases"]
        .as_array()
        .ok_or("Preparation-only source cases")?;
    assert_eq!(cases.len(), 12);
    for case in cases {
        assert!(
            case["scope"]
                .as_str()
                .is_some_and(|scope| scope.contains("preparation proof only"))
        );
        assert!(case.get("output").is_none());
        let tensors = ["input", "mask"]
            .into_iter()
            .map(|name| {
                let shape = serde_json::from_value::<Vec<u64>>(case[name]["shape"].clone())?;
                let values = serde_json::from_value::<Vec<f32>>(case[name]["values"].clone())?;
                Ok(tensor_from_f32(
                    compute.backend(),
                    &shape,
                    &values,
                    &execution,
                )?)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let [pixels, mask] = tensors.as_slice() else {
            return Err("Preparation inputs".into());
        };
        let count = generation.len();
        let (prepared, grown) = sim_nodes::generated_model_latent_07::prepare_source_inpaint(
            &context,
            compute.backend(),
            pixels,
            mask,
            case["compression"].as_u64().ok_or("Compression")?,
            case["options"]["grow_mask_by"].as_u64().ok_or("Grow")?,
            &execution,
        )
        .map_err(|error| format!("Preparation {}: {error:?}", case["label"]))?;
        exact_bits(
            &tensor_numeric(&context, &prepared)?,
            &case["prepared_pixels"],
        )?;
        exact_bits(&tensor_numeric(&context, &grown)?, &case["noise_mask"])?;
        exact_bits(&tensor_numeric(&context, pixels)?, &case["input"])?;
        exact_bits(&tensor_numeric(&context, mask)?, &case["mask"])?;
        assert_eq!(generation.len(), count);
    }
    Ok(())
}

#[derive(Debug)]
struct FailureStore {
    inner: Arc<dyn NativeHandleStore>,
    reject: AtomicBool,
    cancel: AtomicBool,
}
impl NativeHandleStore for FailureStore {
    fn identity(&self) -> NativeHandleStoreIdentity {
        self.inner.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
        self.inner.resolve(handle, expected, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        if self.reject.load(Ordering::Acquire) {
            return Err(NativeHandleStoreError::Rejected(
                "Injected publication failure".to_owned(),
            ));
        }
        let handle = self.inner.publish(payload, cancellation)?;
        if self.cancel.load(Ordering::Acquire) {
            cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}
#[test]
fn task633_cancel_budget_publication_and_stale_handles_leave_no_output() -> TestResult {
    let source = fixture()?;
    let directory = tempfile::tempdir()?;
    let checkpoint = source_checkpoint(&source, directory.path())?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let attempt = AttemptId(Uuid::from_u128(0x633201));
    let store = Arc::new(FailureStore {
        inner: generation.handle_store_for_attempt(attempt),
        reject: AtomicBool::new(false),
        cancel: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        16 * 1024 * 1024,
    )?;
    let vae = canonical_source_image_vae(
        &setup,
        &checkpoint,
        &sim_model::VaeKernelProfile::PixelSpaceV1,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    for name in [
        "VAEEncode",
        "VAEEncodeTiled",
        "VAEDecodeTiled",
        "VAEEncodeForInpaint",
    ] {
        let group = if name == "VAEEncodeForInpaint" {
            "inpaint_cases"
        } else {
            "cases"
        };
        let case = source[group]
            .as_array()
            .ok_or("Cases")?
            .iter()
            .find(|case| {
                case["node"] == name && case.get("output").is_some_and(|output| !output.is_null())
            })
            .ok_or("Source case")?;
        let values = inputs(&setup, case, &vae)?;
        let count = generation.len();
        let node = registry.node(name).ok_or("Node")?;
        for mode in ["precancel", "budget", "reject", "latecancel", "stale"] {
            let token = CancellationToken::default();
            if mode == "precancel" {
                token.cancel();
            }
            let other = NativeHandleStoreGeneration::new()?;
            let execution_store: Arc<dyn NativeHandleStore> = if mode == "stale" {
                other.handle_store_for_attempt(attempt)
            } else {
                store.clone()
            };
            let invocation = context(
                backend.clone(),
                &authority,
                execution_store,
                token,
                if mode == "budget" {
                    1
                } else {
                    16 * 1024 * 1024
                },
            )?;
            store.reject.store(mode == "reject", Ordering::Release);
            store.cancel.store(mode == "latecancel", Ordering::Release);
            let error = smol::block_on(node.execute(invocation, values.clone()))
                .err()
                .ok_or("Failure required")?;
            if matches!(mode, "precancel" | "latecancel") {
                assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
            }
            assert_eq!(generation.len(), count);
            assert!(other.is_empty());
            store.reject.store(false, Ordering::Release);
            store.cancel.store(false, Ordering::Release);
        }
    }
    let case = source["cases"]
        .as_array()
        .ok_or("Cases")?
        .iter()
        .find(|case| case["node"] == "VAEDecodeTiled")
        .ok_or("Tiled decode case")?;
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let video = tensor_from_f32(compute.backend(), &[1, 3, 1, 1, 1], &[0.0; 3], &execution)?;
    let audio = tensor_from_f32(compute.backend(), &[1, 1, 1], &[0.0], &execution)?;
    let nested = NativeLatentBundle::audio_video(
        video,
        audio,
        None,
        None,
        None,
        NativeLatentMetadata::default(),
        &execution,
    )?;
    let nested = setup.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(nested)),
        &setup.cancellation,
    )?;
    let mut values = inputs(&setup, case, &vae)?;
    values.insert("samples".to_owned(), NativeValue::Handle { value: nested });
    let count = generation.len();
    let node = registry.node("VAEDecodeTiled").ok_or("Tiled decode node")?;
    let error = smol::block_on(node.execute(setup, values))
        .err()
        .ok_or("Source tiled decode must reject nested samples")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(error.code, "invalid_vae_inputs");
    assert_eq!(generation.len(), count);
    Ok(())
}

fn source_noise_values(
    context: &NativeNodeContext,
    case: &Value,
) -> Result<(BTreeMap<String, NativeValue>, Tensor), Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let shape = serde_json::from_value::<Vec<u64>>(case["samples"]["shape"].clone())?;
    let values = serde_json::from_value::<Vec<f32>>(case["samples"]["values"].clone())?;
    let samples = tensor_from_f32(compute.backend(), &shape, &values, &execution)?;
    let indices = vec![17; usize::try_from(*shape.first().ok_or("Source batch")?)?];
    let latent = NativeLatentBundle::single(
        samples.clone(),
        None,
        Some(indices),
        NativeLatentMetadata::default(),
        &execution,
    )?;
    let handle = context.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(latent)),
        &context.cancellation,
    )?;
    Ok((
        BTreeMap::from([(
            "warped_noise".to_owned(),
            NativeValue::Handle { value: handle },
        )]),
        samples,
    ))
}

#[test]
fn task633_latent_noise_original_clone_and_ignore_input_contract() -> TestResult {
    let source = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x633401))),
        CancellationToken::default(),
        4 * 1024 * 1024,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("VOIDWarpedNoiseSource")
        .ok_or("Noise source node")?;
    assert_eq!(
        source["noise_source_schema"]["node_id"],
        "VOIDWarpedNoiseSource"
    );
    assert_eq!(
        source["noise_source_schema"]["inputs"],
        json!([{"name":"warped_noise","type":"LATENT","tooltip":"Warped noise latent from VOIDWarpedNoise"}])
    );
    assert_eq!(
        source["noise_source_schema"]["outputs"],
        json!([{"type":"NOISE"}])
    );
    let mut successes = 0;
    let mut gaps = 0;
    let mut errors = 0;
    for case in source["noise_source_cases"]
        .as_array()
        .ok_or("Original noise source cases")?
    {
        if case.get("capability_gap").is_some() {
            assert!(case.get("output").is_some());
            gaps += 1;
            continue;
        }
        if case.get("error_type").is_some() {
            assert_eq!(case["error_type"], "AttributeError");
            errors += 1;
            continue;
        }
        let (values, samples) = source_noise_values(&context, case)?;
        let dependencies = node.cache_dependencies(&context, &values)?;
        assert_eq!(dependencies.artifact_digests.len(), 1);
        let mapped = values
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    NativeValue::List {
                        values: vec![value.clone()],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert!(node.demanded_lazy_inputs(&context, &mapped)?.is_empty());
        assert_eq!(node.cache_dependencies(&context, &mapped)?, dependencies);
        let count = generation.len();
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = smol::block_on(node.execute(context.clone(), values.clone()))?
        else {
            return Err("Noise source values outcome required".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let [NativeValue::Handle { value: output }] = outputs.as_slice() else {
            return Err("Single NOISE handle required".into());
        };
        assert_eq!(output.handle_type().type_id, "NOISE");
        let payload =
            context
                .handle_store()
                .resolve(output, output.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Noise(noise) = payload.as_ref() else {
            return Err("Canonical NOISE resource required".into());
        };
        noise.validate()?;
        assert_eq!(
            noise.kind(),
            sim_sampler::NativeNoiseSourceKind::LatentSource
        );
        assert_eq!(noise.seed(), 0);
        assert_eq!(case["seed"], 0);
        assert_eq!(case["output_device"], "cpu");
        assert_eq!(case["independent_storage"], true);
        assert_eq!(case["repeat_unchanged_after_output_mutation"], true);
        assert_eq!(case["samples_unchanged"], true);
        assert_eq!(
            noise
                .resident_parts()?
                .tensor_allocation()
                .ok_or("Retained samples")?
                .storage_id(),
            samples.storage_id()
        );
        let fixed = sim_sampler::NativeNoisePayload::fixed_latent(0, samples.clone())?;
        assert_ne!(
            noise.semantic_digest_sha256(),
            fixed.semantic_digest_sha256()
        );
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let ignored = match case["supplied_input"].as_str().ok_or("Supplied input")? {
            "none" | "object" | "scalar" => {
                tensor_from_f32(compute.backend(), &[], &[99.0], &execution)?
            }
            "different-shape" => {
                tensor_from_f32(compute.backend(), &[2, 5, 1], &[0.0; 10], &execution)?
            }
            _ => return Err("Unreviewed source input".into()),
        };
        // The Rust API requires a Tensor; scalar substitutes witness that no supplied-input validation occurs.
        let request = sim_sampler::NoiseRequest::native_diffusion("task633", "source-noise")?;
        let first = noise.generate(
            compute.backend(),
            &ignored,
            Some(&[u64::MAX]),
            &request,
            &execution,
        )?;
        let repeated =
            noise.generate(compute.backend(), &ignored, Some(&[]), &request, &execution)?;
        exact_bits(&tensor_numeric(&context, &first.noise)?, &case["output"])?;
        exact_bits(&tensor_numeric(&context, &repeated.noise)?, &case["output"])?;
        assert_ne!(first.noise.storage_id(), samples.storage_id());
        assert_ne!(first.noise.storage_id(), repeated.noise.storage_id());
        assert!(first.before.is_none() && first.after.is_none());
        assert!(repeated.before.is_none() && repeated.after.is_none());
        exact_bits(&tensor_numeric(&context, &samples)?, &case["samples"])?;
        assert_eq!(generation.len(), count + 1);
        assert_eq!(node.cache_dependencies(&context, &values)?, dependencies);
        successes += 1;
    }
    assert_eq!(successes, 16);
    assert_eq!(gaps, 4);
    assert_eq!(errors, 1);
    Ok(())
}

#[test]
fn task633_latent_noise_authority_cancel_and_publication_are_atomic() -> TestResult {
    let source = fixture()?;
    let case = source["noise_source_cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Source noise case")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let attempt = AttemptId(Uuid::from_u128(0x633402));
    let store = Arc::new(FailureStore {
        inner: generation.handle_store_for_attempt(attempt),
        reject: AtomicBool::new(false),
        cancel: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        4 * 1024 * 1024,
    )?;
    let (values, _) = source_noise_values(&setup, case)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("VOIDWarpedNoiseSource")
        .ok_or("Noise source")?;
    let count = generation.len();
    for mode in ["precancel", "reject", "latecancel", "stale"] {
        let cancellation = CancellationToken::default();
        if mode == "precancel" {
            cancellation.cancel();
        }
        let other = NativeHandleStoreGeneration::new()?;
        let execution_store: Arc<dyn NativeHandleStore> = if mode == "stale" {
            other.handle_store_for_attempt(attempt)
        } else {
            store.clone()
        };
        let invocation = context(
            backend.clone(),
            &authority,
            execution_store,
            cancellation,
            4 * 1024 * 1024,
        )?;
        store.reject.store(mode == "reject", Ordering::Release);
        store.cancel.store(mode == "latecancel", Ordering::Release);
        let error = smol::block_on(node.execute(invocation, values.clone()))
            .err()
            .ok_or("Expected noise failure")?;
        if matches!(mode, "precancel" | "latecancel") {
            assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        }
        assert_eq!(generation.len(), count);
        assert!(other.is_empty());
        store.reject.store(false, Ordering::Release);
        store.cancel.store(false, Ordering::Release);
    }
    let mut mapped = values;
    mapped.insert(
        "warped_noise".to_owned(),
        NativeValue::List { values: Vec::new() },
    );
    assert!(node.demanded_lazy_inputs(&setup, &mapped).is_err());
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let nested = NativeLatentBundle::audio_video(
        tensor_from_f32(compute.backend(), &[1, 1, 1, 1, 1], &[0.0], &execution)?,
        tensor_from_f32(compute.backend(), &[1, 1, 1], &[0.0], &execution)?,
        None,
        None,
        None,
        NativeLatentMetadata::default(),
        &execution,
    )?;
    let nested = setup.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(nested)),
        &setup.cancellation,
    )?;
    let count = generation.len();
    let invalid = BTreeMap::from([(
        "warped_noise".to_owned(),
        NativeValue::Handle { value: nested },
    )]);
    assert!(smol::block_on(node.execute(setup, invalid)).is_err());
    assert_eq!(generation.len(), count);
    Ok(())
}

#[test]
fn task633_latent_noise_generation_allocation_failure_has_no_tensor_residue() -> TestResult {
    let source = fixture()?;
    let case = source["noise_source_cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Source noise case")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x633403))),
        CancellationToken::default(),
        32,
    )?;
    let (values, samples) = source_noise_values(&context, case)?;
    let node = generated_native_node_registry_projection(None)?;
    let node = node.node("VOIDWarpedNoiseSource").ok_or("Noise source")?;
    let NativeNodeOutcome::Values { outputs, .. } =
        smol::block_on(node.execute(context.clone(), values))?
    else {
        return Err("Noise source outcome".into());
    };
    let [NativeValue::Handle { value: handle }] = outputs.as_slice() else {
        return Err("Noise source handle".into());
    };
    let payload =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Noise(noise) = payload.as_ref() else {
        return Err("Canonical noise source".into());
    };
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let (blocker, _) = compute.backend().allocate(
        sim_tensor::TensorDescriptor::contiguous(
            vec![1],
            sim_tensor::DType::F32,
            sim_tensor::DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        &execution,
    )?;
    let count = generation.len();
    let bytes = backend.memory_snapshot().current_bytes;
    assert_eq!(bytes, 32);
    let request = sim_sampler::NoiseRequest::native_diffusion("task633", "source-noise")?;
    assert!(matches!(
        noise.generate(compute.backend(), &blocker, None, &request, &execution),
        Err(sim_sampler::NativeSamplerPayloadError::Tensor(
            sim_tensor::TensorError::AllocationFailed { requested: 16, .. }
        ))
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, bytes);
    assert_eq!(generation.len(), count);
    drop(blocker);
    exact_bits(&tensor_numeric(&context, &samples)?, &case["output"])?;
    let bytes = backend.memory_snapshot().current_bytes;
    context.cancellation.cancel();
    assert!(matches!(
        noise.generate(compute.backend(), &samples, None, &request, &execution),
        Err(sim_sampler::NativeSamplerPayloadError::Tensor(
            sim_tensor::TensorError::Cancelled
        ))
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, bytes);
    assert_eq!(generation.len(), count);
    Ok(())
}
