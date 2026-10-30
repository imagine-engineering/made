use sha2::{Digest, Sha256};
use sim_model::NativeModelPayload;
use sim_nodes::{
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeNode, NativeNodeBindingDisposition, NativeNodeComputeSession, NativeNodeContext,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices,
    NativeOpaqueHandle, NativePrimitive, NativeResolvedPayload, NativeStoredModelPayload,
    NativeStoredPayload, NativeValue,
};
use sim_runtime::{
    CacheKey, NativeHandleStoreGeneration, generated_native_node_registry_projection,
};
use sim_sampler::NativeDiffusionPayload;
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ImageTensor,
    NativeTensorPayload, NativeTensorRole, StreamId, TensorBackend, TensorDescriptor,
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
const LIMIT: u64 = 32 * 1024 * 1024;
const CLASSES: [&str; 2] = [
    "CosmosImageToVideoLatent",
    "CosmosPredict2ImageToVideoLatent",
];

fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("conditioning-part-four".to_owned());
    let scratch = authority.authorize_workspace(LIMIT / 2)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x61903), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x61901)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn inputs(
    context: &NativeNodeContext,
    backend: Arc<CpuBackend>,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let bundle = sim_test_support::NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(backend, &execution)?;
    let model = Arc::new(NativeModelPayload::native_vae(bundle.vae().clone())?);
    let vae = context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
            Arc::new(NativeDiffusionPayload::vae(model)?),
        )?)),
        &context.cancellation,
    )?;
    let mut inputs = BTreeMap::from([("vae".to_owned(), NativeValue::Handle { value: vae })]);
    for (name, value) in [
        ("width", 16),
        ("height", 16),
        ("length", 1),
        ("batch_size", 1),
    ] {
        set_integer(&mut inputs, name, value);
    }
    Ok(inputs)
}
fn set_integer(inputs: &mut BTreeMap<String, NativeValue>, name: &str, value: i64) {
    inputs.insert(
        name.to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        },
    );
}
fn output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected Cosmos value outcome".into());
    };
    assert!(ui.is_none());
    assert!(effects.is_empty());
    let [NativeValue::Handle { value }] = outputs.as_slice() else {
        return Err("Expected sole latent handle".into());
    };
    Ok(value.clone())
}
fn image(context: &NativeNodeContext) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let tensor = tensor_from_f32(compute.backend(), &[1, 16, 16, 3], &[0.0; 768], &execution)?;
    Ok(context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            ImageTensor::from_tensor(tensor)?,
        )?)),
        &context.cancellation,
    )?)
}
fn key(
    node: &dyn NativeNode,
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<CacheKey, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let capability_sha = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(compute.backend().capabilities())?)
    );
    let dependencies = node.cache_dependencies(context, inputs)?;
    Ok(CacheKey::from_inputs(
        node.class_type(),
        node.implementation_version(),
        inputs,
        dependencies.artifact_digests,
        capability_sha,
        "cpu-f32",
        None,
        None,
        "task619-cosmos-in-process",
        node.implementation_version(),
        node.cache_change_token(inputs)?,
    )?)
}

#[test]
fn task619_cosmos_generated_registry_preserves_two_exact_source_schemas() -> TestResult {
    let families = sim_nodes::generated_family_node_bindings()?;
    let registry = generated_native_node_registry_projection(None)?;
    for (class_type, feature) in CLASSES
        .into_iter()
        .zip(["COMFY-NODE-0112", "COMFY-NODE-0113"])
    {
        let matching = families
            .iter()
            .filter(|binding| binding.descriptor().class_type == class_type)
            .collect::<Vec<_>>();
        let [binding] = matching.as_slice() else {
            return Err("Expected unique Cosmos owner".into());
        };
        assert_eq!(binding.feature_id(), feature);
        binding.descriptor().validate_exact_schema_v2()?;
        assert_eq!(
            binding.disposition(),
            NativeNodeBindingDisposition::Executable
        );
        let descriptor = registry
            .descriptor(class_type)
            .ok_or("Descriptor missing")?;
        descriptor.validate_exact_schema_v2()?;
        assert!(registry.node(class_type).is_some());
        assert_eq!(descriptor.outputs.len(), 1);
        assert_eq!(
            descriptor.outputs.first().ok_or("Output missing")?.name,
            "latent"
        );
        for name in ["start_image", "end_image"] {
            assert!(
                !descriptor
                    .inputs
                    .iter()
                    .find(|input| input.name == name)
                    .ok_or("Optional source image missing")?
                    .required
            );
        }
    }
    Ok(())
}

#[test]
fn task619_cosmos_empty_source_branches_keep_batch_one_real_zero_bits_and_no_mask() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61911))),
        CancellationToken::default(),
    )?;
    let mut inputs = inputs(&context, backend.clone())?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for (class_type, expected_frames) in CLASSES
        .into_iter()
        .zip([[1, 1, 1, 1, 2, 11], [1, 1, 2, 2, 3, 21]])
    {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        for (length, frames) in [1, 4, 5, 8, 9, 81].into_iter().zip(expected_frames) {
            for batch in [1, 3, 64] {
                set_integer(&mut inputs, "width", 24);
                set_integer(&mut inputs, "height", 16);
                set_integer(&mut inputs, "length", length);
                set_integer(&mut inputs, "batch_size", batch);
                let handle = output(smol::block_on(
                    node.execute(context.clone(), inputs.clone()),
                )?)?;
                {
                    let payload = context.handle_store().resolve(
                        &handle,
                        handle.handle_type(),
                        &context.cancellation,
                    )?;
                    let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                        return Err("Canonical latent missing".into());
                    };
                    let tensor = latent.samples().tensor().ok_or("Samples tensor missing")?;
                    assert_eq!(tensor.descriptor().shape(), &[1, 16, frames, 2, 3]);
                    assert_eq!(tensor.descriptor().dtype(), DType::F32);
                    assert!(latent.noise_mask().is_none());
                    assert!(latent.batch_indices().is_none());
                    assert_eq!(
                        serde_json::to_value(latent.metadata())?,
                        serde_json::json!({})
                    );
                    let compute = context.compute_session()?;
                    let execution = compute.execution_context(&context)?;
                    let values = tensor_to_f32(compute.backend(), tensor, &execution)?;
                    assert_eq!(values.len(), (16 * frames * 2 * 3) as usize);
                    assert!(
                        values
                            .iter()
                            .all(|value| value.to_bits() == 0.0_f32.to_bits())
                    );
                }
                context
                    .handle_store()
                    .revoke(&handle, &context.cancellation)?;
                assert_eq!(
                    (
                        generation.len(),
                        generation.resident_bytes(),
                        backend.memory_snapshot().current_bytes
                    ),
                    before
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct CancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    armed: AtomicBool,
}
impl NativeHandleStore for CancelAfterPublish {
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
        let handle = self.inner.publish(payload, cancellation)?;
        if self.armed.swap(false, Ordering::AcqRel) {
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
fn task619_cosmos_cancellation_resident_oom_publication_rollback_and_recovery() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(CancelAfterPublish {
        inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61921))),
        armed: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
    )?;
    let inputs = inputs(&setup, backend.clone())?;
    for class_type in CLASSES {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        for mode in ["cancelled", "resident_capacity", "after_publish"] {
            let guard = if mode == "resident_capacity" {
                let remaining = LIMIT
                    .checked_sub(backend.memory_snapshot().current_bytes)
                    .ok_or("Limit exceeded")?;
                let compute = setup.compute_session()?;
                let execution = compute.execution_context(&setup)?;
                Some(
                    backend
                        .allocate(
                            TensorDescriptor::contiguous(
                                vec![remaining],
                                DType::U8,
                                DeviceId::CPU,
                                StreamId::DEFAULT,
                            )?,
                            &execution,
                        )?
                        .0,
                )
            } else {
                None
            };
            let before = (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes,
            );
            let cancellation = CancellationToken::default();
            if mode == "cancelled" {
                cancellation.cancel();
            }
            store
                .armed
                .store(mode == "after_publish", Ordering::Release);
            let invocation = context(backend.clone(), &authority, store.clone(), cancellation)?;
            let error = smol::block_on(node.execute(invocation, inputs.clone()))
                .err()
                .ok_or("Failure produced output")?;
            if mode == "resident_capacity" {
                assert_eq!(error.kind, NativeNodeFailureKind::Failure);
                assert_eq!(error.code, "cosmos_execution_failed");
                assert!(
                    error
                        .message
                        .contains("tensor allocation of 256 bytes failed")
                );
                assert!(error.message.contains(&format!(
                    "backend limit is {LIMIT} bytes with {LIMIT} bytes already reserved"
                )));
            } else {
                assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
                assert_eq!(error.code, "execution_interrupted");
            }
            assert_eq!(
                (
                    generation.len(),
                    generation.resident_bytes(),
                    backend.memory_snapshot().current_bytes
                ),
                before
            );
            drop(guard);
        }
        let handle = output(smol::block_on(node.execute(setup.clone(), inputs.clone()))?)?;
        setup.handle_store().revoke(&handle, &setup.cancellation)?;
    }
    Ok(())
}

#[test]
fn task619_cosmos_images_wrong_foreign_and_stale_handles_fail_without_publication() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61931))),
        CancellationToken::default(),
    )?;
    let inputs = inputs(&setup, backend.clone())?;
    let image = image(&setup)?;
    let foreign_generation = NativeHandleStoreGeneration::new()?;
    let foreign = context(
        backend.clone(),
        &authority,
        foreign_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61932))),
        CancellationToken::default(),
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for class_type in CLASSES {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        for name in ["start_image", "end_image"] {
            let mut with_image = inputs.clone();
            with_image.insert(
                name.to_owned(),
                NativeValue::Handle {
                    value: image.clone(),
                },
            );
            let error = smol::block_on(node.execute(setup.clone(), with_image))
                .err()
                .ok_or("Image gate unexpectedly executed")?;
            assert_eq!(error.code, "cosmos_video_conditioning_unbound");
        }
        let mut present_null = inputs.clone();
        present_null.insert(
            "start_image".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Null,
            },
        );
        let error = smol::block_on(node.execute(setup.clone(), present_null))
            .err()
            .ok_or("Nullable image bypassed canonical type admission")?;
        assert_eq!(error.code, "invalid_cosmos_inputs");
        let mut wrong = inputs.clone();
        wrong.insert(
            "vae".to_owned(),
            NativeValue::Handle {
                value: image.clone(),
            },
        );
        let error = smol::block_on(node.execute(setup.clone(), wrong))
            .err()
            .ok_or("Wrong VAE type admitted")?;
        assert_eq!(error.code, "invalid_cosmos_inputs");
        assert!(smol::block_on(node.execute(foreign.clone(), inputs.clone())).is_err());
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        for (name, value) in [("length", 0), ("width", 15), ("batch_size", 0)] {
            let mut invalid = inputs.clone();
            set_integer(&mut invalid, name, value);
            let error = smol::block_on(node.execute(setup.clone(), invalid))
                .err()
                .ok_or("Invalid source schema admitted")?;
            assert_eq!(error.code, "invalid_cosmos_inputs");
        }
    }
    let Some(NativeValue::Handle { value: vae }) = inputs.get("vae") else {
        return Err("VAE absent".into());
    };
    setup.handle_store().revoke(vae, &setup.cancellation)?;
    let after_revoke = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for class_type in CLASSES {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        assert!(smol::block_on(node.execute(setup.clone(), inputs.clone())).is_err());
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            after_revoke
        );
    }
    Ok(())
}

#[test]
fn task619_cosmos_cache_identity_keeps_unused_batch_inputs_and_temporal_profiles_distinct()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61941))),
        CancellationToken::default(),
    )?;
    let mut inputs = inputs(&setup, backend)?;
    let first = registry.node(CLASSES[0]).ok_or("Cosmos binding missing")?;
    let second = registry
        .node(CLASSES[1])
        .ok_or("Cosmos Predict2 binding missing")?;
    let original = key(first.as_ref(), &setup, &inputs)?;
    assert_eq!(key(first.as_ref(), &setup, &inputs)?, original);
    assert_ne!(key(second.as_ref(), &setup, &inputs)?, original);
    set_integer(&mut inputs, "batch_size", 2);
    assert_ne!(key(first.as_ref(), &setup, &inputs)?, original);
    set_integer(&mut inputs, "batch_size", 1);
    set_integer(&mut inputs, "length", 9);
    assert_ne!(key(first.as_ref(), &setup, &inputs)?, original);
    assert!(
        first
            .cache_dependencies(&setup, &inputs)?
            .artifact_digests
            .contains_key("vae.0")
    );
    Ok(())
}

#[test]
fn task619_cosmos_handle_capacity_failure_releases_tensor_and_fresh_store_recovers() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let full = NativeHandleStoreGeneration::with_capacity(1)?;
    let setup = context(
        backend.clone(),
        &authority,
        full.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61951))),
        CancellationToken::default(),
    )?;
    let base_inputs = inputs(&setup, backend.clone())?;
    let before = (
        full.len(),
        full.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert_eq!(before.0, 1);
    for class_type in CLASSES {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        let error = smol::block_on(node.execute(setup.clone(), base_inputs.clone()))
            .err()
            .ok_or("Full handle store produced output")?;
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert_eq!(error.code, "cosmos_execution_failed");
        assert_eq!(
            (
                full.len(),
                full.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    let recovered = NativeHandleStoreGeneration::with_capacity(2)?;
    let recovery = context(
        backend.clone(),
        &authority,
        recovered.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61952))),
        CancellationToken::default(),
    )?;
    let mut recovery_inputs = inputs(&recovery, backend)?;
    recovery_inputs.insert(
        "width".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(16),
        },
    );
    for class_type in CLASSES {
        let node = registry.node(class_type).ok_or("Cosmos binding missing")?;
        let handle = output(smol::block_on(
            node.execute(recovery.clone(), recovery_inputs.clone()),
        )?)?;
        assert_eq!(recovered.len(), 2);
        recovery
            .handle_store()
            .revoke(&handle, &recovery.cancellation)?;
        assert_eq!(recovered.len(), 1);
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalCosmosEmptyOracle {
    format: String,
    producer_sha256: String,
    source_sha256: BTreeMap<String, String>,
    torch_version: String,
    cases: Vec<OriginalCosmosEmptyCase>,
    case_count: usize,
    qualification: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalCosmosEmptyCase {
    class_type: String,
    width: i64,
    height: i64,
    length: i64,
    batch_size: i64,
    start_image: String,
    end_image: String,
    latent_keys: Vec<String>,
    shape: Vec<u64>,
    dtype: String,
    values: Vec<f32>,
    storage_sha256: String,
}

#[test]
fn task619_cosmos_original_imported_36_cases_execute_registered_native_nodes() -> TestResult {
    let bytes = include_bytes!(
        "../fixtures/nodes/model-conditioning-comfy-node-0107/source-original-cosmos-empty.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "6e36ef97a2971af79fbf03219e56a58a4281aee48c622e0c1058daeab8f61e18"
    );
    let oracle: OriginalCosmosEmptyOracle = serde_json::from_slice(bytes)?;
    assert_eq!(
        oracle.format,
        "comfy.source.cosmos-empty.original-cpu-f32.v1"
    );
    assert_eq!(
        oracle.producer_sha256,
        "c45f50841369857e271d8f75ea7d76d8ca07ff842dcb9347ea955312f74df8c7"
    );
    assert_eq!(oracle.case_count, 36);
    assert_eq!(oracle.cases.len(), 36);
    assert!(!oracle.torch_version.is_empty());
    assert!(
        oracle
            .qualification
            .contains("Genuine original imported two Cosmos classes")
    );
    let expected_source_pins: BTreeMap<String, String> = serde_json::from_str(
        r#"{
        "comfy_extras/nodes_cosmos.py":"f0f3e57a13a742720efce54a939760ebccf55d7f7230fa5020404160e5eaf9eb",
        "comfy/model_management.py":"c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
        "comfy_api/latest/_io.py":"ac986092cace7600461b8547e21f770126958d8c15eef59d79d6709cca4b7717",
        "comfy_api/internal/__init__.py":"50786f7e2ebc910ed9becddf066e92d7924b7f6080189651067bbcd839e8a410",
        "nodes.py":"b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b"
    }"#,
    )?;
    assert_eq!(oracle.source_sha256, expected_source_pins);
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61936))),
        CancellationToken::default(),
    )?;
    let mut inputs = inputs(&context, backend.clone())?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let mut observed = std::collections::BTreeSet::new();
    for case in &oracle.cases {
        assert!(CLASSES.contains(&case.class_type.as_str()));
        assert_eq!((case.width, case.height), (24, 16));
        assert!([1, 4, 5, 8, 9, 81].contains(&case.length));
        assert!([1, 3, 64].contains(&case.batch_size));
        assert!(
            observed.insert((case.class_type.as_str(), case.length, case.batch_size)),
            "duplicate Source case"
        );
        assert_eq!(case.start_image, "absent");
        assert_eq!(case.end_image, "absent");
        assert_eq!(case.latent_keys, ["samples"]);
        assert_eq!(case.dtype, "torch.float32");
        assert_eq!(case.shape.first(), Some(&1));
        for (name, value) in [
            ("width", case.width),
            ("height", case.height),
            ("length", case.length),
            ("batch_size", case.batch_size),
        ] {
            set_integer(&mut inputs, name, value);
        }
        let node = registry
            .node(&case.class_type)
            .ok_or("Source Cosmos binding missing")?;
        let handle = output(smol::block_on(
            node.execute(context.clone(), inputs.clone()),
        )?)?;
        {
            let retained = context.handle_store().resolve(
                &handle,
                handle.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Latent(latent) = retained.as_ref() else {
                return Err("Source case returned non-latent payload".into());
            };
            let tensor = latent
                .samples()
                .tensor()
                .ok_or("Source case has no samples tensor")?;
            assert_eq!(
                tensor.descriptor().shape(),
                case.shape.as_slice(),
                "{} length{} batch{}",
                case.class_type,
                case.length,
                case.batch_size
            );
            assert_eq!(tensor.descriptor().dtype(), DType::F32);
            assert_eq!(tensor.descriptor().device(), DeviceId::CPU);
            assert!(latent.noise_mask().is_none());
            assert!(latent.batch_indices().is_none());
            assert_eq!(
                serde_json::to_value(latent.metadata())?,
                serde_json::json!({})
            );
            let compute = context.compute_session()?;
            let execution = compute.execution_context(&context)?;
            let actual = tensor_to_f32(compute.backend(), tensor, &execution)?;
            assert_eq!(actual.len(), case.values.len());
            for (index, (actual, expected)) in actual.iter().zip(&case.values).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{} length{} batch{} value{}",
                    case.class_type,
                    case.length,
                    case.batch_size,
                    index
                );
            }
            assert_eq!(
                format!("{:x}", Sha256::digest(tensor.contiguous_bytes()?)),
                case.storage_sha256
            );
        }
        context
            .handle_store()
            .revoke(&handle, &context.cancellation)?;
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    assert_eq!(observed.len(), 36);
    Ok(())
}

#[test]
fn task619_cosmos_mapped_executor_cache_leases_terminal_release_and_recovery() -> TestResult {
    fn create_executor(
        backend: Arc<CpuBackend>,
    ) -> Result<
        (
            sim_runtime::NativeImageExecutor,
            BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
        ),
        Box<dyn Error>,
    > {
        let mut descriptors = BTreeMap::new();
        let executor =
            sim_runtime::NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
                sim_types::ProfileId(Uuid::from_u128(0x619710)),
                BTreeMap::new(),
                true,
                backend,
                Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()),
                |registry| {
                    for class_type in CLASSES.into_iter().chain(["CheckpointLoaderSimple"]) {
                        let descriptor = registry.descriptor(class_type).ok_or_else(|| {
                            sim_runtime::NativeImageRuntimeError::Registry(format!(
                                "Missing genuine registered {class_type}"
                            ))
                        })?;
                        if class_type != "CheckpointLoaderSimple" {
                            descriptor.validate_exact_schema_v2().map_err(|error| {
                                sim_runtime::NativeImageRuntimeError::Registry(error.to_string())
                            })?;
                        }
                        if registry.binding_disposition(class_type)
                            != Some(NativeNodeBindingDisposition::Executable)
                        {
                            return Err(sim_runtime::NativeImageRuntimeError::Registry(format!(
                                "Non-executable {class_type}"
                            )));
                        }
                        descriptors.insert(class_type.to_owned(), descriptor.clone());
                    }
                    Ok(())
                },
            )?;
        Ok((executor, descriptors))
    }
    fn create_plan(
        descriptors: &BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
        batches: [i64; 2],
    ) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
        let integer_list = |values: [i64; 2]| sim_runtime::InputBinding::Literal {
            value: NativeValue::List {
                values: values
                    .into_iter()
                    .map(|value| NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    })
                    .collect(),
            },
        };
        let mut nodes = BTreeMap::new();
        let mut order = Vec::new();
        let mut outputs = Vec::new();
        for (identifier, class_type) in [
            ("checkpoint", "CheckpointLoaderSimple"),
            ("cosmos", CLASSES[0]),
            ("predict2", CLASSES[1]),
        ] {
            let identifier = NodeId(identifier.to_owned());
            let mut descriptor = descriptors
                .get(class_type)
                .ok_or("Actual mapped workflow descriptor missing")?
                .clone();
            let inputs = if class_type == "CheckpointLoaderSimple" {
                BTreeMap::from([(
                    "ckpt_name".to_owned(),
                    sim_runtime::InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::String("model.safetensors".to_owned()),
                        },
                    },
                )])
            } else {
                descriptor.output_node = true;
                outputs.push(identifier.clone());
                BTreeMap::from([
                    (
                        "vae".to_owned(),
                        sim_runtime::InputBinding::Link {
                            source: NodeId("checkpoint".to_owned()),
                            output_index: 2,
                            lazy: false,
                            mode: sim_nodes::NativePortCardinality::Mapped,
                        },
                    ),
                    ("width".to_owned(), integer_list([16, 24])),
                    ("height".to_owned(), integer_list([24, 16])),
                    ("length".to_owned(), integer_list([9, 8])),
                    ("batch_size".to_owned(), integer_list(batches)),
                ])
            };
            nodes.insert(
                identifier.clone(),
                sim_runtime::CompiledNode {
                    id: identifier.clone(),
                    class_type: class_type.to_owned(),
                    descriptor,
                    inputs,
                    unknown: BTreeMap::new(),
                },
            );
            order.push(identifier);
        }
        outputs.sort();
        let plan = sim_runtime::CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x619711)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes,
            static_required_nodes: order.iter().cloned().collect(),
            topological_order: order,
            output_nodes: outputs,
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        Ok(serde_json::from_slice(&serde_json::to_vec(&plan)?)?)
    }
    fn columns(
        result: &sim_runtime::NativeImageExecutionResult,
        identifier: &str,
    ) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
        let values = result
            .report
            .outputs
            .get(&NodeId(identifier.to_owned()))
            .ok_or("Actual mapped output missing")?;
        let [NativeValue::List { values }] = values.as_slice() else {
            return Err("Expected one real mapped latent output column".into());
        };
        values
            .iter()
            .map(|value| match value {
                NativeValue::Handle { value } => Ok(value.clone()),
                _ => Err("Mapped output must retain a canonical latent handle".into()),
            })
            .collect()
    }
    fn inspect(
        result: &sim_runtime::NativeImageExecutionResult,
        context: &NativeNodeContext,
    ) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
        assert_eq!(
            result.report.state,
            sim_runtime::AttemptState::Succeeded,
            "{:?}",
            result.report.error
        );
        assert_eq!(
            result.report.outputs.keys().cloned().collect::<Vec<_>>(),
            ["checkpoint", "cosmos", "predict2"].map(|identifier| NodeId(identifier.to_owned()))
        );
        let checkpoint = result
            .report
            .outputs
            .get(&NodeId("checkpoint".to_owned()))
            .ok_or("Executed checkpoint outputs missing")?;
        assert_eq!(checkpoint.len(), 3);
        let Some(NativeValue::Handle { value: vae }) = checkpoint.get(2) else {
            return Err("Actual scalar checkpoint VAE handle missing".into());
        };
        let vae_handle_type = NativeHandleType::new(sim_nodes::NativeHandleKind::Vae, "VAE")?;
        assert_eq!(vae.handle_type(), &vae_handle_type);
        assert!(matches!(
            context
                .handle_store()
                .resolve(vae, &vae_handle_type, &context.cancellation)?
                .as_ref(),
            NativeStoredPayload::Model(_)
        ));
        let mut handles = Vec::new();
        for (identifier, temporal) in [("cosmos", [2, 1]), ("predict2", [3, 2])] {
            let column = columns(result, identifier)?;
            assert_eq!(column.len(), 2);
            for (index, handle) in column.iter().enumerate() {
                let retained = context.handle_store().resolve(
                    handle,
                    handle.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(latent) = retained.as_ref() else {
                    return Err("Mapped Cosmos output is not a latent".into());
                };
                let tensor = latent.samples().tensor().ok_or("Samples tensor missing")?;
                let (height, width) = *[(3, 2), (2, 3)]
                    .get(index)
                    .ok_or("Mapped dimensions missing")?;
                let frames = *temporal
                    .get(index)
                    .ok_or("Mapped temporal dimension missing")?;
                assert_eq!(tensor.descriptor().shape(), &[1, 16, frames, height, width]);
                assert_eq!(tensor.descriptor().dtype(), DType::F32);
                assert_eq!(tensor.descriptor().device(), DeviceId::CPU);
                assert!(latent.noise_mask().is_none());
                assert!(latent.batch_indices().is_none());
                assert_eq!(
                    serde_json::to_value(latent.metadata())?,
                    serde_json::json!({})
                );
                let compute = context.compute_session()?;
                let execution = compute.execution_context(context)?;
                assert!(
                    tensor_to_f32(compute.backend(), tensor, &execution)?
                        .iter()
                        .all(|value| value.to_bits() == 0.0_f32.to_bits())
                );
            }
            handles.extend(column);
        }
        Ok(handles)
    }
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = create_executor(backend.clone())?;
    let plan = create_plan(&descriptors, [3, 64])?;
    // The real conformance VAE is linked through its loader; Source's absent-image branch never encodes with it.
    let first = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x619712)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(LIMIT / 2)?,
    )?;
    assert_eq!(first.report.cache_hits, 0);
    let generation = executor.handle_store_generation_for_test();
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x619712))),
        CancellationToken::default(),
    )?;
    let original = inspect(&first, &setup)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
        executor.cache_len_for_test(),
    );
    let cached = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x619713)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(LIMIT / 2)?,
    )?;
    assert_eq!(cached.report.cache_hits, 3);
    let replayed = inspect(&cached, &setup)?;
    assert_eq!(original, replayed);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
            executor.cache_len_for_test()
        ),
        before
    );
    let changed = create_plan(&descriptors, [1, 1])?;
    let changed = executor.execute_blocking(
        &changed,
        AttemptId(Uuid::from_u128(0x619714)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(LIMIT / 2)?,
    )?;
    assert_eq!(changed.report.cache_hits, 1);
    let changed_handles = inspect(&changed, &setup)?;
    for (original, changed) in original.iter().zip(&changed_handles) {
        assert_eq!(original.digest_sha256(), changed.digest_sha256());
        assert_ne!(original.identifier(), changed.identifier());
    }
    drop(executor);
    assert!(
        !generation.is_empty(),
        "Live reports must still lease committed outputs"
    );
    drop(first);
    drop(cached);
    drop(changed);
    assert!(generation.is_empty());
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    for handle in &original {
        assert!(matches!(
            setup
                .handle_store()
                .resolve(handle, handle.handle_type(), &setup.cancellation),
            Err(NativeHandleStoreError::Missing(_))
        ));
    }
    let (recovered, restored_descriptors) = create_executor(backend.clone())?;
    let restored = create_plan(&restored_descriptors, [3, 64])?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        recovered.execute_blocking(
            &restored,
            AttemptId(Uuid::from_u128(0x619715)),
            cancellation,
            0,
            authority.authorize_workspace(LIMIT / 2)?
        ),
        Err(sim_runtime::NativeImageRuntimeError::Cancelled)
    ));
    assert!(recovered.handle_store_generation_for_test().is_empty());
    assert_eq!(recovered.cache_len_for_test(), 0);
    let recovery = recovered.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x619716)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(LIMIT / 2)?,
    )?;
    assert_eq!(recovery.report.cache_hits, 0);
    let recovered_generation = recovered.handle_store_generation_for_test();
    let recovery_context = context(
        backend.clone(),
        &authority,
        recovered_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x619716))),
        CancellationToken::default(),
    )?;
    let recovered_handles = inspect(&recovery, &recovery_context)?;
    for (original, recovered) in original.iter().zip(&recovered_handles) {
        assert_ne!(original.store_identity(), recovered.store_identity());
        assert_eq!(original.digest_sha256(), recovered.digest_sha256());
    }
    drop(recovered);
    drop(recovery);
    assert!(recovered_generation.is_empty());
    assert_eq!(recovered_generation.resident_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}
