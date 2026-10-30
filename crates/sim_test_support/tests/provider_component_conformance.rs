use smol::process::Command;
use std::{error::Error, fs, path::PathBuf};

#[test]
fn provider_component_nested_workspace_is_locked_offline_and_conformant()
-> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = repository.join("crates/sim_plugin_host/provider_components/Cargo.toml");
    let lockfile = repository.join("crates/sim_plugin_host/provider_components/Cargo.lock");
    if !manifest.is_file() || !lockfile.is_file() {
        return Err("nested provider component workspace is not locked".into());
    }
    let target = std::env::temp_dir().join(format!(
        "sim-provider-component-conformance-{}",
        std::process::id()
    ));
    let output = smol::block_on(async {
        Command::new(env!("CARGO"))
            .arg("test")
            .arg("--locked")
            .arg("--offline")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--test")
            .arg("conformance")
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .await
    })?;
    if !output.status.success() {
        return Err(format!(
            "nested provider conformance failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    if target.exists() {
        fs::remove_dir_all(target)?;
    }
    Ok(())
}

mod beeble_alpha {
    use sim_nodes::{
        NativeNodeContext, NativeNodeFailureKind, NativePrimitive, NativeStoredPayload,
        NativeStructuredValue, NativeValue,
    };
    use sim_plugin_host::prepare_native_provider_invocation;
    use sim_plugin_sdk::{
        ApiVersion, CachePolicy, DeterminismPolicy, EffectPolicy, PluginNode, PluginPort,
        PluginValue, PortCardinality, PortDirection, PortPresence, PortSerialization, TypeRegistry,
    };
    use sim_runtime::{AttemptId, NativeHandleStoreGeneration, PromptId};
    use sim_tensor::{CancellationToken, CpuBackend, CpuWorkspaceAuthority, StreamId};
    use sim_types::NodeId;
    type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, error::Error, sync::Arc};
    use uuid::Uuid;

    fn hex_digest(bytes: &[u8; 32]) -> String {
        bytes
            .iter()
            .flat_map(|byte| {
                const DIGITS: &[u8; 16] = b"0123456789abcdef";
                [
                    char::from(DIGITS[usize::from(byte >> 4)]),
                    char::from(DIGITS[usize::from(byte & 15)]),
                ]
            })
            .collect()
    }

    fn input_port(
        registry: &TypeRegistry,
        source_type: &str,
        serialization: PortSerialization,
    ) -> Result<PluginPort, Box<dyn Error>> {
        Ok(PluginPort {
            id: "input".to_owned(),
            name: "input".to_owned(),
            direction: PortDirection::Input,
            type_id: registry.resolve(source_type)?.clone(),
            cardinality: PortCardinality::Singular,
            presence: PortPresence::Required,
            hidden: false,
            lazy: false,
            default: None,
            serialization,
            accepted_legacy_names: Vec::new(),
        })
    }
    #[derive(Debug)]
    struct RecordingAlphaCodec {
        identity: sim_nodes::NativeComponentH264Mp4BackingServiceIdentity,
        backend: Arc<CpuBackend>,
        observed: std::sync::Mutex<Vec<Arc<sim_media::NativeVideoPayload>>>,
        outcome: std::sync::atomic::AtomicU8,
    }

    impl sim_nodes::NativeComponentH264Mp4BackingService for RecordingAlphaCodec {
        fn identity(&self) -> &sim_nodes::NativeComponentH264Mp4BackingServiceIdentity {
            &self.identity
        }

        fn encode_backing(
            &self,
            request: sim_nodes::NativeComponentH264Mp4BackingRequest,
            context: &sim_tensor::ExecutionContext<'_>,
        ) -> BoxFuture<
            'static,
            Result<
                sim_media::NativeVideoPayload,
                sim_nodes::NativeComponentH264Mp4BackingServiceError,
            >,
        > {
            use sim_nodes::NativeComponentH264Mp4BackingServiceError as Error;
            let result = (|| {
                let video = request.into_video();
                self.observed
                    .lock()
                    .map_err(|_| Error::InvalidProjection)?
                    .push(video.clone());
                match self.outcome.load(std::sync::atomic::Ordering::SeqCst) {
                    1 => return Err(Error::Cancelled),
                    2 => return Ok(video.as_ref().clone()),
                    _ => {}
                }
                let bytes = b"recording codec boundary";
                let (encoded, _) = self
                    .backend
                    .upload_bytes(
                        sim_tensor::TensorDescriptor::contiguous(
                            vec![bytes.len() as u64],
                            sim_tensor::DType::U8,
                            sim_tensor::DeviceId::CPU,
                            context.stream,
                        )
                        .map_err(|_| Error::InvalidRequest)?,
                        bytes,
                        context,
                    )
                    .map_err(|_| Error::ResourceExhausted)?;
                sim_media::NativeVideoPayload::checked_h264_mp4_from_component(
                    &video,
                    encoded,
                    Sha256::digest(bytes).into(),
                    video.dimensions(),
                    if self.outcome.load(std::sync::atomic::Ordering::SeqCst) == 3 {
                        (30, 1)
                    } else {
                        (2_997, 100)
                    },
                    video.frame_count(),
                )
                .map_err(|_| Error::InvalidProjection)
            })();
            Box::pin(async move { result })
        }
    }

    #[test]
    fn beeble_video_frame_limit_applies_without_custom_alpha() -> Result<(), Box<dyn Error>> {
        use sim_media::{NativeVideoBitDepth, NativeVideoPayload};
        use sim_runtime::NativeHandleStoreGeneration;
        use sim_tensor::ImageTensor;

        let types = TypeRegistry::built_in()?;
        let mut video_port = input_port(&types, "VIDEO", PortSerialization::Handle)?;
        video_port.id = "video".to_owned();
        let mut mode_port = input_port(&types, "STRING", PortSerialization::Inline)?;
        mode_port.id = "alpha_mode".to_owned();
        let node = PluginNode {
            id: "BeebleSwitchXVideoEdit".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "Beeble".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![video_port, mode_port],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor(&node.id)
            .ok_or("missing Beeble descriptor")?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024)?,
            &cancellation,
        );
        let generation = NativeHandleStoreGeneration::with_capacities(2, 16 * 1024)?;
        let attempt = AttemptId(Uuid::from_u128(0x43650));
        let store = generation.handle_store_for_attempt(attempt);
        let mut videos = Vec::new();
        for frame_count in [240, 241] {
            let images = ImageTensor::from_f32(
                &backend,
                &execution,
                frame_count,
                1,
                1,
                3,
                &vec![0.5; usize::try_from(frame_count)? * 3],
            )?;
            let video = NativeVideoPayload::checked(
                images.tensor().clone(),
                30,
                1,
                NativeVideoBitDepth::Eight,
                None,
                None,
                BTreeMap::new(),
            )?;
            videos.push((
                frame_count,
                store.publish(NativeStoredPayload::Video(Arc::new(video)), &cancellation)?,
            ));
        }
        let context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x43651)),
            attempt,
            NodeId::from("beeble-video-frame-limit"),
            cancellation.clone(),
            authority.authorize_workspace(0)?,
            store,
        )?;
        let resident_bytes = generation.resident_bytes();
        for mode in ["auto", "fill"] {
            for (frame_count, video) in videos.iter().chain(videos.iter().take(1)) {
                let result = smol::block_on(prepare_native_provider_invocation(
                    &node,
                    descriptor,
                    BTreeMap::from([
                        (
                            "video".to_owned(),
                            NativeValue::Handle {
                                value: video.clone(),
                            },
                        ),
                        (
                            "alpha_mode".to_owned(),
                            NativeStructuredValue::checked(
                                "COMFY_DYNAMICCOMBO_V3",
                                BTreeMap::from([(
                                    "alpha_mode".to_owned(),
                                    NativeValue::Primitive {
                                        value: NativePrimitive::String(mode.to_owned()),
                                    },
                                )]),
                            )?
                            .into_runtime_value()?,
                        ),
                    ]),
                    "profile-a",
                    &context,
                ));
                if *frame_count > 240 {
                    let failure = result
                        .err()
                        .ok_or("oversized source passed without custom alpha")?;
                    assert_eq!(failure.kind, NativeNodeFailureKind::Failure);
                    assert!(failure.message.contains("240"));
                } else {
                    result?;
                }
                assert_eq!(generation.len(), 2);
                assert_eq!(generation.resident_bytes(), resident_bytes);
            }
        }
        Ok(())
    }

    #[test]
    fn beeble_alpha_video_preserves_source_frames_and_async_attachment_identity()
    -> Result<(), Box<dyn Error>> {
        use sim_media::{NativeVideoBitDepth, NativeVideoPayload};
        use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
        use sim_tensor::{ImageTensor, NativeTensorPayload, NativeTensorRole};
        #[derive(serde::Deserialize)]
        struct Fixture {
            schema_version: u16,
            sources: BTreeMap<String, String>,
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            shape: Vec<u64>,
            pixels: Vec<f32>,
            maximum_pixels: u64,
            source_frame_count: u64,
            frame_rate: (u64, u64),
            output: Option<Output>,
            error: Option<String>,
        }
        #[derive(serde::Deserialize)]
        struct Output {
            shape: Vec<u64>,
            pixels: Vec<f32>,
            frame_rate: (u64, u64),
            bit_depth: u8,
            audio: bool,
            encode_frame_rate: (u64, u64),
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/providers/beeble/source_alpha_video.json"
        ))?;
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.cases.len(), 16);
        assert_eq!(
            fixture.sources,
            BTreeMap::from([
                (
                    "comfy/utils.py".to_owned(),
                    "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb".to_owned(),
                ),
                (
                    "comfy_api_nodes/util/conversions.py".to_owned(),
                    "632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2".to_owned(),
                ),
                (
                    "comfy_api_nodes/util/upload_helpers.py".to_owned(),
                    "aa761b97035139e84250579e58b85497fb64f6f40137ab8089757fbe2dde2c39".to_owned(),
                ),
                (
                    "comfy_api_nodes/nodes_beeble.py".to_owned(),
                    "a7d98292ba01564a4a8519ba3c8caa30d9e0aaf42bf16ca55e29944222896812".to_owned(),
                ),
                (
                    "comfy_api/latest/_input_impl/video_types.py".to_owned(),
                    "b74216bc76178e9a21769e60cf4fd910183a19236cfa7fb352e74f2b13e482e2".to_owned(),
                ),
            ])
        );
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let backend = Arc::new(backend);
        let codec = Arc::new(RecordingAlphaCodec {
            identity: sim_nodes::NativeComponentH264Mp4BackingServiceIdentity::checked(
                "c".repeat(64),
            )?,
            backend: backend.clone(),
            observed: Default::default(),
            outcome: Default::default(),
        });
        let cancellation = CancellationToken::default();
        let scratch = authority.authorize_workspace(4 * 1024 * 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x43640));
        let node_id = NodeId::from("beeble-alpha-video");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x43641),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let generation = NativeHandleStoreGeneration::with_capacities(3, 1024 * 1024)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x43642)),
            attempt_id,
            node_id,
            cancellation.clone(),
            scratch.clone(),
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?
                .with_component_h264_mp4_backing(codec.clone())?,
        )?;
        let execution = context.compute_session()?.execution_context(&context)?;
        for case in fixture.cases {
            let (mask, _) = backend.upload_f32(
                sim_tensor::TensorDescriptor::contiguous(
                    case.shape,
                    sim_tensor::DType::F32,
                    sim_tensor::DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &case.pixels,
                &execution,
            )?;
            let before = codec.observed.lock().map_err(|_| "poisoned codec")?.len();
            let result = smol::block_on(context.prepare_provider_mask_video(
                &mask,
                case.frame_rate,
                case.source_frame_count,
                case.maximum_pixels,
            ));
            let mut observed = codec.observed.lock().map_err(|_| "poisoned codec")?;
            if let Some(expected) = case.output {
                assert_eq!(result?, b"recording codec boundary");
                assert_eq!(observed.len(), before + 1);
                let video = observed.pop().ok_or("missing video")?;
                let components = video.components().ok_or("missing components")?;
                assert_eq!(components.frames().descriptor().shape(), expected.shape);
                let pixels = components
                    .frames()
                    .contiguous_bytes()?
                    .chunks_exact(4)
                    .map(|bytes| Ok(f32::from_le_bytes(bytes.try_into()?)))
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                assert_eq!(pixels, expected.pixels);
                assert_eq!(video.frame_rate(), expected.frame_rate);
                assert_eq!(video.bit_depth().bits(), expected.bit_depth);
                assert_eq!(expected.encode_frame_rate, (2_997, 100));
                assert_eq!(components.audio().is_some(), expected.audio);
            } else {
                let failure = result.expect_err("mismatched alpha frame count must fail");
                assert_eq!(Some(failure.message), case.error);
                assert_eq!(observed.len(), before);
            }
            assert_eq!(scratch.in_use_bytes(), 0);
        }
        let image = ImageTensor::from_f32(&backend, &execution, 3, 2, 2, 3, &[0.5; 36])?;
        let source = store.publish(
            NativeStoredPayload::Video(Arc::new(NativeVideoPayload::checked(
                image.tensor().clone(),
                30000,
                1001,
                NativeVideoBitDepth::Eight,
                None,
                None,
                BTreeMap::new(),
            )?)),
            &cancellation,
        )?;
        let (mask, _) = backend.upload_f32(
            sim_tensor::TensorDescriptor::contiguous(
                vec![3, 2, 2],
                sim_tensor::DType::F32,
                sim_tensor::DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0.0, 0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0, 0.2, 0.4, 0.6, 0.8],
            &execution,
        )?;
        let mask_handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_tensor(
                NativeTensorRole::Mask,
                mask,
            )?)),
            &cancellation,
        )?;
        let types = TypeRegistry::built_in()?;
        let mut video_port = input_port(&types, "VIDEO", PortSerialization::Handle)?;
        video_port.id = "video".to_owned();
        video_port.name = "video".to_owned();
        let mut mask_port = input_port(&types, "MASK", PortSerialization::Handle)?;
        mask_port.id = "alpha_mask".to_owned();
        mask_port.name = "alpha_mask".to_owned();
        let mut mode_port = input_port(&types, "String", PortSerialization::Inline)?;
        mode_port.id = "alpha_mode".to_owned();
        mode_port.name = "alpha_mode".to_owned();
        let node = PluginNode {
            id: "BeebleSwitchXVideoEdit".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "Beeble".to_owned(),
            category: "test".to_owned(),
            ports: vec![video_port, mask_port, mode_port],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor(&node.id)
            .ok_or("missing Beeble descriptor")?;
        let values = BTreeMap::from([
            ("video".to_owned(), NativeValue::Handle { value: source }),
            (
                "alpha_mode".to_owned(),
                NativeStructuredValue::checked(
                    "COMFY_DYNAMICCOMBO_V3",
                    BTreeMap::from([
                        (
                            "alpha_mode".to_owned(),
                            NativeValue::Primitive {
                                value: NativePrimitive::String("custom".to_owned()),
                            },
                        ),
                        (
                            "alpha_mask".to_owned(),
                            NativeValue::Handle { value: mask_handle },
                        ),
                    ]),
                )?
                .into_runtime_value()?,
            ),
        ]);
        let resident = generation.resident_bytes();
        let prepared = smol::block_on(prepare_native_provider_invocation(
            &node,
            descriptor,
            values.clone(),
            "profile-a",
            &context,
        ))?;
        let input_json = serde_json::to_value(&prepared.inputs)?;
        let attachment = input_json
            .pointer("/media_batches/alpha_mask/0/0")
            .ok_or("missing alpha MP4")?;
        assert_eq!(
            attachment
                .get("media_type")
                .and_then(serde_json::Value::as_str),
            Some("video/mp4")
        );
        let bytes: Vec<u8> = serde_json::from_value(
            attachment
                .get("bytes")
                .ok_or("missing media bytes")?
                .clone(),
        )?;
        assert_eq!(bytes, b"recording codec boundary");
        let mask_value: PluginValue = serde_json::from_value(
            input_json
                .pointer("/values/alpha_mask/0")
                .ok_or("missing mask identity")?
                .clone(),
        )?;
        assert_eq!(
            attachment
                .get("source_value_sha256")
                .and_then(serde_json::Value::as_str),
            Some(hex_digest(&Sha256::digest(mask_value.abi_bytes()?).into()).as_str())
        );
        assert_eq!(
            attachment
                .get("content_sha256")
                .and_then(serde_json::Value::as_str),
            Some(hex_digest(&Sha256::digest(&bytes).into()).as_str())
        );
        let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(128)?;
        let limited_backend = Arc::new(limited_backend);
        let limited_scratch = limited_authority.authorize_workspace(128)?;
        let limited_compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x43643),
                attempt_id,
                NodeId::from("beeble-alpha-video"),
            )?,
            limited_backend.clone(),
            StreamId::DEFAULT,
            &limited_scratch,
        )?;
        let limited = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x43642)),
            attempt_id,
            NodeId::from("beeble-alpha-video"),
            cancellation.clone(),
            limited_scratch.clone(),
            store,
            NativeNodeServices::checked(None, None, Some(limited_compute))?
                .with_component_h264_mp4_backing(codec.clone())?,
        )?;
        let limited_execution = limited.compute_session()?.execution_context(&limited)?;
        let (limited_mask, _) = limited_backend.upload_f32(
            sim_tensor::TensorDescriptor::contiguous(
                vec![3, 2, 2],
                sim_tensor::DType::F32,
                sim_tensor::DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0.5; 12],
            &limited_execution,
        )?;
        let before_memory = limited_backend.memory_snapshot().current_bytes;
        let before = codec.observed.lock().map_err(|_| "poisoned codec")?.len();
        assert!(
            smol::block_on(limited.prepare_provider_mask_video(
                &limited_mask,
                (30000, 1001),
                3,
                2_770_000
            ))
            .is_err()
        );
        assert_eq!(
            codec.observed.lock().map_err(|_| "poisoned codec")?.len(),
            before
        );
        assert_eq!(limited_scratch.in_use_bytes(), 0);
        assert_eq!(
            limited_backend.memory_snapshot().current_bytes,
            before_memory
        );
        assert_eq!(generation.resident_bytes(), resident);
        for mode in [1, 2, 3] {
            codec
                .outcome
                .store(mode, std::sync::atomic::Ordering::SeqCst);
            let error = match smol::block_on(prepare_native_provider_invocation(
                &node,
                descriptor,
                values.clone(),
                "profile-a",
                &context,
            )) {
                Ok(_) => return Err("codec failure unexpectedly prepared an invocation".into()),
                Err(error) => error,
            };
            assert_eq!(
                error.kind,
                if mode == 1 {
                    NativeNodeFailureKind::Interrupted
                } else {
                    NativeNodeFailureKind::Failure
                }
            );
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(generation.len(), 2);
            assert_eq!(scratch.in_use_bytes(), 0);
        }
        codec.outcome.store(0, std::sync::atomic::Ordering::SeqCst);
        smol::block_on(prepare_native_provider_invocation(
            &node,
            descriptor,
            values.clone(),
            "profile-a",
            &context,
        ))?;
        assert_eq!(generation.resident_bytes(), resident);
        let before = codec.observed.lock().map_err(|_| "poisoned codec")?.len();
        assert!(cancellation.cancel());
        assert!(
            smol::block_on(prepare_native_provider_invocation(
                &node,
                descriptor,
                values,
                "profile-a",
                &context,
            ))
            .is_err()
        );
        assert_eq!(
            codec.observed.lock().map_err(|_| "poisoned codec")?.len(),
            before
        );
        Ok(())
    }
}

mod hitpaw_image_input {
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use sim_nodes::{
        NativeHandleKind, NativeHandleType, NativeNodeComputeSession, NativeNodeContext,
        NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativeStoredPayload,
        NativeValue,
    };
    use sim_plugin_host::{PreparedNativeProviderInvocation, prepare_native_provider_invocation};
    use sim_plugin_sdk::{
        ApiVersion, CachePolicy, DeterminismPolicy, EffectPolicy, PluginNode, PluginPort,
        PluginValue, PortCardinality, PortDirection, PortPresence, PortSerialization, TypeRegistry,
    };
    use sim_runtime::{NativeHandleStoreGeneration, ProviderTransportEncoding};
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, ImageTensor, NativeTensorPayload,
        NativeTensorRole, StreamId,
    };
    use sim_types::{AttemptId, NodeId, PromptId};
    use std::{collections::BTreeMap, error::Error, sync::Arc};
    use uuid::Uuid;

    fn node(port_id: &str) -> Result<PluginNode, Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        Ok(PluginNode {
            id: "HitPawGeneralImageEnhance".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "HitPaw input preparation".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![PluginPort {
                id: port_id.to_owned(),
                name: port_id.to_owned(),
                direction: PortDirection::Input,
                type_id: registry.resolve("IMAGE")?.clone(),
                cardinality: PortCardinality::Singular,
                presence: PortPresence::Required,
                hidden: false,
                lazy: false,
                default: None,
                serialization: PortSerialization::Handle,
                accepted_legacy_names: vec![],
            }],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        })
    }

    fn image_context(
        shape: &[u64],
        pixels: &[f32],
        capacity: u64,
        identity: u128,
    ) -> Result<
        (
            NativeNodeContext,
            NativeHandleStoreGeneration,
            NativeOpaqueHandle,
        ),
        Box<dyn Error>,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(capacity)?;
        let backend = Arc::new(backend);
        let workspace = authority.authorize_workspace(capacity)?;
        let attempt_id = AttemptId(Uuid::from_u128(identity));
        let node_id = NodeId::from("hitpaw-image-input");
        let generation =
            NativeHandleStoreGeneration::with_capacities(4, usize::try_from(capacity)?)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(identity + 1),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &workspace,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(identity + 2)),
            attempt_id,
            node_id,
            CancellationToken::default(),
            workspace,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let [batch, height, width, channels] = shape else {
            return Err("expected BHWC shape".into());
        };
        let execution = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(
            &backend, &execution, *batch, *height, *width, *channels, pixels,
        )?;
        let handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                image,
            )?)),
            &context.cancellation,
        )?;
        Ok((context, generation, handle))
    }

    fn prepare(
        node: &PluginNode,
        context: &NativeNodeContext,
        handle: NativeOpaqueHandle,
    ) -> Result<PreparedNativeProviderInvocation, sim_nodes::NativeNodeFailure> {
        let registry =
            sim_runtime::generated_native_node_registry_projection(None).map_err(|error| {
                sim_nodes::NativeNodeFailure {
                    code: "test_registry".to_owned(),
                    message: error.to_string(),
                    kind: sim_nodes::NativeNodeFailureKind::Failure,
                    retryable: false,
                }
            })?;
        let descriptor = registry
            .descriptor("HitPawGeneralImageEnhance")
            .ok_or_else(|| sim_nodes::NativeNodeFailure {
                code: "test_descriptor".to_owned(),
                message: "missing HitPaw descriptor".to_owned(),
                kind: sim_nodes::NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
        smol::block_on(prepare_native_provider_invocation(
            node,
            descriptor,
            BTreeMap::from([(
                node.ports
                    .first()
                    .ok_or_else(|| sim_nodes::NativeNodeFailure {
                        code: "test_port".to_owned(),
                        message: "missing port".to_owned(),
                        kind: sim_nodes::NativeNodeFailureKind::Failure,
                        retryable: false,
                    })?
                    .id
                    .clone(),
                NativeValue::Handle { value: handle },
            )]),
            "profile-a",
            context,
        ))
    }

    fn attachment(prepared: &PreparedNativeProviderInvocation) -> Result<Value, Box<dyn Error>> {
        let inputs = serde_json::to_value(&prepared.inputs)?;
        let batch = inputs
            .pointer("/media_batches/image/0")
            .and_then(Value::as_object)
            .ok_or("missing HitPaw PNG batch")?;
        assert_eq!(batch.len(), 1);
        Ok(batch.get("0").ok_or("missing first frame")?.clone())
    }

    #[test]
    fn hitpaw_nonresizing_input_uses_original_first_frame_pixels_and_digests()
    -> Result<(), Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(
            "../fixtures/providers/common/source_image_upload_linux_x86_64.json"
        ))?;
        let cases = oracle["cases"].as_array().ok_or("missing source cases")?;
        let node = node("image")?;
        let registry = TypeRegistry::built_in()?;
        let mut covered = 0;
        for (index, case) in cases.iter().enumerate() {
            if !case["maximum_pixels"].is_null() || case["batch_index"] != 0 {
                continue;
            }
            let shape: Vec<u64> = serde_json::from_value(case["shape"].clone())?;
            let pixels: Vec<f32> = serde_json::from_value(case["pixels"].clone())?;
            let (context, generation, handle) = image_context(
                &shape,
                &pixels,
                8 * 1024 * 1024,
                0x4430_0000 + u128::try_from(index)? * 4,
            )?;
            let resident = generation.resident_bytes();
            let result = prepare(&node, &context, handle.clone());
            if case.get("error").is_some() {
                assert!(result.is_err());
            } else {
                let prepared = result?;
                let media = attachment(&prepared)?;
                let bytes: Vec<u8> = serde_json::from_value(media["bytes"].clone())?;
                let decoded = sim_media::decode_downloaded_image_rgba(
                    &bytes,
                    sim_media::PngLimits::default(),
                    &context.cancellation,
                )?;
                assert_eq!(
                    u64::from(decoded.width),
                    case["output"]["width"].as_u64().ok_or("missing width")?
                );
                assert_eq!(
                    u64::from(decoded.height),
                    case["output"]["height"].as_u64().ok_or("missing height")?
                );
                let source_pixels: Vec<u8> =
                    serde_json::from_value(case["output"]["pixels"].clone())?;
                let channels = match case["output"]["mode"].as_str() {
                    Some("L") => 1,
                    Some("RGB") => 3,
                    Some("RGBA") => 4,
                    _ => return Err("unknown source mode".into()),
                };
                let expected = source_pixels
                    .chunks_exact(channels)
                    .flat_map(|pixel| {
                        let rgba = match channels {
                            1 => [pixel[0], pixel[0], pixel[0], 255],
                            3 => [pixel[0], pixel[1], pixel[2], 255],
                            _ => [pixel[0], pixel[1], pixel[2], pixel[3]],
                        };
                        rgba.map(|channel| f32::from(channel) / 255.0)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(decoded.pixels_bhwc, expected);
                assert_eq!(media["media_type"], "image/png");
                assert_eq!(
                    media["content_sha256"],
                    format!("{:x}", Sha256::digest(&bytes))
                );
                let values = prepared
                    .request
                    .ports()
                    .first()
                    .ok_or("missing request port")?
                    .values();
                assert_eq!(values.len(), 1);
                let value = values.first().ok_or("missing encoded value")?;
                assert_eq!(value.encoding(), ProviderTransportEncoding::PluginValue);
                assert_eq!(
                    media["source_value_sha256"],
                    format!("{:x}", Sha256::digest(value.abi_bytes()))
                );
                let plugin = PluginValue::from_abi_bytes(value.abi_bytes(), &registry)?;
                assert_eq!(plugin.type_id(), registry.resolve("IMAGE")?);
                assert!(prepared.request.to_bytes()?.len() < 4096);
                assert!(
                    context
                        .handle_store()
                        .resolve(&handle, handle.handle_type(), &context.cancellation)
                        .is_ok()
                );
                covered += 1;
            }
            assert_eq!(generation.len(), 1);
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        assert_eq!(covered, 9);
        Ok(())
    }

    #[test]
    fn hitpaw_second_frame_preserves_first_png_and_changes_tensor_identity()
    -> Result<(), Box<dyn Error>> {
        let node = node("image")?;
        let mut first_png = None;
        let mut source_digest = None;
        for (index, second) in [[1.0, 0.5, 0.0], [0.25, 0.75, 0.5]].into_iter().enumerate() {
            let (context, generation, handle) = image_context(
                &[2, 1, 1, 3],
                &[0.0, 0.5, 1.0, second[0], second[1], second[2]],
                64 * 1024,
                0x4431_0000 + u128::try_from(index)? * 4,
            )?;
            let prepared = prepare(&node, &context, handle)?;
            let media = attachment(&prepared)?;
            let png: Vec<u8> = serde_json::from_value(media["bytes"].clone())?;
            let decoded = sim_media::decode_downloaded_image_rgba(
                &png,
                sim_media::PngLimits::default(),
                &context.cancellation,
            )?;
            assert_eq!(decoded.pixels_bhwc, vec![0.0, 127.0 / 255.0, 1.0, 1.0]);
            if let Some(previous) = &first_png {
                assert_eq!(&png, previous);
            } else {
                first_png = Some(png);
            }
            if let Some(previous) = &source_digest {
                assert_ne!(&media["source_value_sha256"], previous);
            }
            source_digest = Some(media["source_value_sha256"].clone());
            assert_eq!(generation.len(), 1);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }

    #[test]
    fn hitpaw_input_rejects_foreign_stale_wrong_kind_and_cancelled_handles_without_leaks()
    -> Result<(), Box<dyn Error>> {
        let node = node("image")?;
        let (context, generation, handle) =
            image_context(&[1, 1, 1, 3], &[0.0, 0.5, 1.0], 64 * 1024, 0x4432_0000)?;
        let (foreign_context, foreign_generation, foreign) =
            image_context(&[1, 1, 1, 3], &[1.0, 0.5, 0.0], 64 * 1024, 0x4432_0010)?;
        let stale = NativeOpaqueHandle::new(
            handle.handle_type().clone(),
            handle.store_identity(),
            handle.identifier(),
            handle.generation() + 1,
            handle.digest_sha256().map(str::to_owned),
        )?;
        let wrong_kind = NativeOpaqueHandle::new(
            NativeHandleType::new(NativeHandleKind::Mask, "MASK")?,
            handle.store_identity(),
            handle.identifier(),
            handle.generation(),
            handle.digest_sha256().map(str::to_owned),
        )?;
        let resident = generation.resident_bytes();
        for rejected in [foreign, stale, wrong_kind] {
            assert!(prepare(&node, &context, rejected).is_err());
            assert_eq!(generation.len(), 1);
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        assert_eq!(foreign_generation.len(), 1);
        assert_eq!(foreign_context.scratch.in_use_bytes(), 0);
        context.cancellation.cancel();
        assert!(prepare(&node, &context, handle).is_err());
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn hitpaw_input_backend_allocation_failure_and_wrong_port_leave_no_media_or_handles()
    -> Result<(), Box<dyn Error>> {
        let (context, generation, handle) =
            image_context(&[1, 4, 4, 3], &[0.5; 48], 48 * 4, 0x4433_0000)?;
        let resident = generation.resident_bytes();
        let result = prepare(&node("image")?, &context, handle);
        let failure = result
            .err()
            .ok_or("PNG preparation succeeded despite exhausted actual backend")?;
        assert_eq!(failure.code, "provider_image_preparation_failed");
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        let (context, generation, handle) = image_context(
            &[2, 1, 1, 3],
            &[0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
            64 * 1024,
            0x4433_0010,
        )?;
        let prepared = prepare(&node("reference_image")?, &context, handle)?;
        let json = serde_json::to_value(&prepared.inputs)?;
        assert!(json.pointer("/media_batches/image").is_none());
        assert_eq!(
            prepared
                .request
                .ports()
                .first()
                .ok_or("missing request")?
                .values()
                .first()
                .ok_or("missing value")?
                .encoding(),
            ProviderTransportEncoding::NativePayload
        );
        assert_eq!(generation.len(), 1);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
