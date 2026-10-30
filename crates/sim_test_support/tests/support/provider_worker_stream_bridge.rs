use clock::SystemClock;
use extension_host::{ComponentLifecycleAdapter, ComponentRuntime, InstalledComponent};
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
use sim_plugin_host::{
    CancellationToken, ComponentExecutionBoundary, ComponentHost, ComponentHostError,
    ComponentHostProviderInvocationAuthority, ComponentHostRouter, ComponentLimits,
    InvocationInputs, PluginError, PrivateWorkerPluginExecutor,
    private_worker_provider_v2_actuator_route,
};
use sim_plugin_sdk::{
    ApiRequirement, ApiVersion, CachePolicy, CapabilityKind, CapabilityQuota, CapabilityRequest,
    DeterminismPolicy, ED25519_SIGNATURE_BYTES, EffectPolicy, InvocationError,
    MAX_PROVIDER_COST_REQUEST_BYTES, MAX_PROVIDER_COST_RESPONSE_BYTES, ManifestProvenance,
    ManifestSignature, PLUGIN_SIGNATURE_ALGORITHM, PROVIDER_BINDING_API_FEATURE,
    PROVIDER_COMPONENT_WORLD_V2, PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
    PROVIDER_STREAMING_API_FEATURE_V2, PluginManifest, PluginNode, PluginPort, PluginSigningKey,
    PluginValue, PortCardinality, PortDirection, PortPresence, PortSerialization,
    ProviderBindingClaim, ProviderBindingSet, ProviderHttpMethodV2, ProviderPluginManifestV2,
    ProviderStreamingContractV2, ScalarValue, TypeRegistry,
};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, Capability, CapabilitySet,
    CredentialPresenceActuator, CredentialScope, ExecutionCommandOutcome, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionController, ExecutionDataSource, ExecutionEventBus,
    ExecutionPresentationService, ExecutionSnapshotStatus, NativeExecutionController,
    NativeExecutionControllerConfig, NativeExecutionRegistryBundle, NativeHandleStoreGeneration,
    NativeProviderExecutionIdentity, NativeProviderWorkerV2ActuatorRoute, NodeContext,
    PermissionGrant, PermissionPolicy, PluginCapabilityBroker, PluginRngPolicy,
    PluginServiceActuatorError, PluginServiceOperationContext, PluginTrustPolicy,
    PluginVerificationKey, ProfileId, ProviderCostAcceptanceIssuer, ProviderCostApprovalAuthority,
    ProviderCostNonce, ProviderEndpoint, ProviderMode, ProviderPolicy, ProviderRequestActuator,
    ProviderResultReceiptIssuer, ProviderRuntimeReceiptIssuerV2, ProviderTransportResponse,
    RuntimeSupervisorError, SecretId, SecretValue, SharedAssetService, SupervisorPolicy,
    WorkerLaunchConfig,
};
use sim_tensor::{CpuWorkspaceAuthority, RngAlgorithm, RngProfileVersion, StreamId};
use sim_types::{
    ApiPrompt, AttemptId, NodeId, PromptId, PromptNode, PromptSubmission, RequestId, WorkerId,
    WorkerProviderInvocationContext, WorkerProviderResponseFrame, WorkerProviderResponseFrameEvent,
    WorkerProviderResponseHead, WorkerProviderStreamRequest, WorkerProviderTerminal,
    WorkerProviderWaitOutcome,
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;

pub(super) const PROFILE: &str = "00000000-0000-0000-0000-000000042500";
const KEY_ID: &str = "task425.publisher";
const KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
pub(super) const NODE_ID: &str = "OpenRouterLLMNode";
const FIXTURE_MODEL: &str = "deepseek/deepseek-v3.2";

pub(super) fn task425_entrypoint_count() -> usize {
    let entrypoints: [fn() -> Result<(), Box<dyn Error>>; 3] = [
        first_valid_grant_reaches_verified_materialization_and_finalization_ack,
        cancellation_between_response_chunks_revokes_without_publication_and_clean_retry_is_unique,
        worker_crash_revokes_pending_route_and_restarts_without_duplicate_publication,
    ];
    entrypoints.len()
}

#[derive(Default)]
struct UnreachableProvider;

impl ProviderRequestActuator for UnreachableProvider {
    fn execute(
        &self,
        _request: &sim_runtime::AuthorizedProviderRequest,
        _secret: Option<&SecretValue>,
        _body: &[u8],
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        Err(PluginServiceActuatorError::new(
            "legacy provider actuator is outside the provider-v2 test route",
        ))
    }
}

#[derive(Default)]
struct NoCredentials;

impl CredentialPresenceActuator for NoCredentials {
    fn is_present(
        &self,
        _request: &sim_runtime::AuthorizedCredentialPresenceRequest,
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<bool, PluginServiceActuatorError> {
        Ok(true)
    }

    fn read_for_provider(
        &self,
        _request: &sim_runtime::AuthorizedCredentialPresenceRequest,
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<Option<SecretValue>, PluginServiceActuatorError> {
        Ok(Some(SecretValue::new(b"fixture-secret-value".to_vec())))
    }
}

#[derive(Default)]
struct FixedClock;

impl SystemClock for FixedClock {
    fn utc_now(&self) -> Instant {
        Instant::now()
    }
}

pub(super) fn first_valid_grant_reaches_verified_materialization_and_finalization_ack()
-> Result<(), Box<dyn Error>> {
    smol::block_on(async {
        let profile_id = ProfileId(Uuid::parse_str(PROFILE)?);
        let base_registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let contract_sha256 = base_registry
            .provider_binding_contract_sha256(
                NODE_ID,
                "zed:sim-provider-transport@1",
                "zed:sim-provider-materializer@1",
            )?
            .ok_or("provider fixture contract is absent")?;
        let component = provider_component()?;
        let manifest =
            provider_manifest(format!("{:x}", Sha256::digest(&component)), contract_sha256)?;
        let trust = trust_policy()?;
        let permission_policy = permission_policy(&manifest.manifest)?;
        let provider_policy = provider_policy()?;
        let invocation_timeout = Duration::from_secs(30);
        let broker = provider_broker(provider_policy.clone())?;
        let receipt_issuer = Arc::new(ProviderResultReceiptIssuer::generate(Instant::now())?);
        let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
            Arc::new(ProviderCostAcceptanceIssuer::from_seed(
                [0x25; 32],
                Instant::now(),
            )?),
            Arc::new(FixedClock),
        ));
        let mut plugin_worker = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x4251)),
            "task425-plugin-worker-v1",
            2 * 1024 * 1024 * 1024,
        );
        plugin_worker.policy = worker_policy();
        let executor = PrivateWorkerPluginExecutor::new_with_provider_authorities(
            plugin_worker,
            broker.clone(),
            PROFILE,
            receipt_issuer.clone(),
            invocation_timeout,
            cost_authority.clone(),
        )?;
        let (actuator_attachment, actuator_receiver) = private_worker_provider_v2_actuator_route();
        executor.attach_provider_v2_actuator(actuator_attachment)?;
        let host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust,
            permission_policy,
            ComponentExecutionBoundary::private_worker(executor.clone()),
            provider_component_limits(),
            base_registry,
        )?;
        let router = ComponentHostRouter::new(host.clone());
        router
            .synchronize(vec![InstalledComponent::checked(
                Arc::from("task425-provider"),
                Arc::from("1.0.0"),
                serde_json::to_vec(&manifest)?.into(),
                component.into(),
            )?])
            .await
            .map_err(|error| format!("component synchronization failed: {error}"))?;
        let bundle = router.active_execution_registry_bundle()?;

        let controller_fixture = ControllerFixture::new(profile_id)?;
        let presentation = controller_fixture.presentation()?;
        let event_bus = ExecutionEventBus::new(64)?;
        let event_receiver = event_bus.subscribe();
        let provider_authority = Arc::new(ComponentHostProviderInvocationAuthority::new(
            host.clone(),
            broker,
            PROFILE,
            receipt_issuer,
            invocation_timeout,
        )?);
        let mut controller_worker = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_test_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x4252)),
            sim_runtime::NATIVE_IMAGE_REGISTRY_VERSION,
            2 * 1024 * 1024 * 1024,
        )
        .with_registry_deployment(bundle.worker_deployment().clone());
        controller_worker.policy = worker_policy();
        let controller_config = NativeExecutionControllerConfig::new(
            controller_fixture.assets.clone(),
            presentation.clone(),
            controller_worker,
            false,
        )?
        .with_provider_registry(
            bundle
                .provider_registry()
                .cloned()
                .ok_or("provider registry pin is absent")?,
        )?
        .with_provider_invocation_authority(provider_authority);
        let registration = NativeExecutionController::start_with_provider_worker_bridge(
            controller_config,
            event_bus,
        )?;
        let (controller, controller_bridge) = registration.into_parts();
        executor.attach_provider_worker_bridge(controller_bridge)?;

        let prompt_id = PromptId(Uuid::from_u128(0x4253));
        let plan = bundle.compile(PromptSubmission {
            prompt: ApiPrompt(BTreeMap::from([
                (
                    NodeId::from("provider"),
                    PromptNode {
                        class_type: NODE_ID.to_owned(),
                        inputs: BTreeMap::from([
                            ("prompt".to_owned(), json!("fixture prompt")),
                            ("model".to_owned(), json!(FIXTURE_MODEL)),
                            ("model.reasoning_effort".to_owned(), json!("off")),
                            ("seed".to_owned(), json!(425)),
                        ]),
                        unknown: BTreeMap::new(),
                    },
                ),
                (
                    NodeId::from("load"),
                    PromptNode {
                        class_type: "LoadImage".to_owned(),
                        inputs: BTreeMap::from([("image".to_owned(), json!("fixture.png"))]),
                        unknown: BTreeMap::new(),
                    },
                ),
                (
                    NodeId::from("save"),
                    PromptNode {
                        class_type: "SaveImage".to_owned(),
                        inputs: BTreeMap::from([
                            ("images".to_owned(), json!(["load", 0])),
                            ("filename_prefix".to_owned(), json!(["provider", 0])),
                        ]),
                        unknown: BTreeMap::new(),
                    },
                ),
            ])),
            prompt_id: Some(prompt_id),
            client_id: None,
            number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
        })?;
        let compiled_plan_sha256 = plan
            .provider_execution
            .as_ref()
            .ok_or("compiled provider identity is absent")?
            .compiled_plan_sha256()
            .to_owned();
        let acknowledgement = presentation
            .dispatch_durable(
                ExecutionControlCommand {
                    request_id: RequestId(Uuid::from_u128(0x4254)),
                    profile_id,
                    expected_revision: None,
                    kind: ExecutionControlCommandKind::Queue {
                        plan,
                        priority: 0,
                        front: false,
                    },
                },
                controller.as_ref(),
            )
            .await?;
        let attempt_id = match acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } => attempt_id,
            outcome => return Err(format!("provider attempt was not assigned: {outcome:?}").into()),
        };
        wait_until_started(&event_receiver, attempt_id).await?;

        let cancellation = CancellationToken::default();
        let generation = NativeHandleStoreGeneration::new()?;
        let context = provider_context(
            prompt_id,
            attempt_id,
            NodeId::from("provider"),
            cancellation.clone(),
            generation.handle_store_for_attempt(attempt_id),
            &compiled_plan_sha256,
        )?;
        let plugin = host.installed_plugin("task425-provider")?;
        let actuator_policy = provider_policy.clone();
        let actuator_cancellation = cancellation.clone();
        let actuator = smol::spawn(async move {
            let run = async move {
                let mut route = actuator_receiver.recv().await?;
                let receipt_origin = Instant::now();
                let receipt_issuer =
                    ProviderRuntimeReceiptIssuerV2::from_seed([0x42; 32], receipt_origin)?;
                let receipt_verifier = receipt_issuer.verifier()?;
                let cost_issuer =
                    ProviderCostAcceptanceIssuer::from_seed([0x43; 32], receipt_origin)?;
                let terminal_receipt = b"task425-terminal-receipt".to_vec();
                let mut next_response_sequence = 0_u64;
                let mut transcript = Vec::new();
                let mut call_ids = Vec::new();
                loop {
                    let call = route.receive_call(invocation_timeout).await?;
                    call_ids.push(call.call_id());
                    transcript.push(match call.request() {
                        WorkerProviderStreamRequest::StartRequest { context, .. } => {
                            if context.session_generation == 0
                                || context.invocation == 0
                                || context.generation == 0
                            {
                                anyhow::bail!("provider start context is not canonical");
                            }
                            "start"
                        }
                        WorkerProviderStreamRequest::WriteRequestChunk(_) => "request-body",
                        WorkerProviderStreamRequest::CheckCancelled(_) => "check-cancelled",
                        WorkerProviderStreamRequest::StartUpload(_) => "start-upload",
                        WorkerProviderStreamRequest::WriteUploadChunk(_) => "upload-body",
                        WorkerProviderStreamRequest::CompleteUpload(_) => "complete-upload",
                        WorkerProviderStreamRequest::RetryRequest(_) => "retry",
                        WorkerProviderStreamRequest::RequestCost(_) => "cost",
                        WorkerProviderStreamRequest::ReportProgress(_) => "progress",
                        WorkerProviderStreamRequest::WaitResponse(_) => "wait",
                    });
                    match call.request() {
                        WorkerProviderStreamRequest::StartRequest { .. } => {
                            route.respond_start(call, &actuator_policy)?;
                            assert!(matches!(
                                route.prepare_actuation(),
                                Err(sim_runtime::RuntimeSupervisorError::Protocol(message))
                                    if message
                                        == "provider streaming contract rejected the operation: provider stream event is out of order"
                            ));
                        }
                        WorkerProviderStreamRequest::CompleteUpload(_) => {
                            route.respond_upload_completion(
                                call,
                                "https://fixture.invalid/uploads/reference".to_owned(),
                            )?;
                        }
                        WorkerProviderStreamRequest::RequestCost(_) => {
                            route.respond_cost(
                                call,
                                &cost_issuer,
                                receipt_origin,
                                receipt_origin + Duration::from_secs(30),
                                ProviderCostNonce::new([0x44; 32])?,
                            )?;
                        }
                        WorkerProviderStreamRequest::WaitResponse(request) => {
                            if next_response_sequence == 0 {
                                assert!(matches!(
                                    route.finish(
                                        &receipt_issuer,
                                        &receipt_verifier,
                                        receipt_origin,
                                        receipt_origin + Duration::from_secs(30),
                                        [0x45; 32],
                                        receipt_origin + Duration::from_millis(1),
                                    ),
                                    Err(sim_runtime::RuntimeSupervisorError::Protocol(message))
                                        if message
                                            == "provider-v2 actuator finish preceded actuation preparation"
                                ));
                                route.prepare_actuation()?;
                                assert!(matches!(
                                    route.prepare_actuation(),
                                    Err(sim_runtime::RuntimeSupervisorError::Protocol(message))
                                        if message
                                            == "provider-v2 actuator actuation was already prepared"
                                ));
                            }
                            let handle = request.handle;
                            let event = if next_response_sequence == 0 {
                                WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                                    status: 200,
                                    headers: Vec::new(),
                                })
                            } else {
                                WorkerProviderResponseFrameEvent::Terminal(
                                    WorkerProviderTerminal::Completed(terminal_receipt.clone()),
                                )
                            };
                            let terminal = next_response_sequence == 1;
                            route.respond_call(
                                call,
                                Some(WorkerProviderWaitOutcome::Frame(
                                    WorkerProviderResponseFrame {
                                        handle,
                                        sequence: next_response_sequence,
                                        event,
                                    },
                                )),
                            )?;
                            next_response_sequence += 1;
                            if terminal {
                                route.finish(
                                    &receipt_issuer,
                                    &receipt_verifier,
                                    receipt_origin,
                                    receipt_origin + Duration::from_secs(30),
                                    [0x45; 32],
                                    receipt_origin + Duration::from_millis(1),
                                )?;
                                break;
                            }
                        }
                        _ => route.respond_call(call, None)?,
                    }
                }
                assert_eq!(
                    transcript,
                    [
                        "start",
                        "request-body",
                        "check-cancelled",
                        "start-upload",
                        "upload-body",
                        "complete-upload",
                        "cost",
                        "progress",
                        "wait",
                        "wait",
                    ]
                );
                assert_eq!(call_ids, (1_u64..=10).collect::<Vec<_>>());
                let proposal = route.receive_proposal(invocation_timeout).await?;
                match route.receive_call(Duration::from_millis(20)).await {
                    Err(sim_runtime::RuntimeSupervisorError::Timeout {
                        stage: "provider-v2 actuator call",
                    }) => {}
                    Err(error) => anyhow::bail!(
                        "unexpected pre-finalization call error: {error}; cancellation={}",
                        actuator_cancellation.is_cancelled()
                    ),
                    Ok(call) => anyhow::bail!(
                        "unexpected pre-finalization call {}; cancellation={}",
                        call.call_id(),
                        actuator_cancellation.is_cancelled()
                    ),
                }
                route.finalize_proposal(proposal, TypeRegistry::built_in()?)?;
                assert!(matches!(
                    route.receive_call(invocation_timeout).await,
                    Err(sim_runtime::RuntimeSupervisorError::Protocol(message))
                        if message == "provider-v2 actuator call route closed"
                ));
                assert!(!actuator_cancellation.is_cancelled());
                Ok::<_, anyhow::Error>(())
            };
            run.await
        });
        let response = host
            .execute_provider_v2_worker(&plugin, NODE_ID, provider_inputs()?, context)
            .await;
        actuator.await?;
        let response = response?;
        let encoded = response.to_bytes()?;
        assert!(!encoded.is_empty());

        cancellation.cancel();
        controller
            .accept(
                &ExecutionControlCommand {
                    request_id: RequestId(Uuid::from_u128(0x4255)),
                    profile_id,
                    expected_revision: None,
                    kind: ExecutionControlCommandKind::Cancel {
                        attempt_id,
                        reason: "task425 hermetic route completed".to_owned(),
                    },
                },
                None,
            )
            .map_err(|failure| format!("provider attempt cancellation failed: {failure:?}"))?;
        drop(controller);
        Ok::<(), Box<dyn Error>>(())
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProviderBridgeRunMode {
    Complete,
    CancelAfterResponseHead,
    CrashAfterFirstRequestChunk,
}

#[derive(Debug, Default)]
pub(super) struct ProviderBridgeRunEvidence {
    context: Option<WorkerProviderInvocationContext>,
    pub(super) call_ids: Vec<u64>,
    transcript: Vec<&'static str>,
    terminal_receipts: usize,
    proposals: usize,
    finalizations: usize,
    pub(super) materializations: usize,
    pub(super) outputs: usize,
    published_effects: usize,
}

struct ProviderBridgeRunOutcome {
    execution: Result<ProviderTransportResponse, ComponentHostError>,
    evidence: ProviderBridgeRunEvidence,
}

pub(super) struct ProviderBridgeHarness {
    pub(super) profile_id: ProfileId,
    pub(super) host: ComponentHost,
    pub(super) executor: Arc<PrivateWorkerPluginExecutor>,
    pub(super) bundle: NativeExecutionRegistryBundle,
    pub(super) broker: PluginCapabilityBroker,
    pub(super) receipt_issuer: Arc<ProviderResultReceiptIssuer>,
    pub(super) provider_policy: ProviderPolicy,
    pub(super) actuator_receiver: async_channel::Receiver<NativeProviderWorkerV2ActuatorRoute>,
    pub(super) invocation_timeout: Duration,
}

impl ProviderBridgeHarness {
    pub(super) async fn new(plugin_worker_arguments: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let profile_id = ProfileId(Uuid::parse_str(PROFILE)?);
        let base_registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let contract_sha256 = base_registry
            .provider_binding_contract_sha256(
                NODE_ID,
                "zed:sim-provider-transport@1",
                "zed:sim-provider-materializer@1",
            )?
            .ok_or("provider fixture contract is absent")?;
        let component = provider_component()?;
        let manifest =
            provider_manifest(format!("{:x}", Sha256::digest(&component)), contract_sha256)?;
        let provider_policy = provider_policy()?;
        let invocation_timeout = Duration::from_secs(30);
        let broker = provider_broker(provider_policy.clone())?;
        let receipt_issuer = Arc::new(ProviderResultReceiptIssuer::generate(Instant::now())?);
        let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
            Arc::new(ProviderCostAcceptanceIssuer::from_seed(
                [0x25; 32],
                Instant::now(),
            )?),
            Arc::new(FixedClock),
        ));
        let mut plugin_worker = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x4251)),
            "task425-plugin-worker-v1",
            2 * 1024 * 1024 * 1024,
        );
        plugin_worker.policy = worker_policy();
        plugin_worker.arguments = plugin_worker_arguments;
        let executor = PrivateWorkerPluginExecutor::new_with_provider_authorities(
            plugin_worker,
            broker.clone(),
            PROFILE,
            receipt_issuer.clone(),
            invocation_timeout,
            cost_authority,
        )?;
        let (actuator_attachment, actuator_receiver) = private_worker_provider_v2_actuator_route();
        executor.attach_provider_v2_actuator(actuator_attachment)?;
        let host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust_policy()?,
            permission_policy(&manifest.manifest)?,
            ComponentExecutionBoundary::private_worker(executor.clone()),
            provider_component_limits(),
            base_registry,
        )?;
        let router = ComponentHostRouter::new(host.clone());
        router
            .synchronize(vec![InstalledComponent::checked(
                Arc::from("task425-provider"),
                Arc::from("1.0.0"),
                serde_json::to_vec(&manifest)?.into(),
                component.into(),
            )?])
            .await
            .map_err(|error| format!("component synchronization failed: {error}"))?;
        let bundle = router.active_execution_registry_bundle()?;
        Ok(Self {
            profile_id,
            host,
            executor,
            bundle,
            broker,
            receipt_issuer,
            provider_policy,
            actuator_receiver,
            invocation_timeout,
        })
    }

    async fn run(
        &self,
        run_identity: u128,
        mode: ProviderBridgeRunMode,
    ) -> Result<ProviderBridgeRunOutcome, Box<dyn Error>> {
        let controller_fixture = ControllerFixture::new(self.profile_id)?;
        let presentation = controller_fixture.presentation()?;
        let event_bus = ExecutionEventBus::new(64)?;
        let event_receiver = event_bus.subscribe();
        let provider_authority = Arc::new(ComponentHostProviderInvocationAuthority::new(
            self.host.clone(),
            self.broker.clone(),
            PROFILE,
            self.receipt_issuer.clone(),
            self.invocation_timeout,
        )?);
        let mut controller_worker = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_test_worker_fixture"),
            self.profile_id,
            WorkerId(Uuid::from_u128(0x4252_0000 + run_identity)),
            sim_runtime::NATIVE_IMAGE_REGISTRY_VERSION,
            2 * 1024 * 1024 * 1024,
        )
        .with_registry_deployment(self.bundle.worker_deployment().clone());
        controller_worker.policy = worker_policy();
        let controller_config = NativeExecutionControllerConfig::new(
            controller_fixture.assets.clone(),
            presentation.clone(),
            controller_worker,
            false,
        )?
        .with_provider_registry(
            self.bundle
                .provider_registry()
                .cloned()
                .ok_or("provider registry pin is absent")?,
        )?
        .with_provider_invocation_authority(provider_authority);
        let registration = NativeExecutionController::start_with_provider_worker_bridge(
            controller_config,
            event_bus,
        )?;
        let (controller, controller_bridge) = registration.into_parts();
        self.executor
            .attach_provider_worker_bridge(controller_bridge)?;

        let prompt_id = PromptId(Uuid::from_u128(0x4253_0000 + run_identity));
        let plan = provider_plan(&self.bundle, prompt_id)?;
        let compiled_plan_sha256 = plan
            .provider_execution
            .as_ref()
            .ok_or("compiled provider identity is absent")?
            .compiled_plan_sha256()
            .to_owned();
        let acknowledgement = presentation
            .dispatch_durable(
                ExecutionControlCommand {
                    request_id: RequestId(Uuid::from_u128(0x4254_0000 + run_identity)),
                    profile_id: self.profile_id,
                    expected_revision: None,
                    kind: ExecutionControlCommandKind::Queue {
                        plan,
                        priority: 0,
                        front: false,
                    },
                },
                controller.as_ref(),
            )
            .await?;
        let attempt_id = match acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } => attempt_id,
            outcome => {
                return Err(format!("provider attempt was not assigned: {outcome:?}").into());
            }
        };
        wait_until_started(&event_receiver, attempt_id).await?;

        let cancellation = CancellationToken::default();
        let generation = NativeHandleStoreGeneration::new()?;
        let context = provider_context(
            prompt_id,
            attempt_id,
            NodeId::from("provider"),
            cancellation.clone(),
            generation.handle_store_for_attempt(attempt_id),
            &compiled_plan_sha256,
        )?;
        let plugin = self.host.installed_plugin("task425-provider")?;
        let actuator = smol::spawn(drive_provider_v2_actuator(
            self.actuator_receiver.clone(),
            self.provider_policy.clone(),
            self.invocation_timeout,
            cancellation.clone(),
            mode,
        ));
        let execution = self
            .host
            .execute_provider_v2_worker(&plugin, NODE_ID, provider_inputs()?, context)
            .await;
        let mut evidence = actuator.await.map_err(|error| {
            format!(
                "provider actuator failed: execution={execution:?}, snapshot={:?}, error={error}",
                presentation.snapshot(self.profile_id)
            )
        })?;
        if let Ok(materialization) = &execution {
            evidence.materializations += 1;
            evidence.outputs = materialization.ports().len();
        }

        controller
            .accept(
                &ExecutionControlCommand {
                    request_id: RequestId(Uuid::from_u128(0x4255_0000 + run_identity)),
                    profile_id: self.profile_id,
                    expected_revision: None,
                    kind: ExecutionControlCommandKind::Cancel {
                        attempt_id,
                        reason: "task425 provider bridge scenario completed".to_owned(),
                    },
                },
                None,
            )
            .map_err(|failure| format!("provider attempt cancellation failed: {failure:?}"))?;
        controller
            .shutdown()
            .map_err(|failure| format!("provider controller shutdown failed: {failure:?}"))?;
        Ok(ProviderBridgeRunOutcome {
            execution,
            evidence,
        })
    }
}

pub(super) fn provider_plan(
    bundle: &NativeExecutionRegistryBundle,
    prompt_id: PromptId,
) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    Ok(bundle.compile(PromptSubmission {
        prompt: ApiPrompt(BTreeMap::from([
            (
                NodeId::from("provider"),
                PromptNode {
                    class_type: NODE_ID.to_owned(),
                    inputs: BTreeMap::from([
                        ("prompt".to_owned(), json!("fixture prompt")),
                        ("model".to_owned(), json!(FIXTURE_MODEL)),
                        ("model.reasoning_effort".to_owned(), json!("off")),
                        ("seed".to_owned(), json!(425)),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                NodeId::from("load"),
                PromptNode {
                    class_type: "LoadImage".to_owned(),
                    inputs: BTreeMap::from([("image".to_owned(), json!("fixture.png"))]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                NodeId::from("save"),
                PromptNode {
                    class_type: "SaveImage".to_owned(),
                    inputs: BTreeMap::from([
                        ("images".to_owned(), json!(["load", 0])),
                        ("filename_prefix".to_owned(), json!(["provider", 0])),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
        ])),
        prompt_id: Some(prompt_id),
        client_id: None,
        number: None,
        extra_data: BTreeMap::from([("zed_native_delay_millis".to_owned(), json!(10_000))]),
        unknown: BTreeMap::new(),
    })?)
}

pub(super) async fn drive_provider_v2_actuator(
    actuator_receiver: async_channel::Receiver<NativeProviderWorkerV2ActuatorRoute>,
    provider_policy: ProviderPolicy,
    invocation_timeout: Duration,
    cancellation: CancellationToken,
    mode: ProviderBridgeRunMode,
) -> Result<ProviderBridgeRunEvidence, anyhow::Error> {
    let mut route = actuator_receiver.recv().await?;
    let receipt_origin = Instant::now();
    let receipt_issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x42; 32], receipt_origin)?;
    let receipt_verifier = receipt_issuer.verifier()?;
    let cost_issuer = ProviderCostAcceptanceIssuer::from_seed([0x43; 32], receipt_origin)?;
    let terminal_receipt = b"task425-terminal-receipt".to_vec();
    let mut next_response_sequence = 0_u64;
    let mut evidence = ProviderBridgeRunEvidence::default();
    loop {
        let expected_ordinal = evidence
            .call_ids
            .len()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("provider-v2 actuator call ordinal overflowed"))?;
        let call = route.receive_call(invocation_timeout).await.map_err(|error| {
            anyhow::anyhow!(
                "provider-v2 actuator call receive failed: mode={mode:?}, expected_ordinal={expected_ordinal}, transcript={:?}, cancellation={}, error={error}",
                evidence.transcript,
                cancellation.is_cancelled(),
            )
        })?;
        evidence.call_ids.push(call.call_id());
        evidence.transcript.push(match call.request() {
            WorkerProviderStreamRequest::StartRequest { context, .. } => {
                if context.session_generation == 0
                    || context.invocation == 0
                    || context.generation == 0
                {
                    anyhow::bail!("provider start context is not canonical");
                }
                evidence.context = Some(context.clone());
                "start"
            }
            WorkerProviderStreamRequest::WriteRequestChunk(_) => "request-body",
            WorkerProviderStreamRequest::CheckCancelled(_) => "check-cancelled",
            WorkerProviderStreamRequest::StartUpload(_) => "start-upload",
            WorkerProviderStreamRequest::WriteUploadChunk(_) => "upload-body",
            WorkerProviderStreamRequest::CompleteUpload(_) => "complete-upload",
            WorkerProviderStreamRequest::RetryRequest(_) => "retry",
            WorkerProviderStreamRequest::RequestCost(_) => "cost",
            WorkerProviderStreamRequest::ReportProgress(_) => "progress",
            WorkerProviderStreamRequest::WaitResponse(_) => "wait",
        });

        if mode == ProviderBridgeRunMode::CrashAfterFirstRequestChunk
            && matches!(
                call.request(),
                WorkerProviderStreamRequest::WriteRequestChunk(_)
            )
        {
            sim_runtime::supervisor_test_delay(Duration::from_millis(5_500)).await;
            let response = route.respond_call(call, None);
            if response.is_ok() {
                assert!(matches!(
                    route.receive_call(invocation_timeout).await,
                    Err(RuntimeSupervisorError::Protocol(message))
                        if message == "provider-v2 actuator call route closed"
                ));
            }
            assert!(matches!(
                route.receive_proposal(Duration::from_millis(20)).await,
                Err(RuntimeSupervisorError::Protocol(message))
                    if message == "provider-v2 actuator proposal route closed"
            ));
            return Ok(evidence);
        }

        match call.request() {
            WorkerProviderStreamRequest::StartRequest { .. } => {
                route.respond_start(call, &provider_policy)?;
            }
            WorkerProviderStreamRequest::CompleteUpload(_) => {
                route.respond_upload_completion(
                    call,
                    "https://fixture.invalid/uploads/reference".to_owned(),
                )?;
            }
            WorkerProviderStreamRequest::RequestCost(_) => {
                route.respond_cost(
                    call,
                    &cost_issuer,
                    receipt_origin,
                    receipt_origin + Duration::from_secs(30),
                    ProviderCostNonce::new([0x44; 32])?,
                )?;
            }
            WorkerProviderStreamRequest::WaitResponse(request) => {
                if next_response_sequence == 0 {
                    route.prepare_actuation()?;
                }
                let handle = request.handle;
                let event = if next_response_sequence == 0 {
                    WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    })
                } else {
                    WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Completed(
                        terminal_receipt.clone(),
                    ))
                };
                let terminal = next_response_sequence == 1;
                route.respond_call(
                    call,
                    Some(WorkerProviderWaitOutcome::Frame(
                        WorkerProviderResponseFrame {
                            handle,
                            sequence: next_response_sequence,
                            event,
                        },
                    )),
                )?;
                next_response_sequence += 1;
                if mode == ProviderBridgeRunMode::CancelAfterResponseHead {
                    cancellation.cancel();
                    assert!(matches!(
                        route.receive_call(invocation_timeout).await,
                        Err(RuntimeSupervisorError::Protocol(message))
                            if message == "provider-v2 actuator call route closed"
                    ));
                    assert!(matches!(
                        route.receive_proposal(Duration::from_millis(20)).await,
                        Err(RuntimeSupervisorError::Protocol(message))
                            if message == "provider-v2 actuator proposal route closed"
                    ));
                    return Ok(evidence);
                }
                if terminal {
                    evidence.terminal_receipts += 1;
                    route.finish(
                        &receipt_issuer,
                        &receipt_verifier,
                        receipt_origin,
                        receipt_origin + Duration::from_secs(30),
                        [0x45; 32],
                        receipt_origin + Duration::from_millis(1),
                    )?;
                    break;
                }
            }
            _ => route.respond_call(call, None)?,
        }
    }
    let proposal = route
        .receive_proposal(invocation_timeout)
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "provider proposal receive failed after mode={mode:?}, transcript={:?}, cancellation={}: {error}",
                evidence.transcript,
                cancellation.is_cancelled()
            )
        })?;
    evidence.proposals += 1;
    route.finalize_proposal(proposal, TypeRegistry::built_in()?)?;
    evidence.finalizations += 1;
    assert!(!cancellation.is_cancelled());
    Ok(evidence)
}

pub(super) fn cancellation_between_response_chunks_revokes_without_publication_and_clean_retry_is_unique()
-> Result<(), Box<dyn Error>> {
    smol::block_on(async {
        let harness = ProviderBridgeHarness::new(Vec::new()).await?;
        let cancelled = harness
            .run(0x10, ProviderBridgeRunMode::CancelAfterResponseHead)
            .await?;
        assert!(
            matches!(
                &cancelled.execution,
                Err(ComponentHostError::Plugin(PluginError::Invocation(
                    InvocationError::Cancelled
                )))
            ),
            "unexpected cancellation outcome: {:?}",
            cancelled.execution,
        );
        assert_eq!(cancelled.evidence.terminal_receipts, 0);
        assert_eq!(cancelled.evidence.proposals, 0);
        assert_eq!(cancelled.evidence.finalizations, 0);
        assert_eq!(cancelled.evidence.materializations, 0);
        assert_eq!(cancelled.evidence.outputs, 0);
        assert_eq!(cancelled.evidence.published_effects, 0);
        assert_eq!(cancelled.evidence.call_ids, (1_u64..=9).collect::<Vec<_>>());
        assert_eq!(
            cancelled.evidence.transcript,
            [
                "start",
                "request-body",
                "check-cancelled",
                "start-upload",
                "upload-body",
                "complete-upload",
                "cost",
                "progress",
                "wait",
            ]
        );

        let retry = harness.run(0x11, ProviderBridgeRunMode::Complete).await?;
        assert!(retry.execution.is_ok());
        assert_eq!(retry.evidence.call_ids, (1_u64..=10).collect::<Vec<_>>());
        assert_eq!(retry.evidence.terminal_receipts, 1);
        assert_eq!(retry.evidence.proposals, 1);
        assert_eq!(retry.evidence.finalizations, 1);
        assert_eq!(retry.evidence.materializations, 1);
        assert_eq!(retry.evidence.outputs, 1);
        assert_eq!(retry.evidence.published_effects, 0);
        assert_ne!(cancelled.evidence.context, retry.evidence.context);
        assert_eq!(cancelled.evidence.call_ids.first(), Some(&1));
        assert_eq!(
            retry
                .evidence
                .call_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            retry.evidence.call_ids.len()
        );
        Ok::<(), Box<dyn Error>>(())
    })
}

pub(super) fn worker_crash_revokes_pending_route_and_restarts_without_duplicate_publication()
-> Result<(), Box<dyn Error>> {
    smol::block_on(async {
        let marker_directory = tempfile::tempdir()?;
        let marker = marker_directory
            .path()
            .join("task425-worker-exit-once.marker");
        let harness = ProviderBridgeHarness::new(vec![
            "--exit-after-ms-once".to_owned(),
            "5000".to_owned(),
            "--exit-marker".to_owned(),
            marker.to_string_lossy().into_owned(),
        ])
        .await?;
        let crashed = harness
            .run(0x20, ProviderBridgeRunMode::CrashAfterFirstRequestChunk)
            .await?;
        assert!(marker.try_exists()?);
        assert!(matches!(
            crashed.execution,
            Err(ComponentHostError::ExecutionBoundary(message))
                if message.contains("worker") || message.contains("EOF") || message.contains("closed")
        ));
        assert_eq!(crashed.evidence.call_ids, [1, 2]);
        assert_eq!(crashed.evidence.terminal_receipts, 0);
        assert_eq!(crashed.evidence.proposals, 0);
        assert_eq!(crashed.evidence.finalizations, 0);
        assert_eq!(crashed.evidence.materializations, 0);
        assert_eq!(crashed.evidence.outputs, 0);
        assert_eq!(crashed.evidence.published_effects, 0);

        let retry = harness.run(0x21, ProviderBridgeRunMode::Complete).await?;
        assert!(retry.execution.is_ok());
        assert_ne!(crashed.evidence.context, retry.evidence.context);
        assert_eq!(retry.evidence.call_ids, (1_u64..=10).collect::<Vec<_>>());
        assert_eq!(retry.evidence.terminal_receipts, 1);
        assert_eq!(retry.evidence.proposals, 1);
        assert_eq!(retry.evidence.finalizations, 1);
        assert_eq!(retry.evidence.materializations, 1);
        assert_eq!(retry.evidence.outputs, 1);
        assert_eq!(retry.evidence.published_effects, 0);
        Ok::<(), Box<dyn Error>>(())
    })
}

pub(super) struct ControllerFixture {
    _directory: tempfile::TempDir,
    pub(super) assets: SharedAssetService,
}

impl ControllerFixture {
    pub(super) fn new(profile_id: ProfileId) -> Result<Self, Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let roots = AssetRoots::new(
            profile_id.0.to_string(),
            [
                AssetNamespace::Input,
                AssetNamespace::Output,
                AssetNamespace::Temporary,
                AssetNamespace::Model,
                AssetNamespace::Plugin,
            ]
            .into_iter()
            .map(|namespace| {
                let path = directory.path().join(namespace.locator_type());
                fs::create_dir_all(&path)?;
                Ok((namespace, path))
            })
            .collect::<Result<Vec<_>, std::io::Error>>()?,
        )?;
        let input = encode_png_frame(
            &[0.25, 0.5, 0.75],
            1,
            1,
            1,
            3,
            0,
            &BTreeMap::new(),
            PngLimits::default(),
        )?;
        fs::write(
            roots
                .test_root_path(AssetNamespace::Input)?
                .join("fixture.png"),
            input,
        )?;
        Ok(Self {
            assets: Arc::new(std::sync::Mutex::new(AssetService::open(roots)?)),
            _directory: directory,
        })
    }

    fn presentation(
        &self,
    ) -> Result<sim_runtime::SharedExecutionPresentationService, Box<dyn Error>> {
        let profile_id = ProfileId(Uuid::parse_str(
            &self
                .assets
                .lock()
                .map_err(|_| "asset lock is unavailable")?
                .roots()
                .profile_id,
        )?);
        let mut service = ExecutionPresentationService::new(64)?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        Ok(sim_runtime::ExecutionPresentationOwner::ephemeral(service))
    }
}

pub(super) async fn wait_until_started(
    receiver: &async_channel::Receiver<sim_runtime::AttemptEvent>,
    attempt_id: AttemptId,
) -> Result<(), Box<dyn Error>> {
    smol::future::race(
        async {
            loop {
                let event = receiver.recv().await?;
                if event.attempt_id == attempt_id
                    && matches!(event.kind, sim_runtime::AttemptEventKind::Started)
                {
                    return Ok::<(), Box<dyn Error>>(());
                }
            }
        },
        async {
            sim_runtime::supervisor_test_delay(Duration::from_secs(10)).await;
            Err("timed out waiting for provider attempt start".into())
        },
    )
    .await
}

pub(super) fn provider_context(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    execution_node_id: NodeId,
    cancellation: CancellationToken,
    store: Arc<dyn sim_runtime::NativeHandleStore>,
    compiled_plan_sha256: &str,
) -> Result<NodeContext, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let scratch = workspace_authority.authorize_workspace(64 * 1024 * 1024)?;
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x4256),
        attempt_id,
        execution_node_id.clone(),
    )?;
    let compute = NativeNodeComputeSession::checked(
        identity,
        Arc::new(backend),
        StreamId::DEFAULT,
        &scratch,
    )?;
    let services = NativeNodeServices::checked(None, None, Some(compute))?.with_provider_execution(
        NativeProviderExecutionIdentity::checked(compiled_plan_sha256)?,
    );
    Ok(NodeContext::new_with_services(
        prompt_id,
        attempt_id,
        execution_node_id,
        cancellation,
        scratch,
        store,
        services,
    )?)
}

pub(super) fn provider_inputs() -> Result<InvocationInputs, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut inputs = InvocationInputs::default();
    inputs.set_present(
        "prompt",
        vec![PluginValue::scalar(
            registry.resolve("STRING")?.clone(),
            ScalarValue::String("fixture prompt".to_owned()),
            &registry,
        )?],
    );
    inputs.set_present(
        "model",
        vec![PluginValue::scalar(
            registry.resolve("COMFY_DYNAMICCOMBO_V3")?.clone(),
            ScalarValue::String(FIXTURE_MODEL.to_owned()),
            &registry,
        )?],
    );
    inputs.set_present(
        "seed",
        vec![PluginValue::scalar(
            registry.resolve("INT")?.clone(),
            ScalarValue::Integer(425),
            &registry,
        )?],
    );
    Ok(inputs)
}

fn provider_broker(policy: ProviderPolicy) -> Result<PluginCapabilityBroker, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let assets = sim_runtime::open_native_profile_asset_service(PROFILE, directory.path(), &[])?;
    Ok(PluginCapabilityBroker::new(
        assets,
        sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
        policy,
        Arc::new(UnreachableProvider),
        Arc::new(NoCredentials),
        Arc::new(FixedClock),
        PluginRngPolicy::new(RngProfileVersion::V2, RngAlgorithm::Philox4x32_10, 425),
    ))
}

pub(super) fn provider_policy() -> Result<ProviderPolicy, Box<dyn Error>> {
    Ok(ProviderPolicy::new(
        PROFILE,
        ProviderMode::Enabled,
        [ProviderEndpoint::new(
            "zed.sim.provider.openrouter",
            "https://fixture.invalid/v2/stream",
        )?],
        [CredentialScope::new(
            PROFILE,
            "zed.sim.provider.openrouter",
            "zed.sim.provider.openrouter",
            SecretId::new("fixture-secret")?,
        )?],
    )?)
}

fn permission_policy(manifest: &PluginManifest) -> Result<PermissionPolicy, Box<dyn Error>> {
    let capabilities = manifest
        .capabilities
        .iter()
        .map(Capability::from_plugin_request)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PermissionPolicy::new(
        PROFILE,
        [PermissionGrant::new(
            PROFILE,
            manifest.identifier.clone(),
            CapabilitySet::new(capabilities),
            "task425 hermetic fixture",
        )?],
    )?)
}

fn trust_policy() -> Result<PluginTrustPolicy, Box<dyn Error>> {
    let signing_key = PluginSigningKey::new(KEY_ID, KEY)?;
    Ok(PluginTrustPolicy::new([PluginVerificationKey::new(
        KEY_ID,
        signing_key.verification_key_bytes()?,
    )?])?)
}

pub(super) fn provider_manifest(
    component_digest: String,
    contract_sha256: String,
) -> Result<ProviderPluginManifestV2, Box<dyn Error>> {
    provider_manifest_with_provenance_source(
        component_digest,
        contract_sha256,
        "fixture://task425-provider".to_owned(),
    )
}

pub(super) fn provider_manifest_with_provenance_source(
    component_digest: String,
    contract_sha256: String,
    provenance_source: String,
) -> Result<ProviderPluginManifestV2, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut bindings = ProviderBindingSet {
        schema_version: 1,
        implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
        bindings_sha256: "0".repeat(64),
        bindings: vec![ProviderBindingClaim {
            feature_id: "COMFY-NODE-0466".to_owned(),
            node_id: NODE_ID.to_owned(),
            contract_sha256,
            transport_schema: "zed:sim-provider-transport@1".parse()?,
            materializer_schema: "zed:sim-provider-materializer@1".parse()?,
        }],
    };
    bindings.bindings_sha256 = bindings.canonical_bindings_sha256()?;
    let port = |id: &str, direction| -> Result<PluginPort, Box<dyn Error>> {
        Ok(PluginPort {
            id: id.to_owned(),
            name: id.to_owned(),
            direction,
            type_id: registry.resolve("STRING")?.clone(),
            cardinality: PortCardinality::Singular,
            presence: PortPresence::Required,
            hidden: false,
            lazy: false,
            default: None,
            serialization: PortSerialization::Inline,
            accepted_legacy_names: Vec::new(),
        })
    };
    let signing_key = PluginSigningKey::new(KEY_ID, KEY)?;
    let mut manifest = PluginManifest {
        schema_version: 1,
        identifier: "zed.sim.provider.openrouter".to_owned(),
        plugin_version: ApiVersion::new(1, 0, 0),
        api: ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                PROVIDER_BINDING_API_FEATURE.to_owned(),
                PROVIDER_STREAMING_API_FEATURE_V2.to_owned(),
            ],
        },
        digest_sha256: component_digest,
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
        provenance: ManifestProvenance {
            source: provenance_source,
            publisher: "Task425".to_owned(),
            registry: Some("fixture://task425-registry".to_owned()),
        },
        provider_binding: Some(bindings),
        nodes: vec![PluginNode {
            id: NODE_ID.to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "OpenRouter LLM".to_owned(),
            category: "partner/text/OpenRouter".to_owned(),
            ports: vec![
                port("prompt", PortDirection::Input)?,
                PluginPort {
                    id: "model".to_owned(),
                    name: "model".to_owned(),
                    direction: PortDirection::Input,
                    type_id: registry.resolve("COMFY_DYNAMICCOMBO_V3")?.clone(),
                    cardinality: PortCardinality::Singular,
                    presence: PortPresence::Required,
                    hidden: false,
                    lazy: false,
                    default: None,
                    serialization: PortSerialization::Inline,
                    accepted_legacy_names: Vec::new(),
                },
                PluginPort {
                    id: "seed".to_owned(),
                    name: "seed".to_owned(),
                    direction: PortDirection::Input,
                    type_id: registry.resolve("INT")?.clone(),
                    cardinality: PortCardinality::Singular,
                    presence: PortPresence::Required,
                    hidden: false,
                    lazy: false,
                    default: None,
                    serialization: PortSerialization::Inline,
                    accepted_legacy_names: Vec::new(),
                },
                PluginPort {
                    id: "system_prompt".to_owned(),
                    name: "system_prompt".to_owned(),
                    direction: PortDirection::Input,
                    type_id: registry.resolve("STRING")?.clone(),
                    cardinality: PortCardinality::Singular,
                    presence: PortPresence::Optional,
                    hidden: false,
                    lazy: false,
                    default: None,
                    serialization: PortSerialization::Inline,
                    accepted_legacy_names: Vec::new(),
                },
                port("output_0", PortDirection::Output)?,
            ],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        }],
        capabilities: vec![
            CapabilityRequest {
                kind: CapabilityKind::NetworkProvider,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 16_384,
                    maximum_response_bytes: 65_536,
                    maximum_total_bytes: 81_920,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::Secret,
                scope: "fixture-secret".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 14,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 15,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::ProviderUpload,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 4_096,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 4_096,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::ProviderCost,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: MAX_PROVIDER_COST_REQUEST_BYTES,
                    maximum_response_bytes: MAX_PROVIDER_COST_RESPONSE_BYTES,
                    maximum_total_bytes: MAX_PROVIDER_COST_REQUEST_BYTES
                        + MAX_PROVIDER_COST_RESPONSE_BYTES,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
        ],
        ui: Vec::new(),
        routes: Vec::new(),
        legacy_mappings: Vec::new(),
    };
    manifest.signature.value = signing_key.sign_manifest(&manifest)?;
    let mut provider = ProviderPluginManifestV2 {
        schema_version: PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
        component_world: PROVIDER_COMPONENT_WORLD_V2.to_owned(),
        manifest,
        streaming: ProviderStreamingContractV2 {
            methods: vec![
                ProviderHttpMethodV2::Delete,
                ProviderHttpMethodV2::Get,
                ProviderHttpMethodV2::Head,
                ProviderHttpMethodV2::Options,
                ProviderHttpMethodV2::Patch,
                ProviderHttpMethodV2::Post,
                ProviderHttpMethodV2::Put,
            ],
            maximum_headers: 8,
            maximum_header_bytes: 4_096,
            maximum_request_body_bytes: 16_384,
            maximum_response_body_bytes: 65_536,
            maximum_chunk_bytes: 4_096,
            maximum_ndjson_line_bytes: 4_096,
            maximum_wait_milliseconds: 1_000,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 4_096,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        },
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
    };
    provider.signature.value = signing_key.sign_provider_manifest_v2(&provider)?;
    Ok(provider)
}

pub(super) fn provider_component() -> Result<Vec<u8>, Box<dyn Error>> {
    decode_base64(
        include_str!("../../../sim_plugin_host/tests/fixtures/provider_streaming_component").trim(),
    )
}

fn decode_base64(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let encoded = value
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if encoded.is_empty() || encoded.len() % 4 != 0 {
        return Err("provider component base64 is invalid".into());
    }
    let mut decoded = Vec::with_capacity(encoded.len() / 4 * 3);
    for chunk in encoded.chunks_exact(4) {
        let mut values = [0_u8; 4];
        let mut padding = 0;
        for (index, byte) in chunk.iter().copied().enumerate() {
            if byte == b'=' {
                padding += 1;
                values[index] = 0;
            } else {
                values[index] = u8::try_from(
                    ALPHABET
                        .iter()
                        .position(|candidate| *candidate == byte)
                        .ok_or("provider component base64 contains an invalid byte")?,
                )?;
            }
        }
        let packed = (u32::from(values[0]) << 18)
            | (u32::from(values[1]) << 12)
            | (u32::from(values[2]) << 6)
            | u32::from(values[3]);
        decoded.push((packed >> 16) as u8);
        if padding < 2 {
            decoded.push((packed >> 8) as u8);
        }
        if padding == 0 {
            decoded.push(packed as u8);
        }
    }
    Ok(decoded)
}

pub(super) fn worker_policy() -> SupervisorPolicy {
    SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(10),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    }
}

fn provider_component_limits() -> ComponentLimits {
    ComponentLimits {
        epoch_deadline_ticks: 300,
        ..ComponentLimits::default()
    }
}
