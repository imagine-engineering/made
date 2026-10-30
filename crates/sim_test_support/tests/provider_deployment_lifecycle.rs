#[path = "support/provider_worker_stream_bridge.rs"]
mod task425_shared;

mod task425_fixture {
    use super::task425_shared::*;
    use extension_host::ComponentLifecycleAdapter;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use sim_api::NativeApiHostError;
    use sim_plugin_host::{
        CancellationToken, ComponentHostProviderInvocationAuthority, ComponentHostRouter,
    };
    use sim_runtime::{
        ExecutionCommandOutcome, ExecutionControlCommand, ExecutionControlCommandKind,
        ExecutionController, ExecutionEventBus, NativeExecutionController,
        NativeExecutionControllerConfig, NativeExecutionRegistryBundle,
        NativeHandleStoreGeneration, PermissionPolicy, ProfileId, WorkerLaunchConfig,
    };
    use sim_types::{NodeId, PromptId, RequestId, WorkerId};
    use std::{error::Error, sync::Arc};
    use uuid::Uuid;

    pub(super) fn profile_id() -> Result<ProfileId, Box<dyn Error>> {
        Ok(ProfileId(Uuid::parse_str(PROFILE)?))
    }

    pub(super) async fn canonical_candidate(
        cx: &mut gpui::TestAppContext,
        candidate_revision: &str,
    ) -> Result<extension_host::ComponentInventoryCandidate, Box<dyn Error>> {
        let base_registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let contract_sha256 = base_registry
            .provider_binding_contract_sha256(
                NODE_ID,
                "zed:sim-provider-transport@1",
                "zed:sim-provider-materializer@1",
            )?
            .ok_or("provider fixture contract is absent")?;
        let component = provider_component()?;
        let manifest = provider_manifest_with_provenance_source(
            format!("{:x}", Sha256::digest(&component)),
            contract_sha256,
            format!("fixture://task433-provider/{candidate_revision}"),
        )?;
        let filesystem = ::fs::FakeFs::new(cx.executor());
        let extensions_dir = std::path::Path::new("/task433-provider-inventory");
        filesystem
            .insert_tree(extensions_dir.join("installed/task425-provider"), json!({}))
            .await;
        filesystem
            .insert_file(
                extensions_dir.join("installed/task425-provider/sim-plugin.json"),
                serde_json::to_vec(&manifest)?,
            )
            .await;
        filesystem
            .insert_file(
                extensions_dir.join("installed/task425-provider/sim-plugin.wasm"),
                component,
            )
            .await;
        filesystem
            .insert_file(
                extensions_dir.join("index.json"),
                serde_json::to_vec(&json!({
                    "extensions": {
                        "task425-provider": {
                            "manifest": {
                                "id": "task425-provider",
                                "name": "task425-provider",
                                "version": "1.0.0",
                                "schema_version": 0
                            },
                            "dev": false
                        }
                    },
                    "themes": {},
                    "icon_themes": {},
                    "languages": {}
                }))?,
            )
            .await;
        Ok(
            extension_host::ExtensionStore::canonical_component_inventory_candidate(
                filesystem,
                extensions_dir,
            )
            .await?,
        )
    }

    pub(super) struct SignedProviderHarness(Arc<ProviderBridgeHarness>);

    pub(super) async fn provider_harness(
        candidate: &extension_host::ComponentInventoryCandidate,
        generation: u64,
    ) -> Result<Arc<SignedProviderHarness>, Box<dyn Error>> {
        let mut harness = ProviderBridgeHarness::new(Vec::new()).await?;
        let router =
            ComponentHostRouter::with_initial_generation(harness.host.clone(), generation)?;
        router
            .synchronize(candidate.components().to_vec())
            .await
            .map_err(|error| format!("component synchronization failed: {error}"))?;
        harness.bundle = router.active_execution_registry_bundle()?;
        Ok(Arc::new(SignedProviderHarness(Arc::new(harness))))
    }

    pub(super) fn bundle(harness: &SignedProviderHarness) -> Arc<NativeExecutionRegistryBundle> {
        Arc::new(harness.0.bundle.clone())
    }

    pub(super) struct LiveProviderRuntime {
        harness: Arc<ProviderBridgeHarness>,
        _controller_fixture: ControllerFixture,
        controller: Arc<dyn ExecutionController>,
        presentation: sim_runtime::SharedExecutionPresentationService,
        event_receiver: async_channel::Receiver<sim_runtime::AttemptEvent>,
    }

    pub(super) struct LiveProviderExecutionEvidence {
        pub(super) succeeded: bool,
        pub(super) call_ids: Vec<u64>,
        pub(super) materializations: usize,
    }

    impl LiveProviderRuntime {
        pub(super) async fn execute(
            &self,
            run_identity: u128,
        ) -> Result<LiveProviderExecutionEvidence, Box<dyn Error>> {
            let prompt_id = PromptId(Uuid::from_u128(0x4333_0000 + run_identity));
            let plan = provider_plan(&self.harness.bundle, prompt_id)?;
            let compiled_plan_sha256 = plan
                .provider_execution
                .as_ref()
                .ok_or("compiled provider identity is absent")?
                .compiled_plan_sha256()
                .to_owned();
            let acknowledgement = self
                .presentation
                .dispatch_durable(
                    ExecutionControlCommand {
                        request_id: RequestId(Uuid::from_u128(0x4334_0000 + run_identity)),
                        profile_id: self.harness.profile_id,
                        expected_revision: None,
                        kind: ExecutionControlCommandKind::Queue {
                            plan,
                            priority: 0,
                            front: false,
                        },
                    },
                    self.controller.as_ref(),
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
            wait_until_started(&self.event_receiver, attempt_id).await?;

            let cancellation = CancellationToken::default();
            let actuator_cancellation = cancellation.clone();
            let generation = NativeHandleStoreGeneration::new()?;
            let context = provider_context(
                prompt_id,
                attempt_id,
                NodeId::from("provider"),
                cancellation,
                generation.handle_store_for_attempt(attempt_id),
                &compiled_plan_sha256,
            )?;
            let plugin = self.harness.host.installed_plugin("task425-provider")?;
            let actuator = smol::spawn(drive_provider_v2_actuator(
                self.harness.actuator_receiver.clone(),
                self.harness.provider_policy.clone(),
                self.harness.invocation_timeout,
                actuator_cancellation,
                ProviderBridgeRunMode::Complete,
            ));
            let execution = self
                .harness
                .host
                .execute_provider_v2_worker(&plugin, NODE_ID, provider_inputs()?, context)
                .await;
            let mut evidence = actuator.await?;
            if let Ok(materialization) = &execution {
                evidence.materializations += 1;
                evidence.outputs = materialization.ports().len();
            }
            self.controller
                .accept(
                    &ExecutionControlCommand {
                        request_id: RequestId(Uuid::from_u128(0x4335_0000 + run_identity)),
                        profile_id: self.harness.profile_id,
                        expected_revision: None,
                        kind: ExecutionControlCommandKind::Cancel {
                            attempt_id,
                            reason: "task433 provider deployment scenario completed".to_owned(),
                        },
                    },
                    None,
                )
                .map_err(|failure| format!("provider attempt cancellation failed: {failure:?}"))?;
            Ok(LiveProviderExecutionEvidence {
                succeeded: execution.is_ok(),
                call_ids: evidence.call_ids,
                materializations: evidence.materializations,
            })
        }
    }

    pub(super) fn activate_runtime(
        harness: Arc<SignedProviderHarness>,
        presentation: sim_runtime::SharedExecutionPresentationService,
        bundle: Arc<NativeExecutionRegistryBundle>,
        candidate_identity: extension_host::ComponentInventoryCandidateIdentity,
    ) -> Result<(sim_api::NativeRuntimeApiHost, Arc<LiveProviderRuntime>), NativeApiHostError> {
        let harness = harness.0.clone();
        if bundle.identity_sha256() != harness.bundle.identity_sha256() {
            return Err(NativeApiHostError::InvalidConfiguration(
                "scripted provider harness received a foreign registry bundle".into(),
            ));
        }
        let controller_fixture = ControllerFixture::new(harness.profile_id)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let event_bus = ExecutionEventBus::new(64)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let event_receiver = event_bus.subscribe();
        let provider_authority = Arc::new(
            ComponentHostProviderInvocationAuthority::new(
                harness.host.clone(),
                harness.broker.clone(),
                PROFILE,
                harness.receipt_issuer.clone(),
                harness.invocation_timeout,
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
        );
        let mut worker = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_test_worker_fixture"),
            harness.profile_id,
            WorkerId(Uuid::new_v4()),
            sim_runtime::NATIVE_IMAGE_REGISTRY_VERSION,
            2 * 1024 * 1024 * 1024,
        )
        .with_registry_deployment(bundle.worker_deployment().clone());
        worker.policy = worker_policy();
        let controller_config =
            NativeExecutionControllerConfig::new(
                controller_fixture.assets.clone(),
                presentation.clone(),
                worker,
                false,
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?
            .with_provider_registry(bundle.provider_registry().cloned().ok_or_else(|| {
                NativeApiHostError::Runtime("provider registry pin is absent".into())
            })?)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?
            .with_provider_invocation_authority(provider_authority);
        let registration = NativeExecutionController::start_with_provider_worker_bridge(
            controller_config,
            event_bus.clone(),
        )
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let (controller, controller_bridge) = registration.into_parts();
        harness
            .executor
            .attach_provider_worker_bridge(controller_bridge)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let controller: Arc<dyn ExecutionController> = controller;
        let runtime = sim_api::NativeRuntimeApiHost::with_registry_bundle(
            bundle,
            &candidate_identity,
            presentation.clone(),
            controller.clone(),
            &event_bus,
            Some(controller_fixture.assets.clone()),
            sim_api::HttpLimits::default(),
            sim_api::WebSocketLimits::default(),
            sim_api::security::ApiSecurityConfig::loopback(),
            Arc::new(
                PermissionPolicy::native_runtime_services(PROFILE)
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
            ),
            Arc::new(super::MemoryIdempotencyStore::default()),
        )?;
        let driver = Arc::new(LiveProviderRuntime {
            harness,
            _controller_fixture: controller_fixture,
            controller,
            presentation,
            event_receiver,
        });
        Ok((runtime, driver))
    }
}

use sim_api::{
    NativeApiHostError, NativeHeadlessError, NativeHeadlessPolicy, NativeHeadlessService,
    NativeHeadlessState, NativeProviderDeploymentIdentity, PreparedNativeHeadlessRuntime,
    security::{ApiSecurityError, IdempotencySnapshot, IdempotencySnapshotStore},
};
use sim_runtime::{
    ExecutionDataSource, ExecutionPresentationOwner, ExecutionPresentationService,
    ExecutionSnapshotStatus,
};
use std::{
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Default)]
struct MemoryIdempotencyStore {
    snapshot: Mutex<Option<IdempotencySnapshot>>,
}

impl IdempotencySnapshotStore for MemoryIdempotencyStore {
    fn load(&self) -> Result<Option<IdempotencySnapshot>, ApiSecurityError> {
        self.snapshot
            .lock()
            .map(|snapshot| snapshot.clone())
            .map_err(|_| ApiSecurityError::SecurityStateUnavailable)
    }

    fn save(&self, snapshot: &IdempotencySnapshot) -> Result<(), ApiSecurityError> {
        self.snapshot
            .lock()
            .map_err(|_| ApiSecurityError::SecurityStateUnavailable)?
            .replace(snapshot.clone());
        Ok(())
    }
}

fn presentation() -> Result<sim_runtime::SharedExecutionPresentationService, Box<dyn Error>> {
    let profile = task425_fixture::profile_id()?;
    let mut presentation = ExecutionPresentationService::new(128)?;
    presentation.initialize_profile(
        profile,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    Ok(ExecutionPresentationOwner::ephemeral(presentation))
}

fn prepared_live_runtime(
    harness: Arc<task425_fixture::SignedProviderHarness>,
    bundle: Arc<sim_runtime::NativeExecutionRegistryBundle>,
    candidate_identity: extension_host::ComponentInventoryCandidateIdentity,
    driver: Arc<Mutex<Option<Arc<task425_fixture::LiveProviderRuntime>>>>,
    fail_after_quiescence: bool,
) -> Result<PreparedNativeHeadlessRuntime, NativeApiHostError> {
    PreparedNativeHeadlessRuntime::checked(
        bundle,
        candidate_identity,
        move |presentation, bundle, candidate_identity| {
            *driver
                .lock()
                .map_err(|_| NativeApiHostError::StateUnavailable)? = None;
            if fail_after_quiescence {
                return Err(NativeApiHostError::Runtime(
                    "scripted replacement activation failed after quiescence".into(),
                ));
            }
            let (runtime, active_driver) = task425_fixture::activate_runtime(
                harness,
                presentation,
                bundle,
                candidate_identity,
            )?;
            driver
                .lock()
                .map_err(|_| NativeApiHostError::StateUnavailable)?
                .replace(active_driver);
            Ok(runtime)
        },
    )
}

#[gpui::test]
async fn provider_deployment_lifecycle_preserves_cross_mode_generation_and_recovery(
    cx: &mut gpui::TestAppContext,
) {
    cx.executor().allow_parking();
    provider_deployment_lifecycle_result(cx)
        .await
        .expect("provider deployment lifecycle must preserve generation and recovery");
}

async fn provider_deployment_lifecycle_result(
    cx: &mut gpui::TestAppContext,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(task425_shared::task425_entrypoint_count(), 3);
    let candidate = task425_fixture::canonical_candidate(cx, "1.0.0").await?;
    let replacement_candidate = task425_fixture::canonical_candidate(cx, "1.0.1").await?;
    let retry_candidate = task425_fixture::canonical_candidate(cx, "1.0.2").await?;
    assert_ne!(candidate.identity(), replacement_candidate.identity());
    assert_ne!(replacement_candidate.identity(), retry_candidate.identity());
    let desktop_harness = task425_fixture::provider_harness(&candidate, 433).await?;
    let headless_harness = task425_fixture::provider_harness(&candidate, 433).await?;
    let replacement_harness =
        task425_fixture::provider_harness(&replacement_candidate, 434).await?;
    let retry_harness = task425_fixture::provider_harness(&retry_candidate, 435).await?;
    let desktop_bundle = task425_fixture::bundle(&desktop_harness);
    let headless_bundle = task425_fixture::bundle(&headless_harness);
    let replacement_bundle = task425_fixture::bundle(&replacement_harness);
    let retry_bundle = task425_fixture::bundle(&retry_harness);
    let desktop = NativeProviderDeploymentIdentity::from_registry_bundle(
        &desktop_bundle,
        candidate.identity(),
    )?;
    let headless = NativeProviderDeploymentIdentity::from_registry_bundle(
        &headless_bundle,
        candidate.identity(),
    )?;
    assert!(desktop.same_signed_deployment(&headless));
    assert_eq!(desktop.profile_id(), headless.profile_id());
    assert_eq!(
        desktop.component_inventory_candidate_sha256(),
        candidate.identity_sha256()
    );
    assert_eq!(
        desktop.component_generation(),
        headless.component_generation()
    );
    assert_eq!(
        desktop.signed_component_snapshot_sha256(),
        headless.signed_component_snapshot_sha256()
    );
    assert_eq!(
        desktop.provider_binding_set_sha256(),
        headless.provider_binding_set_sha256()
    );
    assert_ne!(
        desktop.registry_digest_sha256(),
        headless.registry_digest_sha256()
    );
    assert_ne!(
        desktop.execution_bundle_identity_sha256(),
        headless.execution_bundle_identity_sha256()
    );
    assert!(
        PreparedNativeHeadlessRuntime::checked(
            headless_bundle.clone(),
            replacement_candidate.identity().clone(),
            |_presentation, _bundle, _identity| Err(NativeApiHostError::StateUnavailable),
        )
        .is_err()
    );

    let foreign_driver = Arc::new(Mutex::new(None));
    let foreign = PreparedNativeHeadlessRuntime::checked(
        headless_bundle.clone(),
        candidate.identity().clone(),
        {
            let desktop_harness = desktop_harness.clone();
            let desktop_bundle = desktop_bundle.clone();
            move |presentation, _certified_bundle, candidate_identity| {
                let (runtime, driver) = task425_fixture::activate_runtime(
                    desktop_harness,
                    presentation,
                    desktop_bundle,
                    candidate_identity,
                )?;
                foreign_driver
                    .lock()
                    .map_err(|_| NativeApiHostError::StateUnavailable)?
                    .replace(driver);
                Ok(runtime)
            }
        },
    )?;
    assert!(matches!(
        foreign.activate(presentation()?),
        Err(NativeApiHostError::InvalidConfiguration(_))
    ));
    let mode = Arc::new(AtomicUsize::new(0));
    let preparations = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(Mutex::new(None));
    let service = NativeHeadlessService::offline_prepared(
        presentation()?,
        {
            let mode = mode.clone();
            let preparations = preparations.clone();
            let driver = driver.clone();
            let headless_harness = headless_harness.clone();
            let headless_bundle = headless_bundle.clone();
            let replacement_harness = replacement_harness.clone();
            let replacement_bundle = replacement_bundle.clone();
            let retry_harness = retry_harness.clone();
            let retry_bundle = retry_bundle.clone();
            let candidate_identity = candidate.identity().clone();
            let replacement_identity = replacement_candidate.identity().clone();
            let retry_identity = retry_candidate.identity().clone();
            move |_presentation| {
                preparations.fetch_add(1, Ordering::AcqRel);
                let (harness, bundle, candidate_identity, fail_after_quiescence) =
                    match mode.load(Ordering::Acquire) {
                        1 => {
                            return Err(NativeApiHostError::Runtime(
                                "signed provider candidate rejected before quiescence".into(),
                            ));
                        }
                        2 => (
                            replacement_harness.clone(),
                            replacement_bundle.clone(),
                            replacement_identity.clone(),
                            false,
                        ),
                        3 => (
                            retry_harness.clone(),
                            retry_bundle.clone(),
                            retry_identity.clone(),
                            true,
                        ),
                        4 => (
                            retry_harness.clone(),
                            retry_bundle.clone(),
                            retry_identity.clone(),
                            false,
                        ),
                        _ => (
                            headless_harness.clone(),
                            headless_bundle.clone(),
                            candidate_identity.clone(),
                            false,
                        ),
                    };
                prepared_live_runtime(
                    harness,
                    bundle,
                    candidate_identity,
                    driver.clone(),
                    fail_after_quiescence,
                )
            }
        },
        NativeHeadlessPolicy {
            maximum_restarts: 2,
            ..NativeHeadlessPolicy::default()
        },
    )?;

    let initial = service.start()?;
    assert!(matches!(initial.state, NativeHeadlessState::Ready));
    assert_eq!(
        initial.deployment_identity_sha256.as_deref(),
        Some(headless.execution_bundle_identity_sha256())
    );
    assert_eq!(initial.provider_deployment.as_ref(), Some(&headless));
    let initial_driver = driver
        .lock()
        .map_err(|_| "live provider driver is unavailable")?
        .clone()
        .ok_or("live provider driver is absent")?;
    let initial_execution = initial_driver.execute(1).await?;
    assert!(initial_execution.succeeded);
    assert_eq!(initial_execution.call_ids, (1_u64..=10).collect::<Vec<_>>());
    assert_eq!(initial_execution.materializations, 1);
    let replay = service.restart()?;
    assert!(matches!(replay.state, NativeHeadlessState::Ready));
    assert_eq!(replay.restart_count, 0);
    assert_eq!(replay.provider_deployment.as_ref(), Some(&headless));
    assert_eq!(preparations.load(Ordering::Acquire), 2);
    assert!(Arc::ptr_eq(
        &initial_driver,
        driver
            .lock()
            .map_err(|_| "live provider driver is unavailable")?
            .as_ref()
            .ok_or("live provider driver is absent")?
    ));

    mode.store(1, Ordering::Release);
    assert!(matches!(
        service.restart(),
        Err(NativeHeadlessError::DeploymentRejected(_))
    ));
    let retained = service.snapshot()?;
    assert!(matches!(retained.state, NativeHeadlessState::Ready));
    assert_eq!(retained.restart_count, 0);
    assert!(Arc::ptr_eq(
        &initial_driver,
        driver
            .lock()
            .map_err(|_| "live provider driver is unavailable")?
            .as_ref()
            .ok_or("live provider driver is absent")?
    ));

    mode.store(2, Ordering::Release);
    let replaced = service.restart()?;
    assert!(matches!(replaced.state, NativeHeadlessState::Ready));
    assert_eq!(replaced.restart_count, 1);
    assert_eq!(
        replaced.deployment_identity_sha256.as_deref(),
        Some(replacement_bundle.identity_sha256())
    );
    let replacement = NativeProviderDeploymentIdentity::from_registry_bundle(
        &replacement_bundle,
        replacement_candidate.identity(),
    )?;
    assert_eq!(replaced.provider_deployment, Some(replacement));
    assert!(initial_driver.execute(2).await.is_err());
    let replacement_driver = driver
        .lock()
        .map_err(|_| "replacement provider driver is unavailable")?
        .clone()
        .ok_or("replacement provider driver is absent")?;
    assert!(!Arc::ptr_eq(&initial_driver, &replacement_driver));
    assert!(replacement_driver.execute(3).await?.succeeded);

    mode.store(3, Ordering::Release);
    assert!(matches!(
        service.restart(),
        Err(NativeHeadlessError::RuntimeBuild(_))
    ));
    let offline = service.snapshot()?;
    assert!(matches!(offline.state, NativeHeadlessState::Failed { .. }));
    assert!(
        driver
            .lock()
            .map_err(|_| "offline provider driver is unavailable")?
            .is_none()
    );

    mode.store(4, Ordering::Release);
    let recovered = service.start()?;
    assert!(matches!(recovered.state, NativeHeadlessState::Ready));
    assert_eq!(recovered.restart_count, 2);
    assert_eq!(
        recovered.deployment_identity_sha256.as_deref(),
        Some(retry_bundle.identity_sha256())
    );
    let retry = NativeProviderDeploymentIdentity::from_registry_bundle(
        &retry_bundle,
        retry_candidate.identity(),
    )?;
    assert_eq!(recovered.provider_deployment, Some(retry));
    let retry_driver = driver
        .lock()
        .map_err(|_| "retry provider driver is unavailable")?
        .clone()
        .ok_or("retry provider driver is absent")?;
    assert!(retry_driver.execute(4).await?.succeeded);

    let headless_source = include_str!("../../sim_api/src/headless.rs");
    assert!(!headless_source.contains("deployment_identity_sha256: impl Into<String>"));
    assert!(!headless_source.contains("candidate_identity: impl Into<String>"));
    assert!(headless_source.contains("runtime.provider_deployment()"));
    Ok(())
}
