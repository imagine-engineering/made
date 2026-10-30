#[cfg(feature = "test-support")]
use crate::component_host::PreparedProviderV2SupervisorExecution;
use crate::{
    ComponentHostError, InvocationResult, PluginError, PluginInvocationExecutor,
    PreparedPluginInvocation, ProviderInvocationResult,
};
use sim_plugin_sdk::InvocationError;
use sim_plugin_sdk::{
    PortCardinality, PortDirection, PortPresence, ProviderResultReceiptSet, TypeRegistry,
    ValueFamily,
};
#[cfg(feature = "test-support")]
use sim_runtime::NativeProviderWorkerV2ActuatorRoute;
use sim_runtime::{
    NativeProviderWorkerBridgeAttachment, PluginAuthorizationVerifier, PluginCapabilityBroker,
    PluginCapabilityInvocation, PluginServiceInvocationContext, ProviderCostAuthorizationAuthority,
    ProviderInvocationIdentity, ProviderResultReceipt, ProviderResultReceiptAuthority,
    ProviderResultReceiptIssuer, ProviderTransportPort, ProviderTransportResponse,
    ProviderTransportValue, RetainedPluginExecution, RuntimeSupervisor, RuntimeSupervisorError,
    WorkerLaunchConfig, WorkerRegistryDeploymentPlan,
};
use sim_types::{
    AttemptId, ProfileId, PromptId, WorkerPluginExecutionFailure, WorkerPluginExecutionOutcome,
    WorkerRegistryGeneration,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub struct PrivateWorkerPluginExecutor {
    launch: WorkerLaunchConfig,
    broker: PluginCapabilityBroker,
    provider_result_receipts: Option<PrivateWorkerProviderResultReceipts>,
    provider_worker_bridge: Mutex<Option<NativeProviderWorkerBridgeAttachment>>,
    #[cfg(feature = "test-support")]
    provider_v2_actuator: Mutex<Option<PrivateWorkerProviderV2ActuatorAttachment>>,
    commands: async_channel::Sender<PrivateWorkerCommand>,
}

#[cfg(feature = "test-support")]
#[derive(Clone)]
pub struct PrivateWorkerProviderV2ActuatorAttachment {
    sender: async_channel::Sender<NativeProviderWorkerV2ActuatorRoute>,
}

#[cfg(feature = "test-support")]
pub fn private_worker_provider_v2_actuator_route() -> (
    PrivateWorkerProviderV2ActuatorAttachment,
    async_channel::Receiver<NativeProviderWorkerV2ActuatorRoute>,
) {
    let (sender, receiver) = async_channel::bounded(1);
    (
        PrivateWorkerProviderV2ActuatorAttachment { sender },
        receiver,
    )
}

#[derive(Clone)]
struct PrivateWorkerProviderResultReceipts {
    principal_id: Arc<str>,
    issuer: Arc<ProviderResultReceiptIssuer>,
    lifetime: Duration,
    cost_authority: Option<Arc<dyn ProviderCostAuthorizationAuthority>>,
}

enum PrivateWorkerCommand {
    Legacy {
        deployment: WorkerRegistryDeploymentPlan,
        invocation: Vec<u8>,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        capability_invocation: PluginCapabilityInvocation,
        response: async_channel::Sender<Result<RetainedPluginExecution, RuntimeSupervisorError>>,
    },
    #[cfg(feature = "test-support")]
    ProviderV2 {
        deployment: WorkerRegistryDeploymentPlan,
        execution: PreparedProviderV2SupervisorExecution,
        response: async_channel::Sender<
            Result<
                (
                    WorkerPluginExecutionOutcome,
                    Option<ProviderTransportResponse>,
                ),
                RuntimeSupervisorError,
            >,
        >,
    },
}

struct PrivateWorkerState {
    supervisor: Option<RuntimeSupervisor>,
    deployed_registry_identity: Option<(WorkerRegistryGeneration, String)>,
    authorization_verifier: Option<PluginAuthorizationVerifier>,
}

impl PrivateWorkerPluginExecutor {
    pub fn new(
        launch: WorkerLaunchConfig,
        broker: PluginCapabilityBroker,
    ) -> Result<Arc<Self>, ComponentHostError> {
        Self::new_internal(launch, broker, None)
    }

    pub fn new_with_provider_result_receipts(
        launch: WorkerLaunchConfig,
        broker: PluginCapabilityBroker,
        principal_id: impl Into<Arc<str>>,
        issuer: Arc<ProviderResultReceiptIssuer>,
        lifetime: Duration,
    ) -> Result<Arc<Self>, ComponentHostError> {
        let principal_id = principal_id.into();
        if principal_id.is_empty() || lifetime.is_zero() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider result receipt authority is invalid".to_owned(),
            ));
        }
        Self::new_internal(
            launch,
            broker,
            Some(PrivateWorkerProviderResultReceipts {
                principal_id,
                issuer,
                lifetime,
                cost_authority: None,
            }),
        )
    }

    pub fn new_with_provider_authorities(
        launch: WorkerLaunchConfig,
        broker: PluginCapabilityBroker,
        principal_id: impl Into<Arc<str>>,
        issuer: Arc<ProviderResultReceiptIssuer>,
        lifetime: Duration,
        cost_authority: Arc<dyn ProviderCostAuthorizationAuthority>,
    ) -> Result<Arc<Self>, ComponentHostError> {
        let principal_id = principal_id.into();
        if principal_id.is_empty() || lifetime.is_zero() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider authority configuration is invalid".to_owned(),
            ));
        }
        Self::new_internal(
            launch,
            broker,
            Some(PrivateWorkerProviderResultReceipts {
                principal_id,
                issuer,
                lifetime,
                cost_authority: Some(cost_authority),
            }),
        )
    }

    fn new_internal(
        launch: WorkerLaunchConfig,
        broker: PluginCapabilityBroker,
        provider_result_receipts: Option<PrivateWorkerProviderResultReceipts>,
    ) -> Result<Arc<Self>, ComponentHostError> {
        let (commands, receiver) = async_channel::bounded(64);
        let actor_launch = launch.clone();
        std::thread::Builder::new()
            .name("sim-plugin-worker".to_owned())
            .spawn(move || run_private_worker_actor(actor_launch, receiver))
            .map_err(worker_boundary_error)?;
        Ok(Arc::new(Self {
            launch,
            broker,
            provider_result_receipts,
            provider_worker_bridge: Mutex::new(None),
            #[cfg(feature = "test-support")]
            provider_v2_actuator: Mutex::new(None),
            commands,
        }))
    }

    pub fn attach_provider_worker_bridge(
        &self,
        attachment: NativeProviderWorkerBridgeAttachment,
    ) -> Result<(), ComponentHostError> {
        let attachment = attachment
            .bind_to_worker_profile(self.launch.profile_id)
            .map_err(worker_boundary_error)?;
        let mut current = self.provider_worker_bridge.lock().map_err(|error| {
            ComponentHostError::ExecutionBoundary(format!(
                "private worker provider bridge slot is unavailable: {error}"
            ))
        })?;
        if current
            .as_ref()
            .is_some_and(NativeProviderWorkerBridgeAttachment::is_live)
        {
            return Err(ComponentHostError::ExecutionBoundary(
                "private worker already has a live native provider bridge".to_owned(),
            ));
        }
        *current = Some(attachment);
        Ok(())
    }

    #[cfg(feature = "test-support")]
    pub fn attach_provider_v2_actuator(
        &self,
        actuator: PrivateWorkerProviderV2ActuatorAttachment,
    ) -> Result<(), ComponentHostError> {
        let mut current = self.provider_v2_actuator.lock().map_err(|error| {
            ComponentHostError::ExecutionBoundary(format!(
                "private worker provider-v2 actuator slot is unavailable: {error}"
            ))
        })?;
        if current
            .as_ref()
            .is_some_and(|attachment| !attachment.sender.is_closed())
        {
            return Err(ComponentHostError::ExecutionBoundary(
                "private worker already has a live provider-v2 actuator route".to_owned(),
            ));
        }
        *current = Some(actuator);
        Ok(())
    }

    async fn execute_prepared(
        &self,
        invocation: PreparedPluginInvocation,
    ) -> Result<RetainedPluginExecution, ComponentHostError> {
        let context = invocation.context();
        context.cancellation.check().map_err(|_| {
            ComponentHostError::Plugin(PluginError::Invocation(InvocationError::Cancelled))
        })?;
        verify_profile(
            self.launch.profile_id,
            invocation.authorization().capabilities().profile_id(),
        )?;
        if invocation.is_provider_v2() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 production actuation is unavailable until a deployment-owned actuator is installed"
                    .to_owned(),
            ));
        }
        let timeout =
            std::time::Duration::from_millis(invocation.worker_invocation().timeout_milliseconds());
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            ComponentHostError::ExecutionBoundary(
                "plugin invocation deadline overflowed".to_owned(),
            )
        })?;
        let service_context = if invocation.worker_invocation().provider_request().is_some() {
            let receipt_configuration =
                self.provider_result_receipts.as_ref().ok_or_else(|| {
                    ComponentHostError::ExecutionBoundary(
                        "provider invocation requires app-owned result receipt authority"
                            .to_owned(),
                    )
                })?;
            let provider_execution = context
                .provider_execution()
                .map_err(worker_boundary_error)?;
            let provider_binding_sha256 =
                invocation.provider_binding_sha256().ok_or_else(|| {
                    ComponentHostError::ExecutionBoundary(
                        "provider invocation is missing its signed binding identity".to_owned(),
                    )
                })?;
            let authority = ProviderResultReceiptAuthority::new(
                receipt_configuration.principal_id.as_ref(),
                provider_execution.compiled_plan_sha256(),
                provider_binding_sha256,
                receipt_configuration.issuer.clone(),
                receipt_configuration.lifetime,
            )
            .map_err(worker_boundary_error)?;
            let service_context = PluginServiceInvocationContext::new_with_principal(
                self.launch.profile_id,
                context.prompt_id,
                context.attempt_id,
                context.node_id.clone(),
                receipt_configuration.principal_id.as_ref(),
                invocation.authorization().clone(),
                context.cancellation.clone(),
                deadline,
                invocation.worker_invocation().maximum_response_bytes(),
            )
            .and_then(|context| context.with_provider_result_authority(authority))
            .map_err(worker_boundary_error)?;
            match invocation.provider_price_badge() {
                Some(price_badge) => service_context
                    .with_provider_cost_requirement(
                        price_badge.clone(),
                        receipt_configuration.cost_authority.clone(),
                    )
                    .map_err(worker_boundary_error)?,
                None => service_context,
            }
        } else {
            PluginServiceInvocationContext::new(
                self.launch.profile_id,
                context.prompt_id,
                context.attempt_id,
                context.node_id.clone(),
                invocation.authorization().clone(),
                context.cancellation.clone(),
                deadline,
                invocation.worker_invocation().maximum_response_bytes(),
            )
            .map_err(worker_boundary_error)?
        };
        let capability_invocation = self
            .broker
            .begin_invocation(service_context)
            .map_err(worker_boundary_error)?;
        let (response, result) = async_channel::bounded(1);
        let command = PrivateWorkerCommand::Legacy {
            deployment: invocation.deployment().clone(),
            invocation: invocation.worker_invocation().to_bytes()?,
            prompt_id: context.prompt_id,
            attempt_id: context.attempt_id,
            capability_invocation,
            response,
        };
        self.commands.send(command).await.map_err(|_| {
            ComponentHostError::ExecutionBoundary(
                "private plugin worker actor is unavailable".to_owned(),
            )
        })?;
        result
            .recv()
            .await
            .map_err(|_| {
                ComponentHostError::ExecutionBoundary(
                    "private plugin worker actor closed its response".to_owned(),
                )
            })?
            .map_err(worker_boundary_error)
    }

    #[cfg(feature = "test-support")]
    async fn execute_provider_v2_prepared(
        &self,
        invocation: PreparedPluginInvocation,
    ) -> Result<
        (
            WorkerPluginExecutionOutcome,
            Option<ProviderTransportResponse>,
        ),
        ComponentHostError,
    > {
        let context = invocation.context();
        context.cancellation.check().map_err(|_| {
            ComponentHostError::Plugin(PluginError::Invocation(InvocationError::Cancelled))
        })?;
        verify_profile(
            self.launch.profile_id,
            invocation.authorization().capabilities().profile_id(),
        )?;
        let deployment = invocation.deployment().clone();
        let prepared = {
            let bridge = self.provider_worker_bridge.lock().map_err(|error| {
                ComponentHostError::ExecutionBoundary(format!(
                    "private worker provider bridge slot is unavailable: {error}"
                ))
            })?;
            let attachment = bridge.as_ref().ok_or_else(|| {
                ComponentHostError::ExecutionBoundary(
                    "provider-v2 private worker execution requires the live controller bridge"
                        .to_owned(),
                )
            })?;
            invocation.activate_provider_v2(attachment)?
        };
        let actuator_sender = self
            .provider_v2_actuator
            .lock()
            .map_err(|error| {
                ComponentHostError::ExecutionBoundary(format!(
                    "private worker provider-v2 actuator slot is unavailable: {error}"
                ))
            })?
            .as_ref()
            .filter(|attachment| !attachment.sender.is_closed())
            .map(|attachment| attachment.sender.clone())
            .ok_or_else(|| {
                ComponentHostError::ExecutionBoundary(
                    "provider-v2 production actuation is unavailable until a deployment-owned actuator is installed"
                        .to_owned(),
                )
            })?;
        let (execution, actuator) = prepared.into_supervised_parts(&self.broker)?;
        actuator_sender.try_send(actuator).map_err(|error| {
            ComponentHostError::ExecutionBoundary(format!(
                "provider-v2 actuator capacity-one route rejected execution: {error}"
            ))
        })?;
        let (response, result) = async_channel::bounded(1);
        self.commands
            .send(PrivateWorkerCommand::ProviderV2 {
                deployment,
                execution,
                response,
            })
            .await
            .map_err(|_| {
                ComponentHostError::ExecutionBoundary(
                    "private plugin worker actor is unavailable".to_owned(),
                )
            })?;
        let outcome = result
            .recv()
            .await
            .map_err(|_| {
                ComponentHostError::ExecutionBoundary(
                    "private plugin worker actor closed its provider-v2 response".to_owned(),
                )
            })?
            .map_err(worker_boundary_error)?;
        validate_provider_v2_worker_outcome(outcome)
    }
}

impl PluginInvocationExecutor for PrivateWorkerPluginExecutor {
    fn execute<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<InvocationResult, ComponentHostError>> + Send + 'a>>
    {
        Box::pin(async move {
            let retained = self.execute_prepared(invocation).await?;
            match decode_worker_result(retained.outcome()) {
                Ok(result) => {
                    retained.finish().map_err(worker_boundary_error)?;
                    Ok(result)
                }
                Err(error) => {
                    retained.abort();
                    Err(error)
                }
            }
        })
    }

    fn execute_provider<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<dyn Future<Output = Result<ProviderInvocationResult, ComponentHostError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let context = invocation.context().clone();
            let node = invocation.provider_node()?.clone();
            let namespace = invocation.provider_namespace().map(str::to_owned);
            let mut retained = self.execute_prepared(invocation).await?;
            let result = (|| {
                let mut result = decode_worker_provider_result(retained.outcome())?;
                if !result.outputs.is_empty()
                    || !result.output_presence.is_empty()
                    || !result.effects.outputs.is_empty()
                    || !result.effects.logs.is_empty()
                    || !result.effects.ui_state.is_empty()
                    || !result.effects.routes.is_empty()
                {
                    return Err(ComponentHostError::ExecutionBoundary(
                        "provider result contains guest-authored outputs or effects".to_owned(),
                    ));
                }
                if let Some((identity, original_receipt, digest, response)) =
                    materialize_private_worker_image_result(
                        &retained,
                        namespace.as_deref(),
                        &node,
                        result.receipts(),
                        &context,
                    )?
                {
                    result.set_materialized_provider_response(
                        identity,
                        original_receipt,
                        digest,
                        response,
                    );
                }
                let receipt_set = ProviderResultReceiptSet::new(result.receipts().to_vec())
                    .map_err(worker_boundary_error)?;
                let resolved = retained
                    .resolve_provider_result_receipt_set(&receipt_set)
                    .map_err(worker_boundary_error)?;
                result.set_resolved_provider_results(resolved);
                if let Some(namespace) = namespace.as_deref() {
                    result
                        .verify_materialized_provider_response(namespace, &node.id)
                        .map_err(ComponentHostError::Plugin)?;
                }
                Ok(result)
            })();
            match result {
                Ok(result) => {
                    retained.finish().map_err(worker_boundary_error)?;
                    Ok(result)
                }
                Err(error) => {
                    retained.abort();
                    Err(error)
                }
            }
        })
    }

    fn execute_provider_v2<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<dyn Future<Output = Result<ProviderTransportResponse, ComponentHostError>> + Send + 'a>,
    > {
        Box::pin(async move {
            #[cfg(feature = "test-support")]
            {
                let (outcome, materialization) =
                    self.execute_provider_v2_prepared(invocation).await?;
                return match outcome {
                    WorkerPluginExecutionOutcome::Succeeded(_) => {
                        materialization.ok_or_else(|| {
                            ComponentHostError::ExecutionBoundary(
                                "provider-v2 successful execution lost its materialization"
                                    .to_owned(),
                            )
                        })
                    }
                    WorkerPluginExecutionOutcome::Failed(failure) => {
                        decode_worker_failure(&failure)
                    }
                };
            }
            #[cfg(not(feature = "test-support"))]
            {
                drop(invocation);
                Err(ComponentHostError::ExecutionBoundary(
                    "provider-v2 production actuation is unavailable until Task427 installs the deployment-owned actuator"
                        .to_owned(),
                ))
            }
        })
    }
}

fn run_private_worker_actor(
    launch: WorkerLaunchConfig,
    commands: async_channel::Receiver<PrivateWorkerCommand>,
) {
    smol::block_on(async move {
        let mut state = PrivateWorkerState {
            supervisor: None,
            deployed_registry_identity: None,
            authorization_verifier: None,
        };
        while let Ok(command) = commands.recv().await {
            match command {
                PrivateWorkerCommand::Legacy {
                    deployment,
                    invocation,
                    prompt_id,
                    attempt_id,
                    capability_invocation,
                    response,
                } => {
                    let result = execute_private_worker_command(
                        &launch,
                        &mut state,
                        deployment,
                        invocation,
                        prompt_id,
                        attempt_id,
                        capability_invocation,
                    )
                    .await;
                    let reset = result
                        .as_ref()
                        .is_err_and(worker_failure_requires_supervisor_reset);
                    if response.send(result).await.is_err() || reset {
                        reset_private_worker_state(&mut state);
                    }
                }
                #[cfg(feature = "test-support")]
                PrivateWorkerCommand::ProviderV2 {
                    deployment,
                    execution,
                    response,
                } => {
                    let result = execute_private_worker_provider_v2_command(
                        &launch, &mut state, deployment, execution,
                    )
                    .await;
                    let reset = result
                        .as_ref()
                        .is_err_and(worker_failure_requires_supervisor_reset);
                    if response.send(result).await.is_err() || reset {
                        reset_private_worker_state(&mut state);
                    }
                }
            }
        }
    });
}

async fn execute_private_worker_command(
    launch: &WorkerLaunchConfig,
    state: &mut PrivateWorkerState,
    deployment: WorkerRegistryDeploymentPlan,
    invocation: Vec<u8>,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    capability_invocation: PluginCapabilityInvocation,
) -> Result<RetainedPluginExecution, RuntimeSupervisorError> {
    ensure_private_worker_supervisor(launch, state, &deployment).await?;
    let supervisor = state
        .supervisor
        .as_mut()
        .ok_or(RuntimeSupervisorError::NotRunning)?;
    match supervisor
        .execute_plugin_retaining_capabilities(
            prompt_id,
            attempt_id,
            invocation,
            capability_invocation,
        )
        .await
    {
        Ok(outcome) => Ok(outcome),
        Err(error) => Err(worker_failure_with_logs(error, supervisor)),
    }
}

#[cfg(feature = "test-support")]
async fn execute_private_worker_provider_v2_command(
    launch: &WorkerLaunchConfig,
    state: &mut PrivateWorkerState,
    deployment: WorkerRegistryDeploymentPlan,
    execution: PreparedProviderV2SupervisorExecution,
) -> Result<
    (
        WorkerPluginExecutionOutcome,
        Option<ProviderTransportResponse>,
    ),
    RuntimeSupervisorError,
> {
    ensure_private_worker_supervisor(launch, state, &deployment).await?;
    let supervisor = state
        .supervisor
        .as_mut()
        .ok_or(RuntimeSupervisorError::NotRunning)?;
    execution
        .execute(supervisor)
        .await
        .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))
}

async fn ensure_private_worker_supervisor(
    launch: &WorkerLaunchConfig,
    state: &mut PrivateWorkerState,
    deployment: &WorkerRegistryDeploymentPlan,
) -> Result<(), RuntimeSupervisorError> {
    let registry_identity = (
        deployment.begin().generation(),
        deployment
            .begin()
            .registry_digest_sha256()
            .as_str()
            .to_owned(),
    );
    if state.authorization_verifier.as_ref() != Some(deployment.authorization_verifier()) {
        reset_private_worker_state(state);
    }
    if state.supervisor.is_none() {
        let mut launch = launch.clone();
        launch.plugin_authorization_verifier = Some(deployment.authorization_verifier().clone());
        let mut supervisor = RuntimeSupervisor::start(launch).await?;
        if let Err(error) = supervisor.deploy_registry(deployment).await {
            return Err(worker_failure_with_logs(error, &supervisor));
        }
        state.supervisor = Some(supervisor);
        state.deployed_registry_identity = Some(registry_identity.clone());
        state.authorization_verifier = Some(deployment.authorization_verifier().clone());
    } else if state.deployed_registry_identity.as_ref() != Some(&registry_identity) {
        let supervisor = state
            .supervisor
            .as_mut()
            .ok_or(RuntimeSupervisorError::NotRunning)?;
        if let Err(error) = supervisor.deploy_registry(deployment).await {
            return Err(worker_failure_with_logs(error, supervisor));
        }
        state.deployed_registry_identity = Some(registry_identity);
    }
    Ok(())
}

fn reset_private_worker_state(state: &mut PrivateWorkerState) {
    state.supervisor = None;
    state.deployed_registry_identity = None;
    state.authorization_verifier = None;
}

fn worker_failure_with_logs(
    error: RuntimeSupervisorError,
    supervisor: &RuntimeSupervisor,
) -> RuntimeSupervisorError {
    worker_failure_with_log_lines(error, &supervisor.logs())
}

fn worker_failure_with_log_lines(
    error: RuntimeSupervisorError,
    logs: &[String],
) -> RuntimeSupervisorError {
    if logs.is_empty() {
        error
    } else if let RuntimeSupervisorError::InvalidRegistryDeployment(reason) = error {
        RuntimeSupervisorError::InvalidRegistryDeployment(format!(
            "{reason}; worker diagnostics: {}",
            logs.join(" | ")
        ))
    } else {
        RuntimeSupervisorError::Protocol(format!(
            "private worker operation failed: {error}; worker diagnostics: {}",
            logs.join(" | ")
        ))
    }
}

fn worker_failure_requires_supervisor_reset(error: &RuntimeSupervisorError) -> bool {
    !matches!(error, RuntimeSupervisorError::InvalidRegistryDeployment(_))
}

fn verify_profile(
    profile_id: ProfileId,
    authorization_profile: &str,
) -> Result<(), ComponentHostError> {
    if authorization_profile != profile_id.0.to_string() {
        return Err(ComponentHostError::ExecutionBoundary(
            "worker and plugin authorization profiles differ".to_owned(),
        ));
    }
    Ok(())
}

fn decode_worker_result(
    outcome: &WorkerPluginExecutionOutcome,
) -> Result<InvocationResult, ComponentHostError> {
    match outcome {
        WorkerPluginExecutionOutcome::Succeeded(bytes) => {
            serde_json::from_slice(&bytes).map_err(|_| {
                ComponentHostError::ExecutionBoundary(
                    "private worker returned an invalid plugin result".to_owned(),
                )
            })
        }
        WorkerPluginExecutionOutcome::Failed(failure) => Err(match failure {
            WorkerPluginExecutionFailure::Cancelled => {
                ComponentHostError::Plugin(PluginError::Invocation(InvocationError::Cancelled))
            }
            WorkerPluginExecutionFailure::TimedOut => {
                ComponentHostError::Plugin(PluginError::Invocation(InvocationError::TimedOut))
            }
            WorkerPluginExecutionFailure::Trap { diagnostic } => {
                ComponentHostError::Plugin(PluginError::WasmTrap(diagnostic.clone()))
            }
            WorkerPluginExecutionFailure::InvalidInvocation => {
                ComponentHostError::ExecutionBoundary(
                    "private worker rejected the plugin invocation".to_owned(),
                )
            }
            WorkerPluginExecutionFailure::CapabilityDenied => {
                ComponentHostError::ExecutionBoundary(
                    "private worker plugin capability was denied".to_owned(),
                )
            }
            WorkerPluginExecutionFailure::HostFailure => ComponentHostError::ExecutionBoundary(
                "private worker plugin execution failed".to_owned(),
            ),
        }),
    }
}

fn decode_worker_provider_result(
    outcome: &WorkerPluginExecutionOutcome,
) -> Result<ProviderInvocationResult, ComponentHostError> {
    match outcome {
        WorkerPluginExecutionOutcome::Succeeded(bytes) => {
            serde_json::from_slice(&bytes).map_err(|_| {
                ComponentHostError::ExecutionBoundary(
                    "private worker returned an invalid provider result".to_owned(),
                )
            })
        }
        WorkerPluginExecutionOutcome::Failed(failure) => decode_worker_failure(failure),
    }
}

fn decode_worker_failure<T>(
    failure: &WorkerPluginExecutionFailure,
) -> Result<T, ComponentHostError> {
    Err(match failure {
        WorkerPluginExecutionFailure::Cancelled => {
            ComponentHostError::Plugin(PluginError::Invocation(InvocationError::Cancelled))
        }
        WorkerPluginExecutionFailure::TimedOut => {
            ComponentHostError::Plugin(PluginError::Invocation(InvocationError::TimedOut))
        }
        WorkerPluginExecutionFailure::Trap { diagnostic } => {
            ComponentHostError::Plugin(PluginError::WasmTrap(diagnostic.clone()))
        }
        WorkerPluginExecutionFailure::InvalidInvocation => ComponentHostError::ExecutionBoundary(
            "private worker rejected the plugin invocation".to_owned(),
        ),
        WorkerPluginExecutionFailure::CapabilityDenied => ComponentHostError::ExecutionBoundary(
            "private worker plugin capability was denied".to_owned(),
        ),
        WorkerPluginExecutionFailure::HostFailure => ComponentHostError::ExecutionBoundary(
            "private worker plugin execution failed".to_owned(),
        ),
    })
}

#[cfg(any(test, feature = "test-support"))]
fn validate_provider_v2_worker_outcome(
    outcome: (
        WorkerPluginExecutionOutcome,
        Option<ProviderTransportResponse>,
    ),
) -> Result<
    (
        WorkerPluginExecutionOutcome,
        Option<ProviderTransportResponse>,
    ),
    ComponentHostError,
> {
    if let WorkerPluginExecutionOutcome::Failed(failure) = &outcome.0 {
        return decode_worker_failure(failure);
    }
    Ok(outcome)
}

fn materialize_private_worker_image_result(
    retained: &RetainedPluginExecution,
    namespace: Option<&str>,
    node: &sim_plugin_sdk::PluginNode,
    receipts: &[Vec<u8>],
    context: &sim_runtime::NodeContext,
) -> Result<
    Option<(
        ProviderInvocationIdentity,
        Vec<u8>,
        String,
        ProviderTransportResponse,
    )>,
    ComponentHostError,
> {
    if namespace != Some("zed.sim.provider.bfl") || node.id != "FluxProUltraImageNode" {
        return Ok(None);
    }
    let outputs = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .collect::<Vec<_>>();
    let [output] = outputs.as_slice() else {
        return Ok(None);
    };
    let registry = TypeRegistry::built_in().map_err(worker_boundary_error)?;
    if output.type_id != *registry.resolve("IMAGE").map_err(worker_boundary_error)? {
        return Ok(None);
    }
    if output.cardinality != PortCardinality::Singular || output.presence != PortPresence::Required
    {
        return Err(ComponentHostError::ExecutionBoundary(
            "PNG receipt requires one required singular IMAGE output".to_owned(),
        ));
    }
    let [receipt_bytes] = receipts else {
        return Err(ComponentHostError::ExecutionBoundary(
            "PNG receipt requires exactly one original result receipt".to_owned(),
        ));
    };
    // Receipt fields select an expected identity only. The retained issuer session
    // authenticates that identity and the original bytes before native decoding.
    let receipt =
        ProviderResultReceipt::from_bytes(receipt_bytes).map_err(worker_boundary_error)?;
    if retained
        .is_provider_result_structured_transport_response(
            receipt_bytes,
            receipt.identity(),
            context,
        )
        .map_err(worker_boundary_error)?
    {
        return Ok(None);
    }
    let payload = retained
        .materialize_provider_downloaded_image(receipt_bytes, receipt.identity(), context)
        .map_err(worker_boundary_error)?;
    let value = ProviderTransportValue::from_native_payload(
        output.type_id.to_string(),
        ValueFamily::Tensor,
        &payload,
    )
    .map_err(worker_boundary_error)?;
    context
        .cancellation
        .check()
        .map_err(worker_boundary_error)?;
    let response = ProviderTransportResponse::checked(
        node.id.clone(),
        vec![
            ProviderTransportPort::checked(output.id.clone(), true, vec![value])
                .map_err(worker_boundary_error)?,
        ],
    )
    .map_err(worker_boundary_error)?;
    Ok(Some((
        receipt.identity().clone(),
        receipt_bytes.to_vec(),
        receipt.result_sha256().to_owned(),
        response,
    )))
}

fn worker_boundary_error(error: impl std::fmt::Display) -> ComponentHostError {
    ComponentHostError::ExecutionBoundary(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_v2_cancelled_worker_outcome_preserves_typed_cancellation() {
        assert!(matches!(
            validate_provider_v2_worker_outcome((
                WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::Cancelled),
                None,
            )),
            Err(ComponentHostError::Plugin(PluginError::Invocation(
                InvocationError::Cancelled
            )))
        ));
    }

    #[test]
    fn worker_failure_with_logs_preserves_timeout_cause_and_empty_log_discriminant() {
        let original = RuntimeSupervisorError::Timeout {
            stage: "worker response",
        };
        let failure = worker_failure_with_log_lines(
            original.clone(),
            &["registry component compilation began".to_owned()],
        );
        assert!(matches!(
            &failure,
            RuntimeSupervisorError::Protocol(diagnostic)
                if diagnostic.contains("native worker timed out during worker response")
                    && diagnostic.contains("registry component compilation began")
        ));
        assert!(worker_failure_requires_supervisor_reset(&failure));
        assert_eq!(
            worker_failure_with_log_lines(original.clone(), &[]),
            original
        );
    }

    #[test]
    fn worker_failure_with_logs_preserves_fatal_code_and_stderr_order() {
        let original = RuntimeSupervisorError::WorkerFatal {
            code: "worker_protocol_error".to_owned(),
            message: "registry acknowledgement was rejected".to_owned(),
        };
        let failure = worker_failure_with_log_lines(
            original.clone(),
            &[
                "first child diagnostic".to_owned(),
                "second child diagnostic".to_owned(),
            ],
        );
        assert!(matches!(
            &failure,
            RuntimeSupervisorError::Protocol(diagnostic)
                if diagnostic.contains("native worker reported fatal worker_protocol_error: registry acknowledgement was rejected")
        ));
        let diagnostic = failure.to_string();
        assert!(matches!(
            (
                diagnostic.find("first child diagnostic"),
                diagnostic.find("second child diagnostic"),
            ),
            (Some(first), Some(second)) if first < second
        ));
        assert!(worker_failure_requires_supervisor_reset(&failure));
        assert_eq!(
            worker_failure_with_log_lines(original.clone(), &[]),
            original
        );
    }

    #[test]
    fn rejected_registry_deployment_retains_diagnostics_without_resetting_supervisor() {
        let original = RuntimeSupervisorError::InvalidRegistryDeployment(
            "worker rejected registry generation 1: ComponentCompilationFailed".to_owned(),
        );
        let failure = worker_failure_with_log_lines(
            original.clone(),
            &[
                "component instantiation failed".to_owned(),
                "epoch deadline reached".to_owned(),
            ],
        );
        assert!(matches!(
            &failure,
            RuntimeSupervisorError::InvalidRegistryDeployment(reason)
                if reason == "worker rejected registry generation 1: ComponentCompilationFailed; worker diagnostics: component instantiation failed | epoch deadline reached"
        ));
        assert!(!worker_failure_requires_supervisor_reset(&failure));
        assert_eq!(
            worker_failure_with_log_lines(original.clone(), &[]),
            original
        );
    }

    #[test]
    fn rejected_registry_replacement_preserves_the_live_supervisor() {
        assert!(!worker_failure_requires_supervisor_reset(
            &RuntimeSupervisorError::InvalidRegistryDeployment(
                "component compilation failed".to_owned(),
            )
        ));
        assert!(worker_failure_requires_supervisor_reset(
            &RuntimeSupervisorError::Protocol("invalid worker frame".to_owned())
        ));
        assert!(worker_failure_requires_supervisor_reset(
            &RuntimeSupervisorError::WorkerFatal {
                code: "worker_protocol_error".to_owned(),
                message: "fatal".to_owned(),
            }
        ));
    }

    #[test]
    fn private_worker_provider_bridge_attachment_is_one_live_weak_consumer() {
        let source = include_str!("private_worker.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map(|(production, _)| production)
            .expect("private worker production source");
        for required in [
            "provider_worker_bridge: Mutex<Option<NativeProviderWorkerBridgeAttachment>>",
            "bind_to_worker_profile(self.launch.profile_id)",
            "is_some_and(NativeProviderWorkerBridgeAttachment::is_live)",
            "private worker already has a live native provider bridge",
            "*current = Some(attachment)",
        ] {
            assert!(
                production.contains(required),
                "private worker bridge attachment lacks {required}"
            );
        }
        for forbidden in [
            concat!("ProviderRuntimeStream", "Service::new()"),
            concat!("ProviderRuntimeActivation", "GrantSource"),
            concat!("provider_runtime_stream", "_service"),
            concat!("provider_runtime_activation", "_grants"),
        ] {
            assert!(
                !production.contains(forbidden),
                "private worker bridge attachment exposes {forbidden}"
            );
        }
    }
}
