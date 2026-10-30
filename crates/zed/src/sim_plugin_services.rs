#[cfg(not(test))]
use futures::AsyncReadExt as _;
#[cfg(not(test))]
use gpui::{App, BackgroundExecutor};
#[cfg(not(test))]
use http_client::{AsyncBody, HttpClient, Request, http};
use sim_model::{ModelStore, ParserLimits};
use sim_plugin_host::{
    ComponentExecutionBoundary, ComponentHost, ComponentHostError,
    ComponentHostProviderInvocationAuthority, PrivateWorkerPluginExecutor,
};
use sim_runtime::{
    AuthorizedCredentialPresenceRequest, AuthorizedProviderRequest, CredentialPresenceActuator,
    PluginCapabilityBroker, PluginRngPolicy, PluginServiceActuatorError,
    PluginServiceOperationContext, ProviderCostAcceptanceIssuer, ProviderCostApprovalAuthority,
    ProviderPolicy, ProviderRequestActuator, ProviderResultReceiptIssuer, SecretValue,
    SharedAssetService, WorkerLaunchConfig,
};
use sim_tensor::{RngAlgorithm, RngProfileVersion};
#[cfg(not(test))]
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct SimPluginServices {
    pub boundary: ComponentExecutionBoundary,
    private_worker_executor: Arc<PrivateWorkerPluginExecutor>,
    broker: PluginCapabilityBroker,
    principal_id: String,
    receipt_issuer: Arc<ProviderResultReceiptIssuer>,
    receipt_lifetime: Duration,
    cost_authority: Arc<ProviderCostApprovalAuthority>,
}

impl SimPluginServices {
    pub fn invocation_authority(
        &self,
        host: ComponentHost,
    ) -> Result<Arc<ComponentHostProviderInvocationAuthority>, ComponentHostError> {
        Ok(Arc::new(
            ComponentHostProviderInvocationAuthority::new(
                host,
                self.broker.clone(),
                &self.principal_id,
                self.receipt_issuer.clone(),
                self.receipt_lifetime,
            )
            .map_err(component_boundary_error)?
            .with_cost_authority(self.cost_authority.clone()),
        ))
    }

    #[cfg(not(test))]
    pub fn cost_authority(&self) -> Arc<ProviderCostApprovalAuthority> {
        self.cost_authority.clone()
    }

    pub(crate) fn private_worker_executor(&self) -> Arc<PrivateWorkerPluginExecutor> {
        self.private_worker_executor.clone()
    }
}

#[cfg(not(test))]
pub fn private_worker_services(
    launch: WorkerLaunchConfig,
    assets: SharedAssetService,
    provider_policy: ProviderPolicy,
    profile_seed: u64,
    cx: &mut App,
) -> Result<SimPluginServices, ComponentHostError> {
    let credentials = credential_bridge(cx);
    let clock: Arc<dyn clock::SystemClock> = Arc::new(clock::RealSystemClock);
    let cost_issuer = Arc::new(
        ProviderCostAcceptanceIssuer::generate(clock.utc_now())
            .map_err(component_boundary_error)?,
    );
    let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
        cost_issuer,
        clock.clone(),
    ));
    let broker = PluginCapabilityBroker::new_with_provider_cost_acceptance(
        assets,
        ModelStore::new(ParserLimits::default()).map_err(component_boundary_error)?,
        provider_policy,
        cost_authority
            .verifier()
            .map_err(component_boundary_error)?,
        Arc::new(SimProviderActuator {
            client: cx.http_client(),
            executor: cx.background_executor().clone(),
        }),
        Arc::new(SimCredentialActuator { credentials }),
        clock,
        PluginRngPolicy::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            profile_seed,
        ),
    );
    let receipt_issuer = Arc::new(
        ProviderResultReceiptIssuer::generate(Instant::now()).map_err(component_boundary_error)?,
    );
    let receipt_lifetime = Duration::from_secs(5 * 60);
    let principal_id = launch.profile_id.0.to_string();
    let private_worker_executor = PrivateWorkerPluginExecutor::new_with_provider_authorities(
        launch,
        broker.clone(),
        principal_id.clone(),
        receipt_issuer.clone(),
        receipt_lifetime,
        cost_authority.clone(),
    )?;
    let boundary = ComponentExecutionBoundary::private_worker(private_worker_executor.clone());
    Ok(SimPluginServices {
        boundary,
        private_worker_executor,
        broker,
        principal_id,
        receipt_issuer,
        receipt_lifetime,
        cost_authority,
    })
}

pub fn deny_only_private_worker_services(
    launch: WorkerLaunchConfig,
    assets: SharedAssetService,
    provider_policy: ProviderPolicy,
    profile_seed: u64,
) -> Result<SimPluginServices, ComponentHostError> {
    let clock: Arc<dyn clock::SystemClock> = Arc::new(clock::RealSystemClock);
    let cost_issuer = Arc::new(
        ProviderCostAcceptanceIssuer::generate(clock.utc_now())
            .map_err(component_boundary_error)?,
    );
    let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
        cost_issuer,
        clock.clone(),
    ));
    let broker = PluginCapabilityBroker::new_with_provider_cost_acceptance(
        assets,
        ModelStore::new(ParserLimits::default()).map_err(component_boundary_error)?,
        provider_policy,
        cost_authority
            .verifier()
            .map_err(component_boundary_error)?,
        Arc::new(DenyOnlyProviderActuator),
        Arc::new(DenyOnlyCredentialActuator),
        clock,
        PluginRngPolicy::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            profile_seed,
        ),
    );
    let receipt_issuer = Arc::new(
        ProviderResultReceiptIssuer::generate(Instant::now()).map_err(component_boundary_error)?,
    );
    let receipt_lifetime = Duration::from_secs(5 * 60);
    let principal_id = launch.profile_id.0.to_string();
    let private_worker_executor = PrivateWorkerPluginExecutor::new_with_provider_authorities(
        launch,
        broker.clone(),
        principal_id.clone(),
        receipt_issuer.clone(),
        receipt_lifetime,
        cost_authority.clone(),
    )?;
    let boundary = ComponentExecutionBoundary::private_worker(private_worker_executor.clone());
    Ok(SimPluginServices {
        boundary,
        private_worker_executor,
        broker,
        principal_id,
        receipt_issuer,
        receipt_lifetime,
        cost_authority,
    })
}

struct DenyOnlyProviderActuator;

impl ProviderRequestActuator for DenyOnlyProviderActuator {
    fn execute(
        &self,
        _request: &AuthorizedProviderRequest,
        _secret: Option<&SecretValue>,
        _body: &[u8],
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        Err(PluginServiceActuatorError::new(
            "headless provider actuation is unavailable without a separately owned actuator",
        ))
    }
}

struct DenyOnlyCredentialActuator;

impl CredentialPresenceActuator for DenyOnlyCredentialActuator {
    fn is_present(
        &self,
        _request: &AuthorizedCredentialPresenceRequest,
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<bool, PluginServiceActuatorError> {
        Ok(false)
    }

    fn read_for_provider(
        &self,
        _request: &AuthorizedCredentialPresenceRequest,
        _context: &PluginServiceOperationContext<'_>,
    ) -> Result<Option<SecretValue>, PluginServiceActuatorError> {
        Ok(None)
    }
}

#[cfg(not(test))]
struct CredentialCommand {
    secret_id: String,
    response: async_channel::Sender<Result<Option<Vec<u8>>, String>>,
}

#[cfg(not(test))]
#[derive(Clone)]
struct CredentialBridge {
    commands: async_channel::Sender<CredentialCommand>,
}

#[cfg(not(test))]
impl CredentialBridge {
    fn read(&self, secret_id: &str) -> Result<Option<Vec<u8>>, PluginServiceActuatorError> {
        let (response, result) = async_channel::bounded(1);
        self.commands
            .send_blocking(CredentialCommand {
                secret_id: secret_id.to_owned(),
                response,
            })
            .map_err(|_| PluginServiceActuatorError::new("credential bridge is unavailable"))?;
        result
            .recv_blocking()
            .map_err(|_| PluginServiceActuatorError::new("credential response is unavailable"))?
            .map_err(PluginServiceActuatorError::new)
    }
}

#[cfg(not(test))]
fn credential_bridge(cx: &mut App) -> CredentialBridge {
    let credentials = zed_credentials_provider::global(cx);
    let (commands, receiver) = async_channel::bounded::<CredentialCommand>(64);
    cx.spawn(async move |cx| {
        while let Ok(command) = receiver.recv().await {
            let result = credentials
                .read_credentials(&command.secret_id, cx)
                .await
                .map(|credential| credential.map(|(_, secret)| secret))
                .map_err(|error| error.to_string());
            if command.response.send(result).await.is_err() {
                continue;
            }
        }
    })
    .detach();
    CredentialBridge { commands }
}

#[cfg(not(test))]
struct SimCredentialActuator {
    credentials: CredentialBridge,
}

#[cfg(not(test))]
impl CredentialPresenceActuator for SimCredentialActuator {
    fn is_present(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<bool, PluginServiceActuatorError> {
        check_context(context)?;
        let present = self
            .credentials
            .read(request.secret_id().as_str())?
            .is_some();
        check_context(context)?;
        Ok(present)
    }

    fn read_for_provider(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Option<SecretValue>, PluginServiceActuatorError> {
        check_context(context)?;
        let secret = self
            .credentials
            .read(request.secret_id().as_str())?
            .map(SecretValue::new);
        check_context(context)?;
        Ok(secret)
    }
}

#[cfg(not(test))]
struct SimProviderActuator {
    client: Arc<dyn HttpClient>,
    executor: BackgroundExecutor,
}

#[cfg(not(test))]
impl ProviderRequestActuator for SimProviderActuator {
    fn execute(
        &self,
        request: &AuthorizedProviderRequest,
        secret: Option<&SecretValue>,
        body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        check_context(context)?;
        let mut builder = Request::builder()
            .method(http::Method::POST)
            .uri(request.endpoint())
            .header(http::header::CONTENT_TYPE, "application/octet-stream")
            .header("x-zed-sim-provider", request.provider());
        if let Some(idempotency_key) = request.idempotency_key_sha256() {
            builder = builder.header("idempotency-key", idempotency_key);
        }
        if let Some(secret) = secret {
            let authorization = secret.expose_to(|bytes| {
                let mut value = Vec::with_capacity(bytes.len().saturating_add(7));
                value.extend_from_slice(b"Bearer ");
                value.extend_from_slice(bytes);
                http::HeaderValue::from_bytes(&value)
            });
            builder = builder.header(
                http::header::AUTHORIZATION,
                authorization.map_err(|_| {
                    PluginServiceActuatorError::new(
                        "provider credential cannot be represented as an authorization header",
                    )
                })?,
            );
        } else if request.secret_id().is_some() {
            return Err(PluginServiceActuatorError::new(
                "authorized provider credential is unavailable",
            ));
        }
        let provider_request = builder
            .body(AsyncBody::from(body.to_vec()))
            .map_err(component_actuator_error)?;
        let mut response = block_on_provider_operation(
            self.client.send(provider_request),
            context,
            &self.executor,
        )?;
        if !response.status().is_success() {
            return Err(PluginServiceActuatorError::new(format!(
                "provider returned HTTP status {}",
                response.status().as_u16()
            )));
        }
        let maximum = context.maximum_response_bytes();
        let read_limit = maximum.saturating_add(1);
        let mut bytes = Vec::new();
        block_on_provider_operation(
            response.body_mut().take(read_limit).read_to_end(&mut bytes),
            context,
            &self.executor,
        )?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum {
            return Err(PluginServiceActuatorError::new(
                "provider response exceeds the invocation bound",
            ));
        }
        check_context(context)?;
        Ok(bytes)
    }
}

#[cfg(not(test))]
fn block_on_provider_operation<T, E>(
    operation: impl Future<Output = Result<T, E>>,
    context: &PluginServiceOperationContext<'_>,
    executor: &BackgroundExecutor,
) -> Result<T, PluginServiceActuatorError>
where
    E: std::fmt::Display,
{
    smol::block_on(smol::future::race(
        async { operation.await.map_err(component_actuator_error) },
        async {
            loop {
                context.check_active().map_err(component_actuator_error)?;
                executor.timer(Duration::from_millis(5)).await;
            }
        },
    ))
}

#[cfg(not(test))]
fn check_context(
    context: &PluginServiceOperationContext<'_>,
) -> Result<(), PluginServiceActuatorError> {
    context
        .check_active()
        .map_err(|error| PluginServiceActuatorError::new(error.to_string()))
}

fn component_boundary_error(error: impl std::fmt::Display) -> ComponentHostError {
    ComponentHostError::ExecutionBoundary(error.to_string())
}

#[cfg(not(test))]
fn component_actuator_error(error: impl std::fmt::Display) -> PluginServiceActuatorError {
    PluginServiceActuatorError::new(error.to_string())
}
