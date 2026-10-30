use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    net::IpAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

pub mod headless;
pub mod http;
pub mod security;
pub mod services;
pub mod transport;
pub mod websocket;

pub use headless::*;
pub use http::*;
pub use services::*;
pub use transport::*;
pub use websocket::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApiVersion {
    pub protocol: u16,
}

impl Default for ApiVersion {
    fn default() -> Self {
        Self {
            protocol: sim_types::NATIVE_PROTOCOL_VERSION,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HostRequestContext {
    peer_address: IpAddr,
    forwarded_for: Option<IpAddr>,
    transport_tls: bool,
    now_epoch_seconds: u64,
}

impl HostRequestContext {
    pub fn embedded_loopback(
        peer_address: IpAddr,
        now_epoch_seconds: u64,
    ) -> Result<Self, NativeApiHostError> {
        if !peer_address.is_loopback() {
            return Err(NativeApiHostError::InvalidConfiguration(
                "embedded native API requests must use a loopback peer identity".into(),
            ));
        }
        Ok(Self {
            peer_address,
            forwarded_for: None,
            transport_tls: false,
            now_epoch_seconds,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum DurableHttpBody {
    Empty,
    Bytes(Vec<u8>),
    Json(Value),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct DurableHttpResponse {
    status: u16,
    content_type: String,
    headers: BTreeMap<String, String>,
    body: DurableHttpBody,
}

impl DurableHttpResponse {
    fn capture(response: &HttpResponse) -> Result<Option<Self>, NativeApiHostError> {
        let body = match &response.body {
            HttpBody::Empty => DurableHttpBody::Empty,
            HttpBody::Bytes(bytes) => DurableHttpBody::Bytes(bytes.to_vec()),
            HttpBody::Json(value) => DurableHttpBody::Json(value.clone()),
            HttpBody::Stream(_) => return Ok(None),
        };
        Ok(Some(Self {
            status: response.status,
            content_type: response.content_type.clone(),
            headers: response.headers.clone(),
            body,
        }))
    }

    fn encode(&self) -> Result<Vec<u8>, NativeApiHostError> {
        serde_json::to_vec(self).map_err(|error| NativeApiHostError::Persistence(error.to_string()))
    }

    fn decode(bytes: &[u8]) -> Result<Self, NativeApiHostError> {
        serde_json::from_slice(bytes)
            .map_err(|error| NativeApiHostError::Persistence(error.to_string()))
    }

    fn into_response(self) -> HttpResponse {
        HttpResponse {
            status: self.status,
            content_type: self.content_type,
            headers: self.headers,
            body: match self.body {
                DurableHttpBody::Empty => HttpBody::Empty,
                DurableHttpBody::Bytes(bytes) => HttpBody::Bytes(bytes.into()),
                DurableHttpBody::Json(value) => HttpBody::Json(value),
            },
        }
    }
}

pub struct NativeApiHost<S>
where
    S: NativeHttpServices,
{
    version: ApiVersion,
    profile_id: String,
    security: security::ApiSecurityGate,
    http: NativeHttpRouter<S>,
    websocket: Mutex<NativeWebSocketEventBus>,
    websocket_limits: WebSocketLimits,
    status_sequence: AtomicU64,
    idempotency: Mutex<security::IdempotencyLedger>,
    idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    idempotency_maximum_records: usize,
    idempotency_maximum_response_bytes: usize,
}

pub struct NativeRuntimeApiHost {
    host: Arc<NativeApiHost<NativeRuntimeHttpServices>>,
    controller: Arc<dyn sim_runtime::ExecutionController>,
    profile_id: sim_runtime::ProfileId,
    presentation: sim_runtime::SharedExecutionPresentationService,
    event_bridge_diagnostic: Arc<Mutex<Option<String>>>,
    provider_deployment: Option<NativeProviderDeploymentIdentity>,
    _event_bridge: smol::Task<Result<(), NativeApiHostError>>,
}

impl NativeRuntimeApiHost {
    #[allow(clippy::too_many_arguments)]
    pub fn with_registry_bundle(
        bundle: Arc<sim_runtime::NativeExecutionRegistryBundle>,
        candidate_identity: &extension_host::ComponentInventoryCandidateIdentity,
        presentation: sim_runtime::SharedExecutionPresentationService,
        controller: Arc<dyn sim_runtime::ExecutionController>,
        event_bus: &sim_runtime::ExecutionEventBus,
        assets: Option<sim_runtime::SharedAssetService>,
        http_limits: HttpLimits,
        websocket_limits: WebSocketLimits,
        security_config: security::ApiSecurityConfig,
        permission_policy: Arc<sim_runtime::PermissionPolicy>,
        idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<Self, NativeApiHostError> {
        let mut services = NativeRuntimeHttpServices::from_registry_bundle(
            &bundle,
            presentation.clone(),
            controller.clone(),
        )
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        if let Some(assets) = assets {
            let asset_authorization =
                sim_runtime::authorize_native_api_asset_reader(&permission_policy)
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
            services = services
                .with_assets(assets, asset_authorization)
                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        }
        let mut runtime = Self::from_services(
            bundle.profile_id(),
            presentation,
            controller,
            event_bus,
            services,
            http_limits,
            websocket_limits,
            security_config,
            permission_policy,
            idempotency_store,
        )?;
        runtime.provider_deployment = Some(NativeProviderDeploymentIdentity::from_registry_bundle(
            &bundle,
            candidate_identity,
        )?);
        Ok(runtime)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn native_image(
        profile_id: sim_runtime::ProfileId,
        presentation: sim_runtime::SharedExecutionPresentationService,
        controller: Arc<dyn sim_runtime::ExecutionController>,
        event_bus: &sim_runtime::ExecutionEventBus,
        assets: Option<sim_runtime::SharedAssetService>,
        http_limits: HttpLimits,
        websocket_limits: WebSocketLimits,
        security_config: security::ApiSecurityConfig,
        permission_policy: Arc<sim_runtime::PermissionPolicy>,
        idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<Self, NativeApiHostError> {
        let registry = sim_runtime::generated_native_node_registry_projection(None)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        Self::with_registry(
            profile_id,
            presentation,
            controller,
            event_bus,
            assets,
            registry,
            http_limits,
            websocket_limits,
            security_config,
            permission_policy,
            idempotency_store,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_registry(
        profile_id: sim_runtime::ProfileId,
        presentation: sim_runtime::SharedExecutionPresentationService,
        controller: Arc<dyn sim_runtime::ExecutionController>,
        event_bus: &sim_runtime::ExecutionEventBus,
        assets: Option<sim_runtime::SharedAssetService>,
        registry: sim_runtime::NativeNodeRegistry,
        http_limits: HttpLimits,
        websocket_limits: WebSocketLimits,
        security_config: security::ApiSecurityConfig,
        permission_policy: Arc<sim_runtime::PermissionPolicy>,
        idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<Self, NativeApiHostError> {
        let mut services = NativeRuntimeHttpServices::new(
            profile_id,
            presentation.clone(),
            controller.clone(),
            registry,
        )
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        if let Some(assets) = assets {
            let asset_authorization =
                sim_runtime::authorize_native_api_asset_reader(&permission_policy)
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
            services = services
                .with_assets(assets, asset_authorization)
                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        }
        Self::from_services(
            profile_id,
            presentation,
            controller,
            event_bus,
            services,
            http_limits,
            websocket_limits,
            security_config,
            permission_policy,
            idempotency_store,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_services(
        profile_id: sim_runtime::ProfileId,
        presentation: sim_runtime::SharedExecutionPresentationService,
        controller: Arc<dyn sim_runtime::ExecutionController>,
        event_bus: &sim_runtime::ExecutionEventBus,
        services: NativeRuntimeHttpServices,
        http_limits: HttpLimits,
        websocket_limits: WebSocketLimits,
        security_config: security::ApiSecurityConfig,
        permission_policy: Arc<sim_runtime::PermissionPolicy>,
        idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<Self, NativeApiHostError> {
        let capabilities = services
            .http_capabilities()
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let host = Arc::new(NativeApiHost::new(
            profile_id.0.to_string(),
            Arc::new(services),
            http_limits,
            capabilities,
            websocket_limits,
            security_config,
            permission_policy,
            idempotency_store,
        )?);
        let receiver = event_bus.subscribe();
        let weak_host = Arc::downgrade(&host);
        let bridge_presentation = presentation.clone();
        let event_bridge_diagnostic = Arc::new(Mutex::new(None));
        let bridge_diagnostic = event_bridge_diagnostic.clone();
        let event_bridge = smol::spawn(async move {
            let mut execution_clients = BTreeMap::new();
            while let Ok(event) = receiver.recv().await {
                let Some(host) = weak_host.upgrade() else {
                    break;
                };
                let terminal = matches!(
                    &event.kind,
                    sim_runtime::AttemptEventKind::Succeeded
                        | sim_runtime::AttemptEventKind::Failed { .. }
                        | sim_runtime::AttemptEventKind::Cancelled
                        | sim_runtime::AttemptEventKind::Interrupted { .. }
                        | sim_runtime::AttemptEventKind::RecoveryInterrupted { .. }
                );
                let result = (|| {
                    let client_id = {
                        if let std::collections::btree_map::Entry::Vacant(entry) =
                            execution_clients.entry(event.attempt_id.0)
                        {
                            let client_id = bridge_presentation
                                .persisted_attempts(event.profile_id)
                                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?
                                .into_iter()
                                .find(|attempt| attempt.record.attempt_id == event.attempt_id)
                                .and_then(|attempt| attempt.plan)
                                .and_then(|plan| plan.client_id);
                            if let Some(client_id) = client_id {
                                entry.insert(client_id);
                            }
                        }
                        execution_clients.get(&event.attempt_id.0).cloned()
                    };
                    let mut wire_event = event.clone();
                    if let Some(client_id) = client_id {
                        let mut data = match wire_event.data.take() {
                            Some(Value::Object(data)) => data,
                            Some(data) => {
                                serde_json::Map::from_iter([("runtime_data".into(), data)])
                            }
                            None => serde_json::Map::new(),
                        };
                        data.insert("client_id".into(), Value::String(client_id));
                        wire_event.data = Some(Value::Object(data));
                    }
                    host.publish_execution_event(wire_event)?;
                    host.publish_status_projection()?;
                    Ok::<_, NativeApiHostError>(())
                })();
                if terminal {
                    execution_clients.remove(&event.attempt_id.0);
                }
                if let Err(error) = result {
                    let mut diagnostic = bridge_diagnostic
                        .lock()
                        .map_err(|_| NativeApiHostError::StateUnavailable)?;
                    *diagnostic = Some(error.to_string());
                }
            }
            Ok(())
        });
        Ok(Self {
            host,
            controller,
            profile_id,
            presentation,
            event_bridge_diagnostic,
            provider_deployment: None,
            _event_bridge: event_bridge,
        })
    }

    pub fn host(&self) -> Arc<NativeApiHost<NativeRuntimeHttpServices>> {
        self.host.clone()
    }

    pub fn presentation(&self) -> sim_runtime::SharedExecutionPresentationService {
        self.presentation.clone()
    }

    pub fn server_config(&self, bind_address: std::net::SocketAddr) -> NativeApiServerConfig {
        let profile_id = self.profile_id;
        let presentation = self.presentation.clone();
        let mut config = NativeApiServerConfig::new(bind_address);
        config.reconnect_projection = Arc::new(move |_, client_id| {
            native_reconnect_projection(&presentation, profile_id, client_id)
        });
        config
    }

    pub fn event_bridge_diagnostic(&self) -> Result<Option<String>, NativeApiHostError> {
        self.event_bridge_diagnostic
            .lock()
            .map(|diagnostic| diagnostic.clone())
            .map_err(|_| NativeApiHostError::StateUnavailable)
    }

    pub fn deployment_identity_sha256(&self) -> Option<&str> {
        self.provider_deployment
            .as_ref()
            .map(NativeProviderDeploymentIdentity::execution_bundle_identity_sha256)
    }

    pub fn provider_deployment(&self) -> Option<&NativeProviderDeploymentIdentity> {
        self.provider_deployment.as_ref()
    }

    pub fn shutdown(&self, reason: impl Into<String>) -> Result<(), NativeApiHostError> {
        self.host.shutdown(reason)?;
        self.controller
            .shutdown()
            .map_err(|failure| NativeApiHostError::Runtime(failure.message))
    }
}

fn native_reconnect_projection(
    presentation: &sim_runtime::SharedExecutionPresentationService,
    profile_id: sim_runtime::ProfileId,
    client_id: &ClientId,
) -> Result<ReconnectProjection, NativeApiHostError> {
    let (snapshot, persisted_attempts) = presentation
        .snapshot_with_persisted_attempts(profile_id)
        .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
    let client_attempts = persisted_attempts
        .into_iter()
        .filter_map(|attempt| {
            let belongs_to_client = attempt
                .plan
                .as_ref()
                .and_then(|plan| plan.client_id.as_deref())
                == Some(client_id.as_str());
            belongs_to_client.then_some(attempt.record.attempt_id)
        })
        .collect::<Vec<_>>();
    let queue_remaining = snapshot
        .queue
        .len()
        .saturating_add(
            snapshot
                .attempts
                .iter()
                .filter(|attempt| {
                    matches!(
                        attempt.state,
                        sim_runtime::AttemptState::Running | sim_runtime::AttemptState::Cancelling
                    )
                })
                .count(),
        )
        .try_into()
        .map_err(|_| NativeApiHostError::Runtime("queue size cannot fit the wire schema".into()))?;
    let mut current_execution = Vec::new();
    let mut history_reconciliation = Vec::new();
    for attempt in snapshot
        .attempts
        .iter()
        .filter(|attempt| client_attempts.contains(&attempt.attempt_id))
    {
        let prompt_id = attempt.prompt_id.0.to_string();
        let attempt_id = attempt.attempt_id.0.to_string();
        let association = |node_id: Option<String>| EventAssociation {
            prompt_id: Some(prompt_id.clone()),
            node_id,
            attempt_id: Some(attempt_id.clone()),
        };
        if matches!(
            attempt.state,
            sim_runtime::AttemptState::Running | sim_runtime::AttemptState::Cancelling
        ) {
            current_execution.push(ReconnectJsonEvent {
                event_type: "execution_start".into(),
                data: json!({
                    "prompt_id": prompt_id,
                    "timestamp": attempt.created_at.timestamp_millis(),
                }),
                association: association(None),
            });
            let node_id = attempt
                .progress
                .as_ref()
                .and_then(|progress| progress.node_id.as_ref())
                .map(|node_id| node_id.0.clone());
            if let Some(node_id) = node_id.as_ref() {
                current_execution.push(ReconnectJsonEvent {
                    event_type: "executing".into(),
                    data: json!({
                        "node": node_id,
                        "display_node": node_id,
                        "prompt_id": prompt_id,
                    }),
                    association: association(Some(node_id.clone())),
                });
            }
            let nodes = attempt
                .node_progress
                .iter()
                .map(|(node_id, progress)| {
                    let state = if progress.total > 0 && progress.completed >= progress.total {
                        "finished"
                    } else {
                        "running"
                    };
                    (
                        node_id.0.clone(),
                        progress_state_node(
                            &prompt_id,
                            &node_id.0,
                            progress.completed,
                            progress.total,
                            state,
                        ),
                    )
                })
                .collect::<serde_json::Map<_, _>>();
            current_execution.push(ReconnectJsonEvent {
                event_type: "progress_state".into(),
                data: json!({"prompt_id": prompt_id, "nodes": nodes}),
                association: association(node_id),
            });
            continue;
        }
        for output in &attempt.outputs {
            let node_id = output.node_id.0.clone();
            let output = serde_json::to_value(output)
                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
            history_reconciliation.push(ReconnectJsonEvent {
                event_type: "executed".into(),
                data: json!({
                    "node": node_id,
                    "display_node": node_id,
                    "output": output,
                    "prompt_id": prompt_id,
                }),
                association: association(Some(node_id)),
            });
        }
        let finished_at = attempt.finished_at.unwrap_or(attempt.created_at);
        let terminal = match attempt.state {
            sim_runtime::AttemptState::Succeeded => ReconnectJsonEvent {
                event_type: "execution_success".into(),
                data: json!({
                    "prompt_id": prompt_id,
                    "timestamp": finished_at.timestamp_millis(),
                }),
                association: association(None),
            },
            sim_runtime::AttemptState::Failed => {
                let failure = attempt.failure.as_ref();
                let node_id = failure
                    .and_then(|failure| failure.node_id.as_ref())
                    .map(|node_id| node_id.0.clone())
                    .unwrap_or_default();
                let code = failure
                    .map(|failure| failure.code.as_str())
                    .unwrap_or("native_execution_failed");
                let message = failure
                    .map(|failure| failure.message.as_str())
                    .unwrap_or("native execution failed without retained details");
                ReconnectJsonEvent {
                    event_type: "execution_error".into(),
                    data: json!({
                        "prompt_id": prompt_id,
                        "node_id": node_id,
                        "node_type": code,
                        "exception_message": message,
                        "exception_type": code,
                        "executed": [],
                        "traceback": [],
                        "current_inputs": {},
                        "current_outputs": {},
                        "timestamp": finished_at.timestamp_millis(),
                    }),
                    association: association(Some(node_id)),
                }
            }
            sim_runtime::AttemptState::Cancelled | sim_runtime::AttemptState::Interrupted => {
                ReconnectJsonEvent {
                    event_type: "execution_interrupted".into(),
                    data: json!({
                        "prompt_id": prompt_id,
                        "node_id": "",
                        "node_type": "native",
                        "executed": [],
                        "timestamp": finished_at.timestamp_millis(),
                    }),
                    association: association(None),
                }
            }
            sim_runtime::AttemptState::Queued
            | sim_runtime::AttemptState::Running
            | sim_runtime::AttemptState::Cancelling => continue,
        };
        history_reconciliation.push(terminal);
    }
    Ok(ReconnectProjection {
        queue_remaining,
        current_execution,
        history_reconciliation,
    })
}

impl<S> NativeApiHost<S>
where
    S: NativeHttpServices,
{
    pub fn new(
        profile_id: impl Into<String>,
        services: Arc<S>,
        http_limits: HttpLimits,
        http_capabilities: HttpCapabilities,
        websocket_limits: WebSocketLimits,
        security_config: security::ApiSecurityConfig,
        permission_policy: Arc<sim_runtime::PermissionPolicy>,
        idempotency_store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<Self, NativeApiHostError> {
        let profile_id = profile_id.into();
        if profile_id.is_empty() {
            return Err(NativeApiHostError::InvalidConfiguration(
                "native API profile identity cannot be empty".into(),
            ));
        }
        if permission_policy.profile_id() != profile_id {
            return Err(NativeApiHostError::InvalidConfiguration(
                "native API permission policy belongs to another runtime profile".into(),
            ));
        }
        let idempotency_maximum_records = http_limits.idempotency_capacity;
        let idempotency_maximum_response_bytes = http_limits.maximum_response_bytes;
        let idempotency = match idempotency_store
            .load()
            .map_err(NativeApiHostError::from_security)?
        {
            Some(snapshot) => security::IdempotencyLedger::restore(
                snapshot,
                idempotency_maximum_records,
                idempotency_maximum_response_bytes,
            )
            .map_err(NativeApiHostError::from_security)?,
            None => security::IdempotencyLedger::new(
                profile_id.clone(),
                idempotency_maximum_records,
                idempotency_maximum_response_bytes,
            )
            .map_err(NativeApiHostError::from_security)?,
        };
        if idempotency.profile_id() != profile_id {
            return Err(NativeApiHostError::InvalidConfiguration(
                "idempotency snapshot belongs to another runtime profile".into(),
            ));
        }
        Ok(Self {
            version: ApiVersion::default(),
            profile_id,
            security: security::ApiSecurityGate::new(security_config, permission_policy)
                .map_err(NativeApiHostError::from_security)?,
            http: NativeHttpRouter::new(services, http_limits, http_capabilities)
                .map_err(NativeApiHostError::from_http)?,
            websocket: Mutex::new(
                NativeWebSocketEventBus::new(websocket_limits.clone())
                    .map_err(NativeApiHostError::from_websocket)?,
            ),
            websocket_limits,
            status_sequence: AtomicU64::new(1),
            idempotency: Mutex::new(idempotency),
            idempotency_store,
            idempotency_maximum_records,
            idempotency_maximum_response_bytes,
        })
    }

    pub fn version(&self) -> &ApiVersion {
        &self.version
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn security_config(&self) -> &security::ApiSecurityConfig {
        self.security.config()
    }

    pub fn websocket_limits(&self) -> &WebSocketLimits {
        &self.websocket_limits
    }

    pub fn idempotency_snapshot(
        &self,
    ) -> Result<security::IdempotencySnapshot, NativeApiHostError> {
        self.idempotency
            .lock()
            .map(|ledger| ledger.snapshot())
            .map_err(|_| NativeApiHostError::StateUnavailable)
    }

    pub fn route_http(
        &self,
        mut request: HttpRequest,
        context: HostRequestContext,
    ) -> Result<HttpResponse, NativeApiHostError> {
        let matched = match_http_route(request.method, &request.path)
            .map_err(|error| NativeApiHostError::Http {
                status: 500,
                code: "http_catalog_invalid".into(),
                message: error.to_string(),
            })?
            .ok_or_else(|| NativeApiHostError::Http {
                status: 404,
                code: "route_not_found".into(),
                message: "native route not found".into(),
            })?;
        let descriptor = http_route_catalog()
            .map_err(|error| NativeApiHostError::Http {
                status: 500,
                code: "http_catalog_invalid".into(),
                message: error.to_string(),
            })?
            .iter()
            .find(|descriptor| descriptor.feature_id() == matched.requested_feature_id)
            .ok_or_else(|| NativeApiHostError::Http {
                status: 500,
                code: "catalog_descriptor_missing".into(),
                message: "matched route descriptor is missing".into(),
            })?;
        http::validate_request_headers(&request, &matched)
            .map_err(NativeApiHostError::from_http)?;
        let plugin_authority = plugin_route_request(&request, &self.profile_id)?;
        let mutation = descriptor.is_mutation();
        let wire_json_body = serde_json::from_slice::<Value>(&request.body).ok();
        let durable_identity = mutation
            .then(|| {
                http::mutation_identity(
                    descriptor,
                    &request,
                    wire_json_body.as_ref(),
                    &matched,
                    None,
                )
            })
            .filter(|identity| *identity != MutationIdentity::Untracked);
        let header_bytes = request
            .headers
            .iter()
            .map(|(name, values)| {
                name.len() + values.iter().map(|value| value.len()).sum::<usize>()
            })
            .sum();
        let required_scope = self
            .security
            .config()
            .require_authentication
            .then(|| if mutation { "api:write" } else { "api:read" }.to_owned());
        let authorization = request_header(&request, "authorization").map(str::to_owned);
        let origin = request_header(&request, "origin").map(str::to_owned);
        let security_context = security::RequestSecurityContext {
            method: http_method_name(request.method).to_owned(),
            canonical_path: matched.canonical_path.clone(),
            body_bytes: request.body.len(),
            header_bytes,
            header_count: request.headers.values().map(Vec::len).sum(),
            origin,
            authorization,
            peer_address: context.peer_address,
            forwarded_for: context.forwarded_for,
            transport_tls: context.transport_tls,
            required_scope,
            plugin: plugin_authority.clone(),
            mutation_identity: durable_identity.clone(),
            now_epoch_seconds: context.now_epoch_seconds,
        };
        let authorization = self
            .security
            .authorize(&security_context)
            .map_err(NativeApiHostError::from_security)?;
        let allowed_origin = authorization.allowed_origin.as_deref();
        let native_authority = NativeRequestAuthority {
            profile_id: self.profile_id.clone(),
            principal: authorization.principal.identity.clone(),
            scopes: authorization.principal.scopes.clone(),
            plugin_id: plugin_authority
                .as_ref()
                .map(|plugin| plugin.plugin_id.clone()),
            plugin_digest: plugin_authority.map(|plugin| plugin.plugin_digest),
        };
        if matched.canonical_path == "/prompt" && request.method == sim_types::HttpMethod::Post {
            scope_prompt_client_id(&mut request, &native_authority)?;
        }

        let Some(identity) = durable_identity else {
            let response = self
                .http
                .route_authorized(request, Some(native_authority))
                .map_err(NativeApiHostError::from_http)?;
            if mutation && (200..300).contains(&response.status) {
                self.publish_status_projection()?;
            }
            return Ok(apply_cors(response, allowed_origin));
        };
        let (client_key, client_operation_id) = match identity {
            MutationIdentity::IdempotencyKey(key) => (key.clone(), key),
            MutationIdentity::DurableAttempt(operation_id) => (operation_id.clone(), operation_id),
            MutationIdentity::Untracked => return Err(NativeApiHostError::StateUnavailable),
        };
        let key = scoped_idempotency_key(&client_key, &native_authority);
        let operation_id = scoped_operation_id(
            &client_operation_id,
            &matched.canonical_feature_id,
            &native_authority,
        );
        let request_digest =
            stable_request_digest(&request, &matched.canonical_feature_id, &native_authority);
        let decision = {
            let mut ledger = self
                .idempotency
                .lock()
                .map_err(|_| NativeApiHostError::StateUnavailable)?;
            let previous = ledger.snapshot();
            let decision = ledger
                .begin(
                    key.clone(),
                    operation_id,
                    request_digest,
                    context.now_epoch_seconds,
                )
                .map_err(NativeApiHostError::from_security)?;
            if decision == security::IdempotencyDecision::Begin
                && let Err(error) = self.idempotency_store.save(&ledger.snapshot())
            {
                *ledger = security::IdempotencyLedger::restore(
                    previous,
                    self.idempotency_maximum_records,
                    self.idempotency_maximum_response_bytes,
                )
                .map_err(NativeApiHostError::from_security)?;
                return Err(NativeApiHostError::from_security(error));
            }
            decision
        };
        let decision = match decision {
            security::IdempotencyDecision::Replay {
                status: _,
                response_body,
            } => {
                return Ok(apply_cors(
                    DurableHttpResponse::decode(&response_body)?.into_response(),
                    allowed_origin,
                ));
            }
            security::IdempotencyDecision::Pending { operation_id } => {
                return Ok(apply_cors(
                    HttpResponse::json(
                        409,
                        json!({
                            "error": {
                                "code": "mutation_in_progress",
                                "message": "the native mutation is still in progress",
                                "operation_id": operation_id,
                            }
                        }),
                    ),
                    allowed_origin,
                ));
            }
            security::IdempotencyDecision::Reconcile { operation_id } => {
                match self
                    .http
                    .reconcile_authorized(request.clone(), native_authority.clone())
                    .map_err(NativeApiHostError::from_http)?
                {
                    RoutedMutationReconciliation::Committed(response) => {
                        let captured =
                            DurableHttpResponse::capture(&response)?.ok_or_else(|| {
                                NativeApiHostError::Persistence(
                                    "a reconciled mutation response must be replayable".into(),
                                )
                            })?;
                        let status = captured.status;
                        let response_body = captured.encode()?;
                        let state = if (200..400).contains(&status) {
                            security::IdempotencyState::Committed {
                                status,
                                response_body,
                            }
                        } else {
                            security::IdempotencyState::Failed {
                                status,
                                response_body,
                            }
                        };
                        let mut ledger = self
                            .idempotency
                            .lock()
                            .map_err(|_| NativeApiHostError::StateUnavailable)?;
                        let previous = ledger.snapshot();
                        ledger
                            .complete(&key, state, context.now_epoch_seconds)
                            .map_err(NativeApiHostError::from_security)?;
                        if let Err(error) = self.idempotency_store.save(&ledger.snapshot()) {
                            *ledger = security::IdempotencyLedger::restore(
                                previous,
                                self.idempotency_maximum_records,
                                self.idempotency_maximum_response_bytes,
                            )
                            .map_err(NativeApiHostError::from_security)?;
                            return Err(NativeApiHostError::from_security(error));
                        }
                        return Ok(apply_cors(response, allowed_origin));
                    }
                    RoutedMutationReconciliation::NotApplied => {
                        let mut ledger = self
                            .idempotency
                            .lock()
                            .map_err(|_| NativeApiHostError::StateUnavailable)?;
                        let previous = ledger.snapshot();
                        ledger
                            .reopen_after_not_applied(&key, context.now_epoch_seconds)
                            .map_err(NativeApiHostError::from_security)?;
                        if let Err(error) = self.idempotency_store.save(&ledger.snapshot()) {
                            *ledger = security::IdempotencyLedger::restore(
                                previous,
                                self.idempotency_maximum_records,
                                self.idempotency_maximum_response_bytes,
                            )
                            .map_err(NativeApiHostError::from_security)?;
                            return Err(NativeApiHostError::from_security(error));
                        }
                        security::IdempotencyDecision::Begin
                    }
                    RoutedMutationReconciliation::Unresolved { reason } => {
                        return Ok(apply_cors(
                            HttpResponse::json(
                                409,
                                json!({
                                    "error": {
                                        "code": "ambiguous_mutation_reconciliation_required",
                                        "message": "the mutation may already have been applied; canonical state could not resolve it",
                                        "operation_id": operation_id,
                                        "reason": reason,
                                    }
                                }),
                            ),
                            allowed_origin,
                        ));
                    }
                }
            }
            security::IdempotencyDecision::Begin => security::IdempotencyDecision::Begin,
        };
        match decision {
            security::IdempotencyDecision::Begin => {
                let response = self
                    .http
                    .route_authorized(request, Some(native_authority))
                    .unwrap_or_else(HttpRouteError::into_response);
                let status_projection_error = if (200..300).contains(&response.status) {
                    self.publish_status_projection().err()
                } else {
                    None
                };
                let captured = DurableHttpResponse::capture(&response)?;
                let state = match (captured, status_projection_error.as_ref()) {
                    (_, Some(_)) => security::IdempotencyState::Interrupted,
                    (Some(captured), None) => {
                        let status = captured.status;
                        let response_body = captured.encode()?;
                        if status == 504 {
                            security::IdempotencyState::Interrupted
                        } else if (200..400).contains(&status) {
                            security::IdempotencyState::Committed {
                                status,
                                response_body,
                            }
                        } else {
                            security::IdempotencyState::Failed {
                                status,
                                response_body,
                            }
                        }
                    }
                    (None, None) => security::IdempotencyState::Interrupted,
                };
                let mut ledger = self
                    .idempotency
                    .lock()
                    .map_err(|_| NativeApiHostError::StateUnavailable)?;
                let prepared = ledger.snapshot();
                ledger
                    .complete(&key, state, context.now_epoch_seconds)
                    .map_err(NativeApiHostError::from_security)?;
                if let Err(error) = self.idempotency_store.save(&ledger.snapshot()) {
                    *ledger = security::IdempotencyLedger::restore(
                        prepared,
                        self.idempotency_maximum_records,
                        self.idempotency_maximum_response_bytes,
                    )
                    .map_err(NativeApiHostError::from_security)?;
                    ledger
                        .complete(
                            &key,
                            security::IdempotencyState::Interrupted,
                            context.now_epoch_seconds,
                        )
                        .map_err(NativeApiHostError::from_security)?;
                    return Err(NativeApiHostError::from_security(error));
                }
                if let Some(error) = status_projection_error {
                    return Err(error);
                }
                Ok(apply_cors(response, allowed_origin))
            }
            security::IdempotencyDecision::Replay { .. }
            | security::IdempotencyDecision::Pending { .. }
            | security::IdempotencyDecision::Reconcile { .. } => {
                Err(NativeApiHostError::StateUnavailable)
            }
        }
    }

    pub fn handle_http(&self, request: HttpRequest, context: HostRequestContext) -> HttpResponse {
        let allowed_origin = request_header(&request, "origin")
            .filter(|origin| self.security.config().allowed_origins.contains(*origin))
            .map(str::to_owned);
        let response = self
            .route_http(request, context)
            .unwrap_or_else(NativeApiHostError::into_http_response);
        apply_cors(response, allowed_origin.as_deref())
    }

    pub fn handle_preflight(
        &self,
        path: &str,
        headers: &BTreeMap<String, Vec<String>>,
        context: HostRequestContext,
    ) -> HttpResponse {
        let error = |status, code: &str, message: &str| {
            HttpResponse::json(status, json!({"error": {"code": code, "message": message}}))
        };
        let preflight = security::PreflightSecurityContext {
            canonical_path: path.to_owned(),
            header_bytes: headers
                .iter()
                .map(|(name, values)| name.len() + values.iter().map(String::len).sum::<usize>())
                .sum(),
            header_count: headers.values().map(Vec::len).sum(),
            origin: map_header(headers, "origin").map(str::to_owned),
            peer_address: context.peer_address,
            forwarded_for: context.forwarded_for,
            transport_tls: context.transport_tls,
        };
        let authorized = match self.security.authorize_preflight(&preflight) {
            Ok(authorized) => authorized,
            Err(error) => return NativeApiHostError::from_security(error).into_http_response(),
        };
        let origin = authorized.allowed_origin.as_str();
        let Some(requested_method) = map_header(headers, "access-control-request-method") else {
            return apply_cors(
                error(
                    400,
                    "preflight_method_required",
                    "CORS preflight requires Access-Control-Request-Method",
                ),
                Some(origin),
            );
        };
        let Some(method) = parse_http_method(requested_method) else {
            return apply_cors(
                error(
                    405,
                    "method_not_allowed",
                    "requested method is not supported",
                ),
                Some(origin),
            );
        };
        match match_http_route(method, path) {
            Ok(Some(_)) => {}
            Ok(None) => {
                return apply_cors(
                    error(404, "route_not_found", "native route not found"),
                    Some(origin),
                );
            }
            Err(catalog_error) => {
                return apply_cors(
                    error(500, "http_catalog_invalid", &catalog_error.to_string()),
                    Some(origin),
                );
            }
        }
        const ALLOWED_HEADERS: [&str; 6] = [
            "authorization",
            "content-type",
            "idempotency-key",
            "range",
            "x-operation-id",
            "x-request-id",
        ];
        let requested_headers = map_header(headers, "access-control-request-headers")
            .map(|value| {
                value
                    .split(',')
                    .map(|header| header.trim().to_ascii_lowercase())
                    .filter(|header| !header.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if requested_headers
            .iter()
            .any(|header| !ALLOWED_HEADERS.contains(&header.as_str()))
        {
            return apply_cors(
                error(
                    403,
                    "preflight_header_denied",
                    "one or more requested headers are not allowed",
                ),
                Some(origin),
            );
        }
        let mut response = HttpResponse {
            status: 204,
            content_type: "application/octet-stream".into(),
            headers: BTreeMap::from([
                (
                    "access-control-allow-methods".into(),
                    requested_method.to_ascii_uppercase(),
                ),
                ("access-control-max-age".into(), "600".into()),
            ]),
            body: HttpBody::Empty,
        };
        if !requested_headers.is_empty() {
            response.headers.insert(
                "access-control-allow-headers".into(),
                requested_headers.join(", "),
            );
        }
        response.headers.insert(
            "vary".into(),
            "Origin, Access-Control-Request-Method, Access-Control-Request-Headers".into(),
        );
        apply_cors(response, Some(origin))
    }

    pub fn connect_websocket(
        &self,
        client_id: ClientId,
        projection: ReconnectProjection,
        headers: BTreeMap<String, Vec<String>>,
        context: HostRequestContext,
    ) -> Result<ClientId, NativeApiHostError> {
        self.connect_websocket_projected(client_id, headers, context, move |_, _| Ok(projection))
    }

    pub fn connect_websocket_projected<F>(
        &self,
        client_id: ClientId,
        headers: BTreeMap<String, Vec<String>>,
        context: HostRequestContext,
        projection: F,
    ) -> Result<ClientId, NativeApiHostError>
    where
        F: FnOnce(&str, &ClientId) -> Result<ReconnectProjection, NativeApiHostError>,
    {
        let request = HttpRequest {
            method: sim_types::HttpMethod::Get,
            path: "/ws".into(),
            query: BTreeMap::new(),
            headers,
            body: bytes::Bytes::new(),
        };
        let security_context = security::RequestSecurityContext {
            method: "GET".into(),
            canonical_path: "/ws".into(),
            body_bytes: 0,
            header_bytes: request
                .headers
                .iter()
                .map(|(name, values)| name.len() + values.iter().map(String::len).sum::<usize>())
                .sum(),
            header_count: request.headers.values().map(Vec::len).sum(),
            origin: request_header(&request, "origin").map(str::to_owned),
            authorization: request_header(&request, "authorization").map(str::to_owned),
            peer_address: context.peer_address,
            forwarded_for: context.forwarded_for,
            transport_tls: context.transport_tls,
            required_scope: self
                .security
                .config()
                .require_authentication
                .then(|| "api:read".to_owned()),
            plugin: None,
            mutation_identity: None,
            now_epoch_seconds: context.now_epoch_seconds,
        };
        let authorized = self
            .security
            .authorize(&security_context)
            .map_err(NativeApiHostError::from_security)?;
        let principal = authorized.principal;
        let requested_client_id = client_id;
        let client_id = scoped_websocket_client_id(
            &self.profile_id,
            &principal.identity,
            &requested_client_id,
        )?;
        let projection = projection(&principal.identity, &client_id)?;
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .connect_authenticated_with_session_id(
                client_id.clone(),
                requested_client_id,
                principal,
                projection,
            )
            .map_err(|error| match error {
                WebSocketContractError::ClientAlreadyConnected(_)
                | WebSocketContractError::PrincipalMismatch(_) => NativeApiHostError::Security {
                    status: 409,
                    code: "websocket_client_identity_conflict",
                    message: error.to_string(),
                },
                error => NativeApiHostError::from_websocket(error),
            })?;
        Ok(client_id)
    }

    pub fn process_websocket_fragment(
        &self,
        client_id: &ClientId,
        fragment: InputFragment,
    ) -> Result<InputReport, NativeApiHostError> {
        self.ensure_websocket_principal(client_id)?;
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .process_input_fragment(client_id, fragment)
            .map_err(NativeApiHostError::from_websocket)
    }

    pub fn set_asset_seeder_events_enabled(&self, enabled: bool) -> Result<(), NativeApiHostError> {
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .set_asset_seeder_enabled(enabled);
        Ok(())
    }

    pub fn set_terminal_log_subscription(
        &self,
        client_id: &ClientId,
        subscribed: bool,
    ) -> Result<(), NativeApiHostError> {
        self.ensure_websocket_principal(client_id)?;
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .set_log_subscription(client_id, subscribed)
            .map_err(NativeApiHostError::from_websocket)
    }

    pub fn drain_websocket(
        &self,
        client_id: &ClientId,
    ) -> Result<Vec<OutboundMessage>, NativeApiHostError> {
        self.ensure_websocket_principal(client_id)?;
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .drain_client(client_id)
            .map_err(NativeApiHostError::from_websocket)
    }

    pub fn disconnect_websocket(&self, client_id: &ClientId) -> Result<bool, NativeApiHostError> {
        Ok(self
            .websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .disconnect(client_id))
    }

    fn publish_websocket_json(
        &self,
        event: NativeJsonEvent,
    ) -> Result<PublishReport, NativeApiHostError> {
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .publish_json(event)
            .map_err(NativeApiHostError::from_websocket)
    }

    fn publish_websocket_preview(
        &self,
        event: NativePreviewEvent,
    ) -> Result<PublishReport, NativeApiHostError> {
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .publish_preview(event)
            .map_err(NativeApiHostError::from_websocket)
    }

    fn publish_status_projection(&self) -> Result<Option<PublishReport>, NativeApiHostError> {
        let Some(status) = self
            .http
            .status_projection()
            .map_err(NativeApiHostError::from_http)?
        else {
            return Ok(None);
        };
        let sequence = self
            .status_sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |sequence| {
                sequence.checked_add(1)
            })
            .map_err(|_| {
                NativeApiHostError::WebSocket("native status event sequence is exhausted".into())
            })?;
        self.publish_websocket_json(NativeJsonEvent {
            sequence,
            event_type: "status".into(),
            data: json!({"status": status}),
            target: EventTarget::Broadcast,
            source: NativeEventSource::Runtime,
            association: EventAssociation::default(),
        })
        .map(Some)
    }

    pub fn publish_execution_event(
        &self,
        event: sim_runtime::AttemptEvent,
    ) -> Result<Vec<PublishReport>, NativeApiHostError> {
        if event.profile_id.0.to_string() != self.profile_id {
            return Err(NativeApiHostError::WebSocket(
                "execution event belongs to another runtime profile".into(),
            ));
        }
        let prompt_id = event.prompt_id.0.to_string();
        let attempt_id = event.attempt_id.0.to_string();
        let wire_sequence = event
            .sequence
            .checked_mul(4)
            .and_then(|sequence| sequence.checked_add(1))
            .ok_or_else(|| {
                NativeApiHostError::WebSocket(
                    "execution event sequence cannot be represented on the native wire protocol"
                        .into(),
                )
            })?;
        let node_id = event.node_id.as_ref().map(|node_id| node_id.0.clone());
        let client_id = event
            .data
            .as_ref()
            .and_then(|data| data.get("client_id"))
            .and_then(Value::as_str)
            .map(|client_id| ClientId::new(client_id.to_owned()))
            .transpose()
            .map_err(NativeApiHostError::from_websocket)?;
        let association = EventAssociation {
            prompt_id: Some(prompt_id.clone()),
            node_id: node_id.clone(),
            attempt_id: Some(attempt_id),
        };
        let timestamp = event.at.timestamp_millis();
        let publish_json = |sequence, event_type: &str, data: Value, target: EventTarget| {
            self.publish_websocket_json(NativeJsonEvent {
                sequence,
                event_type: event_type.into(),
                data,
                target,
                source: NativeEventSource::Runtime,
                association: association.clone(),
            })
        };
        let client_target = client_id.map(EventTarget::Client);
        let publish_client_json =
            |sequence, event_type: &str, data: Value| -> Result<Vec<_>, NativeApiHostError> {
                match client_target.clone() {
                    Some(target) => {
                        publish_json(sequence, event_type, data, target).map(|report| vec![report])
                    }
                    None => Ok(Vec::new()),
                }
            };
        let publish_terminal =
            |sequence: u64, event_type: &str, data: Value, target: EventTarget| {
                let terminal_sequence = sequence.checked_add(1).ok_or_else(|| {
                    NativeApiHostError::WebSocket("execution terminal sequence is exhausted".into())
                })?;
                let terminal_association = EventAssociation {
                    prompt_id: association.prompt_id.clone(),
                    node_id: None,
                    attempt_id: association.attempt_id.clone(),
                };
                let executing = self.publish_websocket_json(NativeJsonEvent {
                    sequence,
                    event_type: "executing".into(),
                    data: json!({
                        "node": Value::Null,
                        "display_node": Value::Null,
                        "prompt_id": prompt_id,
                    }),
                    target: target.clone(),
                    source: NativeEventSource::Runtime,
                    association: terminal_association,
                })?;
                let terminal = self.publish_websocket_json(NativeJsonEvent {
                    sequence: terminal_sequence,
                    event_type: event_type.into(),
                    data,
                    target,
                    source: NativeEventSource::Runtime,
                    association: association.clone(),
                })?;
                Ok(vec![executing, terminal])
            };
        match event.kind {
            sim_runtime::AttemptEventKind::Started => publish_client_json(
                wire_sequence,
                "execution_start",
                json!({"prompt_id": prompt_id, "timestamp": timestamp}),
            ),
            sim_runtime::AttemptEventKind::Progress { completed, total } => {
                let Some(target) = client_target.clone() else {
                    return Ok(Vec::new());
                };
                let node = node_id.ok_or_else(|| {
                    NativeApiHostError::WebSocket(
                        "native execution progress requires a canonical node identity".into(),
                    )
                })?;
                let progress_sequence = wire_sequence.checked_add(1).ok_or_else(|| {
                    NativeApiHostError::WebSocket("execution progress sequence is exhausted".into())
                })?;
                let state_sequence = wire_sequence.checked_add(2).ok_or_else(|| {
                    NativeApiHostError::WebSocket(
                        "execution progress-state sequence is exhausted".into(),
                    )
                })?;
                let mut reports = Vec::with_capacity(3);
                let progress_nodes = serde_json::Map::from_iter([(
                    node.clone(),
                    progress_state_node(
                        &prompt_id,
                        &node,
                        completed,
                        total,
                        if total > 0 && completed >= total {
                            "finished"
                        } else {
                            "running"
                        },
                    ),
                )]);
                reports.push(publish_json(
                    wire_sequence,
                    "executing",
                    json!({
                        "node": node,
                        "display_node": node,
                        "prompt_id": prompt_id,
                    }),
                    target.clone(),
                )?);
                reports.push(publish_json(
                    progress_sequence,
                    "progress",
                    json!({
                        "value": completed,
                        "max": total,
                        "prompt_id": prompt_id,
                        "node": node,
                    }),
                    target.clone(),
                )?);
                reports.push(publish_json(
                    state_sequence,
                    "progress_state",
                    json!({
                        "prompt_id": prompt_id,
                        "nodes": progress_nodes,
                    }),
                    target,
                )?);
                Ok(reports)
            }
            sim_runtime::AttemptEventKind::Preview { preview } => {
                let Some(target) = client_target else {
                    return Ok(Vec::new());
                };
                let format = match preview.media_type.as_str() {
                    "image/jpeg" | "image/jpg" => PreviewImageFormat::Jpeg,
                    "image/png" => PreviewImageFormat::Png,
                    media_type => {
                        return Err(NativeApiHostError::WebSocket(format!(
                            "preview media type {media_type} is not supported by the native wire protocol"
                        )));
                    }
                };
                let metadata = serde_json::Map::from_iter([
                    (
                        "preview_id".into(),
                        Value::String(preview.preview_id.to_string()),
                    ),
                    ("revision".into(), Value::from(preview.revision)),
                    ("media_type".into(), Value::String(preview.media_type)),
                ]);
                self.publish_websocket_preview(NativePreviewEvent {
                    sequence: wire_sequence,
                    format,
                    encoded_image: preview.encoded_bytes,
                    target,
                    source: NativeEventSource::Runtime,
                    association,
                    metadata,
                })
                .map(|report| vec![report])
            }
            sim_runtime::AttemptEventKind::CacheHit => publish_client_json(
                wire_sequence,
                "execution_cached",
                json!({
                    "nodes": node_id.into_iter().collect::<Vec<_>>(),
                    "prompt_id": prompt_id,
                    "timestamp": timestamp,
                }),
            ),
            sim_runtime::AttemptEventKind::OutputAvailable { output } => {
                let node = output.node_id.0.clone();
                let output = serde_json::to_value(output)
                    .map_err(|error| NativeApiHostError::WebSocket(error.to_string()))?;
                publish_client_json(
                    wire_sequence,
                    "executed",
                    json!({
                        "node": node,
                        "display_node": node,
                        "output": output,
                        "prompt_id": prompt_id,
                    }),
                )
            }
            sim_runtime::AttemptEventKind::Succeeded => match client_target {
                Some(target) => publish_terminal(
                    wire_sequence,
                    "execution_success",
                    json!({"prompt_id": prompt_id, "timestamp": timestamp}),
                    target,
                ),
                None => Ok(Vec::new()),
            },
            sim_runtime::AttemptEventKind::Failed { failure } => {
                let node = failure
                    .node_id
                    .as_ref()
                    .map(|node_id| node_id.0.clone())
                    .or(node_id)
                    .unwrap_or_default();
                match client_target {
                    Some(target) => publish_terminal(
                        wire_sequence,
                        "execution_error",
                        json!({
                            "prompt_id": prompt_id,
                            "node_id": node,
                            "node_type": failure.code,
                            "exception_message": failure.message,
                            "exception_type": failure.code,
                            "executed": [],
                            "traceback": [],
                            "current_inputs": {},
                            "current_outputs": {},
                            "timestamp": timestamp,
                        }),
                        target,
                    ),
                    None => Ok(Vec::new()),
                }
            }
            sim_runtime::AttemptEventKind::Cancelled
            | sim_runtime::AttemptEventKind::Interrupted { .. }
            | sim_runtime::AttemptEventKind::RecoveryInterrupted { .. } => publish_terminal(
                wire_sequence,
                "execution_interrupted",
                json!({
                    "prompt_id": prompt_id,
                    "node_id": node_id.unwrap_or_default(),
                    "node_type": "native",
                    "executed": [],
                    "timestamp": timestamp,
                }),
                EventTarget::Broadcast,
            ),
            sim_runtime::AttemptEventKind::OutputPrepared { .. }
            | sim_runtime::AttemptEventKind::CancelRequested { .. } => Ok(Vec::new()),
        }
    }

    pub fn shutdown(&self, reason: impl Into<String>) -> Result<(), NativeApiHostError> {
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .shutdown(reason);
        Ok(())
    }

    fn ensure_websocket_principal(&self, client_id: &ClientId) -> Result<(), NativeApiHostError> {
        self.websocket
            .lock()
            .map_err(|_| NativeApiHostError::StateUnavailable)?
            .authenticated_principal(client_id)
            .map(|_| ())
            .map_err(NativeApiHostError::from_websocket)
    }
}

fn progress_state_node(
    prompt_id: &str,
    node_id: &str,
    completed: u64,
    total: u64,
    state: &str,
) -> Value {
    json!({
        "value": completed,
        "max": total,
        "state": state,
        "node_id": node_id,
        "prompt_id": prompt_id,
        "display_node_id": node_id,
        "real_node_id": node_id,
        "parent_node_id": Value::Null,
    })
}

fn request_header<'a>(request: &'a HttpRequest, name: &str) -> Option<&'a str> {
    map_header(&request.headers, name)
}

fn plugin_route_request(
    request: &HttpRequest,
    host_profile_id: &str,
) -> Result<Option<security::PluginRouteRequest>, NativeApiHostError> {
    let profile_id = request_header(request, "x-zed-plugin-profile");
    let plugin_id = request_header(request, "x-zed-plugin-id");
    let plugin_digest = request_header(request, "x-zed-plugin-digest");
    let capabilities = request_header(request, "x-zed-plugin-capabilities");
    if profile_id.is_none()
        && plugin_id.is_none()
        && plugin_digest.is_none()
        && capabilities.is_none()
    {
        return Ok(None);
    }
    let malformed = || NativeApiHostError::Http {
        status: 400,
        code: "invalid_plugin_authority".into(),
        message: "plugin route authority requires exact profile, plugin, and digest headers".into(),
    };
    let profile_id = profile_id
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(malformed)?;
    let plugin_id = plugin_id
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(malformed)?;
    let plugin_digest = plugin_digest
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(malformed)?;
    if profile_id != host_profile_id {
        return Err(NativeApiHostError::Security {
            status: 403,
            code: "cross_profile_plugin_route",
            message: "plugin route authority belongs to another runtime profile".into(),
        });
    }
    let required_capabilities = capabilities
        .map(|capabilities| {
            capabilities
                .split(',')
                .map(str::trim)
                .map(|capability| {
                    if capability.is_empty() {
                        Err(malformed())
                    } else {
                        sim_runtime::Capability::parse_wire_identifier(capability)
                            .map_err(|_| malformed())
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map(sim_runtime::CapabilitySet::new)
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Some(security::PluginRouteRequest {
        profile_id: profile_id.to_owned(),
        plugin_id: plugin_id.to_owned(),
        plugin_digest: plugin_digest.to_owned(),
        required_capabilities,
    }))
}

fn map_header<'a>(headers: &'a BTreeMap<String, Vec<String>>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.first())
        .map(String::as_str)
}

fn apply_cors(mut response: HttpResponse, allowed_origin: Option<&str>) -> HttpResponse {
    if let Some(origin) = allowed_origin {
        response
            .headers
            .insert("access-control-allow-origin".into(), origin.into());
        response
            .headers
            .entry("vary".into())
            .and_modify(|value| {
                if !value
                    .split(',')
                    .any(|name| name.trim().eq_ignore_ascii_case("origin"))
                {
                    value.push_str(", Origin");
                }
            })
            .or_insert_with(|| "Origin".into());
    }
    response
}

fn parse_http_method(method: &str) -> Option<sim_types::HttpMethod> {
    match method.to_ascii_uppercase().as_str() {
        "GET" => Some(sim_types::HttpMethod::Get),
        "POST" => Some(sim_types::HttpMethod::Post),
        "PUT" => Some(sim_types::HttpMethod::Put),
        "PATCH" => Some(sim_types::HttpMethod::Patch),
        "DELETE" => Some(sim_types::HttpMethod::Delete),
        "HEAD" => Some(sim_types::HttpMethod::Head),
        _ => None,
    }
}

fn http_method_name(method: sim_types::HttpMethod) -> &'static str {
    match method {
        sim_types::HttpMethod::Get => "GET",
        sim_types::HttpMethod::Post => "POST",
        sim_types::HttpMethod::Put => "PUT",
        sim_types::HttpMethod::Patch => "PATCH",
        sim_types::HttpMethod::Delete => "DELETE",
        sim_types::HttpMethod::Head => "HEAD",
    }
}

fn stable_request_digest(
    request: &HttpRequest,
    canonical_feature_id: &str,
    authority: &NativeRequestAuthority,
) -> String {
    let mut digest = Sha256::new();
    update_digest(&mut digest, b"zed-native-http-request-v2");
    update_digest(&mut digest, authority.profile_id.as_bytes());
    update_digest(&mut digest, authority.principal.as_bytes());
    update_digest(
        &mut digest,
        authority
            .plugin_id
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    for scope in &authority.scopes {
        update_digest(&mut digest, scope.as_bytes());
    }
    update_digest(
        &mut digest,
        authority
            .plugin_digest
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    update_digest(&mut digest, canonical_feature_id.as_bytes());
    update_digest(&mut digest, http_method_name(request.method).as_bytes());
    update_digest(&mut digest, request.path.as_bytes());
    for (name, values) in &request.query {
        update_digest(&mut digest, name.as_bytes());
        for value in values {
            update_digest(&mut digest, value.as_bytes());
        }
    }
    const SEMANTIC_HEADERS: [&str; 10] = [
        "accept",
        "sim-user",
        "content-encoding",
        "content-type",
        "if-match",
        "if-none-match",
        "range",
        "x-zed-plugin-capabilities",
        "x-zed-plugin-digest",
        "x-zed-plugin-id",
    ];
    for header_name in SEMANTIC_HEADERS {
        update_digest(&mut digest, header_name.as_bytes());
        for value in request.headers.iter().filter_map(|(candidate, values)| {
            candidate
                .eq_ignore_ascii_case(header_name)
                .then_some(values.as_slice())
        }) {
            for value in value {
                update_digest(&mut digest, value.as_bytes());
            }
        }
    }
    update_digest(&mut digest, &request.body);
    format!("sha256:{}", encode_digest(digest.finalize().as_slice()))
}

fn scoped_idempotency_key(client_key: &str, authority: &NativeRequestAuthority) -> String {
    scoped_identity_digest(b"idempotency", client_key, "", authority)
}

fn scoped_operation_id(
    client_operation_id: &str,
    canonical_feature_id: &str,
    authority: &NativeRequestAuthority,
) -> String {
    scoped_identity_digest(
        b"operation",
        client_operation_id,
        canonical_feature_id,
        authority,
    )
}

fn scoped_websocket_client_id(
    profile_id: &str,
    principal: &str,
    requested_client_id: &ClientId,
) -> Result<ClientId, NativeApiHostError> {
    let mut digest = Sha256::new();
    update_digest(&mut digest, b"zed-native-websocket-client-v1");
    update_digest(&mut digest, profile_id.as_bytes());
    update_digest(&mut digest, principal.as_bytes());
    update_digest(&mut digest, requested_client_id.as_str().as_bytes());
    ClientId::new(format!("principal-{}", encode_digest(&digest.finalize())))
        .map_err(NativeApiHostError::from_websocket)
}

fn scope_prompt_client_id(
    request: &mut HttpRequest,
    authority: &NativeRequestAuthority,
) -> Result<(), NativeApiHostError> {
    let Ok(mut value) = serde_json::from_slice::<Value>(&request.body) else {
        return Ok(());
    };
    let Some(client_id) = value
        .as_object()
        .and_then(|object| object.get("client_id"))
        .and_then(Value::as_str)
    else {
        return Ok(());
    };
    let requested =
        ClientId::new(client_id.to_owned()).map_err(|error| NativeApiHostError::Http {
            status: 400,
            code: "invalid_websocket_client_id".into(),
            message: error.to_string(),
        })?;
    let scoped =
        scoped_websocket_client_id(&authority.profile_id, &authority.principal, &requested)?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "client_id".into(),
            Value::String(scoped.as_str().to_owned()),
        );
    }
    request.body = serde_json::to_vec(&value)
        .map(bytes::Bytes::from)
        .map_err(|error| NativeApiHostError::Http {
            status: 500,
            code: "prompt_client_id_serialization_failed".into(),
            message: error.to_string(),
        })?;
    Ok(())
}

fn scoped_identity_digest(
    domain: &[u8],
    client_identity: &str,
    feature_id: &str,
    authority: &NativeRequestAuthority,
) -> String {
    let mut digest = Sha256::new();
    update_digest(&mut digest, b"zed-native-http-identity-v1");
    update_digest(&mut digest, domain);
    update_digest(&mut digest, authority.profile_id.as_bytes());
    update_digest(&mut digest, authority.principal.as_bytes());
    update_digest(
        &mut digest,
        authority
            .plugin_id
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    update_digest(
        &mut digest,
        authority
            .plugin_digest
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    update_digest(&mut digest, feature_id.as_bytes());
    update_digest(&mut digest, client_identity.as_bytes());
    format!("sha256:{}", encode_digest(digest.finalize().as_slice()))
}

fn update_digest(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}

fn encode_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NativeApiHostError {
    #[error("invalid native API host configuration: {0}")]
    InvalidConfiguration(String),
    #[error("native API security rejected the request: {message}")]
    Security {
        status: u16,
        code: &'static str,
        message: String,
    },
    #[error("native HTTP host rejected the request: {message}")]
    Http {
        status: u16,
        code: String,
        message: String,
    },
    #[error("native WebSocket host rejected the request: {0}")]
    WebSocket(String),
    #[error("native API host state is unavailable")]
    StateUnavailable,
    #[error("a native API mutation requires an idempotency key or durable operation identity")]
    MutationIdentityRequired,
    #[error("native API persistence failed: {0}")]
    Persistence(String),
    #[error("native runtime API service failed: {0}")]
    Runtime(String),
}

impl NativeApiHostError {
    fn from_security(error: security::ApiSecurityError) -> Self {
        let error_message = error.to_string();
        match error {
            security::ApiSecurityError::MutationIdentityRequired => Self::MutationIdentityRequired,
            security::ApiSecurityError::Persistence(message) => Self::Persistence(message),
            security::ApiSecurityError::AuthenticationRequired
            | security::ApiSecurityError::InvalidCredential
            | security::ApiSecurityError::ExpiredCredential => Self::Security {
                status: 401,
                code: "authentication_required",
                message: error_message,
            },
            security::ApiSecurityError::OriginRequired => Self::Security {
                status: 403,
                code: "origin_required",
                message: error_message,
            },
            security::ApiSecurityError::OriginDenied => Self::Security {
                status: 403,
                code: "origin_denied",
                message: error_message,
            },
            security::ApiSecurityError::UntrustedForwardedAddress => Self::Security {
                status: 403,
                code: "untrusted_forwarded_address",
                message: error_message,
            },
            security::ApiSecurityError::ForbiddenScope(_)
            | security::ApiSecurityError::PluginRouteDenied => Self::Security {
                status: 403,
                code: "request_forbidden",
                message: error_message,
            },
            security::ApiSecurityError::BodyTooLarge
            | security::ApiSecurityError::HeadersTooLarge => Self::Security {
                status: 413,
                code: "request_too_large",
                message: error_message,
            },
            security::ApiSecurityError::TlsRequired => Self::Security {
                status: 426,
                code: "tls_required",
                message: error_message,
            },
            security::ApiSecurityError::RateLimited => Self::Security {
                status: 429,
                code: "rate_limited",
                message: error_message,
            },
            security::ApiSecurityError::TooManyConcurrentRequests
            | security::ApiSecurityError::SecurityStateUnavailable => Self::Security {
                status: 503,
                code: "security_state_unavailable",
                message: error_message,
            },
            security::ApiSecurityError::UnsafePath
            | security::ApiSecurityError::IdempotencyConflict
            | security::ApiSecurityError::IdempotencyLedgerFull
            | security::ApiSecurityError::InvalidIdempotencySnapshot
            | security::ApiSecurityError::InvalidIdempotencyTransition
            | security::ApiSecurityError::UnknownIdempotencyKey => Self::Security {
                status: 409,
                code: "request_conflict",
                message: error_message,
            },
            _ => Self::Security {
                status: 500,
                code: "security_configuration_invalid",
                message: error_message,
            },
        }
    }

    fn from_http(error: HttpRouteError) -> Self {
        Self::Http {
            status: error.status,
            code: error.code,
            message: error.message,
        }
    }

    fn from_websocket(error: WebSocketContractError) -> Self {
        Self::WebSocket(error.to_string())
    }

    fn into_http_response(self) -> HttpResponse {
        let message = self.to_string();
        let (status, code) = match &self {
            Self::Security { status, code, .. } => (*status, *code),
            Self::Http { status, code, .. } => (*status, code.as_str()),
            Self::WebSocket(_) => (400, "websocket_request_rejected"),
            Self::StateUnavailable | Self::Persistence(_) => (503, "host_state_unavailable"),
            Self::Runtime(_) => (503, "native_runtime_unavailable"),
            Self::MutationIdentityRequired => (428, "mutation_identity_required"),
            Self::InvalidConfiguration(_) => (500, "host_configuration_invalid"),
        };
        HttpResponse::json(
            status,
            json!({
                "error": {
                    "code": code,
                    "message": message,
                }
            }),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        error::Error,
        fs,
        net::{IpAddr, Ipv4Addr, Ipv6Addr},
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct ProbeServices {
        calls: AtomicUsize,
    }

    impl NativeHttpServices for ProbeServices {
        fn dispatch(
            &self,
            request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(NativeServiceResponse::json(
                200,
                json!({
                    "native": true,
                    "feature_id": request.route.canonical_feature_id,
                }),
            ))
        }
    }

    struct BlockingProbeServices {
        calls: AtomicUsize,
        started: std::sync::mpsc::Sender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl NativeHttpServices for BlockingProbeServices {
        fn dispatch(
            &self,
            request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                self.started.send(()).map_err(|error| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "fixture_start_failed",
                        error.to_string(),
                    )
                })?;
                self.release
                    .lock()
                    .map_err(|_| {
                        NativeServiceError::new(
                            NativeServiceErrorKind::Internal,
                            "fixture_release_poisoned",
                            "fixture release state was poisoned",
                        )
                    })?
                    .recv()
                    .map_err(|error| {
                        NativeServiceError::new(
                            NativeServiceErrorKind::Internal,
                            "fixture_release_failed",
                            error.to_string(),
                        )
                    })?;
            }
            Ok(NativeServiceResponse::json(
                200,
                json!({
                    "native": true,
                    "feature_id": request.route.canonical_feature_id,
                }),
            ))
        }
    }

    #[derive(Default)]
    struct TimeoutProbeServices {
        calls: AtomicUsize,
    }

    impl NativeHttpServices for TimeoutProbeServices {
        fn dispatch(
            &self,
            _request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(NativeServiceError::new(
                NativeServiceErrorKind::Timeout,
                "fixture_timeout",
                "the native mutation response timed out",
            ))
        }
    }

    fn permission_policy(
        profile_id: &str,
        config: &security::ApiSecurityConfig,
    ) -> Result<Arc<sim_runtime::PermissionPolicy>, Box<dyn Error>> {
        let route_grants = security::plugin_route_permission_grants(&config.plugin_route_grants)?;
        Ok(Arc::new(
            sim_runtime::PermissionPolicy::native_runtime_services(profile_id.to_owned())?
                .with_additional_grants(route_grants)?,
        ))
    }

    #[test]
    fn plugin_authority_headers_require_canonical_capability_identifiers()
    -> Result<(), Box<dyn Error>> {
        let request = HttpRequest::new(sim_types::HttpMethod::Post, "/api/jobs/cancel")
            .with_header("x-zed-plugin-profile", "profile-a")
            .with_header("x-zed-plugin-id", "provider-plugin")
            .with_header("x-zed-plugin-digest", "sha256:provider-plugin")
            .with_header(
                "x-zed-plugin-capabilities",
                "provider_network:payment|https://payment.invalid/api/jobs/cancel",
            );
        let parsed = plugin_route_request(&request, "profile-a")?
            .expect("plugin authority headers were parsed");
        let capability = sim_runtime::Capability::ProviderNetwork {
            provider: "payment".into(),
            endpoint: "https://payment.invalid/api/jobs/cancel".into(),
        };
        assert_eq!(
            parsed.required_capabilities,
            sim_runtime::CapabilitySet::new([capability])
        );

        let malformed = HttpRequest::new(sim_types::HttpMethod::Post, "/api/jobs/cancel")
            .with_header("x-zed-plugin-profile", "profile-a")
            .with_header("x-zed-plugin-id", "provider-plugin")
            .with_header("x-zed-plugin-digest", "sha256:provider-plugin")
            .with_header("x-zed-plugin-capabilities", "provider:payment");
        assert!(matches!(
            plugin_route_request(&malformed, "profile-a"),
            Err(NativeApiHostError::Http {
                status: 400,
                code,
                ..
            }) if code == "invalid_plugin_authority"
        ));
        Ok(())
    }

    struct RecoveryOutcome {
        artifact: Value,
        passed: bool,
    }

    fn recovery_outcome(
        identifier: &str,
        passed: bool,
        side_effect_count: usize,
        evidence: Value,
    ) -> RecoveryOutcome {
        RecoveryOutcome {
            artifact: json!({
                "id": identifier,
                "passed": passed,
                "side_effect_count": side_effect_count,
                "evidence": evidence,
            }),
            passed,
        }
    }

    #[derive(Debug)]
    struct FailingSnapshotStore;

    impl security::IdempotencySnapshotStore for FailingSnapshotStore {
        fn load(
            &self,
        ) -> Result<Option<security::IdempotencySnapshot>, security::ApiSecurityError> {
            Ok(None)
        }

        fn save(
            &self,
            _snapshot: &security::IdempotencySnapshot,
        ) -> Result<(), security::ApiSecurityError> {
            Err(security::ApiSecurityError::Persistence(
                "injected persistence failure".into(),
            ))
        }
    }

    #[derive(Default)]
    struct FailOnceAfterPreparedSnapshotStore {
        snapshot: Mutex<Option<security::IdempotencySnapshot>>,
        saves: AtomicUsize,
    }

    impl security::IdempotencySnapshotStore for FailOnceAfterPreparedSnapshotStore {
        fn load(
            &self,
        ) -> Result<Option<security::IdempotencySnapshot>, security::ApiSecurityError> {
            self.snapshot
                .lock()
                .map(|snapshot| snapshot.clone())
                .map_err(|_| {
                    security::ApiSecurityError::Persistence(
                        "injected snapshot state is unavailable".into(),
                    )
                })
        }

        fn save(
            &self,
            snapshot: &security::IdempotencySnapshot,
        ) -> Result<(), security::ApiSecurityError> {
            if self.saves.fetch_add(1, Ordering::SeqCst) == 1 {
                return Err(security::ApiSecurityError::Persistence(
                    "injected post-side-effect persistence failure".into(),
                ));
            }
            self.snapshot
                .lock()
                .map(|mut stored| *stored = Some(snapshot.clone()))
                .map_err(|_| {
                    security::ApiSecurityError::Persistence(
                        "injected snapshot state is unavailable".into(),
                    )
                })
        }
    }

    fn context(now_epoch_seconds: u64) -> HostRequestContext {
        HostRequestContext {
            peer_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            forwarded_for: None,
            transport_tls: false,
            now_epoch_seconds,
        }
    }

    fn prompt_request(key: &str) -> HttpRequest {
        HttpRequest::new(sim_types::HttpMethod::Post, "/prompt")
            .with_header("content-type", "application/json")
            .with_header("idempotency-key", key)
            .with_body(bytes::Bytes::from_static(br#"{"prompt":{}}"#))
    }

    fn test_directory(name: &str) -> Result<PathBuf, Box<dyn Error>> {
        let path = std::env::temp_dir().join(format!(
            "sim-api-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        if path.try_exists()? {
            fs::remove_dir_all(&path)?;
        }
        fs::create_dir(&path)?;
        Ok(path)
    }

    fn artifact_idempotency_store(
        snapshot_path: &Path,
    ) -> Result<security::ArtifactIdempotencySnapshotStore, security::ApiSecurityError> {
        let parent = snapshot_path.parent().ok_or_else(|| {
            security::ApiSecurityError::InvalidConfiguration(
                "test idempotency snapshot has no parent".into(),
            )
        })?;
        let relative_path = snapshot_path.file_name().ok_or_else(|| {
            security::ApiSecurityError::InvalidConfiguration(
                "test idempotency snapshot has no filename".into(),
            )
        })?;
        security::ArtifactIdempotencySnapshotStore::from_directory(parent, relative_path)
    }

    fn test_presentation(
        profile_id: sim_runtime::ProfileId,
    ) -> Result<sim_runtime::SharedExecutionPresentationService, Box<dyn Error>> {
        let mut presentation = sim_runtime::ExecutionPresentationService::new(4_096)?;
        presentation.initialize_profile(
            profile_id,
            sim_runtime::ExecutionDataSource::Live,
            sim_runtime::ExecutionSnapshotStatus::Ready,
        )?;
        Ok(sim_runtime::ExecutionPresentationOwner::ephemeral(
            presentation,
        ))
    }

    fn file_host(
        services: Arc<ProbeServices>,
        snapshot_path: &Path,
    ) -> Result<NativeApiHost<ProbeServices>, NativeApiHostError> {
        let store = Arc::new(
            artifact_idempotency_store(snapshot_path).map_err(NativeApiHostError::from_security)?,
        );
        NativeApiHost::new(
            "profile-a",
            services,
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security::ApiSecurityConfig::loopback(),
            permission_policy("profile-a", &security::ApiSecurityConfig::loopback())
                .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
            store,
        )
    }

    #[test]
    pub(crate) fn native_host_idempotency_contracts() -> Result<(), Box<dyn Error>> {
        let directory = test_directory("host-idempotency-contracts")?;
        let snapshot_path = directory.join("idempotency.json");
        let services = Arc::new(ProbeServices::default());
        let host = file_host(services.clone(), &snapshot_path)?;

        let request = prompt_request("host-replay");
        let first = host.route_http(request.clone(), context(1))?;
        let replay = host.route_http(request.clone(), context(2))?;
        assert_eq!(first.status, 200);
        assert_eq!(body_json(&replay), body_json(&first));
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);

        let conflict = host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Post, "/prompt")
                .with_header("content-type", "application/json")
                .with_header("idempotency-key", "host-replay")
                .with_body(bytes::Bytes::from_static(br#"{"prompt":{"changed":true}}"#)),
            context(3),
        );
        assert_eq!(conflict.status, 409);
        assert_eq!(
            body_json(&conflict)
                .and_then(|body| body.pointer("/error/code"))
                .and_then(Value::as_str),
            Some("request_conflict")
        );
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);

        let durable = HttpRequest::new(sim_types::HttpMethod::Post, "/interrupt")
            .with_header("content-type", "application/json")
            .with_body(bytes::Bytes::from_static(
                br#"{"attempt_id":"00000000-0000-0000-0000-000000000701"}"#,
            ));
        assert_eq!(host.route_http(durable.clone(), context(4))?.status, 200);
        assert_eq!(host.route_http(durable, context(5))?.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 2);
        drop(host);

        let restarted = file_host(services.clone(), &snapshot_path)?;
        assert_eq!(restarted.route_http(request, context(6))?.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 2);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    pub(crate) fn native_host_concurrent_and_ambiguous_reconciliation() -> Result<(), Box<dyn Error>>
    {
        let directory = test_directory("host-concurrent-reconciliation")?;
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let blocking_services = Arc::new(BlockingProbeServices {
            calls: AtomicUsize::new(0),
            started: started_sender,
            release: Mutex::new(release_receiver),
        });
        let security_config = security::ApiSecurityConfig::loopback();
        let blocking_host = Arc::new(NativeApiHost::new(
            "profile-a",
            blocking_services.clone(),
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security_config.clone(),
            permission_policy("profile-a", &security_config)?,
            Arc::new(artifact_idempotency_store(
                &directory.join("blocking-idempotency.json"),
            )?),
        )?);
        let request = prompt_request("host-concurrent");
        let first_host = blocking_host.clone();
        let first_request = request.clone();
        let first = std::thread::spawn(move || first_host.route_http(first_request, context(10)));
        started_receiver.recv_timeout(std::time::Duration::from_secs(5))?;

        let pending = blocking_host.handle_http(request.clone(), context(11));
        assert_eq!(pending.status, 409);
        assert_eq!(
            body_json(&pending)
                .and_then(|body| body.pointer("/error/code"))
                .and_then(Value::as_str),
            Some("mutation_in_progress")
        );
        assert_eq!(blocking_services.calls.load(Ordering::SeqCst), 1);
        release_sender.send(())?;
        let first_response = first
            .join()
            .map_err(|_| "first host mutation thread panicked")??;
        assert_eq!(first_response.status, 200);
        assert_eq!(blocking_host.route_http(request, context(12))?.status, 200);
        assert_eq!(blocking_services.calls.load(Ordering::SeqCst), 1);

        let timeout_services = Arc::new(TimeoutProbeServices::default());
        let timeout_host = NativeApiHost::new(
            "profile-a",
            timeout_services.clone(),
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security_config.clone(),
            permission_policy("profile-a", &security_config)?,
            Arc::new(artifact_idempotency_store(
                &directory.join("timeout-idempotency.json"),
            )?),
        )?;
        let timeout_request = prompt_request("host-timeout");
        let timeout = timeout_host.handle_http(timeout_request.clone(), context(20));
        assert_eq!(timeout.status, 504);
        let unresolved = timeout_host.handle_http(timeout_request, context(21));
        assert_eq!(unresolved.status, 409);
        assert_eq!(
            body_json(&unresolved)
                .and_then(|body| body.pointer("/error/code"))
                .and_then(Value::as_str),
            Some("ambiguous_mutation_reconciliation_required")
        );
        assert_eq!(timeout_services.calls.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    fn restarted_idempotency_decision(
        operation: &str,
        state: security::IdempotencyState,
    ) -> Result<security::IdempotencyDecision, security::ApiSecurityError> {
        let maximum_records = HttpLimits::default().idempotency_capacity;
        let maximum_response_bytes = HttpLimits::default().maximum_response_bytes;
        let key = format!("key:{operation}");
        let operation_id = format!("operation:{operation}");
        let request_digest = format!("digest:{operation}");
        let mut ledger =
            security::IdempotencyLedger::new("profile-a", maximum_records, maximum_response_bytes)?;
        assert_eq!(
            ledger.begin(key.clone(), operation_id.clone(), request_digest.clone(), 1,)?,
            security::IdempotencyDecision::Begin
        );
        if state != security::IdempotencyState::Prepared {
            ledger.complete(&key, state, 2)?;
        }
        let mut recovered = security::IdempotencyLedger::restore(
            ledger.snapshot(),
            maximum_records,
            maximum_response_bytes,
        )?;
        recovered.begin(key, operation_id, request_digest, 3)
    }

    fn durable_catalog_case(
        feature_id: &str,
    ) -> Result<(Value, [bool; 4]), security::ApiSecurityError> {
        let committed_body = br#"{"committed":true}"#.to_vec();
        let failed_body = br#"{"failed":true}"#.to_vec();
        let prepared = restarted_idempotency_decision(
            &format!("{feature_id}:prepared"),
            security::IdempotencyState::Prepared,
        )?;
        let committed = restarted_idempotency_decision(
            &format!("{feature_id}:committed"),
            security::IdempotencyState::Committed {
                status: 200,
                response_body: committed_body.clone(),
            },
        )?;
        let failed = restarted_idempotency_decision(
            &format!("{feature_id}:failed"),
            security::IdempotencyState::Failed {
                status: 409,
                response_body: failed_body.clone(),
            },
        )?;
        let interrupted = restarted_idempotency_decision(
            &format!("{feature_id}:interrupted"),
            security::IdempotencyState::Interrupted,
        )?;
        let prepared_passed = matches!(prepared, security::IdempotencyDecision::Reconcile { .. });
        let committed_passed = matches!(
            committed,
            security::IdempotencyDecision::Replay { status: 200, response_body }
                if response_body == committed_body
        );
        let failed_passed = matches!(
            failed,
            security::IdempotencyDecision::Replay { status: 409, response_body }
                if response_body == failed_body
        );
        let interrupted_passed =
            matches!(interrupted, security::IdempotencyDecision::Reconcile { .. });
        let checks = [
            prepared_passed,
            committed_passed,
            failed_passed,
            interrupted_passed,
        ];
        Ok((
            json!({
                "id": feature_id,
                "scope": "generic ledger transition coverage only; this is not a side-effect fixture",
                "prepared_restart_reconciles": prepared_passed,
                "committed_restart_replays": committed_passed,
                "failed_restart_replays": failed_passed,
                "interrupted_restart_reconciles": interrupted_passed,
            }),
            checks,
        ))
    }

    fn write_validation_artifact(name: &str, value: &Value) -> Result<(), Box<dyn Error>> {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/sim-parity");
        fs::create_dir_all(&directory)?;
        let mut bytes = serde_json::to_vec_pretty(value)?;
        bytes.push(b'\n');
        fs::write(directory.join(name), bytes)?;
        Ok(())
    }

    fn body_json(response: &HttpResponse) -> Option<&Value> {
        match &response.body {
            HttpBody::Json(value) => Some(value),
            HttpBody::Empty | HttpBody::Bytes(_) | HttpBody::Stream(_) => None,
        }
    }

    #[test]
    fn native_host_integration_smoke() -> Result<(), Box<dyn Error>> {
        let directory = test_directory("native-host")?;
        let snapshot_path = directory.join("idempotency.json");
        let services = Arc::new(ProbeServices::default());
        let host = file_host(services.clone(), &snapshot_path)?;

        let response = host.route_http(prompt_request("submit-1"), context(60))?;
        assert_eq!(response.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);
        drop(host);

        let recovered = file_host(services.clone(), &snapshot_path)?;
        let replay = recovered.route_http(prompt_request("submit-1"), context(61))?;
        assert_eq!(replay.status, 200);
        assert_eq!(body_json(&replay), body_json(&response));
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);

        let requested_client_id = ClientId::new("native-client")?;
        let client_id = recovered.connect_websocket(
            requested_client_id.clone(),
            ReconnectProjection::default(),
            BTreeMap::new(),
            context(62),
        )?;
        let initial = recovered.drain_websocket(&client_id)?;
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].event_type, "status");
        assert_eq!(
            serde_json::from_slice::<Value>(&initial[0].payload)?["data"]["sid"],
            requested_client_id.as_str()
        );
        assert!(recovered.disconnect_websocket(&client_id)?);

        let mut remote = security::ApiSecurityConfig::loopback();
        remote.bind_address = IpAddr::V6(Ipv6Addr::UNSPECIFIED);
        assert_eq!(
            remote.validate(),
            Err(security::ApiSecurityError::RemoteExposureNotAcknowledged)
        );
        assert!(!HTTP_FORWARDING_SUPPORTED);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn native_runtime_builder_installs_concrete_services_and_capabilities()
    -> Result<(), Box<dyn Error>> {
        let directory = test_directory("native-runtime-builder")?;
        let profile_id = sim_runtime::ProfileId("00000000-0000-0000-0000-000000000020".parse()?);
        let presentation = test_presentation(profile_id)?;
        let event_bus = sim_runtime::ExecutionEventBus::new(16)?;
        let security_config = security::ApiSecurityConfig::loopback();
        let runtime = NativeRuntimeApiHost::native_image(
            profile_id,
            presentation.clone(),
            Arc::new(crate::services::AcceptingExecutionController),
            &event_bus,
            None,
            HttpLimits::default(),
            WebSocketLimits::default(),
            security_config.clone(),
            permission_policy(&profile_id.0.to_string(), &security_config)?,
            Arc::new(artifact_idempotency_store(
                &directory.join("idempotency.json"),
            )?),
        )?;
        assert!(Arc::ptr_eq(&runtime.presentation(), &presentation));
        let host = runtime.host();
        let features = host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Get, "/features"),
            context(70),
        );
        assert_eq!(features.status, 200);
        assert_eq!(
            body_json(&features)
                .and_then(|value| value.pointer("/zed_native_api/native_execution"))
                .and_then(Value::as_bool),
            Some(true)
        );
        let later_owned = host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Get, "/api/assets"),
            context(71),
        );
        assert_eq!(later_owned.status, 501);
        assert!(
            body_json(&later_owned)
                .and_then(|value| value.pointer("/error/message"))
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("later native Rust service"))
        );

        let requested_client_id = ClientId::new("native-client")?;
        let client_id = host.connect_websocket(
            requested_client_id.clone(),
            ReconnectProjection::default(),
            BTreeMap::new(),
            context(72),
        )?;
        assert_eq!(host.drain_websocket(&client_id)?.len(), 1);
        let prompt = json!({
            "prompt_id": "00000000-0000-0000-0000-000000000101",
            "number": 7,
            "client_id": "native-client",
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
            }
        });
        let prompt_response = host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Post, "/prompt")
                .with_header("content-type", "application/json")
                .with_header("idempotency-key", "runtime-builder-submit")
                .with_body(serde_json::to_vec(&prompt)?),
            context(73),
        );
        assert_eq!(prompt_response.status, 200);
        let prompt_body = body_json(&prompt_response).ok_or("prompt response was not JSON")?;
        let prompt_id: sim_runtime::PromptId =
            serde_json::from_value(prompt_body["prompt_id"].clone())?;
        let attempt_id: sim_runtime::AttemptId =
            serde_json::from_value(prompt_body["attempt_id"].clone())?;
        let at = sim_runtime::AttemptRecord::queued(profile_id, prompt_id, attempt_id).created_at;
        let event = smol::block_on(presentation.apply_actuator_event_batch_durable(
            profile_id,
            prompt_id,
            attempt_id,
            &[sim_runtime::ExecutionActuatorEventInput {
                node_id: None,
                kind: sim_runtime::AttemptEventKind::Started,
                data: None,
                at,
            }],
        ))?
        .into_iter()
        .next()
        .ok_or("durable actuator event batch returned no event")?;
        event_bus.publish(event)?;
        let mut bridged = false;
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            if host
                .drain_websocket(&client_id)?
                .iter()
                .any(|message| message.event_type == "execution_start")
            {
                bridged = true;
                break;
            }
        }
        assert!(
            bridged,
            "runtime event was not bridged to the native WebSocket client"
        );
        let snapshot = presentation.snapshot(profile_id)?;
        assert_eq!(snapshot.attempts[0].canonical_event_count, 1);
        assert!(host.disconnect_websocket(&client_id)?);
        let reconnect_projection = runtime
            .server_config("127.0.0.1:0".parse()?)
            .reconnect_projection;
        let reconnected_client_id = host.connect_websocket_projected(
            requested_client_id,
            BTreeMap::new(),
            context(74),
            |principal, client_id| reconnect_projection(principal, client_id),
        )?;
        assert_eq!(reconnected_client_id, client_id);
        let reconciled = host.drain_websocket(&reconnected_client_id)?;
        assert_eq!(reconciled[0].event_type, "status");
        assert!(
            reconciled
                .iter()
                .any(|message| message.event_type == "execution_start"),
            "reconnect did not project the client's in-flight execution"
        );
        let unrelated_requested_client_id = ClientId::new("other-native-client")?;
        let unrelated_client_id = host.connect_websocket_projected(
            unrelated_requested_client_id,
            BTreeMap::new(),
            context(75),
            |principal, client_id| reconnect_projection(principal, client_id),
        )?;
        let unrelated_projection = host.drain_websocket(&unrelated_client_id)?;
        assert_eq!(
            unrelated_projection
                .iter()
                .map(|message| message.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["status"]
        );
        assert!(host.disconnect_websocket(&reconnected_client_id)?);
        assert!(host.disconnect_websocket(&unrelated_client_id)?);
        drop(host);
        drop(runtime);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    pub(crate) fn cors_preflight_and_error_responses_use_exact_origin_policy()
    -> Result<(), Box<dyn Error>> {
        let directory = test_directory("cors")?;
        let services = Arc::new(ProbeServices::default());
        let mut security = security::ApiSecurityConfig::loopback();
        security
            .allowed_origins
            .insert("https://automation.example".into());
        let host = NativeApiHost::new(
            "profile-a",
            services,
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security.clone(),
            permission_policy("profile-a", &security)?,
            Arc::new(artifact_idempotency_store(
                &directory.join("idempotency.json"),
            )?),
        )?;
        let preflight_headers = BTreeMap::from([
            ("origin".into(), vec!["https://automation.example".into()]),
            ("access-control-request-method".into(), vec!["GET".into()]),
            (
                "access-control-request-headers".into(),
                vec!["authorization, content-type".into()],
            ),
        ]);
        let preflight = host.handle_preflight("/system_stats", &preflight_headers, context(72));
        assert_eq!(preflight.status, 204);
        assert_eq!(
            preflight.headers.get("access-control-allow-origin"),
            Some(&"https://automation.example".into())
        );
        assert_eq!(
            preflight.headers.get("access-control-allow-methods"),
            Some(&"GET".into())
        );

        let mut error_request = HttpRequest::new(sim_types::HttpMethod::Get, "/system_stats")
            .with_header("origin", "https://automation.example");
        for value in 0..=HttpLimits::default().maximum_query_values {
            error_request = error_request.with_query("overflow", value.to_string());
        }
        let error_response = host.handle_http(error_request, context(73));
        assert_eq!(error_response.status, 413);
        assert_eq!(
            error_response.headers.get("access-control-allow-origin"),
            Some(&"https://automation.example".into())
        );

        let denied_headers = BTreeMap::from([
            ("origin".into(), vec!["https://denied.example".into()]),
            ("access-control-request-method".into(), vec!["GET".into()]),
        ]);
        let denied = host.handle_preflight("/system_stats", &denied_headers, context(74));
        assert_eq!(denied.status, 403);
        assert!(!denied.headers.contains_key("access-control-allow-origin"));
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    pub(crate) fn principals_plugins_and_websocket_clients_are_strictly_isolated()
    -> Result<(), Box<dyn Error>> {
        let directory = test_directory("authority-isolation")?;
        let token_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let token_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let mut security = security::ApiSecurityConfig::loopback();
        security.require_authentication = true;
        security.credentials = vec![
            security::BearerCredential::new(
                "principal-a",
                token_a,
                ["api:read".to_owned(), "api:write".to_owned()],
                None,
            )?,
            security::BearerCredential::new(
                "principal-b",
                token_b,
                ["api:read".to_owned(), "api:write".to_owned()],
                None,
            )?,
        ];
        security
            .plugin_route_grants
            .push(security::PluginRouteGrant {
                profile_id: "profile-a".into(),
                principal: "principal-a".into(),
                plugin_id: "provider-plugin".into(),
                plugin_digest: "sha256:provider-plugin".into(),
                methods: ["POST".into()].into_iter().collect(),
                route_prefixes: ["/api/jobs/cancel".into()].into_iter().collect(),
                capabilities: [
                    "provider_network:payment|https://payment.invalid/api/jobs/cancel".into(),
                ]
                .into_iter()
                .collect(),
            });
        let services = Arc::new(ProbeServices::default());
        let permission_policy = permission_policy("profile-a", &security)?;
        let host = NativeApiHost::new(
            "profile-a",
            services.clone(),
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security,
            permission_policy,
            Arc::new(artifact_idempotency_store(
                &directory.join("idempotency.json"),
            )?),
        )?;
        let request_for = |token: &str| {
            prompt_request("shared-client-key")
                .with_header("authorization", format!("Bearer {token}"))
        };
        assert_eq!(
            host.route_http(request_for(token_a), context(80))?.status,
            200
        );
        assert_eq!(
            host.route_http(request_for(token_a), context(81))?.status,
            200
        );
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            host.route_http(request_for(token_b), context(82))?.status,
            200
        );
        assert_eq!(
            host.route_http(request_for(token_b), context(83))?.status,
            200
        );
        assert_eq!(services.calls.load(Ordering::SeqCst), 2);

        let plugin_request = |token: &str| {
            HttpRequest::new(sim_types::HttpMethod::Post, "/api/jobs/cancel")
                .with_header("authorization", format!("Bearer {token}"))
                .with_header("content-type", "application/json")
                .with_header("idempotency-key", format!("plugin-{token}"))
                .with_header("x-zed-plugin-profile", "profile-a")
                .with_header("x-zed-plugin-id", "provider-plugin")
                .with_header("x-zed-plugin-digest", "sha256:provider-plugin")
                .with_header(
                    "x-zed-plugin-capabilities",
                    "provider_network:payment|https://payment.invalid/api/jobs/cancel",
                )
                .with_body(bytes::Bytes::from_static(br#"{}"#))
        };
        assert_eq!(
            host.route_http(plugin_request(token_a), context(84))?
                .status,
            200
        );
        assert_eq!(services.calls.load(Ordering::SeqCst), 3);
        assert!(matches!(
            host.route_http(plugin_request(token_b), context(85)),
            Err(NativeApiHostError::Security { status: 403, .. })
        ));
        assert_eq!(services.calls.load(Ordering::SeqCst), 3);

        let requested_client_id = ClientId::new("shared-client")?;
        let websocket_headers = |token: &str| {
            BTreeMap::from([("authorization".into(), vec![format!("Bearer {token}")])])
        };
        let client_a = host.connect_websocket(
            requested_client_id.clone(),
            ReconnectProjection::default(),
            websocket_headers(token_a),
            context(86),
        )?;
        assert!(matches!(
            host.connect_websocket(
                requested_client_id.clone(),
                ReconnectProjection::default(),
                websocket_headers(token_a),
                context(87),
            ),
            Err(NativeApiHostError::Security { status: 409, .. })
        ));
        let client_b = host.connect_websocket(
            requested_client_id,
            ReconnectProjection::default(),
            websocket_headers(token_b),
            context(88),
        )?;
        assert_ne!(client_a, client_b);
        let status_a = host.drain_websocket(&client_a)?;
        let status_b = host.drain_websocket(&client_b)?;
        assert_eq!(status_a.len(), 1);
        assert_eq!(status_b.len(), 1);
        assert_eq!(
            serde_json::from_slice::<Value>(&status_a[0].payload)?["data"]["sid"],
            "shared-client"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&status_b[0].payload)?["data"]["sid"],
            "shared-client"
        );
        assert!(matches!(
            host.drain_websocket(&ClientId::new("shared-client")?),
            Err(NativeApiHostError::WebSocket(_))
        ));
        assert!(host.disconnect_websocket(&client_a)?);
        assert!(host.disconnect_websocket(&client_b)?);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    fn runtime_recovery_host(
        profile_id: sim_runtime::ProfileId,
        presentation: sim_runtime::SharedExecutionPresentationService,
        store: Arc<dyn security::IdempotencySnapshotStore>,
    ) -> Result<NativeApiHost<NativeRuntimeHttpServices>, Box<dyn Error>> {
        let services = NativeRuntimeHttpServices::native_image(
            profile_id,
            presentation,
            Arc::new(crate::services::AcceptingExecutionController),
        )?;
        let capabilities = services.http_capabilities()?;
        let security_config = security::ApiSecurityConfig::loopback();
        NativeApiHost::new(
            profile_id.0.to_string(),
            Arc::new(services),
            HttpLimits::default(),
            capabilities,
            WebSocketLimits::default(),
            security_config.clone(),
            permission_policy(&profile_id.0.to_string(), &security_config)?,
            store,
        )
        .map_err(Into::into)
    }

    fn recovery_prompt_request(key: Option<&str>, prompt_id: &str) -> HttpRequest {
        let request = HttpRequest::new(sim_types::HttpMethod::Post, "/prompt")
            .with_header("content-type", "application/json")
            .with_body(bytes::Bytes::from(
                serde_json::to_vec(&json!({
                    "prompt_id": prompt_id,
                    "number": 7,
                    "prompt": {
                        "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
                    }
                }))
                .expect("static native prompt fixture must encode"),
            ));
        key.map_or(request.clone(), |key| {
            request.with_header("idempotency-key", key)
        })
    }

    fn canonical_runtime_state(
        presentation: &sim_runtime::SharedExecutionPresentationService,
        profile_id: sim_runtime::ProfileId,
    ) -> Result<(usize, usize, usize), Box<dyn Error>> {
        let snapshot = presentation.snapshot(profile_id)?;
        Ok((
            snapshot.attempts.len(),
            snapshot.queue.len(),
            snapshot.recent_command_results.len(),
        ))
    }

    fn real_host_recovery_cases(
        directory: &Path,
    ) -> Result<(Vec<Value>, Vec<bool>), Box<dyn Error>> {
        let profile_id = sim_runtime::ProfileId("00000000-0000-0000-0000-000000000026".parse()?);
        let mut outcomes = Vec::new();

        let prepare_presentation = test_presentation(profile_id)?;
        let prepare_host = runtime_recovery_host(
            profile_id,
            prepare_presentation.clone(),
            Arc::new(FailingSnapshotStore),
        )?;
        let prepare_request = recovery_prompt_request(
            Some("real-prompt-prepare-failure"),
            "00000000-0000-0000-0000-000000002601",
        );
        let prepare_result = prepare_host.route_http(prepare_request, context(100));
        let prepare_state = canonical_runtime_state(&prepare_presentation, profile_id)?;
        let prepare_passed = matches!(prepare_result, Err(NativeApiHostError::Persistence(_)))
            && prepare_state == (0, 0, 0);
        outcomes.push(recovery_outcome(
            "prepare_persistence_fails_before_canonical_effect",
            prepare_passed,
            prepare_state.0,
            json!({
                "canonical_state": prepare_state,
                "persistence_failed": matches!(
                    prepare_result,
                    Err(NativeApiHostError::Persistence(_))
                ),
            }),
        ));

        let interrupted_presentation = test_presentation(profile_id)?;
        let interrupted_store = Arc::new(FailOnceAfterPreparedSnapshotStore::default());
        let interrupted_request = recovery_prompt_request(
            Some("real-prompt-interrupted-terminal"),
            "00000000-0000-0000-0000-000000002602",
        );
        let interrupted_host = runtime_recovery_host(
            profile_id,
            interrupted_presentation.clone(),
            interrupted_store.clone(),
        )?;
        let interrupted_result =
            interrupted_host.route_http(interrupted_request.clone(), context(101));
        let state_after_effect = canonical_runtime_state(&interrupted_presentation, profile_id)?;
        drop(interrupted_host);
        let recovered_host = runtime_recovery_host(
            profile_id,
            interrupted_presentation.clone(),
            interrupted_store,
        )?;
        let recovered_response = recovered_host.route_http(interrupted_request, context(102))?;
        let recovered_state = canonical_runtime_state(&interrupted_presentation, profile_id)?;
        let interrupted_passed =
            matches!(interrupted_result, Err(NativeApiHostError::Persistence(_)))
                && recovered_response.status == 200
                && state_after_effect.0 == 1
                && recovered_state.0 == 1
                && recovered_state.1 == 1
                && recovered_state.2 == 1;
        outcomes.push(recovery_outcome(
            "committed_prompt_reconciles_after_terminal_persistence_loss",
            interrupted_passed,
            recovered_state.0,
            json!({
                "first_response_lost": matches!(
                    interrupted_result,
                    Err(NativeApiHostError::Persistence(_))
                ),
                "restart_status": recovered_response.status,
                "state_after_effect": state_after_effect,
                "state_after_reconciliation": recovered_state,
            }),
        ));

        let replay_presentation = test_presentation(profile_id)?;
        let replay_path = directory.join("real-prompt-committed-replay.json");
        let replay_store = Arc::new(artifact_idempotency_store(&replay_path)?);
        let replay_request = recovery_prompt_request(
            Some("real-prompt-committed-replay"),
            "00000000-0000-0000-0000-000000002603",
        );
        let replay_host = runtime_recovery_host(
            profile_id,
            replay_presentation.clone(),
            replay_store.clone(),
        )?;
        let initial_response = replay_host.route_http(replay_request.clone(), context(103))?;
        let initial_body = body_json(&initial_response).cloned();
        drop(replay_host);
        let replay_host =
            runtime_recovery_host(profile_id, replay_presentation.clone(), replay_store)?;
        let replay_response = replay_host.route_http(replay_request, context(104))?;
        let replay_state = canonical_runtime_state(&replay_presentation, profile_id)?;
        let replay_passed = initial_response.status == 200
            && replay_response.status == 200
            && body_json(&replay_response).cloned() == initial_body
            && replay_state.0 == 1
            && replay_state.2 == 1;
        outcomes.push(recovery_outcome(
            "committed_response_replays_after_restart",
            replay_passed,
            replay_state.0,
            json!({
                "initial_status": initial_response.status,
                "replay_status": replay_response.status,
                "canonical_state": replay_state,
            }),
        ));

        let not_applied_presentation = test_presentation(profile_id)?;
        let not_applied_path = directory.join("real-prompt-not-applied.json");
        let not_applied_store = Arc::new(artifact_idempotency_store(&not_applied_path)?);
        let not_applied_request = recovery_prompt_request(
            Some("real-prompt-not-applied"),
            "00000000-0000-0000-0000-000000002604",
        );
        let matched = match_http_route(not_applied_request.method, &not_applied_request.path)?
            .ok_or("native prompt route is unavailable")?;
        let authority = NativeRequestAuthority {
            profile_id: profile_id.0.to_string(),
            principal: "anonymous:127.0.0.1".into(),
            scopes: Default::default(),
            plugin_id: None,
            plugin_digest: None,
        };
        let client_key = "real-prompt-not-applied";
        let scoped_key = scoped_idempotency_key(client_key, &authority);
        let operation_id =
            scoped_operation_id(client_key, &matched.canonical_feature_id, &authority);
        let digest = stable_request_digest(
            &not_applied_request,
            &matched.canonical_feature_id,
            &authority,
        );
        let mut prepared_ledger = security::IdempotencyLedger::new(
            profile_id.0.to_string(),
            HttpLimits::default().idempotency_capacity,
            HttpLimits::default().maximum_response_bytes,
        )?;
        assert_eq!(
            prepared_ledger.begin(scoped_key, operation_id, digest, 105)?,
            security::IdempotencyDecision::Begin
        );
        security::IdempotencySnapshotStore::save(&*not_applied_store, &prepared_ledger.snapshot())?;
        let not_applied_host = runtime_recovery_host(
            profile_id,
            not_applied_presentation.clone(),
            not_applied_store,
        )?;
        let not_applied_response =
            not_applied_host.route_http(not_applied_request, context(106))?;
        let not_applied_state = canonical_runtime_state(&not_applied_presentation, profile_id)?;
        let not_applied_passed = not_applied_response.status == 200
            && not_applied_state.0 == 1
            && not_applied_state.1 == 1
            && not_applied_state.2 == 1;
        outcomes.push(recovery_outcome(
            "prepared_but_not_applied_reopens_once",
            not_applied_passed,
            not_applied_state.0,
            json!({
                "restart_status": not_applied_response.status,
                "canonical_state": not_applied_state,
            }),
        ));

        let baseline_presentation = test_presentation(profile_id)?;
        let baseline_path = directory.join("baseline-compatible-prompt.json");
        let baseline_host = runtime_recovery_host(
            profile_id,
            baseline_presentation.clone(),
            Arc::new(artifact_idempotency_store(&baseline_path)?),
        )?;
        let baseline_response = baseline_host.route_http(
            recovery_prompt_request(None, "00000000-0000-0000-0000-000000002605"),
            context(107),
        )?;
        let baseline_before_unavailable =
            canonical_runtime_state(&baseline_presentation, profile_id)?;
        let unavailable_output = baseline_host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Post, "/upload/image")
                .with_header("content-type", "application/octet-stream")
                .with_body(bytes::Bytes::from_static(b"unavailable-output")),
            context(108),
        );
        let unavailable_delete = baseline_host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Delete, "/userdata/unavailable.txt"),
            context(109),
        );
        let unavailable_payment = baseline_host.handle_http(
            HttpRequest::new(sim_types::HttpMethod::Post, "/api/payment")
                .with_header("content-type", "application/json")
                .with_body(bytes::Bytes::from_static(br#"{}"#)),
            context(110),
        );
        let baseline_after_unavailable =
            canonical_runtime_state(&baseline_presentation, profile_id)?;
        let boundary_passed = baseline_response.status == 200
            && baseline_before_unavailable == baseline_after_unavailable
            && unavailable_output.status >= 400
            && unavailable_delete.status >= 400
            && unavailable_payment.status == 404;
        outcomes.push(recovery_outcome(
            "baseline_compatibility_and_unavailable_mutation_boundaries",
            boundary_passed,
            baseline_after_unavailable.0,
            json!({
                "baseline_prompt_status": baseline_response.status,
                "output_route_status": unavailable_output.status,
                "delete_route_status": unavailable_delete.status,
                "payment_route_status": unavailable_payment.status,
                "canonical_state_before": baseline_before_unavailable,
                "canonical_state_after": baseline_after_unavailable,
                "disclosure": "output upload, userdata deletion, and provider payment have no enabled canonical Task 26 mutation owner; they fail before side effects and are not represented by synthetic substitutes",
            }),
        ));

        let checks = outcomes.iter().map(|outcome| outcome.passed).collect();
        let artifacts = outcomes
            .into_iter()
            .map(|outcome| outcome.artifact)
            .collect();
        Ok((artifacts, checks))
    }

    #[test]
    fn val_recovery_004() -> Result<(), Box<dyn Error>> {
        let directory = test_directory("recovery")?;
        let (canonical_host_cases, host_checks) = real_host_recovery_cases(&directory)?;

        let generic_catalog_outcomes = http_route_catalog()?
            .iter()
            .filter(|route| route.is_mutation())
            .map(|route| durable_catalog_case(route.feature_id()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut generic_checks = Vec::new();
        let generic_catalog_cases = generic_catalog_outcomes
            .into_iter()
            .map(|(artifact, checks)| {
                generic_checks.extend(checks);
                artifact
            })
            .collect::<Vec<_>>();
        let passed = host_checks
            .iter()
            .chain(&generic_checks)
            .filter(|passed| **passed)
            .count();
        let failed = host_checks.len() + generic_checks.len() - passed;

        let artifact = json!({
            "schema_version": 2,
            "validation_id": "VAL-RECOVERY-004",
            "fixture": "inject native idempotency persistence loss around real canonical prompt submission and restart reconciliation",
            "environment": {
                "backend": "native-rust-services",
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
            },
            "fixture_digests": {
                "backend_http_routes_fnv1a64": format!("{:016x}", fnv1a64(include_bytes!("../../../.agents/specs/comfy-parity/catalogs/backend-http-routes.csv"))),
            },
            "summary": {"passed": passed, "failed": failed, "skipped": 0},
            "canonical_host_cases": canonical_host_cases,
            "generic_ledger_transition_coverage": {
                "scope": "computed ledger transition checks for every mutating catalog row; not claimed as side-effect evidence",
                "cases": generic_catalog_cases,
            },
            "unsupported_mutation_disclosure": {
                "output_upload": "explicit capability-unavailable response before dispatch",
                "userdata_delete": "explicit capability-unavailable response before dispatch",
                "provider_payment": "no cataloged endpoint; route-not-found before dispatch",
                "synthetic_surrogates": false,
            },
            "skipped": [],
        });
        write_validation_artifact("val-recovery-004.json", &artifact)?;
        fs::remove_dir_all(directory)?;
        assert_eq!(failed, 0, "all recorded VAL-RECOVERY-004 checks must pass");
        Ok(())
    }
    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325_u64, |digest, byte| {
            (digest ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    }
}
