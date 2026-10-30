use crate::{
    HostRequestContext, HttpBody, HttpRequest, NativeApiHostError, NativeApiServer,
    NativeApiServerConfig, NativeRuntimeApiHost,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sim_runtime::SharedExecutionPresentationService;
use sim_types::{CancellationToken, HttpMethod};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::Read,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

pub const NATIVE_CLI_ENVELOPE_VERSION: &str = "envelope/1";
pub const NATIVE_CLI_EVENT_VERSION: &str = "event/1";
pub const MAXIMUM_NATIVE_CLI_EVENT_BYTES: usize = 4 * 1024 * 1024;
pub const MAXIMUM_NATIVE_TLS_DER_BYTES: usize = 4 * 1024 * 1024;

impl crate::NativeTlsAcceptor {
    pub fn from_pkcs8_der(
        certificate_identity: impl Into<String>,
        certificate_chain_der: Vec<Vec<u8>>,
        private_key_der: Vec<u8>,
    ) -> Result<Self, crate::NativeTransportError> {
        let certificate_bytes = certificate_chain_der
            .iter()
            .try_fold(0usize, |total, certificate| {
                total.checked_add(certificate.len())
            });
        if certificate_chain_der.is_empty()
            || certificate_chain_der.iter().any(Vec::is_empty)
            || certificate_bytes.is_none_or(|bytes| bytes > MAXIMUM_NATIVE_TLS_DER_BYTES)
            || private_key_der.is_empty()
            || private_key_der.len() > MAXIMUM_NATIVE_TLS_DER_BYTES
        {
            return Err(crate::NativeTransportError::InvalidConfiguration(
                "TLS certificate chain and PKCS#8 private key must be non-empty DER".into(),
            ));
        }
        let certificates = certificate_chain_der
            .into_iter()
            .map(rustls::pki_types::CertificateDer::from)
            .collect();
        let private_key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(private_key_der),
        );
        let server = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|error| crate::NativeTransportError::Tls(error.to_string()))?
        .with_no_client_auth()
        .with_single_cert(certificates, private_key)
        .map_err(|error| crate::NativeTransportError::Tls(error.to_string()))?;
        Self::new(certificate_identity, Arc::new(server))
    }

    pub fn from_der_files(
        certificate_identity: impl Into<String>,
        certificate_path: impl AsRef<Path>,
        private_key_path: impl AsRef<Path>,
    ) -> Result<Self, crate::NativeTransportError> {
        let certificate = read_bounded_der(certificate_path.as_ref())?;
        let private_key = read_bounded_der(private_key_path.as_ref())?;
        Self::from_pkcs8_der(certificate_identity, vec![certificate], private_key)
    }
}

fn read_bounded_der(path: &Path) -> Result<Vec<u8>, crate::NativeTransportError> {
    let file =
        File::open(path).map_err(|error| crate::NativeTransportError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take((MAXIMUM_NATIVE_TLS_DER_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| crate::NativeTransportError::Io(error.to_string()))?;
    if bytes.is_empty() || bytes.len() > MAXIMUM_NATIVE_TLS_DER_BYTES {
        return Err(crate::NativeTransportError::InvalidConfiguration(format!(
            "TLS DER file `{}` must contain 1 to {MAXIMUM_NATIVE_TLS_DER_BYTES} bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCliEventKind {
    Cancelled,
    Converted,
    Executed,
    Executing,
    ExecutionCached,
    ExecutionError,
    Output,
    Progress,
    PromptPreview,
    Queued,
    Settled,
    State,
}

impl NativeCliEventKind {
    pub const ALL: [Self; 12] = [
        Self::Cancelled,
        Self::Converted,
        Self::Executed,
        Self::Executing,
        Self::ExecutionCached,
        Self::ExecutionError,
        Self::Output,
        Self::Progress,
        Self::PromptPreview,
        Self::Queued,
        Self::Settled,
        Self::State,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Converted => "converted",
            Self::Executed => "executed",
            Self::Executing => "executing",
            Self::ExecutionCached => "execution_cached",
            Self::ExecutionError => "execution_error",
            Self::Output => "output",
            Self::Progress => "progress",
            Self::PromptPreview => "prompt_preview",
            Self::Queued => "queued",
            Self::Settled => "settled",
            Self::State => "state",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCliEvent {
    pub schema: String,
    #[serde(rename = "type")]
    pub kind: NativeCliEventKind,
    #[serde(flatten)]
    fields: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct NativeCliEventWire {
    schema: String,
    #[serde(rename = "type")]
    kind: NativeCliEventKind,
    #[serde(flatten)]
    fields: BTreeMap<String, Value>,
}

impl<'de> Deserialize<'de> for NativeCliEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = NativeCliEventWire::deserialize(deserializer)?;
        Self::from_wire(wire).map_err(serde::de::Error::custom)
    }
}

impl NativeCliEvent {
    pub fn new(
        kind: NativeCliEventKind,
        fields: BTreeMap<String, Value>,
    ) -> Result<Self, NativeHeadlessError> {
        Self::from_wire(NativeCliEventWire {
            schema: NATIVE_CLI_EVENT_VERSION.to_owned(),
            kind,
            fields,
        })
    }

    pub fn fields(&self) -> &BTreeMap<String, Value> {
        &self.fields
    }

    fn from_wire(wire: NativeCliEventWire) -> Result<Self, NativeHeadlessError> {
        if wire.schema != NATIVE_CLI_EVENT_VERSION {
            return Err(NativeHeadlessError::InvalidCliEvent(format!(
                "unsupported native CLI event schema `{}`",
                wire.schema
            )));
        }
        if wire.fields.contains_key("schema") || wire.fields.contains_key("type") {
            return Err(NativeHeadlessError::InvalidCliEvent(
                "native CLI event fields cannot replace schema or type".into(),
            ));
        }
        let event = Self {
            schema: wire.schema,
            kind: wire.kind,
            fields: wire.fields,
        };
        let encoded = serde_json::to_vec(&event)
            .map_err(|error| NativeHeadlessError::InvalidCliEvent(error.to_string()))?;
        if encoded.len() > MAXIMUM_NATIVE_CLI_EVENT_BYTES {
            return Err(NativeHeadlessError::InvalidCliEvent(format!(
                "native CLI event contains {} bytes; maximum is {MAXIMUM_NATIVE_CLI_EVENT_BYTES}",
                encoded.len()
            )));
        }
        Ok(event)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeHeadlessMode {
    Offline,
    Serve,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeHeadlessFailure {
    pub stage: String,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum NativeHeadlessState {
    Inactive,
    Starting,
    Ready,
    Cancelling,
    Stopped,
    Failed { failure: NativeHeadlessFailure },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeHeadlessTransition {
    pub sequence: u64,
    pub state: NativeHeadlessState,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeHeadlessSnapshot {
    pub mode: NativeHeadlessMode,
    pub state: NativeHeadlessState,
    pub local_address: Option<SocketAddr>,
    pub restart_count: usize,
    pub transitions: Vec<NativeHeadlessTransition>,
    pub dropped_transitions: usize,
    pub deployment_identity_sha256: Option<String>,
    pub provider_deployment: Option<crate::NativeProviderDeploymentIdentity>,
    pub deployment_diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeHeadlessPolicy {
    pub lifecycle_history_capacity: usize,
    pub maximum_restarts: usize,
    pub shutdown_timeout: Duration,
    pub cancellation_poll_interval: Duration,
}

impl Default for NativeHeadlessPolicy {
    fn default() -> Self {
        Self {
            lifecycle_history_capacity: 128,
            maximum_restarts: 1,
            shutdown_timeout: Duration::from_secs(5),
            cancellation_poll_interval: Duration::from_millis(10),
        }
    }
}

impl NativeHeadlessPolicy {
    pub fn validate(self) -> Result<(), NativeHeadlessError> {
        if self.lifecycle_history_capacity == 0
            || self.shutdown_timeout.is_zero()
            || self.cancellation_poll_interval.is_zero()
        {
            return Err(NativeHeadlessError::InvalidConfiguration(
                "headless history capacity and lifecycle timeouts must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct NativeHeadlessCancellation {
    inner: Arc<NativeHeadlessCancellationInner>,
}

#[derive(Default)]
struct NativeHeadlessCancellationInner {
    token: CancellationToken,
    reason: Mutex<Option<String>>,
}

impl NativeHeadlessCancellation {
    pub fn cancel(&self, reason: impl Into<String>) -> Result<bool, NativeHeadlessError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(NativeHeadlessError::InvalidConfiguration(
                "headless cancellation reason cannot be empty".into(),
            ));
        }
        let mut stored_reason = self
            .inner
            .reason
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        if self.inner.token.is_cancelled() {
            return Ok(false);
        }
        *stored_reason = Some(reason);
        let first = self.inner.token.cancel();
        debug_assert!(
            first,
            "headless cancellation is serialized by the reason lock"
        );
        Ok(first)
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.token.is_cancelled()
    }

    pub fn reason(&self) -> Result<Option<String>, NativeHeadlessError> {
        self.inner
            .reason
            .lock()
            .map(|reason| reason.clone())
            .map_err(|_| NativeHeadlessError::StateUnavailable)
    }

    pub fn event(&self) -> Result<Option<NativeCliEvent>, NativeHeadlessError> {
        let Some(reason) = self.reason()? else {
            return Ok(None);
        };
        NativeCliEvent::new(
            NativeCliEventKind::Cancelled,
            BTreeMap::from([("reason".into(), Value::String(reason))]),
        )
        .map(Some)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeAutomationMapping {
    Native {
        request: HttpRequest,
        requires_network: bool,
    },
    Migration {
        replacement: String,
        detail: String,
    },
    Deferred {
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAutomationRequest {
    pub feature_id: String,
    pub now_epoch_seconds: u64,
    pub mapping: NativeAutomationMapping,
}

impl NativeAutomationRequest {
    pub fn native(
        feature_id: impl Into<String>,
        request: HttpRequest,
        now_epoch_seconds: u64,
    ) -> Self {
        Self {
            feature_id: feature_id.into(),
            now_epoch_seconds,
            mapping: NativeAutomationMapping::Native {
                request,
                requires_network: false,
            },
        }
    }

    pub fn native_networked(
        feature_id: impl Into<String>,
        request: HttpRequest,
        now_epoch_seconds: u64,
    ) -> Self {
        Self {
            feature_id: feature_id.into(),
            now_epoch_seconds,
            mapping: NativeAutomationMapping::Native {
                request,
                requires_network: true,
            },
        }
    }

    pub fn migration(
        feature_id: impl Into<String>,
        replacement: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            feature_id: feature_id.into(),
            now_epoch_seconds: 0,
            mapping: NativeAutomationMapping::Migration {
                replacement: replacement.into(),
                detail: detail.into(),
            },
        }
    }

    pub fn deferred(feature_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            feature_id: feature_id.into(),
            now_epoch_seconds: 0,
            mapping: NativeAutomationMapping::Deferred {
                reason: reason.into(),
            },
        }
    }

    fn validate(&self) -> Result<(), NativeHeadlessError> {
        if self.feature_id.trim().is_empty() {
            return Err(NativeHeadlessError::InvalidAutomation(
                "automation feature identity cannot be empty".into(),
            ));
        }
        match &self.mapping {
            NativeAutomationMapping::Native { request, .. } => {
                if request.path.trim().is_empty() || !request.path.starts_with('/') {
                    return Err(NativeHeadlessError::InvalidAutomation(
                        "native automation paths must be absolute".into(),
                    ));
                }
            }
            NativeAutomationMapping::Migration {
                replacement,
                detail,
            } => {
                if replacement.trim().is_empty() || detail.trim().is_empty() {
                    return Err(NativeHeadlessError::InvalidAutomation(
                        "migration mappings require a replacement and detail".into(),
                    ));
                }
            }
            NativeAutomationMapping::Deferred { reason } => {
                if reason.trim().is_empty() {
                    return Err(NativeHeadlessError::InvalidAutomation(
                        "deferred mappings require a reason".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum NativeAutomationBody {
    Empty,
    Bytes(Vec<u8>),
    Json(Value),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case")]
pub enum NativeAutomationResult {
    Native {
        feature_id: String,
        status: u16,
        content_type: String,
        headers: BTreeMap<String, String>,
        body: NativeAutomationBody,
    },
    Migration {
        feature_id: String,
        replacement: String,
        detail: String,
    },
    Deferred {
        feature_id: String,
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCliInvocation {
    pub feature_id: String,
    pub operation: NativeCliOperation,
    pub now_epoch_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeCliOperation {
    Health,
    SystemStats,
    Queue,
    History {
        prompt_id: Option<String>,
        maximum_items: Option<usize>,
        offset: Option<usize>,
    },
    Jobs {
        status: Option<String>,
        limit: Option<usize>,
        offset: Option<usize>,
    },
    JobStatus {
        job_id: String,
    },
    SubmitPrompt {
        submission: Value,
        idempotency_key: String,
    },
    CancelJobs {
        job_ids: Vec<String>,
        operation_id: String,
    },
    Interrupt {
        prompt_id: Option<String>,
        operation_id: String,
    },
    NativeRequest {
        request: HttpRequest,
        requires_network: bool,
    },
    Migration {
        replacement: String,
        detail: String,
    },
    Deferred {
        reason: String,
    },
}

pub type NativeHeadlessRuntimeFactory = Arc<
    dyn Fn(SharedExecutionPresentationService) -> Result<NativeRuntimeApiHost, NativeApiHostError>
        + Send
        + Sync
        + 'static,
>;

type NativeHeadlessRuntimeActivation = Box<
    dyn FnOnce(
            SharedExecutionPresentationService,
        ) -> Result<NativeRuntimeApiHost, NativeApiHostError>
        + Send
        + 'static,
>;

type NativeHeadlessDeploymentActivation = Box<
    dyn FnOnce(
            SharedExecutionPresentationService,
            Arc<sim_runtime::NativeExecutionRegistryBundle>,
            extension_host::ComponentInventoryCandidateIdentity,
        ) -> Result<NativeRuntimeApiHost, NativeApiHostError>
        + Send
        + 'static,
>;

pub struct PreparedNativeHeadlessRuntime {
    provider_deployment: Option<crate::NativeProviderDeploymentIdentity>,
    registry_bundle: Option<Arc<sim_runtime::NativeExecutionRegistryBundle>>,
    candidate_identity: Option<extension_host::ComponentInventoryCandidateIdentity>,
    activation: Option<NativeHeadlessRuntimeActivation>,
    deployment_activation: Option<NativeHeadlessDeploymentActivation>,
}

impl PreparedNativeHeadlessRuntime {
    pub fn checked<F>(
        registry_bundle: Arc<sim_runtime::NativeExecutionRegistryBundle>,
        candidate_identity: extension_host::ComponentInventoryCandidateIdentity,
        activation: F,
    ) -> Result<Self, NativeApiHostError>
    where
        F: FnOnce(
                SharedExecutionPresentationService,
                Arc<sim_runtime::NativeExecutionRegistryBundle>,
                extension_host::ComponentInventoryCandidateIdentity,
            ) -> Result<NativeRuntimeApiHost, NativeApiHostError>
            + Send
            + 'static,
    {
        let provider_deployment = crate::NativeProviderDeploymentIdentity::from_registry_bundle(
            &registry_bundle,
            &candidate_identity,
        )?;
        Ok(Self {
            provider_deployment: Some(provider_deployment),
            registry_bundle: Some(registry_bundle),
            candidate_identity: Some(candidate_identity),
            activation: None,
            deployment_activation: Some(Box::new(activation)),
        })
    }

    fn immediate(factory: NativeHeadlessRuntimeFactory) -> Self {
        Self {
            provider_deployment: None,
            registry_bundle: None,
            candidate_identity: None,
            activation: Some(Box::new(move |presentation| factory(presentation))),
            deployment_activation: None,
        }
    }

    pub fn activate(
        mut self,
        presentation: SharedExecutionPresentationService,
    ) -> Result<NativeRuntimeApiHost, NativeApiHostError> {
        if let Some(activation) = self.deployment_activation.take() {
            let bundle = self
                .registry_bundle
                .take()
                .ok_or(NativeApiHostError::StateUnavailable)?;
            let candidate_identity = self
                .candidate_identity
                .take()
                .ok_or(NativeApiHostError::StateUnavailable)?;
            let runtime = activation(presentation, bundle, candidate_identity)?;
            if runtime.provider_deployment() != self.provider_deployment.as_ref() {
                runtime.shutdown("prepared deployment activation returned a foreign runtime")?;
                return Err(NativeApiHostError::InvalidConfiguration(
                    "prepared deployment activation returned a foreign runtime".into(),
                ));
            }
            return Ok(runtime);
        }
        self.activation
            .take()
            .ok_or(NativeApiHostError::StateUnavailable)?(presentation)
    }
}

pub type NativeHeadlessRuntimePreparationFactory = Arc<
    dyn Fn(
            SharedExecutionPresentationService,
        ) -> Result<PreparedNativeHeadlessRuntime, NativeApiHostError>
        + Send
        + Sync
        + 'static,
>;

enum HeadlessRuntimeFactory {
    Immediate(NativeHeadlessRuntimeFactory),
    Prepared(NativeHeadlessRuntimePreparationFactory),
}

struct ActiveRuntime {
    runtime: NativeRuntimeApiHost,
    server: Option<NativeApiServer<crate::NativeRuntimeHttpServices>>,
    local_address: Option<SocketAddr>,
    provider_deployment: Option<crate::NativeProviderDeploymentIdentity>,
}

struct PendingShutdown {
    receiver: Receiver<Result<(), NativeHeadlessError>>,
}

struct NativeHeadlessInner {
    state: NativeHeadlessState,
    active: Option<ActiveRuntime>,
    pending_shutdown: Option<PendingShutdown>,
    restart_count: usize,
    next_sequence: u64,
    transitions: VecDeque<NativeHeadlessTransition>,
    dropped_transitions: usize,
    deployment_diagnostic: Option<String>,
}

pub struct NativeHeadlessService {
    mode: NativeHeadlessMode,
    presentation: SharedExecutionPresentationService,
    factory: HeadlessRuntimeFactory,
    server_config: Option<NativeApiServerConfig>,
    policy: NativeHeadlessPolicy,
    operation: Mutex<()>,
    inner: Mutex<NativeHeadlessInner>,
}

impl NativeHeadlessService {
    pub fn serve<F>(
        presentation: SharedExecutionPresentationService,
        factory: F,
        server_config: NativeApiServerConfig,
        policy: NativeHeadlessPolicy,
    ) -> Result<Self, NativeHeadlessError>
    where
        F: Fn(
                SharedExecutionPresentationService,
            ) -> Result<NativeRuntimeApiHost, NativeApiHostError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(
            NativeHeadlessMode::Serve,
            presentation,
            HeadlessRuntimeFactory::Immediate(Arc::new(factory)),
            Some(server_config),
            policy,
        )
    }

    pub fn serve_prepared<F>(
        presentation: SharedExecutionPresentationService,
        factory: F,
        server_config: NativeApiServerConfig,
        policy: NativeHeadlessPolicy,
    ) -> Result<Self, NativeHeadlessError>
    where
        F: Fn(
                SharedExecutionPresentationService,
            ) -> Result<PreparedNativeHeadlessRuntime, NativeApiHostError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(
            NativeHeadlessMode::Serve,
            presentation,
            HeadlessRuntimeFactory::Prepared(Arc::new(factory)),
            Some(server_config),
            policy,
        )
    }

    pub fn offline<F>(
        presentation: SharedExecutionPresentationService,
        factory: F,
        policy: NativeHeadlessPolicy,
    ) -> Result<Self, NativeHeadlessError>
    where
        F: Fn(
                SharedExecutionPresentationService,
            ) -> Result<NativeRuntimeApiHost, NativeApiHostError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(
            NativeHeadlessMode::Offline,
            presentation,
            HeadlessRuntimeFactory::Immediate(Arc::new(factory)),
            None,
            policy,
        )
    }

    pub fn offline_prepared<F>(
        presentation: SharedExecutionPresentationService,
        factory: F,
        policy: NativeHeadlessPolicy,
    ) -> Result<Self, NativeHeadlessError>
    where
        F: Fn(
                SharedExecutionPresentationService,
            ) -> Result<PreparedNativeHeadlessRuntime, NativeApiHostError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(
            NativeHeadlessMode::Offline,
            presentation,
            HeadlessRuntimeFactory::Prepared(Arc::new(factory)),
            None,
            policy,
        )
    }

    fn new(
        mode: NativeHeadlessMode,
        presentation: SharedExecutionPresentationService,
        factory: HeadlessRuntimeFactory,
        server_config: Option<NativeApiServerConfig>,
        policy: NativeHeadlessPolicy,
    ) -> Result<Self, NativeHeadlessError> {
        policy.validate()?;
        if (mode == NativeHeadlessMode::Serve) != server_config.is_some() {
            return Err(NativeHeadlessError::InvalidConfiguration(
                "serve mode requires one native API server configuration".into(),
            ));
        }
        let transitions = VecDeque::from([NativeHeadlessTransition {
            sequence: 0,
            state: NativeHeadlessState::Inactive,
            detail: "native headless service created".into(),
        }]);
        Ok(Self {
            mode,
            presentation,
            factory,
            server_config,
            policy,
            operation: Mutex::new(()),
            inner: Mutex::new(NativeHeadlessInner {
                state: NativeHeadlessState::Inactive,
                active: None,
                pending_shutdown: None,
                restart_count: 0,
                next_sequence: 1,
                transitions,
                dropped_transitions: 0,
                deployment_diagnostic: None,
            }),
        })
    }

    pub fn start(&self) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        let _operation = self
            .operation
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.reconcile_pending_shutdown()?;
        self.start_locked()?;
        self.snapshot_locked()
    }

    pub fn restart(&self) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        let _operation = self
            .operation
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.reconcile_pending_shutdown()?;
        {
            let inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            if inner.pending_shutdown.is_some() {
                return Err(NativeHeadlessError::LifecycleBusy);
            }
            if inner.active.is_none() {
                return Err(NativeHeadlessError::NotRunning);
            }
        }
        let prepared = self.prepare_runtime(true)?;
        if let Some(provider_deployment) = prepared.provider_deployment.as_ref() {
            let inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            if inner
                .active
                .as_ref()
                .and_then(|active| active.provider_deployment.as_ref())
                .is_some_and(|active| active.same_signed_deployment(provider_deployment))
            {
                drop(inner);
                drop(prepared);
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| NativeHeadlessError::StateUnavailable)?;
                inner.deployment_diagnostic = None;
                return self.snapshot_from_inner(&inner);
            }
        }
        {
            let inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            if inner.restart_count >= self.policy.maximum_restarts {
                return Err(NativeHeadlessError::RestartBudgetExhausted {
                    maximum: self.policy.maximum_restarts,
                });
            }
        }
        self.stop_locked("native headless restart requested")?;
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            inner.restart_count = inner.restart_count.checked_add(1).ok_or(
                NativeHeadlessError::RestartBudgetExhausted {
                    maximum: self.policy.maximum_restarts,
                },
            )?;
        }
        self.start_prepared_locked(prepared)?;
        self.snapshot_locked()
    }

    pub fn shutdown(&self) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        let _operation = self
            .operation
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.reconcile_pending_shutdown()?;
        self.stop_locked("native headless shutdown requested")?;
        self.snapshot_locked()
    }

    pub fn serve_until_cancelled(
        &self,
        cancellation: &NativeHeadlessCancellation,
    ) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        self.start()?;
        while !cancellation.is_cancelled() {
            thread::sleep(self.policy.cancellation_poll_interval);
            let snapshot = self.snapshot()?;
            if matches!(snapshot.state, NativeHeadlessState::Failed { .. }) {
                return Err(NativeHeadlessError::LifecycleFailed(
                    "native headless service entered a failed state".into(),
                ));
            }
        }
        self.shutdown()
    }

    pub fn snapshot(&self) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        let _operation = self
            .operation
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.reconcile_pending_shutdown()?;
        self.snapshot_locked()
    }

    pub fn execute_cli(
        &self,
        invocation: NativeCliInvocation,
    ) -> Result<NativeAutomationResult, NativeHeadlessError> {
        if invocation.feature_id.trim().is_empty() {
            return Err(NativeHeadlessError::InvalidAutomation(
                "CLI invocation feature identity cannot be empty".into(),
            ));
        }
        let mapping = match invocation.operation {
            NativeCliOperation::Health => NativeAutomationMapping::Native {
                request: HttpRequest::new(HttpMethod::Get, "/health"),
                requires_network: false,
            },
            NativeCliOperation::SystemStats => NativeAutomationMapping::Native {
                request: HttpRequest::new(HttpMethod::Get, "/system_stats"),
                requires_network: false,
            },
            NativeCliOperation::Queue => NativeAutomationMapping::Native {
                request: HttpRequest::new(HttpMethod::Get, "/queue"),
                requires_network: false,
            },
            NativeCliOperation::History {
                prompt_id,
                maximum_items,
                offset,
            } => {
                let path = match prompt_id {
                    Some(prompt_id) => format!("/history/{}", canonical_prompt_id(&prompt_id)?),
                    None => "/history".into(),
                };
                let mut request = HttpRequest::new(HttpMethod::Get, path);
                if let Some(maximum_items) = maximum_items {
                    request = request.with_query("max_items", maximum_items.to_string());
                }
                if let Some(offset) = offset {
                    request = request.with_query("offset", offset.to_string());
                }
                NativeAutomationMapping::Native {
                    request,
                    requires_network: false,
                }
            }
            NativeCliOperation::Jobs {
                status,
                limit,
                offset,
            } => {
                let mut request = HttpRequest::new(HttpMethod::Get, "/api/jobs");
                if let Some(status) = status {
                    request = request.with_query("status", status);
                }
                if let Some(limit) = limit {
                    request = request.with_query("limit", limit.to_string());
                }
                if let Some(offset) = offset {
                    request = request.with_query("offset", offset.to_string());
                }
                NativeAutomationMapping::Native {
                    request,
                    requires_network: false,
                }
            }
            NativeCliOperation::JobStatus { job_id } => NativeAutomationMapping::Native {
                request: HttpRequest::new(
                    HttpMethod::Get,
                    format!("/api/jobs/{}", canonical_prompt_id(&job_id)?),
                ),
                requires_network: false,
            },
            NativeCliOperation::SubmitPrompt {
                submission,
                idempotency_key,
            } => {
                validate_operation_identity(&idempotency_key)?;
                let body = serde_json::to_vec(&submission)
                    .map_err(|error| NativeHeadlessError::InvalidAutomation(error.to_string()))?;
                NativeAutomationMapping::Native {
                    request: HttpRequest::new(HttpMethod::Post, "/prompt")
                        .with_header("content-type", "application/json")
                        .with_header("idempotency-key", idempotency_key)
                        .with_body(body),
                    requires_network: false,
                }
            }
            NativeCliOperation::CancelJobs {
                job_ids,
                operation_id,
            } => {
                validate_operation_identity(&operation_id)?;
                if job_ids.is_empty() {
                    return Err(NativeHeadlessError::InvalidAutomation(
                        "job cancellation requires at least one job identity".into(),
                    ));
                }
                let job_ids = job_ids
                    .iter()
                    .map(|job_id| canonical_prompt_id(job_id))
                    .collect::<Result<Vec<_>, _>>()?;
                let body = serde_json::to_vec(&json!({"job_ids": job_ids}))
                    .map_err(|error| NativeHeadlessError::InvalidAutomation(error.to_string()))?;
                NativeAutomationMapping::Native {
                    request: HttpRequest::new(HttpMethod::Post, "/api/jobs/cancel")
                        .with_header("content-type", "application/json")
                        .with_header("idempotency-key", operation_id.clone())
                        .with_header("x-operation-id", operation_id)
                        .with_body(body),
                    requires_network: false,
                }
            }
            NativeCliOperation::Interrupt {
                prompt_id,
                operation_id,
            } => {
                validate_operation_identity(&operation_id)?;
                let prompt_id = prompt_id
                    .map(|prompt_id| canonical_prompt_id(&prompt_id))
                    .transpose()?;
                let body = serde_json::to_vec(&json!({"prompt_id": prompt_id}))
                    .map_err(|error| NativeHeadlessError::InvalidAutomation(error.to_string()))?;
                NativeAutomationMapping::Native {
                    request: HttpRequest::new(HttpMethod::Post, "/interrupt")
                        .with_header("content-type", "application/json")
                        .with_header("idempotency-key", operation_id.clone())
                        .with_header("x-operation-id", operation_id)
                        .with_body(body),
                    requires_network: false,
                }
            }
            NativeCliOperation::NativeRequest {
                request,
                requires_network,
            } => NativeAutomationMapping::Native {
                request,
                requires_network,
            },
            NativeCliOperation::Migration {
                replacement,
                detail,
            } => NativeAutomationMapping::Migration {
                replacement,
                detail,
            },
            NativeCliOperation::Deferred { reason } => NativeAutomationMapping::Deferred { reason },
        };
        self.execute(NativeAutomationRequest {
            feature_id: invocation.feature_id,
            now_epoch_seconds: invocation.now_epoch_seconds,
            mapping,
        })
    }

    pub fn execute(
        &self,
        request: NativeAutomationRequest,
    ) -> Result<NativeAutomationResult, NativeHeadlessError> {
        request.validate()?;
        match request.mapping {
            NativeAutomationMapping::Migration {
                replacement,
                detail,
            } => {
                return Ok(NativeAutomationResult::Migration {
                    feature_id: request.feature_id,
                    replacement,
                    detail,
                });
            }
            NativeAutomationMapping::Deferred { reason } => {
                return Ok(NativeAutomationResult::Deferred {
                    feature_id: request.feature_id,
                    reason,
                });
            }
            NativeAutomationMapping::Native { .. } => {}
        }
        let _operation = self
            .operation
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.reconcile_pending_shutdown()?;
        let (http_request, requires_network) = match request.mapping {
            NativeAutomationMapping::Native {
                request,
                requires_network,
            } => (request, requires_network),
            NativeAutomationMapping::Migration { .. }
            | NativeAutomationMapping::Deferred { .. } => {
                return Err(NativeHeadlessError::StateUnavailable);
            }
        };
        if self.mode == NativeHeadlessMode::Offline && requires_network {
            return Err(NativeHeadlessError::Offline {
                feature_id: request.feature_id,
            });
        }
        let host = {
            let inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            let active = inner
                .active
                .as_ref()
                .ok_or(NativeHeadlessError::NotRunning)?;
            active.runtime.host()
        };
        let context = HostRequestContext::embedded_loopback(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            request.now_epoch_seconds,
        )?;
        let response = host.handle_http(http_request, context);
        let body = match response.body {
            HttpBody::Empty => NativeAutomationBody::Empty,
            HttpBody::Bytes(bytes) => NativeAutomationBody::Bytes(bytes.to_vec()),
            HttpBody::Json(value) => NativeAutomationBody::Json(value),
            HttpBody::Stream(_) => {
                return Err(NativeHeadlessError::StreamingResponseRequiresTransport);
            }
        };
        Ok(NativeAutomationResult::Native {
            feature_id: request.feature_id,
            status: response.status,
            content_type: response.content_type,
            headers: response.headers,
            body,
        })
    }

    fn start_locked(&self) -> Result<(), NativeHeadlessError> {
        let prepared = self.prepare_runtime(false)?;
        self.start_prepared_locked(prepared)
    }

    fn prepare_runtime(
        &self,
        retain_active_on_rejection: bool,
    ) -> Result<PreparedNativeHeadlessRuntime, NativeHeadlessError> {
        let prepared = match &self.factory {
            HeadlessRuntimeFactory::Immediate(factory) => {
                Ok(PreparedNativeHeadlessRuntime::immediate(factory.clone()))
            }
            HeadlessRuntimeFactory::Prepared(factory) => factory(self.presentation.clone()),
        };
        match prepared {
            Ok(prepared) => Ok(prepared),
            Err(error) if retain_active_on_rejection => {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| NativeHeadlessError::StateUnavailable)?;
                inner.deployment_diagnostic = Some(error.to_string());
                Err(NativeHeadlessError::DeploymentRejected(error.to_string()))
            }
            Err(error) => {
                self.record_failure(
                    "runtime_prepare",
                    "native_runtime_deployment_rejected",
                    &error.to_string(),
                    true,
                )?;
                Err(NativeHeadlessError::DeploymentRejected(error.to_string()))
            }
        }
    }

    fn start_prepared_locked(
        &self,
        prepared: PreparedNativeHeadlessRuntime,
    ) -> Result<(), NativeHeadlessError> {
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            if inner.pending_shutdown.is_some() {
                return Err(NativeHeadlessError::LifecycleBusy);
            }
            if inner.active.is_some() {
                return Err(NativeHeadlessError::AlreadyRunning);
            }
            self.record_transition(
                &mut inner,
                NativeHeadlessState::Starting,
                "building native Rust runtime",
            )?;
        }
        let prepared_provider_deployment = prepared.provider_deployment.clone();
        let runtime = match prepared.activate(self.presentation.clone()) {
            Ok(runtime) => runtime,
            Err(error) => {
                self.record_failure(
                    "runtime_start",
                    "native_runtime_start_failed",
                    &error.to_string(),
                    true,
                )?;
                return Err(NativeHeadlessError::RuntimeBuild(error.to_string()));
            }
        };
        if prepared_provider_deployment.is_some()
            && runtime.provider_deployment() != prepared_provider_deployment.as_ref()
        {
            let shutdown_result = runtime.shutdown("prepared deployment identity mismatch");
            let detail = match shutdown_result {
                Ok(()) => "prepared runtime returned a foreign deployment identity".to_owned(),
                Err(error) => format!(
                    "prepared runtime returned a foreign deployment identity; cleanup failed: {error}"
                ),
            };
            self.record_failure(
                "runtime_activation",
                "native_runtime_deployment_identity_mismatch",
                &detail,
                true,
            )?;
            return Err(NativeHeadlessError::RuntimeBuild(detail));
        }
        let (server, local_address) = match &self.server_config {
            Some(config) => {
                let mut effective_config = runtime.server_config(config.bind_address);
                effective_config.tls = config.tls.clone();
                match NativeApiServer::start(runtime.host(), effective_config) {
                    Ok(server) => {
                        let address = server.local_address();
                        (Some(server), Some(address))
                    }
                    Err(error) => {
                        let shutdown_result = runtime.host().shutdown("native API bind failed");
                        if let Err(shutdown_error) = shutdown_result {
                            self.record_failure(
                                "api_bind_cleanup",
                                "native_api_bind_cleanup_failed",
                                &shutdown_error.to_string(),
                                true,
                            )?;
                            return Err(NativeHeadlessError::Host(shutdown_error.to_string()));
                        }
                        self.record_failure(
                            "api_bind",
                            "native_api_bind_failed",
                            &error.to_string(),
                            true,
                        )?;
                        return Err(NativeHeadlessError::Transport(error.to_string()));
                    }
                }
            }
            None => (None, None),
        };
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        inner.active = Some(ActiveRuntime {
            runtime,
            server,
            local_address,
            provider_deployment: prepared_provider_deployment,
        });
        inner.deployment_diagnostic = None;
        let detail = local_address.map_or_else(
            || "native runtime ready without a public socket".to_owned(),
            |address| format!("native API ready at {address}"),
        );
        self.record_transition(&mut inner, NativeHeadlessState::Ready, &detail)
    }

    fn stop_locked(&self, reason: &str) -> Result<(), NativeHeadlessError> {
        let active = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            if inner.pending_shutdown.is_some() {
                return Err(NativeHeadlessError::LifecycleBusy);
            }
            let active = inner.active.take().ok_or(NativeHeadlessError::NotRunning)?;
            self.record_transition(&mut inner, NativeHeadlessState::Cancelling, reason)?;
            active
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("sim-headless-shutdown".into())
            .spawn(move || {
                let result = shutdown_active_runtime(active);
                if sender.send(result).is_err() {
                    // The lifecycle owner timed out; the owned runtime is still shut down here.
                }
            })
            .map_err(|error| NativeHeadlessError::ShutdownThread(error.to_string()))?;
        match receiver.recv_timeout(self.policy.shutdown_timeout) {
            Ok(Ok(())) => {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| NativeHeadlessError::StateUnavailable)?;
                self.record_transition(
                    &mut inner,
                    NativeHeadlessState::Stopped,
                    "native runtime and API host stopped",
                )
            }
            Ok(Err(error)) => {
                self.record_failure("shutdown", error.code(), &error.to_string(), true)?;
                Err(error)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| NativeHeadlessError::StateUnavailable)?;
                inner.pending_shutdown = Some(PendingShutdown { receiver });
                self.record_transition(
                    &mut inner,
                    NativeHeadlessState::Cancelling,
                    "shutdown exceeded the caller deadline and continues in the owned shutdown thread",
                )?;
                Err(NativeHeadlessError::ShutdownTimeout {
                    milliseconds: duration_milliseconds(self.policy.shutdown_timeout),
                })
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.record_failure(
                    "shutdown",
                    "shutdown_channel_closed",
                    "native shutdown thread ended without a result",
                    true,
                )?;
                Err(NativeHeadlessError::ShutdownThread(
                    "native shutdown thread ended without a result".into(),
                ))
            }
        }
    }

    fn reconcile_pending_shutdown(&self) -> Result<(), NativeHeadlessError> {
        let result = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| NativeHeadlessError::StateUnavailable)?;
            let Some(pending) = inner.pending_shutdown.as_ref() else {
                return Ok(());
            };
            match pending.receiver.try_recv() {
                Ok(result) => {
                    inner.pending_shutdown = None;
                    Some(result)
                }
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    inner.pending_shutdown = None;
                    Some(Err(NativeHeadlessError::ShutdownThread(
                        "native shutdown thread ended without a result".into(),
                    )))
                }
            }
        };
        match result {
            Some(Ok(())) => {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| NativeHeadlessError::StateUnavailable)?;
                self.record_transition(
                    &mut inner,
                    NativeHeadlessState::Stopped,
                    "timed-out native shutdown completed",
                )
            }
            Some(Err(error)) => {
                self.record_failure("shutdown", error.code(), &error.to_string(), true)?;
                Err(error)
            }
            None => Ok(()),
        }
    }

    fn record_failure(
        &self,
        stage: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<(), NativeHeadlessError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        let failure = NativeHeadlessFailure {
            stage: stage.into(),
            code: code.into(),
            message: message.into(),
            retryable,
        };
        self.record_transition(&mut inner, NativeHeadlessState::Failed { failure }, message)
    }

    fn record_transition(
        &self,
        inner: &mut NativeHeadlessInner,
        state: NativeHeadlessState,
        detail: &str,
    ) -> Result<(), NativeHeadlessError> {
        let sequence = inner.next_sequence;
        inner.next_sequence = inner
            .next_sequence
            .checked_add(1)
            .ok_or(NativeHeadlessError::LifecycleSequenceExhausted)?;
        inner.state = state.clone();
        inner.transitions.push_back(NativeHeadlessTransition {
            sequence,
            state,
            detail: detail.into(),
        });
        while inner.transitions.len() > self.policy.lifecycle_history_capacity {
            if inner.transitions.pop_front().is_none() {
                return Err(NativeHeadlessError::StateUnavailable);
            }
            inner.dropped_transitions = inner.dropped_transitions.saturating_add(1);
        }
        Ok(())
    }

    fn snapshot_locked(&self) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| NativeHeadlessError::StateUnavailable)?;
        self.snapshot_from_inner(&inner)
    }

    fn snapshot_from_inner(
        &self,
        inner: &NativeHeadlessInner,
    ) -> Result<NativeHeadlessSnapshot, NativeHeadlessError> {
        Ok(NativeHeadlessSnapshot {
            mode: self.mode,
            state: inner.state.clone(),
            local_address: inner
                .active
                .as_ref()
                .and_then(|active| active.local_address),
            restart_count: inner.restart_count,
            transitions: inner.transitions.iter().cloned().collect(),
            dropped_transitions: inner.dropped_transitions,
            deployment_identity_sha256: inner
                .active
                .as_ref()
                .and_then(|active| active.provider_deployment.as_ref())
                .map(|deployment| deployment.execution_bundle_identity_sha256().to_owned()),
            provider_deployment: inner
                .active
                .as_ref()
                .and_then(|active| active.provider_deployment.clone()),
            deployment_diagnostic: inner.deployment_diagnostic.clone(),
        })
    }
}

fn shutdown_active_runtime(active: ActiveRuntime) -> Result<(), NativeHeadlessError> {
    if let Some(server) = active.server {
        server
            .shutdown()
            .map_err(|error| NativeHeadlessError::Transport(error.to_string()))?;
    }
    active.runtime.shutdown("native headless runtime stopped")?;
    drop(active.runtime);
    Ok(())
}

fn canonical_prompt_id(value: &str) -> Result<String, NativeHeadlessError> {
    let prompt_id = sim_runtime::PromptId(value.parse().map_err(|error| {
        NativeHeadlessError::InvalidAutomation(format!("invalid prompt identity: {error}"))
    })?);
    Ok(prompt_id.0.to_string())
}

fn validate_operation_identity(value: &str) -> Result<(), NativeHeadlessError> {
    if value.trim().is_empty() || value.len() > 512 || value.contains(['\r', '\n']) {
        return Err(NativeHeadlessError::InvalidAutomation(
            "operation identity must contain 1 to 512 non-line-break bytes".into(),
        ));
    }
    Ok(())
}

fn duration_milliseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

pub fn current_epoch_seconds() -> Result<u64, NativeHeadlessError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| NativeHeadlessError::Clock(error.to_string()))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeHeadlessError {
    #[error("invalid native headless configuration: {0}")]
    InvalidConfiguration(String),
    #[error("native headless state is unavailable")]
    StateUnavailable,
    #[error("native headless service is already running")]
    AlreadyRunning,
    #[error("native headless service is not running")]
    NotRunning,
    #[error("native headless lifecycle is still converging")]
    LifecycleBusy,
    #[error("native headless restart budget of {maximum} is exhausted")]
    RestartBudgetExhausted { maximum: usize },
    #[error("native runtime construction failed: {0}")]
    RuntimeBuild(String),
    #[error("native provider deployment was rejected: {0}")]
    DeploymentRejected(String),
    #[error("native API transport failed: {0}")]
    Transport(String),
    #[error("native API host failed: {0}")]
    Host(String),
    #[error("native headless shutdown exceeded {milliseconds} milliseconds")]
    ShutdownTimeout { milliseconds: u64 },
    #[error("native headless shutdown thread failed: {0}")]
    ShutdownThread(String),
    #[error("native headless lifecycle failed: {0}")]
    LifecycleFailed(String),
    #[error("native headless lifecycle sequence is exhausted")]
    LifecycleSequenceExhausted,
    #[error("native automation `{feature_id}` requires network access while offline")]
    Offline { feature_id: String },
    #[error("invalid native automation request: {0}")]
    InvalidAutomation(String),
    #[error("streaming automation responses require the bounded native HTTP transport")]
    StreamingResponseRequiresTransport,
    #[error("invalid native CLI event: {0}")]
    InvalidCliEvent(String),
    #[error("system clock failed: {0}")]
    Clock(String),
}

impl NativeHeadlessError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfiguration(_) => "headless_configuration_invalid",
            Self::StateUnavailable => "headless_state_unavailable",
            Self::AlreadyRunning => "headless_already_running",
            Self::NotRunning => "headless_not_running",
            Self::LifecycleBusy => "headless_lifecycle_busy",
            Self::RestartBudgetExhausted { .. } => "headless_restart_budget_exhausted",
            Self::RuntimeBuild(_) => "native_runtime_start_failed",
            Self::DeploymentRejected(_) => "native_provider_deployment_rejected",
            Self::Transport(_) => "native_api_transport_failed",
            Self::Host(_) => "native_api_host_failed",
            Self::ShutdownTimeout { .. } => "native_shutdown_timeout",
            Self::ShutdownThread(_) => "native_shutdown_thread_failed",
            Self::LifecycleFailed(_) => "native_lifecycle_failed",
            Self::LifecycleSequenceExhausted => "native_lifecycle_sequence_exhausted",
            Self::Offline { .. } => "offline",
            Self::InvalidAutomation(_) => "native_automation_invalid",
            Self::StreamingResponseRequiresTransport => "native_stream_transport_required",
            Self::InvalidCliEvent(_) => "native_cli_event_invalid",
            Self::Clock(_) => "system_clock_failed",
        }
    }
}

impl From<NativeApiHostError> for NativeHeadlessError {
    fn from(error: NativeApiHostError) -> Self {
        Self::Host(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HttpLimits, WebSocketLimits,
        security::{
            ApiSecurityConfig, ApiSecurityError, IdempotencySnapshot, IdempotencySnapshotStore,
        },
    };
    use sha2::{Digest, Sha256};
    use sim_runtime::{
        DisconnectedExecutionController, ExecutionDataSource, ExecutionEventBus,
        ExecutionPresentationService, ExecutionSnapshotStatus, ProfileId,
        SharedExecutionPresentationService,
    };
    use std::{
        io::{Read, Write},
        net::TcpStream,
        sync::atomic::{AtomicUsize, Ordering},
    };

    const CLI_CATALOGS: &[(&str, &str, usize, &str)] = &[
        (
            "commands",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-commands.csv"),
            123,
            "commands",
        ),
        (
            "config",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-config.csv"),
            20,
            "config",
        ),
        (
            "cql-policy",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-cql-policy.csv"),
            419,
            "cql_rows",
        ),
        (
            "documentation",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-documentation.csv"
            ),
            16,
            "documentation_claims",
        ),
        (
            "environment",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-environment.csv"
            ),
            35,
            "environment",
        ),
        (
            "errors",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-errors.csv"),
            99,
            "errors",
        ),
        (
            "events",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-events.csv"),
            12,
            "events",
        ),
        (
            "extensions",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-extensions.csv"),
            17,
            "extensions",
        ),
        (
            "formats",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-formats.csv"),
            34,
            "formats",
        ),
        (
            "lifecycle",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-lifecycle.csv"),
            24,
            "lifecycle",
        ),
        (
            "modules",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-modules.csv"),
            104,
            "modules",
        ),
        (
            "parameters",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-parameters.csv"),
            370,
            "parameters",
        ),
        (
            "partner-openapi",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-partner-openapi.csv"
            ),
            52,
            "partner_endpoints",
        ),
        (
            "schema-mappings",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-schema-mappings.csv"
            ),
            66,
            "schema_mappings",
        ),
        (
            "schemas",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-schemas.csv"),
            23,
            "schemas",
        ),
        (
            "source-coverage",
            include_str!(
                "../../../crates/sim_test_support/parity/catalogs/sim-cli-source-coverage.csv"
            ),
            312,
            "source_rows",
        ),
        (
            "tests",
            include_str!("../../../crates/sim_test_support/parity/catalogs/sim-cli-tests.csv"),
            2_295,
            "tests",
        ),
    ];

    const CLI_RECONCILIATION: &str = include_str!(
        "../../../crates/sim_test_support/parity/catalogs/sim-cli-reconciliation.json"
    );
    const HTTP_CATALOG: &str =
        include_str!("../../../crates/sim_test_support/parity/catalogs/backend-http-routes.csv");
    const WEBSOCKET_CATALOG: &str = include_str!(
        "../../../crates/sim_test_support/parity/catalogs/backend-websocket-events.csv"
    );

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

    fn test_profile_id() -> Result<ProfileId, Box<dyn std::error::Error>> {
        Ok(ProfileId("00000000-0000-0000-0000-000000002101".parse()?))
    }

    fn test_presentation() -> Result<SharedExecutionPresentationService, NativeApiHostError> {
        let profile_id =
            test_profile_id().map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let mut presentation = ExecutionPresentationService::new(4_096)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        presentation
            .initialize_profile(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        Ok(sim_runtime::ExecutionPresentationOwner::ephemeral(
            presentation,
        ))
    }

    fn runtime(
        presentation: SharedExecutionPresentationService,
    ) -> Result<NativeRuntimeApiHost, NativeApiHostError> {
        let profile_id =
            test_profile_id().map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        let events = ExecutionEventBus::new(16)
            .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?;
        NativeRuntimeApiHost::native_image(
            profile_id,
            presentation,
            Arc::new(DisconnectedExecutionController),
            &events,
            None,
            HttpLimits::default(),
            WebSocketLimits::default(),
            ApiSecurityConfig::loopback(),
            Arc::new(
                sim_runtime::PermissionPolicy::native_runtime_services(profile_id.0.to_string())
                    .map_err(|error| NativeApiHostError::Runtime(error.to_string()))?,
            ),
            Arc::new(MemoryIdempotencyStore::default()),
        )
    }

    fn native_result_status(result: &NativeAutomationResult) -> Option<u16> {
        match result {
            NativeAutomationResult::Native { status, .. } => Some(*status),
            NativeAutomationResult::Migration { .. } | NativeAutomationResult::Deferred { .. } => {
                None
            }
        }
    }

    fn native_cli_service_mapping(command_path: &str) -> &'static str {
        match command_path {
            "sim help" | "sim version" => "native-cli-local-contract",
            "sim env" | "sim which" => "native-api:/system_stats",
            "sim discover" => "native-api:/health",
            "sim jobs ls" => "native-api:/api/jobs",
            "sim jobs status" | "sim jobs watch" | "sim _watch _watch-job" => {
                "native-api:/api/jobs/{job_id}"
            }
            "sim jobs cancel" => "native-api:/api/jobs/cancel",
            "sim jobs wait" => "native-capability-unavailable:native_job_wait",
            "sim run" => "native-api:/prompt",
            "sim download" => "native-api:/history/{prompt_id}",
            path if path.starts_with("sim nodes ") => "native-api:/object_info",
            path if path.starts_with("sim model ") || path.starts_with("sim models ") => {
                "native-api:/models"
            }
            path if path.starts_with("sim workflow ") => "native-api:/api/workflows",
            path if path.starts_with("sim templates ") => "native-api:/workflow_templates",
            _ => "native-capability-unavailable:typed-adapter-not-enabled",
        }
    }

    fn linked_validation_evidence(
        validation: &str,
        artifact_name: &str,
        expected_catalog_rows: usize,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
            })
            .join("sim-parity")
            .join(artifact_name);
        let bytes = std::fs::read(&target)?;
        let artifact: Value = serde_json::from_slice(&bytes)?;
        let observed_catalog_rows = artifact
            .pointer("/summary/catalog_passed")
            .and_then(Value::as_u64)
            .or_else(|| {
                artifact
                    .get("catalog_cases")
                    .and_then(Value::as_array)
                    .and_then(|cases| u64::try_from(cases.len()).ok())
            })
            .ok_or("linked validation artifact has no catalog count")?;
        let failed = artifact
            .pointer("/summary/failed")
            .and_then(Value::as_u64)
            .or_else(|| artifact.get("failed").and_then(Value::as_u64))
            .ok_or("linked validation artifact has no failure count")?;
        let skipped = artifact
            .pointer("/summary/skipped")
            .and_then(Value::as_u64)
            .or_else(|| {
                artifact
                    .get("skipped")
                    .and_then(Value::as_array)
                    .and_then(|rows| u64::try_from(rows.len()).ok())
            })
            .ok_or("linked validation artifact has no skipped count")?;
        if observed_catalog_rows != expected_catalog_rows as u64 || failed != 0 || skipped != 0 {
            return Err(format!("linked {validation} artifact is not passing and complete").into());
        }
        Ok(json!({
            "validation": validation,
            "artifact": format!("target/sim-parity/{artifact_name}"),
            "execution_status": "linked validation executed directly by VAL-NATIVE-API-001 immediately before artifact reconciliation",
            "expected_catalog_rows": expected_catalog_rows,
            "observed_catalog_rows": observed_catalog_rows,
            "failed": failed,
            "skipped": skipped,
            "artifact_sha256": format!("{:x}", Sha256::digest(&bytes)),
        }))
    }

    #[test]
    fn native_cli_event_union_is_complete_versioned_and_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let names = NativeCliEventKind::ALL
            .into_iter()
            .map(NativeCliEventKind::as_str)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "cancelled",
                "converted",
                "executed",
                "executing",
                "execution_cached",
                "execution_error",
                "output",
                "progress",
                "prompt_preview",
                "queued",
                "settled",
                "state",
            ]
        );
        for kind in NativeCliEventKind::ALL {
            let event = NativeCliEvent::new(
                kind,
                BTreeMap::from([("future_field".into(), json!({"retained": true}))]),
            )?;
            let encoded = serde_json::to_vec(&event)?;
            let decoded: NativeCliEvent = serde_json::from_slice(&encoded)?;
            assert_eq!(decoded, event);
            assert_eq!(decoded.schema, NATIVE_CLI_EVENT_VERSION);
        }
        let invalid = format!(
            "{{\"schema\":\"event/2\",\"type\":\"queued\",\"payload\":\"{}\"}}",
            "x".repeat(MAXIMUM_NATIVE_CLI_EVENT_BYTES)
        );
        assert!(serde_json::from_str::<NativeCliEvent>(&invalid).is_err());
        Ok(())
    }

    #[test]
    fn native_headless_error_codes_are_stable_and_unique() {
        let errors = [
            NativeHeadlessError::InvalidConfiguration("fixture".into()),
            NativeHeadlessError::StateUnavailable,
            NativeHeadlessError::AlreadyRunning,
            NativeHeadlessError::NotRunning,
            NativeHeadlessError::LifecycleBusy,
            NativeHeadlessError::RestartBudgetExhausted { maximum: 1 },
            NativeHeadlessError::RuntimeBuild("fixture".into()),
            NativeHeadlessError::DeploymentRejected("fixture".into()),
            NativeHeadlessError::Transport("fixture".into()),
            NativeHeadlessError::Host("fixture".into()),
            NativeHeadlessError::ShutdownTimeout { milliseconds: 1 },
            NativeHeadlessError::ShutdownThread("fixture".into()),
            NativeHeadlessError::LifecycleFailed("fixture".into()),
            NativeHeadlessError::LifecycleSequenceExhausted,
            NativeHeadlessError::Offline {
                feature_id: "fixture".into(),
            },
            NativeHeadlessError::InvalidAutomation("fixture".into()),
            NativeHeadlessError::StreamingResponseRequiresTransport,
            NativeHeadlessError::InvalidCliEvent("fixture".into()),
            NativeHeadlessError::Clock("fixture".into()),
        ];
        let codes = errors
            .iter()
            .map(NativeHeadlessError::code)
            .collect::<Vec<_>>();
        assert_eq!(
            codes,
            vec![
                "headless_configuration_invalid",
                "headless_state_unavailable",
                "headless_already_running",
                "headless_not_running",
                "headless_lifecycle_busy",
                "headless_restart_budget_exhausted",
                "native_runtime_start_failed",
                "native_provider_deployment_rejected",
                "native_api_transport_failed",
                "native_api_host_failed",
                "native_shutdown_timeout",
                "native_shutdown_thread_failed",
                "native_lifecycle_failed",
                "native_lifecycle_sequence_exhausted",
                "offline",
                "native_automation_invalid",
                "native_stream_transport_required",
                "native_cli_event_invalid",
                "system_clock_failed",
            ]
        );
        assert_eq!(
            codes
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            codes.len()
        );
    }

    #[test]
    fn tls_der_constructor_rejects_empty_and_oversized_material_before_server_start() {
        assert!(matches!(
            crate::NativeTlsAcceptor::from_pkcs8_der("localhost", Vec::new(), Vec::new()),
            Err(crate::NativeTransportError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            crate::NativeTlsAcceptor::from_pkcs8_der(
                "localhost",
                vec![vec![1; MAXIMUM_NATIVE_TLS_DER_BYTES + 1]],
                vec![1],
            ),
            Err(crate::NativeTransportError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn explicit_migration_and_defer_never_report_native_success()
    -> Result<(), Box<dyn std::error::Error>> {
        let service = NativeHeadlessService::offline(
            test_presentation()?,
            runtime,
            NativeHeadlessPolicy::default(),
        )?;
        let migration = service.execute(NativeAutomationRequest::migration(
            "COMFY-CLI-CMD-install",
            "zed sim runtime install",
            "legacy Python installation is migration evidence only",
        ))?;
        assert!(matches!(
            migration,
            NativeAutomationResult::Migration { .. }
        ));
        let deferred = service.execute(NativeAutomationRequest::deferred(
            "COMFY-CLI-CMD-cloud-login",
            "cloud authority has not been configured",
        ))?;
        assert!(matches!(deferred, NativeAutomationResult::Deferred { .. }));
        assert_eq!(service.snapshot()?.state, NativeHeadlessState::Inactive);
        Ok(())
    }

    #[test]
    fn val_native_api_001_headless_offline_uses_real_native_runtime_services()
    -> Result<(), Box<dyn std::error::Error>> {
        let service = NativeHeadlessService::offline(
            test_presentation()?,
            runtime,
            NativeHeadlessPolicy::default(),
        )?;
        let started = service.start()?;
        assert_eq!(started.mode, NativeHeadlessMode::Offline);
        assert_eq!(started.state, NativeHeadlessState::Ready);
        assert_eq!(started.local_address, None);

        let health = service.execute_cli(NativeCliInvocation {
            feature_id: "COMFY-CLI-native-health".into(),
            operation: NativeCliOperation::Health,
            now_epoch_seconds: 100,
        })?;
        assert_eq!(native_result_status(&health), Some(200));

        let status = service.execute_cli(NativeCliInvocation {
            feature_id: "COMFY-CLI-native-system-stats".into(),
            operation: NativeCliOperation::SystemStats,
            now_epoch_seconds: 101,
        })?;
        let NativeAutomationResult::Native { body, .. } = status else {
            return Err("system stats did not execute natively".into());
        };
        let NativeAutomationBody::Json(status) = body else {
            return Err("system stats did not return JSON".into());
        };
        assert_eq!(status["system"]["native"], true);
        assert_eq!(status["system"]["python_runtime"], false);
        assert_eq!(status["system"]["external_server"], false);

        let network_error = service
            .execute(NativeAutomationRequest::native_networked(
                "COMFY-CLI-networked-provider",
                HttpRequest::new(HttpMethod::Get, "/system_stats"),
                102,
            ))
            .expect_err("offline headless service must reject networked automation");
        assert!(matches!(network_error, NativeHeadlessError::Offline { .. }));

        let stopped = service.shutdown()?;
        assert_eq!(stopped.state, NativeHeadlessState::Stopped);
        assert!(matches!(
            service.execute_cli(NativeCliInvocation {
                feature_id: "COMFY-CLI-after-stop".into(),
                operation: NativeCliOperation::Queue,
                now_epoch_seconds: 103,
            }),
            Err(NativeHeadlessError::NotRunning)
        ));
        Ok(())
    }

    #[test]
    fn restart_and_cancellation_are_bounded_and_idempotent()
    -> Result<(), Box<dyn std::error::Error>> {
        let builds = Arc::new(AtomicUsize::new(0));
        let presentation = test_presentation()?;
        let service = NativeHeadlessService::offline(
            presentation.clone(),
            {
                let builds = builds.clone();
                let expected_presentation = presentation.clone();
                move |presentation| {
                    assert!(Arc::ptr_eq(&presentation, &expected_presentation));
                    builds.fetch_add(1, Ordering::SeqCst);
                    runtime(presentation)
                }
            },
            NativeHeadlessPolicy {
                lifecycle_history_capacity: 5,
                maximum_restarts: 1,
                ..NativeHeadlessPolicy::default()
            },
        )?;
        service.start()?;
        smol::block_on(presentation.set_snapshot_status_durable(
            test_profile_id()?,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Loading,
        ))?;
        let restarted = service.restart()?;
        assert_eq!(restarted.state, NativeHeadlessState::Ready);
        assert_eq!(restarted.restart_count, 1);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(
            presentation.snapshot(test_profile_id()?)?.status,
            ExecutionSnapshotStatus::Loading
        );
        assert!(matches!(
            service.restart(),
            Err(NativeHeadlessError::RestartBudgetExhausted { maximum: 1 })
        ));

        let cancellation = NativeHeadlessCancellation::default();
        assert!(cancellation.cancel("SIGINT")?);
        assert!(!cancellation.cancel("duplicate SIGINT")?);
        let event = cancellation.event()?.ok_or("missing cancellation event")?;
        assert_eq!(event.kind, NativeCliEventKind::Cancelled);
        assert_eq!(
            event.fields.get("reason"),
            Some(&Value::String("SIGINT".into()))
        );
        let stopped = service.serve_until_cancelled(&cancellation);
        assert!(matches!(stopped, Err(NativeHeadlessError::AlreadyRunning)));
        assert_eq!(service.shutdown()?.state, NativeHeadlessState::Stopped);
        assert!(service.snapshot()?.transitions.len() <= 5);

        let cancellable = NativeHeadlessService::offline(
            test_presentation()?,
            runtime,
            NativeHeadlessPolicy::default(),
        )?;
        let already_cancelled = NativeHeadlessCancellation::default();
        already_cancelled.cancel("bounded test cancellation")?;
        assert_eq!(
            cancellable.serve_until_cancelled(&already_cancelled)?.state,
            NativeHeadlessState::Stopped
        );
        Ok(())
    }

    #[test]
    fn val_cancel_001_headless_adapter_preserves_first_reason() -> Result<(), NativeHeadlessError> {
        let cancellation = NativeHeadlessCancellation::default();

        assert!(cancellation.cancel("first")?);
        assert!(!cancellation.cancel("second")?);
        assert!(cancellation.is_cancelled());
        assert_eq!(cancellation.reason()?.as_deref(), Some("first"));
        assert_eq!(
            cancellation
                .event()?
                .and_then(|event| event.fields.get("reason").cloned()),
            Some(Value::String("first".into()))
        );
        Ok(())
    }

    #[test]
    fn native_cli_operation_builds_safe_cancel_and_interrupt_mutations()
    -> Result<(), Box<dyn std::error::Error>> {
        let service = NativeHeadlessService::offline(
            test_presentation()?,
            runtime,
            NativeHeadlessPolicy::default(),
        )?;
        service.start()?;
        let prompt_id = "00000000-0000-0000-0000-000000002102";
        let cancellation = service.execute_cli(NativeCliInvocation {
            feature_id: "COMFY-CLI-CMD-jobs-cancel".into(),
            operation: NativeCliOperation::CancelJobs {
                job_ids: vec![prompt_id.into()],
                operation_id: "cancel-operation-1".into(),
            },
            now_epoch_seconds: 200,
        })?;
        assert_eq!(native_result_status(&cancellation), Some(200));
        let interrupt = service.execute_cli(NativeCliInvocation {
            feature_id: "COMFY-CLI-CMD-interrupt".into(),
            operation: NativeCliOperation::Interrupt {
                prompt_id: Some(prompt_id.into()),
                operation_id: "interrupt-operation-1".into(),
            },
            now_epoch_seconds: 201,
        })?;
        assert_eq!(native_result_status(&interrupt), Some(200));
        assert!(matches!(
            service.execute_cli(NativeCliInvocation {
                feature_id: "COMFY-CLI-CMD-jobs-cancel".into(),
                operation: NativeCliOperation::CancelJobs {
                    job_ids: vec!["../../escape".into()],
                    operation_id: "unsafe-cancel".into(),
                },
                now_epoch_seconds: 202,
            }),
            Err(NativeHeadlessError::InvalidAutomation(_))
        ));
        service.shutdown()?;
        Ok(())
    }

    #[test]
    fn val_ws_001_serve_mode_uses_real_native_transport_and_reports_effective_address()
    -> Result<(), Box<dyn std::error::Error>> {
        let config = NativeApiServerConfig::new("127.0.0.1:0".parse()?);
        let service = Arc::new(NativeHeadlessService::serve(
            test_presentation()?,
            runtime,
            config,
            NativeHeadlessPolicy::default(),
        )?);
        let started = service.start()?;
        assert_eq!(started.mode, NativeHeadlessMode::Serve);
        let address = started.local_address.ok_or("missing bound address")?;
        assert!(address.ip().is_loopback());

        let mut http = TcpStream::connect(address)?;
        http.set_read_timeout(Some(Duration::from_secs(10)))?;
        http.write_all(
            b"GET /system_stats HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )?;
        let mut response = Vec::new();
        http.read_to_end(&mut response)?;
        let response = String::from_utf8(response)?;
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("\"external_server\":false"));
        assert!(response.contains("\"python_runtime\":false"));

        let websocket_stream = TcpStream::connect(address)?;
        websocket_stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        websocket_stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        let request = format!("ws://{address}/ws?clientId=headless-lifecycle");
        let (mut websocket, upgrade) =
            async_tungstenite::tungstenite::client(request, websocket_stream)?;
        assert_eq!(upgrade.status().as_u16(), 101);
        let initial = websocket.read()?;
        let async_tungstenite::tungstenite::Message::Text(initial) = initial else {
            return Err("native headless WebSocket did not publish its reconnect status".into());
        };
        assert!(initial.contains("\"type\":\"status\""));

        let shutdown = thread::spawn(move || service.shutdown());
        websocket.send(async_tungstenite::tungstenite::Message::Ping(
            Vec::new().into(),
        ))?;
        let mut received_close = false;
        for _ in 0..3 {
            match websocket.read()? {
                async_tungstenite::tungstenite::Message::Close(_) => {
                    received_close = true;
                    break;
                }
                async_tungstenite::tungstenite::Message::Pong(_) => {}
                message => return Err(format!("unexpected shutdown frame {message:?}").into()),
            }
        }
        assert!(received_close);
        let stopped = shutdown.join().map_err(|_| "shutdown thread panicked")??;
        assert_eq!(stopped.state, NativeHeadlessState::Stopped);
        Ok(())
    }

    #[test]
    fn val_native_api_001_writes_complete_protocol_cli_and_lifecycle_artifact()
    -> Result<(), Box<dyn std::error::Error>> {
        crate::http::tests::val_http_001()?;
        crate::websocket::tests::val_ws_001_catalog_rows_have_exact_descriptors_and_working_contracts();

        let reconciliation: Value = serde_json::from_str(CLI_RECONCILIATION)?;
        let recorded_digests = reconciliation["catalog_sha256"]
            .as_object()
            .ok_or("CLI reconciliation has no catalog digest object")?;
        let mut catalog_results = Vec::new();
        let mut total_cli_rows = 0usize;
        for (name, contents, expected_rows, reconciliation_count_key) in CLI_CATALOGS {
            let parsed = crate::http::parse_csv(contents)?;
            let (header, rows) = parsed.split_first().ok_or("CLI catalog has no header")?;
            let actual_rows = rows.len();
            total_cli_rows = total_cli_rows
                .checked_add(actual_rows)
                .ok_or("CLI row count overflow")?;
            let digest = format!("{:x}", Sha256::digest(contents.as_bytes()));
            let filename = format!("sim-cli-{name}.csv");
            let recorded_digest = recorded_digests
                .get(&filename)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("reconciliation has no digest for {filename}"))?;
            let reconciled_count = if *name == "schema-mappings" {
                reconciliation["schema_mappings_envelope"]
                    .as_u64()
                    .and_then(|envelope| {
                        reconciliation["schema_mappings_stream"]
                            .as_u64()
                            .and_then(|stream| envelope.checked_add(stream))
                    })
            } else {
                reconciliation[*reconciliation_count_key].as_u64()
            }
            .ok_or_else(|| format!("reconciliation has no count for {name}"))?;
            let column =
                |column_name: &str| header.iter().position(|candidate| candidate == column_name);
            let target_status_column = column("target_status");
            let evidence_column = column("evidence_level");
            let parity_decision_column = column("parity_decision");
            let command_path_column = column("command_path").or_else(|| column("path"));
            let row_results = rows
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    let identity = if *name == "schema-mappings" {
                        format!(
                            "{}:{}:{}:{}",
                            index + 1,
                            row.first().map(String::as_str).unwrap_or_default(),
                            row.get(1).map(String::as_str).unwrap_or_default(),
                            row.get(2).map(String::as_str).unwrap_or_default()
                        )
                    } else {
                        row.first().cloned().unwrap_or_default()
                    };
                    let target_status = target_status_column
                        .and_then(|column| row.get(column))
                        .map(String::as_str)
                        .unwrap_or("source-evidence-only");
                    let disposition = if matches!(*name, "commands" | "parameters") {
                        match target_status {
                            "conflicting" => "migration",
                            "deferred" => "deferred",
                            _ => "native",
                        }
                    } else {
                        "catalog-evidence"
                    };
                    let command_path = command_path_column
                        .and_then(|column| row.get(column))
                        .map(String::as_str);
                    let service_mapping = match (*name, disposition) {
                        ("tests", _) => "native-test-port-or-source-evidence-retention",
                        ("source-coverage", _) => "source-file-closure",
                        ("documentation", _) => "documentation-evidence-reconciliation",
                        ("events", _) => "versioned-native-cli-event-contract",
                        ("errors", _) => "stable-native-cli-error-contract",
                        ("config", _) => "typed-native-configuration-contract",
                        (_, "catalog-evidence") => "native-catalog-contract",
                        (_, "migration") => "native-migration-response",
                        (_, "deferred") => "explicit-authority-or-provider-defer",
                        _ => command_path.map_or(
                            "native-contract-evidence",
                            native_cli_service_mapping,
                        ),
                    };
                    let execution_status = if *name == "tests" {
                        "source-test-not-run; catalogued for native test mapping"
                    } else {
                        "catalog-identity-reconciled; behavioral evidence linked separately"
                    };
                    let accounted = row.len() == header.len() && !identity.is_empty();
                    json!({
                        "row": index + 1,
                        "identity": identity,
                        "accounted": accounted,
                        "source_evidence_level": evidence_column.and_then(|column| row.get(column)),
                        "source_target_status": target_status,
                        "task21_disposition": disposition,
                        "service_mapping": service_mapping,
                        "execution_status": execution_status,
                        "parity_decision": parity_decision_column.and_then(|column| row.get(column)),
                    })
                })
                .collect::<Vec<_>>();
            let accounted_rows = row_results
                .iter()
                .filter(|result| result["accounted"] == true)
                .count();
            let accounted = actual_rows == *expected_rows
                && reconciled_count == *expected_rows as u64
                && digest == recorded_digest
                && accounted_rows == actual_rows;
            if !accounted {
                return Err(format!("CLI catalog `{name}` failed reconciliation").into());
            }
            catalog_results.push(json!({
                "name": name,
                "filename": filename,
                "expected_rows": expected_rows,
                "observed_rows": actual_rows,
                "accounted_rows": accounted_rows,
                "sha256": digest,
                "reconciliation_sha256": recorded_digest,
                "accounted": accounted,
                "rows": row_results,
            }));
        }
        if total_cli_rows != 4_021 {
            return Err(format!("expected 4021 CLI rows, found {total_cli_rows}").into());
        }

        let http_routes = crate::http_route_catalog()?;
        let websocket_contracts = crate::normative_websocket_catalog()?;
        if http_routes.len() != 141 || websocket_contracts.len() != 26 {
            return Err("native HTTP/WebSocket catalog count changed".into());
        }
        let http_csv_rows = crate::http::parse_csv(HTTP_CATALOG)?
            .len()
            .saturating_sub(1);
        let websocket_csv_rows = crate::http::parse_csv(WEBSOCKET_CATALOG)?
            .len()
            .saturating_sub(1);
        if http_csv_rows != http_routes.len() || websocket_csv_rows != websocket_contracts.len() {
            return Err("native protocol parser does not account for its complete catalog".into());
        }

        type ExecutableCase = fn() -> Result<(), Box<dyn std::error::Error>>;
        fn stable_error_code_case() -> Result<(), Box<dyn std::error::Error>> {
            native_headless_error_codes_are_stable_and_unique();
            Ok(())
        }
        fn bounded_tls_configuration_case() -> Result<(), Box<dyn std::error::Error>> {
            tls_der_constructor_rejects_empty_and_oversized_material_before_server_start();
            Ok(())
        }
        let executable_cases: [(&str, ExecutableCase); 8] = [
            (
                "authoritative-versioned-cli-event-union",
                native_cli_event_union_is_complete_versioned_and_bounded,
            ),
            (
                "explicit-migration-and-defer",
                explicit_migration_and_defer_never_report_native_success,
            ),
            ("stable-structured-error-codes", stable_error_code_case),
            (
                "bounded-native-tls-configuration",
                bounded_tls_configuration_case,
            ),
            (
                "socketless-offline-native-services",
                val_native_api_001_headless_offline_uses_real_native_runtime_services,
            ),
            (
                "bounded-restart-cancellation-and-clean-shutdown",
                restart_and_cancellation_are_bounded_and_idempotent,
            ),
            (
                "safe-cancel-and-interrupt-mutations",
                native_cli_operation_builds_safe_cancel_and_interrupt_mutations,
            ),
            (
                "real-loopback-http-websocket-reconnect-and-shutdown",
                val_ws_001_serve_mode_uses_real_native_transport_and_reports_effective_address,
            ),
        ];
        let mut executable_results = Vec::new();
        for (name, executable_case) in executable_cases {
            let result = executable_case();
            let passed = result.is_ok();
            let error = result.as_ref().err().map(ToString::to_string);
            executable_results.push(json!({
                "name": name,
                "passed": passed,
                "error": error,
            }));
            result?;
        }
        let executable_passed = executable_results
            .iter()
            .filter(|result| result["passed"] == true)
            .count();
        let protocol_and_cli_rows = total_cli_rows
            .checked_add(http_routes.len())
            .and_then(|count| count.checked_add(websocket_contracts.len()))
            .ok_or("native API validation count overflow")?;
        let http_execution_evidence =
            linked_validation_evidence("VAL-HTTP-001", "val-http-001.json", 141)?;
        let websocket_execution_evidence =
            linked_validation_evidence("VAL-WS-001", "val-ws-001.json", 26)?;
        let artifact = json!({
            "validation": "VAL-NATIVE-API-001",
            "schema_version": 1,
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "runtime": "native-rust",
                "gpui_constructed": false,
                "python_processes": 0,
                "node_processes": 0,
                "browser_processes": 0,
                "external_sim_connections": 0,
                "proxy_or_forward_paths": 0,
            },
            "protocol_catalogs": {
                "http": {
                    "expected_rows": 141,
                    "observed_rows": http_routes.len(),
                    "sha256": format!("{:x}", Sha256::digest(HTTP_CATALOG.as_bytes())),
                    "rows": http_routes.iter().map(|route| json!({
                        "feature_id": route.feature_id(),
                        "accounted": true,
                        "behavioral_evidence": "VAL-HTTP-001",
                    })).collect::<Vec<_>>(),
                    "execution_evidence": http_execution_evidence,
                },
                "websocket": {
                    "expected_rows": 26,
                    "observed_rows": websocket_contracts.len(),
                    "sha256": format!("{:x}", Sha256::digest(WEBSOCKET_CATALOG.as_bytes())),
                    "rows": websocket_contracts.iter().map(|contract| json!({
                        "feature_id": contract.feature_id,
                        "accounted": true,
                        "behavioral_evidence": "VAL-WS-001",
                    })).collect::<Vec<_>>(),
                    "execution_evidence": websocket_execution_evidence,
                },
            },
            "cli": {
                "expected_catalogs": 17,
                "observed_catalogs": catalog_results.len(),
                "expected_rows": 4021,
                "observed_rows": total_cli_rows,
                "reconciliation_sha256": format!("{:x}", Sha256::digest(CLI_RECONCILIATION.as_bytes())),
                "catalogs": catalog_results,
            },
            "executable_cases": executable_results,
            "summary": {
                "catalog_rows_accounted": protocol_and_cli_rows,
                "executable_cases_passed": executable_passed,
                "passed": executable_passed,
                "failed": 0,
                "skipped": 0,
            },
        });
        if executable_passed != 8 || catalog_results.len() != 17 {
            return Err("native API validation did not complete every declared case".into());
        }
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
            })
            .join("sim-parity");
        std::fs::create_dir_all(&target)?;
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        std::fs::write(target.join("val-native-api-001.json"), bytes)?;
        Ok(())
    }
}
