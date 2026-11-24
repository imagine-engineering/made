use std::sync::Arc;

use agent_client_protocol::schema::v1 as acp;
use async_trait::async_trait;
use collaboration_domain::{
    JobIdentity, PresenceError, PresenceMutationOutcome, PresenceProjection, PresenceSnapshot,
    PrincipalId, SignedPresenceObservation, TenantContext,
};
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use remote::{
    agent_provider_discovery::AgentProviderDiscoveryReport,
    agent_provider_lifecycle::{
        AgentProviderDeployDisposition, AgentProviderDeploymentRecord,
        AgentProviderDeploymentRunner, AgentProviderExecutionApproval, AgentProviderLifecycleError,
        RemoteAgentLifecycleState, RemoteAgentProviderLifecycle, RemoteAgentShutdown,
        RemoteAgentTerminationDisposition,
    },
    agent_provider_protocol::AgentProviderCancellation,
};
use uuid::Uuid;

use crate::{
    collaboration_session::{CollaborationSessionIdentity, CollaborationSessionScope},
    jobs::{
        JobExecutionAuthority, JobExecutionCoordinator, JobExecutionDisposition, JobExecutionError,
        JobExecutionRequest, NativeJobRunOutcome, NativeJobRuntime, NativeJobRuntimeError,
    },
    remote_provider_config::RemoteProviderDeployTemplate,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemoteExecutionBinding {
    job_identity: JobIdentity,
    executor_principal_id: PrincipalId,
    session_identity: CollaborationSessionIdentity,
}

impl RemoteExecutionBinding {
    pub fn new(
        job_identity: JobIdentity,
        executor_principal_id: PrincipalId,
        channel_id: Uuid,
        thread_id: Option<Uuid>,
    ) -> Result<Self, RemoteExecutionError> {
        if executor_principal_id.as_uuid().is_nil() {
            return Err(RemoteExecutionError::InvalidBinding);
        }
        let session_identity = CollaborationSessionIdentity::new(
            job_identity.community_id().as_uuid(),
            CollaborationSessionScope::Job {
                channel_id,
                thread_id,
                job_id: job_identity.job_id().as_uuid(),
            },
        )
        .map_err(|_| RemoteExecutionError::InvalidBinding)?;
        Ok(Self {
            job_identity,
            executor_principal_id,
            session_identity,
        })
    }

    pub const fn job_identity(self) -> JobIdentity {
        self.job_identity
    }

    pub const fn executor_principal_id(self) -> PrincipalId {
        self.executor_principal_id
    }

    pub const fn session_identity(self) -> CollaborationSessionIdentity {
        self.session_identity
    }

    fn matches_request(self, request: &JobExecutionRequest) -> bool {
        self.job_identity == request.identity()
            && self.executor_principal_id == request.executor_principal_id()
            && self.session_identity.community_id() == request.identity().community_id().as_uuid()
            && self.session_identity.scope()
                == (CollaborationSessionScope::Job {
                    channel_id: request.channel_id(),
                    thread_id: request.thread_id(),
                    job_id: request.identity().job_id().as_uuid(),
                })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemoteSubstrateCapabilities {
    pub canonical_inspection: bool,
    pub canonical_signed_presence: bool,
    pub owner_authorized_shutdown: bool,
    pub provider_inspection: bool,
    pub provider_termination: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemotePresenceFreshness {
    NeverObserved,
    Fresh,
    Stale,
    Offline,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemoteExecutionPresence {
    pub snapshot: PresenceSnapshot,
    pub freshness: RemotePresenceFreshness,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteExecutionPhase {
    Prepared,
    Running,
    Completed,
    Failed,
    Cancelled,
    Disconnected,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteExecutionInspection {
    pub binding: RemoteExecutionBinding,
    pub session_id: Option<acp::SessionId>,
    pub deployment: Option<AgentProviderDeploymentRecord>,
    pub lifecycle_state: RemoteAgentLifecycleState,
    pub phase: RemoteExecutionPhase,
    pub presence: RemoteExecutionPresence,
    pub capabilities: RemoteSubstrateCapabilities,
}

#[async_trait(?Send)]
pub trait RemoteAgentSessionTransport {
    async fn create_session(
        &mut self,
        identity: CollaborationSessionIdentity,
        deployment: &AgentProviderDeploymentRecord,
    ) -> Result<acp::SessionId, RemoteExecutionTransportError>;

    async fn resume_session(
        &mut self,
        session_id: &acp::SessionId,
        deployment: &AgentProviderDeploymentRecord,
    ) -> Result<(), RemoteExecutionTransportError>;

    async fn run_job(
        &mut self,
        request: acp::PromptRequest,
        deployment: &AgentProviderDeploymentRecord,
    ) -> Result<NativeJobRunOutcome, RemoteExecutionTransportError>;

    async fn cancel_session(
        &mut self,
        session_id: &acp::SessionId,
        deployment: &AgentProviderDeploymentRecord,
    ) -> Result<(), RemoteExecutionTransportError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("remote agent session transport is unavailable")]
pub struct RemoteExecutionTransportError;

pub struct RemoteExecutionRuntime<R, T> {
    binding: RemoteExecutionBinding,
    lifecycle: RemoteAgentProviderLifecycle<R>,
    discovery: AgentProviderDiscoveryReport,
    approval: AgentProviderExecutionApproval,
    deploy_template: Option<RemoteProviderDeployTemplate>,
    credentials: Arc<dyn CredentialsProvider>,
    cx: AsyncApp,
    provider_cancellation: AgentProviderCancellation,
    background_executor: gpui::BackgroundExecutor,
    transport: T,
    presence: PresenceProjection,
    last_signed_presence_was_live: bool,
    signed_presence_observed: bool,
    session_id: Option<acp::SessionId>,
    deployment: Option<AgentProviderDeploymentRecord>,
    phase: RemoteExecutionPhase,
}

impl<R, T> RemoteExecutionRuntime<R, T>
where
    R: AgentProviderDeploymentRunner,
    T: RemoteAgentSessionTransport + RemoteAgentShutdown,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        binding: RemoteExecutionBinding,
        lifecycle: RemoteAgentProviderLifecycle<R>,
        discovery: AgentProviderDiscoveryReport,
        approval: AgentProviderExecutionApproval,
        deploy_template: RemoteProviderDeployTemplate,
        credentials: Arc<dyn CredentialsProvider>,
        cx: AsyncApp,
        provider_cancellation: AgentProviderCancellation,
        background_executor: gpui::BackgroundExecutor,
        transport: T,
        presence: PresenceProjection,
    ) -> Result<Self, RemoteExecutionError> {
        let presence_subject = presence.subject();
        let lifecycle_identity = lifecycle.inspect().agent_identity;
        if presence_subject.community_id != binding.job_identity.community_id()
            || presence_subject.principal_id != binding.executor_principal_id
            || lifecycle_identity != encode_hex(presence_subject.nostr_public_key.as_bytes())
            || deploy_template.agent_identity() != lifecycle_identity
        {
            return Err(RemoteExecutionError::InvalidBinding);
        }
        Ok(Self {
            binding,
            lifecycle,
            discovery,
            approval,
            deploy_template: Some(deploy_template),
            credentials,
            cx,
            provider_cancellation,
            background_executor,
            transport,
            presence,
            last_signed_presence_was_live: false,
            signed_presence_observed: false,
            session_id: None,
            deployment: None,
            phase: RemoteExecutionPhase::Prepared,
        })
    }

    pub fn apply_signed_presence(
        &mut self,
        observation: SignedPresenceObservation,
        observed_at_millis: u64,
    ) -> Result<PresenceMutationOutcome, RemoteExecutionError> {
        let outcome = self.presence.apply_signed(observation)?;
        if outcome == PresenceMutationOutcome::IgnoredStale {
            return Ok(outcome);
        }
        let snapshot = self.presence.snapshot(observed_at_millis);
        self.signed_presence_observed = true;
        self.last_signed_presence_was_live = snapshot.active_sources.signed;
        Ok(outcome)
    }

    pub fn inspection(&self, now_millis: u64) -> RemoteExecutionInspection {
        let lifecycle = self.lifecycle.inspect();
        let presence = self.presence.snapshot(now_millis);
        let freshness = if !self.signed_presence_observed {
            RemotePresenceFreshness::NeverObserved
        } else if presence.active_sources.signed {
            RemotePresenceFreshness::Fresh
        } else if self.last_signed_presence_was_live {
            RemotePresenceFreshness::Stale
        } else {
            RemotePresenceFreshness::Offline
        };
        RemoteExecutionInspection {
            binding: self.binding,
            session_id: self.session_id.clone(),
            deployment: self.deployment.clone(),
            lifecycle_state: lifecycle.state,
            phase: self.phase,
            presence: RemoteExecutionPresence {
                snapshot: presence,
                freshness,
            },
            capabilities: RemoteSubstrateCapabilities {
                canonical_inspection: true,
                canonical_signed_presence: true,
                owner_authorized_shutdown: true,
                provider_inspection: lifecycle.provider_inspection_supported,
                provider_termination: lifecycle.provider_termination_supported,
            },
        }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn confirm_shutdown_disconnect(
        &mut self,
        provider_agent_id: &str,
    ) -> Result<(), RemoteExecutionError> {
        self.lifecycle.confirm_terminated(provider_agent_id)?;
        Ok(())
    }

    async fn deployment(&mut self) -> Result<AgentProviderDeploymentRecord, NativeJobRuntimeError> {
        if let Some(deployment) = &self.deployment {
            return Ok(deployment.clone());
        }
        let template = self
            .deploy_template
            .take()
            .ok_or(NativeJobRuntimeError::Unavailable)?;
        let input = template
            .resolve(self.credentials.as_ref(), &self.cx)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        let disposition = self
            .lifecycle
            .deploy(
                &self.discovery,
                &self.approval,
                input,
                &self.provider_cancellation,
                &self.background_executor,
            )
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        let deployment = match disposition {
            AgentProviderDeployDisposition::Deployed(deployment)
            | AgentProviderDeployDisposition::AlreadyDeployed(deployment) => deployment,
        };
        self.deployment = Some(deployment.clone());
        Ok(deployment)
    }
}

#[async_trait(?Send)]
impl<R, T> NativeJobRuntime for RemoteExecutionRuntime<R, T>
where
    R: AgentProviderDeploymentRunner,
    T: RemoteAgentSessionTransport + RemoteAgentShutdown,
{
    async fn create_session(
        &mut self,
        identity: CollaborationSessionIdentity,
    ) -> Result<acp::SessionId, NativeJobRuntimeError> {
        if identity != self.binding.session_identity {
            return Err(NativeJobRuntimeError::Unavailable);
        }
        let deployment = self.deployment().await?;
        let session_id = self
            .transport
            .create_session(identity, &deployment)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        self.session_id = Some(session_id.clone());
        self.phase = RemoteExecutionPhase::Running;
        Ok(session_id)
    }

    async fn resume_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError> {
        let deployment = self.deployment().await?;
        self.transport
            .resume_session(session_id, &deployment)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        self.session_id = Some(session_id.clone());
        self.phase = RemoteExecutionPhase::Running;
        Ok(())
    }

    async fn run_job(
        &mut self,
        request: acp::PromptRequest,
    ) -> Result<NativeJobRunOutcome, NativeJobRuntimeError> {
        if self.session_id.as_ref() != Some(&request.session_id) {
            return Err(NativeJobRuntimeError::Unavailable);
        }
        let deployment = self.deployment().await?;
        let outcome = self
            .transport
            .run_job(request, &deployment)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        self.phase = match outcome {
            NativeJobRunOutcome::Completed { .. } => RemoteExecutionPhase::Completed,
            NativeJobRunOutcome::Failed { .. } => RemoteExecutionPhase::Failed,
            NativeJobRunOutcome::CancellationRequested { .. } => RemoteExecutionPhase::Running,
            NativeJobRunOutcome::Crashed => RemoteExecutionPhase::Disconnected,
        };
        Ok(outcome)
    }

    async fn cancel_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError> {
        if self.session_id.as_ref() != Some(session_id) {
            return Err(NativeJobRuntimeError::Unavailable);
        }
        let deployment = self.deployment().await?;
        self.transport
            .cancel_session(session_id, &deployment)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?;
        let operation_id = format!("cancel-{}", deployment.generation);
        let (lifecycle, transport) = (&mut self.lifecycle, &self.transport);
        match lifecycle
            .terminate(operation_id, transport)
            .await
            .map_err(|_| NativeJobRuntimeError::Unavailable)?
        {
            RemoteAgentTerminationDisposition::Requested(_)
            | RemoteAgentTerminationDisposition::AlreadyRequested(_)
            | RemoteAgentTerminationDisposition::AlreadyTerminated(_) => {}
            RemoteAgentTerminationDisposition::NotDeployed => {
                return Err(NativeJobRuntimeError::Unavailable);
            }
        }
        self.phase = RemoteExecutionPhase::Cancelled;
        Ok(())
    }
}

#[derive(Default)]
pub struct RemoteExecutionCoordinator {
    jobs: JobExecutionCoordinator,
}

impl RemoteExecutionCoordinator {
    pub async fn execute_once<A, R, T>(
        &mut self,
        authority: &A,
        runtime: &mut RemoteExecutionRuntime<R, T>,
        tenant: &TenantContext,
        request: &JobExecutionRequest,
    ) -> Result<JobExecutionDisposition, RemoteExecutionError>
    where
        A: JobExecutionAuthority,
        R: AgentProviderDeploymentRunner,
        T: RemoteAgentSessionTransport + RemoteAgentShutdown,
    {
        if !runtime.binding.matches_request(request) {
            return Err(RemoteExecutionError::InvalidBinding);
        }
        self.jobs
            .execute_once(authority, runtime, tenant, request)
            .await
            .map_err(RemoteExecutionError::Job)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RemoteExecutionError {
    #[error("remote execution binding is invalid")]
    InvalidBinding,
    #[error(transparent)]
    Job(#[from] JobExecutionError),
    #[error(transparent)]
    Presence(#[from] PresenceError),
    #[error(transparent)]
    Provider(#[from] AgentProviderLifecycleError),
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::{BTreeMap, VecDeque},
        fs,
        future::Future,
        path::Path,
        pin::Pin,
        sync::{Arc, Mutex},
    };

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;

    use collaboration_domain::{
        AggregateId, AggregateVersion, CommunityId, CommunityMembership, Job, JobCommand,
        JobCommandKind, JobCommandOutcome, JobState, MembershipRole, MembershipStatus,
        NostrEventId, NostrPublicKey, OperationId, PresenceStatus, PresenceSubject,
        TrustedTenantRoute,
    };
    use gpui::TestAppContext;
    use remote::{
        agent_provider_discovery::{AgentProviderSearchDirectory, AgentProviderTrust},
        agent_provider_lifecycle::{
            AgentProviderDeployInput, RemoteAgentShutdownError, RemoteAgentShutdownRequest,
            StagedAgentProviderDeployment,
        },
        agent_provider_protocol::AgentProviderInfo,
    };
    use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
    use serde_json::json;

    use crate::jobs::{
        JobExecutionAuthorityError, JobExecutorLease, JobExecutorLeaseRequest,
        JobLeaseMutationOutcome, JobLeaseReleaseReason,
    };
    use crate::remote_provider_config::{
        RemoteProviderProjectConfiguration, RemoteProviderSecretReferences,
    };
    use agent_settings::managed_agent::ProtectedCredentialReference;

    use super::*;

    const COMMUNITY: u128 = 1;
    const JOB: u128 = 2;
    const REQUESTER: u128 = 3;
    const EXECUTOR: u128 = 4;
    const CHANNEL: u128 = 5;
    const AGENT_SECRET: [u8; 32] = [6; 32];
    const IDENTITY_REFERENCE: &str = "credentials/remote-execution/identity";

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn job_identity() -> JobIdentity {
        JobIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(COMMUNITY)),
            AggregateId::from_uuid(Uuid::from_u128(JOB)),
        )
        .expect("job identity")
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(
                    CommunityId::from_uuid(Uuid::from_u128(COMMUNITY)),
                    "remote-execution-test",
                )
                .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn job_command(version: u64, occurred_at_millis: u64, kind: JobCommandKind) -> JobCommand {
        JobCommand::new(
            job_identity(),
            OperationId::from_uuid(Uuid::from_u128(100 + u128::from(version))),
            AggregateVersion::new(version).expect("positive version"),
            occurred_at_millis,
            kind,
        )
        .expect("job command")
    }

    fn accepted_job() -> Job {
        let mut job = Job::request(job_command(
            1,
            10,
            JobCommandKind::Request {
                requester_principal_id: principal(REQUESTER),
                target_executor_principal_id: principal(EXECUTOR),
            },
        ))
        .expect("requested job");
        job.apply(job_command(
            2,
            20,
            JobCommandKind::Accept {
                executor_principal_id: principal(EXECUTOR),
            },
        ))
        .expect("accepted job");
        job
    }

    fn execution_request(lease_id: u128) -> JobExecutionRequest {
        JobExecutionRequest::new(
            job_identity(),
            principal(EXECUTOR),
            Uuid::from_u128(CHANNEL),
            None,
            Uuid::from_u128(lease_id),
            30,
            40,
            50,
            "Run the remote collaboration job",
        )
        .expect("execution request")
    }

    struct AuthorityState {
        job: Job,
        active_lease: Option<JobExecutorLease>,
        next_generation: u64,
        released: Vec<(JobExecutorLease, JobLeaseReleaseReason)>,
        result_publications: usize,
    }

    struct FakeAuthority {
        state: RefCell<AuthorityState>,
    }

    impl FakeAuthority {
        fn new() -> Self {
            Self {
                state: RefCell::new(AuthorityState {
                    job: accepted_job(),
                    active_lease: None,
                    next_generation: 0,
                    released: Vec::new(),
                    result_publications: 0,
                }),
            }
        }
    }

    struct FakeCredentialsProvider {
        read_count: Arc<Mutex<usize>>,
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move {
                *self.read_count.lock().expect("credential read count lock") += 1;
                Ok((url == IDENTITY_REFERENCE)
                    .then(|| ("remote-agent".to_owned(), AGENT_SECRET.to_vec())))
            })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            _username: &'a str,
            _password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
            Box::pin(async { anyhow::bail!("read-only fixture") })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>> {
            Box::pin(async { anyhow::bail!("read-only fixture") })
        }
    }

    #[async_trait(?Send)]
    impl JobExecutionAuthority for FakeAuthority {
        async fn load_job(
            &self,
            tenant: &TenantContext,
            identity: JobIdentity,
        ) -> Result<Option<Job>, JobExecutionAuthorityError> {
            if tenant.community_id() != identity.community_id() {
                return Err(JobExecutionAuthorityError::Unavailable);
            }
            let state = self.state.borrow();
            Ok((state.job.identity() == identity).then(|| state.job.clone()))
        }

        async fn acquire_executor_lease(
            &self,
            _tenant: &TenantContext,
            request: JobExecutorLeaseRequest,
        ) -> Result<JobExecutorLease, JobExecutionAuthorityError> {
            let mut state = self.state.borrow_mut();
            if let Some(active) = state.active_lease {
                if active.identity() == request.identity()
                    && active.job_version() == request.job_version()
                    && active.lease_id() == request.lease_id()
                    && active.executor_principal_id() == request.executor_principal_id()
                    && active.acquired_at_millis() == request.acquired_at_millis()
                    && active.expires_at_millis() == request.expires_at_millis()
                    && active.recovery_after_millis() == request.recovery_after_millis()
                {
                    return Ok(active);
                }
                return Err(JobExecutionAuthorityError::LeaseUnavailable);
            }
            state.next_generation = state
                .next_generation
                .checked_add(1)
                .ok_or(JobExecutionAuthorityError::Unavailable)?;
            let generation = AggregateVersion::new(state.next_generation)
                .ok_or(JobExecutionAuthorityError::Unavailable)?;
            let lease = JobExecutorLease::new(request, generation)
                .map_err(|_| JobExecutionAuthorityError::Unavailable)?;
            state.active_lease = Some(lease);
            Ok(lease)
        }

        async fn active_executor_lease(
            &self,
            _tenant: &TenantContext,
            identity: JobIdentity,
        ) -> Result<Option<JobExecutorLease>, JobExecutionAuthorityError> {
            Ok(self
                .state
                .borrow()
                .active_lease
                .filter(|lease| lease.identity() == identity))
        }

        async fn transition(
            &self,
            _tenant: &TenantContext,
            command: JobCommand,
        ) -> Result<JobCommandOutcome, JobExecutionAuthorityError> {
            let mut state = self.state.borrow_mut();
            let outcome = state
                .job
                .apply(command.clone())
                .map_err(|_| JobExecutionAuthorityError::Conflict)?;
            if outcome == JobCommandOutcome::Applied
                && matches!(command.kind(), JobCommandKind::Result { .. })
            {
                state.result_publications += 1;
            }
            Ok(outcome)
        }

        async fn release_executor_lease(
            &self,
            _tenant: &TenantContext,
            lease: JobExecutorLease,
            _released_at_millis: u64,
            reason: JobLeaseReleaseReason,
        ) -> Result<JobLeaseMutationOutcome, JobExecutionAuthorityError> {
            let mut state = self.state.borrow_mut();
            if state.active_lease == Some(lease) {
                state.active_lease = None;
                state.released.push((lease, reason));
                Ok(JobLeaseMutationOutcome::Applied)
            } else if state.released.contains(&(lease, reason)) {
                Ok(JobLeaseMutationOutcome::Duplicate)
            } else {
                Err(JobExecutionAuthorityError::LeaseLost)
            }
        }
    }

    #[derive(Clone)]
    struct FakeRunner {
        deploy_count: Arc<Mutex<usize>>,
    }

    #[async_trait]
    impl AgentProviderDeploymentRunner for FakeRunner {
        async fn deploy(
            &self,
            _candidate: &remote::agent_provider_discovery::AgentProviderCandidate,
            _input: &AgentProviderDeployInput,
            _cancellation: &AgentProviderCancellation,
            _background_executor: &gpui::BackgroundExecutor,
        ) -> Result<StagedAgentProviderDeployment, AgentProviderLifecycleError> {
            *self.deploy_count.lock().expect("deploy count lock") += 1;
            Ok(StagedAgentProviderDeployment {
                info: AgentProviderInfo {
                    name: "fixture".to_owned(),
                    version: "1".to_owned(),
                    protocol_version: 1,
                    description: "fixture".to_owned(),
                    config_schema: json!({}),
                },
                agent_id: "remote-agent-1".to_owned(),
                provider_sha256: "fixture-sha256".to_owned(),
            })
        }
    }

    #[derive(Default)]
    struct TransportState {
        outcomes: VecDeque<NativeJobRunOutcome>,
        create_count: usize,
        resume_count: usize,
        run_count: usize,
        cancel_count: usize,
        shutdown_requests: Vec<RemoteAgentShutdownRequest>,
    }

    #[derive(Clone)]
    struct FakeTransport {
        state: Arc<Mutex<TransportState>>,
    }

    impl FakeTransport {
        fn new(outcomes: impl IntoIterator<Item = NativeJobRunOutcome>) -> Self {
            Self {
                state: Arc::new(Mutex::new(TransportState {
                    outcomes: outcomes.into_iter().collect(),
                    ..TransportState::default()
                })),
            }
        }
    }

    #[async_trait(?Send)]
    impl RemoteAgentSessionTransport for FakeTransport {
        async fn create_session(
            &mut self,
            _identity: CollaborationSessionIdentity,
            _deployment: &AgentProviderDeploymentRecord,
        ) -> Result<acp::SessionId, RemoteExecutionTransportError> {
            self.state
                .lock()
                .expect("transport state lock")
                .create_count += 1;
            Ok(acp::SessionId::new("remote-session-1"))
        }

        async fn resume_session(
            &mut self,
            _session_id: &acp::SessionId,
            _deployment: &AgentProviderDeploymentRecord,
        ) -> Result<(), RemoteExecutionTransportError> {
            self.state
                .lock()
                .expect("transport state lock")
                .resume_count += 1;
            Ok(())
        }

        async fn run_job(
            &mut self,
            _request: acp::PromptRequest,
            _deployment: &AgentProviderDeploymentRecord,
        ) -> Result<NativeJobRunOutcome, RemoteExecutionTransportError> {
            let mut state = self.state.lock().expect("transport state lock");
            state.run_count += 1;
            state
                .outcomes
                .pop_front()
                .ok_or(RemoteExecutionTransportError)
        }

        async fn cancel_session(
            &mut self,
            _session_id: &acp::SessionId,
            _deployment: &AgentProviderDeploymentRecord,
        ) -> Result<(), RemoteExecutionTransportError> {
            self.state
                .lock()
                .expect("transport state lock")
                .cancel_count += 1;
            Ok(())
        }
    }

    #[async_trait]
    impl RemoteAgentShutdown for FakeTransport {
        async fn request_shutdown(
            &self,
            request: &RemoteAgentShutdownRequest,
        ) -> Result<(), RemoteAgentShutdownError> {
            self.state
                .lock()
                .expect("transport state lock")
                .shutdown_requests
                .push(request.clone());
            Ok(())
        }
    }

    fn presence_subject() -> PresenceSubject {
        PresenceSubject {
            community_id: job_identity().community_id(),
            principal_id: principal(EXECUTOR),
            nostr_public_key: NostrPublicKey::from_bytes(agent_public_key()),
        }
    }

    fn agent_public_key() -> [u8; 32] {
        let secret = SecretKey::from_slice(&AGENT_SECRET).expect("fixture secret key");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
        let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
        public_key.serialize()
    }

    fn presence() -> PresenceProjection {
        PresenceProjection::new(
            presence_subject(),
            CommunityMembership {
                community_id: job_identity().community_id(),
                principal_id: principal(EXECUTOR),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        )
        .expect("presence projection")
    }

    fn signed_presence(
        status: PresenceStatus,
        observed_at_millis: u64,
        expires_at_millis: Option<u64>,
    ) -> SignedPresenceObservation {
        SignedPresenceObservation::from_verified_event(
            presence_subject(),
            NostrEventId::from_bytes([observed_at_millis as u8; 32]),
            presence_subject().nostr_public_key,
            status,
            observed_at_millis,
            expires_at_millis,
        )
        .expect("signed presence")
    }

    fn provider_fixture(
        directory: &Path,
    ) -> (AgentProviderDiscoveryReport, AgentProviderExecutionApproval) {
        let provider_path = directory.join("buzz-backend-fixture");
        fs::write(&provider_path, b"fixture").expect("write provider fixture");
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&provider_path)
                .expect("provider metadata")
                .permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&provider_path, permissions).expect("provider permissions");
        }
        let discovery =
            AgentProviderDiscoveryReport::discover([AgentProviderSearchDirectory::new(
                directory,
                AgentProviderTrust::Untrusted,
            )]);
        let approval = AgentProviderExecutionApproval {
            executable: discovery
                .resolve("fixture")
                .expect("resolve provider")
                .executable_reference(),
            allow_untrusted: true,
        };
        (discovery, approval)
    }

    fn deploy_template(directory: &Path) -> RemoteProviderDeployTemplate {
        let serde_json::Value::Object(agent) = json!({"name": "fixture"}) else {
            panic!("fixture agent object")
        };
        RemoteProviderDeployTemplate::new(
            "deploy-job-2",
            directory.to_owned(),
            &encode_hex(&agent_public_key()),
            agent,
            RemoteProviderProjectConfiguration::new(BTreeMap::new())
                .expect("provider configuration"),
            RemoteProviderSecretReferences::new(
                ProtectedCredentialReference::parse(IDENTITY_REFERENCE)
                    .expect("identity credential reference"),
                None,
                BTreeMap::new(),
            )
            .expect("secret references"),
        )
        .expect("deploy template")
    }

    fn runtime(
        cx: &TestAppContext,
        outcomes: impl IntoIterator<Item = NativeJobRunOutcome>,
    ) -> (
        RemoteExecutionRuntime<FakeRunner, FakeTransport>,
        Arc<Mutex<usize>>,
        Arc<Mutex<TransportState>>,
        Arc<Mutex<usize>>,
        tempfile::TempDir,
    ) {
        let directory = tempfile::tempdir().expect("provider directory");
        let (discovery, approval) = provider_fixture(directory.path());
        let deploy_count = Arc::new(Mutex::new(0));
        let runner = FakeRunner {
            deploy_count: deploy_count.clone(),
        };
        let transport = FakeTransport::new(outcomes);
        let transport_state = transport.state.clone();
        let credential_read_count = Arc::new(Mutex::new(0));
        let agent_identity = encode_hex(&agent_public_key());
        let lifecycle =
            RemoteAgentProviderLifecycle::with_runner("fixture", agent_identity, runner)
                .expect("provider lifecycle");
        let binding = RemoteExecutionBinding::new(
            job_identity(),
            principal(EXECUTOR),
            Uuid::from_u128(CHANNEL),
            None,
        )
        .expect("remote binding");
        let runtime = RemoteExecutionRuntime::new(
            binding,
            lifecycle,
            discovery,
            approval,
            deploy_template(directory.path()),
            Arc::new(FakeCredentialsProvider {
                read_count: credential_read_count.clone(),
            }),
            cx.to_async(),
            AgentProviderCancellation::default(),
            cx.background_executor.clone(),
            transport,
            presence(),
        )
        .expect("remote runtime");
        (
            runtime,
            deploy_count,
            transport_state,
            credential_read_count,
            directory,
        )
    }

    #[gpui::test]
    async fn launch_result_and_terminal_retry_execute_exactly_once(cx: &mut TestAppContext) {
        let authority = FakeAuthority::new();
        let (mut runtime, deploy_count, transport_state, credential_read_count, _directory) =
            runtime(
                cx,
                [NativeJobRunOutcome::Completed {
                    completed_at_millis: 35,
                }],
            );
        let request = execution_request(200);
        let mut coordinator = RemoteExecutionCoordinator::default();

        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &request)
                .await
                .expect("complete remote job"),
            JobExecutionDisposition::Completed
        );
        assert_eq!(
            runtime.inspection(35).phase,
            RemoteExecutionPhase::Completed
        );
        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &request)
                .await
                .expect("terminal retry"),
            JobExecutionDisposition::AlreadyTerminal(collaboration_domain::JobStateKind::Completed)
        );

        assert_eq!(*deploy_count.lock().expect("deploy count lock"), 1);
        assert_eq!(
            *credential_read_count
                .lock()
                .expect("credential read count lock"),
            1
        );
        let transport_state = transport_state.lock().expect("transport state lock");
        assert_eq!(transport_state.create_count, 1);
        assert_eq!(transport_state.run_count, 1);
        let authority_state = authority.state.borrow();
        assert!(matches!(
            authority_state.job.state(),
            JobState::Completed { .. }
        ));
        assert_eq!(authority_state.result_publications, 1);
    }

    #[gpui::test]
    async fn signed_heartbeat_expires_without_controlling_execution(cx: &mut TestAppContext) {
        let (mut runtime, _deploy_count, _transport_state, _credential_read_count, _directory) =
            runtime(
                cx,
                [NativeJobRunOutcome::Completed {
                    completed_at_millis: 35,
                }],
            );
        runtime
            .apply_signed_presence(
                signed_presence(PresenceStatus::Online, 1_000, Some(181_000)),
                1_000,
            )
            .expect("apply heartbeat");

        let fresh = runtime.inspection(180_999);
        assert_eq!(fresh.presence.freshness, RemotePresenceFreshness::Fresh);
        assert_eq!(fresh.presence.snapshot.status, PresenceStatus::Online);
        let stale = runtime.inspection(181_000);
        assert_eq!(stale.presence.freshness, RemotePresenceFreshness::Stale);
        assert_eq!(stale.presence.snapshot.status, PresenceStatus::Offline);
        assert_eq!(stale.phase, RemoteExecutionPhase::Prepared);
        assert!(stale.capabilities.canonical_signed_presence);
        assert!(!stale.capabilities.provider_inspection);
        assert!(!stale.capabilities.provider_termination);
    }

    #[gpui::test]
    async fn cancellation_stops_the_owned_session_and_requests_shutdown(cx: &mut TestAppContext) {
        let authority = FakeAuthority::new();
        let (mut runtime, _deploy_count, transport_state, _credential_read_count, _directory) =
            runtime(
                cx,
                [NativeJobRunOutcome::CancellationRequested {
                    actor_principal_id: principal(REQUESTER),
                    cancelled_at_millis: 35,
                }],
            );
        let mut coordinator = RemoteExecutionCoordinator::default();
        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &execution_request(201),)
                .await
                .expect("cancel remote job"),
            JobExecutionDisposition::Cancelled
        );

        assert_eq!(
            runtime.inspection(35).phase,
            RemoteExecutionPhase::Cancelled
        );
        let state = transport_state.lock().expect("transport state lock");
        assert_eq!(state.cancel_count, 1);
        assert_eq!(state.shutdown_requests.len(), 1);
        assert_eq!(state.shutdown_requests[0].agent_id, "remote-agent-1");
        drop(state);
        assert!(matches!(
            runtime.confirm_shutdown_disconnect("different-agent"),
            Err(RemoteExecutionError::Provider(
                AgentProviderLifecycleError::AgentIdMismatch
            ))
        ));
        runtime
            .confirm_shutdown_disconnect("remote-agent-1")
            .expect("confirm exact disconnected provider agent");
        assert!(matches!(
            runtime.inspection(35).lifecycle_state,
            RemoteAgentLifecycleState::Terminated { .. }
        ));
    }

    #[gpui::test]
    async fn disconnect_retains_canonical_lease_for_recovery(cx: &mut TestAppContext) {
        let authority = FakeAuthority::new();
        let (mut runtime, _deploy_count, transport_state, _credential_read_count, _directory) =
            runtime(cx, [NativeJobRunOutcome::Crashed]);
        let mut coordinator = RemoteExecutionCoordinator::default();
        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &execution_request(202),)
                .await
                .expect("record disconnect"),
            JobExecutionDisposition::Crashed
        );

        assert_eq!(
            runtime.inspection(35).phase,
            RemoteExecutionPhase::Disconnected
        );
        assert!(authority.state.borrow().active_lease.is_some());
        let state = transport_state.lock().expect("transport state lock");
        assert_eq!(state.shutdown_requests.len(), 0);
        assert_eq!(state.cancel_count, 0);
    }
}
