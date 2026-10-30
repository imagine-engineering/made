use std::{error::Error, fmt};

use collaboration_domain::{AggregateVersion, OperationId};

use crate::{
    collaboration_command::{
        CommandAdapter, DomainCommand, DomainCommandReceipt, DomainCommandSink,
        DomainCommandSubmissionError,
    },
    tenant_admission::AuthorizedRpcRequest,
};

pub const MINIMUM_NOSTR_INGRESS_VERSION: u16 = 1;
pub const CURRENT_NOSTR_INGRESS_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrIngressDeployment {
    InProcess,
    TemporarySidecar,
}

impl NostrIngressDeployment {
    const fn command_adapter(self) -> CommandAdapter {
        match self {
            Self::InProcess => CommandAdapter::NostrInProcess,
            Self::TemporarySidecar => CommandAdapter::NostrTemporarySidecar,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrIngressRequest<P> {
    ingress_version: u16,
    minimum_service_version: u16,
    operation_id: OperationId,
    expected_version: Option<AggregateVersion>,
    predecessor: Option<AggregateVersion>,
    deployment: NostrIngressDeployment,
    payload: P,
}

impl<P> NostrIngressRequest<P> {
    pub fn new(
        ingress_version: u16,
        minimum_service_version: u16,
        operation_id: OperationId,
        expected_version: Option<AggregateVersion>,
        predecessor: Option<AggregateVersion>,
        deployment: NostrIngressDeployment,
        payload: P,
    ) -> Self {
        Self {
            ingress_version,
            minimum_service_version,
            operation_id,
            expected_version,
            predecessor,
            deployment,
            payload,
        }
    }
}

pub struct VersionedNostrIngress<S> {
    command_sink: S,
}

impl<S> VersionedNostrIngress<S> {
    pub const fn new(command_sink: S) -> Self {
        Self { command_sink }
    }

    pub async fn submit<P>(
        &self,
        admission: AuthorizedRpcRequest,
        request: NostrIngressRequest<P>,
    ) -> Result<DomainCommandReceipt, NostrIngressError>
    where
        P: Send,
        S: DomainCommandSink<P>,
    {
        validate_version(request.ingress_version, request.minimum_service_version)?;
        admission
            .run(move |tenant, principal| async move {
                self.command_sink
                    .submit(DomainCommand::new(
                        request.operation_id,
                        tenant,
                        principal,
                        request.expected_version,
                        request.predecessor,
                        request.deployment.command_adapter(),
                        request.payload,
                    ))
                    .await
                    .map_err(NostrIngressError::Command)
            })
            .await
    }
}

fn validate_version(
    ingress_version: u16,
    minimum_service_version: u16,
) -> Result<(), NostrIngressError> {
    if !(MINIMUM_NOSTR_INGRESS_VERSION..=CURRENT_NOSTR_INGRESS_VERSION).contains(&ingress_version)
        || minimum_service_version == 0
        || minimum_service_version > CURRENT_NOSTR_INGRESS_VERSION
    {
        return Err(NostrIngressError::UnsupportedVersion {
            minimum: MINIMUM_NOSTR_INGRESS_VERSION,
            current: CURRENT_NOSTR_INGRESS_VERSION,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrIngressError {
    UnsupportedVersion { minimum: u16, current: u16 },
    Command(DomainCommandSubmissionError),
}

impl fmt::Display for NostrIngressError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { minimum, current } => write!(
                formatter,
                "unsupported Nostr ingress version; supported versions are {minimum}..={current}"
            ),
            Self::Command(error) => error.fmt(formatter),
        }
    }
}

impl Error for NostrIngressError {}
