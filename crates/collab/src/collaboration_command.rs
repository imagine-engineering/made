use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{AggregateVersion, AuthenticatedPrincipal, OperationId, TenantContext};

pub const CURRENT_COMMAND_CONTRACT_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandAdapter {
    NostrInProcess,
    NostrTemporarySidecar,
    ZedRpc,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainCommand<P> {
    contract_version: u16,
    operation_id: OperationId,
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    expected_version: Option<AggregateVersion>,
    predecessor: Option<AggregateVersion>,
    originating_adapter: CommandAdapter,
    payload: P,
}

impl<P> DomainCommand<P> {
    pub fn new(
        operation_id: OperationId,
        tenant: TenantContext,
        principal: AuthenticatedPrincipal,
        expected_version: Option<AggregateVersion>,
        predecessor: Option<AggregateVersion>,
        originating_adapter: CommandAdapter,
        payload: P,
    ) -> Self {
        Self {
            contract_version: CURRENT_COMMAND_CONTRACT_VERSION,
            operation_id,
            tenant,
            principal,
            expected_version,
            predecessor,
            originating_adapter,
            payload,
        }
    }

    pub const fn contract_version(&self) -> u16 {
        self.contract_version
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn tenant(&self) -> &TenantContext {
        &self.tenant
    }

    pub const fn principal(&self) -> &AuthenticatedPrincipal {
        &self.principal
    }

    pub const fn expected_version(&self) -> Option<AggregateVersion> {
        self.expected_version
    }

    pub const fn predecessor(&self) -> Option<AggregateVersion> {
        self.predecessor
    }

    pub const fn originating_adapter(&self) -> CommandAdapter {
        self.originating_adapter
    }

    pub const fn payload(&self) -> &P {
        &self.payload
    }

    pub fn into_payload(self) -> P {
        self.payload
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DomainCommandReceipt {
    operation_id: OperationId,
    authoritative_version: AggregateVersion,
    disposition: DomainCommandDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainCommandDisposition {
    Applied,
    Duplicate,
}

impl DomainCommandReceipt {
    pub const fn new(operation_id: OperationId, authoritative_version: AggregateVersion) -> Self {
        Self {
            operation_id,
            authoritative_version,
            disposition: DomainCommandDisposition::Applied,
        }
    }

    pub const fn duplicate(
        operation_id: OperationId,
        authoritative_version: AggregateVersion,
    ) -> Self {
        Self {
            operation_id,
            authoritative_version,
            disposition: DomainCommandDisposition::Duplicate,
        }
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn authoritative_version(self) -> AggregateVersion {
        self.authoritative_version
    }

    pub const fn disposition(self) -> DomainCommandDisposition {
        self.disposition
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainCommandSubmissionError {
    Unavailable,
    Rejected,
}

impl fmt::Display for DomainCommandSubmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "domain command service is unavailable",
            Self::Rejected => "domain command was rejected",
        })
    }
}

impl Error for DomainCommandSubmissionError {}

#[async_trait]
pub trait DomainCommandSink<P>: Send + Sync {
    async fn submit(
        &self,
        command: DomainCommand<P>,
    ) -> Result<DomainCommandReceipt, DomainCommandSubmissionError>;
}
