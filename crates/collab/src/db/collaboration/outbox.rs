use async_trait::async_trait;
use collaboration_domain::{
    AggregateVersion, IntegrityAlgorithm, Provenance, SourceSystem, TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend,
    QueryResult, Statement, TransactionTrait, Value,
};
use sha2::{Digest, Sha256};

use crate::collaboration_command::{
    CommandAdapter, DomainCommand, DomainCommandReceipt, DomainCommandSink,
    DomainCommandSubmissionError,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const RESERVE_COMMAND_SQL: &str = r#"
INSERT INTO public.collaboration_command_receipts (
    community_id,
    operation_id,
    contract_version,
    principal_id,
    originating_adapter,
    command_kind,
    command_fingerprint,
    expected_version,
    predecessor_version
) VALUES (
    $1, $2, $3, $4, $5, $6, $7,
    CAST($8 AS numeric), CAST($9 AS numeric)
)
ON CONFLICT (community_id, operation_id) DO NOTHING
"#;
const SELECT_RECEIPT_SQL: &str = r#"
SELECT
    receipt.contract_version,
    receipt.principal_id,
    receipt.originating_adapter,
    receipt.command_kind,
    receipt.command_fingerprint,
    receipt.authoritative_version::text AS authoritative_version_text,
    EXISTS (
        SELECT 1
        FROM public.collaboration_outbox AS outbox
        WHERE outbox.community_id = receipt.community_id
          AND outbox.operation_id = receipt.operation_id
    ) AS has_outbox
FROM public.collaboration_command_receipts AS receipt
WHERE receipt.community_id = $1 AND receipt.operation_id = $2
"#;
const ACQUIRE_TOPIC_LOCK_SQL: &str = r#"
SELECT pg_advisory_xact_lock(
    hashtextextended('zed_collaboration_outbox:' || $1::text || ':' || $2::text, 0)
)
"#;
const INSERT_OUTBOX_SQL: &str = r#"
INSERT INTO public.collaboration_outbox (
    community_id,
    operation_id,
    authoritative_version,
    topic,
    source_system,
    source_record_id,
    source_version,
    source_observed_at,
    source_integrity_algorithm,
    source_integrity_value,
    payload
) VALUES (
    $1, $2, CAST($3 AS numeric), $4, $5, $6, $7,
    to_timestamp($8::double precision / 1000), $9, $10, $11
)
"#;
const COMPLETE_RECEIPT_SQL: &str = r#"
UPDATE public.collaboration_command_receipts
SET authoritative_version = CAST($3 AS numeric), accepted_at = clock_timestamp()
WHERE community_id = $1
  AND operation_id = $2
  AND authoritative_version IS NULL
"#;

pub const MAX_COMMAND_KIND_BYTES: usize = 128;
pub const MAX_OUTBOX_TOPIC_BYTES: usize = 128;
pub const MAX_OUTBOX_PAYLOAD_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandFingerprint {
    command_kind: String,
    digest: [u8; 32],
}

impl CommandFingerprint {
    pub fn new(
        command_kind: impl Into<String>,
        canonical_command_bytes: &[u8],
    ) -> Result<Self, DomainCommandSubmissionError> {
        let command_kind = command_kind.into();
        if command_kind.is_empty()
            || command_kind.len() > MAX_COMMAND_KIND_BYTES
            || command_kind.trim() != command_kind
            || command_kind.chars().any(char::is_control)
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        Ok(Self {
            command_kind,
            digest: Sha256::digest(canonical_command_bytes).into(),
        })
    }

    pub fn command_kind(&self) -> &str {
        &self.command_kind
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutboxOperation {
    topic: String,
    provenance: Provenance,
    payload: Vec<u8>,
}

impl OutboxOperation {
    pub fn new(
        topic: impl Into<String>,
        provenance: Provenance,
        payload: Vec<u8>,
    ) -> Result<Self, DomainCommandSubmissionError> {
        let topic = topic.into();
        if topic.is_empty()
            || topic.len() > MAX_OUTBOX_TOPIC_BYTES
            || topic.trim() != topic
            || topic.chars().any(char::is_control)
            || payload.len() > MAX_OUTBOX_PAYLOAD_BYTES
            || provenance
                .source_version
                .as_ref()
                .is_some_and(|version| version.is_empty() || version.len() > 1024)
            || provenance
                .integrity
                .as_ref()
                .is_some_and(|integrity| integrity.value.is_empty() || integrity.value.len() > 1024)
            || i64::try_from(provenance.observed_at_millis).is_err()
        {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        Ok(Self {
            topic,
            provenance,
            payload,
        })
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppliedCommand {
    authoritative_version: AggregateVersion,
    outbox_operation: OutboxOperation,
}

impl AppliedCommand {
    pub const fn new(
        authoritative_version: AggregateVersion,
        outbox_operation: OutboxOperation,
    ) -> Self {
        Self {
            authoritative_version,
            outbox_operation,
        }
    }

    pub const fn authoritative_version(&self) -> AggregateVersion {
        self.authoritative_version
    }

    pub const fn outbox_operation(&self) -> &OutboxOperation {
        &self.outbox_operation
    }
}

#[async_trait]
pub trait TransactionalCommandMutation<P: Sync>: Send + Sync {
    fn transaction_isolation(&self) -> Option<sea_orm::IsolationLevel> {
        None
    }

    fn fingerprint(
        &self,
        command: &DomainCommand<P>,
    ) -> Result<CommandFingerprint, DomainCommandSubmissionError>;

    async fn admit_in_transaction(
        &self,
        _transaction: &DatabaseTransaction,
        _command: &DomainCommand<P>,
    ) -> Result<(), DomainCommandSubmissionError> {
        Ok(())
    }

    async fn apply(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<P>,
    ) -> Result<AppliedCommand, DomainCommandSubmissionError>;
}

pub struct TransactionalCommandOutbox<M> {
    connection: DatabaseConnection,
    mutation: M,
}

impl<M> TransactionalCommandOutbox<M> {
    pub fn new(connection: DatabaseConnection, mutation: M) -> Result<Self, DbBackend> {
        if connection.get_database_backend() != DatabaseBackend::Postgres {
            return Err(connection.get_database_backend());
        }
        Ok(Self {
            connection,
            mutation,
        })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }
}

#[async_trait]
impl<P, M> DomainCommandSink<P> for TransactionalCommandOutbox<M>
where
    P: Send + Sync + 'static,
    M: TransactionalCommandMutation<P>,
{
    async fn submit(
        &self,
        command: DomainCommand<P>,
    ) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
        validate_command_boundary(&command)?;
        let fingerprint = self.mutation.fingerprint(&command)?;
        let transaction = self
            .connection
            .begin_with_config(self.mutation.transaction_isolation(), None)
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        if let Err(error) = set_tenant(&transaction, command.tenant()).await {
            rollback(transaction).await;
            return Err(error);
        }

        if let Err(error) = self
            .mutation
            .admit_in_transaction(&transaction, &command)
            .await
        {
            rollback(transaction).await;
            return Err(error);
        }

        let reservation = match transaction
            .execute(reserve_statement(&command, &fingerprint))
            .await
        {
            Ok(reservation) => reservation,
            Err(_) => {
                rollback(transaction).await;
                return Err(DomainCommandSubmissionError::Unavailable);
            }
        };
        if reservation.rows_affected() == 0 {
            let receipt = duplicate_receipt(&transaction, &command, &fingerprint).await;
            match receipt {
                Ok(receipt) => {
                    transaction
                        .commit()
                        .await
                        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
                    return Ok(receipt);
                }
                Err(error) => {
                    rollback(transaction).await;
                    return Err(error);
                }
            }
        }
        if reservation.rows_affected() != 1 {
            rollback(transaction).await;
            return Err(DomainCommandSubmissionError::Unavailable);
        }

        let applied = match self.mutation.apply(&transaction, &command).await {
            Ok(applied) => applied,
            Err(error) => {
                rollback(transaction).await;
                return Err(error);
            }
        };
        if let Err(error) = insert_outbox(&transaction, &command, &applied).await {
            rollback(transaction).await;
            return Err(error);
        }
        if let Err(error) = complete_receipt(&transaction, &command, &applied).await {
            rollback(transaction).await;
            return Err(error);
        }
        transaction
            .commit()
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        Ok(DomainCommandReceipt::new(
            command.operation_id(),
            applied.authoritative_version(),
        ))
    }
}

async fn rollback(transaction: DatabaseTransaction) {
    if let Err(error) = transaction.rollback().await {
        log::error!("failed to roll back collaboration command transaction: {error}");
    }
}

fn validate_command_boundary<P>(
    command: &DomainCommand<P>,
) -> Result<(), DomainCommandSubmissionError> {
    if command.tenant().community_id() != command.principal().community_id() {
        return Err(DomainCommandSubmissionError::Rejected);
    }
    Ok(())
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), DomainCommandSubmissionError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    Ok(())
}

fn reserve_statement<P>(command: &DomainCommand<P>, fingerprint: &CommandFingerprint) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        RESERVE_COMMAND_SQL,
        [
            Value::Uuid(Some(Box::new(command.tenant().community_id().as_uuid()))),
            Value::Uuid(Some(Box::new(command.operation_id().as_uuid()))),
            i32::from(command.contract_version()).into(),
            Value::Uuid(Some(Box::new(command.principal().principal_id().as_uuid()))),
            adapter_name(command.originating_adapter()).into(),
            fingerprint.command_kind().into(),
            fingerprint.digest().to_vec().into(),
            version_value(command.expected_version()),
            version_value(command.predecessor()),
        ],
    )
}

async fn duplicate_receipt<P>(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<P>,
    fingerprint: &CommandFingerprint,
) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_RECEIPT_SQL,
            [
                Value::Uuid(Some(Box::new(command.tenant().community_id().as_uuid()))),
                Value::Uuid(Some(Box::new(command.operation_id().as_uuid()))),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?
        .ok_or(DomainCommandSubmissionError::Unavailable)?;
    validate_duplicate(row, command, fingerprint)
}

fn validate_duplicate<P>(
    row: QueryResult,
    command: &DomainCommand<P>,
    fingerprint: &CommandFingerprint,
) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
    let contract_version: i32 = row
        .try_get("", "contract_version")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let principal_id: uuid::Uuid = row
        .try_get("", "principal_id")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let adapter: String = row
        .try_get("", "originating_adapter")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let command_kind: String = row
        .try_get("", "command_kind")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let stored_fingerprint: Vec<u8> = row
        .try_get("", "command_fingerprint")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let authoritative_version: String = row
        .try_get("", "authoritative_version_text")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let has_outbox: bool = row
        .try_get("", "has_outbox")
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;

    if contract_version != i32::from(command.contract_version())
        || principal_id != command.principal().principal_id().as_uuid()
        || adapter != adapter_name(command.originating_adapter())
        || command_kind != fingerprint.command_kind()
        || stored_fingerprint != fingerprint.digest().as_slice()
    {
        return Err(DomainCommandSubmissionError::Rejected);
    }
    if !has_outbox {
        return Err(DomainCommandSubmissionError::Unavailable);
    }
    let authoritative_version = authoritative_version
        .parse::<u64>()
        .ok()
        .and_then(AggregateVersion::new)
        .ok_or(DomainCommandSubmissionError::Unavailable)?;
    Ok(DomainCommandReceipt::duplicate(
        command.operation_id(),
        authoritative_version,
    ))
}

async fn insert_outbox<P>(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<P>,
    applied: &AppliedCommand,
) -> Result<(), DomainCommandSubmissionError> {
    let operation = applied.outbox_operation();
    let provenance = operation.provenance();
    let observed_at = i64::try_from(provenance.observed_at_millis)
        .map_err(|_| DomainCommandSubmissionError::Rejected)?;
    let integrity_algorithm = provenance
        .integrity
        .as_ref()
        .map(|integrity| integrity_algorithm_name(integrity.algorithm));
    let integrity_value = provenance
        .integrity
        .as_ref()
        .map(|integrity| integrity.value.as_str());
    // Identity values are allocated before commit. Serialize writers for this replay
    // scope until commit so a visible higher cursor cannot hide an earlier transaction.
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            ACQUIRE_TOPIC_LOCK_SQL,
            [
                command.tenant().community_id().to_string().into(),
                operation.topic().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            INSERT_OUTBOX_SQL,
            [
                Value::Uuid(Some(Box::new(command.tenant().community_id().as_uuid()))),
                Value::Uuid(Some(Box::new(command.operation_id().as_uuid()))),
                applied.authoritative_version().to_string().into(),
                operation.topic().into(),
                source_system_name(provenance.source_system).into(),
                provenance.source_record_id.as_str().into(),
                provenance.source_version.as_deref().into(),
                observed_at.into(),
                integrity_algorithm.into(),
                integrity_value.into(),
                operation.payload().to_vec().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    if result.rows_affected() != 1 {
        return Err(DomainCommandSubmissionError::Unavailable);
    }
    Ok(())
}

async fn complete_receipt<P>(
    transaction: &DatabaseTransaction,
    command: &DomainCommand<P>,
    applied: &AppliedCommand,
) -> Result<(), DomainCommandSubmissionError> {
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            COMPLETE_RECEIPT_SQL,
            [
                Value::Uuid(Some(Box::new(command.tenant().community_id().as_uuid()))),
                Value::Uuid(Some(Box::new(command.operation_id().as_uuid()))),
                applied.authoritative_version().to_string().into(),
            ],
        ))
        .await
        .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
    if result.rows_affected() != 1 {
        return Err(DomainCommandSubmissionError::Unavailable);
    }
    Ok(())
}

fn version_value(version: Option<AggregateVersion>) -> Value {
    version.map(|version| version.to_string()).into()
}

const fn adapter_name(adapter: CommandAdapter) -> &'static str {
    match adapter {
        CommandAdapter::NostrInProcess => "nostr_in_process",
        CommandAdapter::NostrTemporarySidecar => "nostr_temporary_sidecar",
        CommandAdapter::ZedRpc => "zed_rpc",
    }
}

const fn source_system_name(source_system: SourceSystem) -> &'static str {
    match source_system {
        SourceSystem::Zed => "zed",
        SourceSystem::Buzz => "buzz",
        SourceSystem::Nostr => "nostr",
        SourceSystem::Acp => "acp",
        SourceSystem::ExternalGit => "external_git",
    }
}

const fn integrity_algorithm_name(algorithm: IntegrityAlgorithm) -> &'static str {
    match algorithm {
        IntegrityAlgorithm::Sha256 => "sha256",
        IntegrityAlgorithm::NostrEventId => "nostr_event_id",
        IntegrityAlgorithm::GitObjectId => "git_object_id",
    }
}
