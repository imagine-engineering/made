use async_trait::async_trait;
use collaboration_domain::{
    IntegrityAlgorithm, IntegrityReference, OperationId, Provenance, SourceRecordId, SourceSystem,
    TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DbBackend, Statement, TransactionTrait,
};
use sha2::{Digest as _, Sha256};

use super::{
    envelope::FanoutEnvelope,
    subscription_bus::{FanoutReplayStore, MAX_REPLAY_BATCH, SubscriptionBusError},
};

pub struct PostgresFanoutReplayStore {
    connection: DatabaseConnection,
}

impl PostgresFanoutReplayStore {
    pub fn new(connection: DatabaseConnection) -> Result<Self, SubscriptionBusError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(SubscriptionBusError::Unavailable);
        }
        Ok(Self { connection })
    }

    pub async fn payload(
        &self,
        tenant: &TenantContext,
        sequence: u64,
    ) -> Result<Vec<u8>, SubscriptionBusError> {
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        set_tenant(&transaction, tenant).await?;
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT payload FROM public.collaboration_outbox WHERE community_id = $1 AND outbox_sequence = $2",
                [tenant.community_id().as_uuid().into(), i64::try_from(sequence).map_err(|_| SubscriptionBusError::InvalidRequest)?.into()],
            ))
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?
            .ok_or(SubscriptionBusError::InvalidRequest)?;
        let payload = row
            .try_get("", "payload")
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        transaction
            .commit()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        Ok(payload)
    }

    pub async fn envelope_for_operation(
        &self,
        tenant: &TenantContext,
        operation_id: OperationId,
    ) -> Result<FanoutEnvelope, SubscriptionBusError> {
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        set_tenant(&transaction, tenant).await?;
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                format!("{ENVELOPE_COLUMNS} WHERE community_id = $1 AND operation_id = $2"),
                [
                    tenant.community_id().as_uuid().into(),
                    operation_id.as_uuid().into(),
                ],
            ))
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?
            .ok_or(SubscriptionBusError::InvalidRequest)?;
        let envelope = envelope_from_row(tenant, row, None)?;
        transaction
            .commit()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        Ok(envelope)
    }

    /// Redis hints must match authority before changing local replay cursors or deduplication state.
    pub async fn validate_notification(
        &self,
        tenant: &TenantContext,
        notification: &FanoutEnvelope,
    ) -> Result<(), SubscriptionBusError> {
        if notification.community_id() != tenant.community_id() {
            return Err(SubscriptionBusError::TenantMismatch);
        }
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        set_tenant(&transaction, tenant).await?;
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                format!("{ENVELOPE_COLUMNS} WHERE community_id = $1 AND outbox_sequence = $2"),
                [
                    tenant.community_id().as_uuid().into(),
                    i64::try_from(notification.outbox_sequence())
                        .map_err(|_| SubscriptionBusError::InvalidRequest)?
                        .into(),
                ],
            ))
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?
            .ok_or(SubscriptionBusError::InvalidRequest)?;
        if envelope_from_row(tenant, row, None)? != *notification {
            return Err(SubscriptionBusError::InvalidRequest);
        }
        transaction
            .commit()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)
    }
}

const ENVELOPE_COLUMNS: &str = r#"
SELECT
    outbox_sequence,
    topic,
    source_system,
    source_record_id,
    source_version,
    floor(extract(epoch FROM source_observed_at) * 1000)::bigint AS observed_at_millis,
    source_integrity_algorithm,
    source_integrity_value,
    payload
FROM public.collaboration_outbox
"#;

#[async_trait]
impl FanoutReplayStore for PostgresFanoutReplayStore {
    async fn load_after(
        &self,
        tenant: &TenantContext,
        topic: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<FanoutEnvelope>, SubscriptionBusError> {
        if limit == 0 || limit > MAX_REPLAY_BATCH {
            return Err(SubscriptionBusError::InvalidRequest);
        }
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        set_tenant(&transaction, tenant).await?;
        let rows = transaction
            .query_all(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                format!(
                    r#"{ENVELOPE_COLUMNS}
WHERE community_id = $1 AND topic = $2 AND outbox_sequence > $3
ORDER BY outbox_sequence ASC
LIMIT $4
"#
                ),
                [
                    tenant.community_id().as_uuid().into(),
                    topic.into(),
                    i64::try_from(after_sequence)
                        .map_err(|_| SubscriptionBusError::InvalidRequest)?
                        .into(),
                    i64::try_from(limit)
                        .map_err(|_| SubscriptionBusError::InvalidRequest)?
                        .into(),
                ],
            ))
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        let envelopes = rows
            .into_iter()
            .map(|row| envelope_from_row(tenant, row, Some(topic)))
            .collect::<Result<Vec<_>, _>>()?;
        transaction
            .commit()
            .await
            .map_err(|_| SubscriptionBusError::Unavailable)?;
        Ok(envelopes)
    }
}

fn envelope_from_row(
    tenant: &TenantContext,
    row: sea_orm::QueryResult,
    expected_topic: Option<&str>,
) -> Result<FanoutEnvelope, SubscriptionBusError> {
    let topic = row
        .try_get::<String>("", "topic")
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    if expected_topic.is_some_and(|expected| expected != topic) {
        return Err(SubscriptionBusError::Unavailable);
    }
    let source_system = match row
        .try_get::<String>("", "source_system")
        .map_err(|_| SubscriptionBusError::Unavailable)?
        .as_str()
    {
        "zed" => SourceSystem::Zed,
        "buzz" => SourceSystem::Buzz,
        "nostr" => SourceSystem::Nostr,
        "acp" => SourceSystem::Acp,
        "external_git" => SourceSystem::ExternalGit,
        _ => return Err(SubscriptionBusError::Unavailable),
    };
    let source_record_id = SourceRecordId::new(
        row.try_get::<String>("", "source_record_id")
            .map_err(|_| SubscriptionBusError::Unavailable)?,
    )
    .ok_or(SubscriptionBusError::Unavailable)?;
    let source_version = row
        .try_get::<Option<String>>("", "source_version")
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    let integrity = match (
        row.try_get::<Option<String>>("", "source_integrity_algorithm")
            .map_err(|_| SubscriptionBusError::Unavailable)?,
        row.try_get::<Option<String>>("", "source_integrity_value")
            .map_err(|_| SubscriptionBusError::Unavailable)?,
    ) {
        (None, None) => None,
        (Some(algorithm), Some(value)) => Some(IntegrityReference {
            algorithm: match algorithm.as_str() {
                "sha256" => IntegrityAlgorithm::Sha256,
                "nostr_event_id" => IntegrityAlgorithm::NostrEventId,
                "git_object_id" => IntegrityAlgorithm::GitObjectId,
                _ => return Err(SubscriptionBusError::Unavailable),
            },
            value,
        }),
        _ => return Err(SubscriptionBusError::Unavailable),
    };
    let observed_at_millis = u64::try_from(
        row.try_get::<i64>("", "observed_at_millis")
            .map_err(|_| SubscriptionBusError::Unavailable)?,
    )
    .map_err(|_| SubscriptionBusError::Unavailable)?;
    let payload: Vec<u8> = row
        .try_get("", "payload")
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    let provenance = Provenance {
        source_system,
        source_record_id,
        source_version,
        observed_at_millis,
        integrity,
    };
    FanoutEnvelope::new(
        tenant.community_id(),
        u64::try_from(
            row.try_get::<i64>("", "outbox_sequence")
                .map_err(|_| SubscriptionBusError::Unavailable)?,
        )
        .map_err(|_| SubscriptionBusError::Unavailable)?,
        topic,
        provenance,
        Sha256::digest(&payload).into(),
    )
    .map_err(SubscriptionBusError::Envelope)
}

async fn set_tenant(
    transaction: &sea_orm::DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), SubscriptionBusError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)?;
    Ok(())
}
