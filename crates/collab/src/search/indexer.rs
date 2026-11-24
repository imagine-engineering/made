use collaboration_domain::TenantContext;
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use serde::{Deserialize, Serialize};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SELECT_OUTBOX_SQL: &str = r#"
SELECT outbox.outbox_sequence,
       outbox.topic,
       outbox.source_system,
       outbox.source_record_id,
       outbox.source_version,
       floor(extract(epoch FROM outbox.source_observed_at) * 1000)::bigint
           AS source_observed_at_millis,
       outbox.source_integrity_algorithm,
       outbox.source_integrity_value,
       outbox.payload
FROM public.collaboration_outbox AS outbox
WHERE outbox.community_id = $1
  AND outbox.outbox_sequence = $2
FOR SHARE
"#;
const UPSERT_DOCUMENT_SQL: &str = r#"
INSERT INTO public.collaboration_search_documents (
    community_id,
    source_system,
    source_record_id,
    source_version,
    source_observed_at,
    projection_version,
    document_type,
    visibility_scope,
    title,
    body,
    projected_at
) VALUES (
    $1, $2, $3, $4,
    to_timestamp($5::double precision / 1000), CAST($6 AS numeric),
    $7, $8, $9, $10, clock_timestamp()
)
ON CONFLICT (community_id, source_system, source_record_id)
DO UPDATE SET
    source_version = EXCLUDED.source_version,
    source_observed_at = EXCLUDED.source_observed_at,
    projection_version = EXCLUDED.projection_version,
    document_type = EXCLUDED.document_type,
    visibility_scope = EXCLUDED.visibility_scope,
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    projected_at = EXCLUDED.projected_at
WHERE public.collaboration_search_documents.projection_version
    < EXCLUDED.projection_version
"#;
const UPSERT_CHECKPOINT_SQL: &str = r#"
INSERT INTO public.collaboration_projection_checkpoints (
    community_id,
    projection_name,
    source_system,
    source_record_id,
    source_version,
    source_observed_at,
    source_integrity_algorithm,
    source_integrity_value,
    projection_version,
    reset_generation,
    cursor,
    drift_state,
    authoritative_hash,
    projection_hash,
    projected_at,
    reset_at,
    last_error
) VALUES (
    $1, 'collaboration_search', $2, $3, $4,
    to_timestamp($5::double precision / 1000), $6, $7,
    1, 1, $8, 'clean', NULL, NULL, clock_timestamp(), NULL, NULL
)
ON CONFLICT (community_id, projection_name, source_system, source_record_id)
DO UPDATE SET
    source_version = EXCLUDED.source_version,
    source_observed_at = EXCLUDED.source_observed_at,
    source_integrity_algorithm = EXCLUDED.source_integrity_algorithm,
    source_integrity_value = EXCLUDED.source_integrity_value,
    projection_version = public.collaboration_projection_checkpoints.projection_version + 1,
    cursor = EXCLUDED.cursor,
    drift_state = 'clean',
    authoritative_hash = NULL,
    projection_hash = NULL,
    projected_at = EXCLUDED.projected_at,
    last_error = NULL
WHERE public.collaboration_projection_checkpoints.cursor IS NULL
   OR public.collaboration_projection_checkpoints.cursor < EXCLUDED.cursor
"#;

pub const SEARCH_DOCUMENT_OUTBOX_TOPIC: &str = "collaboration.search.document.v1";
pub const MAX_SEARCH_TITLE_BYTES: usize = 32_768;
pub const MAX_SEARCH_BODY_BYTES: usize = 262_144;
const SEARCH_DOCUMENT_CONTRACT_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchDocumentType {
    Profile,
    Community,
    Channel,
    Project,
    Repository,
    Task,
    Agent,
    Workflow,
    Media,
}

impl SearchDocumentType {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Community => "community",
            Self::Channel => "channel",
            Self::Project => "project",
            Self::Repository => "repository",
            Self::Task => "task",
            Self::Agent => "agent",
            Self::Workflow => "workflow",
            Self::Media => "media",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchExclusionReason {
    Deleted,
    RetentionExpired,
    AuthorizedRestricted,
    DirectMessage,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum SearchProjectionMutation {
    UpsertCommunity { title: String, body: String },
    Exclude { reason: SearchExclusionReason },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SearchProjectionWire {
    contract_version: u16,
    document_type: SearchDocumentType,
    mutation: SearchProjectionMutation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchProjectionOperation(SearchProjectionWire);

impl SearchProjectionOperation {
    pub fn upsert_community(
        document_type: SearchDocumentType,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<Self, SearchIndexerError> {
        let operation = Self(SearchProjectionWire {
            contract_version: SEARCH_DOCUMENT_CONTRACT_VERSION,
            document_type,
            mutation: SearchProjectionMutation::UpsertCommunity {
                title: title.into(),
                body: body.into(),
            },
        });
        operation.validate()?;
        Ok(operation)
    }

    pub const fn exclude(document_type: SearchDocumentType, reason: SearchExclusionReason) -> Self {
        Self(SearchProjectionWire {
            contract_version: SEARCH_DOCUMENT_CONTRACT_VERSION,
            document_type,
            mutation: SearchProjectionMutation::Exclude { reason },
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, SearchIndexerError> {
        serde_json::to_vec(&self.0).map_err(|_| SearchIndexerError::InvalidInput)
    }

    fn decode(payload: &[u8]) -> Result<Self, SearchIndexerError> {
        let operation =
            Self(serde_json::from_slice(payload).map_err(|_| SearchIndexerError::InvalidInput)?);
        operation.validate()?;
        Ok(operation)
    }

    fn validate(&self) -> Result<(), SearchIndexerError> {
        if self.0.contract_version != SEARCH_DOCUMENT_CONTRACT_VERSION {
            return Err(SearchIndexerError::InvalidInput);
        }
        if let SearchProjectionMutation::UpsertCommunity { title, body } = &self.0.mutation
            && (title.len() > MAX_SEARCH_TITLE_BYTES
                || body.len() > MAX_SEARCH_BODY_BYTES
                || title.contains('\0')
                || body.contains('\0'))
        {
            return Err(SearchIndexerError::InvalidInput);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchIndexingOutcome {
    Indexed,
    Excluded(SearchExclusionReason),
    IgnoredTopic,
    IgnoredReplay,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchIndexerError {
    #[error("collaboration search indexing requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("collaboration search projection input is invalid")]
    InvalidInput,
    #[error("the authoritative collaboration outbox record does not exist")]
    OutboxRecordNotFound,
    #[error("collaboration search indexing is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct CollaborationSearchIndexer {
    connection: DatabaseConnection,
}

impl CollaborationSearchIndexer {
    pub fn new(connection: DatabaseConnection) -> Result<Self, SearchIndexerError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(SearchIndexerError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn index_outbox_sequence(
        &self,
        tenant: &TenantContext,
        outbox_sequence: u64,
    ) -> Result<SearchIndexingOutcome, SearchIndexerError> {
        self.index_outbox_sequence_with_retention_requirement(tenant, outbox_sequence, false)
            .await
    }

    pub async fn index_retention_expiry_outbox_sequence(
        &self,
        tenant: &TenantContext,
        outbox_sequence: u64,
    ) -> Result<SearchIndexingOutcome, SearchIndexerError> {
        self.index_outbox_sequence_with_retention_requirement(tenant, outbox_sequence, true)
            .await
    }

    async fn index_outbox_sequence_with_retention_requirement(
        &self,
        tenant: &TenantContext,
        outbox_sequence: u64,
        require_retention_expiry: bool,
    ) -> Result<SearchIndexingOutcome, SearchIndexerError> {
        let outbox_sequence = i64::try_from(outbox_sequence)
            .ok()
            .filter(|sequence| *sequence > 0)
            .ok_or(SearchIndexerError::InvalidInput)?;
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(SearchIndexerError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            let row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_OUTBOX_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        outbox_sequence.into(),
                    ],
                ))
                .await
                .map_err(SearchIndexerError::Unavailable)?
                .ok_or(SearchIndexerError::OutboxRecordNotFound)?;
            let record = SearchOutboxRecord::from_row(row)?;
            if record.topic != SEARCH_DOCUMENT_OUTBOX_TOPIC {
                return Ok(SearchIndexingOutcome::IgnoredTopic);
            }
            let operation = SearchProjectionOperation::decode(&record.payload)?;
            if require_retention_expiry
                && !matches!(
                    &operation.0.mutation,
                    SearchProjectionMutation::Exclude {
                        reason: SearchExclusionReason::RetentionExpired
                    }
                )
            {
                return Err(SearchIndexerError::InvalidInput);
            }
            apply_operation(&transaction, tenant, &record, &operation).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }
}

struct SearchOutboxRecord {
    outbox_sequence: u64,
    topic: String,
    source_system: String,
    source_record_id: String,
    source_version: String,
    source_observed_at_millis: i64,
    source_integrity_algorithm: Option<String>,
    source_integrity_value: Option<String>,
    payload: Vec<u8>,
}

impl SearchOutboxRecord {
    fn from_row(row: QueryResult) -> Result<Self, SearchIndexerError> {
        let outbox_sequence = read_u64(&row, "outbox_sequence")?;
        let source_observed_at_millis = row
            .try_get::<i64>("", "source_observed_at_millis")
            .map_err(|_| SearchIndexerError::InvalidInput)?;
        let record = Self {
            outbox_sequence,
            topic: read_string(&row, "topic")?,
            source_system: read_string(&row, "source_system")?,
            source_record_id: read_string(&row, "source_record_id")?,
            source_version: read_string(&row, "source_version")?,
            source_observed_at_millis,
            source_integrity_algorithm: row
                .try_get("", "source_integrity_algorithm")
                .map_err(|_| SearchIndexerError::InvalidInput)?,
            source_integrity_value: row
                .try_get("", "source_integrity_value")
                .map_err(|_| SearchIndexerError::InvalidInput)?,
            payload: row
                .try_get("", "payload")
                .map_err(|_| SearchIndexerError::InvalidInput)?,
        };
        if !matches!(
            record.source_system.as_str(),
            "zed" | "buzz" | "nostr" | "acp" | "external_git"
        ) || record.source_record_id.is_empty()
            || record.source_record_id.len() > 1024
            || record.source_version.is_empty()
            || record.source_version.len() > 1024
            || record.source_observed_at_millis < 0
            || record.source_integrity_algorithm.is_some()
                != record.source_integrity_value.is_some()
            || record
                .source_integrity_algorithm
                .as_deref()
                .is_some_and(|algorithm| {
                    !matches!(algorithm, "sha256" | "nostr_event_id" | "git_object_id")
                })
            || record
                .source_integrity_value
                .as_ref()
                .is_some_and(|value| value.is_empty() || value.len() > 1024)
        {
            return Err(SearchIndexerError::InvalidInput);
        }
        Ok(record)
    }
}

async fn apply_operation(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    record: &SearchOutboxRecord,
    operation: &SearchProjectionOperation,
) -> Result<SearchIndexingOutcome, SearchIndexerError> {
    let (visibility_scope, title, body, applied_outcome) = match &operation.0.mutation {
        SearchProjectionMutation::UpsertCommunity { title, body } => (
            "community",
            title.as_str(),
            body.as_str(),
            SearchIndexingOutcome::Indexed,
        ),
        SearchProjectionMutation::Exclude { reason } => {
            ("excluded", "", "", SearchIndexingOutcome::Excluded(*reason))
        }
    };
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            UPSERT_DOCUMENT_SQL,
            [
                tenant.community_id().as_uuid().into(),
                record.source_system.as_str().into(),
                record.source_record_id.as_str().into(),
                record.source_version.as_str().into(),
                record.source_observed_at_millis.into(),
                record.outbox_sequence.to_string().into(),
                operation.0.document_type.database_name().into(),
                visibility_scope.into(),
                title.into(),
                body.into(),
            ],
        ))
        .await
        .map_err(SearchIndexerError::Unavailable)?;
    if result.rows_affected() == 0 {
        return Ok(SearchIndexingOutcome::IgnoredReplay);
    }
    if result.rows_affected() != 1 {
        return Err(SearchIndexerError::InvalidInput);
    }
    let checkpoint = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            UPSERT_CHECKPOINT_SQL,
            [
                tenant.community_id().as_uuid().into(),
                record.source_system.as_str().into(),
                record.source_record_id.as_str().into(),
                record.source_version.as_str().into(),
                record.source_observed_at_millis.into(),
                record.source_integrity_algorithm.as_deref().into(),
                record.source_integrity_value.as_deref().into(),
                record.outbox_sequence.to_be_bytes().to_vec().into(),
            ],
        ))
        .await
        .map_err(SearchIndexerError::Unavailable)?;
    if checkpoint.rows_affected() != 1 {
        return Err(SearchIndexerError::InvalidInput);
    }
    Ok(applied_outcome)
}

fn read_string(row: &QueryResult, column: &str) -> Result<String, SearchIndexerError> {
    row.try_get("", column)
        .map_err(|_| SearchIndexerError::InvalidInput)
}

fn read_u64(row: &QueryResult, column: &str) -> Result<u64, SearchIndexerError> {
    row.try_get::<i64>("", column)
        .ok()
        .and_then(|value| u64::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or(SearchIndexerError::InvalidInput)
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), SearchIndexerError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(SearchIndexerError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, SearchIndexerError>,
) -> Result<T, SearchIndexerError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(SearchIndexerError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(SearchIndexerError::Unavailable)?;
            Err(error)
        }
    }
}
