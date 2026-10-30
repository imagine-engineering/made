use collaboration_domain::{CommunityId, TenantContext};
use nostr_compat::{
    CanonicalEvent, EventCodecError, EventId, EventSignature, PublicKey, SignedEvent,
    TimestampPolicy, VerificationError,
    filter::EventFilter,
    head::{
        HeadError, PersistenceClass, ReplacementCoordinate, persistence_class,
        replacement_coordinate,
    },
    verify_signed_event,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait, Value,
};
use serde_json::json;

use super::persistence_policy::{
    EventDurability, EventPersistenceDecision, EventPersistencePolicy, PersistencePolicyError,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const EVENT_COLUMNS: &str = r#"
    e.community_id,
    e.event_id,
    e.author_public_key,
    e.event_created_at::text AS event_created_at_text,
    e.kind,
    e.tags,
    e.content,
    e.canonical_event_bytes,
    e.signature,
    e.signature_state,
    floor(extract(epoch FROM e.verified_at) * 1000)::bigint AS verified_at_millis
"#;
const INSERT_EVENT_SQL: &str = r#"
INSERT INTO public.collaboration_events (
    community_id,
    event_id,
    author_public_key,
    event_created_at,
    kind,
    tags,
    content,
    canonical_event_bytes,
    signature,
    signature_state,
    verified_at,
    persistence_class,
    discriminator
) VALUES (
    $1, $2, $3, CAST($4 AS numeric), $5, $6, $7, $8, $9, $10,
    to_timestamp($11::double precision / 1000), $12, $13
)
ON CONFLICT (community_id, event_id) DO NOTHING
"#;
const UPSERT_HEAD_SQL: &str = r#"
INSERT INTO public.collaboration_event_heads (
    community_id,
    kind,
    author_public_key,
    discriminator,
    head_event_created_at,
    head_event_id,
    live_event_id
) VALUES ($1, $2, $3, $4, CAST($5 AS numeric), $6, $6)
ON CONFLICT (community_id, kind, author_public_key, discriminator) DO UPDATE SET
    head_event_created_at = EXCLUDED.head_event_created_at,
    head_event_id = EXCLUDED.head_event_id,
    live_event_id = EXCLUDED.live_event_id,
    updated_at = clock_timestamp()
WHERE EXCLUDED.head_event_created_at > public.collaboration_event_heads.head_event_created_at
   OR (
        EXCLUDED.head_event_created_at = public.collaboration_event_heads.head_event_created_at
        AND EXCLUDED.head_event_id < public.collaboration_event_heads.head_event_id
   )
"#;
const SELECT_HEAD_WATERMARK_SQL: &str = r#"
SELECT head_event_id, live_event_id
FROM public.collaboration_event_heads
WHERE community_id = $1
  AND kind = $2
  AND author_public_key = $3
  AND discriminator = $4
FOR UPDATE
"#;
const SELECT_EXACT_SQL: &str = r#"
SELECT
"#;
const DELETE_HEAD_SQL: &str = r#"
UPDATE public.collaboration_event_heads
SET live_event_id = NULL, updated_at = clock_timestamp()
WHERE community_id = $1 AND live_event_id = $2
"#;
const DELETE_EVENT_SQL: &str = r#"
DELETE FROM public.collaboration_events
WHERE community_id = $1 AND event_id = $2
"#;

pub const MAX_EVENT_QUERY_FILTERS: usize = 10;
pub const MAX_EVENT_QUERY_RESULTS: u32 = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventVerificationState {
    Live,
    Historical,
}

impl EventVerificationState {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Live => "verified_live",
            Self::Historical => "verified_historical",
        }
    }

    fn from_database(value: &str) -> Result<Self, EventRepositoryError> {
        match value {
            "verified_live" => Ok(Self::Live),
            "verified_historical" => Ok(Self::Historical),
            _ => Err(EventRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedEventRecord {
    community_id: CommunityId,
    signed_event: SignedEvent,
    verification_state: EventVerificationState,
    verified_at_millis: u64,
    canonical_event_bytes: Vec<u8>,
}

impl VerifiedEventRecord {
    pub fn new(
        community_id: CommunityId,
        signed_event: SignedEvent,
        verification_state: EventVerificationState,
        verified_at_millis: u64,
        timestamp_policy: TimestampPolicy,
    ) -> Result<Self, EventRepositoryError> {
        if !matches!(
            (verification_state, timestamp_policy),
            (
                EventVerificationState::Historical,
                TimestampPolicy::Historical
            ) | (
                EventVerificationState::Live,
                TimestampPolicy::Bounded { .. }
            )
        ) {
            return Err(EventRepositoryError::InvalidRecord);
        }
        verify_signed_event(&signed_event, timestamp_policy)?;
        let canonical_event_bytes = signed_event.event.canonical_bytes()?;
        Ok(Self {
            community_id,
            signed_event,
            verification_state,
            verified_at_millis,
            canonical_event_bytes,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn signed_event(&self) -> &SignedEvent {
        &self.signed_event
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredEventRecord {
    community_id: CommunityId,
    signed_event: SignedEvent,
    verification_state: EventVerificationState,
    verified_at_millis: u64,
}

impl StoredEventRecord {
    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn signed_event(&self) -> &SignedEvent {
        &self.signed_event
    }

    pub const fn verification_state(&self) -> EventVerificationState {
        self.verification_state
    }

    pub const fn verified_at_millis(&self) -> u64 {
        self.verified_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventStoreOutcome {
    Inserted,
    Duplicate,
    Stale,
    EphemeralNotPersisted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventRepositoryQuery {
    filters: Vec<EventFilter>,
    limit: u32,
}

impl EventRepositoryQuery {
    pub fn new(filters: Vec<EventFilter>, limit: u32) -> Result<Self, EventRepositoryError> {
        if filters.len() > MAX_EVENT_QUERY_FILTERS
            || limit == 0
            || limit > MAX_EVENT_QUERY_RESULTS
            || filters.iter().any(|filter| filter.validate().is_err())
        {
            return Err(EventRepositoryError::InvalidQuery);
        }
        Ok(Self { filters, limit })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EventRepositoryError {
    #[error("event repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("event repository request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("event repository query is invalid or exceeds a bound")]
    InvalidQuery,
    #[error("event repository record is invalid")]
    InvalidRecord,
    #[error(transparent)]
    Codec(#[from] EventCodecError),
    #[error(transparent)]
    Verification(#[from] VerificationError),
    #[error(transparent)]
    Head(#[from] HeadError),
    #[error(transparent)]
    PersistencePolicy(#[from] PersistencePolicyError),
    #[error("event repository is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct EventRepository {
    connection: DatabaseConnection,
}

impl EventRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, EventRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(EventRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn store(
        &self,
        tenant: &TenantContext,
        record: &VerifiedEventRecord,
        persistence_decision: EventPersistenceDecision,
    ) -> Result<EventStoreOutcome, EventRepositoryError> {
        if tenant.community_id() != record.community_id {
            return Err(EventRepositoryError::TenantBoundaryViolation);
        }
        let persistence_decision = EventPersistencePolicy::validate_for_event(
            record.signed_event.event.kind,
            persistence_decision,
        )?;
        if persistence_decision.durability() == EventDurability::TransientOnly {
            return Ok(EventStoreOutcome::EphemeralNotPersisted);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            self.store_in_transaction(&transaction, tenant, record, persistence_decision)
                .await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn store_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        record: &VerifiedEventRecord,
        persistence_decision: EventPersistenceDecision,
    ) -> Result<EventStoreOutcome, EventRepositoryError> {
        if tenant.community_id() != record.community_id {
            return Err(EventRepositoryError::TenantBoundaryViolation);
        }
        let persistence_decision = EventPersistencePolicy::validate_for_event(
            record.signed_event.event.kind,
            persistence_decision,
        )?;
        if persistence_decision.durability() == EventDurability::TransientOnly {
            return Ok(EventStoreOutcome::EphemeralNotPersisted);
        }
        if let Some(coordinate) = event_coordinate(&record.signed_event.event)? {
            let head_result = transaction
                .execute(head_statement(tenant.community_id(), record, &coordinate))
                .await
                .map_err(EventRepositoryError::Unavailable)?;
            if head_result.rows_affected() == 0
                && !same_live_head(transaction, tenant.community_id(), record, &coordinate).await?
            {
                return Ok(EventStoreOutcome::Stale);
            }
        }
        let result = transaction
            .execute(insert_statement(record)?)
            .await
            .map_err(EventRepositoryError::Unavailable)?;
        Ok(if result.rows_affected() == 0 {
            EventStoreOutcome::Duplicate
        } else {
            EventStoreOutcome::Inserted
        })
    }

    pub async fn exact(
        &self,
        tenant: &TenantContext,
        event_id: EventId,
    ) -> Result<Option<StoredEventRecord>, EventRepositoryError> {
        let transaction = self.begin().await?;
        let result = self
            .exact_in_transaction(&transaction, tenant, event_id)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn exact_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        event_id: EventId,
    ) -> Result<Option<StoredEventRecord>, EventRepositoryError> {
        set_tenant(transaction, tenant.community_id()).await?;
        let sql = format!(
            "{SELECT_EXACT_SQL}{EVENT_COLUMNS} FROM public.collaboration_events e WHERE e.community_id = $1 AND e.event_id = $2"
        );
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                sql,
                [
                    tenant.community_id().as_uuid().into(),
                    event_id.as_bytes().to_vec().into(),
                ],
            ))
            .await
            .map_err(EventRepositoryError::Unavailable)?;
        row.map(|row| stored_event_from_row(row, tenant.community_id()))
            .transpose()
    }

    pub async fn query(
        &self,
        tenant: &TenantContext,
        query: &EventRepositoryQuery,
    ) -> Result<Vec<StoredEventRecord>, EventRepositoryError> {
        if query.filters.is_empty() {
            return Ok(Vec::new());
        }
        let transaction = self.begin().await?;
        let result = self.query_in_transaction(&transaction, tenant, query).await;
        finish_transaction(transaction, result).await
    }

    pub async fn query_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        query: &EventRepositoryQuery,
    ) -> Result<Vec<StoredEventRecord>, EventRepositoryError> {
        set_tenant(transaction, tenant.community_id()).await?;
        if query.filters.is_empty() {
            return Ok(Vec::new());
        }
        let rows = transaction
            .query_all(query_statement(tenant.community_id(), query))
            .await
            .map_err(EventRepositoryError::Unavailable)?;
        rows.into_iter()
            .map(|row| stored_event_from_row(row, tenant.community_id()))
            .collect()
    }

    pub async fn head(
        &self,
        tenant: &TenantContext,
        coordinate: &ReplacementCoordinate,
    ) -> Result<Option<StoredEventRecord>, EventRepositoryError> {
        validate_coordinate(coordinate)?;
        let transaction = self.begin().await?;
        let result = self
            .head_in_transaction(&transaction, tenant, coordinate)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn head_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        coordinate: &ReplacementCoordinate,
    ) -> Result<Option<StoredEventRecord>, EventRepositoryError> {
        validate_coordinate(coordinate)?;
        set_tenant(transaction, tenant.community_id()).await?;
        let sql = format!(
            "SELECT {EVENT_COLUMNS} FROM public.collaboration_event_heads h JOIN public.collaboration_events e ON e.community_id = h.community_id AND e.event_id = h.live_event_id WHERE h.community_id = $1 AND h.kind = $2 AND h.author_public_key = $3 AND h.discriminator = $4"
        );
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                sql,
                [
                    tenant.community_id().as_uuid().into(),
                    i32::from(coordinate.kind).into(),
                    coordinate.author.as_bytes().to_vec().into(),
                    coordinate.discriminator.clone().unwrap_or_default().into(),
                ],
            ))
            .await
            .map_err(EventRepositoryError::Unavailable)?;
        row.map(|row| stored_event_from_row(row, tenant.community_id()))
            .transpose()
    }

    pub async fn delete(
        &self,
        tenant: &TenantContext,
        event_id: EventId,
    ) -> Result<bool, EventRepositoryError> {
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    DELETE_HEAD_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        event_id.as_bytes().to_vec().into(),
                    ],
                ))
                .await
                .map_err(EventRepositoryError::Unavailable)?;
            let deleted = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    DELETE_EVENT_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        event_id.as_bytes().to_vec().into(),
                    ],
                ))
                .await
                .map_err(EventRepositoryError::Unavailable)?;
            Ok(deleted.rows_affected() == 1)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, EventRepositoryError> {
        self.connection
            .begin()
            .await
            .map_err(EventRepositoryError::Unavailable)
    }
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, EventRepositoryError>,
) -> Result<T, EventRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(EventRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(EventRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), EventRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(EventRepositoryError::Unavailable)?;
    Ok(())
}

fn event_coordinate(
    event: &CanonicalEvent,
) -> Result<Option<ReplacementCoordinate>, EventRepositoryError> {
    match persistence_class(event.kind) {
        PersistenceClass::Replaceable | PersistenceClass::ParameterizedReplaceable => {
            Ok(Some(replacement_coordinate(event)?))
        }
        PersistenceClass::Regular | PersistenceClass::Ephemeral => Ok(None),
    }
}

fn validate_coordinate(coordinate: &ReplacementCoordinate) -> Result<(), EventRepositoryError> {
    match (
        persistence_class(coordinate.kind),
        &coordinate.discriminator,
    ) {
        (PersistenceClass::Replaceable, None) => Ok(()),
        (PersistenceClass::ParameterizedReplaceable, Some(value)) if value.len() <= 1024 => Ok(()),
        _ => Err(EventRepositoryError::InvalidQuery),
    }
}

fn head_statement(
    community_id: CommunityId,
    record: &VerifiedEventRecord,
    coordinate: &ReplacementCoordinate,
) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPSERT_HEAD_SQL,
        [
            community_id.as_uuid().into(),
            i32::from(coordinate.kind).into(),
            coordinate.author.as_bytes().to_vec().into(),
            coordinate.discriminator.clone().unwrap_or_default().into(),
            record.signed_event.event.created_at.to_string().into(),
            record.signed_event.claimed_id.as_bytes().to_vec().into(),
        ],
    )
}

async fn same_live_head(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    record: &VerifiedEventRecord,
    coordinate: &ReplacementCoordinate,
) -> Result<bool, EventRepositoryError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_HEAD_WATERMARK_SQL,
            [
                community_id.as_uuid().into(),
                i32::from(coordinate.kind).into(),
                coordinate.author.as_bytes().to_vec().into(),
                coordinate.discriminator.clone().unwrap_or_default().into(),
            ],
        ))
        .await
        .map_err(EventRepositoryError::Unavailable)?;
    let Some(row) = row else {
        return Err(EventRepositoryError::InvalidRecord);
    };
    let head_event_id: Vec<u8> = row_value(&row, "head_event_id")?;
    let live_event_id: Option<Vec<u8>> = row_value(&row, "live_event_id")?;
    Ok(head_event_id == record.signed_event.claimed_id.as_bytes()
        && live_event_id.as_deref() == Some(record.signed_event.claimed_id.as_bytes()))
}

fn insert_statement(record: &VerifiedEventRecord) -> Result<Statement, EventRepositoryError> {
    let event = &record.signed_event.event;
    let (persistence_name, discriminator): (&str, Option<String>) =
        match persistence_class(event.kind) {
            PersistenceClass::Regular => ("regular", None),
            PersistenceClass::Replaceable => ("replaceable", Some(String::new())),
            PersistenceClass::ParameterizedReplaceable => (
                "parameterized_replaceable",
                replacement_coordinate(event)?.discriminator,
            ),
            PersistenceClass::Ephemeral => return Err(EventRepositoryError::InvalidRecord),
        };
    let verified_at_millis = i64::try_from(record.verified_at_millis)
        .map_err(|_| EventRepositoryError::InvalidRecord)?;
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_EVENT_SQL,
        [
            record.community_id.as_uuid().into(),
            record.signed_event.claimed_id.as_bytes().to_vec().into(),
            event.public_key.as_bytes().to_vec().into(),
            event.created_at.to_string().into(),
            i32::from(event.kind).into(),
            serde_json::to_value(&event.tags)
                .map_err(|_| EventRepositoryError::InvalidRecord)?
                .into(),
            event.content.clone().into(),
            record.canonical_event_bytes.clone().into(),
            record.signed_event.signature.as_bytes().to_vec().into(),
            record.verification_state.database_name().to_owned().into(),
            verified_at_millis.into(),
            persistence_name.to_owned().into(),
            discriminator.into(),
        ],
    ))
}

fn query_statement(community_id: CommunityId, query: &EventRepositoryQuery) -> Statement {
    let mut sql = format!(
        "SELECT {EVENT_COLUMNS} FROM public.collaboration_events e WHERE e.community_id = $1 AND (e.persistence_class = 'regular' OR EXISTS (SELECT 1 FROM public.collaboration_event_heads h WHERE h.community_id = e.community_id AND h.live_event_id = e.event_id)) AND ("
    );
    let mut values = vec![community_id.as_uuid().into()];
    for (filter_index, filter) in query.filters.iter().enumerate() {
        if filter_index > 0 {
            sql.push_str(" OR ");
        }
        sql.push('(');
        append_filter(&mut sql, &mut values, filter);
        sql.push(')');
    }
    sql.push_str(" ) ORDER BY e.event_created_at DESC, e.event_id ASC LIMIT ");
    sql.push_str(&bind_value(&mut values, i64::from(query.limit).into()));
    Statement::from_sql_and_values(DatabaseBackend::Postgres, sql, values)
}

fn append_filter(sql: &mut String, values: &mut Vec<Value>, filter: &EventFilter) {
    let mut clauses = Vec::new();
    if !filter.ids.is_empty() {
        let matches = filter
            .ids
            .iter()
            .map(|prefix| {
                let parameter = bind_value(values, format!("{}%", prefix.as_str()).into());
                format!("encode(e.event_id, 'hex') LIKE {parameter}")
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        clauses.push(format!("({matches})"));
    }
    if !filter.authors.is_empty() {
        let parameters = filter
            .authors
            .iter()
            .map(|author| bind_value(values, author.as_bytes().to_vec().into()))
            .collect::<Vec<_>>()
            .join(", ");
        clauses.push(format!("e.author_public_key IN ({parameters})"));
    }
    if !filter.kinds.is_empty() {
        let parameters = filter
            .kinds
            .iter()
            .map(|kind| bind_value(values, i32::from(*kind).into()))
            .collect::<Vec<_>>()
            .join(", ");
        clauses.push(format!("e.kind IN ({parameters})"));
    }
    if let Some(since) = filter.since {
        let parameter = bind_value(values, since.to_string().into());
        clauses.push(format!(
            "e.event_created_at >= CAST({parameter} AS numeric)"
        ));
    }
    if let Some(until) = filter.until {
        let parameter = bind_value(values, until.to_string().into());
        clauses.push(format!(
            "e.event_created_at <= CAST({parameter} AS numeric)"
        ));
    }
    for (tag, tag_values) in &filter.generic_tags {
        if tag_values.is_empty() {
            clauses.push("FALSE".to_owned());
            continue;
        }
        let matches = tag_values
            .iter()
            .map(|value| {
                let parameter = bind_value(values, json!([[tag.to_string(), value]]).into());
                format!("e.tags @> CAST({parameter} AS jsonb)")
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        clauses.push(format!("({matches})"));
    }
    if clauses.is_empty() {
        sql.push_str("TRUE");
    } else {
        sql.push_str(&clauses.join(" AND "));
    }
}

fn bind_value(values: &mut Vec<Value>, value: Value) -> String {
    values.push(value);
    format!("${}", values.len())
}

fn stored_event_from_row(
    row: QueryResult,
    expected_community: CommunityId,
) -> Result<StoredEventRecord, EventRepositoryError> {
    let community_id = CommunityId::from_uuid(row_value(&row, "community_id")?);
    if community_id != expected_community {
        return Err(EventRepositoryError::TenantBoundaryViolation);
    }
    let event_id = fixed_bytes::<32>(row_value(&row, "event_id")?)?;
    let public_key = fixed_bytes::<32>(row_value(&row, "author_public_key")?)?;
    let created_at = row_value::<String>(&row, "event_created_at_text")?
        .parse::<u64>()
        .map_err(|_| EventRepositoryError::InvalidRecord)?;
    let kind = u16::try_from(row_value::<i32>(&row, "kind")?)
        .map_err(|_| EventRepositoryError::InvalidRecord)?;
    let tags = serde_json::from_value(row_value(&row, "tags")?)
        .map_err(|_| EventRepositoryError::InvalidRecord)?;
    let content = row_value(&row, "content")?;
    let canonical_event_bytes: Vec<u8> = row_value(&row, "canonical_event_bytes")?;
    let signature = fixed_bytes::<64>(row_value(&row, "signature")?)?;
    let verification_state =
        EventVerificationState::from_database(&row_value::<String>(&row, "signature_state")?)?;
    let verified_at_millis = u64::try_from(row_value::<i64>(&row, "verified_at_millis")?)
        .map_err(|_| EventRepositoryError::InvalidRecord)?;
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key),
        created_at,
        kind,
        tags,
        content,
    );
    if event.canonical_bytes()? != canonical_event_bytes
        || event.event_id()? != EventId::from_bytes(event_id)
    {
        return Err(EventRepositoryError::InvalidRecord);
    }
    Ok(StoredEventRecord {
        community_id,
        signed_event: SignedEvent {
            claimed_id: EventId::from_bytes(event_id),
            event,
            signature: EventSignature::from_hex(&hex::encode(signature))?,
        },
        verification_state,
        verified_at_millis,
    })
}

fn fixed_bytes<const LENGTH: usize>(value: Vec<u8>) -> Result<[u8; LENGTH], EventRepositoryError> {
    value
        .try_into()
        .map_err(|_| EventRepositoryError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, EventRepositoryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| EventRepositoryError::InvalidRecord)
}
