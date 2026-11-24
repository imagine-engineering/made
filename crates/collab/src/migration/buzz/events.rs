use std::collections::HashSet;

use collaboration_domain::{CommunityId, TenantContext};
use nostr_compat::{
    EventCodecError, SignedEvent, TimestampPolicy, VerificationError,
    head::{
        HeadError, PersistenceClass, ReplacementCoordinate, persistence_class,
        replacement_coordinate,
    },
    verify_signed_event,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use sha2::{Digest, Sha256};

const MAX_IMPORT_BATCH_SIZE: usize = 1_000;
const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const INSERT_EVENT_SQL: &str = r#"
INSERT INTO public.collaboration_events (
    community_id, event_id, author_public_key, event_created_at, kind, tags, content,
    canonical_event_bytes, signature, signature_state, verified_at, persistence_class,
    discriminator
) VALUES (
    $1, $2, $3, CAST($4 AS numeric), $5, $6, $7, $8, $9,
    'verified_historical', to_timestamp($10::double precision / 1000), $11, $12
)
ON CONFLICT (community_id, event_id) DO NOTHING
"#;
const SELECT_EVENT_SQL: &str = r#"
SELECT event_id, author_public_key, event_created_at::text AS event_created_at_text,
       kind, tags, content, canonical_event_bytes, signature
FROM public.collaboration_events
WHERE community_id = $1 AND event_id = $2
"#;
const UPSERT_HEAD_SQL: &str = r#"
INSERT INTO public.collaboration_event_heads (
    community_id, kind, author_public_key, discriminator,
    head_event_created_at, head_event_id, live_event_id
) VALUES ($1, $2, $3, $4, CAST($5 AS numeric), $6, $7)
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzEventSourceRecord {
    community_id: CommunityId,
    source_sequence: u64,
    signed_event: SignedEvent,
    received_at_millis: u64,
    deleted_at_millis: Option<u64>,
    canonical_event_bytes: Vec<u8>,
    coordinate: Option<ReplacementCoordinate>,
}

impl BuzzEventSourceRecord {
    pub fn new(
        community_id: CommunityId,
        source_sequence: u64,
        signed_event: SignedEvent,
        received_at_millis: u64,
        deleted_at_millis: Option<u64>,
    ) -> Result<Self, BuzzEventImportError> {
        if source_sequence == 0
            || i64::try_from(received_at_millis).is_err()
            || deleted_at_millis.is_some_and(|value| i64::try_from(value).is_err())
        {
            return Err(BuzzEventImportError::InvalidSourceRecord);
        }
        verify_signed_event(&signed_event, TimestampPolicy::Historical)?;
        let canonical_event_bytes = signed_event.event.canonical_bytes()?;
        let coordinate = match persistence_class(signed_event.event.kind) {
            PersistenceClass::Regular => None,
            PersistenceClass::Replaceable | PersistenceClass::ParameterizedReplaceable => {
                Some(replacement_coordinate(&signed_event.event)?)
            }
            PersistenceClass::Ephemeral => {
                return Err(BuzzEventImportError::EphemeralSourceRecord);
            }
        };
        Ok(Self {
            community_id,
            source_sequence,
            signed_event,
            received_at_millis,
            deleted_at_millis,
            canonical_event_bytes,
            coordinate,
        })
    }

    pub const fn source_sequence(&self) -> u64 {
        self.source_sequence
    }

    pub const fn signed_event(&self) -> &SignedEvent {
        &self.signed_event
    }

    pub const fn is_deleted(&self) -> bool {
        self.deleted_at_millis.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzEventImportResult {
    pub scanned: u64,
    pub inserted: u64,
    pub duplicates: u64,
    pub addressable_coordinates: u64,
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub target_hash: [u8; 32],
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzEventImportError {
    #[error("Buzz event import requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("Buzz event import input is invalid")]
    InvalidSourceRecord,
    #[error("Buzz never persisted ephemeral events")]
    EphemeralSourceRecord,
    #[error("Buzz event import crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("Buzz event import batch is empty, oversized or out of order")]
    InvalidBatch,
    #[error("an existing canonical event differs from the Buzz signed event")]
    IntegrityConflict,
    #[error(transparent)]
    Codec(#[from] EventCodecError),
    #[error(transparent)]
    Verification(#[from] VerificationError),
    #[error(transparent)]
    Head(#[from] HeadError),
    #[error("Buzz event import storage is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct BuzzEventImporter {
    connection: DatabaseConnection,
}

impl BuzzEventImporter {
    pub fn new(connection: DatabaseConnection) -> Result<Self, BuzzEventImportError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(BuzzEventImportError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn import_batch(
        &self,
        tenant: &TenantContext,
        records: &[BuzzEventSourceRecord],
    ) -> Result<BuzzEventImportResult, BuzzEventImportError> {
        validate_batch(tenant, records)?;
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(BuzzEventImportError::Unavailable)?;
        let result = self
            .import_in_transaction(&transaction, tenant, records)
            .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    async fn import_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        records: &[BuzzEventSourceRecord],
    ) -> Result<BuzzEventImportResult, BuzzEventImportError> {
        set_tenant(transaction, tenant.community_id()).await?;
        let mut inserted = 0_u64;
        let mut duplicates = 0_u64;
        let mut source_hasher = Sha256::new();
        let mut target_hasher = Sha256::new();
        let mut coordinates = HashSet::new();

        for record in records {
            hash_signed_record(&mut source_hasher, record);
            let result = transaction
                .execute(insert_statement(record)?)
                .await
                .map_err(BuzzEventImportError::Unavailable)?;
            if result.rows_affected() == 1 {
                inserted += 1;
            } else {
                duplicates += 1;
            }
            let stored = read_stored_event(transaction, record).await?;
            hash_stored_record(&mut target_hasher, record.source_sequence, &stored);

            if let Some(coordinate) = &record.coordinate {
                coordinates.insert(coordinate.clone());
                transaction
                    .execute(head_statement(record, coordinate))
                    .await
                    .map_err(BuzzEventImportError::Unavailable)?;
            }
        }

        let scanned =
            u64::try_from(records.len()).map_err(|_| BuzzEventImportError::InvalidBatch)?;
        let addressable_coordinates =
            u64::try_from(coordinates.len()).map_err(|_| BuzzEventImportError::InvalidBatch)?;
        let final_source_sequence = records
            .last()
            .map(BuzzEventSourceRecord::source_sequence)
            .ok_or(BuzzEventImportError::InvalidBatch)?;
        Ok(BuzzEventImportResult {
            scanned,
            inserted,
            duplicates,
            addressable_coordinates,
            final_source_sequence,
            source_hash: source_hasher.finalize().into(),
            target_hash: target_hasher.finalize().into(),
        })
    }
}

fn validate_batch(
    tenant: &TenantContext,
    records: &[BuzzEventSourceRecord],
) -> Result<(), BuzzEventImportError> {
    if records.is_empty() || records.len() > MAX_IMPORT_BATCH_SIZE {
        return Err(BuzzEventImportError::InvalidBatch);
    }
    let mut previous_sequence = None;
    for record in records {
        if record.community_id != tenant.community_id() {
            return Err(BuzzEventImportError::TenantBoundaryViolation);
        }
        if previous_sequence.is_some_and(|previous| record.source_sequence <= previous) {
            return Err(BuzzEventImportError::InvalidBatch);
        }
        previous_sequence = Some(record.source_sequence);
    }
    Ok(())
}

fn insert_statement(record: &BuzzEventSourceRecord) -> Result<Statement, BuzzEventImportError> {
    let event = &record.signed_event.event;
    let (persistence_name, discriminator): (&str, Option<String>) = match &record.coordinate {
        None => ("regular", None),
        Some(coordinate) if coordinate.discriminator.is_none() => {
            ("replaceable", Some(String::new()))
        }
        Some(coordinate) => (
            "parameterized_replaceable",
            coordinate.discriminator.clone(),
        ),
    };
    let verified_at = i64::try_from(record.received_at_millis)
        .map_err(|_| BuzzEventImportError::InvalidSourceRecord)?;
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
                .map_err(|_| BuzzEventImportError::InvalidSourceRecord)?
                .into(),
            event.content.clone().into(),
            record.canonical_event_bytes.clone().into(),
            record.signed_event.signature.as_bytes().to_vec().into(),
            verified_at.into(),
            persistence_name.to_owned().into(),
            discriminator.into(),
        ],
    ))
}

fn head_statement(record: &BuzzEventSourceRecord, coordinate: &ReplacementCoordinate) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPSERT_HEAD_SQL,
        [
            record.community_id.as_uuid().into(),
            i32::from(coordinate.kind).into(),
            coordinate.author.as_bytes().to_vec().into(),
            coordinate.discriminator.clone().unwrap_or_default().into(),
            record.signed_event.event.created_at.to_string().into(),
            record.signed_event.claimed_id.as_bytes().to_vec().into(),
            (!record.is_deleted())
                .then(|| record.signed_event.claimed_id.as_bytes().to_vec())
                .into(),
        ],
    )
}

struct StoredSignedEvent {
    event_id: Vec<u8>,
    public_key: Vec<u8>,
    created_at: String,
    kind: i32,
    tags: serde_json::Value,
    content: String,
    canonical_event_bytes: Vec<u8>,
    signature: Vec<u8>,
}

async fn read_stored_event(
    transaction: &DatabaseTransaction,
    expected: &BuzzEventSourceRecord,
) -> Result<StoredSignedEvent, BuzzEventImportError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_EVENT_SQL,
            [
                expected.community_id.as_uuid().into(),
                expected.signed_event.claimed_id.as_bytes().to_vec().into(),
            ],
        ))
        .await
        .map_err(BuzzEventImportError::Unavailable)?
        .ok_or(BuzzEventImportError::IntegrityConflict)?;
    let stored = StoredSignedEvent {
        event_id: column(&row, "event_id")?,
        public_key: column(&row, "author_public_key")?,
        created_at: column(&row, "event_created_at_text")?,
        kind: column(&row, "kind")?,
        tags: column(&row, "tags")?,
        content: column(&row, "content")?,
        canonical_event_bytes: column(&row, "canonical_event_bytes")?,
        signature: column(&row, "signature")?,
    };
    let event = &expected.signed_event.event;
    if stored.event_id != expected.signed_event.claimed_id.as_bytes()
        || stored.public_key != event.public_key.as_bytes()
        || stored.created_at != event.created_at.to_string()
        || stored.kind != i32::from(event.kind)
        || stored.tags
            != serde_json::to_value(&event.tags)
                .map_err(|_| BuzzEventImportError::InvalidSourceRecord)?
        || stored.content != event.content
        || stored.canonical_event_bytes != expected.canonical_event_bytes
        || stored.signature != expected.signed_event.signature.as_bytes()
    {
        return Err(BuzzEventImportError::IntegrityConflict);
    }
    Ok(stored)
}

fn column<T: sea_orm::TryGetable>(
    row: &QueryResult,
    name: &str,
) -> Result<T, BuzzEventImportError> {
    row.try_get("", name)
        .map_err(|_| BuzzEventImportError::IntegrityConflict)
}

fn hash_signed_record(hasher: &mut Sha256, record: &BuzzEventSourceRecord) {
    hash_part(hasher, &record.source_sequence.to_be_bytes());
    hash_part(hasher, record.signed_event.claimed_id.as_bytes());
    hash_part(hasher, &record.canonical_event_bytes);
    hash_part(hasher, record.signed_event.signature.as_bytes());
}

fn hash_stored_record(hasher: &mut Sha256, sequence: u64, record: &StoredSignedEvent) {
    hash_part(hasher, &sequence.to_be_bytes());
    hash_part(hasher, &record.event_id);
    hash_part(hasher, &record.canonical_event_bytes);
    hash_part(hasher, &record.signature);
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), BuzzEventImportError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(BuzzEventImportError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, BuzzEventImportError>,
) -> Result<T, BuzzEventImportError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(BuzzEventImportError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(BuzzEventImportError::Unavailable)?;
            Err(error)
        }
    }
}
