use collaboration_domain::{
    AuditAction, AuditChainBridge, AuditChainPosition, AuditChainSource, AuditEntry, AuditError,
    AuditFields, AuditHash, AuditOutcome, AuditRecord, CommunityId, OperationId, PrincipalId,
    TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const ACQUIRE_WRITER_LOCK_SQL: &str = r#"
SELECT pg_advisory_xact_lock(
    hashtextextended('zed_collaboration_audit:' || $1::text, 0)
)
"#;
const SELECT_HEAD_SQL: &str = r#"
SELECT sequence::text AS sequence_text, entry_hash
FROM public.collaboration_audit_heads
WHERE community_id = $1
"#;
const INSERT_ENTRY_SQL: &str = r#"
INSERT INTO public.collaboration_audit_entries (
    community_id, sequence, entry_hash, previous_hash, operation_id,
    action, actor_principal_id, outcome, occurred_at_millis, fields,
    bridge_source, bridge_source_sequence, bridge_source_head
) VALUES (
    $1, CAST($2 AS numeric), $3, $4, $5, $6, $7, $8,
    CAST($9 AS numeric), CAST($10 AS jsonb), $11,
    CASE WHEN $12::text IS NULL THEN NULL ELSE CAST($12 AS numeric) END,
    $13
)
"#;
const INSERT_HEAD_SQL: &str = r#"
INSERT INTO public.collaboration_audit_heads (
    community_id, sequence, entry_hash
) VALUES ($1, CAST($2 AS numeric), $3)
"#;
const UPDATE_HEAD_SQL: &str = r#"
UPDATE public.collaboration_audit_heads
SET sequence = CAST($4 AS numeric), entry_hash = $5
WHERE community_id = $1
  AND sequence = CAST($2 AS numeric)
  AND entry_hash = $3
"#;
const ENTRY_COLUMNS: &str = r#"
    community_id,
    sequence::text AS sequence_text,
    entry_hash,
    previous_hash,
    operation_id,
    action,
    actor_principal_id,
    outcome,
    occurred_at_millis::text AS occurred_at_millis_text,
    fields::text AS fields_json,
    bridge_source,
    bridge_source_sequence::text AS bridge_source_sequence_text,
    bridge_source_head
"#;
const SELECT_SEGMENT_SQL: &str = r#"
SELECT
"#;
const SELECT_CURSOR_SQL: &str = r#"
SELECT exporter_id, cursor_version::text AS cursor_version_text,
       exported_through_sequence::text AS exported_through_sequence_text,
       exported_through_hash
FROM public.collaboration_audit_export_cursors
WHERE community_id = $1 AND exporter_id = $2
"#;
const INSERT_CURSOR_SQL: &str = r#"
INSERT INTO public.collaboration_audit_export_cursors (
    community_id, exporter_id, cursor_version,
    exported_through_sequence, exported_through_hash
) VALUES ($1, $2, 1, CAST($3 AS numeric), $4)
"#;
const UPDATE_CURSOR_SQL: &str = r#"
UPDATE public.collaboration_audit_export_cursors
SET cursor_version = CAST($3 AS numeric),
    exported_through_sequence = CAST($4 AS numeric),
    exported_through_hash = $5
WHERE community_id = $1
  AND exporter_id = $2
  AND cursor_version = CAST($6 AS numeric)
  AND exported_through_sequence = CAST($7 AS numeric)
  AND exported_through_hash = $8
"#;

pub const MAX_AUDIT_SEGMENT_ENTRIES: u32 = 1_000;
const MAX_EXPORTER_ID_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuditHead {
    community_id: CommunityId,
    sequence: u64,
    hash: AuditHash,
}

impl AuditHead {
    pub fn from_entry(entry: &AuditEntry) -> Self {
        Self {
            community_id: entry.community_id(),
            sequence: entry.sequence(),
            hash: entry.hash(),
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    pub const fn hash(self) -> AuditHash {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedAuditHead {
    Empty,
    Imported(AuditChainBridge),
    Entry(AuditHead),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditSegment {
    entries: Vec<AuditEntry>,
}

impl AuditSegment {
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn end_head(&self) -> Option<AuditHead> {
        self.entries.last().map(AuditHead::from_entry)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditExportCursor {
    community_id: CommunityId,
    exporter_id: String,
    version: u64,
    head: AuditHead,
}

impl AuditExportCursor {
    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub fn exporter_id(&self) -> &str {
        &self.exporter_id
    }

    pub const fn version(&self) -> u64 {
        self.version
    }

    pub const fn head(&self) -> AuditHead {
        self.head
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuditRepositoryError {
    #[error("audit repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("audit repository request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("audit repository input is invalid or exceeds a bound")]
    InvalidInput,
    #[error("audit repository head is no longer authoritative")]
    StaleHead,
    #[error("audit export cursor is no longer authoritative")]
    StaleCursor,
    #[error("audit repository record is invalid")]
    InvalidRecord,
    #[error("audit record violates the canonical chain contract")]
    Domain(#[source] AuditError),
    #[error("audit fields could not be encoded")]
    Encoding(#[source] serde_json::Error),
    #[error("audit repository is unavailable")]
    Unavailable(#[source] DbErr),
}

impl From<AuditError> for AuditRepositoryError {
    fn from(error: AuditError) -> Self {
        Self::Domain(error)
    }
}

pub struct AuditRepository {
    connection: DatabaseConnection,
}

impl AuditRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, AuditRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(AuditRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn load_head(
        &self,
        tenant: &TenantContext,
    ) -> Result<Option<AuditHead>, AuditRepositoryError> {
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            query_head(&transaction, tenant.community_id(), false).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn load_head_in_transaction(
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
    ) -> Result<Option<AuditHead>, AuditRepositoryError> {
        validate_transaction_tenant(transaction, tenant).await?;
        query_head(transaction, tenant.community_id(), false).await
    }

    pub async fn load_operation(
        &self,
        tenant: &TenantContext,
        operation_id: OperationId,
    ) -> Result<Option<AuditEntry>, AuditRepositoryError> {
        if operation_id.as_uuid().is_nil() {
            return Err(AuditRepositoryError::InvalidInput);
        }
        let transaction = self.begin().await?;
        let result = Self::load_operation_in_transaction(&transaction, tenant, operation_id).await;
        finish_transaction(transaction, result).await
    }

    pub async fn load_operation_in_transaction(
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        operation_id: OperationId,
    ) -> Result<Option<AuditEntry>, AuditRepositoryError> {
        if operation_id.as_uuid().is_nil() {
            return Err(AuditRepositoryError::InvalidInput);
        }
        validate_transaction_tenant(transaction, tenant).await?;
        transaction
            .query_one(operation_statement(tenant.community_id(), operation_id))
            .await
            .map_err(AuditRepositoryError::Unavailable)?
            .map(|row| entry_from_row(&row, tenant.community_id(), None))
            .transpose()
    }

    pub async fn append(
        &self,
        tenant: &TenantContext,
        expected_head: ExpectedAuditHead,
        record: AuditRecord,
    ) -> Result<AuditEntry, AuditRepositoryError> {
        validate_expected_head(tenant, expected_head)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            Self::append_bound_in_transaction(&transaction, tenant, expected_head, record).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn append_in_transaction(
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        expected_head: ExpectedAuditHead,
        record: AuditRecord,
    ) -> Result<AuditEntry, AuditRepositoryError> {
        validate_expected_head(tenant, expected_head)?;
        validate_transaction_tenant(transaction, tenant).await?;
        Self::append_bound_in_transaction(transaction, tenant, expected_head, record).await
    }

    async fn append_bound_in_transaction(
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        expected_head: ExpectedAuditHead,
        record: AuditRecord,
    ) -> Result<AuditEntry, AuditRepositoryError> {
        acquire_writer_lock(transaction, tenant.community_id()).await?;
        let stored_head = query_head(transaction, tenant.community_id(), true).await?;
        let position = match (stored_head, expected_head) {
            (None, ExpectedAuditHead::Empty) => AuditChainPosition::genesis(tenant.community_id())?,
            (None, ExpectedAuditHead::Imported(bridge)) => {
                AuditChainPosition::from_imported(tenant.community_id(), bridge)?
            }
            (Some(stored), ExpectedAuditHead::Entry(expected)) if stored == expected => {
                AuditChainPosition::after_stored_head(
                    stored.community_id,
                    stored.sequence,
                    stored.hash,
                )?
            }
            _ => return Err(AuditRepositoryError::StaleHead),
        };
        let entry = AuditEntry::append(position, record)?;
        transaction
            .execute(insert_entry_statement(&entry)?)
            .await
            .map_err(AuditRepositoryError::Unavailable)?;
        match stored_head {
            None => {
                transaction
                    .execute(insert_head_statement(&entry))
                    .await
                    .map_err(AuditRepositoryError::Unavailable)?;
            }
            Some(stored) => {
                let result = transaction
                    .execute(update_head_statement(stored, &entry))
                    .await
                    .map_err(AuditRepositoryError::Unavailable)?;
                if result.rows_affected() != 1 {
                    return Err(AuditRepositoryError::StaleHead);
                }
            }
        }
        Ok(entry)
    }

    pub async fn read_segment(
        &self,
        tenant: &TenantContext,
        from_sequence: u64,
        limit: u32,
    ) -> Result<AuditSegment, AuditRepositoryError> {
        validate_segment(from_sequence, limit)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let rows = transaction
                .query_all(segment_statement(
                    tenant.community_id(),
                    from_sequence,
                    limit,
                ))
                .await
                .map_err(AuditRepositoryError::Unavailable)?;
            hydrate_segment(rows, tenant.community_id(), from_sequence)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn export_segment(
        &self,
        tenant: &TenantContext,
        cursor: Option<&AuditExportCursor>,
        limit: u32,
    ) -> Result<AuditSegment, AuditRepositoryError> {
        if cursor.is_some_and(|cursor| cursor.community_id != tenant.community_id()) {
            return Err(AuditRepositoryError::TenantBoundaryViolation);
        }
        let from_sequence = match cursor {
            Some(cursor) => cursor
                .head
                .sequence
                .checked_add(1)
                .ok_or(AuditRepositoryError::InvalidInput)?,
            None => 1,
        };
        self.read_segment(tenant, from_sequence, limit).await
    }

    pub async fn load_export_cursor(
        &self,
        tenant: &TenantContext,
        exporter_id: &str,
    ) -> Result<Option<AuditExportCursor>, AuditRepositoryError> {
        validate_exporter_id(exporter_id)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            query_cursor(&transaction, tenant.community_id(), exporter_id, false).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn advance_export_cursor(
        &self,
        tenant: &TenantContext,
        exporter_id: &str,
        expected: Option<&AuditExportCursor>,
        exported_through: AuditHead,
    ) -> Result<AuditExportCursor, AuditRepositoryError> {
        validate_exporter_id(exporter_id)?;
        if exported_through.community_id != tenant.community_id()
            || expected.is_some_and(|cursor| {
                cursor.community_id != tenant.community_id() || cursor.exporter_id != exporter_id
            })
        {
            return Err(AuditRepositoryError::TenantBoundaryViolation);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let stored =
                query_cursor(&transaction, tenant.community_id(), exporter_id, true).await?;
            if stored.as_ref() != expected {
                return Err(AuditRepositoryError::StaleCursor);
            }
            let version = match stored {
                None => {
                    transaction
                        .execute(insert_cursor_statement(exporter_id, exported_through))
                        .await
                        .map_err(AuditRepositoryError::Unavailable)?;
                    1
                }
                Some(stored) => {
                    if exported_through.sequence <= stored.head.sequence {
                        return Err(AuditRepositoryError::StaleCursor);
                    }
                    let version = stored
                        .version
                        .checked_add(1)
                        .ok_or(AuditRepositoryError::InvalidInput)?;
                    let update = transaction
                        .execute(update_cursor_statement(&stored, version, exported_through))
                        .await
                        .map_err(AuditRepositoryError::Unavailable)?;
                    if update.rows_affected() != 1 {
                        return Err(AuditRepositoryError::StaleCursor);
                    }
                    version
                }
            };
            Ok(AuditExportCursor {
                community_id: tenant.community_id(),
                exporter_id: exporter_id.to_owned(),
                version,
                head: exported_through,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, AuditRepositoryError> {
        self.connection
            .begin()
            .await
            .map_err(AuditRepositoryError::Unavailable)
    }
}

fn validate_expected_head(
    tenant: &TenantContext,
    expected_head: ExpectedAuditHead,
) -> Result<(), AuditRepositoryError> {
    if matches!(expected_head, ExpectedAuditHead::Entry(head) if head.community_id != tenant.community_id())
    {
        return Err(AuditRepositoryError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_segment(from_sequence: u64, limit: u32) -> Result<(), AuditRepositoryError> {
    if from_sequence == 0 || limit == 0 || limit > MAX_AUDIT_SEGMENT_ENTRIES {
        return Err(AuditRepositoryError::InvalidInput);
    }
    Ok(())
}

fn validate_exporter_id(value: &str) -> Result<(), AuditRepositoryError> {
    let valid = !value.is_empty()
        && value.len() <= MAX_EXPORTER_ID_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        });
    valid
        .then_some(())
        .ok_or(AuditRepositoryError::InvalidInput)
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, AuditRepositoryError>,
) -> Result<T, AuditRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(AuditRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(AuditRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}

async fn validate_transaction_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), AuditRepositoryError> {
    if transaction.get_database_backend() != DatabaseBackend::Postgres {
        return Err(AuditRepositoryError::UnsupportedDatabase);
    }
    let configured = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await
        .map_err(AuditRepositoryError::Unavailable)?
        .ok_or(AuditRepositoryError::InvalidRecord)?;
    let configured: Option<String> = configured
        .try_get("", "community_id")
        .map_err(|_| AuditRepositoryError::InvalidRecord)?;
    if configured.is_some_and(|configured| configured != tenant.community_id().to_string()) {
        return Err(AuditRepositoryError::TenantBoundaryViolation);
    }
    set_tenant(transaction, tenant.community_id()).await?;
    Ok(())
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), AuditRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(AuditRepositoryError::Unavailable)?;
    Ok(())
}

async fn acquire_writer_lock(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), AuditRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            ACQUIRE_WRITER_LOCK_SQL,
            [community_id.as_uuid().into()],
        ))
        .await
        .map_err(AuditRepositoryError::Unavailable)?;
    Ok(())
}

async fn query_head(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    for_update: bool,
) -> Result<Option<AuditHead>, AuditRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            if for_update {
                format!("{SELECT_HEAD_SQL}FOR UPDATE")
            } else {
                SELECT_HEAD_SQL.to_owned()
            },
            [community_id.as_uuid().into()],
        ))
        .await
        .map_err(AuditRepositoryError::Unavailable)?
        .map(|row| head_from_row(&row, community_id))
        .transpose()
}

fn head_from_row(
    row: &QueryResult,
    community_id: CommunityId,
) -> Result<AuditHead, AuditRepositoryError> {
    Ok(AuditHead {
        community_id,
        sequence: parse_u64(row_value(row, "sequence_text")?)?,
        hash: AuditHash::from_bytes(bytes32(row_value(row, "entry_hash")?)?),
    })
}

fn insert_entry_statement(entry: &AuditEntry) -> Result<Statement, AuditRepositoryError> {
    let record = entry.record();
    let fields_json =
        serde_json::to_string(record.fields()).map_err(AuditRepositoryError::Encoding)?;
    let bridge = entry.chain_bridge();
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_ENTRY_SQL,
        vec![
            entry.community_id().as_uuid().into(),
            entry.sequence().to_string().into(),
            entry.hash().as_bytes().to_vec().into(),
            entry
                .previous_hash()
                .map(|hash| hash.as_bytes().to_vec())
                .into(),
            record.operation_id().as_uuid().into(),
            record.action().as_str().into(),
            record.actor_principal_id().map(PrincipalId::as_uuid).into(),
            outcome_name(record.outcome()).into(),
            record.occurred_at_millis().to_string().into(),
            fields_json.into(),
            bridge.map(|bridge| source_name(bridge.source())).into(),
            bridge
                .map(|bridge| bridge.source_sequence().to_string())
                .into(),
            bridge
                .map(|bridge| bridge.source_head().as_bytes().to_vec())
                .into(),
        ],
    ))
}

fn insert_head_statement(entry: &AuditEntry) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_HEAD_SQL,
        vec![
            entry.community_id().as_uuid().into(),
            entry.sequence().to_string().into(),
            entry.hash().as_bytes().to_vec().into(),
        ],
    )
}

fn update_head_statement(stored: AuditHead, entry: &AuditEntry) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_HEAD_SQL,
        vec![
            stored.community_id.as_uuid().into(),
            stored.sequence.to_string().into(),
            stored.hash.as_bytes().to_vec().into(),
            entry.sequence().to_string().into(),
            entry.hash().as_bytes().to_vec().into(),
        ],
    )
}

fn segment_statement(community_id: CommunityId, from_sequence: u64, limit: u32) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        format!(
            "{SELECT_SEGMENT_SQL}{ENTRY_COLUMNS}\nFROM public.collaboration_audit_entries\nWHERE community_id = $1 AND sequence >= CAST($2 AS numeric)\nORDER BY sequence ASC\nLIMIT $3"
        ),
        vec![
            community_id.as_uuid().into(),
            from_sequence.to_string().into(),
            i64::from(limit).into(),
        ],
    )
}

fn operation_statement(community_id: CommunityId, operation_id: OperationId) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        format!(
            "{SELECT_SEGMENT_SQL}{ENTRY_COLUMNS}\nFROM public.collaboration_audit_entries\nWHERE community_id = $1 AND operation_id = $2"
        ),
        vec![community_id.as_uuid().into(), operation_id.as_uuid().into()],
    )
}

fn hydrate_segment(
    rows: Vec<QueryResult>,
    community_id: CommunityId,
    from_sequence: u64,
) -> Result<AuditSegment, AuditRepositoryError> {
    let mut entries = Vec::with_capacity(rows.len());
    for row in rows {
        let entry = entry_from_row(&row, community_id, entries.last())?;
        if entries.is_empty() && entry.sequence() < from_sequence {
            return Err(AuditRepositoryError::InvalidRecord);
        }
        entries.push(entry);
    }
    Ok(AuditSegment { entries })
}

fn entry_from_row(
    row: &QueryResult,
    community_id: CommunityId,
    previous: Option<&AuditEntry>,
) -> Result<AuditEntry, AuditRepositoryError> {
    let stored_community = CommunityId::from_uuid(row_value(row, "community_id")?);
    if stored_community != community_id {
        return Err(AuditRepositoryError::TenantBoundaryViolation);
    }
    let sequence = parse_u64(row_value(row, "sequence_text")?)?;
    let hash = AuditHash::from_bytes(bytes32(row_value(row, "entry_hash")?)?);
    let previous_hash = row_value::<Option<Vec<u8>>>(row, "previous_hash")?
        .map(bytes32)
        .transpose()?
        .map(AuditHash::from_bytes);
    let bridge = bridge_from_row(row)?;
    let position = match previous {
        Some(previous) => {
            if previous_hash != Some(previous.hash()) || bridge.is_some() {
                return Err(AuditRepositoryError::InvalidRecord);
            }
            AuditChainPosition::after(previous)?
        }
        None => match (previous_hash, bridge) {
            (None, None) => AuditChainPosition::genesis(community_id)?,
            (None, Some(bridge)) => AuditChainPosition::from_imported(community_id, bridge)?,
            (Some(previous_hash), None) => {
                let previous_sequence = sequence
                    .checked_sub(1)
                    .ok_or(AuditRepositoryError::InvalidRecord)?;
                AuditChainPosition::after_stored_head(
                    community_id,
                    previous_sequence,
                    previous_hash,
                )?
            }
            (Some(_), Some(_)) => return Err(AuditRepositoryError::InvalidRecord),
        },
    };
    if position.next_sequence() != sequence {
        return Err(AuditRepositoryError::InvalidRecord);
    }
    let fields_json: String = row_value(row, "fields_json")?;
    let fields: AuditFields =
        serde_json::from_str(&fields_json).map_err(AuditRepositoryError::Encoding)?;
    let actor_principal_id =
        row_value::<Option<uuid::Uuid>>(row, "actor_principal_id")?.map(PrincipalId::from_uuid);
    let record = AuditRecord::new(
        OperationId::from_uuid(row_value(row, "operation_id")?),
        AuditAction::new(row_value::<String>(row, "action")?)?,
        actor_principal_id,
        outcome_from_database(&row_value::<String>(row, "outcome")?)?,
        parse_u64(row_value(row, "occurred_at_millis_text")?)?,
        fields,
    )?;
    AuditEntry::from_stored(position, record, hash).map_err(Into::into)
}

fn bridge_from_row(row: &QueryResult) -> Result<Option<AuditChainBridge>, AuditRepositoryError> {
    let source = row_value::<Option<String>>(row, "bridge_source")?;
    let sequence = row_value::<Option<String>>(row, "bridge_source_sequence_text")?;
    let head = row_value::<Option<Vec<u8>>>(row, "bridge_source_head")?;
    match (source, sequence, head) {
        (None, None, None) => Ok(None),
        (Some(source), Some(sequence), Some(head)) => Ok(Some(AuditChainBridge::new(
            source_from_database(&source)?,
            parse_u64(sequence)?,
            AuditHash::from_bytes(bytes32(head)?),
        )?)),
        _ => Err(AuditRepositoryError::InvalidRecord),
    }
}

async fn query_cursor(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    exporter_id: &str,
    for_update: bool,
) -> Result<Option<AuditExportCursor>, AuditRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            if for_update {
                format!("{SELECT_CURSOR_SQL}FOR UPDATE")
            } else {
                SELECT_CURSOR_SQL.to_owned()
            },
            vec![community_id.as_uuid().into(), exporter_id.into()],
        ))
        .await
        .map_err(AuditRepositoryError::Unavailable)?
        .map(|row| cursor_from_row(&row, community_id))
        .transpose()
}

fn cursor_from_row(
    row: &QueryResult,
    community_id: CommunityId,
) -> Result<AuditExportCursor, AuditRepositoryError> {
    let exporter_id: String = row_value(row, "exporter_id")?;
    validate_exporter_id(&exporter_id)?;
    let sequence = parse_u64(row_value(row, "exported_through_sequence_text")?)?;
    Ok(AuditExportCursor {
        community_id,
        exporter_id,
        version: parse_u64(row_value(row, "cursor_version_text")?)?,
        head: AuditHead {
            community_id,
            sequence,
            hash: AuditHash::from_bytes(bytes32(row_value(row, "exported_through_hash")?)?),
        },
    })
}

fn insert_cursor_statement(exporter_id: &str, head: AuditHead) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_CURSOR_SQL,
        vec![
            head.community_id.as_uuid().into(),
            exporter_id.into(),
            head.sequence.to_string().into(),
            head.hash.as_bytes().to_vec().into(),
        ],
    )
}

fn update_cursor_statement(stored: &AuditExportCursor, version: u64, head: AuditHead) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_CURSOR_SQL,
        vec![
            stored.community_id.as_uuid().into(),
            stored.exporter_id.as_str().into(),
            version.to_string().into(),
            head.sequence.to_string().into(),
            head.hash.as_bytes().to_vec().into(),
            stored.version.to_string().into(),
            stored.head.sequence.to_string().into(),
            stored.head.hash.as_bytes().to_vec().into(),
        ],
    )
}

const fn outcome_name(outcome: AuditOutcome) -> &'static str {
    match outcome {
        AuditOutcome::Succeeded => "succeeded",
        AuditOutcome::Failed => "failed",
        AuditOutcome::Denied => "denied",
        AuditOutcome::Cancelled => "cancelled",
    }
}

fn outcome_from_database(value: &str) -> Result<AuditOutcome, AuditRepositoryError> {
    match value {
        "succeeded" => Ok(AuditOutcome::Succeeded),
        "failed" => Ok(AuditOutcome::Failed),
        "denied" => Ok(AuditOutcome::Denied),
        "cancelled" => Ok(AuditOutcome::Cancelled),
        _ => Err(AuditRepositoryError::InvalidRecord),
    }
}

const fn source_name(source: AuditChainSource) -> &'static str {
    match source {
        AuditChainSource::BuzzV1 => "buzz_v1",
    }
}

fn source_from_database(value: &str) -> Result<AuditChainSource, AuditRepositoryError> {
    match value {
        "buzz_v1" => Ok(AuditChainSource::BuzzV1),
        _ => Err(AuditRepositoryError::InvalidRecord),
    }
}

fn parse_u64(value: String) -> Result<u64, AuditRepositoryError> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or(AuditRepositoryError::InvalidRecord)
}

fn bytes32(value: Vec<u8>) -> Result<[u8; 32], AuditRepositoryError> {
    value
        .try_into()
        .map_err(|_| AuditRepositoryError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, AuditRepositoryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| AuditRepositoryError::InvalidRecord)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use collaboration_domain::{
        AuditField, AuditFieldName, AuditRedaction, AuditValue, TrustedTenantRoute,
    };
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
    use uuid::Uuid;

    use super::*;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(value: u128) -> TenantContext {
        let community_id = community(value);
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "audit-repository")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn record(operation: u128) -> AuditRecord {
        AuditRecord::new(
            OperationId::from_uuid(Uuid::from_u128(operation)),
            AuditAction::new("workflow.action.completed").expect("action"),
            Some(PrincipalId::from_uuid(Uuid::from_u128(90))),
            AuditOutcome::Succeeded,
            1_900_000_000_000 + u64::try_from(operation).expect("fixture operation"),
            AuditFields::new(vec![AuditField::new(
                AuditFieldName::new("private_payload").expect("field name"),
                AuditValue::Redacted(AuditRedaction::PrivateContent),
            )])
            .expect("fields"),
        )
        .expect("record")
    }

    fn genesis_entry(community_id: CommunityId) -> AuditEntry {
        AuditEntry::append(
            AuditChainPosition::genesis(community_id).expect("genesis"),
            record(10),
        )
        .expect("entry")
    }

    fn repository(
        query_results: Vec<Vec<BTreeMap<String, SeaValue>>>,
        affected_rows: &[u64],
    ) -> AuditRepository {
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
        AuditRepository::new(database).expect("repository")
    }

    fn head_row(head: AuditHead) -> BTreeMap<String, SeaValue> {
        BTreeMap::from([
            ("sequence_text".into(), head.sequence.to_string().into()),
            ("entry_hash".into(), head.hash.as_bytes().to_vec().into()),
        ])
    }

    fn entry_row(entry: &AuditEntry) -> BTreeMap<String, SeaValue> {
        let bridge = entry.chain_bridge();
        BTreeMap::from([
            ("community_id".into(), entry.community_id().as_uuid().into()),
            ("sequence_text".into(), entry.sequence().to_string().into()),
            ("entry_hash".into(), entry.hash().as_bytes().to_vec().into()),
            (
                "previous_hash".into(),
                entry
                    .previous_hash()
                    .map(|hash| hash.as_bytes().to_vec())
                    .into(),
            ),
            (
                "operation_id".into(),
                entry.record().operation_id().as_uuid().into(),
            ),
            (
                "action".into(),
                entry.record().action().as_str().to_owned().into(),
            ),
            (
                "actor_principal_id".into(),
                entry
                    .record()
                    .actor_principal_id()
                    .map(PrincipalId::as_uuid)
                    .into(),
            ),
            (
                "outcome".into(),
                outcome_name(entry.record().outcome()).to_owned().into(),
            ),
            (
                "occurred_at_millis_text".into(),
                entry.record().occurred_at_millis().to_string().into(),
            ),
            (
                "fields_json".into(),
                serde_json::to_string(entry.record().fields())
                    .expect("fields JSON")
                    .into(),
            ),
            (
                "bridge_source".into(),
                bridge
                    .map(|bridge| source_name(bridge.source()).to_owned())
                    .into(),
            ),
            (
                "bridge_source_sequence_text".into(),
                bridge
                    .map(|bridge| bridge.source_sequence().to_string())
                    .into(),
            ),
            (
                "bridge_source_head".into(),
                bridge
                    .map(|bridge| bridge.source_head().as_bytes().to_vec())
                    .into(),
            ),
        ])
    }

    fn cursor_row(cursor: &AuditExportCursor) -> BTreeMap<String, SeaValue> {
        BTreeMap::from([
            ("exporter_id".into(), cursor.exporter_id.clone().into()),
            (
                "cursor_version_text".into(),
                cursor.version.to_string().into(),
            ),
            (
                "exported_through_sequence_text".into(),
                cursor.head.sequence.to_string().into(),
            ),
            (
                "exported_through_hash".into(),
                cursor.head.hash.as_bytes().to_vec().into(),
            ),
        ])
    }

    fn log(repository: AuditRepository) -> String {
        format!("{:#?}", repository.into_connection().into_transaction_log())
    }

    #[tokio::test]
    async fn concurrent_contenders_commit_one_successor_and_fence_the_stale_writer() {
        let tenant = tenant(1);
        let seed = genesis_entry(tenant.community_id());
        let expected = AuditHead::from_entry(&seed);
        let winner = repository(vec![vec![head_row(expected)]], &[1, 1, 1, 1]);
        let winner_record = record(11);

        let winner_entry = AuditEntry::append(
            AuditChainPosition::after(&seed).expect("successor position"),
            winner_record.clone(),
        )
        .expect("successor");
        let loser = repository(
            vec![vec![head_row(AuditHead::from_entry(&winner_entry))]],
            &[1, 1],
        );

        let (winner_result, loser_result) = tokio::join!(
            winner.append(&tenant, ExpectedAuditHead::Entry(expected), winner_record),
            loser.append(&tenant, ExpectedAuditHead::Entry(expected), record(12)),
        );

        let committed = winner_result.expect("winner");
        committed.verify().expect("valid committed hash");
        assert_eq!(committed.sequence(), 2);
        assert_eq!(committed.previous_hash(), Some(seed.hash()));
        assert!(matches!(loser_result, Err(AuditRepositoryError::StaleHead)));

        let winner_log = log(winner);
        assert!(winner_log.contains("pg_advisory_xact_lock"));
        assert!(winner_log.contains("FOR UPDATE"));
        assert!(winner_log.contains("INSERT INTO public.collaboration_audit_entries"));
        assert!(winner_log.contains("UPDATE public.collaboration_audit_heads"));
        let loser_log = log(loser);
        assert!(loser_log.contains("FOR UPDATE"));
        assert!(!loser_log.contains("INSERT INTO public.collaboration_audit_entries"));
    }

    #[tokio::test]
    async fn append_supports_genesis_and_imported_heads_without_conflating_them() {
        let tenant = tenant(1);
        let genesis_repository = repository(vec![vec![]], &[1, 1, 1, 1]);
        let genesis = genesis_repository
            .append(&tenant, ExpectedAuditHead::Empty, record(20))
            .await
            .expect("genesis append");
        assert_eq!(genesis.sequence(), 1);
        assert_eq!(genesis.previous_hash(), None);
        assert_eq!(genesis.chain_bridge(), None);

        let bridge =
            AuditChainBridge::new(AuditChainSource::BuzzV1, 42, AuditHash::from_bytes([7; 32]))
                .expect("bridge");
        let imported_repository = repository(vec![vec![]], &[1, 1, 1, 1]);
        let imported = imported_repository
            .append(&tenant, ExpectedAuditHead::Imported(bridge), record(21))
            .await
            .expect("imported append");
        assert_eq!(imported.sequence(), 43);
        assert_eq!(imported.chain_bridge(), Some(bridge));
        assert_ne!(genesis.hash(), imported.hash());
        let imported_log = log(imported_repository);
        assert!(imported_log.contains("buzz_v1"));
    }

    #[tokio::test]
    async fn segment_hydration_verifies_order_hashes_and_tenant_scope() {
        let tenant = tenant(1);
        let first = genesis_entry(tenant.community_id());
        let second = AuditEntry::append(
            AuditChainPosition::after(&first).expect("position"),
            record(11),
        )
        .expect("second entry");
        let valid = repository(vec![vec![entry_row(&first), entry_row(&second)]], &[1]);
        let segment = valid
            .read_segment(&tenant, 1, 100)
            .await
            .expect("valid segment");
        assert_eq!(segment.entries(), &[first.clone(), second.clone()]);
        assert_eq!(segment.end_head(), Some(AuditHead::from_entry(&second)));

        let mut corrupted_row = entry_row(&first);
        corrupted_row.insert("outcome".into(), "failed".to_owned().into());
        let corrupted = repository(vec![vec![corrupted_row]], &[1]);
        assert!(matches!(
            corrupted.read_segment(&tenant, 1, 100).await,
            Err(AuditRepositoryError::Domain(AuditError::HashMismatch))
        ));

        let reordered = repository(vec![vec![entry_row(&second), entry_row(&first)]], &[1]);
        assert!(matches!(
            reordered.read_segment(&tenant, 1, 100).await,
            Err(AuditRepositoryError::InvalidRecord)
                | Err(AuditRepositoryError::Domain(AuditError::HashMismatch))
        ));

        let foreign = genesis_entry(community(2));
        let cross_tenant = repository(vec![vec![entry_row(&foreign)]], &[1]);
        assert!(matches!(
            cross_tenant.read_segment(&tenant, 1, 100).await,
            Err(AuditRepositoryError::TenantBoundaryViolation)
        ));
    }

    #[tokio::test]
    async fn export_cursor_advances_exactly_and_cross_tenant_inputs_do_no_database_work() {
        let tenant = tenant(1);
        let first = genesis_entry(tenant.community_id());
        let first_head = AuditHead::from_entry(&first);
        let cursor_repository = repository(vec![vec![]], &[1, 1]);
        let cursor = cursor_repository
            .advance_export_cursor(&tenant, "operator_archive", None, first_head)
            .await
            .expect("initial cursor");
        assert_eq!(cursor.version(), 1);
        assert_eq!(cursor.head(), first_head);

        let second = AuditEntry::append(
            AuditChainPosition::after(&first).expect("position"),
            record(11),
        )
        .expect("second");
        let stale_stored = AuditExportCursor {
            community_id: tenant.community_id(),
            exporter_id: "operator_archive".into(),
            version: 2,
            head: AuditHead::from_entry(&second),
        };
        let stale_repository = repository(vec![vec![cursor_row(&stale_stored)]], &[1]);
        assert!(matches!(
            stale_repository
                .advance_export_cursor(
                    &tenant,
                    "operator_archive",
                    Some(&cursor),
                    AuditHead::from_entry(&second),
                )
                .await,
            Err(AuditRepositoryError::StaleCursor)
        ));
        assert!(
            !log(stale_repository).contains("UPDATE public.collaboration_audit_export_cursors")
        );

        let foreign_head = AuditHead::from_entry(&genesis_entry(community(2)));
        let foreign_repository = repository(Vec::new(), &[]);
        assert!(matches!(
            foreign_repository
                .advance_export_cursor(&tenant, "operator_archive", None, foreign_head)
                .await,
            Err(AuditRepositoryError::TenantBoundaryViolation)
        ));
        assert!(
            foreign_repository
                .into_connection()
                .into_transaction_log()
                .is_empty()
        );
    }
}
