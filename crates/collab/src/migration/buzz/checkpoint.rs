use collaboration_domain::{AggregateVersion, CommunityId, TenantContext};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use uuid::Uuid;

const MAX_CURSOR_TOKEN_BYTES: usize = 65_536;
const MAX_LABEL_BYTES: usize = 256;
const MAX_ERROR_BYTES: usize = 2_048;
const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";

const INSERT_RUN_SQL: &str = r#"
INSERT INTO public.collaboration_migration_runs (
    run_id, community_id, source_revision
) VALUES ($1, $2, $3)
ON CONFLICT (run_id) DO UPDATE
SET source_revision = public.collaboration_migration_runs.source_revision
WHERE public.collaboration_migration_runs.community_id = EXCLUDED.community_id
  AND public.collaboration_migration_runs.source_revision = EXCLUDED.source_revision
RETURNING community_id
"#;

const INSERT_CHECKPOINT_SQL: &str = r#"
INSERT INTO public.collaboration_migration_checkpoints (
    community_id, run_id, stream_name, shard_id, checkpoint_version, status,
    source_cursor_sequence, source_cursor_token,
    target_cursor_sequence, target_cursor_token,
    scanned_count, imported_count, skipped_count, failed_count,
    source_hash, target_hash, rollback_label, rollback_irreversible,
    irreversible_at, last_error
) VALUES (
    $1, $2, $3, $4, 1, 'pending',
    0, NULL, 0, NULL, 0, 0, 0, 0,
    NULL, NULL, $5, false, NULL, NULL
)
"#;

const UPDATE_CHECKPOINT_SQL: &str = r#"
UPDATE public.collaboration_migration_checkpoints
SET checkpoint_version = CAST($5 AS numeric),
    status = $6,
    source_cursor_sequence = CAST($7 AS numeric),
    source_cursor_token = $8,
    target_cursor_sequence = CAST($9 AS numeric),
    target_cursor_token = $10,
    scanned_count = CAST($11 AS numeric),
    imported_count = CAST($12 AS numeric),
    skipped_count = CAST($13 AS numeric),
    failed_count = CAST($14 AS numeric),
    source_hash = $15,
    target_hash = $16,
    rollback_label = $17,
    rollback_irreversible = $18,
    irreversible_at = CASE
        WHEN $19::bigint IS NULL THEN NULL
        ELSE to_timestamp($19::double precision / 1000)
    END,
    last_error = $20
WHERE community_id = $1
  AND run_id = $2
  AND stream_name = $3
  AND shard_id = $4
  AND checkpoint_version = CAST($21 AS numeric)
"#;

const SELECT_CHECKPOINT_SQL: &str = r#"
SELECT checkpoint.community_id,
       checkpoint.run_id,
       run.source_revision,
       checkpoint.stream_name,
       checkpoint.shard_id,
       checkpoint.checkpoint_version::text AS checkpoint_version_text,
       checkpoint.status,
       checkpoint.source_cursor_sequence::text AS source_cursor_sequence_text,
       checkpoint.source_cursor_token,
       checkpoint.target_cursor_sequence::text AS target_cursor_sequence_text,
       checkpoint.target_cursor_token,
       checkpoint.scanned_count::text AS scanned_count_text,
       checkpoint.imported_count::text AS imported_count_text,
       checkpoint.skipped_count::text AS skipped_count_text,
       checkpoint.failed_count::text AS failed_count_text,
       checkpoint.source_hash,
       checkpoint.target_hash,
       checkpoint.rollback_label,
       checkpoint.rollback_irreversible,
       CASE WHEN checkpoint.irreversible_at IS NULL THEN NULL
            ELSE floor(extract(epoch FROM checkpoint.irreversible_at) * 1000)::bigint
       END AS irreversible_at_millis,
       checkpoint.last_error
FROM public.collaboration_migration_checkpoints AS checkpoint
JOIN public.collaboration_migration_runs AS run
  ON run.community_id = checkpoint.community_id
 AND run.run_id = checkpoint.run_id
WHERE checkpoint.community_id = $1
  AND checkpoint.run_id = $2
  AND checkpoint.stream_name = $3
  AND checkpoint.shard_id = $4
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationStream {
    SignedEvents,
    CommunityState,
    ObjectGitMetadata,
    DesktopState,
    AgentState,
    WorkflowState,
    ModerationState,
    MediaState,
}

impl MigrationStream {
    const fn database_name(self) -> &'static str {
        match self {
            Self::SignedEvents => "signed_events",
            Self::CommunityState => "community_state",
            Self::ObjectGitMetadata => "object_git_metadata",
            Self::DesktopState => "desktop_state",
            Self::AgentState => "agent_state",
            Self::WorkflowState => "workflow_state",
            Self::ModerationState => "moderation_state",
            Self::MediaState => "media_state",
        }
    }

    fn from_database(value: &str) -> Result<Self, MigrationCheckpointError> {
        match value {
            "signed_events" => Ok(Self::SignedEvents),
            "community_state" => Ok(Self::CommunityState),
            "object_git_metadata" => Ok(Self::ObjectGitMetadata),
            "desktop_state" => Ok(Self::DesktopState),
            "agent_state" => Ok(Self::AgentState),
            "workflow_state" => Ok(Self::WorkflowState),
            "moderation_state" => Ok(Self::ModerationState),
            "media_state" => Ok(Self::MediaState),
            _ => Err(MigrationCheckpointError::InvalidRecord),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationCheckpointStatus {
    Pending,
    Running,
    Interrupted,
    Verifying,
    Verified,
    Failed,
    RolledBack,
}

impl MigrationCheckpointStatus {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Interrupted => "interrupted",
            Self::Verifying => "verifying",
            Self::Verified => "verified",
            Self::Failed => "failed",
            Self::RolledBack => "rolled_back",
        }
    }

    fn from_database(value: &str) -> Result<Self, MigrationCheckpointError> {
        match value {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "interrupted" => Ok(Self::Interrupted),
            "verifying" => Ok(Self::Verifying),
            "verified" => Ok(Self::Verified),
            "failed" => Ok(Self::Failed),
            "rolled_back" => Ok(Self::RolledBack),
            _ => Err(MigrationCheckpointError::InvalidRecord),
        }
    }

    const fn permits(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Pending, Self::Running | Self::Failed)
                | (
                    Self::Running,
                    Self::Running | Self::Interrupted | Self::Verifying | Self::Failed
                )
                | (
                    Self::Interrupted,
                    Self::Running | Self::Failed | Self::RolledBack
                )
                | (Self::Verifying, Self::Verified | Self::Failed)
                | (Self::Failed, Self::Running | Self::RolledBack)
                | (Self::Verified, Self::RolledBack)
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationCursor {
    sequence: u64,
    token: Option<Vec<u8>>,
}

impl MigrationCursor {
    pub fn new(sequence: u64, token: Option<Vec<u8>>) -> Result<Self, MigrationCheckpointError> {
        if token
            .as_ref()
            .is_some_and(|token| token.len() > MAX_CURSOR_TOKEN_BYTES)
        {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        Ok(Self { sequence, token })
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn token(&self) -> Option<&[u8]> {
        self.token.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MigrationCounts {
    pub scanned: u64,
    pub imported: u64,
    pub skipped: u64,
    pub failed: u64,
}

impl MigrationCounts {
    fn validate(self) -> Result<Self, MigrationCheckpointError> {
        let accounted = self
            .imported
            .checked_add(self.skipped)
            .and_then(|count| count.checked_add(self.failed))
            .ok_or(MigrationCheckpointError::InvalidInput)?;
        if accounted > self.scanned {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        Ok(self)
    }

    const fn follows(self, previous: Self) -> bool {
        self.scanned >= previous.scanned
            && self.imported >= previous.imported
            && self.skipped >= previous.skipped
            && self.failed >= previous.failed
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollbackBoundary {
    label: String,
    irreversible_at_millis: Option<u64>,
}

impl RollbackBoundary {
    pub fn reversible(label: impl Into<String>) -> Result<Self, MigrationCheckpointError> {
        Self::new(label.into(), None)
    }

    pub fn irreversible(
        label: impl Into<String>,
        crossed_at_millis: u64,
    ) -> Result<Self, MigrationCheckpointError> {
        Self::new(label.into(), Some(crossed_at_millis))
    }

    fn new(
        label: String,
        irreversible_at_millis: Option<u64>,
    ) -> Result<Self, MigrationCheckpointError> {
        if label.is_empty()
            || label.len() > MAX_LABEL_BYTES
            || label.trim() != label
            || irreversible_at_millis.is_some_and(|value| i64::try_from(value).is_err())
        {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        Ok(Self {
            label,
            irreversible_at_millis,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn irreversible_at_millis(&self) -> Option<u64> {
        self.irreversible_at_millis
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationCheckpoint {
    community_id: CommunityId,
    run_id: Uuid,
    source_revision: String,
    stream: MigrationStream,
    shard_id: String,
    version: AggregateVersion,
    status: MigrationCheckpointStatus,
    source_cursor: MigrationCursor,
    target_cursor: MigrationCursor,
    counts: MigrationCounts,
    source_hash: Option<[u8; 32]>,
    target_hash: Option<[u8; 32]>,
    rollback_boundary: RollbackBoundary,
    last_error: Option<String>,
}

impl MigrationCheckpoint {
    pub fn new(
        community_id: CommunityId,
        run_id: Uuid,
        source_revision: impl Into<String>,
        stream: MigrationStream,
        shard_id: impl Into<String>,
        rollback_boundary: RollbackBoundary,
    ) -> Result<Self, MigrationCheckpointError> {
        let source_revision = source_revision.into();
        let shard_id = shard_id.into();
        if run_id.is_nil()
            || source_revision.is_empty()
            || source_revision.len() > MAX_LABEL_BYTES
            || source_revision.trim() != source_revision
            || shard_id.is_empty()
            || shard_id.len() > MAX_LABEL_BYTES
            || shard_id.trim() != shard_id
            || rollback_boundary.irreversible_at_millis.is_some()
        {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        Ok(Self {
            community_id,
            run_id,
            source_revision,
            stream,
            shard_id,
            version: AggregateVersion::FIRST,
            status: MigrationCheckpointStatus::Pending,
            source_cursor: MigrationCursor::new(0, None)?,
            target_cursor: MigrationCursor::new(0, None)?,
            counts: MigrationCounts::default(),
            source_hash: None,
            target_hash: None,
            rollback_boundary,
            last_error: None,
        })
    }

    pub fn transition(
        &self,
        update: MigrationCheckpointUpdate,
    ) -> Result<Self, MigrationCheckpointError> {
        update.validate()?;
        if !self.status.permits(update.status)
            || !cursor_follows(&update.source_cursor, &self.source_cursor)
            || !cursor_follows(&update.target_cursor, &self.target_cursor)
            || !update.counts.follows(self.counts)
            || (self.rollback_boundary.irreversible_at_millis.is_some()
                && update.rollback_boundary != self.rollback_boundary)
            || (update.status == MigrationCheckpointStatus::RolledBack
                && update.rollback_boundary.irreversible_at_millis.is_some())
            || (update.counts == self.counts
                && (update.source_hash != self.source_hash
                    || update.target_hash != self.target_hash))
        {
            return Err(MigrationCheckpointError::ProgressRegression);
        }
        let version = self
            .version
            .next()
            .ok_or(MigrationCheckpointError::ProgressRegression)?;
        Ok(Self {
            community_id: self.community_id,
            run_id: self.run_id,
            source_revision: self.source_revision.clone(),
            stream: self.stream,
            shard_id: self.shard_id.clone(),
            version,
            status: update.status,
            source_cursor: update.source_cursor,
            target_cursor: update.target_cursor,
            counts: update.counts,
            source_hash: update.source_hash,
            target_hash: update.target_hash,
            rollback_boundary: update.rollback_boundary,
            last_error: update.last_error,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn run_id(&self) -> Uuid {
        self.run_id
    }

    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    pub const fn stream(&self) -> MigrationStream {
        self.stream
    }

    pub fn shard_id(&self) -> &str {
        &self.shard_id
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }

    pub const fn status(&self) -> MigrationCheckpointStatus {
        self.status
    }

    pub const fn counts(&self) -> MigrationCounts {
        self.counts
    }

    pub const fn source_cursor(&self) -> &MigrationCursor {
        &self.source_cursor
    }

    pub const fn target_cursor(&self) -> &MigrationCursor {
        &self.target_cursor
    }

    pub const fn rollback_boundary(&self) -> &RollbackBoundary {
        &self.rollback_boundary
    }
}

fn cursor_follows(next: &MigrationCursor, previous: &MigrationCursor) -> bool {
    next.sequence > previous.sequence
        || (next.sequence == previous.sequence && next.token == previous.token)
}

pub struct MigrationCheckpointUpdate {
    pub status: MigrationCheckpointStatus,
    pub source_cursor: MigrationCursor,
    pub target_cursor: MigrationCursor,
    pub counts: MigrationCounts,
    pub source_hash: Option<[u8; 32]>,
    pub target_hash: Option<[u8; 32]>,
    pub rollback_boundary: RollbackBoundary,
    pub last_error: Option<String>,
}

impl MigrationCheckpointUpdate {
    fn validate(&self) -> Result<(), MigrationCheckpointError> {
        self.counts.validate()?;
        if self
            .last_error
            .as_ref()
            .is_some_and(|error| error.is_empty() || error.len() > MAX_ERROR_BYTES)
            || (self.status == MigrationCheckpointStatus::Failed && self.last_error.is_none())
            || (matches!(
                self.status,
                MigrationCheckpointStatus::Pending
                    | MigrationCheckpointStatus::Running
                    | MigrationCheckpointStatus::Verifying
                    | MigrationCheckpointStatus::Verified
                    | MigrationCheckpointStatus::RolledBack
            ) && self.last_error.is_some())
        {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MigrationCheckpointError {
    #[error("Buzz migration checkpoints require PostgreSQL")]
    UnsupportedDatabase,
    #[error("Buzz migration checkpoint input is invalid")]
    InvalidInput,
    #[error("Buzz migration checkpoint crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("Buzz migration checkpoint progress or status regressed")]
    ProgressRegression,
    #[error("Buzz migration checkpoint version conflicted")]
    VersionConflict,
    #[error("Buzz migration checkpoint record is invalid")]
    InvalidRecord,
    #[error("Buzz migration checkpoint storage is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct MigrationCheckpointRepository {
    connection: DatabaseConnection,
}

impl MigrationCheckpointRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, MigrationCheckpointError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(MigrationCheckpointError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn create(
        &self,
        tenant: &TenantContext,
        checkpoint: &MigrationCheckpoint,
    ) -> Result<(), MigrationCheckpointError> {
        validate_tenant(tenant, checkpoint)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            let assigned = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    INSERT_RUN_SQL,
                    [
                        checkpoint.run_id.into(),
                        checkpoint.community_id.as_uuid().into(),
                        checkpoint.source_revision.clone().into(),
                    ],
                ))
                .await
                .map_err(MigrationCheckpointError::Unavailable)?;
            if assigned.is_none() {
                return Err(MigrationCheckpointError::TenantBoundaryViolation);
            }
            let inserted = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    INSERT_CHECKPOINT_SQL,
                    [
                        checkpoint.community_id.as_uuid().into(),
                        checkpoint.run_id.into(),
                        checkpoint.stream.database_name().into(),
                        checkpoint.shard_id.clone().into(),
                        checkpoint.rollback_boundary.label.clone().into(),
                    ],
                ))
                .await
                .map_err(MigrationCheckpointError::Unavailable)?;
            if inserted.rows_affected() != 1 {
                return Err(MigrationCheckpointError::VersionConflict);
            }
            Ok(())
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn save_transition(
        &self,
        tenant: &TenantContext,
        current: &MigrationCheckpoint,
        update: MigrationCheckpointUpdate,
    ) -> Result<MigrationCheckpoint, MigrationCheckpointError> {
        validate_tenant(tenant, current)?;
        let next = current.transition(update)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            let saved = transaction
                .execute(update_statement(current, &next))
                .await
                .map_err(MigrationCheckpointError::Unavailable)?;
            if saved.rows_affected() != 1 {
                return Err(MigrationCheckpointError::VersionConflict);
            }
            Ok(next)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn load(
        &self,
        tenant: &TenantContext,
        run_id: Uuid,
        stream: MigrationStream,
        shard_id: &str,
    ) -> Result<Option<MigrationCheckpoint>, MigrationCheckpointError> {
        if run_id.is_nil() || shard_id.is_empty() || shard_id.len() > MAX_LABEL_BYTES {
            return Err(MigrationCheckpointError::InvalidInput);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_CHECKPOINT_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        run_id.into(),
                        stream.database_name().into(),
                        shard_id.into(),
                    ],
                ))
                .await
                .map_err(MigrationCheckpointError::Unavailable)?
                .map(checkpoint_from_row)
                .transpose()
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    async fn begin(&self) -> Result<DatabaseTransaction, MigrationCheckpointError> {
        self.connection
            .begin()
            .await
            .map_err(MigrationCheckpointError::Unavailable)
    }
}

fn update_statement(current: &MigrationCheckpoint, next: &MigrationCheckpoint) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_CHECKPOINT_SQL,
        [
            next.community_id.as_uuid().into(),
            next.run_id.into(),
            next.stream.database_name().into(),
            next.shard_id.clone().into(),
            next.version.to_string().into(),
            next.status.database_name().into(),
            next.source_cursor.sequence.to_string().into(),
            next.source_cursor.token.clone().into(),
            next.target_cursor.sequence.to_string().into(),
            next.target_cursor.token.clone().into(),
            next.counts.scanned.to_string().into(),
            next.counts.imported.to_string().into(),
            next.counts.skipped.to_string().into(),
            next.counts.failed.to_string().into(),
            next.source_hash.map(|hash| hash.to_vec()).into(),
            next.target_hash.map(|hash| hash.to_vec()).into(),
            next.rollback_boundary.label.clone().into(),
            next.rollback_boundary
                .irreversible_at_millis
                .is_some()
                .into(),
            next.rollback_boundary
                .irreversible_at_millis
                .and_then(|value| i64::try_from(value).ok())
                .into(),
            next.last_error.clone().into(),
            current.version.to_string().into(),
        ],
    )
}

fn checkpoint_from_row(row: QueryResult) -> Result<MigrationCheckpoint, MigrationCheckpointError> {
    let community_id = CommunityId::from_uuid(column(&row, "community_id")?);
    let version = number(&row, "checkpoint_version_text")?
        .and_then(AggregateVersion::new)
        .ok_or(MigrationCheckpointError::InvalidRecord)?;
    let rollback_irreversible: bool = column(&row, "rollback_irreversible")?;
    let irreversible_at_millis: Option<i64> = column(&row, "irreversible_at_millis")?;
    let irreversible_at_millis = irreversible_at_millis
        .map(u64::try_from)
        .transpose()
        .map_err(|_| MigrationCheckpointError::InvalidRecord)?;
    if rollback_irreversible != irreversible_at_millis.is_some() {
        return Err(MigrationCheckpointError::InvalidRecord);
    }
    let checkpoint = MigrationCheckpoint {
        community_id,
        run_id: column(&row, "run_id")?,
        source_revision: column(&row, "source_revision")?,
        stream: MigrationStream::from_database(&column::<String>(&row, "stream_name")?)?,
        shard_id: column(&row, "shard_id")?,
        version,
        status: MigrationCheckpointStatus::from_database(&column::<String>(&row, "status")?)?,
        source_cursor: MigrationCursor::new(
            number(&row, "source_cursor_sequence_text")?
                .ok_or(MigrationCheckpointError::InvalidRecord)?,
            column(&row, "source_cursor_token")?,
        )?,
        target_cursor: MigrationCursor::new(
            number(&row, "target_cursor_sequence_text")?
                .ok_or(MigrationCheckpointError::InvalidRecord)?,
            column(&row, "target_cursor_token")?,
        )?,
        counts: MigrationCounts {
            scanned: required_number(&row, "scanned_count_text")?,
            imported: required_number(&row, "imported_count_text")?,
            skipped: required_number(&row, "skipped_count_text")?,
            failed: required_number(&row, "failed_count_text")?,
        }
        .validate()?,
        source_hash: fixed_hash(column(&row, "source_hash")?)?,
        target_hash: fixed_hash(column(&row, "target_hash")?)?,
        rollback_boundary: RollbackBoundary::new(
            column(&row, "rollback_label")?,
            irreversible_at_millis,
        )?,
        last_error: column(&row, "last_error")?,
    };
    MigrationCheckpointUpdate {
        status: checkpoint.status,
        source_cursor: checkpoint.source_cursor.clone(),
        target_cursor: checkpoint.target_cursor.clone(),
        counts: checkpoint.counts,
        source_hash: checkpoint.source_hash,
        target_hash: checkpoint.target_hash,
        rollback_boundary: checkpoint.rollback_boundary.clone(),
        last_error: checkpoint.last_error.clone(),
    }
    .validate()?;
    Ok(checkpoint)
}

fn required_number(row: &QueryResult, name: &str) -> Result<u64, MigrationCheckpointError> {
    number(row, name)?.ok_or(MigrationCheckpointError::InvalidRecord)
}

fn number(row: &QueryResult, name: &str) -> Result<Option<u64>, MigrationCheckpointError> {
    let value: Option<String> = column(row, name)?;
    value
        .map(|value| {
            value
                .parse()
                .map_err(|_| MigrationCheckpointError::InvalidRecord)
        })
        .transpose()
}

fn fixed_hash(value: Option<Vec<u8>>) -> Result<Option<[u8; 32]>, MigrationCheckpointError> {
    value
        .map(|value| {
            value
                .try_into()
                .map_err(|_| MigrationCheckpointError::InvalidRecord)
        })
        .transpose()
}

fn column<T: sea_orm::TryGetable>(
    row: &QueryResult,
    name: &str,
) -> Result<T, MigrationCheckpointError> {
    row.try_get("", name)
        .map_err(|_| MigrationCheckpointError::InvalidRecord)
}

fn validate_tenant(
    tenant: &TenantContext,
    checkpoint: &MigrationCheckpoint,
) -> Result<(), MigrationCheckpointError> {
    if tenant.community_id() != checkpoint.community_id {
        return Err(MigrationCheckpointError::TenantBoundaryViolation);
    }
    Ok(())
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), MigrationCheckpointError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(MigrationCheckpointError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, MigrationCheckpointError>,
) -> Result<T, MigrationCheckpointError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(MigrationCheckpointError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(MigrationCheckpointError::Unavailable)?;
            Err(error)
        }
    }
}
