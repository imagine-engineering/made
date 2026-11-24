use async_trait::async_trait;
use collaboration_domain::{
    CommunityId, IntegrityAlgorithm, Provenance, SourceSystem, TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    Statement, TransactionTrait, Value,
};
use sha2::{Digest, Sha256};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
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
    $1, $2, $3, $4, $5,
    to_timestamp($6::double precision / 1000), $7, $8,
    1, 1, NULL, $9, $10, $11, clock_timestamp(), NULL, $12
)
ON CONFLICT (community_id, projection_name, source_system, source_record_id)
DO UPDATE SET
    source_version = EXCLUDED.source_version,
    source_observed_at = EXCLUDED.source_observed_at,
    source_integrity_algorithm = EXCLUDED.source_integrity_algorithm,
    source_integrity_value = EXCLUDED.source_integrity_value,
    projection_version = public.collaboration_projection_checkpoints.projection_version + 1,
    cursor = NULL,
    drift_state = EXCLUDED.drift_state,
    authoritative_hash = EXCLUDED.authoritative_hash,
    projection_hash = EXCLUDED.projection_hash,
    projected_at = EXCLUDED.projected_at,
    last_error = EXCLUDED.last_error
"#;

pub const MAX_PROJECTION_NAME_BYTES: usize = 128;
pub const MAX_PROJECTION_ROWS: usize = 100_000;
pub const MAX_PROJECTION_ROW_KEY_BYTES: usize = 1024;
pub const MAX_PROJECTION_ROW_PAYLOAD_BYTES: usize = 1024 * 1024;
pub const MAX_PROJECTION_TOTAL_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionSource {
    projection_name: String,
    provenance: Provenance,
}

impl ProjectionSource {
    pub fn new(
        projection_name: impl Into<String>,
        provenance: Provenance,
    ) -> Result<Self, ProjectionRebuildError> {
        let projection_name = projection_name.into();
        if projection_name.is_empty()
            || projection_name.len() > MAX_PROJECTION_NAME_BYTES
            || projection_name.trim() != projection_name
            || projection_name.chars().any(char::is_control)
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
            return Err(ProjectionRebuildError::InvalidInput);
        }
        Ok(Self {
            projection_name,
            provenance,
        })
    }

    pub fn projection_name(&self) -> &str {
        &self.projection_name
    }

    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionRow {
    key: String,
    payload: Vec<u8>,
}

impl ProjectionRow {
    pub fn new(key: impl Into<String>, payload: Vec<u8>) -> Result<Self, ProjectionRebuildError> {
        let key = key.into();
        if key.is_empty()
            || key.len() > MAX_PROJECTION_ROW_KEY_BYTES
            || key.chars().any(char::is_control)
            || payload.len() > MAX_PROJECTION_ROW_PAYLOAD_BYTES
        {
            return Err(ProjectionRebuildError::InvalidInput);
        }
        Ok(Self { key, payload })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionRows(Vec<ProjectionRow>);

impl ProjectionRows {
    pub fn new(mut rows: Vec<ProjectionRow>) -> Result<Self, ProjectionRebuildError> {
        let total_payload_bytes = rows
            .iter()
            .try_fold(0_usize, |total, row| total.checked_add(row.payload.len()))
            .ok_or(ProjectionRebuildError::InvalidInput)?;
        if rows.len() > MAX_PROJECTION_ROWS
            || total_payload_bytes > MAX_PROJECTION_TOTAL_PAYLOAD_BYTES
        {
            return Err(ProjectionRebuildError::InvalidInput);
        }
        rows.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        if rows.windows(2).any(|pair| pair[0].key == pair[1].key) {
            return Err(ProjectionRebuildError::InvalidInput);
        }
        Ok(Self(rows))
    }

    pub fn as_slice(&self) -> &[ProjectionRow] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionMaterialization {
    source_version: Option<String>,
    rows: ProjectionRows,
}

impl ProjectionMaterialization {
    pub fn new(
        source_version: Option<String>,
        rows: ProjectionRows,
    ) -> Result<Self, ProjectionRebuildError> {
        if source_version
            .as_ref()
            .is_some_and(|version| version.is_empty() || version.len() > 1024)
        {
            return Err(ProjectionRebuildError::InvalidInput);
        }
        Ok(Self {
            source_version,
            rows,
        })
    }

    pub fn source_version(&self) -> Option<&str> {
        self.source_version.as_deref()
    }

    pub const fn rows(&self) -> &ProjectionRows {
        &self.rows
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionDriftState {
    Clean,
    Diverged,
}

impl ProjectionDriftState {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Diverged => "diverged",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionRebuildDiagnostic {
    community_id: CommunityId,
    projection_name: String,
    source_record_id: String,
    state: ProjectionDriftState,
    authoritative_source_version: Option<String>,
    projection_source_version: Option<String>,
    authoritative_count: usize,
    projection_count: usize,
    authoritative_hash: [u8; 32],
    projection_hash: [u8; 32],
}

impl ProjectionRebuildDiagnostic {
    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub fn projection_name(&self) -> &str {
        &self.projection_name
    }

    pub fn source_record_id(&self) -> &str {
        &self.source_record_id
    }

    pub const fn state(&self) -> ProjectionDriftState {
        self.state
    }

    pub fn authoritative_source_version(&self) -> Option<&str> {
        self.authoritative_source_version.as_deref()
    }

    pub fn projection_source_version(&self) -> Option<&str> {
        self.projection_source_version.as_deref()
    }

    pub const fn authoritative_count(&self) -> usize {
        self.authoritative_count
    }

    pub const fn projection_count(&self) -> usize {
        self.projection_count
    }

    pub const fn authoritative_hash(&self) -> &[u8; 32] {
        &self.authoritative_hash
    }

    pub const fn projection_hash(&self) -> &[u8; 32] {
        &self.projection_hash
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectionRebuildError {
    #[error("projection rebuild requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("projection rebuild input is invalid or exceeds a bound")]
    InvalidInput,
    #[error("projection adapter rejected the rebuild")]
    AdapterRejected,
    #[error("projection rebuild storage is unavailable")]
    Unavailable(#[source] DbErr),
}

#[async_trait]
pub trait ProjectionRebuildAdapter: Send + Sync {
    async fn load_authority(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionRows, ProjectionRebuildError>;

    async fn replace_projection(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
        rows: &ProjectionRows,
    ) -> Result<(), ProjectionRebuildError>;

    async fn load_projection(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionMaterialization, ProjectionRebuildError>;
}

pub struct ProjectionRebuilder<A> {
    connection: DatabaseConnection,
    adapter: A,
}

impl<A> ProjectionRebuilder<A>
where
    A: ProjectionRebuildAdapter,
{
    pub fn new(connection: DatabaseConnection, adapter: A) -> Result<Self, ProjectionRebuildError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(ProjectionRebuildError::UnsupportedDatabase);
        }
        Ok(Self {
            connection,
            adapter,
        })
    }

    pub async fn rebuild(
        &self,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionRebuildDiagnostic, ProjectionRebuildError> {
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        let result = self
            .rebuild_in_transaction(&transaction, tenant, source)
            .await;
        finish_transaction(transaction, result).await
    }

    async fn rebuild_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionRebuildDiagnostic, ProjectionRebuildError> {
        set_tenant(transaction, tenant.community_id()).await?;
        let authoritative_rows = self
            .adapter
            .load_authority(transaction, tenant, source)
            .await?;
        self.adapter
            .replace_projection(transaction, tenant, source, &authoritative_rows)
            .await?;
        let projection = self
            .adapter
            .load_projection(transaction, tenant, source)
            .await?;
        let authoritative_hash = materialization_hash(
            source.provenance(),
            source.provenance().source_version.as_deref(),
            &authoritative_rows,
        );
        let projection_hash = materialization_hash(
            source.provenance(),
            projection.source_version(),
            projection.rows(),
        );
        let state = if authoritative_hash == projection_hash {
            ProjectionDriftState::Clean
        } else {
            ProjectionDriftState::Diverged
        };
        let diagnostic = ProjectionRebuildDiagnostic {
            community_id: tenant.community_id(),
            projection_name: source.projection_name.clone(),
            source_record_id: source.provenance.source_record_id.as_str().to_owned(),
            state,
            authoritative_source_version: source.provenance.source_version.clone(),
            projection_source_version: projection.source_version.clone(),
            authoritative_count: authoritative_rows.len(),
            projection_count: projection.rows.len(),
            authoritative_hash,
            projection_hash,
        };
        persist_checkpoint(transaction, tenant, source, &diagnostic).await?;
        Ok(diagnostic)
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), ProjectionRebuildError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(ProjectionRebuildError::Unavailable)?;
    Ok(())
}

async fn persist_checkpoint(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    source: &ProjectionSource,
    diagnostic: &ProjectionRebuildDiagnostic,
) -> Result<(), ProjectionRebuildError> {
    let provenance = source.provenance();
    let observed_at = i64::try_from(provenance.observed_at_millis)
        .map_err(|_| ProjectionRebuildError::InvalidInput)?;
    let integrity_algorithm = provenance
        .integrity
        .as_ref()
        .map(|integrity| integrity_algorithm_name(integrity.algorithm));
    let integrity_value = provenance
        .integrity
        .as_ref()
        .map(|integrity| integrity.value.as_str());
    let drift_summary = (diagnostic.state == ProjectionDriftState::Diverged).then(|| {
        format!(
            "projection drift: source_version_match={} authority_count={} projection_count={}",
            diagnostic.authoritative_source_version == diagnostic.projection_source_version,
            diagnostic.authoritative_count,
            diagnostic.projection_count,
        )
    });
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            UPSERT_CHECKPOINT_SQL,
            [
                Value::Uuid(Some(Box::new(tenant.community_id().as_uuid()))),
                source.projection_name().into(),
                source_system_name(provenance.source_system).into(),
                provenance.source_record_id.as_str().into(),
                provenance.source_version.as_deref().into(),
                observed_at.into(),
                integrity_algorithm.into(),
                integrity_value.into(),
                diagnostic.state.database_name().into(),
                diagnostic.authoritative_hash.to_vec().into(),
                diagnostic.projection_hash.to_vec().into(),
                drift_summary.as_deref().into(),
            ],
        ))
        .await
        .map_err(ProjectionRebuildError::Unavailable)?;
    if result.rows_affected() != 1 {
        return Err(ProjectionRebuildError::AdapterRejected);
    }
    Ok(())
}

fn materialization_hash(
    provenance: &Provenance,
    source_version: Option<&str>,
    rows: &ProjectionRows,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_field(
        &mut hasher,
        source_system_name(provenance.source_system).as_bytes(),
    );
    hash_field(&mut hasher, provenance.source_record_id.as_str().as_bytes());
    hash_optional_field(&mut hasher, source_version.map(str::as_bytes));
    hash_field(
        &mut hasher,
        &u64::try_from(rows.len()).unwrap_or(u64::MAX).to_be_bytes(),
    );
    for row in rows.as_slice() {
        hash_field(&mut hasher, row.key().as_bytes());
        hash_field(&mut hasher, row.payload());
    }
    hasher.finalize().into()
}

fn hash_optional_field(hasher: &mut Sha256, value: Option<&[u8]>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_field(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_be_bytes());
    hasher.update(value);
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, ProjectionRebuildError>,
) -> Result<T, ProjectionRebuildError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(ProjectionRebuildError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            if let Err(rollback_error) = transaction.rollback().await {
                log::error!("failed to roll back projection rebuild: {rollback_error}");
            }
            Err(error)
        }
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
