use std::collections::BTreeSet;

use collaboration_domain::{
    AggregateId, AuthorizationAction, AuthorizationDecision, AuthorizationDenial,
    AuthorizationRequest, AuthorizationResourceKind, PrincipalId, TenantContext, authorize,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const HIDE_SQL: &str = r#"
UPDATE public.collaboration_channel_memberships AS membership
SET hidden_at = clock_timestamp(),
    updated_at = GREATEST(membership.updated_at, clock_timestamp()),
    membership_version = membership.membership_version + 1
FROM public.collaboration_channels AS channel
WHERE membership.community_id = $1
  AND membership.channel_id = $2
  AND membership.principal_id = $3
  AND membership.status = 'active'
  AND membership.membership_version = CAST($4 AS numeric)
  AND membership.hidden_at IS NULL
  AND channel.community_id = membership.community_id
  AND channel.channel_id = membership.channel_id
  AND channel.channel_type = 'dm'
  AND channel.lifecycle_state = 'active'
RETURNING membership.channel_id, membership.principal_id
"#;
const REOPEN_SQL: &str = r#"
UPDATE public.collaboration_channel_memberships AS membership
SET hidden_at = NULL,
    updated_at = GREATEST(membership.updated_at, clock_timestamp()),
    membership_version = membership.membership_version + 1
FROM public.collaboration_channels AS channel
WHERE membership.community_id = $1
  AND membership.channel_id = $2
  AND membership.principal_id = $3
  AND membership.status = 'active'
  AND membership.membership_version = CAST($4 AS numeric)
  AND membership.hidden_at IS NOT NULL
  AND channel.community_id = membership.community_id
  AND channel.channel_id = membership.channel_id
  AND channel.channel_type = 'dm'
  AND channel.lifecycle_state = 'active'
RETURNING membership.channel_id, membership.principal_id
"#;
const SNAPSHOT_SQL: &str = r#"
SELECT membership.channel_id
FROM public.collaboration_channel_memberships AS membership
JOIN public.collaboration_channels AS channel
  ON channel.community_id = membership.community_id
 AND channel.channel_id = membership.channel_id
WHERE membership.community_id = $1
  AND membership.principal_id = $2
  AND membership.status = 'active'
  AND membership.hidden_at IS NOT NULL
  AND channel.channel_type = 'dm'
  AND channel.lifecycle_state = 'active'
ORDER BY membership.channel_id ASC
"#;

pub struct DmVisibilityAccess<'a> {
    pub authorization: &'a AuthorizationRequest<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmVisibilityMutation {
    Hidden,
    Reopened,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmVisibilitySnapshot {
    hidden_dm_ids: Vec<AggregateId>,
}

impl DmVisibilitySnapshot {
    pub fn hidden_dm_ids(&self) -> &[AggregateId] {
        &self.hidden_dm_ids
    }

    pub fn hidden_count(&self) -> usize {
        self.hidden_dm_ids.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DmVisibilityError {
    #[error("DM visibility requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("DM visibility request is invalid")]
    InvalidRequest,
    #[error("DM visibility is not authorized: {0:?}")]
    Unauthorized(AuthorizationDenial),
    #[error("DM visibility is unavailable")]
    NotAvailable,
    #[error("DM visibility returned an invalid record")]
    InvalidRecord,
    #[error("DM visibility storage is unavailable")]
    StorageUnavailable(#[source] DbErr),
}

pub struct DmVisibilityRepository {
    connection: DatabaseConnection,
}

impl DmVisibilityRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, DmVisibilityError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(DmVisibilityError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn hide(
        &self,
        access: DmVisibilityAccess<'_>,
        dm_id: AggregateId,
    ) -> Result<DmVisibilityMutation, DmVisibilityError> {
        self.mutate(access, dm_id, HIDE_SQL, DmVisibilityMutation::Hidden)
            .await
    }

    pub async fn reopen(
        &self,
        access: DmVisibilityAccess<'_>,
        dm_id: AggregateId,
    ) -> Result<DmVisibilityMutation, DmVisibilityError> {
        self.mutate(access, dm_id, REOPEN_SQL, DmVisibilityMutation::Reopened)
            .await
    }

    pub async fn snapshot(
        &self,
        access: DmVisibilityAccess<'_>,
    ) -> Result<DmVisibilitySnapshot, DmVisibilityError> {
        let viewer_id = authorize_snapshot(&access)?;
        let tenant = access.authorization.tenant;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            let rows = transaction
                .query_all(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SNAPSHOT_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        viewer_id.as_uuid().into(),
                    ],
                ))
                .await
                .map_err(DmVisibilityError::StorageUnavailable)?;
            snapshot_from_rows(rows)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    async fn mutate(
        &self,
        access: DmVisibilityAccess<'_>,
        dm_id: AggregateId,
        sql: &'static str,
        mutation: DmVisibilityMutation,
    ) -> Result<DmVisibilityMutation, DmVisibilityError> {
        let viewer_id = authorize_mutation(&access, dm_id)?;
        let membership_version = access
            .authorization
            .current_channel_membership_version
            .ok_or(DmVisibilityError::InvalidRequest)?;
        let tenant = access.authorization.tenant;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant).await?;
            let row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    sql,
                    [
                        tenant.community_id().as_uuid().into(),
                        dm_id.as_uuid().into(),
                        viewer_id.as_uuid().into(),
                        membership_version.to_string().into(),
                    ],
                ))
                .await
                .map_err(DmVisibilityError::StorageUnavailable)?
                .ok_or(DmVisibilityError::NotAvailable)?;
            validate_mutation_row(row, dm_id, viewer_id)?;
            Ok(mutation)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, DmVisibilityError> {
        self.connection
            .begin()
            .await
            .map_err(DmVisibilityError::StorageUnavailable)
    }
}

fn authorize_mutation(
    access: &DmVisibilityAccess<'_>,
    dm_id: AggregateId,
) -> Result<PrincipalId, DmVisibilityError> {
    if dm_id.as_uuid().is_nil() {
        return Err(DmVisibilityError::InvalidRequest);
    }
    let request = access.authorization;
    if request.delegation.is_some()
        || request.action != AuthorizationAction::Write
        || request.resource.kind != AuthorizationResourceKind::Channel
        || request.resource.resource_id != dm_id
        || request.resource.channel_id != Some(dm_id)
    {
        return Err(DmVisibilityError::InvalidRequest);
    }
    require_authorized(request)?;
    let membership = request
        .channel_membership
        .ok_or(DmVisibilityError::InvalidRequest)?;
    Ok(membership.principal_id)
}

fn authorize_snapshot(access: &DmVisibilityAccess<'_>) -> Result<PrincipalId, DmVisibilityError> {
    let request = access.authorization;
    let community_id = request.tenant.community_id();
    if request.delegation.is_some()
        || request.action != AuthorizationAction::Read
        || request.resource.kind != AuthorizationResourceKind::Community
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.channel_id.is_some()
    {
        return Err(DmVisibilityError::InvalidRequest);
    }
    require_authorized(request)?;
    let membership = request
        .community_membership
        .ok_or(DmVisibilityError::InvalidRequest)?;
    Ok(membership.principal_id)
}

fn require_authorized(request: &AuthorizationRequest<'_>) -> Result<(), DmVisibilityError> {
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(DmVisibilityError::Unauthorized(denial)),
    }
}

fn validate_mutation_row(
    row: QueryResult,
    expected_dm_id: AggregateId,
    expected_viewer_id: PrincipalId,
) -> Result<(), DmVisibilityError> {
    let dm_id = AggregateId::from_uuid(row_value(&row, "channel_id")?);
    let viewer_id = PrincipalId::from_uuid(row_value(&row, "principal_id")?);
    if dm_id != expected_dm_id || viewer_id != expected_viewer_id {
        return Err(DmVisibilityError::InvalidRecord);
    }
    Ok(())
}

fn snapshot_from_rows(rows: Vec<QueryResult>) -> Result<DmVisibilitySnapshot, DmVisibilityError> {
    let mut hidden_dm_ids = BTreeSet::new();
    for row in rows {
        let dm_id = AggregateId::from_uuid(row_value(&row, "channel_id")?);
        if dm_id.as_uuid().is_nil() || !hidden_dm_ids.insert(dm_id) {
            return Err(DmVisibilityError::InvalidRecord);
        }
    }
    Ok(DmVisibilitySnapshot {
        hidden_dm_ids: hidden_dm_ids.into_iter().collect(),
    })
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, DmVisibilityError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| DmVisibilityError::InvalidRecord)
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), DmVisibilityError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(DmVisibilityError::StorageUnavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, DmVisibilityError>,
) -> Result<T, DmVisibilityError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(DmVisibilityError::StorageUnavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(DmVisibilityError::StorageUnavailable)?;
            Err(error)
        }
    }
}
