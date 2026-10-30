use std::{error::Error, fmt, sync::Arc};

use collaboration_domain::{
    AggregateVersion, AuthorizationRequest, BranchArchiveReason, BranchCollaboration,
    BranchLifecycleState, Channel, ChannelCommandOutcome,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbErr, Statement,
    TransactionTrait,
};

use super::branch_channel::{
    BranchChannelBinding, BranchChannelError, BranchChannelService, SELECT_CHANNEL_SQL,
    branch_channel_key, channel_from_row,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const ARCHIVE_CHANNEL_SQL: &str = r#"
UPDATE public.collaboration_channels
SET lifecycle_state = 'archived', channel_version = $3,
    updated_at = to_timestamp($4::double precision / 1000)
WHERE community_id = $1 AND channel_id = $2
  AND lifecycle_state = 'active' AND channel_version = $5
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchChannelLifecycleCause {
    Deleted,
    Merged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchChannelLifecycleResult {
    cause: BranchChannelLifecycleCause,
    channel: Channel,
    outcome: ChannelCommandOutcome,
}

impl BranchChannelLifecycleResult {
    pub const fn cause(&self) -> BranchChannelLifecycleCause {
        self.cause
    }

    pub const fn channel(&self) -> &Channel {
        &self.channel
    }

    pub const fn outcome(&self) -> ChannelCommandOutcome {
        self.outcome
    }
}

#[derive(Clone)]
pub struct BranchLifecycleService {
    branch_channels: BranchChannelService,
}

impl BranchLifecycleService {
    pub fn new(connection: DatabaseConnection) -> Result<Self, BranchLifecycleError> {
        Ok(Self {
            branch_channels: BranchChannelService::from_shared(Arc::new(connection))?,
        })
    }

    pub async fn apply_archive_transition(
        &self,
        previous: &BranchCollaboration,
        current: &BranchCollaboration,
        expected_channel_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchChannelLifecycleResult, BranchLifecycleError> {
        let cause = classify_archive_transition(previous, current)?;
        let key = branch_channel_key(&current.fields().identity)?;
        let transaction = self
            .branch_channels
            .connection()
            .begin()
            .await
            .map_err(BranchLifecycleError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, current.fields().identity.community_id()).await?;
            let mut channel = select_channel_for_update(
                &transaction,
                current.fields().identity.community_id(),
                &key,
            )
            .await?;
            let prior_version = channel.fields().version;
            if prior_version != expected_channel_version {
                return Err(BranchLifecycleError::StaleChannel);
            }
            let outcome = channel.archive(expected_channel_version, authorization)?;
            if outcome == ChannelCommandOutcome::Applied {
                let updated = transaction
                    .execute(Statement::from_sql_and_values(
                        DatabaseBackend::Postgres,
                        ARCHIVE_CHANNEL_SQL,
                        [
                            channel.fields().community_id.as_uuid().into(),
                            channel.fields().channel_id.as_uuid().into(),
                            version_i64(channel.fields().version)?.into(),
                            millis_i64(authorization.now_millis)?.into(),
                            version_i64(prior_version)?.into(),
                        ],
                    ))
                    .await
                    .map_err(BranchLifecycleError::Unavailable)?;
                if updated.rows_affected() != 1 {
                    return Err(BranchLifecycleError::StaleChannel);
                }
            }
            Ok(BranchChannelLifecycleResult {
                cause,
                channel,
                outcome,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn reopen(
        &self,
        archived: &BranchCollaboration,
        reopened: &BranchCollaboration,
        creator_principal_id: collaboration_domain::PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchChannelBinding, BranchLifecycleError> {
        validate_reopen(archived, reopened)?;
        let archived_key = branch_channel_key(&archived.fields().identity)?;
        let transaction = self
            .branch_channels
            .connection()
            .begin()
            .await
            .map_err(BranchLifecycleError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, archived.fields().identity.community_id()).await?;
            let channel = select_channel_for_update(
                &transaction,
                archived.fields().identity.community_id(),
                &archived_key,
            )
            .await?;
            if channel.fields().lifecycle_state
                != collaboration_domain::ChannelLifecycleState::Archived
            {
                return Err(BranchLifecycleError::InvalidTransition);
            }
            self.branch_channels
                .bind_in_transaction(&transaction, reopened, creator_principal_id, authorization)
                .await
                .map_err(BranchLifecycleError::Channel)
        }
        .await;
        finish_transaction(transaction, result).await
    }
}

fn classify_archive_transition(
    previous: &BranchCollaboration,
    current: &BranchCollaboration,
) -> Result<BranchChannelLifecycleCause, BranchLifecycleError> {
    let previous_fields = previous.fields();
    let current_fields = current.fields();
    if previous_fields.identity != current_fields.identity
        || !current_fields.version.follows(previous_fields.version)
        || previous_fields.head_commit != current_fields.head_commit
    {
        return Err(BranchLifecycleError::InvalidTransition);
    }
    match (
        previous_fields.lifecycle_state,
        current_fields.lifecycle_state,
    ) {
        (BranchLifecycleState::Active, BranchLifecycleState::Merged)
            if current_fields.merge.is_some() =>
        {
            Ok(BranchChannelLifecycleCause::Merged)
        }
        (
            BranchLifecycleState::Active,
            BranchLifecycleState::Archived(BranchArchiveReason::Deleted),
        ) if current_fields.merge.is_none() => Ok(BranchChannelLifecycleCause::Deleted),
        (
            BranchLifecycleState::Merged,
            BranchLifecycleState::Archived(BranchArchiveReason::Merged),
        ) if previous_fields.merge == current_fields.merge && current_fields.merge.is_some() => {
            Ok(BranchChannelLifecycleCause::Merged)
        }
        _ => Err(BranchLifecycleError::InvalidTransition),
    }
}

fn validate_reopen(
    archived: &BranchCollaboration,
    reopened: &BranchCollaboration,
) -> Result<(), BranchLifecycleError> {
    let archived_identity = &archived.fields().identity;
    let reopened_identity = &reopened.fields().identity;
    if !matches!(
        archived.fields().lifecycle_state,
        BranchLifecycleState::Archived(_)
    ) || reopened.fields().lifecycle_state != BranchLifecycleState::Active
        || reopened.fields().version != AggregateVersion::FIRST
        || reopened.fields().last_head_update.is_some()
        || reopened.fields().merge.is_some()
        || archived_identity.community_id() != reopened_identity.community_id()
        || archived_identity.repository_id() != reopened_identity.repository_id()
        || archived_identity.branch_ref() != reopened_identity.branch_ref()
        || archived_identity.generation().next() != Some(reopened_identity.generation())
    {
        return Err(BranchLifecycleError::InvalidTransition);
    }
    Ok(())
}

async fn select_channel_for_update(
    transaction: &DatabaseTransaction,
    community_id: collaboration_domain::CommunityId,
    key: &super::branch_channel::BranchChannelKey,
) -> Result<Channel, BranchLifecycleError> {
    let sql = format!("{SELECT_CHANNEL_SQL} FOR UPDATE");
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            [
                community_id.as_uuid().into(),
                key.channel_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(BranchLifecycleError::Unavailable)?
        .ok_or(BranchLifecycleError::ChannelMissing)?;
    channel_from_row(&row, community_id, key).map_err(BranchLifecycleError::Channel)
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: collaboration_domain::CommunityId,
) -> Result<(), BranchLifecycleError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(BranchLifecycleError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, BranchLifecycleError>,
) -> Result<T, BranchLifecycleError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(BranchLifecycleError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(BranchLifecycleError::Unavailable)?;
            Err(error)
        }
    }
}

fn millis_i64(value: u64) -> Result<i64, BranchLifecycleError> {
    i64::try_from(value).map_err(|_| BranchLifecycleError::InvalidRecord)
}

fn version_i64(value: AggregateVersion) -> Result<i64, BranchLifecycleError> {
    i64::try_from(value.get()).map_err(|_| BranchLifecycleError::InvalidRecord)
}

#[derive(Debug)]
pub enum BranchLifecycleError {
    InvalidTransition,
    StaleChannel,
    ChannelMissing,
    InvalidRecord,
    Channel(BranchChannelError),
    Domain(collaboration_domain::ChannelError),
    Unavailable(DbErr),
}

impl From<BranchChannelError> for BranchLifecycleError {
    fn from(error: BranchChannelError) -> Self {
        Self::Channel(error)
    }
}

impl From<collaboration_domain::ChannelError> for BranchLifecycleError {
    fn from(error: collaboration_domain::ChannelError) -> Self {
        Self::Domain(error)
    }
}

impl fmt::Display for BranchLifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition => {
                formatter.write_str("branch lifecycle transition is invalid")
            }
            Self::StaleChannel => formatter.write_str("branch channel version is stale"),
            Self::ChannelMissing => formatter.write_str("branch channel is unavailable"),
            Self::InvalidRecord => formatter.write_str("branch lifecycle record is invalid"),
            Self::Channel(_) | Self::Domain(_) => {
                formatter.write_str("branch channel transition was rejected")
            }
            Self::Unavailable(_) => formatter.write_str("branch lifecycle storage is unavailable"),
        }
    }
}

impl Error for BranchLifecycleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Channel(error) => Some(error),
            Self::Domain(error) => Some(error),
            Self::Unavailable(error) => Some(error),
            Self::InvalidTransition
            | Self::StaleChannel
            | Self::ChannelMissing
            | Self::InvalidRecord => None,
        }
    }
}
