use std::{error::Error, fmt, sync::Arc};

use collaboration_domain::{
    AggregateId, AggregateVersion, AuthorizationRequest, BranchCollaboration,
    BranchCollaborationIdentity, BranchLifecycleState, Channel, ChannelCreateFields,
    ChannelDescription, ChannelError, ChannelLifecycleState, ChannelName, ChannelRecordFields,
    ChannelType, ChannelVisibility, CommunityId, PrincipalId,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const BRANCH_CHANNEL_NAMESPACE: Uuid = Uuid::from_u128(0x4f94f0ab_4ad8_58df_9fd4_32aad3f2db8c);
const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const INSERT_CHANNEL_SQL: &str = r#"
INSERT INTO public.collaboration_channels (
    community_id, channel_id, name, channel_type, visibility, lifecycle_state,
    description, creator_principal_id, channel_version, source_system, source_record_id,
    source_version, source_observed_at, integrity_algorithm, integrity_value,
    created_at, updated_at
) VALUES (
    $1, $2, $3, 'stream', 'private', 'active', $4, $5, 1, 'zed', $6, $7,
    to_timestamp($8::double precision / 1000), 'sha256', $9,
    to_timestamp($8::double precision / 1000),
    to_timestamp($8::double precision / 1000)
)
ON CONFLICT (community_id, channel_id) DO NOTHING
"#;
pub(super) const SELECT_CHANNEL_SQL: &str = r#"
SELECT community_id, channel_id, name, channel_type, visibility, lifecycle_state,
       description, creator_principal_id, channel_version::bigint AS channel_version,
       ttl_seconds IS NOT NULL AS has_ttl,
       expires_at IS NOT NULL AS has_expiration,
       source_system, source_record_id, source_version,
       integrity_algorithm, integrity_value
FROM public.collaboration_channels
WHERE community_id = $1 AND channel_id = $2
"#;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BranchChannelKey {
    pub(super) channel_id: AggregateId,
    pub(super) source_record_id: String,
    pub(super) source_version: String,
    pub(super) integrity_value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchChannelBinding {
    branch: BranchCollaborationIdentity,
    channel: Channel,
}

impl BranchChannelBinding {
    pub const fn branch(&self) -> &BranchCollaborationIdentity {
        &self.branch
    }

    pub const fn channel(&self) -> &Channel {
        &self.channel
    }

    pub fn channel_id(&self) -> AggregateId {
        self.channel.fields().channel_id
    }
}

#[derive(Clone)]
pub struct BranchChannelService {
    connection: Arc<DatabaseConnection>,
}

impl BranchChannelService {
    pub fn new(connection: DatabaseConnection) -> Result<Self, BranchChannelError> {
        Self::from_shared(Arc::new(connection))
    }

    pub(super) fn from_shared(
        connection: Arc<DatabaseConnection>,
    ) -> Result<Self, BranchChannelError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(BranchChannelError::UnsupportedBackend);
        }
        Ok(Self { connection })
    }

    pub async fn bind(
        &self,
        branch: &BranchCollaboration,
        creator_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchChannelBinding, BranchChannelError> {
        let (key, proposed_channel) = prepare_binding(branch, creator_principal_id, authorization)?;
        let transaction = self
            .connection
            .as_ref()
            .begin()
            .await
            .map_err(BranchChannelError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, branch.fields().identity.community_id()).await?;
            self.bind_prepared_in_transaction(
                &transaction,
                branch,
                &key,
                &proposed_channel,
                authorization.now_millis,
                false,
            )
            .await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub(super) async fn bind_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        branch: &BranchCollaboration,
        creator_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchChannelBinding, BranchChannelError> {
        let (key, proposed_channel) = prepare_binding(branch, creator_principal_id, authorization)?;
        self.bind_prepared_in_transaction(
            transaction,
            branch,
            &key,
            &proposed_channel,
            authorization.now_millis,
            false,
        )
        .await
    }

    pub(super) async fn resolve_for_activity(
        &self,
        branch: &BranchCollaboration,
        creator_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<BranchChannelBinding, BranchChannelError> {
        let (key, proposed_channel) = prepare_binding(branch, creator_principal_id, authorization)?;
        let transaction = self
            .connection
            .as_ref()
            .begin()
            .await
            .map_err(BranchChannelError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, branch.fields().identity.community_id()).await?;
            self.bind_prepared_in_transaction(
                &transaction,
                branch,
                &key,
                &proposed_channel,
                authorization.now_millis,
                true,
            )
            .await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn bind_prepared_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        branch: &BranchCollaboration,
        key: &BranchChannelKey,
        proposed_channel: &Channel,
        now_millis: u64,
        allow_archived: bool,
    ) -> Result<BranchChannelBinding, BranchChannelError> {
        insert_channel(transaction, proposed_channel, key, now_millis).await?;
        let channel =
            select_channel(transaction, branch.fields().identity.community_id(), key).await?;
        if channel.fields().lifecycle_state != ChannelLifecycleState::Active
            && !(allow_archived
                && channel.fields().lifecycle_state == ChannelLifecycleState::Archived)
        {
            return Err(BranchChannelError::BindingConflict);
        }
        Ok(BranchChannelBinding {
            branch: branch.fields().identity.clone(),
            channel,
        })
    }

    pub(super) fn connection(&self) -> &DatabaseConnection {
        self.connection.as_ref()
    }
}

fn prepare_binding(
    branch: &BranchCollaboration,
    creator_principal_id: PrincipalId,
    authorization: &AuthorizationRequest<'_>,
) -> Result<(BranchChannelKey, Channel), BranchChannelError> {
    if branch.fields().lifecycle_state != BranchLifecycleState::Active {
        return Err(BranchChannelError::BranchUnavailable);
    }
    let key = branch_channel_key(&branch.fields().identity)?;
    let proposed_channel = Channel::create(
        ChannelCreateFields {
            community_id: branch.fields().identity.community_id(),
            channel_id: key.channel_id,
            name: branch_channel_name(&branch.fields().identity, key.channel_id)?,
            channel_type: ChannelType::Stream,
            visibility: ChannelVisibility::Private,
            description: Some(
                ChannelDescription::new(format!(
                    "Branch conversation for {}",
                    branch.fields().identity.branch_ref().as_str()
                ))
                .map_err(BranchChannelError::Domain)?,
            ),
            creator_principal_id,
            ttl_seconds: None,
            now_millis: authorization.now_millis,
        },
        authorization,
    )?;
    Ok((key, proposed_channel))
}

pub fn branch_channel_id(
    identity: &BranchCollaborationIdentity,
) -> Result<AggregateId, BranchChannelError> {
    branch_channel_key(identity).map(|key| key.channel_id)
}

pub(super) fn branch_channel_key(
    identity: &BranchCollaborationIdentity,
) -> Result<BranchChannelKey, BranchChannelError> {
    let branch = identity.branch_ref().as_str().as_bytes();
    let branch_length =
        u32::try_from(branch.len()).map_err(|_| BranchChannelError::InvalidRecord)?;
    let mut canonical = Vec::with_capacity(32 + 16 + 16 + 8 + 4 + branch.len());
    canonical.extend_from_slice(b"collaboration-branch-channel/v1\0");
    canonical.extend_from_slice(identity.community_id().as_uuid().as_bytes());
    canonical.extend_from_slice(identity.repository_id().as_uuid().as_bytes());
    canonical.extend_from_slice(&identity.generation().get().to_be_bytes());
    canonical.extend_from_slice(&branch_length.to_be_bytes());
    canonical.extend_from_slice(branch);
    let channel_uuid = Uuid::new_v5(&BRANCH_CHANNEL_NAMESPACE, &canonical);
    let integrity_value = hex::encode(Sha256::digest(&canonical));
    Ok(BranchChannelKey {
        channel_id: AggregateId::from_uuid(channel_uuid),
        source_record_id: format!("branch-channel:v1:{channel_uuid}"),
        source_version: identity.generation().get().to_string(),
        integrity_value,
    })
}

fn branch_channel_name(
    identity: &BranchCollaborationIdentity,
    channel_id: AggregateId,
) -> Result<ChannelName, BranchChannelError> {
    let short_name = identity
        .branch_ref()
        .as_str()
        .strip_prefix("refs/heads/")
        .ok_or(BranchChannelError::InvalidRecord)?;
    let mut name = format!("branch/{short_name}");
    if name.len() > 255 {
        let suffix = format!(
            "-{}",
            channel_id.to_string().chars().take(12).collect::<String>()
        );
        let mut end = 255usize.saturating_sub(suffix.len());
        while !name.is_char_boundary(end) {
            end = end.saturating_sub(1);
        }
        name.truncate(end);
        name.push_str(&suffix);
    }
    ChannelName::new(name).map_err(BranchChannelError::Domain)
}

async fn insert_channel(
    transaction: &DatabaseTransaction,
    channel: &Channel,
    key: &BranchChannelKey,
    now_millis: u64,
) -> Result<(), BranchChannelError> {
    let fields = channel.fields();
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            INSERT_CHANNEL_SQL,
            [
                fields.community_id.as_uuid().into(),
                fields.channel_id.as_uuid().into(),
                fields.name.as_str().to_owned().into(),
                fields
                    .description
                    .as_ref()
                    .map(|description| description.as_str().to_owned())
                    .into(),
                fields.creator_principal_id.as_uuid().into(),
                key.source_record_id.clone().into(),
                key.source_version.clone().into(),
                millis_i64(now_millis)?.into(),
                key.integrity_value.clone().into(),
            ],
        ))
        .await
        .map_err(BranchChannelError::Unavailable)?;
    Ok(())
}

async fn select_channel(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    key: &BranchChannelKey,
) -> Result<Channel, BranchChannelError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_CHANNEL_SQL,
            [
                community_id.as_uuid().into(),
                key.channel_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(BranchChannelError::Unavailable)?
        .ok_or(BranchChannelError::InvalidRecord)?;
    channel_from_row(&row, community_id, key)
}

pub(super) fn channel_from_row(
    row: &QueryResult,
    expected_community_id: CommunityId,
    key: &BranchChannelKey,
) -> Result<Channel, BranchChannelError> {
    let community_id = CommunityId::from_uuid(row_value(row, "community_id")?);
    let channel_id = AggregateId::from_uuid(row_value(row, "channel_id")?);
    let source_system: String = row_value(row, "source_system")?;
    let source_record_id: String = row_value(row, "source_record_id")?;
    let source_version: Option<String> = row_value(row, "source_version")?;
    let integrity_algorithm: Option<String> = row_value(row, "integrity_algorithm")?;
    let integrity_value: Option<String> = row_value(row, "integrity_value")?;
    let channel_type: String = row_value(row, "channel_type")?;
    let visibility: String = row_value(row, "visibility")?;
    let lifecycle_state: String = row_value(row, "lifecycle_state")?;
    let has_ttl: bool = row_value(row, "has_ttl")?;
    let has_expiration: bool = row_value(row, "has_expiration")?;
    if community_id != expected_community_id
        || channel_id != key.channel_id
        || source_system != "zed"
        || source_record_id != key.source_record_id
        || source_version.as_deref() != Some(key.source_version.as_str())
        || integrity_algorithm.as_deref() != Some("sha256")
        || integrity_value.as_deref() != Some(key.integrity_value.as_str())
        || channel_type != "stream"
        || visibility != "private"
        || has_ttl
        || has_expiration
    {
        return Err(BranchChannelError::BindingConflict);
    }
    let lifecycle_state = match lifecycle_state.as_str() {
        "active" => ChannelLifecycleState::Active,
        "archived" => ChannelLifecycleState::Archived,
        "deleted" => ChannelLifecycleState::Deleted,
        "expired" => ChannelLifecycleState::Expired,
        _ => return Err(BranchChannelError::InvalidRecord),
    };
    let creator_principal_id = PrincipalId::from_uuid(row_value(row, "creator_principal_id")?);
    if creator_principal_id.as_uuid().is_nil() {
        return Err(BranchChannelError::InvalidRecord);
    }
    Channel::from_record(ChannelRecordFields {
        community_id,
        channel_id,
        name: ChannelName::new(row_value::<String>(row, "name")?)?,
        channel_type: ChannelType::Stream,
        visibility: ChannelVisibility::Private,
        lifecycle_state,
        description: row_value::<Option<String>>(row, "description")?
            .map(ChannelDescription::new)
            .transpose()?,
        creator_principal_id,
        expiration: None,
        version: aggregate_version(row_value(row, "channel_version")?)?,
    })
    .map_err(BranchChannelError::Domain)
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), BranchChannelError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(BranchChannelError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, BranchChannelError>,
) -> Result<T, BranchChannelError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(BranchChannelError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(BranchChannelError::Unavailable)?;
            Err(error)
        }
    }
}

fn millis_i64(value: u64) -> Result<i64, BranchChannelError> {
    i64::try_from(value).map_err(|_| BranchChannelError::InvalidRecord)
}

fn aggregate_version(value: i64) -> Result<AggregateVersion, BranchChannelError> {
    u64::try_from(value)
        .ok()
        .and_then(AggregateVersion::new)
        .ok_or(BranchChannelError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, BranchChannelError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| BranchChannelError::InvalidRecord)
}

#[derive(Debug)]
pub enum BranchChannelError {
    UnsupportedBackend,
    BranchUnavailable,
    BindingConflict,
    InvalidRecord,
    Domain(ChannelError),
    Unavailable(DbErr),
}

impl From<ChannelError> for BranchChannelError {
    fn from(error: ChannelError) -> Self {
        Self::Domain(error)
    }
}

impl fmt::Display for BranchChannelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend => formatter.write_str("branch channels require PostgreSQL"),
            Self::BranchUnavailable => formatter.write_str("branch is not active"),
            Self::BindingConflict => formatter.write_str("branch channel binding conflicts"),
            Self::InvalidRecord => formatter.write_str("branch channel record is invalid"),
            Self::Domain(_) => formatter.write_str("branch channel command is not authorized"),
            Self::Unavailable(_) => formatter.write_str("branch channel storage is unavailable"),
        }
    }
}

impl Error for BranchChannelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Unavailable(error) => Some(error),
            Self::UnsupportedBackend
            | Self::BranchUnavailable
            | Self::BindingConflict
            | Self::InvalidRecord => None,
        }
    }
}
