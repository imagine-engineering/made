use collaboration_domain::{CommunityId, PrincipalId, TenantContext};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    Statement, TransactionTrait,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const SUPPORTED_BUZZ_SCHEMA_VERSION: u32 = 30;
const MAX_IMPORT_BATCH_SIZE: usize = 1_000;
const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzCommunityRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub host: String,
    pub icon: Option<String>,
    pub lifecycle_state: String,
    pub join_policy_version: Option<String>,
    pub aggregate_version: u64,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzCommunityMembershipRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub principal_id: PrincipalId,
    pub public_key: String,
    pub role: String,
    pub status: String,
    pub membership_version: u64,
    pub added_by_principal_id: Option<PrincipalId>,
    pub added_by_public_key: Option<String>,
    pub joined_at_millis: u64,
    pub updated_at_millis: u64,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzJoinPolicyAcceptanceRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub principal_id: PrincipalId,
    pub public_key: String,
    pub policy_version: String,
    pub accepted_at_millis: u64,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzChannelRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub channel_id: Uuid,
    pub name: String,
    pub channel_type: String,
    pub visibility: String,
    pub lifecycle_state: String,
    pub description: Option<String>,
    pub creator_principal_id: PrincipalId,
    pub creator_public_key: String,
    pub nip29_group_id: Option<String>,
    pub topic_required: bool,
    pub max_members: Option<u32>,
    pub ttl_seconds: Option<u32>,
    pub expires_at_millis: Option<u64>,
    pub channel_version: u64,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzInviteRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub invite_id: Uuid,
    pub channel_id: Option<Uuid>,
    pub token_hash: [u8; 32],
    pub role: String,
    pub status: String,
    pub max_uses: Option<u32>,
    pub use_count: u32,
    pub expires_at_millis: u64,
    pub created_by_principal_id: PrincipalId,
    pub created_by_source_identity: String,
    pub invite_version: u64,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzChannelMembershipRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub channel_id: Uuid,
    pub principal_id: PrincipalId,
    pub public_key: String,
    pub role: String,
    pub status: String,
    pub membership_version: u64,
    pub invited_by_principal_id: Option<PrincipalId>,
    pub invited_by_public_key: Option<String>,
    pub joined_at_millis: u64,
    pub updated_at_millis: u64,
    pub hidden_at_millis: Option<u64>,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzCommunityStateRecord {
    Community(BuzzCommunityRecord),
    CommunityMembership(BuzzCommunityMembershipRecord),
    JoinPolicyAcceptance(BuzzJoinPolicyAcceptanceRecord),
    Channel(BuzzChannelRecord),
    Invite(BuzzInviteRecord),
    ChannelMembership(BuzzChannelMembershipRecord),
}

impl BuzzCommunityStateRecord {
    fn community_id(&self) -> CommunityId {
        match self {
            Self::Community(record) => record.community_id,
            Self::CommunityMembership(record) => record.community_id,
            Self::JoinPolicyAcceptance(record) => record.community_id,
            Self::Channel(record) => record.community_id,
            Self::Invite(record) => record.community_id,
            Self::ChannelMembership(record) => record.community_id,
        }
    }

    fn source_sequence(&self) -> u64 {
        match self {
            Self::Community(record) => record.source_sequence,
            Self::CommunityMembership(record) => record.source_sequence,
            Self::JoinPolicyAcceptance(record) => record.source_sequence,
            Self::Channel(record) => record.source_sequence,
            Self::Invite(record) => record.source_sequence,
            Self::ChannelMembership(record) => record.source_sequence,
        }
    }

    fn observed_at_millis(&self) -> u64 {
        match self {
            Self::Community(record) => record.observed_at_millis,
            Self::CommunityMembership(record) => record.observed_at_millis,
            Self::JoinPolicyAcceptance(record) => record.observed_at_millis,
            Self::Channel(record) => record.observed_at_millis,
            Self::Invite(record) => record.observed_at_millis,
            Self::ChannelMembership(record) => record.observed_at_millis,
        }
    }

    fn source_record_id(&self) -> String {
        match self {
            Self::Community(record) => format!("communities:{}", record.community_id),
            Self::CommunityMembership(record) => {
                format!("relay_members:{}", record.public_key)
            }
            Self::JoinPolicyAcceptance(record) => format!(
                "join_policy_acceptances:{}:{}",
                record.public_key, record.policy_version
            ),
            Self::Channel(record) => format!("channels:{}", record.channel_id),
            Self::Invite(record) => format!("relay_invites:{}", record.invite_id),
            Self::ChannelMembership(record) => {
                format!(
                    "channel_members:{}:{}",
                    record.channel_id, record.public_key
                )
            }
        }
    }

    fn validate(&self) -> Result<(), BuzzCommunityStateImportError> {
        if self.source_sequence() == 0
            || self.source_record_id().len() > 1024
            || !valid_timestamp(self.observed_at_millis())
        {
            return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
        }
        match self {
            Self::Community(record) => validate_community(record),
            Self::CommunityMembership(record) => validate_community_membership(record),
            Self::JoinPolicyAcceptance(record) => validate_policy_acceptance(record),
            Self::Channel(record) => validate_channel(record),
            Self::Invite(record) => validate_invite(record),
            Self::ChannelMembership(record) => validate_channel_membership(record),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzCommunityStateBatch {
    schema_version: u32,
    records: Vec<BuzzCommunityStateRecord>,
}

impl BuzzCommunityStateBatch {
    pub fn new(
        schema_version: u32,
        records: Vec<BuzzCommunityStateRecord>,
    ) -> Result<Self, BuzzCommunityStateImportError> {
        if schema_version != SUPPORTED_BUZZ_SCHEMA_VERSION {
            return Err(BuzzCommunityStateImportError::UnsupportedSchemaVersion(
                schema_version,
            ));
        }
        if records.is_empty() || records.len() > MAX_IMPORT_BATCH_SIZE {
            return Err(BuzzCommunityStateImportError::InvalidBatch);
        }
        let mut previous_sequence = None;
        for record in &records {
            record.validate()?;
            if previous_sequence.is_some_and(|previous| record.source_sequence() <= previous) {
                return Err(BuzzCommunityStateImportError::InvalidBatch);
            }
            previous_sequence = Some(record.source_sequence());
        }
        Ok(Self {
            schema_version,
            records,
        })
    }

    pub fn records(&self) -> &[BuzzCommunityStateRecord] {
        &self.records
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzCommunityStateImportResult {
    pub scanned: u64,
    pub inserted: u64,
    pub duplicates: u64,
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub target_hash: [u8; 32],
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzCommunityStateImportError {
    #[error("Buzz community-state import requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("Buzz community-state schema version {0} is unsupported")]
    UnsupportedSchemaVersion(u32),
    #[error("Buzz community-state source record is invalid")]
    InvalidSourceRecord,
    #[error("Buzz community-state batch is empty, oversized or out of order")]
    InvalidBatch,
    #[error("Buzz community-state import crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("an existing canonical community-state row differs from Buzz")]
    IntegrityConflict,
    #[error("Buzz community-state import storage is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct BuzzCommunityStateImporter {
    connection: DatabaseConnection,
}

impl BuzzCommunityStateImporter {
    pub fn new(connection: DatabaseConnection) -> Result<Self, BuzzCommunityStateImportError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(BuzzCommunityStateImportError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn import_batch(
        &self,
        tenant: &TenantContext,
        batch: &BuzzCommunityStateBatch,
    ) -> Result<BuzzCommunityStateImportResult, BuzzCommunityStateImportError> {
        if batch
            .records
            .iter()
            .any(|record| record.community_id() != tenant.community_id())
        {
            return Err(BuzzCommunityStateImportError::TenantBoundaryViolation);
        }
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(BuzzCommunityStateImportError::Unavailable)?;
        let result = import_in_transaction(&transaction, tenant, batch).await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }
}

async fn import_in_transaction(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    batch: &BuzzCommunityStateBatch,
) -> Result<BuzzCommunityStateImportResult, BuzzCommunityStateImportError> {
    set_tenant(transaction, tenant.community_id()).await?;
    let source_version = batch.schema_version.to_string();
    let mut source_hasher = Sha256::new();
    let mut target_hasher = Sha256::new();
    let mut inserted = 0_u64;
    let mut duplicates = 0_u64;
    for record in &batch.records {
        let integrity_hash = record_hash(record)?;
        hash_part(&mut source_hasher, &record.source_sequence().to_be_bytes());
        hash_part(&mut source_hasher, &integrity_hash);
        let result = transaction
            .execute(insert_statement(record, &source_version, integrity_hash)?)
            .await
            .map_err(BuzzCommunityStateImportError::Unavailable)?;
        if result.rows_affected() == 1 {
            inserted += 1;
        } else {
            duplicates += 1;
        }
        let target_hash = read_integrity_hash(
            transaction,
            record,
            &source_version,
            hex::encode(integrity_hash),
        )
        .await?;
        hash_part(&mut target_hasher, &record.source_sequence().to_be_bytes());
        hash_part(&mut target_hasher, &target_hash);
    }
    Ok(BuzzCommunityStateImportResult {
        scanned: u64::try_from(batch.records.len())
            .map_err(|_| BuzzCommunityStateImportError::InvalidBatch)?,
        inserted,
        duplicates,
        final_source_sequence: batch
            .records
            .last()
            .map(BuzzCommunityStateRecord::source_sequence)
            .ok_or(BuzzCommunityStateImportError::InvalidBatch)?,
        source_hash: source_hasher.finalize().into(),
        target_hash: target_hasher.finalize().into(),
    })
}

fn insert_statement(
    record: &BuzzCommunityStateRecord,
    source_version: &str,
    integrity_hash: [u8; 32],
) -> Result<Statement, BuzzCommunityStateImportError> {
    let source_record_id = record.source_record_id();
    let integrity_value = hex::encode(integrity_hash);
    let observed_at = millis(record.observed_at_millis())?;
    let statement = match record {
        BuzzCommunityStateRecord::Community(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_communities (community_id, host, icon, lifecycle_state, join_policy_version, aggregate_version, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value, created_at, updated_at) VALUES ($1,$2,$3,$4,$5,CAST($6 AS numeric),'buzz',$7,$8,to_timestamp($9::double precision / 1000),'sha256',$10,to_timestamp($11::double precision / 1000),to_timestamp($12::double precision / 1000)) ON CONFLICT (community_id) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.host.clone().into(),
                record.icon.clone().into(),
                record.lifecycle_state.clone().into(),
                record.join_policy_version.clone().into(),
                record.aggregate_version.to_string().into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
                millis(record.created_at_millis)?.into(),
                millis(record.updated_at_millis)?.into(),
            ],
        ),
        BuzzCommunityStateRecord::CommunityMembership(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, added_by_principal_id, joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value) VALUES ($1,$2,$3,$4,CAST($5 AS numeric),$6,to_timestamp($7::double precision / 1000),to_timestamp($8::double precision / 1000),'buzz',$9,$10,to_timestamp($11::double precision / 1000),'sha256',$12) ON CONFLICT (community_id, principal_id) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.principal_id.as_uuid().into(),
                record.role.clone().into(),
                record.status.clone().into(),
                record.membership_version.to_string().into(),
                record
                    .added_by_principal_id
                    .map(PrincipalId::as_uuid)
                    .into(),
                millis(record.joined_at_millis)?.into(),
                millis(record.updated_at_millis)?.into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
            ],
        ),
        BuzzCommunityStateRecord::JoinPolicyAcceptance(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_join_policy_acceptances (community_id, principal_id, policy_version, accepted_at, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value) VALUES ($1,$2,$3,to_timestamp($4::double precision / 1000),'buzz',$5,$6,to_timestamp($7::double precision / 1000),'sha256',$8) ON CONFLICT (community_id, principal_id, policy_version) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.principal_id.as_uuid().into(),
                record.policy_version.clone().into(),
                millis(record.accepted_at_millis)?.into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
            ],
        ),
        BuzzCommunityStateRecord::Channel(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_channels (community_id, channel_id, name, channel_type, visibility, lifecycle_state, description, creator_principal_id, nip29_group_id, topic_required, max_members, ttl_seconds, expires_at, channel_version, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value, created_at, updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,CASE WHEN $13::bigint IS NULL THEN NULL ELSE to_timestamp($13::double precision / 1000) END,CAST($14 AS numeric),'buzz',$15,$16,to_timestamp($17::double precision / 1000),'sha256',$18,to_timestamp($19::double precision / 1000),to_timestamp($20::double precision / 1000)) ON CONFLICT (community_id, channel_id) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.channel_id.into(),
                record.name.clone().into(),
                record.channel_type.clone().into(),
                record.visibility.clone().into(),
                record.lifecycle_state.clone().into(),
                record.description.clone().into(),
                record.creator_principal_id.as_uuid().into(),
                record.nip29_group_id.clone().into(),
                record.topic_required.into(),
                record.max_members.map(i64::from).into(),
                record.ttl_seconds.map(i64::from).into(),
                optional_millis(record.expires_at_millis)?.into(),
                record.channel_version.to_string().into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
                millis(record.created_at_millis)?.into(),
                millis(record.updated_at_millis)?.into(),
            ],
        ),
        BuzzCommunityStateRecord::Invite(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_channel_invites (community_id, invite_id, channel_id, token_hash, role, status, max_uses, use_count, expires_at, created_by_principal_id, invite_version, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value, created_at, updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,to_timestamp($9::double precision / 1000),$10,CAST($11 AS numeric),'buzz',$12,$13,to_timestamp($14::double precision / 1000),'sha256',$15,to_timestamp($16::double precision / 1000),to_timestamp($17::double precision / 1000)) ON CONFLICT (community_id, invite_id) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.invite_id.into(),
                record.channel_id.into(),
                record.token_hash.to_vec().into(),
                record.role.clone().into(),
                record.status.clone().into(),
                record.max_uses.map(i64::from).into(),
                i64::from(record.use_count).into(),
                millis(record.expires_at_millis)?.into(),
                record.created_by_principal_id.as_uuid().into(),
                record.invite_version.to_string().into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
                millis(record.created_at_millis)?.into(),
                millis(record.updated_at_millis)?.into(),
            ],
        ),
        BuzzCommunityStateRecord::ChannelMembership(record) => Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO public.collaboration_channel_memberships (community_id, channel_id, principal_id, role, status, membership_version, invited_by_principal_id, joined_at, updated_at, hidden_at, source_system, source_record_id, source_version, source_observed_at, integrity_algorithm, integrity_value) VALUES ($1,$2,$3,$4,$5,CAST($6 AS numeric),$7,to_timestamp($8::double precision / 1000),to_timestamp($9::double precision / 1000),CASE WHEN $10::bigint IS NULL THEN NULL ELSE to_timestamp($10::double precision / 1000) END,'buzz',$11,$12,to_timestamp($13::double precision / 1000),'sha256',$14) ON CONFLICT (community_id, channel_id, principal_id) DO NOTHING",
            vec![
                record.community_id.as_uuid().into(),
                record.channel_id.into(),
                record.principal_id.as_uuid().into(),
                record.role.clone().into(),
                record.status.clone().into(),
                record.membership_version.to_string().into(),
                record
                    .invited_by_principal_id
                    .map(PrincipalId::as_uuid)
                    .into(),
                millis(record.joined_at_millis)?.into(),
                millis(record.updated_at_millis)?.into(),
                optional_millis(record.hidden_at_millis)?.into(),
                source_record_id.into(),
                source_version.into(),
                observed_at.into(),
                integrity_value.into(),
            ],
        ),
    };
    Ok(statement)
}

async fn read_integrity_hash(
    transaction: &DatabaseTransaction,
    record: &BuzzCommunityStateRecord,
    source_version: &str,
    expected_integrity: String,
) -> Result<[u8; 32], BuzzCommunityStateImportError> {
    let (table, key_sql, mut values): (&str, &str, Vec<sea_orm::Value>) = match record {
        BuzzCommunityStateRecord::Community(record) => (
            "collaboration_communities",
            "community_id = $1",
            vec![record.community_id.as_uuid().into()],
        ),
        BuzzCommunityStateRecord::CommunityMembership(record) => (
            "collaboration_community_memberships",
            "community_id = $1 AND principal_id = $2",
            vec![
                record.community_id.as_uuid().into(),
                record.principal_id.as_uuid().into(),
            ],
        ),
        BuzzCommunityStateRecord::JoinPolicyAcceptance(record) => (
            "collaboration_join_policy_acceptances",
            "community_id = $1 AND principal_id = $2 AND policy_version = $3",
            vec![
                record.community_id.as_uuid().into(),
                record.principal_id.as_uuid().into(),
                record.policy_version.clone().into(),
            ],
        ),
        BuzzCommunityStateRecord::Channel(record) => (
            "collaboration_channels",
            "community_id = $1 AND channel_id = $2",
            vec![
                record.community_id.as_uuid().into(),
                record.channel_id.into(),
            ],
        ),
        BuzzCommunityStateRecord::Invite(record) => (
            "collaboration_channel_invites",
            "community_id = $1 AND invite_id = $2",
            vec![
                record.community_id.as_uuid().into(),
                record.invite_id.into(),
            ],
        ),
        BuzzCommunityStateRecord::ChannelMembership(record) => (
            "collaboration_channel_memberships",
            "community_id = $1 AND channel_id = $2 AND principal_id = $3",
            vec![
                record.community_id.as_uuid().into(),
                record.channel_id.into(),
                record.principal_id.as_uuid().into(),
            ],
        ),
    };
    let source_index = values.len() + 1;
    let integrity_index = values.len() + 2;
    values.push(source_version.into());
    values.push(expected_integrity.into());
    let sql = format!(
        "SELECT integrity_value FROM public.{table} WHERE {key_sql} AND source_system = 'buzz' AND source_version = ${source_index} AND integrity_algorithm = 'sha256' AND integrity_value = ${integrity_index}"
    );
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            values,
        ))
        .await
        .map_err(BuzzCommunityStateImportError::Unavailable)?
        .ok_or(BuzzCommunityStateImportError::IntegrityConflict)?;
    let value: String = row
        .try_get("", "integrity_value")
        .map_err(|_| BuzzCommunityStateImportError::IntegrityConflict)?;
    let bytes = hex::decode(value).map_err(|_| BuzzCommunityStateImportError::IntegrityConflict)?;
    bytes
        .try_into()
        .map_err(|_| BuzzCommunityStateImportError::IntegrityConflict)
}

fn validate_community(record: &BuzzCommunityRecord) -> Result<(), BuzzCommunityStateImportError> {
    if record.host.is_empty()
        || record.host.len() > 255
        || record.host != record.host.to_ascii_lowercase()
        || record.host.trim() != record.host
        || !matches!(
            record.lifecycle_state.as_str(),
            "active" | "archived" | "quiescing" | "fenced" | "tombstone"
        )
        || record
            .join_policy_version
            .as_ref()
            .is_some_and(|value| value.is_empty() || value.len() > 256)
        || record.aggregate_version == 0
        || record
            .icon
            .as_ref()
            .is_some_and(|value| value.len() > 262_144)
        || !ordered_timestamps(record.created_at_millis, record.updated_at_millis)
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_community_membership(
    record: &BuzzCommunityMembershipRecord,
) -> Result<(), BuzzCommunityStateImportError> {
    if !valid_public_key(&record.public_key)
        || record
            .added_by_public_key
            .as_ref()
            .is_some_and(|value| !valid_public_key(value))
        || !valid_role(&record.role)
        || !valid_membership_status(&record.status)
        || record.membership_version == 0
        || !ordered_timestamps(record.joined_at_millis, record.updated_at_millis)
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_policy_acceptance(
    record: &BuzzJoinPolicyAcceptanceRecord,
) -> Result<(), BuzzCommunityStateImportError> {
    if !valid_public_key(&record.public_key)
        || record.policy_version.len() != 64
        || !record
            .policy_version
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || !valid_timestamp(record.accepted_at_millis)
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_channel(record: &BuzzChannelRecord) -> Result<(), BuzzCommunityStateImportError> {
    if record.channel_id.is_nil()
        || record.name.is_empty()
        || record.name.len() > 255
        || !matches!(
            record.channel_type.as_str(),
            "stream" | "forum" | "dm" | "workflow" | "ephemeral" | "huddle"
        )
        || !matches!(record.visibility.as_str(), "open" | "private")
        || !matches!(
            record.lifecycle_state.as_str(),
            "active" | "archived" | "deleted" | "expired"
        )
        || !valid_public_key(&record.creator_public_key)
        || record
            .description
            .as_ref()
            .is_some_and(|value| value.len() > 65_536)
        || record
            .nip29_group_id
            .as_ref()
            .is_some_and(|value| value.is_empty() || value.len() > 255)
        || record.max_members == Some(0)
        || record.ttl_seconds == Some(0)
        || record.ttl_seconds.is_some() != record.expires_at_millis.is_some()
        || record.channel_version == 0
        || !ordered_timestamps(record.created_at_millis, record.updated_at_millis)
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    if record
        .expires_at_millis
        .is_some_and(|value| !valid_timestamp(value))
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_invite(record: &BuzzInviteRecord) -> Result<(), BuzzCommunityStateImportError> {
    if record.invite_id.is_nil()
        || record.channel_id.is_some_and(|value| value.is_nil())
        || !matches!(record.role.as_str(), "member" | "guest")
        || !matches!(
            record.status.as_str(),
            "active" | "revoked" | "exhausted" | "expired"
        )
        || record
            .max_uses
            .is_some_and(|value| value == 0 || value > 10_000)
        || record
            .max_uses
            .is_some_and(|value| record.use_count > value)
        || record.created_by_source_identity.is_empty()
        || record.created_by_source_identity.len() > 1024
        || record.invite_version == 0
        || !valid_timestamp(record.expires_at_millis)
        || !ordered_timestamps(record.created_at_millis, record.updated_at_millis)
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_channel_membership(
    record: &BuzzChannelMembershipRecord,
) -> Result<(), BuzzCommunityStateImportError> {
    if record.channel_id.is_nil()
        || !valid_public_key(&record.public_key)
        || record
            .invited_by_public_key
            .as_ref()
            .is_some_and(|value| !valid_public_key(value))
        || !valid_role(&record.role)
        || !valid_membership_status(&record.status)
        || record.membership_version == 0
        || !ordered_timestamps(record.joined_at_millis, record.updated_at_millis)
        || record
            .hidden_at_millis
            .is_some_and(|value| !valid_timestamp(value))
    {
        return Err(BuzzCommunityStateImportError::InvalidSourceRecord);
    }
    Ok(())
}

fn valid_role(value: &str) -> bool {
    matches!(value, "owner" | "admin" | "member" | "guest" | "bot")
}

fn valid_membership_status(value: &str) -> bool {
    matches!(value, "active" | "revoked" | "archived")
}

fn valid_public_key(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn ordered_timestamps(created: u64, updated: u64) -> bool {
    valid_timestamp(created) && valid_timestamp(updated) && updated >= created
}

fn valid_timestamp(value: u64) -> bool {
    i64::try_from(value).is_ok()
}

fn millis(value: u64) -> Result<i64, BuzzCommunityStateImportError> {
    i64::try_from(value).map_err(|_| BuzzCommunityStateImportError::InvalidSourceRecord)
}

fn optional_millis(value: Option<u64>) -> Result<Option<i64>, BuzzCommunityStateImportError> {
    value.map(millis).transpose()
}

fn record_hash(
    record: &BuzzCommunityStateRecord,
) -> Result<[u8; 32], BuzzCommunityStateImportError> {
    let bytes = serde_json::to_vec(record)
        .map_err(|_| BuzzCommunityStateImportError::InvalidSourceRecord)?;
    Ok(Sha256::digest(bytes).into())
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), BuzzCommunityStateImportError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(BuzzCommunityStateImportError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, BuzzCommunityStateImportError>,
) -> Result<T, BuzzCommunityStateImportError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(BuzzCommunityStateImportError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(BuzzCommunityStateImportError::Unavailable)?;
            Err(error)
        }
    }
}
