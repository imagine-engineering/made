use collaboration_domain::{
    AccountBinding, AccountBindingFields, AggregateVersion, BindingId, BindingStatus,
    BindingVerification, BindingVerificationMethod, BindingVersionReference, CommunityId,
    EvidenceReference, NostrPublicKey, OperationId, OrganizationPolicyVersion, PrincipalId,
    ProfileId, ServiceAccountId,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, SqlErr, Statement, TransactionTrait,
};
use uuid::Uuid;

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SELECT_CURRENT_SQL: &str = r#"
SELECT
    community_id,
    binding_id,
    version,
    service_account_id,
    profile_id,
    nostr_public_key,
    status,
    verification_method,
    evidence_reference,
    CASE WHEN verified_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM verified_at) * 1000)::bigint END AS verified_at_millis,
    predecessor_binding_id,
    predecessor_version,
    successor_binding_id,
    successor_version,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
    CASE WHEN activated_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM activated_at) * 1000)::bigint END AS activated_at_millis,
    CASE WHEN terminal_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM terminal_at) * 1000)::bigint END AS terminal_at_millis,
    organization_policy_version,
    actor_principal_id,
    audit_reference
FROM public.collaboration_identity_bindings
WHERE community_id = $1 AND binding_id = $2 AND is_current
"#;
const SELECT_CURRENT_FOR_UPDATE_SQL: &str = r#"
SELECT
    community_id,
    binding_id,
    version,
    service_account_id,
    profile_id,
    nostr_public_key,
    status,
    verification_method,
    evidence_reference,
    CASE WHEN verified_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM verified_at) * 1000)::bigint END AS verified_at_millis,
    predecessor_binding_id,
    predecessor_version,
    successor_binding_id,
    successor_version,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
    CASE WHEN activated_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM activated_at) * 1000)::bigint END AS activated_at_millis,
    CASE WHEN terminal_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM terminal_at) * 1000)::bigint END AS terminal_at_millis,
    organization_policy_version,
    actor_principal_id,
    audit_reference
FROM public.collaboration_identity_bindings
WHERE community_id = $1 AND binding_id = $2 AND is_current
FOR UPDATE
"#;
const CLEAR_CURRENT_SQL: &str = r#"
UPDATE public.collaboration_identity_bindings
SET is_current = false
WHERE community_id = $1 AND binding_id = $2 AND version = $3 AND is_current
"#;
const INSERT_VERSION_SQL: &str = r#"
INSERT INTO public.collaboration_identity_bindings (
    community_id,
    binding_id,
    version,
    is_current,
    service_account_id,
    profile_id,
    nostr_public_key,
    status,
    verification_method,
    evidence_reference,
    verified_at,
    predecessor_binding_id,
    predecessor_version,
    successor_binding_id,
    successor_version,
    created_at,
    activated_at,
    terminal_at,
    organization_policy_version,
    actor_principal_id,
    audit_reference
) VALUES (
    $1, $2, $3, true, $4, $5, $6, $7, $8, $9,
    CASE WHEN $10::bigint IS NULL THEN NULL ELSE to_timestamp($10::double precision / 1000) END,
    $11, $12, $13, $14,
    to_timestamp($15::double precision / 1000),
    CASE WHEN $16::bigint IS NULL THEN NULL ELSE to_timestamp($16::double precision / 1000) END,
    CASE WHEN $17::bigint IS NULL THEN NULL ELSE to_timestamp($17::double precision / 1000) END,
    $18, $19, $20
)
"#;

#[derive(Debug, thiserror::Error)]
pub enum IdentityBindingRepositoryError {
    #[error("identity-binding repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("identity-binding request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("identity-binding optimistic version does not match current state")]
    VersionConflict,
    #[error("identity-binding record is invalid or cannot be represented")]
    InvalidRecord,
    #[error("identity-binding repository is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct IdentityBindingRepository {
    connection: DatabaseConnection,
}

impl IdentityBindingRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, IdentityBindingRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(IdentityBindingRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn current(
        &self,
        community_id: CommunityId,
        binding_id: BindingId,
    ) -> Result<Option<AccountBinding>, IdentityBindingRepositoryError> {
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(IdentityBindingRepositoryError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, community_id).await?;
            let binding = select_current(&transaction, community_id, binding_id, false).await?;
            if binding
                .as_ref()
                .is_some_and(|binding| binding.community_id() != community_id)
            {
                return Err(IdentityBindingRepositoryError::TenantBoundaryViolation);
            }
            Ok(binding)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn append_version(
        &self,
        community_id: CommunityId,
        expected_current_version: Option<AggregateVersion>,
        binding: &AccountBinding,
    ) -> Result<AccountBinding, IdentityBindingRepositoryError> {
        if binding.community_id() != community_id {
            return Err(IdentityBindingRepositoryError::TenantBoundaryViolation);
        }
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(IdentityBindingRepositoryError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, community_id).await?;
            let current =
                select_current(&transaction, community_id, binding.binding_id(), true).await?;
            validate_append(current.as_ref(), expected_current_version, binding)?;
            if let Some(current) = current {
                let result = transaction
                    .execute(Statement::from_sql_and_values(
                        DatabaseBackend::Postgres,
                        CLEAR_CURRENT_SQL,
                        [
                            community_id.as_uuid().into(),
                            binding.binding_id().as_uuid().into(),
                            to_i64(current.version().get())?.into(),
                        ],
                    ))
                    .await
                    .map_err(map_write_error)?;
                if result.rows_affected() != 1 {
                    return Err(IdentityBindingRepositoryError::VersionConflict);
                }
            }
            transaction
                .execute(insert_statement(binding)?)
                .await
                .map_err(map_write_error)?;
            Ok(binding.clone())
        }
        .await;
        finish_transaction(transaction, result).await
    }
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, IdentityBindingRepositoryError>,
) -> Result<T, IdentityBindingRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(IdentityBindingRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(IdentityBindingRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), IdentityBindingRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(IdentityBindingRepositoryError::Unavailable)?;
    Ok(())
}

async fn select_current(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    binding_id: BindingId,
    for_update: bool,
) -> Result<Option<AccountBinding>, IdentityBindingRepositoryError> {
    let statement = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        if for_update {
            SELECT_CURRENT_FOR_UPDATE_SQL
        } else {
            SELECT_CURRENT_SQL
        },
        [community_id.as_uuid().into(), binding_id.as_uuid().into()],
    );
    transaction
        .query_one(statement)
        .await
        .map_err(IdentityBindingRepositoryError::Unavailable)?
        .map(binding_from_row)
        .transpose()
}

fn validate_append(
    current: Option<&AccountBinding>,
    expected_current_version: Option<AggregateVersion>,
    binding: &AccountBinding,
) -> Result<(), IdentityBindingRepositoryError> {
    match (current, expected_current_version) {
        (None, None) if binding.version() == AggregateVersion::FIRST => Ok(()),
        (Some(current), Some(expected))
            if current.version() == expected
                && binding.binding_id() == current.binding_id()
                && binding.version().follows(current.version()) =>
        {
            Ok(())
        }
        _ => Err(IdentityBindingRepositoryError::VersionConflict),
    }
}

fn insert_statement(binding: &AccountBinding) -> Result<Statement, IdentityBindingRepositoryError> {
    let fields = binding.fields();
    let verification_method = fields
        .verification
        .as_ref()
        .map(|verification| verification_method_name(verification.method).to_owned());
    let evidence_reference = fields
        .verification
        .as_ref()
        .map(|verification| verification.evidence_reference.as_str().to_owned());
    let verified_at = fields
        .verification
        .as_ref()
        .map(|verification| to_i64(verification.verified_at_millis))
        .transpose()?;
    let predecessor = fields.predecessor;
    let successor = fields.successor;
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_VERSION_SQL,
        [
            fields.community_id.as_uuid().into(),
            fields.binding_id.as_uuid().into(),
            to_i64(fields.version.get())?.into(),
            to_i64(fields.service_account_id.get())?.into(),
            fields.profile_id.as_uuid().into(),
            fields.public_key.as_bytes().to_vec().into(),
            status_name(fields.status).to_owned().into(),
            verification_method.into(),
            evidence_reference.into(),
            verified_at.into(),
            predecessor.map(|value| value.binding_id.as_uuid()).into(),
            predecessor
                .map(|value| to_i64(value.version.get()))
                .transpose()?
                .into(),
            successor.map(|value| value.binding_id.as_uuid()).into(),
            successor
                .map(|value| to_i64(value.version.get()))
                .transpose()?
                .into(),
            to_i64(fields.created_at_millis)?.into(),
            fields.activated_at_millis.map(to_i64).transpose()?.into(),
            fields.terminal_at_millis.map(to_i64).transpose()?.into(),
            to_i64(fields.organization_policy_version.get())?.into(),
            fields.actor_principal_id.as_uuid().into(),
            fields.audit_reference.as_uuid().into(),
        ],
    ))
}

fn binding_from_row(row: QueryResult) -> Result<AccountBinding, IdentityBindingRepositoryError> {
    let community_id = CommunityId::from_uuid(row_value(&row, "community_id")?);
    let binding_id = BindingId::from_uuid(row_value(&row, "binding_id")?);
    let version = aggregate_version(row_value(&row, "version")?)?;
    let service_account_id =
        ServiceAccountId::new(nonnegative_u64(row_value(&row, "service_account_id")?)?);
    let profile_id = ProfileId::from_uuid(row_value(&row, "profile_id")?);
    let public_key_bytes: Vec<u8> = row_value(&row, "nostr_public_key")?;
    let public_key = NostrPublicKey::from_bytes(
        public_key_bytes
            .try_into()
            .map_err(|_| IdentityBindingRepositoryError::InvalidRecord)?,
    );
    let status = parse_status(&row_value::<String>(&row, "status")?)?;
    let verification_method: Option<String> = row_value(&row, "verification_method")?;
    let evidence_reference: Option<String> = row_value(&row, "evidence_reference")?;
    let verified_at_millis: Option<i64> = row_value(&row, "verified_at_millis")?;
    let verification = match (verification_method, evidence_reference, verified_at_millis) {
        (None, None, None) => None,
        (Some(method), Some(evidence), Some(verified_at)) => Some(BindingVerification {
            method: parse_verification_method(&method)?,
            evidence_reference: EvidenceReference::new(evidence)
                .ok_or(IdentityBindingRepositoryError::InvalidRecord)?,
            verified_at_millis: nonnegative_u64(verified_at)?,
        }),
        _ => return Err(IdentityBindingRepositoryError::InvalidRecord),
    };
    let predecessor = parse_version_reference(
        row_value(&row, "predecessor_binding_id")?,
        row_value(&row, "predecessor_version")?,
    )?;
    let successor = parse_version_reference(
        row_value(&row, "successor_binding_id")?,
        row_value(&row, "successor_version")?,
    )?;
    AccountBinding::new(AccountBindingFields {
        binding_id,
        community_id,
        service_account_id,
        profile_id,
        public_key,
        status,
        verification,
        predecessor,
        successor,
        created_at_millis: nonnegative_u64(row_value(&row, "created_at_millis")?)?,
        activated_at_millis: optional_nonnegative_u64(row_value(&row, "activated_at_millis")?)?,
        terminal_at_millis: optional_nonnegative_u64(row_value(&row, "terminal_at_millis")?)?,
        organization_policy_version: OrganizationPolicyVersion::new(nonnegative_u64(row_value(
            &row,
            "organization_policy_version",
        )?)?)
        .ok_or(IdentityBindingRepositoryError::InvalidRecord)?,
        actor_principal_id: PrincipalId::from_uuid(row_value(&row, "actor_principal_id")?),
        version,
        audit_reference: OperationId::from_uuid(row_value(&row, "audit_reference")?),
    })
    .map_err(|_| IdentityBindingRepositoryError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, IdentityBindingRepositoryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| IdentityBindingRepositoryError::InvalidRecord)
}

fn parse_version_reference(
    binding_id: Option<Uuid>,
    version: Option<i64>,
) -> Result<Option<BindingVersionReference>, IdentityBindingRepositoryError> {
    match (binding_id, version) {
        (None, None) => Ok(None),
        (Some(binding_id), Some(version)) => Ok(Some(BindingVersionReference {
            binding_id: BindingId::from_uuid(binding_id),
            version: aggregate_version(version)?,
        })),
        _ => Err(IdentityBindingRepositoryError::InvalidRecord),
    }
}

fn aggregate_version(value: i64) -> Result<AggregateVersion, IdentityBindingRepositoryError> {
    AggregateVersion::new(nonnegative_u64(value)?)
        .ok_or(IdentityBindingRepositoryError::InvalidRecord)
}

fn optional_nonnegative_u64(
    value: Option<i64>,
) -> Result<Option<u64>, IdentityBindingRepositoryError> {
    value.map(nonnegative_u64).transpose()
}

fn nonnegative_u64(value: i64) -> Result<u64, IdentityBindingRepositoryError> {
    u64::try_from(value).map_err(|_| IdentityBindingRepositoryError::InvalidRecord)
}

fn to_i64(value: u64) -> Result<i64, IdentityBindingRepositoryError> {
    i64::try_from(value).map_err(|_| IdentityBindingRepositoryError::InvalidRecord)
}

fn map_write_error(error: DbErr) -> IdentityBindingRepositoryError {
    if matches!(error.sql_err(), Some(SqlErr::UniqueConstraintViolation(_))) {
        IdentityBindingRepositoryError::VersionConflict
    } else {
        IdentityBindingRepositoryError::Unavailable(error)
    }
}

fn status_name(status: BindingStatus) -> &'static str {
    match status {
        BindingStatus::Pending => "pending",
        BindingStatus::Verified => "verified",
        BindingStatus::Active => "active",
        BindingStatus::Rotated => "rotated",
        BindingStatus::Revoked => "revoked",
        BindingStatus::Archived => "archived",
    }
}

fn parse_status(value: &str) -> Result<BindingStatus, IdentityBindingRepositoryError> {
    match value {
        "pending" => Ok(BindingStatus::Pending),
        "verified" => Ok(BindingStatus::Verified),
        "active" => Ok(BindingStatus::Active),
        "rotated" => Ok(BindingStatus::Rotated),
        "revoked" => Ok(BindingStatus::Revoked),
        "archived" => Ok(BindingStatus::Archived),
        _ => Err(IdentityBindingRepositoryError::InvalidRecord),
    }
}

fn verification_method_name(method: BindingVerificationMethod) -> &'static str {
    match method {
        BindingVerificationMethod::GeneratedKeyChallenge => "generated_key_challenge",
        BindingVerificationMethod::ExistingKeyChallenge => "existing_key_challenge",
        BindingVerificationMethod::ImportedKeyChallenge => "imported_key_challenge",
        BindingVerificationMethod::PairedKeyChallenge => "paired_key_challenge",
        BindingVerificationMethod::RestoredKeyChallenge => "restored_key_challenge",
        BindingVerificationMethod::MigratedEvidence => "migrated_evidence",
    }
}

fn parse_verification_method(
    value: &str,
) -> Result<BindingVerificationMethod, IdentityBindingRepositoryError> {
    match value {
        "generated_key_challenge" => Ok(BindingVerificationMethod::GeneratedKeyChallenge),
        "existing_key_challenge" => Ok(BindingVerificationMethod::ExistingKeyChallenge),
        "imported_key_challenge" => Ok(BindingVerificationMethod::ImportedKeyChallenge),
        "paired_key_challenge" => Ok(BindingVerificationMethod::PairedKeyChallenge),
        "restored_key_challenge" => Ok(BindingVerificationMethod::RestoredKeyChallenge),
        "migrated_evidence" => Ok(BindingVerificationMethod::MigratedEvidence),
        _ => Err(IdentityBindingRepositoryError::InvalidRecord),
    }
}
