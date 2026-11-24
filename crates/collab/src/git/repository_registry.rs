use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, NostrPublicKey, PrincipalId, Provenance, SourceRecordId, SourceSystem,
    TenantContext, authorize,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use uuid::Uuid;

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const REPOSITORY_COLUMNS: &str = r#"
    repository.community_id,
    repository.repository_id,
    repository.repository_owner_public_key,
    repository.repository_discriminator,
    repository.authority_kind,
    repository.authority_version::bigint AS authority_version,
    repository.lifecycle_state,
    repository.provider_kind,
    repository.provider_instance,
    repository.provider_owner,
    repository.provider_repository,
    repository.source_system,
    repository.source_record_id,
    repository.source_version,
    floor(extract(epoch FROM repository.source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    CASE WHEN repository.archived_at IS NULL THEN NULL
         ELSE floor(extract(epoch FROM repository.archived_at) * 1000)::bigint
         END AS archived_at_millis,
    floor(extract(epoch FROM repository.created_at) * 1000)::bigint AS created_at_millis,
    floor(extract(epoch FROM repository.updated_at) * 1000)::bigint AS updated_at_millis,
    storage.storage_handle_id,
    storage.lifecycle_state AS storage_lifecycle_state
"#;
const REPOSITORY_FROM_SQL: &str = r#"
FROM public.collaboration_hosted_repositories AS repository
LEFT JOIN public.collaboration_git_storage_handles AS storage
    ON storage.community_id = repository.community_id
    AND storage.repository_id = repository.repository_id
"#;
const INSERT_REPOSITORY_SQL: &str = r#"
INSERT INTO public.collaboration_hosted_repositories (
    community_id, repository_id, repository_owner_public_key, repository_discriminator,
    authority_kind, authority_version, lifecycle_state, provider_kind, provider_instance,
    provider_owner, provider_repository, source_system, source_record_id, source_version,
    source_observed_at, created_at, updated_at
) VALUES (
    $1, $2, $3, $4, $5, 1, 'active', $6, $7, $8, $9, $10, $11, $12,
    to_timestamp($13::double precision / 1000),
    to_timestamp($14::double precision / 1000),
    to_timestamp($14::double precision / 1000)
)
"#;
const INSERT_STORAGE_SQL: &str = r#"
INSERT INTO public.collaboration_git_storage_handles (
    community_id, storage_handle_id, repository_id, handle_version, lifecycle_state,
    created_at, updated_at
) VALUES (
    $1, $2, $3, 1, 'active',
    to_timestamp($4::double precision / 1000),
    to_timestamp($4::double precision / 1000)
)
"#;
const INSERT_INITIAL_ADMIN_SQL: &str = r#"
INSERT INTO public.collaboration_git_repository_grants (
    community_id, repository_id, grantee_principal_id, permission, grant_version,
    grant_state, granted_by_principal_id, created_at, updated_at
) VALUES (
    $1, $2, $3, 'admin', 1, 'active', $3,
    to_timestamp($4::double precision / 1000),
    to_timestamp($4::double precision / 1000)
)
"#;
const RENAME_SQL: &str = r#"
UPDATE public.collaboration_hosted_repositories
SET repository_discriminator = $3,
    authority_version = authority_version + 1,
    updated_at = to_timestamp($5::double precision / 1000)
WHERE community_id = $1 AND repository_id = $2 AND authority_version = $4
  AND lifecycle_state = 'active'
"#;
const UPSERT_GRANT_SQL: &str = r#"
INSERT INTO public.collaboration_git_repository_grants (
    community_id, repository_id, grantee_principal_id, permission, grant_version,
    grant_state, granted_by_principal_id, created_at, updated_at
)
SELECT $1, $2, membership.principal_id, $4, 1, 'active', $5,
       to_timestamp($6::double precision / 1000),
       to_timestamp($6::double precision / 1000)
FROM public.collaboration_community_memberships AS membership
WHERE membership.community_id = $1 AND membership.principal_id = $3
  AND membership.status = 'active'
ON CONFLICT (community_id, repository_id, grantee_principal_id, permission)
DO UPDATE SET
    grant_version = collaboration_git_repository_grants.grant_version + 1,
    grant_state = 'active', granted_by_principal_id = EXCLUDED.granted_by_principal_id,
    revoked_at = NULL, updated_at = EXCLUDED.updated_at
"#;
const REVOKE_GRANT_SQL: &str = r#"
UPDATE public.collaboration_git_repository_grants
SET grant_version = grant_version + 1, grant_state = 'revoked',
    revoked_at = to_timestamp($5::double precision / 1000),
    updated_at = to_timestamp($5::double precision / 1000)
WHERE community_id = $1 AND repository_id = $2 AND grantee_principal_id = $3
  AND permission = $4 AND grant_state = 'active'
"#;
const ARCHIVE_GRANTS_SQL: &str = r#"
UPDATE public.collaboration_git_repository_grants
SET grant_version = grant_version + 1, grant_state = 'revoked',
    revoked_at = to_timestamp($3::double precision / 1000),
    updated_at = to_timestamp($3::double precision / 1000)
WHERE community_id = $1 AND repository_id = $2 AND grant_state = 'active'
"#;
const ARCHIVE_STORAGE_SQL: &str = r#"
UPDATE public.collaboration_git_storage_handles
SET handle_version = handle_version + 1, lifecycle_state = 'archived',
    archived_at = to_timestamp($3::double precision / 1000),
    updated_at = to_timestamp($3::double precision / 1000)
WHERE community_id = $1 AND repository_id = $2 AND lifecycle_state = 'active'
"#;
const ARCHIVE_REPOSITORY_SQL: &str = r#"
UPDATE public.collaboration_hosted_repositories
SET authority_version = authority_version + 1, lifecycle_state = 'archived',
    archived_at = to_timestamp($4::double precision / 1000),
    updated_at = to_timestamp($4::double precision / 1000)
WHERE community_id = $1 AND repository_id = $2 AND authority_version = $3
  AND lifecycle_state = 'active'
"#;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RepositoryPermission {
    Read,
    Write,
    Admin,
}

impl RepositoryPermission {
    fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
        }
    }

    fn required_scope(self) -> &'static str {
        match self {
            Self::Read => "git:read",
            Self::Write => "git:write",
            Self::Admin => "git:admin",
        }
    }

    fn required_action(self) -> AuthorizationAction {
        match self {
            Self::Read => AuthorizationAction::Read,
            Self::Write => AuthorizationAction::Write,
            Self::Admin => AuthorizationAction::Manage,
        }
    }

    fn permits(self, required: Self) -> bool {
        self >= required
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryCoordinate {
    pub owner_public_key: NostrPublicKey,
    pub discriminator: String,
}

impl RepositoryCoordinate {
    pub fn new(
        owner_public_key: NostrPublicKey,
        discriminator: impl Into<String>,
    ) -> Result<Self, HostedRepositoryRegistryError> {
        let discriminator = discriminator.into();
        validate_discriminator(&discriminator)?;
        Ok(Self {
            owner_public_key,
            discriminator,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalProviderCoordinate {
    pub provider_kind: String,
    pub provider_instance: String,
    pub owner: String,
    pub repository: String,
}

impl ExternalProviderCoordinate {
    pub fn new(
        provider_kind: impl Into<String>,
        provider_instance: impl Into<String>,
        owner: impl Into<String>,
        repository: impl Into<String>,
    ) -> Result<Self, HostedRepositoryRegistryError> {
        let coordinate = Self {
            provider_kind: provider_kind.into(),
            provider_instance: provider_instance.into(),
            owner: owner.into(),
            repository: repository.into(),
        };
        validate_provider_field(&coordinate.provider_kind, 64)?;
        validate_provider_field(&coordinate.provider_instance, 512)?;
        validate_provider_field(&coordinate.owner, 512)?;
        validate_provider_field(&coordinate.repository, 512)?;
        Ok(coordinate)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostedAuthority {
    SimHostedNip34 { storage_handle_id: Uuid },
    ExternalProvider(ExternalProviderCoordinate),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostedRepositoryLifecycle {
    Active,
    Archived,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedRepository {
    pub community_id: CommunityId,
    pub repository_id: AggregateId,
    pub coordinate: RepositoryCoordinate,
    pub authority: HostedAuthority,
    pub authority_version: AggregateVersion,
    pub lifecycle: HostedRepositoryLifecycle,
    pub provenance: Provenance,
    pub archived_at_millis: Option<u64>,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedRepositoryDraft {
    pub community_id: CommunityId,
    pub repository_id: AggregateId,
    pub coordinate: RepositoryCoordinate,
    pub authority: HostedAuthority,
    pub provenance: Provenance,
    pub created_at_millis: u64,
}

impl HostedRepositoryDraft {
    pub fn validate(&self) -> Result<(), HostedRepositoryRegistryError> {
        validate_discriminator(&self.coordinate.discriminator)?;
        validate_provenance(&self.provenance)?;
        validate_millis(self.created_at_millis)?;
        match &self.authority {
            HostedAuthority::SimHostedNip34 { storage_handle_id } if storage_handle_id.is_nil() => {
                Err(HostedRepositoryRegistryError::InvalidRecord)
            }
            HostedAuthority::SimHostedNip34 { .. } => Ok(()),
            HostedAuthority::ExternalProvider(coordinate) => {
                validate_provider_field(&coordinate.provider_kind, 64)?;
                validate_provider_field(&coordinate.provider_instance, 512)?;
                validate_provider_field(&coordinate.owner, 512)?;
                validate_provider_field(&coordinate.repository, 512)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HostedRepositoryRegistryError {
    #[error("hosted repository registry requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("hosted repository request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("hosted repository request is not authorized: {0:?}")]
    Unauthorized(AuthorizationDenial),
    #[error("hosted repository permission is unavailable")]
    PermissionDenied,
    #[error("hosted repository does not exist")]
    NotFound,
    #[error("hosted repository optimistic version does not match current state")]
    VersionConflict,
    #[error("hosted repository record is invalid or exceeds a bound")]
    InvalidRecord,
    #[error("hosted repository registry is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct HostedRepositoryRegistry {
    connection: DatabaseConnection,
}

impl HostedRepositoryRegistry {
    pub fn new(connection: DatabaseConnection) -> Result<Self, HostedRepositoryRegistryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(HostedRepositoryRegistryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn create(
        &self,
        authorization: &AuthorizationRequest<'_>,
        draft: &HostedRepositoryDraft,
    ) -> Result<HostedRepository, HostedRepositoryRegistryError> {
        draft.validate()?;
        require_authorization_shape(
            authorization,
            draft.community_id,
            draft.repository_id,
            RepositoryPermission::Admin,
        )?;
        authorize_common(authorization)?;
        let actor = authorization_subject(authorization);
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, draft.community_id).await?;
            transaction
                .execute(insert_repository_statement(draft)?)
                .await
                .map_err(map_write_error)?;
            if let HostedAuthority::SimHostedNip34 { storage_handle_id } = draft.authority {
                transaction
                    .execute(Statement::from_sql_and_values(
                        DatabaseBackend::Postgres,
                        INSERT_STORAGE_SQL,
                        [
                            draft.community_id.as_uuid().into(),
                            storage_handle_id.into(),
                            draft.repository_id.as_uuid().into(),
                            millis_i64(draft.created_at_millis)?.into(),
                        ],
                    ))
                    .await
                    .map_err(map_write_error)?;
            }
            transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    INSERT_INITIAL_ADMIN_SQL,
                    [
                        draft.community_id.as_uuid().into(),
                        draft.repository_id.as_uuid().into(),
                        actor.as_uuid().into(),
                        millis_i64(draft.created_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(map_write_error)?;
            select_repository(&transaction, draft.community_id, draft.repository_id).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn authorize_access(
        &self,
        authorization: &AuthorizationRequest<'_>,
        required_permission: RepositoryPermission,
    ) -> Result<HostedRepository, HostedRepositoryRegistryError> {
        let repository_id = repository_id_from_authorization(authorization)?;
        require_authorization_shape(
            authorization,
            authorization.tenant.community_id(),
            repository_id,
            required_permission,
        )?;
        authorize_common(authorization)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, authorization.tenant.community_id()).await?;
            select_authorized(
                &transaction,
                authorization.tenant,
                repository_id,
                authorization_subject(authorization),
                required_permission,
                false,
            )
            .await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn rename(
        &self,
        authorization: &AuthorizationRequest<'_>,
        expected_version: AggregateVersion,
        discriminator: impl Into<String>,
        updated_at_millis: u64,
    ) -> Result<HostedRepository, HostedRepositoryRegistryError> {
        let discriminator = discriminator.into();
        validate_discriminator(&discriminator)?;
        validate_millis(updated_at_millis)?;
        let repository_id = repository_id_from_authorization(authorization)?;
        require_admin_shape(authorization, repository_id)?;
        authorize_common(authorization)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, authorization.tenant.community_id()).await?;
            let current = select_authorized(
                &transaction,
                authorization.tenant,
                repository_id,
                authorization_subject(authorization),
                RepositoryPermission::Admin,
                true,
            )
            .await?;
            if current.authority_version != expected_version {
                return Err(HostedRepositoryRegistryError::VersionConflict);
            }
            let renamed = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    RENAME_SQL,
                    [
                        current.community_id.as_uuid().into(),
                        repository_id.as_uuid().into(),
                        discriminator.into(),
                        version_i64(expected_version)?.into(),
                        millis_i64(updated_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(map_write_error)?;
            if renamed.rows_affected() != 1 {
                return Err(HostedRepositoryRegistryError::VersionConflict);
            }
            select_repository(&transaction, current.community_id, repository_id).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn grant(
        &self,
        authorization: &AuthorizationRequest<'_>,
        grantee_principal_id: PrincipalId,
        permission: RepositoryPermission,
        updated_at_millis: u64,
    ) -> Result<(), HostedRepositoryRegistryError> {
        validate_millis(updated_at_millis)?;
        let repository_id = repository_id_from_authorization(authorization)?;
        require_admin_shape(authorization, repository_id)?;
        authorize_common(authorization)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, authorization.tenant.community_id()).await?;
            select_authorized(
                &transaction,
                authorization.tenant,
                repository_id,
                authorization_subject(authorization),
                RepositoryPermission::Admin,
                true,
            )
            .await?;
            let granted = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    UPSERT_GRANT_SQL,
                    [
                        authorization.tenant.community_id().as_uuid().into(),
                        repository_id.as_uuid().into(),
                        grantee_principal_id.as_uuid().into(),
                        permission.name().to_owned().into(),
                        authorization_subject(authorization).as_uuid().into(),
                        millis_i64(updated_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(map_write_error)?;
            if granted.rows_affected() != 1 {
                return Err(HostedRepositoryRegistryError::PermissionDenied);
            }
            Ok(())
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn revoke(
        &self,
        authorization: &AuthorizationRequest<'_>,
        grantee_principal_id: PrincipalId,
        permission: RepositoryPermission,
        updated_at_millis: u64,
    ) -> Result<(), HostedRepositoryRegistryError> {
        validate_millis(updated_at_millis)?;
        let repository_id = repository_id_from_authorization(authorization)?;
        require_admin_shape(authorization, repository_id)?;
        authorize_common(authorization)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, authorization.tenant.community_id()).await?;
            select_authorized(
                &transaction,
                authorization.tenant,
                repository_id,
                authorization_subject(authorization),
                RepositoryPermission::Admin,
                true,
            )
            .await?;
            let revoked = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    REVOKE_GRANT_SQL,
                    [
                        authorization.tenant.community_id().as_uuid().into(),
                        repository_id.as_uuid().into(),
                        grantee_principal_id.as_uuid().into(),
                        permission.name().to_owned().into(),
                        millis_i64(updated_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(map_write_error)?;
            if revoked.rows_affected() != 1 {
                return Err(HostedRepositoryRegistryError::PermissionDenied);
            }
            Ok(())
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn archive(
        &self,
        authorization: &AuthorizationRequest<'_>,
        expected_version: AggregateVersion,
        archived_at_millis: u64,
    ) -> Result<HostedRepository, HostedRepositoryRegistryError> {
        validate_millis(archived_at_millis)?;
        let repository_id = repository_id_from_authorization(authorization)?;
        require_admin_shape(authorization, repository_id)?;
        authorize_common(authorization)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, authorization.tenant.community_id()).await?;
            let current = select_authorized(
                &transaction,
                authorization.tenant,
                repository_id,
                authorization_subject(authorization),
                RepositoryPermission::Admin,
                true,
            )
            .await?;
            if current.authority_version != expected_version {
                return Err(HostedRepositoryRegistryError::VersionConflict);
            }
            for statement in archive_dependent_statements(&current, archived_at_millis)? {
                transaction
                    .execute(statement)
                    .await
                    .map_err(map_write_error)?;
            }
            let archived = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    ARCHIVE_REPOSITORY_SQL,
                    [
                        current.community_id.as_uuid().into(),
                        repository_id.as_uuid().into(),
                        version_i64(expected_version)?.into(),
                        millis_i64(archived_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(map_write_error)?;
            if archived.rows_affected() != 1 {
                return Err(HostedRepositoryRegistryError::VersionConflict);
            }
            select_repository(&transaction, current.community_id, repository_id).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, HostedRepositoryRegistryError> {
        self.connection
            .begin()
            .await
            .map_err(HostedRepositoryRegistryError::Unavailable)
    }
}

async fn select_authorized(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    repository_id: AggregateId,
    grantee_principal_id: PrincipalId,
    required_permission: RepositoryPermission,
    for_update: bool,
) -> Result<HostedRepository, HostedRepositoryRegistryError> {
    let mut sql = format!(
        "SELECT {REPOSITORY_COLUMNS}, repository_grant.permission {REPOSITORY_FROM_SQL} \
         LEFT JOIN public.collaboration_git_repository_grants AS repository_grant \
         ON repository_grant.community_id = repository.community_id \
         AND repository_grant.repository_id = repository.repository_id \
         AND repository_grant.grantee_principal_id = $3 \
         AND repository_grant.grant_state = 'active' \
         WHERE repository.community_id = $1 AND repository.repository_id = $2"
    );
    if for_update {
        sql.push_str(" FOR UPDATE OF repository");
    }
    let rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            [
                tenant.community_id().as_uuid().into(),
                repository_id.as_uuid().into(),
                grantee_principal_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(HostedRepositoryRegistryError::Unavailable)?;
    let Some(first) = rows.first() else {
        return Err(HostedRepositoryRegistryError::PermissionDenied);
    };
    let repository = repository_from_row(first)?;
    if repository.community_id != tenant.community_id() || repository.repository_id != repository_id
    {
        return Err(HostedRepositoryRegistryError::TenantBoundaryViolation);
    }
    if repository.lifecycle != HostedRepositoryLifecycle::Active {
        return Err(HostedRepositoryRegistryError::PermissionDenied);
    }
    let permitted = rows.iter().try_fold(false, |permitted, row| {
        let permission = row_value::<Option<String>>(row, "permission")?
            .map(|permission| parse_permission(&permission))
            .transpose()?;
        Ok::<_, HostedRepositoryRegistryError>(
            permitted
                || permission.is_some_and(|permission| permission.permits(required_permission)),
        )
    })?;
    if !permitted {
        return Err(HostedRepositoryRegistryError::PermissionDenied);
    }
    Ok(repository)
}

async fn select_repository(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    repository_id: AggregateId,
) -> Result<HostedRepository, HostedRepositoryRegistryError> {
    let sql = format!(
        "SELECT {REPOSITORY_COLUMNS} {REPOSITORY_FROM_SQL} \
         WHERE repository.community_id = $1 AND repository.repository_id = $2"
    );
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            [
                community_id.as_uuid().into(),
                repository_id.as_uuid().into(),
            ],
        ))
        .await
        .map_err(HostedRepositoryRegistryError::Unavailable)?
        .ok_or(HostedRepositoryRegistryError::NotFound)?;
    let repository = repository_from_row(&row)?;
    if repository.community_id != community_id || repository.repository_id != repository_id {
        return Err(HostedRepositoryRegistryError::TenantBoundaryViolation);
    }
    Ok(repository)
}

fn repository_from_row(
    row: &QueryResult,
) -> Result<HostedRepository, HostedRepositoryRegistryError> {
    let community_id = CommunityId::from_uuid(row_value(row, "community_id")?);
    let repository_id = AggregateId::from_uuid(row_value(row, "repository_id")?);
    let owner_bytes: Vec<u8> = row_value(row, "repository_owner_public_key")?;
    let owner_public_key = NostrPublicKey::from_bytes(
        owner_bytes
            .try_into()
            .map_err(|_| HostedRepositoryRegistryError::InvalidRecord)?,
    );
    let coordinate = RepositoryCoordinate::new(
        owner_public_key,
        row_value::<String>(row, "repository_discriminator")?,
    )?;
    let authority_kind: String = row_value(row, "authority_kind")?;
    let provider_kind: Option<String> = row_value(row, "provider_kind")?;
    let provider_instance: Option<String> = row_value(row, "provider_instance")?;
    let provider_owner: Option<String> = row_value(row, "provider_owner")?;
    let provider_repository: Option<String> = row_value(row, "provider_repository")?;
    let storage_handle_id: Option<Uuid> = row_value(row, "storage_handle_id")?;
    let storage_lifecycle: Option<String> = row_value(row, "storage_lifecycle_state")?;
    let lifecycle = parse_lifecycle(&row_value::<String>(row, "lifecycle_state")?)?;
    let authority = match (
        authority_kind.as_str(),
        provider_kind,
        provider_instance,
        provider_owner,
        provider_repository,
        storage_handle_id,
        storage_lifecycle.as_deref(),
    ) {
        (
            "sim_hosted_nip34",
            None,
            None,
            None,
            None,
            Some(storage_handle_id),
            Some(storage_state),
        ) if storage_state == lifecycle_name(lifecycle) => {
            HostedAuthority::SimHostedNip34 { storage_handle_id }
        }
        (
            "external_provider",
            Some(provider_kind),
            Some(provider_instance),
            Some(owner),
            Some(repository),
            None,
            None,
        ) => HostedAuthority::ExternalProvider(ExternalProviderCoordinate::new(
            provider_kind,
            provider_instance,
            owner,
            repository,
        )?),
        _ => return Err(HostedRepositoryRegistryError::InvalidRecord),
    };
    let provenance = Provenance {
        source_system: parse_source_system(&row_value::<String>(row, "source_system")?)?,
        source_record_id: SourceRecordId::new(row_value::<String>(row, "source_record_id")?)
            .ok_or(HostedRepositoryRegistryError::InvalidRecord)?,
        source_version: row_value(row, "source_version")?,
        observed_at_millis: nonnegative_u64(row_value(row, "source_observed_at_millis")?)?,
        integrity: None,
    };
    validate_provenance(&provenance)?;
    Ok(HostedRepository {
        community_id,
        repository_id,
        coordinate,
        authority,
        authority_version: aggregate_version(row_value(row, "authority_version")?)?,
        lifecycle,
        provenance,
        archived_at_millis: optional_nonnegative_u64(row_value(row, "archived_at_millis")?)?,
        created_at_millis: nonnegative_u64(row_value(row, "created_at_millis")?)?,
        updated_at_millis: nonnegative_u64(row_value(row, "updated_at_millis")?)?,
    })
}

fn insert_repository_statement(
    draft: &HostedRepositoryDraft,
) -> Result<Statement, HostedRepositoryRegistryError> {
    let (authority_kind, provider_kind, provider_instance, provider_owner, provider_repository) =
        match &draft.authority {
            HostedAuthority::SimHostedNip34 { .. } => ("sim_hosted_nip34", None, None, None, None),
            HostedAuthority::ExternalProvider(coordinate) => (
                "external_provider",
                Some(coordinate.provider_kind.clone()),
                Some(coordinate.provider_instance.clone()),
                Some(coordinate.owner.clone()),
                Some(coordinate.repository.clone()),
            ),
        };
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_REPOSITORY_SQL,
        [
            draft.community_id.as_uuid().into(),
            draft.repository_id.as_uuid().into(),
            draft.coordinate.owner_public_key.as_bytes().to_vec().into(),
            draft.coordinate.discriminator.clone().into(),
            authority_kind.to_owned().into(),
            provider_kind.into(),
            provider_instance.into(),
            provider_owner.into(),
            provider_repository.into(),
            source_system_name(draft.provenance.source_system)
                .to_owned()
                .into(),
            draft.provenance.source_record_id.as_str().to_owned().into(),
            draft.provenance.source_version.clone().into(),
            millis_i64(draft.provenance.observed_at_millis)?.into(),
            millis_i64(draft.created_at_millis)?.into(),
        ],
    ))
}

fn archive_dependent_statements(
    repository: &HostedRepository,
    archived_at_millis: u64,
) -> Result<[Statement; 2], HostedRepositoryRegistryError> {
    let archived_at_millis = millis_i64(archived_at_millis)?;
    let values = || {
        [
            repository.community_id.as_uuid().into(),
            repository.repository_id.as_uuid().into(),
            archived_at_millis.into(),
        ]
    };
    Ok([
        Statement::from_sql_and_values(DatabaseBackend::Postgres, ARCHIVE_GRANTS_SQL, values()),
        Statement::from_sql_and_values(DatabaseBackend::Postgres, ARCHIVE_STORAGE_SQL, values()),
    ])
}

fn require_admin_shape(
    authorization: &AuthorizationRequest<'_>,
    repository_id: AggregateId,
) -> Result<(), HostedRepositoryRegistryError> {
    require_authorization_shape(
        authorization,
        authorization.tenant.community_id(),
        repository_id,
        RepositoryPermission::Admin,
    )
}

fn require_authorization_shape(
    authorization: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    repository_id: AggregateId,
    permission: RepositoryPermission,
) -> Result<(), HostedRepositoryRegistryError> {
    if authorization.tenant.community_id() != community_id
        || authorization.resource.community_id != community_id
        || authorization.resource.kind != AuthorizationResourceKind::Repository
        || authorization.resource.resource_id != repository_id
    {
        return Err(HostedRepositoryRegistryError::TenantBoundaryViolation);
    }
    if authorization.action != permission.required_action()
        || authorization.required_scope.as_str() != permission.required_scope()
    {
        return Err(HostedRepositoryRegistryError::PermissionDenied);
    }
    Ok(())
}

fn authorize_common(
    authorization: &AuthorizationRequest<'_>,
) -> Result<(), HostedRepositoryRegistryError> {
    match authorize(authorization) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => {
            Err(HostedRepositoryRegistryError::Unauthorized(denial))
        }
    }
}

fn repository_id_from_authorization(
    authorization: &AuthorizationRequest<'_>,
) -> Result<AggregateId, HostedRepositoryRegistryError> {
    if authorization.resource.kind != AuthorizationResourceKind::Repository {
        return Err(HostedRepositoryRegistryError::TenantBoundaryViolation);
    }
    Ok(authorization.resource.resource_id)
}

fn authorization_subject(authorization: &AuthorizationRequest<'_>) -> PrincipalId {
    match authorization.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => authorization.principal.principal_id(),
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), HostedRepositoryRegistryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(HostedRepositoryRegistryError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, HostedRepositoryRegistryError>,
) -> Result<T, HostedRepositoryRegistryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(HostedRepositoryRegistryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(HostedRepositoryRegistryError::Unavailable)?;
            Err(error)
        }
    }
}

fn validate_discriminator(value: &str) -> Result<(), HostedRepositoryRegistryError> {
    if value.is_empty()
        || value.len() > 64
        || value.trim() != value
        || matches!(value, "." | "..")
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.bytes().any(|byte| matches!(byte, b'/' | b'\\'))
    {
        return Err(HostedRepositoryRegistryError::InvalidRecord);
    }
    Ok(())
}

fn validate_provider_field(
    value: &str,
    max_bytes: usize,
) -> Result<(), HostedRepositoryRegistryError> {
    if value.is_empty()
        || value.len() > max_bytes
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(HostedRepositoryRegistryError::InvalidRecord);
    }
    Ok(())
}

fn validate_provenance(provenance: &Provenance) -> Result<(), HostedRepositoryRegistryError> {
    if provenance.integrity.is_some()
        || provenance
            .source_version
            .as_ref()
            .is_some_and(|version| version.is_empty() || version.len() > 256)
    {
        return Err(HostedRepositoryRegistryError::InvalidRecord);
    }
    validate_millis(provenance.observed_at_millis)
}

fn validate_millis(value: u64) -> Result<(), HostedRepositoryRegistryError> {
    millis_i64(value).map(|_| ())
}

fn millis_i64(value: u64) -> Result<i64, HostedRepositoryRegistryError> {
    i64::try_from(value).map_err(|_| HostedRepositoryRegistryError::InvalidRecord)
}

fn version_i64(value: AggregateVersion) -> Result<i64, HostedRepositoryRegistryError> {
    i64::try_from(value.get()).map_err(|_| HostedRepositoryRegistryError::InvalidRecord)
}

fn aggregate_version(value: i64) -> Result<AggregateVersion, HostedRepositoryRegistryError> {
    AggregateVersion::new(nonnegative_u64(value)?)
        .ok_or(HostedRepositoryRegistryError::InvalidRecord)
}

fn nonnegative_u64(value: i64) -> Result<u64, HostedRepositoryRegistryError> {
    u64::try_from(value).map_err(|_| HostedRepositoryRegistryError::InvalidRecord)
}

fn optional_nonnegative_u64(
    value: Option<i64>,
) -> Result<Option<u64>, HostedRepositoryRegistryError> {
    value.map(nonnegative_u64).transpose()
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, HostedRepositoryRegistryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| HostedRepositoryRegistryError::InvalidRecord)
}

fn parse_permission(value: &str) -> Result<RepositoryPermission, HostedRepositoryRegistryError> {
    match value {
        "read" => Ok(RepositoryPermission::Read),
        "write" => Ok(RepositoryPermission::Write),
        "admin" => Ok(RepositoryPermission::Admin),
        _ => Err(HostedRepositoryRegistryError::InvalidRecord),
    }
}

fn parse_lifecycle(
    value: &str,
) -> Result<HostedRepositoryLifecycle, HostedRepositoryRegistryError> {
    match value {
        "active" => Ok(HostedRepositoryLifecycle::Active),
        "archived" => Ok(HostedRepositoryLifecycle::Archived),
        _ => Err(HostedRepositoryRegistryError::InvalidRecord),
    }
}

fn lifecycle_name(value: HostedRepositoryLifecycle) -> &'static str {
    match value {
        HostedRepositoryLifecycle::Active => "active",
        HostedRepositoryLifecycle::Archived => "archived",
    }
}

fn source_system_name(value: SourceSystem) -> &'static str {
    match value {
        SourceSystem::Zed => "zed",
        SourceSystem::Buzz => "buzz",
        SourceSystem::Nostr => "nostr",
        SourceSystem::Acp => "acp",
        SourceSystem::ExternalGit => "external_git",
    }
}

fn parse_source_system(value: &str) -> Result<SourceSystem, HostedRepositoryRegistryError> {
    match value {
        "zed" => Ok(SourceSystem::Zed),
        "buzz" => Ok(SourceSystem::Buzz),
        "nostr" => Ok(SourceSystem::Nostr),
        "acp" => Ok(SourceSystem::Acp),
        "external_git" => Ok(SourceSystem::ExternalGit),
        _ => Err(HostedRepositoryRegistryError::InvalidRecord),
    }
}

fn map_write_error(error: DbErr) -> HostedRepositoryRegistryError {
    if matches!(
        error.sql_err(),
        Some(sea_orm::SqlErr::UniqueConstraintViolation(_))
    ) {
        HostedRepositoryRegistryError::VersionConflict
    } else {
        HostedRepositoryRegistryError::Unavailable(error)
    }
}
