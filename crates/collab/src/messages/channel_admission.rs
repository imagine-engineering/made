use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, Result, anyhow};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
    UntrustedTenantClaim, UntrustedTenantClaimSource, channel_id_for_legacy_channel,
    community_id_for_legacy_root_channel, principal_id_for_legacy_user,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, QueryResult,
    Statement, TransactionTrait,
};

use crate::{
    db::{ChannelRole, Database},
    entities::User,
    tenant_admission::{AuthorizedRpcRequest, bind_rpc_tenant},
};

pub const CHANNEL_READ_SCOPE: &str = "collaboration:channel:read";
pub const CHANNEL_WRITE_SCOPE: &str = "collaboration:channel:write";

pub(crate) fn channel_scope(
    principal: &AuthenticatedPrincipal,
    action: AuthorizationAction,
) -> Result<AuthorizationScope> {
    let nostr = matches!(
        principal.kind(),
        collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { .. }
            | collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
    );
    AuthorizationScope::new(match (nostr, action) {
        (true, AuthorizationAction::Read) => "messages:read",
        (true, _) => "messages:write",
        (false, AuthorizationAction::Read) => CHANNEL_READ_SCOPE,
        (false, _) => CHANNEL_WRITE_SCOPE,
    })
    .map_err(Into::into)
}

#[derive(Clone)]
pub struct AuthorizedChannel {
    pub tenant: TenantContext,
    pub principal: AuthenticatedPrincipal,
    pub community_membership: CommunityMembership,
    pub channel_membership: ChannelMembership,
    pub channel_id: AggregateId,
    pub signing_public_key: Option<[u8; 32]>,
}

impl AuthorizedChannel {
    pub fn authorization_request<'a>(
        &'a self,
        action: AuthorizationAction,
        required_scope: &'a AuthorizationScope,
        now_millis: u64,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope,
            action,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: self.channel_id,
                owner_principal_id: None,
                channel_id: Some(self.channel_id),
            },
            current_membership_version: self.community_membership.version,
            community_membership: Some(self.community_membership),
            current_channel_membership_version: Some(self.channel_membership.version),
            channel_membership: Some(self.channel_membership),
            delegation: None,
            now_millis,
        }
    }

    pub fn message_authorization_request<'a>(
        &'a self,
        message_id: AggregateId,
        author_principal_id: PrincipalId,
        action: AuthorizationAction,
        required_scope: &'a AuthorizationScope,
        now_millis: u64,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope,
            action,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Conversation,
                resource_id: message_id,
                owner_principal_id: Some(author_principal_id),
                channel_id: Some(self.channel_id),
            },
            current_membership_version: self.community_membership.version,
            community_membership: Some(self.community_membership),
            current_channel_membership_version: Some(self.channel_membership.version),
            channel_membership: Some(self.channel_membership),
            delegation: None,
            now_millis,
        }
    }

    pub fn authorize(
        &self,
        action: AuthorizationAction,
        now_millis: u64,
    ) -> Result<AuthorizedRpcRequest> {
        let scope = channel_scope(&self.principal, action)?;
        AuthorizedRpcRequest::authorize(&self.authorization_request(action, &scope, now_millis))
            .map_err(|_| anyhow!("channel access denied"))
    }
}

pub(crate) struct CurrentMessageAdmission {
    pub authorization: AuthorizedChannel,
    pub agent_author: Option<crate::identity::profile_repository::VerifiedAgentMessageAuthor>,
    pub owner_membership: Option<CommunityMembership>,
    pub message_author: Option<collaboration_domain::MessageAuthor>,
    moderation: Option<CurrentMessageModeration>,
}

struct CurrentMessageModeration {
    restriction: Option<collaboration_domain::ModerationRestriction>,
    principal_archive:
        collaboration_domain::ModerationSnapshot<collaboration_domain::IdentityArchiveSnapshot>,
    community_archive:
        collaboration_domain::ModerationSnapshot<collaboration_domain::CommunityArchiveSnapshot>,
}

impl CurrentMessageAdmission {
    pub(crate) fn authorize(
        &self,
        action: AuthorizationAction,
        now_millis: u64,
    ) -> Result<AuthorizedRpcRequest> {
        if let Some(moderation) = &self.moderation {
            let scope = channel_scope(&self.authorization.principal, action)?;
            let authorization = self
                .authorization
                .authorization_request(action, &scope, now_millis);
            let decision = collaboration_domain::authorize_with_moderation(
                &collaboration_domain::ModerationAuthorizationRequest {
                    authorization: &authorization,
                    restriction: moderation.restriction.as_ref().map_or(
                        collaboration_domain::ModerationSnapshot::Absent,
                        collaboration_domain::ModerationSnapshot::Current,
                    ),
                    principal_archive: moderation.principal_archive,
                    community_archive: moderation.community_archive,
                    resource_context: collaboration_domain::ModerationResourceContext::Current,
                },
            );
            if !matches!(
                decision,
                collaboration_domain::ModerationAuthorizationDecision::Allowed(_)
            ) {
                return Err(anyhow!("channel access denied"));
            }
        }
        self.authorization.authorize(action, now_millis)
    }
}

pub async fn admit_authenticated_channel_in_transaction(
    transaction: &DatabaseTransaction,
    profiles: &crate::identity::profile_repository::IdentityProfileRepository,
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    channel_id: AggregateId,
    now_millis: u64,
) -> Result<AuthorizedChannel> {
    Ok(current_message_admission_in_transaction(
        transaction,
        profiles,
        tenant,
        principal,
        channel_id,
        now_millis,
    )
    .await?
    .authorization)
}

pub(crate) async fn current_message_admission_in_transaction(
    transaction: &DatabaseTransaction,
    profiles: &crate::identity::profile_repository::IdentityProfileRepository,
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    channel_id: AggregateId,
    now_millis: u64,
) -> Result<CurrentMessageAdmission> {
    if principal.community_id() != tenant.community_id() || channel_id.as_uuid().is_nil() {
        return Err(anyhow!("channel access denied"));
    }
    let canonical_nostr = matches!(
        principal.kind(),
        collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { .. }
            | collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
    );
    if canonical_nostr {
        // Restriction writers take this same lock before identity and membership locks.
        crate::admin::moderation::lock_moderation_in_transaction(transaction, tenant).await?;
    }
    current_principal_memberships_in_transaction(transaction, tenant, principal, Some(channel_id))
        .await?;
    if canonical_nostr {
        let community = current_community_serving_state_in_transaction(transaction, tenant)
            .await?
            .ok_or_else(|| anyhow!("channel access denied"))?;
        if community.lifecycle != collaboration_domain::CommunityLifecycleState::Active {
            return Err(anyhow!("channel access denied"));
        }
    }
    let agent_author = profiles
        .verified_message_agent_in_transaction(transaction, tenant, principal, now_millis / 1000)
        .await?;
    let owner_principal_id = agent_author
        .as_ref()
        .map(|author| author.owner_principal_id());
    let subject_principal_id =
        collaboration_domain::authorization::authorization_subject(principal);
    // Hold current membership rows through the command commit, including duplicate receipts.
    transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "SELECT principal_id FROM public.collaboration_community_memberships WHERE community_id=$1 AND (principal_id=$2 OR principal_id=$3) ORDER BY principal_id FOR SHARE",
        [tenant.community_id().as_uuid().into(), subject_principal_id.as_uuid().into(), owner_principal_id.map(|id| id.as_uuid()).into()])).await?;
    transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "SELECT principal_id FROM public.collaboration_channel_memberships WHERE community_id=$1 AND channel_id=$2 AND principal_id=$3 FOR SHARE",
        [tenant.community_id().as_uuid().into(), channel_id.as_uuid().into(), subject_principal_id.as_uuid().into()])).await?;
    let memberships = current_principal_memberships_in_transaction(
        transaction,
        tenant,
        principal,
        Some(channel_id),
    )
    .await?;
    let channel_membership = memberships
        .channel
        .ok_or_else(|| anyhow!("channel access denied"))?;
    let (community_membership, owner_membership, message_author) =
        if let Some(author) = &agent_author {
            if memberships
                .community
                .is_some_and(|membership| membership.status != MembershipStatus::Active)
            {
                return Err(anyhow!("channel access denied"));
            }
            let owner = current_subject_memberships_in_transaction(
                transaction,
                tenant,
                author.owner_principal_id(),
                None,
            )
            .await?
            .community
            .ok_or_else(|| anyhow!("channel access denied"))?;
            let virtual_membership = collaboration_domain::VirtualAgentMembershipEvidence::derive(
                principal,
                author.owner_principal_id(),
                owner,
                owner.version,
            )?;
            (
                virtual_membership.policy_membership_snapshot(),
                Some(owner),
                Some(collaboration_domain::MessageAuthor::from_virtual_agent(
                    virtual_membership,
                )),
            )
        } else {
            (
                memberships
                    .community
                    .ok_or_else(|| anyhow!("channel access denied"))?,
                None,
                None,
            )
        };
    let moderation = if canonical_nostr {
        let restriction = crate::admin::moderation::current_restriction_in_transaction(
            transaction,
            tenant,
            subject_principal_id,
        )
        .await?;
        let (principal_archive, community_archive) =
            crate::admin::moderation::current_actor_archives_in_transaction(
                transaction,
                tenant,
                subject_principal_id,
            )
            .await?;
        if let Some(owner) = owner_principal_id {
            let (owner_archive, _) =
                crate::admin::moderation::current_actor_archives_in_transaction(
                    transaction,
                    tenant,
                    owner,
                )
                .await?;
            if !matches!(
                owner_archive,
                collaboration_domain::ModerationSnapshot::Absent
                    | collaboration_domain::ModerationSnapshot::Current(
                        collaboration_domain::IdentityArchiveSnapshot {
                            state: collaboration_domain::IdentityArchivePolicyState::Visible,
                            ..
                        }
                    )
            ) {
                return Err(anyhow!("channel access denied"));
            }
            let owner_restriction = crate::admin::moderation::current_restriction_in_transaction(
                transaction,
                tenant,
                owner,
            )
            .await?;
            if owner_restriction
                .as_ref()
                .is_some_and(|restriction| restriction.fields().ban.is_active_at(now_millis))
            {
                return Err(anyhow!("channel access denied"));
            }
        }
        Some(CurrentMessageModeration {
            restriction,
            principal_archive,
            community_archive,
        })
    } else {
        None
    };
    let signing_public_key = match principal.kind() {
        collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } => {
            Some(*public_key.as_bytes())
        }
        collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent {
            agent_public_key,
            ..
        } => Some(*agent_public_key.as_bytes()),
        _ => None,
    };
    Ok(CurrentMessageAdmission {
        authorization: AuthorizedChannel {
            tenant: tenant.clone(),
            principal: principal.clone(),
            community_membership,
            channel_membership,
            channel_id,
            signing_public_key,
        },
        agent_author,
        owner_membership,
        message_author,
        moderation,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CurrentCommunityServingState {
    pub lifecycle: collaboration_domain::CommunityLifecycleState,
    pub version: AggregateVersion,
    pub host: collaboration_domain::CommunityHost,
    pub icon: Option<collaboration_domain::CommunityIcon>,
}

pub async fn current_community_serving_state_in_transaction(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<Option<CurrentCommunityServingState>> {
    if transaction.get_database_backend() != DatabaseBackend::Postgres {
        return Err(anyhow!("invalid community serving snapshot"));
    }
    let configured = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await?
        .ok_or_else(|| anyhow!("invalid community serving snapshot"))?;
    let configured: Option<String> = configured.try_get("", "community_id")?;
    if configured.as_deref() != Some(tenant.community_id().to_string().as_str()) {
        return Err(anyhow!("community serving access denied"));
    }
    let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "SELECT community_id, host, icon, lifecycle_state, aggregate_version::text AS aggregate_version FROM public.collaboration_communities WHERE community_id = $1 FOR SHARE",
        [tenant.community_id().as_uuid().into()])).await?;
    row.map(|row| {
        if row.try_get::<uuid::Uuid>("", "community_id")? != tenant.community_id().as_uuid() {
            return Err(anyhow!("community serving access denied"));
        }
        let lifecycle = match row.try_get::<String>("", "lifecycle_state")?.as_str() {
            "active" => collaboration_domain::CommunityLifecycleState::Active,
            "archived" => collaboration_domain::CommunityLifecycleState::Archived,
            "quiescing" => collaboration_domain::CommunityLifecycleState::Quiescing,
            "fenced" => collaboration_domain::CommunityLifecycleState::Fenced,
            "tombstone" => collaboration_domain::CommunityLifecycleState::Tombstone,
            _ => return Err(anyhow!("invalid community serving snapshot")),
        };
        Ok(CurrentCommunityServingState {
            lifecycle,
            icon: row
                .try_get::<Option<String>>("", "icon")?
                .map(collaboration_domain::CommunityIcon::new)
                .transpose()?,
            host: collaboration_domain::CommunityHost::new(row.try_get::<String>("", "host")?)?,
            version: AggregateVersion::new(
                row.try_get::<String>("", "aggregate_version")?.parse()?,
            )
            .ok_or_else(|| anyhow!("invalid community serving snapshot"))?,
        })
    })
    .transpose()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurrentPrincipalMemberships {
    pub community: Option<CommunityMembership>,
    pub channel: Option<ChannelMembership>,
}

pub async fn current_principal_memberships_in_transaction(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    channel_id: Option<AggregateId>,
) -> Result<CurrentPrincipalMemberships> {
    if principal.community_id() != tenant.community_id() {
        return Err(anyhow!("channel access denied"));
    }
    let subject_principal_id =
        collaboration_domain::authorization::authorization_subject(principal);
    current_subject_memberships_in_transaction(
        transaction,
        tenant,
        subject_principal_id,
        channel_id,
    )
    .await
}

pub(crate) async fn current_subject_memberships_in_transaction(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    subject_principal_id: collaboration_domain::PrincipalId,
    channel_id: Option<AggregateId>,
) -> Result<CurrentPrincipalMemberships> {
    if transaction.get_database_backend() != DatabaseBackend::Postgres
        || subject_principal_id.as_uuid().is_nil()
    {
        return Err(anyhow!("invalid membership snapshot"));
    }
    let configured = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await?
        .ok_or_else(|| anyhow!("invalid membership snapshot"))?;
    let configured: Option<String> = configured.try_get("", "community_id")?;
    if configured.is_some_and(|value| value != tenant.community_id().to_string()) {
        return Err(anyhow!("channel access denied"));
    }
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [tenant.community_id().to_string().into()],
        ))
        .await?;
    let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
SELECT community_membership.community_id AS community_membership_community_id,
       community_membership.principal_id AS community_membership_principal_id,
       community_membership.role AS community_role,
       community_membership.status AS community_status,
       community_membership.membership_version::text AS community_version,
       channel_membership.community_id AS channel_membership_community_id,
       channel_membership.channel_id AS channel_membership_channel_id,
       channel_membership.principal_id AS channel_membership_principal_id,
       channel_membership.role AS channel_role,
       channel_membership.status AS channel_status,
       channel_membership.membership_version::text AS channel_version
FROM (SELECT $1::uuid AS community_id, $2::uuid AS principal_id, $3::uuid AS channel_id) AS requested
LEFT JOIN public.collaboration_community_memberships AS community_membership
  ON community_membership.community_id = requested.community_id
 AND community_membership.principal_id = requested.principal_id
LEFT JOIN public.collaboration_channel_memberships AS channel_membership
  ON channel_membership.community_id = requested.community_id
 AND channel_membership.principal_id = requested.principal_id
 AND channel_membership.channel_id = requested.channel_id
"#, [tenant.community_id().as_uuid().into(), subject_principal_id.as_uuid().into(),
      channel_id.map(|id| id.as_uuid()).into()])).await?.ok_or_else(|| anyhow!("invalid membership snapshot"))?;
    let community = if let Some(community_id) =
        row.try_get::<Option<uuid::Uuid>>("", "community_membership_community_id")?
    {
        if community_id != tenant.community_id().as_uuid()
            || row.try_get::<uuid::Uuid>("", "community_membership_principal_id")?
                != subject_principal_id.as_uuid()
        {
            return Err(anyhow!("channel access denied"));
        }
        let (role, status, version) = membership_from_row(&row, false)?;
        Some(CommunityMembership {
            community_id: tenant.community_id(),
            principal_id: subject_principal_id,
            role,
            status,
            version,
        })
    } else {
        None
    };
    let channel = if let Some(stored_channel_id) =
        row.try_get::<Option<uuid::Uuid>>("", "channel_membership_channel_id")?
    {
        let channel_id = channel_id.ok_or_else(|| anyhow!("invalid membership snapshot"))?;
        if stored_channel_id != channel_id.as_uuid()
            || row.try_get::<uuid::Uuid>("", "channel_membership_community_id")?
                != tenant.community_id().as_uuid()
            || row.try_get::<uuid::Uuid>("", "channel_membership_principal_id")?
                != subject_principal_id.as_uuid()
        {
            return Err(anyhow!("channel access denied"));
        }
        let (role, status, version) = membership_from_row(&row, true)?;
        Some(ChannelMembership {
            community_id: tenant.community_id(),
            channel_id,
            principal_id: subject_principal_id,
            role,
            status,
            version,
        })
    } else {
        None
    };
    Ok(CurrentPrincipalMemberships { community, channel })
}

#[derive(Clone)]
struct LegacyChannel {
    id: u64,
    root_id: u64,
    name: String,
    visibility: String,
}

#[derive(Clone, Copy)]
struct LegacyMembership {
    user_id: u64,
    role: ChannelRole,
}

pub async fn bootstrap_canonical_channels(database: &Database) -> Result<()> {
    if database.pool.get_database_backend() != DatabaseBackend::Postgres {
        return Ok(());
    }
    let channels = load_channels(&database.pool).await?;
    let memberships = load_memberships(&database.pool).await?;
    let channels_by_root = channels.into_iter().fold(
        BTreeMap::<u64, Vec<LegacyChannel>>::new(),
        |mut grouped, channel| {
            grouped.entry(channel.root_id).or_default().push(channel);
            grouped
        },
    );
    let memberships_by_root = memberships.into_iter().fold(
        BTreeMap::<u64, Vec<LegacyMembership>>::new(),
        |mut grouped, (root_id, membership)| {
            grouped.entry(root_id).or_default().push(membership);
            grouped
        },
    );

    for (root_id, channels) in channels_by_root {
        let Some(memberships) = memberships_by_root.get(&root_id) else {
            continue;
        };
        if memberships.is_empty() {
            continue;
        }
        bootstrap_community(&database.pool, root_id, &channels, memberships).await?;
    }
    Ok(())
}

pub async fn admit_channel(
    connection: &DatabaseConnection,
    user: &User,
    claimed_community_id: CommunityId,
    claimed_channel_id: AggregateId,
) -> Result<AuthorizedChannel> {
    let route_row = connection
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
SELECT
    channel_binding.community_id,
    channel_binding.channel_id,
    channel_binding.legacy_channel_id,
    community_binding.legacy_root_channel_id,
    principal_binding.principal_id,
    principal_binding.signing_public_key
FROM public.collaboration_zed_channel_bindings AS channel_binding
JOIN public.collaboration_zed_community_bindings AS community_binding
  ON community_binding.community_id = channel_binding.community_id
JOIN public.collaboration_zed_principal_bindings AS principal_binding
  ON principal_binding.community_id = channel_binding.community_id
 AND principal_binding.legacy_user_id = $3
WHERE channel_binding.community_id = $1
  AND channel_binding.channel_id = $2
"#,
            [
                claimed_community_id.as_uuid().into(),
                claimed_channel_id.as_uuid().into(),
                user.id.0.into(),
            ],
        ))
        .await
        .context("failed to resolve trusted channel route")?
        .ok_or_else(|| anyhow!("channel access denied"))?;

    let community_id = CommunityId::from_uuid(route_row.try_get("", "community_id")?);
    let channel_id = AggregateId::from_uuid(route_row.try_get("", "channel_id")?);
    if community_id != claimed_community_id || channel_id != claimed_channel_id {
        return Err(anyhow!("channel access denied"));
    }
    let legacy_root_channel_id =
        u64::try_from(route_row.try_get::<i64>("", "legacy_root_channel_id")?)?;
    let legacy_channel_id = u64::try_from(route_row.try_get::<i64>("", "legacy_channel_id")?)?;
    if community_id_for_legacy_root_channel(legacy_root_channel_id) != community_id
        || channel_id_for_legacy_channel(legacy_channel_id) != channel_id
    {
        return Err(anyhow!("channel access denied"));
    }

    let principal_id = PrincipalId::from_uuid(route_row.try_get("", "principal_id")?);
    let signing_public_key = route_row
        .try_get::<Option<Vec<u8>>>("", "signing_public_key")?
        .map(|bytes| {
            bytes
                .try_into()
                .map_err(|_| anyhow!("invalid signing identity"))
        })
        .transpose()?;
    let expected_principal = principal_id_for_legacy_user(community_id, user.id.0 as u64);
    if principal_id != expected_principal {
        return Err(anyhow!("channel access denied"));
    }
    let route = TrustedTenantRoute::from_deployment(
        community_id,
        format!("zed-channel:{legacy_root_channel_id}"),
    )?;
    let claim =
        UntrustedTenantClaim::new(claimed_community_id, UntrustedTenantClaimSource::BodyField);
    let tenant =
        bind_rpc_tenant(Some(route), &[claim]).map_err(|_| anyhow!("channel access denied"))?;
    let scopes = PrincipalScopes::new([
        AuthorizationScope::new(CHANNEL_READ_SCOPE)?,
        AuthorizationScope::new(CHANNEL_WRITE_SCOPE)?,
    ])?;
    let service_account_id = u64::try_from(user.id.0)?;
    let principal = AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(service_account_id),
        scopes,
    );
    let transaction = connection.begin().await?;
    let memberships = current_principal_memberships_in_transaction(
        &transaction,
        &tenant,
        &principal,
        Some(channel_id),
    )
    .await?;
    let community_membership = memberships
        .community
        .ok_or_else(|| anyhow!("channel access denied"))?;
    let channel_membership = memberships
        .channel
        .ok_or_else(|| anyhow!("channel access denied"))?;
    transaction.commit().await?;
    Ok(AuthorizedChannel {
        tenant,
        principal,
        community_membership,
        channel_membership,
        channel_id,
        signing_public_key,
    })
}

pub async fn update_principal_presentation(
    connection: &DatabaseConnection,
    authorized: &AuthorizedChannel,
    user: &User,
    signing_public_key: Option<&[u8]>,
) -> Result<()> {
    if signing_public_key.is_some_and(|key| key.len() != 32) {
        return Err(anyhow!("invalid signing public key"));
    }
    let transaction = connection.begin().await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [authorized.tenant.community_id().to_string().into()],
        ))
        .await?;
    let result = transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
UPDATE public.collaboration_zed_principal_bindings
SET display_name = $3,
    avatar_url = $4,
    signing_public_key = CASE
        WHEN signing_public_key IS NULL THEN $5
        WHEN $5 IS NULL OR signing_public_key = $5 THEN signing_public_key
        ELSE signing_public_key
    END,
    updated_at = clock_timestamp()
WHERE community_id = $1 AND legacy_user_id = $2
  AND (signing_public_key IS NULL OR $5 IS NULL OR signing_public_key = $5)
"#,
            [
                authorized.tenant.community_id().as_uuid().into(),
                user.id.0.into(),
                user.username.clone().into(),
                user.avatar_url.clone().into(),
                signing_public_key.map(ToOwned::to_owned).into(),
            ],
        ))
        .await?;
    if result.rows_affected() != 1 {
        transaction.rollback().await?;
        return Err(anyhow!(
            "signing identity does not match the authenticated account"
        ));
    }
    transaction.commit().await?;
    Ok(())
}

async fn load_channels(connection: &DatabaseConnection) -> Result<Vec<LegacyChannel>> {
    let rows = connection
        .query_all(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT id::bigint AS id, name, visibility, parent_path FROM public.channels ORDER BY id",
        ))
        .await?;
    rows.into_iter().map(channel_from_row).collect()
}

fn channel_from_row(row: QueryResult) -> Result<LegacyChannel> {
    let id = u64::try_from(row.try_get::<i64>("", "id")?)?;
    let parent_path: String = row.try_get("", "parent_path")?;
    let root_id = parent_path
        .split('/')
        .find(|part| !part.is_empty())
        .map(str::parse::<u64>)
        .transpose()?
        .unwrap_or(id);
    Ok(LegacyChannel {
        id,
        root_id,
        name: row.try_get("", "name")?,
        visibility: row.try_get("", "visibility")?,
    })
}

async fn load_memberships(connection: &DatabaseConnection) -> Result<Vec<(u64, LegacyMembership)>> {
    let rows = connection
        .query_all(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT channel_id::bigint AS channel_id, user_id::bigint AS user_id, role FROM public.channel_members WHERE accepted = true ORDER BY channel_id, user_id",
        ))
        .await?;
    rows.into_iter()
        .map(|row| {
            let root_id = u64::try_from(row.try_get::<i64>("", "channel_id")?)?;
            let user_id = u64::try_from(row.try_get::<i64>("", "user_id")?)?;
            let role = match row.try_get::<String>("", "role")?.as_str() {
                "admin" => ChannelRole::Admin,
                "member" => ChannelRole::Member,
                "talker" => ChannelRole::Talker,
                "guest" => ChannelRole::Guest,
                "banned" => ChannelRole::Banned,
                _ => return Err(anyhow!("invalid legacy channel role")),
            };
            Ok((root_id, LegacyMembership { user_id, role }))
        })
        .collect()
}

async fn bootstrap_community(
    connection: &DatabaseConnection,
    root_id: u64,
    channels: &[LegacyChannel],
    memberships: &[LegacyMembership],
) -> Result<()> {
    let community_id = community_id_for_legacy_root_channel(root_id);
    let transaction = connection.begin().await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [community_id.to_string().into()],
        ))
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
INSERT INTO public.collaboration_communities (
    community_id, host, lifecycle_state, aggregate_version, source_system,
    source_record_id, source_version, source_observed_at, created_at, updated_at
) VALUES ($1, $2, 'active', 1, 'zed', $3, '1', clock_timestamp(), clock_timestamp(), clock_timestamp())
ON CONFLICT (community_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
"#,
            [
                community_id.as_uuid().into(),
                format!("zed-channel-{root_id}").into(),
                format!("channel-root:{root_id}").into(),
            ],
        ))
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            r#"
INSERT INTO public.collaboration_zed_community_bindings (legacy_root_channel_id, community_id)
VALUES ($1, $2)
ON CONFLICT (legacy_root_channel_id) DO UPDATE SET community_id = EXCLUDED.community_id
"#,
            [
                i64::try_from(root_id)?.into(),
                community_id.as_uuid().into(),
            ],
        ))
        .await?;

    let mut active_principals = BTreeSet::new();
    let current_legacy_users = memberships
        .iter()
        .map(|membership| membership.user_id)
        .collect::<BTreeSet<_>>();
    let existing_bindings = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT legacy_user_id::bigint AS legacy_user_id, principal_id FROM public.collaboration_zed_principal_bindings WHERE community_id = $1",
            [community_id.as_uuid().into()],
        ))
        .await?;
    for binding in existing_bindings {
        let legacy_user_id = u64::try_from(binding.try_get::<i64>("", "legacy_user_id")?)?;
        if current_legacy_users.contains(&legacy_user_id) {
            continue;
        }
        let principal_id: uuid::Uuid = binding.try_get("", "principal_id")?;
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
UPDATE public.collaboration_community_memberships
SET status = 'revoked', membership_version = membership_version + 1,
    updated_at = clock_timestamp()
WHERE community_id = $1 AND principal_id = $2 AND status <> 'revoked'
"#,
                [community_id.as_uuid().into(), principal_id.into()],
            ))
            .await?;
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
UPDATE public.collaboration_channel_memberships
SET status = 'revoked', membership_version = membership_version + 1,
    updated_at = clock_timestamp()
WHERE community_id = $1 AND principal_id = $2 AND status <> 'revoked'
"#,
                [community_id.as_uuid().into(), principal_id.into()],
            ))
            .await?;
    }
    for membership in memberships {
        let principal_id = principal_id_for_legacy_user(community_id, membership.user_id);
        let (role, status) = canonical_membership(membership.role);
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_community_memberships (
    community_id, principal_id, role, status, membership_version, joined_at, updated_at,
    source_system, source_record_id, source_version, source_observed_at
) VALUES ($1, $2, $3, $4, 1, clock_timestamp(), clock_timestamp(), 'zed', $5, '1', clock_timestamp())
ON CONFLICT (community_id, principal_id) DO UPDATE SET
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    membership_version = CASE
        WHEN collaboration_community_memberships.role IS DISTINCT FROM EXCLUDED.role
          OR collaboration_community_memberships.status IS DISTINCT FROM EXCLUDED.status
        THEN collaboration_community_memberships.membership_version + 1
        ELSE collaboration_community_memberships.membership_version
    END,
    updated_at = EXCLUDED.updated_at
"#,
                [
                    community_id.as_uuid().into(),
                    principal_id.as_uuid().into(),
                    role.into(),
                    status.into(),
                    format!("channel-member:{}:{}", root_id, membership.user_id).into(),
                ],
            ))
            .await?;
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_zed_principal_bindings (
    legacy_user_id, community_id, principal_id, display_name
) VALUES ($1, $2, $3, $4)
ON CONFLICT (legacy_user_id, community_id) DO NOTHING
"#,
                [
                    i64::try_from(membership.user_id)?.into(),
                    community_id.as_uuid().into(),
                    principal_id.as_uuid().into(),
                    format!("user-{}", membership.user_id).into(),
                ],
            ))
            .await?;
        if status == "active" {
            active_principals.insert(principal_id);
        }
    }
    let creator = active_principals
        .first()
        .copied()
        .ok_or_else(|| anyhow!("channel community has no active member"))?;

    for channel in channels {
        let channel_id = channel_id_for_legacy_channel(channel.id);
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_channels (
    community_id, channel_id, name, channel_type, visibility, lifecycle_state,
    creator_principal_id, channel_version, source_system, source_record_id,
    source_version, source_observed_at, created_at, updated_at
) VALUES ($1, $2, $3, 'stream', $4, 'active', $5, 1, 'zed', $6, '1', clock_timestamp(), clock_timestamp(), clock_timestamp())
ON CONFLICT (community_id, channel_id) DO UPDATE SET
    name = EXCLUDED.name, visibility = EXCLUDED.visibility, updated_at = EXCLUDED.updated_at
"#,
                [
                    community_id.as_uuid().into(),
                    channel_id.as_uuid().into(),
                    channel.name.clone().into(),
                    (if channel.visibility == "public" { "open" } else { "private" }).into(),
                    creator.as_uuid().into(),
                    format!("channel:{}", channel.id).into(),
                ],
            ))
            .await?;
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_zed_channel_bindings (legacy_channel_id, community_id, channel_id)
VALUES ($1, $2, $3)
ON CONFLICT (legacy_channel_id) DO UPDATE SET
    community_id = EXCLUDED.community_id, channel_id = EXCLUDED.channel_id
"#,
                [
                    i64::try_from(channel.id)?.into(),
                    community_id.as_uuid().into(),
                    channel_id.as_uuid().into(),
                ],
            ))
            .await?;
        for membership in memberships {
            let principal_id = principal_id_for_legacy_user(community_id, membership.user_id);
            let (role, status) = canonical_membership(membership.role);
            transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    r#"
INSERT INTO public.collaboration_channel_memberships (
    community_id, channel_id, principal_id, role, status, membership_version,
    joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at
) VALUES ($1, $2, $3, $4, $5, 1, clock_timestamp(), clock_timestamp(), 'zed', $6, '1', clock_timestamp())
ON CONFLICT (community_id, channel_id, principal_id) DO UPDATE SET
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    membership_version = CASE
        WHEN collaboration_channel_memberships.role IS DISTINCT FROM EXCLUDED.role
          OR collaboration_channel_memberships.status IS DISTINCT FROM EXCLUDED.status
        THEN collaboration_channel_memberships.membership_version + 1
        ELSE collaboration_channel_memberships.membership_version
    END,
    updated_at = EXCLUDED.updated_at
"#,
                    [
                        community_id.as_uuid().into(),
                        channel_id.as_uuid().into(),
                        principal_id.as_uuid().into(),
                        role.into(),
                        status.into(),
                        format!("channel-member:{}:{}", channel.id, membership.user_id).into(),
                    ],
                ))
                .await?;
        }
    }
    transaction.commit().await?;
    Ok(())
}

fn canonical_membership(role: ChannelRole) -> (&'static str, &'static str) {
    match role {
        ChannelRole::Admin => ("admin", "active"),
        ChannelRole::Member | ChannelRole::Talker => ("member", "active"),
        ChannelRole::Guest => ("guest", "active"),
        ChannelRole::Banned => ("guest", "revoked"),
    }
}

fn membership_from_row(
    row: &QueryResult,
    channel: bool,
) -> Result<(MembershipRole, MembershipStatus, AggregateVersion)> {
    let prefix = if channel { "channel" } else { "community" };
    let role = match row
        .try_get::<String>("", &format!("{prefix}_role"))?
        .as_str()
    {
        "owner" => MembershipRole::Owner,
        "admin" => MembershipRole::Admin,
        "member" => MembershipRole::Member,
        "guest" => MembershipRole::Guest,
        "bot" => MembershipRole::Bot,
        _ => return Err(anyhow!("invalid membership role")),
    };
    let status = match row
        .try_get::<String>("", &format!("{prefix}_status"))?
        .as_str()
    {
        "active" => MembershipStatus::Active,
        "revoked" => MembershipStatus::Revoked,
        "archived" => MembershipStatus::Archived,
        _ => return Err(anyhow!("invalid membership status")),
    };
    let version = row
        .try_get::<String>("", &format!("{prefix}_version"))?
        .parse::<u64>()?;
    let version =
        AggregateVersion::new(version).ok_or_else(|| anyhow!("invalid membership version"))?;
    Ok((role, status, version))
}

pub(crate) async fn admit_nostr_channel_transport(
    connection: &DatabaseConnection,
    profiles: &crate::identity::profile_repository::IdentityProfileRepository,
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    kind: u16,
    now_seconds: u64,
) -> Result<AuthorizedRpcRequest> {
    let transaction = connection
        .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
        .await?;
    let result = async {
        let member = current_nostr_community_for_write(
            &transaction,
            profiles,
            tenant,
            principal,
            now_seconds,
        )
        .await?;
        let scope = AuthorizationScope::new(match u32::from(kind) {
            nostr_compat::generated_kinds::KIND_NIP29_PUT_USER => "admin:channels",
            nostr_compat::generated_kinds::KIND_AGENT_PROFILE => "users:write",
            nostr_compat::generated_kinds::KIND_STREAM_MESSAGE
            | nostr_compat::generated_kinds::KIND_DELETION => "messages:write",
            _ => return Err(anyhow!("unsupported channel event")),
        })?;
        AuthorizedRpcRequest::authorize(&community_channel_request(
            tenant,
            principal,
            &scope,
            member,
            now_seconds * 1000,
        ))
        .map_err(|_| anyhow!("channel access denied"))
    }
    .await;
    match result {
        Ok(admitted) => {
            transaction.commit().await?;
            Ok(admitted)
        }
        Err(error) => {
            transaction.rollback().await?;
            Err(error)
        }
    }
}

fn community_channel_request<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    scope: &'a AuthorizationScope,
    member: CommunityMembership,
    now_millis: u64,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope: scope,
        action: AuthorizationAction::Read,
        resource: AuthorizationResource {
            community_id: tenant.community_id(),
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(tenant.community_id().as_uuid()),
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: member.version,
        community_membership: Some(member),
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis,
    }
}

async fn current_nostr_community_for_write(
    transaction: &DatabaseTransaction,
    profiles: &crate::identity::profile_repository::IdentityProfileRepository,
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    now_seconds: u64,
) -> Result<CommunityMembership> {
    crate::admin::moderation::lock_moderation_in_transaction(transaction, tenant).await?;
    let member = profiles
        .current_closed_nip42_membership_in_transaction(transaction, tenant, principal, now_seconds)
        .await?;
    let author = profiles
        .verified_message_agent_in_transaction(transaction, tenant, principal, now_seconds)
        .await?;
    for subject in std::iter::once(principal.principal_id())
        .chain(author.as_ref().map(|agent| agent.owner_principal_id()))
    {
        let restriction = crate::admin::moderation::current_restriction_in_transaction(
            transaction,
            tenant,
            subject,
        )
        .await?;
        if restriction.as_ref().is_some_and(|restriction| {
            restriction.fields().ban.is_active_at(now_seconds * 1000)
                || (subject == principal.principal_id()
                    && restriction
                        .fields()
                        .timeout
                        .is_active_at(now_seconds * 1000))
        }) {
            return Err(anyhow!("channel access denied"));
        }
    }
    Ok(member)
}

pub(crate) struct CanonicalNostrMembershipMutation {
    profiles: crate::identity::profile_repository::IdentityProfileRepository,
    events: crate::db::collaboration::event_repository::EventRepository,
}

impl CanonicalNostrMembershipMutation {
    pub(crate) fn new(connection: DatabaseConnection) -> Result<Self> {
        if !matches!(
            &connection,
            DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(crate::identity::profile_repository::IdentityProfileRepositoryError::UnsupportedDatabase.into());
        }
        Ok(Self {
            profiles: crate::identity::profile_repository::IdentityProfileRepository::new(
                DatabaseConnection::from(connection.get_postgres_connection_pool().clone()),
            )?,
            events: crate::db::collaboration::event_repository::EventRepository::new(connection)?,
        })
    }

    async fn prepare_put(
        &self,
        transaction: &DatabaseTransaction,
        command: &crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> Result<(
        collaboration_domain::Membership,
        collaboration_domain::MembershipCommandOutcome,
        Option<uuid::Uuid>,
    )> {
        use collaboration_domain::{
            Membership, MembershipRecordFields, MembershipScope, ProfileKind,
        };
        let signed = command.payload().signed_event();
        let channel = AggregateId::from_uuid(uuid::Uuid::parse_str(exact_channel_event_tag(
            &signed.event.tags,
            "h",
        )?)?);
        let key = signed
            .event
            .tags
            .iter()
            .filter(|tag| tag.first().is_some_and(|name| name == "p"))
            .filter_map(|tag| tag.get(1))
            .find_map(|value| match nostr_compat::PublicKey::from_hex(value) {
                Ok(key) => Some(collaboration_domain::NostrPublicKey::from_bytes(
                    *key.as_bytes(),
                )),
                Err(_) => None,
            })
            .ok_or_else(|| anyhow!("missing valid membership target"))?;
        let now = nostr_channel_now()?;
        crate::admin::moderation::lock_moderation_in_transaction(transaction, command.tenant())
            .await?;
        self.profiles
            .lock_nostr_channel_membership_identities_in_transaction(
                transaction,
                command.tenant(),
                command.principal(),
                key,
            )
            .await?;
        let member = current_nostr_community_for_write(
            transaction,
            &self.profiles,
            command.tenant(),
            command.principal(),
            now,
        )
        .await?;
        // The canonical community policy lock precedes this channel lock on every Nostr membership writer.
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                [format!(
                    "collaboration-channel-membership:{}:{channel}",
                    command.tenant().community_id()
                )
                .into()],
            ))
            .await?;
        let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT visibility, lifecycle_state FROM public.collaboration_channels WHERE community_id=$1 AND channel_id=$2 FOR SHARE",
            [command.tenant().community_id().as_uuid().into(), channel.as_uuid().into()])).await?.ok_or_else(||anyhow!("channel access denied"))?;
        let visibility = match row.try_get::<String>("", "visibility")?.as_str() {
            "open" => collaboration_domain::ChannelVisibility::Open,
            "private" => collaboration_domain::ChannelVisibility::Private,
            _ => return Err(anyhow!("invalid channel visibility")),
        };
        if row.try_get::<String>("", "lifecycle_state")? != "active" {
            return Err(anyhow!("channel access denied"));
        }
        let policy = self
            .profiles
            .current_channel_add_policy_in_transaction(transaction, command.tenant(), key)
            .await?;
        let target = self
            .profiles
            .current_registration_in_transaction(transaction, command.tenant(), key)
            .await?
            .ok_or_else(|| anyhow!("channel target must be canonically registered"))?;
        let agent = match target.profile().map(|profile| profile.kind()) {
            Some(ProfileKind::Agent(agent)) => Some(agent),
            _ => None,
        };
        transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT principal_id FROM public.collaboration_channel_memberships WHERE community_id=$1 AND channel_id=$2 AND (principal_id=$3 OR principal_id=$4) ORDER BY principal_id FOR UPDATE",
            [command.tenant().community_id().as_uuid().into(), channel.as_uuid().into(), command.principal().principal_id().as_uuid().into(), target.principal_id().as_uuid().into()])).await?;
        let actor = current_principal_memberships_in_transaction(
            transaction,
            command.tenant(),
            command.principal(),
            Some(channel),
        )
        .await?
        .channel
        .filter(|member| member.status == MembershipStatus::Active);
        let target_members = current_subject_memberships_in_transaction(
            transaction,
            command.tenant(),
            target.principal_id(),
            Some(channel),
        )
        .await?;
        if agent.is_none()
            && !target_members
                .community
                .is_some_and(|member| member.status == MembershipStatus::Active)
        {
            return Err(anyhow!("channel target lacks current community membership"));
        }
        let existing = target_members.channel.map(|member| MembershipRecordFields {
            community_id: member.community_id,
            scope: MembershipScope::Channel(channel),
            principal_id: member.principal_id,
            role: member.role,
            status: member.status,
            version: member.version,
            added_by_principal_id: None,
        });
        let role = optional_channel_event_tag(&signed.event.tags, "role")?
            .map(|value| match value {
                "owner" => Ok(MembershipRole::Owner),
                "admin" => Ok(MembershipRole::Admin),
                "member" => Ok(MembershipRole::Member),
                "guest" => Ok(MembershipRole::Guest),
                "bot" => Ok(MembershipRole::Bot),
                _ => Err(anyhow!("invalid membership role")),
            })
            .transpose()?;
        let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT count(*)::text AS owner_count FROM public.collaboration_channel_memberships WHERE community_id=$1 AND channel_id=$2 AND status='active' AND role='owner'",
            [command.tenant().community_id().as_uuid().into(), channel.as_uuid().into()])).await?.ok_or_else(||anyhow!("invalid channel snapshot"))?;
        let scope = AuthorizationScope::new("admin:channels")?;
        let authorization = community_channel_request(
            command.tenant(),
            command.principal(),
            &scope,
            member,
            now * 1000,
        );
        let (membership, outcome) = Membership::put_channel_member(
            collaboration_domain::membership::ChannelMemberPutRequest {
                authorization: &authorization,
                channel_id: channel,
                visibility,
                lifecycle: collaboration_domain::ChannelLifecycleState::Active,
                actor_channel_membership: actor,
                current_actor_channel_version: actor.map(|member| member.version),
                target_principal_id: target.principal_id(),
                target_public_key: key,
                existing,
                requested_role: role,
                target_policy: policy,
                target_owner_public_key: agent.and_then(|agent| agent.claimed_owner),
                current_owner_count: row.try_get::<String>("", "owner_count")?.parse()?,
            },
        )?;
        Ok((
            membership,
            outcome,
            agent.map(|_| target.profile_id().as_uuid()),
        ))
    }
}

pub(crate) fn optional_channel_event_tag<'a>(
    tags: &'a [Vec<String>],
    name: &str,
) -> Result<Option<&'a str>> {
    tags.iter()
        .find(|tag| tag.first().is_some_and(|value| value == name))
        .map(|tag| {
            tag.get(1)
                .map(String::as_str)
                .ok_or_else(|| anyhow!("invalid channel event tag"))
        })
        .transpose()
}

pub(crate) fn exact_channel_event_tag<'a>(tags: &'a [Vec<String>], name: &str) -> Result<&'a str> {
    optional_channel_event_tag(tags, name)?.ok_or_else(|| anyhow!("missing channel event tag"))
}

pub(crate) fn nostr_channel_now() -> Result<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    now.checked_mul(1000)
        .ok_or_else(|| anyhow!("invalid event time"))?;
    Ok(now)
}

#[async_trait::async_trait]
impl
    crate::db::collaboration::outbox::TransactionalCommandMutation<
        crate::nostr::event_ingest::NostrEventCommand,
    > for CanonicalNostrMembershipMutation
{
    fn transaction_isolation(&self) -> Option<sea_orm::IsolationLevel> {
        Some(sea_orm::IsolationLevel::ReadCommitted)
    }

    fn fingerprint(
        &self,
        command: &crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> std::result::Result<
        crate::db::collaboration::outbox::CommandFingerprint,
        crate::collaboration_command::DomainCommandSubmissionError,
    > {
        let mut bytes = command
            .payload()
            .signed_event()
            .event
            .canonical_bytes()
            .map_err(|_| crate::collaboration_command::DomainCommandSubmissionError::Rejected)?;
        bytes.extend_from_slice(command.payload().signed_event().signature.as_bytes());
        crate::db::collaboration::outbox::CommandFingerprint::new(
            "nostr_channel_membership",
            &bytes,
        )
    }

    async fn admit_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        command: &crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> std::result::Result<(), crate::collaboration_command::DomainCommandSubmissionError> {
        use crate::collaboration_command::DomainCommandSubmissionError;
        if !crate::nostr::event_ingest::principal_matches_author(
            command.principal().kind(),
            command.payload().signed_event().event.public_key,
        ) {
            return Err(DomainCommandSubmissionError::Rejected);
        }
        let result: Result<()> = async {
            match u32::from(command.payload().signed_event().event.kind) {
                nostr_compat::generated_kinds::KIND_NIP29_PUT_USER => {
                    self.prepare_put(transaction, command).await?;
                }
                nostr_compat::generated_kinds::KIND_AGENT_PROFILE => {
                    let member = current_nostr_community_for_write(
                        transaction,
                        &self.profiles,
                        command.tenant(),
                        command.principal(),
                        nostr_channel_now()?,
                    )
                    .await?;
                    let scope = AuthorizationScope::new("users:write")?;
                    let mut authorization = community_channel_request(
                        command.tenant(),
                        command.principal(),
                        &scope,
                        member,
                        nostr_channel_now()? * 1000,
                    );
                    authorization.action = AuthorizationAction::Write;
                    AuthorizedRpcRequest::authorize(&authorization)
                        .map_err(|_| anyhow!("profile policy access denied"))?;
                    // Validate content before reserving a receipt, including duplicate attempts.
                    crate::identity::profile_repository::validate_channel_add_policy(
                        &command.payload().signed_event().event.content,
                    )
                    .map_err(|_| anyhow!("invalid channel add policy"))?;
                }
                _ => return Err(anyhow!("unsupported channel event")),
            }
            Ok(())
        }
        .await;
        result.map_err(channel_command_error)
    }

    async fn apply(
        &self,
        transaction: &DatabaseTransaction,
        command: &crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> std::result::Result<
        crate::db::collaboration::outbox::AppliedCommand,
        crate::collaboration_command::DomainCommandSubmissionError,
    > {
        use crate::db::collaboration::{
            event_repository::{EventVerificationState, VerifiedEventRecord},
            outbox::{AppliedCommand, OutboxOperation},
            persistence_policy::{EventPersistencePolicy, PrivacyAdmission},
        };
        let result: Result<AppliedCommand> = async {
            let signed = command.payload().signed_event();
            let now = nostr_channel_now()?;
            let record = VerifiedEventRecord::new(command.tenant().community_id(), signed.clone(), EventVerificationState::Live, now * 1000, nostr_compat::TimestampPolicy::Bounded { now, max_past_seconds: 900, max_future_seconds: 900 })?;
            let version = if u32::from(signed.event.kind) == nostr_compat::generated_kinds::KIND_AGENT_PROFILE {
                let stored = self.profiles.store_channel_add_policy_in_transaction(transaction, command.tenant(), &record).await?;
                if stored == crate::db::collaboration::event_repository::EventStoreOutcome::Stale { return Err(anyhow!("stale channel add policy")); }
                AggregateVersion::FIRST
            } else {
                let (membership, outcome, profile) = self.prepare_put(transaction, command).await?;
                let fields = membership.fields();
                let collaboration_domain::MembershipScope::Channel(channel) = fields.scope else { return Err(anyhow!("invalid channel membership")); };
                self.events.store_in_transaction(transaction, command.tenant(), &record, EventPersistencePolicy::evaluate(signed.event.kind, PrivacyAdmission::author())?).await?;
                if outcome == collaboration_domain::MembershipCommandOutcome::Applied {
                    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
                        "INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,agent_profile_id,role,status,membership_version,invited_by_principal_id,joined_at,updated_at,source_system,source_record_id,source_version,source_observed_at,integrity_algorithm,integrity_value) VALUES ($1,$2,$3,$4,$5,'active',CAST($6 AS numeric),$7,clock_timestamp(),clock_timestamp(),'nostr',$8,$6,clock_timestamp(),'nostr_event_id',$8) ON CONFLICT (community_id,channel_id,principal_id) DO UPDATE SET agent_profile_id=EXCLUDED.agent_profile_id,role=EXCLUDED.role,status='active',membership_version=EXCLUDED.membership_version,updated_at=clock_timestamp(),source_system=EXCLUDED.source_system,source_record_id=EXCLUDED.source_record_id,source_version=EXCLUDED.source_version,source_observed_at=EXCLUDED.source_observed_at,integrity_algorithm=EXCLUDED.integrity_algorithm,integrity_value=EXCLUDED.integrity_value",
                        [command.tenant().community_id().as_uuid().into(), channel.as_uuid().into(), fields.principal_id.as_uuid().into(), profile.into(), membership_role_name(fields.role).into(), fields.version.to_string().into(), command.principal().principal_id().as_uuid().into(), signed.claimed_id.to_hex().into()])).await?;
                }
                fields.version
            };
            let source = signed.claimed_id.to_hex();
            let provenance = collaboration_domain::Provenance::new(collaboration_domain::SourceSystem::Nostr, collaboration_domain::SourceRecordId::new(source.clone()).ok_or_else(||anyhow!("invalid source event"))?, now * 1000).with_source_version(version.to_string()).with_integrity(collaboration_domain::IntegrityReference { algorithm: collaboration_domain::IntegrityAlgorithm::NostrEventId, value: source.clone() });
            let encoded = serde_json::to_vec(&serde_json::json!({"operation_id":command.operation_id(),"source_event_id":source,"kind":signed.event.kind}))?;
            Ok(AppliedCommand::new(version, OutboxOperation::new("nostr_channel_membership", provenance, encoded).map_err(|error|anyhow!(error))?))
        }.await;
        result.map_err(channel_command_error)
    }
}

fn membership_role_name(role: MembershipRole) -> &'static str {
    match role {
        MembershipRole::Owner => "owner",
        MembershipRole::Admin => "admin",
        MembershipRole::Member => "member",
        MembershipRole::Guest => "guest",
        MembershipRole::Bot => "bot",
    }
}

pub(crate) fn channel_command_error(
    error: anyhow::Error,
) -> crate::collaboration_command::DomainCommandSubmissionError {
    use crate::admin::moderation::ModerationRestrictionReadError;
    use crate::identity::profile_repository::IdentityProfileRepositoryError;
    use crate::nostr::auth::NostrPrincipalResolutionError;
    if error.downcast_ref::<sea_orm::DbErr>().is_some()
        || matches!(
            error.downcast_ref::<IdentityProfileRepositoryError>(),
            Some(
                IdentityProfileRepositoryError::Unavailable(_)
                    | IdentityProfileRepositoryError::EventRepository(_)
                    | IdentityProfileRepositoryError::InvalidRecord
                    | IdentityProfileRepositoryError::UnsupportedDatabase
            )
        )
        || matches!(
            error.downcast_ref::<NostrPrincipalResolutionError>(),
            Some(NostrPrincipalResolutionError::Unavailable)
        )
        || matches!(
            error.downcast_ref::<ModerationRestrictionReadError>(),
            Some(
                ModerationRestrictionReadError::Unavailable
                    | ModerationRestrictionReadError::InvalidHistory
                    | ModerationRestrictionReadError::UnsupportedDatabase
            )
        )
        || error
            .downcast_ref::<crate::db::collaboration::event_repository::EventRepositoryError>()
            .is_some()
    {
        tracing::warn!("canonical channel command infrastructure or retained event failure");
        crate::collaboration_command::DomainCommandSubmissionError::Unavailable
    } else {
        crate::collaboration_command::DomainCommandSubmissionError::Rejected
    }
}
