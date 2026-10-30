use std::{error::Error, fmt};

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResource,
    AuthorizationResourceKind, AuthorizationScope, CommunityId, CommunityMembership,
    DelegationGrant, TenantContext, authorize,
};

const COMMUNITY_MANAGE_SCOPE: &str = "communities:manage";
const MAX_COMMUNITY_HOST_BYTES: usize = 255;
const MAX_COMMUNITY_ICON_BYTES: usize = 262_144;
const MAX_JOIN_POLICY_VERSION_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct CommunityHost(String);

impl CommunityHost {
    pub fn new(value: impl Into<String>) -> Result<Self, CommunityError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_COMMUNITY_HOST_BYTES
            || value.trim() != value
            || value != value.to_ascii_lowercase()
            || value.chars().any(char::is_control)
        {
            return Err(CommunityError::InvalidHost);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CommunityHost {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct CommunityIcon(String);

impl CommunityIcon {
    pub fn new(value: impl Into<String>) -> Result<Self, CommunityError> {
        let value = value.into();
        if value.len() > MAX_COMMUNITY_ICON_BYTES || value.chars().any(char::is_control) {
            return Err(CommunityError::InvalidIcon);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CommunityIcon {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct JoinPolicyVersion(String);

impl JoinPolicyVersion {
    pub fn new(value: impl Into<String>) -> Result<Self, CommunityError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_JOIN_POLICY_VERSION_BYTES
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(CommunityError::InvalidJoinPolicyVersion);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for JoinPolicyVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommunityJoinPolicy {
    Open,
    AcceptanceRequired(JoinPolicyVersion),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommunityLifecycleState {
    Active,
    Archived,
    Quiescing,
    Fenced,
    Tombstone,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityRecordFields {
    pub community_id: CommunityId,
    pub host: CommunityHost,
    pub icon: Option<CommunityIcon>,
    pub lifecycle_state: CommunityLifecycleState,
    pub join_policy: CommunityJoinPolicy,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityCreateFields {
    pub community_id: CommunityId,
    pub host: CommunityHost,
    pub icon: Option<CommunityIcon>,
    pub join_policy: CommunityJoinPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommunityIconUpdate {
    Unchanged,
    Clear,
    Set(CommunityIcon),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityUpdate {
    pub host: Option<CommunityHost>,
    pub icon: CommunityIconUpdate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityCommandOutcome {
    Applied,
    Unchanged,
}

pub struct CommunityCommandContext<'a> {
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    current_membership_version: AggregateVersion,
    community_membership: Option<CommunityMembership>,
    delegation: Option<DelegationGrant>,
    now_millis: u64,
}

impl<'a> CommunityCommandContext<'a> {
    pub fn new(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        current_membership_version: AggregateVersion,
        community_membership: Option<CommunityMembership>,
        now_millis: u64,
    ) -> Self {
        Self {
            tenant,
            principal,
            current_membership_version,
            community_membership,
            delegation: None,
            now_millis,
        }
    }

    pub fn with_delegation(mut self, delegation: DelegationGrant) -> Self {
        self.delegation = Some(delegation);
        self
    }

    fn authorize(
        &self,
        community_id: CommunityId,
        action: AuthorizationAction,
    ) -> Result<(), CommunityError> {
        let required_scope = AuthorizationScope::new(COMMUNITY_MANAGE_SCOPE)
            .map_err(|_| CommunityError::AuthorizationConfiguration)?;
        let resource_id = AggregateId::from_uuid(community_id.as_uuid());
        let request = AuthorizationRequest {
            tenant: self.tenant,
            principal: self.principal,
            required_scope: &required_scope,
            action,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Community,
                resource_id,
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: self.current_membership_version,
            community_membership: self.community_membership,
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: self.delegation,
            now_millis: self.now_millis,
        };
        match authorize(&request) {
            AuthorizationDecision::Allowed => Ok(()),
            AuthorizationDecision::Denied(denial) => Err(CommunityError::Unauthorized(denial)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Community {
    fields: CommunityRecordFields,
}

impl Community {
    pub fn from_record(fields: CommunityRecordFields) -> Self {
        Self { fields }
    }

    pub fn create(
        fields: CommunityCreateFields,
        context: &CommunityCommandContext<'_>,
    ) -> Result<Self, CommunityError> {
        context.authorize(fields.community_id, AuthorizationAction::Delete)?;
        Ok(Self {
            fields: CommunityRecordFields {
                community_id: fields.community_id,
                host: fields.host,
                icon: fields.icon,
                lifecycle_state: CommunityLifecycleState::Active,
                join_policy: fields.join_policy,
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn fields(&self) -> &CommunityRecordFields {
        &self.fields
    }

    pub fn update(
        &mut self,
        expected_version: AggregateVersion,
        update: CommunityUpdate,
        context: &CommunityCommandContext<'_>,
    ) -> Result<CommunityCommandOutcome, CommunityError> {
        context.authorize(self.fields.community_id, AuthorizationAction::Manage)?;
        self.require_version(expected_version)?;
        self.require_active()?;
        if update.host.is_none() && update.icon == CommunityIconUpdate::Unchanged {
            return Err(CommunityError::EmptyUpdate);
        }

        let host_changed = update
            .host
            .as_ref()
            .is_some_and(|host| host != &self.fields.host);
        let icon_changed = match &update.icon {
            CommunityIconUpdate::Unchanged => false,
            CommunityIconUpdate::Clear => self.fields.icon.is_some(),
            CommunityIconUpdate::Set(icon) => self.fields.icon.as_ref() != Some(icon),
        };
        if !host_changed && !icon_changed {
            return Ok(CommunityCommandOutcome::Unchanged);
        }

        if let Some(host) = update.host {
            self.fields.host = host;
        }
        match update.icon {
            CommunityIconUpdate::Unchanged => {}
            CommunityIconUpdate::Clear => self.fields.icon = None,
            CommunityIconUpdate::Set(icon) => self.fields.icon = Some(icon),
        }
        self.advance_version()?;
        Ok(CommunityCommandOutcome::Applied)
    }

    pub fn set_join_policy(
        &mut self,
        expected_version: AggregateVersion,
        join_policy: CommunityJoinPolicy,
        context: &CommunityCommandContext<'_>,
    ) -> Result<CommunityCommandOutcome, CommunityError> {
        context.authorize(self.fields.community_id, AuthorizationAction::Manage)?;
        self.require_version(expected_version)?;
        self.require_active()?;
        if self.fields.join_policy == join_policy {
            return Ok(CommunityCommandOutcome::Unchanged);
        }
        self.fields.join_policy = join_policy;
        self.advance_version()?;
        Ok(CommunityCommandOutcome::Applied)
    }

    pub fn archive(
        &mut self,
        expected_version: AggregateVersion,
        context: &CommunityCommandContext<'_>,
    ) -> Result<CommunityCommandOutcome, CommunityError> {
        context.authorize(self.fields.community_id, AuthorizationAction::Delete)?;
        self.require_version(expected_version)?;
        match self.fields.lifecycle_state {
            CommunityLifecycleState::Active => {
                self.fields.lifecycle_state = CommunityLifecycleState::Archived;
                self.advance_version()?;
                Ok(CommunityCommandOutcome::Applied)
            }
            CommunityLifecycleState::Archived => Ok(CommunityCommandOutcome::Unchanged),
            CommunityLifecycleState::Quiescing
            | CommunityLifecycleState::Fenced
            | CommunityLifecycleState::Tombstone => Err(CommunityError::InvalidTransition),
        }
    }

    pub fn restore(
        &mut self,
        expected_version: AggregateVersion,
        context: &CommunityCommandContext<'_>,
    ) -> Result<CommunityCommandOutcome, CommunityError> {
        context.authorize(self.fields.community_id, AuthorizationAction::Delete)?;
        self.require_version(expected_version)?;
        match self.fields.lifecycle_state {
            CommunityLifecycleState::Archived => {
                self.fields.lifecycle_state = CommunityLifecycleState::Active;
                self.advance_version()?;
                Ok(CommunityCommandOutcome::Applied)
            }
            CommunityLifecycleState::Active => Ok(CommunityCommandOutcome::Unchanged),
            CommunityLifecycleState::Quiescing
            | CommunityLifecycleState::Fenced
            | CommunityLifecycleState::Tombstone => Err(CommunityError::InvalidTransition),
        }
    }

    fn require_version(&self, expected_version: AggregateVersion) -> Result<(), CommunityError> {
        if self.fields.version != expected_version {
            return Err(CommunityError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_active(&self) -> Result<(), CommunityError> {
        if self.fields.lifecycle_state != CommunityLifecycleState::Active {
            return Err(CommunityError::InvalidTransition);
        }
        Ok(())
    }

    fn advance_version(&mut self) -> Result<(), CommunityError> {
        self.fields.version = self
            .fields
            .version
            .next()
            .ok_or(CommunityError::VersionExhausted)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityError {
    InvalidHost,
    InvalidIcon,
    InvalidJoinPolicyVersion,
    EmptyUpdate,
    Unauthorized(AuthorizationDenial),
    AuthorizationConfiguration,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTransition,
    VersionExhausted,
}

impl fmt::Display for CommunityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHost => formatter.write_str("community host is invalid"),
            Self::InvalidIcon => formatter.write_str("community icon is invalid"),
            Self::InvalidJoinPolicyVersion => {
                formatter.write_str("community join-policy version is invalid")
            }
            Self::EmptyUpdate => formatter.write_str("community update is empty"),
            Self::Unauthorized(_) | Self::AuthorizationConfiguration => {
                formatter.write_str("community command is not authorized")
            }
            Self::StaleVersion { .. } => formatter.write_str("community version is stale"),
            Self::InvalidTransition => formatter.write_str("community transition is invalid"),
            Self::VersionExhausted => formatter.write_str("community version is exhausted"),
        }
    }
}

impl Error for CommunityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MembershipRole, MembershipStatus, PrincipalId, PrincipalScopes, ServiceAccountId,
        TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn principal_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(2))
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_direct_host(community_id(), "community.example")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn principal() -> AuthenticatedPrincipal {
        let scope = AuthorizationScope::new(COMMUNITY_MANAGE_SCOPE).expect("scope");
        let scopes = PrincipalScopes::new([scope]).expect("scopes");
        AuthenticatedPrincipal::zed_account(
            principal_id(),
            community_id(),
            ServiceAccountId::new(1),
            scopes,
        )
    }

    fn membership(role: MembershipRole) -> CommunityMembership {
        CommunityMembership {
            community_id: community_id(),
            principal_id: principal_id(),
            role,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }
    }

    fn context<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        role: MembershipRole,
    ) -> CommunityCommandContext<'a> {
        CommunityCommandContext::new(
            tenant,
            principal,
            AggregateVersion::FIRST,
            Some(membership(role)),
            100,
        )
    }

    fn create_fields() -> CommunityCreateFields {
        CommunityCreateFields {
            community_id: community_id(),
            host: CommunityHost::new("community.example").expect("host"),
            icon: None,
            join_policy: CommunityJoinPolicy::Open,
        }
    }

    fn owner_community(tenant: &TenantContext, principal: &AuthenticatedPrincipal) -> Community {
        Community::create(
            create_fields(),
            &context(tenant, principal, MembershipRole::Owner),
        )
        .expect("community")
    }

    #[test]
    fn owner_creates_updates_and_transitions_join_policy() {
        let tenant = tenant();
        let principal = principal();
        let owner = context(&tenant, &principal, MembershipRole::Owner);
        let mut community = Community::create(create_fields(), &owner).expect("create community");
        assert_eq!(community.fields().version, AggregateVersion::FIRST);

        assert_eq!(
            community
                .update(
                    AggregateVersion::FIRST,
                    CommunityUpdate {
                        host: None,
                        icon: CommunityIconUpdate::Set(
                            CommunityIcon::new("https://community.example/icon.png").expect("icon"),
                        ),
                    },
                    &owner,
                )
                .expect("update"),
            CommunityCommandOutcome::Applied
        );
        let second = AggregateVersion::FIRST.next().expect("second");
        let policy = CommunityJoinPolicy::AcceptanceRequired(
            JoinPolicyVersion::new("policy-sha256-v1").expect("policy version"),
        );
        assert_eq!(
            community
                .set_join_policy(second, policy.clone(), &owner)
                .expect("set policy"),
            CommunityCommandOutcome::Applied
        );
        let third = second.next().expect("third");
        assert_eq!(community.fields().join_policy, policy);
        assert_eq!(
            community
                .set_join_policy(third, community.fields().join_policy.clone(), &owner)
                .expect("idempotent policy"),
            CommunityCommandOutcome::Unchanged
        );
        assert_eq!(community.fields().version, third);
    }

    #[test]
    fn stale_update_does_not_mutate_community() {
        let tenant = tenant();
        let principal = principal();
        let owner = context(&tenant, &principal, MembershipRole::Owner);
        let mut community = owner_community(&tenant, &principal);
        community
            .archive(AggregateVersion::FIRST, &owner)
            .expect("archive");
        let before = community.clone();

        assert!(matches!(
            community.restore(AggregateVersion::FIRST, &owner),
            Err(CommunityError::StaleVersion { .. })
        ));
        assert_eq!(community, before);
    }

    #[test]
    fn only_delete_authority_can_archive_or_restore() {
        let tenant = tenant();
        let principal = principal();
        let mut community = owner_community(&tenant, &principal);
        let admin = context(&tenant, &principal, MembershipRole::Admin);
        let before = community.clone();

        assert_eq!(
            community.archive(AggregateVersion::FIRST, &admin),
            Err(CommunityError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );
        assert_eq!(community, before);
    }

    #[test]
    fn archive_and_restore_reject_protected_lifecycle_states() {
        let tenant = tenant();
        let principal = principal();
        let owner = context(&tenant, &principal, MembershipRole::Owner);
        let mut community = owner_community(&tenant, &principal);
        assert_eq!(
            community
                .archive(AggregateVersion::FIRST, &owner)
                .expect("archive"),
            CommunityCommandOutcome::Applied
        );
        let second = AggregateVersion::FIRST.next().expect("second");
        assert_eq!(
            community.restore(second, &owner).expect("restore"),
            CommunityCommandOutcome::Applied
        );

        let third = second.next().expect("third");
        let mut fenced = Community::from_record(CommunityRecordFields {
            community_id: community_id(),
            host: CommunityHost::new("community.example").expect("host"),
            icon: None,
            lifecycle_state: CommunityLifecycleState::Fenced,
            join_policy: CommunityJoinPolicy::Open,
            version: third,
        });
        assert_eq!(
            fenced.archive(third, &owner),
            Err(CommunityError::InvalidTransition)
        );
    }
}
