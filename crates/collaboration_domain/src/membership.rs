use std::{error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    ChannelMembership, CommunityId, CommunityMembership, InviteRedemption, MembershipRole,
    MembershipStatus, PrincipalId, VirtualAgentMembershipEvidence, authorize,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipScope {
    Community,
    Channel(AggregateId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MembershipRecordFields {
    pub community_id: CommunityId,
    pub scope: MembershipScope,
    pub principal_id: PrincipalId,
    pub role: MembershipRole,
    pub status: MembershipStatus,
    pub version: AggregateVersion,
    pub added_by_principal_id: Option<PrincipalId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MembershipCreateFields {
    pub community_id: CommunityId,
    pub scope: MembershipScope,
    pub principal_id: PrincipalId,
    pub role: MembershipRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentChannelAddPolicy {
    Anyone,
    OwnerOnly,
    Nobody,
}

pub struct ChannelMemberPutRequest<'a> {
    pub authorization: &'a AuthorizationRequest<'a>,
    pub channel_id: AggregateId,
    pub visibility: crate::ChannelVisibility,
    pub lifecycle: crate::ChannelLifecycleState,
    pub actor_channel_membership: Option<ChannelMembership>,
    pub current_actor_channel_version: Option<AggregateVersion>,
    pub target_principal_id: PrincipalId,
    pub target_public_key: crate::NostrPublicKey,
    pub existing: Option<MembershipRecordFields>,
    pub requested_role: Option<MembershipRole>,
    pub target_policy: AgentChannelAddPolicy,
    pub target_owner_public_key: Option<crate::NostrPublicKey>,
    pub current_owner_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipPolicyInput {
    PersistentCommunity(CommunityMembership),
    PersistentChannel(ChannelMembership),
    VirtualCommunity(CommunityMembership),
}

impl MembershipPolicyInput {
    pub const fn community_membership(self) -> Option<CommunityMembership> {
        match self {
            Self::PersistentCommunity(membership) | Self::VirtualCommunity(membership) => {
                Some(membership)
            }
            Self::PersistentChannel(_) => None,
        }
    }

    pub const fn channel_membership(self) -> Option<ChannelMembership> {
        match self {
            Self::PersistentChannel(membership) => Some(membership),
            Self::PersistentCommunity(_) | Self::VirtualCommunity(_) => None,
        }
    }

    pub const fn is_virtual(self) -> bool {
        matches!(self, Self::VirtualCommunity(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InviteMembershipProjection {
    membership: Membership,
    inserted: bool,
}

impl InviteMembershipProjection {
    pub const fn membership(self) -> Membership {
        self.membership
    }

    pub const fn inserted(self) -> bool {
        self.inserted
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Membership {
    fields: MembershipRecordFields,
}

impl Membership {
    pub const fn from_record(fields: MembershipRecordFields) -> Self {
        Self { fields }
    }

    pub fn add(
        fields: MembershipCreateFields,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, MembershipError> {
        authorize_membership(authorization, fields.community_id, fields.scope)?;
        require_distinct_actor(authorization, fields.principal_id)?;
        require_role_grant(actor_role(authorization, fields.scope)?, fields.role)?;
        Ok(Self {
            fields: MembershipRecordFields {
                community_id: fields.community_id,
                scope: fields.scope,
                principal_id: fields.principal_id,
                role: fields.role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
                added_by_principal_id: Some(authorization_subject(authorization)),
            },
        })
    }

    pub fn put_channel_member(
        request: ChannelMemberPutRequest<'_>,
    ) -> Result<(Self, MembershipCommandOutcome), MembershipError> {
        let authorization = request.authorization;
        if authorization.action != AuthorizationAction::Read
            || authorization.required_scope.as_str() != "admin:channels"
            || authorization.resource.kind != AuthorizationResourceKind::Community
            || authorization.resource.channel_id.is_some()
            || authorization.resource.resource_id
                != AggregateId::from_uuid(authorization.tenant.community_id().as_uuid())
            || request.channel_id.as_uuid().is_nil()
            || request.target_principal_id.as_uuid().is_nil()
            || request.lifecycle != crate::ChannelLifecycleState::Active
        {
            return Err(MembershipError::AuthorizationShape);
        }
        if let AuthorizationDecision::Denied(denial) = authorize(authorization) {
            return Err(MembershipError::Unauthorized(denial));
        }
        let actor_key = match authorization.principal.kind() {
            AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } => *public_key,
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                agent_public_key, ..
            } => *agent_public_key,
            _ => return Err(MembershipError::AuthorizationShape),
        };
        if (actor_key == request.target_public_key)
            != (authorization_subject(authorization) == request.target_principal_id)
        {
            return Err(MembershipError::AuthorizationShape);
        }
        let actor = request.actor_channel_membership;
        if let Some(actor) = actor {
            if actor.community_id != authorization.tenant.community_id()
                || actor.channel_id != request.channel_id
                || actor.principal_id != authorization_subject(authorization)
                || actor.status != MembershipStatus::Active
                || Some(actor.version) != request.current_actor_channel_version
            {
                return Err(MembershipError::AuthorizationShape);
            }
        } else if request.current_actor_channel_version.is_some() {
            return Err(MembershipError::AuthorizationShape);
        }
        if request.visibility == crate::ChannelVisibility::Private && actor.is_none() {
            return Err(MembershipError::MissingActorMembership);
        }
        let existing = request.existing;
        if existing.is_some_and(|member| {
            member.community_id != authorization.tenant.community_id()
                || member.scope != MembershipScope::Channel(request.channel_id)
                || member.principal_id != request.target_principal_id
        }) {
            return Err(MembershipError::AuthorizationShape);
        }
        let active = existing.filter(|member| member.status == MembershipStatus::Active);
        let role = request
            .requested_role
            .unwrap_or_else(|| active.map_or(MembershipRole::Member, |member| member.role));
        let elevated = actor.is_some_and(|member| {
            matches!(member.role, MembershipRole::Owner | MembershipRole::Admin)
        });
        if matches!(role, MembershipRole::Owner | MembershipRole::Admin) && !elevated {
            return Err(MembershipError::RoleEscalation);
        }
        if active.is_some_and(|member| member.role != role) && !elevated {
            return Err(MembershipError::RoleEscalation);
        }
        if active.is_some_and(|member| member.role == MembershipRole::Owner)
            && role != MembershipRole::Owner
            && request.current_owner_count <= 1
        {
            return Err(MembershipError::ProtectedOwner);
        }
        if actor_key != request.target_public_key {
            match request.target_policy {
                AgentChannelAddPolicy::Anyone => {}
                AgentChannelAddPolicy::OwnerOnly
                    if request.target_owner_public_key == Some(actor_key) => {}
                _ => return Err(MembershipError::RoleEscalation),
            }
        }
        if let Some(fields) = active.filter(|member| member.role == role) {
            return Ok((
                Self::from_record(fields),
                MembershipCommandOutcome::Unchanged,
            ));
        }
        let version = match existing {
            Some(member) => member
                .version
                .next()
                .ok_or(MembershipError::VersionExhausted)?,
            None => AggregateVersion::FIRST,
        };
        Ok((
            Self::from_record(MembershipRecordFields {
                community_id: authorization.tenant.community_id(),
                scope: MembershipScope::Channel(request.channel_id),
                principal_id: request.target_principal_id,
                role,
                status: MembershipStatus::Active,
                version,
                added_by_principal_id: Some(authorization_subject(authorization)),
            }),
            MembershipCommandOutcome::Applied,
        ))
    }

    pub fn from_invite_redemption(
        redemption: &InviteRedemption,
        added_by_principal_id: Option<PrincipalId>,
    ) -> InviteMembershipProjection {
        let source = redemption.membership();
        let inserted = matches!(redemption, InviteRedemption::Joined { .. });
        InviteMembershipProjection {
            membership: Self::from_record(MembershipRecordFields {
                community_id: source.community_id,
                scope: MembershipScope::Community,
                principal_id: source.principal_id,
                role: source.role,
                status: source.status,
                version: source.version,
                added_by_principal_id,
            }),
            inserted,
        }
    }

    pub const fn virtual_policy_input(
        evidence: VirtualAgentMembershipEvidence,
    ) -> MembershipPolicyInput {
        MembershipPolicyInput::VirtualCommunity(evidence.policy_membership_snapshot())
    }

    pub const fn fields(self) -> MembershipRecordFields {
        self.fields
    }

    pub const fn policy_input(self) -> MembershipPolicyInput {
        match self.fields.scope {
            MembershipScope::Community => {
                MembershipPolicyInput::PersistentCommunity(CommunityMembership {
                    community_id: self.fields.community_id,
                    principal_id: self.fields.principal_id,
                    role: self.fields.role,
                    status: self.fields.status,
                    version: self.fields.version,
                })
            }
            MembershipScope::Channel(channel_id) => {
                MembershipPolicyInput::PersistentChannel(ChannelMembership {
                    community_id: self.fields.community_id,
                    channel_id,
                    principal_id: self.fields.principal_id,
                    role: self.fields.role,
                    status: self.fields.status,
                    version: self.fields.version,
                })
            }
        }
    }

    pub fn change_role(
        &mut self,
        expected_version: AggregateVersion,
        new_role: MembershipRole,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MembershipCommandOutcome, MembershipError> {
        self.authorize_mutation(authorization)?;
        self.require_version(expected_version)?;
        self.require_active()?;
        require_distinct_actor(authorization, self.fields.principal_id)?;
        let actor_role = actor_role(authorization, self.fields.scope)?;
        if actor_role != MembershipRole::Owner {
            return Err(MembershipError::RoleEscalation);
        }
        if new_role == MembershipRole::Owner
            || (self.fields.scope == MembershipScope::Community
                && self.fields.role == MembershipRole::Owner)
        {
            return Err(MembershipError::ProtectedOwner);
        }
        if self.fields.role == new_role {
            return Ok(MembershipCommandOutcome::Unchanged);
        }
        self.fields.role = new_role;
        self.advance_version()?;
        Ok(MembershipCommandOutcome::Applied)
    }

    pub fn revoke(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MembershipCommandOutcome, MembershipError> {
        self.authorize_mutation(authorization)?;
        self.require_version(expected_version)?;
        require_distinct_actor(authorization, self.fields.principal_id)?;
        self.require_removable_by(actor_role(authorization, self.fields.scope)?)?;
        match self.fields.status {
            MembershipStatus::Active | MembershipStatus::Archived => {
                self.fields.status = MembershipStatus::Revoked;
                self.advance_version()?;
                Ok(MembershipCommandOutcome::Applied)
            }
            MembershipStatus::Revoked => Ok(MembershipCommandOutcome::Unchanged),
        }
    }

    pub fn archive(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MembershipCommandOutcome, MembershipError> {
        self.authorize_mutation(authorization)?;
        self.require_version(expected_version)?;
        require_distinct_actor(authorization, self.fields.principal_id)?;
        self.require_removable_by(actor_role(authorization, self.fields.scope)?)?;
        match self.fields.status {
            MembershipStatus::Active => {
                self.fields.status = MembershipStatus::Archived;
                self.advance_version()?;
                Ok(MembershipCommandOutcome::Applied)
            }
            MembershipStatus::Archived => Ok(MembershipCommandOutcome::Unchanged),
            MembershipStatus::Revoked => Err(MembershipError::InvalidTransition),
        }
    }

    pub fn restore(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MembershipCommandOutcome, MembershipError> {
        self.authorize_mutation(authorization)?;
        self.require_version(expected_version)?;
        require_distinct_actor(authorization, self.fields.principal_id)?;
        self.require_removable_by(actor_role(authorization, self.fields.scope)?)?;
        match self.fields.status {
            MembershipStatus::Archived => {
                self.fields.status = MembershipStatus::Active;
                self.advance_version()?;
                Ok(MembershipCommandOutcome::Applied)
            }
            MembershipStatus::Active => Ok(MembershipCommandOutcome::Unchanged),
            MembershipStatus::Revoked => Err(MembershipError::InvalidTransition),
        }
    }

    fn authorize_mutation(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<(), MembershipError> {
        authorize_membership(authorization, self.fields.community_id, self.fields.scope)
    }

    fn require_version(&self, expected_version: AggregateVersion) -> Result<(), MembershipError> {
        if self.fields.version != expected_version {
            return Err(MembershipError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_active(&self) -> Result<(), MembershipError> {
        if self.fields.status != MembershipStatus::Active {
            return Err(MembershipError::InvalidTransition);
        }
        Ok(())
    }

    fn require_removable_by(&self, actor_role: MembershipRole) -> Result<(), MembershipError> {
        if self.fields.scope == MembershipScope::Community
            && self.fields.role == MembershipRole::Owner
        {
            return Err(MembershipError::ProtectedOwner);
        }
        let allowed = match actor_role {
            MembershipRole::Owner => true,
            MembershipRole::Admin => matches!(
                self.fields.role,
                MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot
            ),
            MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot => false,
        };
        if !allowed {
            return Err(MembershipError::RoleEscalation);
        }
        Ok(())
    }

    fn advance_version(&mut self) -> Result<(), MembershipError> {
        self.fields.version = self
            .fields
            .version
            .next()
            .ok_or(MembershipError::VersionExhausted)?;
        Ok(())
    }
}

fn authorize_membership(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    scope: MembershipScope,
) -> Result<(), MembershipError> {
    if request.action != AuthorizationAction::Manage
        || request.resource.community_id != community_id
    {
        return Err(MembershipError::AuthorizationShape);
    }
    let resource_matches = match scope {
        MembershipScope::Community => {
            request.resource.kind == AuthorizationResourceKind::Community
                && request.resource.resource_id == AggregateId::from_uuid(community_id.as_uuid())
                && request.resource.channel_id.is_none()
        }
        MembershipScope::Channel(channel_id) => {
            request.resource.kind == AuthorizationResourceKind::Channel
                && request.resource.resource_id == channel_id
                && request.resource.channel_id == Some(channel_id)
        }
    };
    if !resource_matches {
        return Err(MembershipError::AuthorizationShape);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(MembershipError::Unauthorized(denial)),
    }
}

fn actor_role(
    request: &AuthorizationRequest<'_>,
    scope: MembershipScope,
) -> Result<MembershipRole, MembershipError> {
    let role = match scope {
        MembershipScope::Community => request
            .community_membership
            .map(|membership| membership.role),
        MembershipScope::Channel(_) => request.channel_membership.map(|membership| membership.role),
    };
    role.ok_or(MembershipError::MissingActorMembership)
}

fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

fn require_distinct_actor(
    request: &AuthorizationRequest<'_>,
    target_principal_id: PrincipalId,
) -> Result<(), MembershipError> {
    if authorization_subject(request) == target_principal_id {
        return Err(MembershipError::SelfMutation);
    }
    Ok(())
}

fn require_role_grant(
    actor_role: MembershipRole,
    target_role: MembershipRole,
) -> Result<(), MembershipError> {
    let allowed = match actor_role {
        MembershipRole::Owner => target_role != MembershipRole::Owner,
        MembershipRole::Admin => matches!(
            target_role,
            MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot
        ),
        MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot => false,
    };
    if !allowed {
        return Err(if target_role == MembershipRole::Owner {
            MembershipError::ProtectedOwner
        } else {
            MembershipError::RoleEscalation
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipError {
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    MissingActorMembership,
    SelfMutation,
    ProtectedOwner,
    RoleEscalation,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTransition,
    VersionExhausted,
}

impl fmt::Display for MembershipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorizationShape | Self::Unauthorized(_) | Self::MissingActorMembership => {
                formatter.write_str("membership command is not authorized")
            }
            Self::SelfMutation => formatter.write_str("membership self-mutation is forbidden"),
            Self::ProtectedOwner => formatter.write_str("owner membership is protected"),
            Self::RoleEscalation => formatter.write_str("membership role change is forbidden"),
            Self::StaleVersion { .. } => formatter.write_str("membership version is stale"),
            Self::InvalidTransition => formatter.write_str("membership transition is invalid"),
            Self::VersionExhausted => formatter.write_str("membership version is exhausted"),
        }
    }
}

impl Error for MembershipError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthorizationResource, AuthorizationScope, InviteAdmissionEvidence, InviteId,
        PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("communities:manage").expect("scope")
    }

    fn principal_scopes() -> PrincipalScopes {
        PrincipalScopes::new([scope()]).expect("scopes")
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "membership-test").expect("route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn authenticated(
        community_id: CommunityId,
        actor_principal_id: PrincipalId,
    ) -> crate::AuthenticatedPrincipal {
        crate::AuthenticatedPrincipal::zed_account(
            actor_principal_id,
            community_id,
            ServiceAccountId::new(1),
            principal_scopes(),
        )
    }

    fn policy_membership(
        community_id: CommunityId,
        actor_principal_id: PrincipalId,
        role: MembershipRole,
        version: AggregateVersion,
    ) -> CommunityMembership {
        CommunityMembership {
            community_id,
            principal_id: actor_principal_id,
            role,
            status: MembershipStatus::Active,
            version,
        }
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a crate::AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        membership: CommunityMembership,
        current_version: AggregateVersion,
    ) -> AuthorizationRequest<'a> {
        let community_id = tenant.community_id();
        AuthorizationRequest {
            tenant,
            principal: authenticated,
            required_scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id.as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: current_version,
            community_membership: Some(membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn active_member(
        community_id: CommunityId,
        target_principal_id: PrincipalId,
        role: MembershipRole,
    ) -> Membership {
        Membership::from_record(MembershipRecordFields {
            community_id,
            scope: MembershipScope::Community,
            principal_id: target_principal_id,
            role,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
            added_by_principal_id: None,
        })
    }

    #[test]
    fn signed_channel_put_ordinary_self_policy_and_role_fences() {
        let community = community(1);
        let actor_id = principal(2);
        let target_id = principal(3);
        let channel = aggregate(4);
        let tenant = tenant(community);
        let scope = AuthorizationScope::new("admin:channels").expect("scope");
        let key = crate::NostrPublicKey::from_bytes([2; 32]);
        let target_key = crate::NostrPublicKey::from_bytes([3; 32]);
        let actor = crate::AuthenticatedPrincipal::nostr_identity(
            actor_id,
            community,
            key,
            crate::NostrAuthenticationMethod::Nip42,
            PrincipalScopes::new([scope.clone()]).expect("scopes"),
        );
        let community_member = policy_membership(
            community,
            actor_id,
            MembershipRole::Member,
            AggregateVersion::FIRST,
        );
        let mut authorization = request(
            &tenant,
            &actor,
            &scope,
            community_member,
            AggregateVersion::FIRST,
        );
        authorization.action = AuthorizationAction::Read;
        let actor_member = ChannelMembership {
            community_id: community,
            channel_id: channel,
            principal_id: actor_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        let make = |actor_channel_membership: Option<ChannelMembership>,
                    target,
                    requested_role,
                    policy| ChannelMemberPutRequest {
            authorization: &authorization,
            channel_id: channel,
            visibility: crate::ChannelVisibility::Private,
            lifecycle: crate::ChannelLifecycleState::Active,
            actor_channel_membership,
            current_actor_channel_version: actor_channel_membership.map(|member| member.version),
            target_principal_id: target,
            target_public_key: if target == actor_id { key } else { target_key },
            existing: None,
            requested_role,
            target_policy: policy,
            target_owner_public_key: None,
            current_owner_count: 1,
        };
        for actor_role in [
            MembershipRole::Member,
            MembershipRole::Guest,
            MembershipRole::Bot,
        ] {
            for granted in [
                MembershipRole::Member,
                MembershipRole::Guest,
                MembershipRole::Bot,
            ] {
                let mut current = actor_member;
                current.role = actor_role;
                let (added, outcome) = Membership::put_channel_member(make(
                    Some(current),
                    target_id,
                    Some(granted),
                    AgentChannelAddPolicy::Anyone,
                ))
                .expect("ordinary private add");
                assert_eq!(added.fields().role, granted);
                assert_eq!(outcome, MembershipCommandOutcome::Applied);
            }
        }
        assert!(
            Membership::put_channel_member(make(
                None,
                target_id,
                None,
                AgentChannelAddPolicy::Anyone
            ))
            .is_err()
        );
        assert!(
            Membership::put_channel_member(make(
                Some(actor_member),
                target_id,
                Some(MembershipRole::Admin),
                AgentChannelAddPolicy::Anyone
            ))
            .is_err()
        );
        assert!(
            Membership::put_channel_member(make(
                Some(actor_member),
                target_id,
                None,
                AgentChannelAddPolicy::Nobody
            ))
            .is_err()
        );
        for elevated_role in [MembershipRole::Owner, MembershipRole::Admin] {
            let mut elevated_actor = actor_member;
            elevated_actor.role = elevated_role;
            let granted = Membership::put_channel_member(make(
                Some(elevated_actor),
                target_id,
                Some(MembershipRole::Owner),
                AgentChannelAddPolicy::Anyone,
            ))
            .expect("actual elevated owner grant");
            assert_eq!(granted.0.fields().role, MembershipRole::Owner);
        }
        let mut own = make(
            Some(actor_member),
            target_id,
            None,
            AgentChannelAddPolicy::OwnerOnly,
        );
        own.target_owner_public_key = Some(key);
        assert!(Membership::put_channel_member(own).is_ok());
        assert!(
            Membership::put_channel_member(make(
                Some(actor_member),
                target_id,
                None,
                AgentChannelAddPolicy::OwnerOnly
            ))
            .is_err()
        );
        let mut self_add = make(None, actor_id, None, AgentChannelAddPolicy::Nobody);
        self_add.visibility = crate::ChannelVisibility::Open;
        assert!(Membership::put_channel_member(self_add).is_ok());
        let mut existing_owner = actor_member;
        existing_owner.role = MembershipRole::Owner;
        let existing = MembershipRecordFields {
            community_id: community,
            scope: MembershipScope::Channel(channel),
            principal_id: actor_id,
            role: MembershipRole::Owner,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
            added_by_principal_id: Some(actor_id),
        };
        let mut preserved = make(
            Some(existing_owner),
            actor_id,
            None,
            AgentChannelAddPolicy::Nobody,
        );
        preserved.existing = Some(existing);
        let (unchanged, outcome) =
            Membership::put_channel_member(preserved).expect("bare owner retry");
        assert_eq!(unchanged.fields().role, MembershipRole::Owner);
        assert_eq!(outcome, MembershipCommandOutcome::Unchanged);
        let mut last_owner = make(
            Some(existing_owner),
            actor_id,
            Some(MembershipRole::Member),
            AgentChannelAddPolicy::Anyone,
        );
        last_owner.existing = Some(existing);
        assert_eq!(
            Membership::put_channel_member(last_owner),
            Err(MembershipError::ProtectedOwner)
        );
        let mut stale = make(
            Some(actor_member),
            target_id,
            None,
            AgentChannelAddPolicy::Anyone,
        );
        stale.current_actor_channel_version = AggregateVersion::new(2);
        assert!(Membership::put_channel_member(stale).is_err());
        let mut archived = make(
            Some(actor_member),
            target_id,
            None,
            AgentChannelAddPolicy::Anyone,
        );
        archived.lifecycle = crate::ChannelLifecycleState::Archived;
        assert!(Membership::put_channel_member(archived).is_err());
    }

    #[test]
    fn signed_channel_put_requires_exact_scope_current_community_and_tenant() {
        let community = community(11);
        let actor_id = principal(12);
        let tenant = tenant(community);
        let scope = AuthorizationScope::new("admin:channels").expect("scope");
        let actor = crate::AuthenticatedPrincipal::nostr_identity(
            actor_id,
            community,
            crate::NostrPublicKey::from_bytes([12; 32]),
            crate::NostrAuthenticationMethod::Nip42,
            PrincipalScopes::new([]).expect("scopes"),
        );
        let member = policy_membership(
            community,
            actor_id,
            MembershipRole::Member,
            AggregateVersion::FIRST,
        );
        let mut authorization = request(&tenant, &actor, &scope, member, AggregateVersion::FIRST);
        authorization.action = AuthorizationAction::Read;
        let denied = |authorization: &AuthorizationRequest<'_>| {
            Membership::put_channel_member(ChannelMemberPutRequest {
                authorization,
                channel_id: aggregate(13),
                visibility: crate::ChannelVisibility::Open,
                lifecycle: crate::ChannelLifecycleState::Active,
                actor_channel_membership: None,
                current_actor_channel_version: None,
                target_principal_id: actor_id,
                target_public_key: crate::NostrPublicKey::from_bytes([12; 32]),
                existing: None,
                requested_role: None,
                target_policy: AgentChannelAddPolicy::Anyone,
                target_owner_public_key: None,
                current_owner_count: 0,
            })
        };
        assert!(denied(&authorization).is_err());
        let valid = crate::AuthenticatedPrincipal::nostr_identity(
            actor_id,
            community,
            crate::NostrPublicKey::from_bytes([12; 32]),
            crate::NostrAuthenticationMethod::Nip42,
            PrincipalScopes::new([scope.clone()]).expect("scopes"),
        );
        authorization.principal = &valid;
        assert!(denied(&authorization).is_ok());
        authorization.community_membership = Some(CommunityMembership {
            role: MembershipRole::Guest,
            ..member
        });
        assert!(denied(&authorization).is_ok());
        authorization.community_membership = Some(CommunityMembership {
            role: MembershipRole::Bot,
            ..member
        });
        assert!(denied(&authorization).is_err());
        authorization.community_membership = Some(member);
        authorization.current_membership_version = AggregateVersion::new(2).expect("version");
        assert!(denied(&authorization).is_err());
        authorization.current_membership_version = AggregateVersion::FIRST;
        authorization.resource.community_id = crate::CommunityId::from_uuid(Uuid::from_u128(99));
        assert!(denied(&authorization).is_err());
        authorization.resource.community_id = community;
        let legacy = authenticated(community, actor_id);
        authorization.principal = &legacy;
        assert!(denied(&authorization).is_err());
        authorization.current_membership_version = AggregateVersion::new(2).expect("version");
        assert!(denied(&authorization).is_err());
    }

    #[test]
    fn invite_projection_preserves_join_and_retry_semantics() {
        let community_id = community(1);
        let target = principal(2);
        let evidence = InviteAdmissionEvidence::new(
            InviteId::from_uuid(Uuid::from_u128(3)),
            community_id,
            Some(2),
            0,
            1_000,
            false,
            AggregateVersion::FIRST,
        )
        .expect("invite");
        let redemption = evidence
            .redeem(
                community_id,
                target,
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            )
            .expect("redemption");
        let projection = Membership::from_invite_redemption(&redemption, None);
        assert!(projection.inserted());
        assert_eq!(
            projection.membership().policy_input(),
            MembershipPolicyInput::PersistentCommunity(redemption.membership())
        );

        let retry = redemption
            .evidence()
            .clone()
            .redeem(
                community_id,
                target,
                Some(redemption.membership()),
                redemption.evidence().version(),
                AggregateVersion::FIRST,
                100,
            )
            .expect("retry");
        assert!(!Membership::from_invite_redemption(&retry, None).inserted());
    }

    #[test]
    fn channel_membership_projects_an_exact_nip29_policy_input() {
        let community_id = community(1);
        let channel_id = aggregate(2);
        let member = Membership::from_record(MembershipRecordFields {
            community_id,
            scope: MembershipScope::Channel(channel_id),
            principal_id: principal(3),
            role: MembershipRole::Guest,
            status: MembershipStatus::Archived,
            version: AggregateVersion::new(4).expect("version"),
            added_by_principal_id: Some(principal(1)),
        });

        assert_eq!(
            member.policy_input(),
            MembershipPolicyInput::PersistentChannel(ChannelMembership {
                community_id,
                channel_id,
                principal_id: principal(3),
                role: MembershipRole::Guest,
                status: MembershipStatus::Archived,
                version: AggregateVersion::new(4).expect("version"),
            })
        );
    }

    #[test]
    fn owner_changes_roles_while_admin_cannot_escalate() {
        let community_id = community(1);
        let owner_id = principal(1);
        let target = principal(2);
        let tenant = tenant(community_id);
        let owner_principal = authenticated(community_id, owner_id);
        let required_scope = scope();
        let owner = policy_membership(
            community_id,
            owner_id,
            MembershipRole::Owner,
            AggregateVersion::FIRST,
        );
        let owner_request = request(
            &tenant,
            &owner_principal,
            &required_scope,
            owner,
            AggregateVersion::FIRST,
        );
        let mut membership = active_member(community_id, target, MembershipRole::Member);
        assert_eq!(
            membership
                .change_role(
                    AggregateVersion::FIRST,
                    MembershipRole::Admin,
                    &owner_request,
                )
                .expect("owner role change"),
            MembershipCommandOutcome::Applied
        );

        let admin_id = principal(3);
        let admin_principal = authenticated(community_id, admin_id);
        let admin = policy_membership(
            community_id,
            admin_id,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
        );
        let admin_request = request(
            &tenant,
            &admin_principal,
            &required_scope,
            admin,
            AggregateVersion::FIRST,
        );
        let version = AggregateVersion::FIRST.next().expect("second");
        assert_eq!(
            membership.change_role(version, MembershipRole::Member, &admin_request),
            Err(MembershipError::RoleEscalation)
        );
    }

    #[test]
    fn removal_and_archive_preserve_owner_and_revocation_guards() {
        let community_id = community(1);
        let owner_id = principal(1);
        let target = principal(2);
        let tenant = tenant(community_id);
        let authenticated = authenticated(community_id, owner_id);
        let required_scope = scope();
        let owner = policy_membership(
            community_id,
            owner_id,
            MembershipRole::Owner,
            AggregateVersion::FIRST,
        );
        let owner_request = request(
            &tenant,
            &authenticated,
            &required_scope,
            owner,
            AggregateVersion::FIRST,
        );
        let mut membership = active_member(community_id, target, MembershipRole::Member);
        assert_eq!(
            membership
                .archive(AggregateVersion::FIRST, &owner_request)
                .expect("archive"),
            MembershipCommandOutcome::Applied
        );
        let second = AggregateVersion::FIRST.next().expect("second");
        assert_eq!(
            membership.restore(second, &owner_request).expect("restore"),
            MembershipCommandOutcome::Applied
        );
        let third = second.next().expect("third");
        membership
            .revoke(third, &owner_request)
            .expect("revoke membership");
        let fourth = third.next().expect("fourth");
        assert_eq!(
            membership.restore(fourth, &owner_request),
            Err(MembershipError::InvalidTransition)
        );

        let mut protected_owner = active_member(community_id, target, MembershipRole::Owner);
        assert_eq!(
            protected_owner.revoke(AggregateVersion::FIRST, &owner_request),
            Err(MembershipError::ProtectedOwner)
        );
    }

    #[test]
    fn changed_membership_invalidates_a_cached_authorization_snapshot() {
        let community_id = community(1);
        let actor_id = principal(1);
        let target = principal(2);
        let tenant = tenant(community_id);
        let authenticated = authenticated(community_id, actor_id);
        let required_scope = scope();
        let owner = policy_membership(
            community_id,
            actor_id,
            MembershipRole::Owner,
            AggregateVersion::FIRST,
        );
        let owner_request = request(
            &tenant,
            &authenticated,
            &required_scope,
            owner,
            AggregateVersion::FIRST,
        );
        let mut membership = active_member(community_id, target, MembershipRole::Member);
        let cached = membership
            .policy_input()
            .community_membership()
            .expect("community membership");
        membership
            .change_role(
                AggregateVersion::FIRST,
                MembershipRole::Guest,
                &owner_request,
            )
            .expect("role change");
        let current = membership.fields().version;
        let read_scope = AuthorizationScope::new("messages:read").expect("read scope");
        let target_scopes = PrincipalScopes::new([read_scope.clone()]).expect("target scopes");
        let target_principal = crate::AuthenticatedPrincipal::zed_account(
            target,
            community_id,
            ServiceAccountId::new(2),
            target_scopes,
        );
        let cached_request = AuthorizationRequest {
            tenant: &tenant,
            principal: &target_principal,
            required_scope: &read_scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id,
                kind: AuthorizationResourceKind::Community,
                resource_id: aggregate(9),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: current,
            community_membership: Some(cached),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        };
        assert_eq!(
            authorize(&cached_request),
            AuthorizationDecision::Denied(AuthorizationDenial::StaleMembership)
        );
    }
}
