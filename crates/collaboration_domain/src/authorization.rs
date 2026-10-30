use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthenticatedPrincipalKind,
    AuthorizationScope, CommunityId, PrincipalId, TenantContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipRole {
    Owner,
    Admin,
    Member,
    Guest,
    Bot,
}

impl MembershipRole {
    fn permits(self, action: AuthorizationAction) -> bool {
        match self {
            Self::Owner => true,
            Self::Admin => action != AuthorizationAction::Delete,
            Self::Member => matches!(
                action,
                AuthorizationAction::Read | AuthorizationAction::Write
            ),
            Self::Guest => action == AuthorizationAction::Read,
            Self::Bot => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipStatus {
    Active,
    Revoked,
    Archived,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityMembership {
    pub community_id: CommunityId,
    pub principal_id: PrincipalId,
    pub role: MembershipRole,
    pub status: MembershipStatus,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelMembership {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub principal_id: PrincipalId,
    pub role: MembershipRole,
    pub status: MembershipStatus,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorizationResourceKind {
    Community,
    Channel,
    Project,
    Repository,
    Conversation,
    Workflow,
    AgentSession,
    Media,
    Administration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorizationResource {
    pub community_id: CommunityId,
    pub kind: AuthorizationResourceKind,
    pub resource_id: AggregateId,
    pub owner_principal_id: Option<PrincipalId>,
    pub channel_id: Option<AggregateId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorizationAction {
    Read,
    Write,
    Manage,
    Delete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DelegationGrant {
    pub community_id: CommunityId,
    pub delegate_principal_id: PrincipalId,
    pub resource_id: AggregateId,
    pub action: AuthorizationAction,
    pub membership_version: AggregateVersion,
    pub expires_at_millis: u64,
    pub revoked: bool,
}

pub struct AuthorizationRequest<'a> {
    pub tenant: &'a TenantContext,
    pub principal: &'a AuthenticatedPrincipal,
    pub required_scope: &'a AuthorizationScope,
    pub action: AuthorizationAction,
    pub resource: AuthorizationResource,
    pub current_membership_version: AggregateVersion,
    pub community_membership: Option<CommunityMembership>,
    pub current_channel_membership_version: Option<AggregateVersion>,
    pub channel_membership: Option<ChannelMembership>,
    pub delegation: Option<DelegationGrant>,
    pub now_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorizationDecision {
    Allowed,
    Denied(AuthorizationDenial),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorizationDenial {
    TenantMismatch,
    MissingScope,
    MissingMembership,
    InactiveMembership,
    StaleMembership,
    MissingChannelMembership,
    InactiveChannelMembership,
    StaleChannelMembership,
    InsufficientRole,
}

pub fn authorize(request: &AuthorizationRequest<'_>) -> AuthorizationDecision {
    let subject_principal_id = authorization_subject(request.principal);
    if request.tenant.community_id() != request.principal.community_id()
        || request.resource.community_id != request.tenant.community_id()
    {
        return AuthorizationDecision::Denied(AuthorizationDenial::TenantMismatch);
    }
    if !request.principal.scopes().contains(request.required_scope) {
        return AuthorizationDecision::Denied(AuthorizationDenial::MissingScope);
    }
    if request.resource.kind == AuthorizationResourceKind::Channel
        && request.resource.channel_id.is_none()
    {
        return AuthorizationDecision::Denied(AuthorizationDenial::MissingChannelMembership);
    }

    if delegation_permits(request) {
        return AuthorizationDecision::Allowed;
    }

    let Some(membership) = request.community_membership else {
        return AuthorizationDecision::Denied(AuthorizationDenial::MissingMembership);
    };
    if membership.community_id != request.tenant.community_id()
        || membership.principal_id != subject_principal_id
    {
        return AuthorizationDecision::Denied(AuthorizationDenial::TenantMismatch);
    }
    if membership.version != request.current_membership_version {
        return AuthorizationDecision::Denied(AuthorizationDenial::StaleMembership);
    }
    if membership.status != MembershipStatus::Active {
        return AuthorizationDecision::Denied(AuthorizationDenial::InactiveMembership);
    }

    if let Some(channel_id) = request.resource.channel_id {
        let Some(channel_membership) = request.channel_membership else {
            return AuthorizationDecision::Denied(AuthorizationDenial::MissingChannelMembership);
        };
        if channel_membership.community_id != request.tenant.community_id()
            || channel_membership.channel_id != channel_id
            || channel_membership.principal_id != subject_principal_id
        {
            return AuthorizationDecision::Denied(AuthorizationDenial::TenantMismatch);
        }
        if Some(channel_membership.version) != request.current_channel_membership_version {
            return AuthorizationDecision::Denied(AuthorizationDenial::StaleChannelMembership);
        }
        if channel_membership.status != MembershipStatus::Active {
            return AuthorizationDecision::Denied(AuthorizationDenial::InactiveChannelMembership);
        }
        return role_decision(channel_membership.role, request.action);
    }

    if request.resource.kind != AuthorizationResourceKind::Community
        && request.resource.kind != AuthorizationResourceKind::Administration
        && request.resource.owner_principal_id == Some(subject_principal_id)
    {
        return AuthorizationDecision::Allowed;
    }
    role_decision(membership.role, request.action)
}

pub fn authorization_subject(principal: &AuthenticatedPrincipal) -> PrincipalId {
    match principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => principal.principal_id(),
    }
}

fn delegation_permits(request: &AuthorizationRequest<'_>) -> bool {
    let Some(delegation) = request.delegation else {
        return false;
    };
    delegation.community_id == request.tenant.community_id()
        && delegation.delegate_principal_id == request.principal.principal_id()
        && delegation.resource_id == request.resource.resource_id
        && delegation.action == request.action
        && delegation.membership_version == request.current_membership_version
        && delegation.expires_at_millis >= request.now_millis
        && !delegation.revoked
}

fn role_decision(role: MembershipRole, action: AuthorizationAction) -> AuthorizationDecision {
    if role.permits(action) {
        AuthorizationDecision::Allowed
    } else {
        AuthorizationDecision::Denied(AuthorizationDenial::InsufficientRole)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentProfile, IdentityProfile, NostrAuthenticationMethod, NostrEventId, NostrPublicKey,
        OwnerAttestationEvidence, PrincipalScopes, ProfileId, ProfileKind, ProfileRecordFields,
        ServiceAccountId, TokenId, TrustedTenantRoute,
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
        AuthorizationScope::new("messages:read").expect("scope")
    }

    fn scopes() -> PrincipalScopes {
        PrincipalScopes::new([scope()]).expect("scopes")
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(TrustedTenantRoute::from_listener(community_id, "policy-test").expect("route")),
            &[],
        )
        .expect("tenant")
    }

    fn membership(
        community_id: CommunityId,
        principal_id: PrincipalId,
        role: MembershipRole,
    ) -> CommunityMembership {
        CommunityMembership {
            community_id,
            principal_id,
            role,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }
    }

    fn resource(
        community_id: CommunityId,
        kind: AuthorizationResourceKind,
    ) -> AuthorizationResource {
        AuthorizationResource {
            community_id,
            kind,
            resource_id: aggregate(10),
            owner_principal_id: None,
            channel_id: None,
        }
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        authenticated: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        resource: AuthorizationResource,
        membership: Option<CommunityMembership>,
        action: AuthorizationAction,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal: authenticated,
            required_scope,
            action,
            resource,
            current_membership_version: AggregateVersion::FIRST,
            community_membership: membership,
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn agent_profile(community_id: CommunityId) -> IdentityProfile {
        let agent = NostrPublicKey::from_bytes([6; 32]);
        let owner = NostrPublicKey::from_bytes([7; 32]);
        IdentityProfile::new(ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(30)),
            community_id,
            author_public_key: agent,
            kind: ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(owner),
                owner_attestation: Some(OwnerAttestationEvidence {
                    owner_public_key: owner,
                    agent_public_key: agent,
                    proof_event_id: NostrEventId::from_bytes([8; 32]),
                    exact_conditions: "kind=1".to_owned(),
                    verified_at: 1,
                }),
            }),
            metadata: None,
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("profile")
    }

    #[test]
    fn authorization_policy_covers_every_principal_kind() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let subject = principal(2);
        let agent_profile = agent_profile(community_id);
        let principals = [
            AuthenticatedPrincipal::zed_account(
                subject,
                community_id,
                ServiceAccountId::new(1),
                scopes(),
            ),
            AuthenticatedPrincipal::nostr_identity(
                subject,
                community_id,
                NostrPublicKey::from_bytes([1; 32]),
                NostrAuthenticationMethod::Nip42,
                scopes(),
            ),
            AuthenticatedPrincipal::owner_attested_agent(
                subject,
                community_id,
                &agent_profile,
                NostrAuthenticationMethod::Nip98,
                scopes(),
            )
            .expect("agent"),
            AuthenticatedPrincipal::scoped_token(
                principal(3),
                community_id,
                TokenId::from_uuid(Uuid::from_u128(40)),
                subject,
                scopes(),
            ),
            AuthenticatedPrincipal::service(subject, community_id, "workflow", scopes())
                .expect("service"),
        ];
        let required_scope = scope();

        for authenticated in principals {
            let request = request(
                &tenant,
                &authenticated,
                &required_scope,
                resource(community_id, AuthorizationResourceKind::Project),
                Some(membership(community_id, subject, MembershipRole::Member)),
                AuthorizationAction::Read,
            );
            assert_eq!(authorize(&request), AuthorizationDecision::Allowed);
        }
    }

    #[test]
    fn authorization_policy_applies_the_complete_role_table() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let subject = principal(2);
        let authenticated = AuthenticatedPrincipal::zed_account(
            subject,
            community_id,
            ServiceAccountId::new(1),
            scopes(),
        );
        let required_scope = scope();
        let cases = [
            (MembershipRole::Owner, AuthorizationAction::Delete, true),
            (MembershipRole::Admin, AuthorizationAction::Manage, true),
            (MembershipRole::Admin, AuthorizationAction::Delete, false),
            (MembershipRole::Member, AuthorizationAction::Write, true),
            (MembershipRole::Member, AuthorizationAction::Manage, false),
            (MembershipRole::Guest, AuthorizationAction::Read, true),
            (MembershipRole::Guest, AuthorizationAction::Write, false),
            (MembershipRole::Bot, AuthorizationAction::Read, false),
        ];

        for (role, action, expected) in cases {
            let request = request(
                &tenant,
                &authenticated,
                &required_scope,
                resource(community_id, AuthorizationResourceKind::Community),
                Some(membership(community_id, subject, role)),
                action,
            );
            assert_eq!(
                authorize(&request) == AuthorizationDecision::Allowed,
                expected
            );
        }
    }

    #[test]
    fn authorization_policy_covers_every_resource_kind_and_ownership() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let subject = principal(2);
        let authenticated = AuthenticatedPrincipal::zed_account(
            subject,
            community_id,
            ServiceAccountId::new(1),
            scopes(),
        );
        let required_scope = scope();
        for kind in [
            AuthorizationResourceKind::Community,
            AuthorizationResourceKind::Project,
            AuthorizationResourceKind::Repository,
            AuthorizationResourceKind::Conversation,
            AuthorizationResourceKind::Workflow,
            AuthorizationResourceKind::AgentSession,
            AuthorizationResourceKind::Media,
            AuthorizationResourceKind::Administration,
        ] {
            let mut target = resource(community_id, kind);
            target.owner_principal_id = Some(subject);
            let request = request(
                &tenant,
                &authenticated,
                &required_scope,
                target,
                Some(membership(community_id, subject, MembershipRole::Member)),
                AuthorizationAction::Manage,
            );
            let ownership_applies = !matches!(
                kind,
                AuthorizationResourceKind::Community | AuthorizationResourceKind::Administration
            );
            assert_eq!(
                authorize(&request) == AuthorizationDecision::Allowed,
                ownership_applies
            );
        }
    }

    #[test]
    fn authorization_policy_requires_current_channel_membership() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let subject = principal(2);
        let authenticated = AuthenticatedPrincipal::zed_account(
            subject,
            community_id,
            ServiceAccountId::new(1),
            scopes(),
        );
        let required_scope = scope();
        let channel_id = aggregate(20);
        let mut target = resource(community_id, AuthorizationResourceKind::Channel);
        let malformed = request(
            &tenant,
            &authenticated,
            &required_scope,
            target,
            Some(membership(community_id, subject, MembershipRole::Owner)),
            AuthorizationAction::Write,
        );
        assert_eq!(
            authorize(&malformed),
            AuthorizationDecision::Denied(AuthorizationDenial::MissingChannelMembership)
        );
        target.channel_id = Some(channel_id);
        let mut request = request(
            &tenant,
            &authenticated,
            &required_scope,
            target,
            Some(membership(community_id, subject, MembershipRole::Owner)),
            AuthorizationAction::Write,
        );
        assert_eq!(
            authorize(&request),
            AuthorizationDecision::Denied(AuthorizationDenial::MissingChannelMembership)
        );
        request.current_channel_membership_version = Some(AggregateVersion::FIRST);
        request.channel_membership = Some(ChannelMembership {
            community_id,
            channel_id,
            principal_id: subject,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        });
        assert_eq!(authorize(&request), AuthorizationDecision::Allowed);
    }

    #[test]
    fn authorization_policy_rejects_stale_membership_and_missing_scope() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let subject = principal(2);
        let authenticated = AuthenticatedPrincipal::zed_account(
            subject,
            community_id,
            ServiceAccountId::new(1),
            PrincipalScopes::default(),
        );
        let required_scope = scope();
        let mut request = request(
            &tenant,
            &authenticated,
            &required_scope,
            resource(community_id, AuthorizationResourceKind::Project),
            Some(membership(community_id, subject, MembershipRole::Owner)),
            AuthorizationAction::Read,
        );
        assert_eq!(
            authorize(&request),
            AuthorizationDecision::Denied(AuthorizationDenial::MissingScope)
        );

        let authenticated = AuthenticatedPrincipal::zed_account(
            subject,
            community_id,
            ServiceAccountId::new(1),
            scopes(),
        );
        request.principal = &authenticated;
        request.current_membership_version = AggregateVersion::FIRST.next().expect("version two");
        assert_eq!(
            authorize(&request),
            AuthorizationDecision::Denied(AuthorizationDenial::StaleMembership)
        );
    }

    #[test]
    fn authorization_policy_accepts_only_current_exact_delegation() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let authenticated =
            AuthenticatedPrincipal::service(principal(9), community_id, "workflow", scopes())
                .expect("service");
        let required_scope = scope();
        let target = resource(community_id, AuthorizationResourceKind::Workflow);
        let mut request = request(
            &tenant,
            &authenticated,
            &required_scope,
            target,
            None,
            AuthorizationAction::Write,
        );
        request.delegation = Some(DelegationGrant {
            community_id,
            delegate_principal_id: authenticated.principal_id(),
            resource_id: target.resource_id,
            action: AuthorizationAction::Write,
            membership_version: AggregateVersion::FIRST,
            expires_at_millis: 100,
            revoked: false,
        });
        assert_eq!(authorize(&request), AuthorizationDecision::Allowed);

        request.delegation.as_mut().expect("delegation").revoked = true;
        assert_eq!(
            authorize(&request),
            AuthorizationDecision::Denied(AuthorizationDenial::MissingMembership)
        );
    }
}
