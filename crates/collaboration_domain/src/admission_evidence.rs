use std::{error::Error, fmt};

use uuid::Uuid;

use crate::{
    AggregateVersion, AuthenticatedPrincipal, AuthenticatedPrincipalKind, CommunityId,
    CommunityMembership, MembershipRole, MembershipStatus, NostrEventId, NostrPublicKey,
    PrincipalId, PrincipalScopes, TokenId,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InviteId(Uuid);

impl InviteId {
    pub const fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayChallengeId(Uuid);

impl ReplayChallengeId {
    pub const fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayProtectionEvidence {
    challenge_id: ReplayChallengeId,
    community_id: CommunityId,
    token_id: TokenId,
    expires_at_millis: u64,
    consumed: bool,
    version: AggregateVersion,
}

impl ReplayProtectionEvidence {
    pub const fn new(
        challenge_id: ReplayChallengeId,
        community_id: CommunityId,
        token_id: TokenId,
        expires_at_millis: u64,
        consumed: bool,
        version: AggregateVersion,
    ) -> Self {
        Self {
            challenge_id,
            community_id,
            token_id,
            expires_at_millis,
            consumed,
            version,
        }
    }

    fn consume(
        self,
        community_id: CommunityId,
        token_id: TokenId,
        current_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<Self, AdmissionEvidenceError> {
        if self.community_id != community_id || self.token_id != token_id {
            return Err(AdmissionEvidenceError::TenantMismatch);
        }
        if self.version != current_version {
            return Err(AdmissionEvidenceError::StaleEvidence);
        }
        if self.consumed {
            return Err(AdmissionEvidenceError::ReplayDetected);
        }
        if self.expires_at_millis < now_millis {
            return Err(AdmissionEvidenceError::Expired);
        }
        let version = self
            .version
            .next()
            .ok_or(AdmissionEvidenceError::VersionExhausted)?;
        Ok(Self {
            consumed: true,
            version,
            ..self
        })
    }

    pub const fn challenge_id(&self) -> ReplayChallengeId {
        self.challenge_id
    }

    pub const fn is_consumed(&self) -> bool {
        self.consumed
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScopedTokenEvidence {
    token_id: TokenId,
    token_principal_id: PrincipalId,
    subject_principal_id: PrincipalId,
    community_id: CommunityId,
    granted_scopes: PrincipalScopes,
    expires_at_millis: u64,
    revoked: bool,
}

impl ScopedTokenEvidence {
    pub const fn new(
        token_id: TokenId,
        token_principal_id: PrincipalId,
        subject_principal_id: PrincipalId,
        community_id: CommunityId,
        granted_scopes: PrincipalScopes,
        expires_at_millis: u64,
        revoked: bool,
    ) -> Self {
        Self {
            token_id,
            token_principal_id,
            subject_principal_id,
            community_id,
            granted_scopes,
            expires_at_millis,
            revoked,
        }
    }

    pub fn admit(
        &self,
        requested_scopes: PrincipalScopes,
        replay_evidence: ReplayProtectionEvidence,
        current_replay_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<ScopedTokenAdmission, AdmissionEvidenceError> {
        if self.revoked {
            return Err(AdmissionEvidenceError::Revoked);
        }
        if self.expires_at_millis < now_millis {
            return Err(AdmissionEvidenceError::Expired);
        }
        if requested_scopes
            .iter()
            .any(|scope| !self.granted_scopes.contains(scope))
        {
            return Err(AdmissionEvidenceError::ScopeEscalation);
        }

        let replay_evidence = replay_evidence.consume(
            self.community_id,
            self.token_id,
            current_replay_version,
            now_millis,
        )?;
        Ok(ScopedTokenAdmission {
            principal: AuthenticatedPrincipal::scoped_token(
                self.token_principal_id,
                self.community_id,
                self.token_id,
                self.subject_principal_id,
                requested_scopes,
            ),
            replay_evidence,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScopedTokenAdmission {
    principal: AuthenticatedPrincipal,
    replay_evidence: ReplayProtectionEvidence,
}

impl ScopedTokenAdmission {
    pub const fn principal(&self) -> &AuthenticatedPrincipal {
        &self.principal
    }

    pub const fn replay_evidence(&self) -> &ReplayProtectionEvidence {
        &self.replay_evidence
    }

    pub fn into_parts(self) -> (AuthenticatedPrincipal, ReplayProtectionEvidence) {
        (self.principal, self.replay_evidence)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InviteAdmissionEvidence {
    invite_id: InviteId,
    community_id: CommunityId,
    max_uses: Option<u32>,
    use_count: u32,
    expires_at_millis: u64,
    revoked: bool,
    version: AggregateVersion,
}

impl InviteAdmissionEvidence {
    pub fn new(
        invite_id: InviteId,
        community_id: CommunityId,
        max_uses: Option<u32>,
        use_count: u32,
        expires_at_millis: u64,
        revoked: bool,
        version: AggregateVersion,
    ) -> Result<Self, AdmissionEvidenceError> {
        if max_uses == Some(0) || max_uses.is_some_and(|maximum| use_count > maximum) {
            return Err(AdmissionEvidenceError::InvalidInviteState);
        }
        Ok(Self {
            invite_id,
            community_id,
            max_uses,
            use_count,
            expires_at_millis,
            revoked,
            version,
        })
    }

    pub fn redeem(
        self,
        community_id: CommunityId,
        principal_id: PrincipalId,
        existing_membership: Option<CommunityMembership>,
        current_version: AggregateVersion,
        membership_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<InviteRedemption, AdmissionEvidenceError> {
        if self.community_id != community_id {
            return Err(AdmissionEvidenceError::TenantMismatch);
        }
        if self.version != current_version {
            return Err(AdmissionEvidenceError::StaleEvidence);
        }
        if self.revoked {
            return Err(AdmissionEvidenceError::Revoked);
        }
        if self.expires_at_millis < now_millis {
            return Err(AdmissionEvidenceError::Expired);
        }
        if let Some(membership) = existing_membership {
            if membership.community_id != community_id || membership.principal_id != principal_id {
                return Err(AdmissionEvidenceError::TenantMismatch);
            }
            return Ok(InviteRedemption::AlreadyMember {
                evidence: self,
                membership,
            });
        }
        if self
            .max_uses
            .is_some_and(|maximum| self.use_count >= maximum)
        {
            return Err(AdmissionEvidenceError::InviteExhausted);
        }

        let use_count = self
            .use_count
            .checked_add(1)
            .ok_or(AdmissionEvidenceError::InviteExhausted)?;
        let version = self
            .version
            .next()
            .ok_or(AdmissionEvidenceError::VersionExhausted)?;
        Ok(InviteRedemption::Joined {
            evidence: Self {
                use_count,
                version,
                ..self
            },
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: membership_version,
            },
        })
    }

    pub const fn invite_id(&self) -> InviteId {
        self.invite_id
    }

    pub const fn use_count(&self) -> u32 {
        self.use_count
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InviteRedemption {
    Joined {
        evidence: InviteAdmissionEvidence,
        membership: CommunityMembership,
    },
    AlreadyMember {
        evidence: InviteAdmissionEvidence,
        membership: CommunityMembership,
    },
}

impl InviteRedemption {
    pub const fn evidence(&self) -> &InviteAdmissionEvidence {
        match self {
            Self::Joined { evidence, .. } | Self::AlreadyMember { evidence, .. } => evidence,
        }
    }

    pub const fn membership(&self) -> CommunityMembership {
        match self {
            Self::Joined { membership, .. } | Self::AlreadyMember { membership, .. } => *membership,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualAgentMembershipEvidence {
    community_id: CommunityId,
    agent_principal_id: PrincipalId,
    agent_public_key: NostrPublicKey,
    owner_principal_id: PrincipalId,
    owner_public_key: NostrPublicKey,
    proof_event_id: NostrEventId,
    owner_membership_version: AggregateVersion,
}

impl VirtualAgentMembershipEvidence {
    pub fn derive(
        principal: &AuthenticatedPrincipal,
        owner_principal_id: PrincipalId,
        owner_membership: CommunityMembership,
        current_owner_membership_version: AggregateVersion,
    ) -> Result<Self, AdmissionEvidenceError> {
        let AuthenticatedPrincipalKind::OwnerAttestedAgent {
            agent_public_key,
            owner_public_key,
            proof_event_id,
            ..
        } = principal.kind()
        else {
            return Err(AdmissionEvidenceError::OwnerAttestationRequired);
        };
        if principal.principal_id() == owner_principal_id {
            return Err(AdmissionEvidenceError::InvalidOwnerAttestation);
        }
        if owner_membership.community_id != principal.community_id()
            || owner_membership.principal_id != owner_principal_id
        {
            return Err(AdmissionEvidenceError::TenantMismatch);
        }
        if owner_membership.version != current_owner_membership_version {
            return Err(AdmissionEvidenceError::StaleEvidence);
        }
        if owner_membership.status != MembershipStatus::Active {
            return Err(AdmissionEvidenceError::OwnerMembershipRequired);
        }
        Ok(Self {
            community_id: principal.community_id(),
            agent_principal_id: principal.principal_id(),
            agent_public_key: *agent_public_key,
            owner_principal_id,
            owner_public_key: *owner_public_key,
            proof_event_id: *proof_event_id,
            owner_membership_version: current_owner_membership_version,
        })
    }

    pub const fn policy_membership_snapshot(self) -> CommunityMembership {
        CommunityMembership {
            community_id: self.community_id,
            principal_id: self.agent_principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: self.owner_membership_version,
        }
    }

    pub const fn owner_principal_id(self) -> PrincipalId {
        self.owner_principal_id
    }

    pub const fn owner_public_key(self) -> NostrPublicKey {
        self.owner_public_key
    }

    pub const fn agent_public_key(self) -> NostrPublicKey {
        self.agent_public_key
    }

    pub const fn proof_event_id(self) -> NostrEventId {
        self.proof_event_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionEvidenceError {
    TenantMismatch,
    Expired,
    Revoked,
    ReplayDetected,
    ScopeEscalation,
    InviteExhausted,
    InvalidInviteState,
    StaleEvidence,
    VersionExhausted,
    OwnerAttestationRequired,
    InvalidOwnerAttestation,
    OwnerMembershipRequired,
}

impl fmt::Display for AdmissionEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TenantMismatch => "admission evidence tenant does not match",
            Self::Expired => "admission evidence has expired",
            Self::Revoked => "admission evidence has been revoked",
            Self::ReplayDetected => "admission evidence has already been consumed",
            Self::ScopeEscalation => "requested token scopes exceed the grant",
            Self::InviteExhausted => "invite has no remaining uses",
            Self::InvalidInviteState => "invite evidence is internally inconsistent",
            Self::StaleEvidence => "admission evidence version is stale",
            Self::VersionExhausted => "admission evidence version is exhausted",
            Self::OwnerAttestationRequired => "virtual membership requires owner attestation",
            Self::InvalidOwnerAttestation => "virtual membership owner attestation is invalid",
            Self::OwnerMembershipRequired => "virtual membership owner is not an active member",
        })
    }
}

impl Error for AdmissionEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentProfile, AuthorizationScope, IdentityProfile, NostrAuthenticationMethod,
        OwnerAttestationEvidence, ProfileId, ProfileKind, ProfileRecordFields,
    };

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn scope(value: &str) -> AuthorizationScope {
        AuthorizationScope::new(value).expect("valid test scope")
    }

    fn scopes(values: &[&str]) -> PrincipalScopes {
        PrincipalScopes::new(values.iter().map(|value| scope(value))).expect("valid test scopes")
    }

    fn token_evidence(community_id: CommunityId) -> ScopedTokenEvidence {
        ScopedTokenEvidence::new(
            TokenId::from_uuid(Uuid::from_u128(1)),
            principal(2),
            principal(3),
            community_id,
            scopes(&["messages:read", "messages:write"]),
            1_000,
            false,
        )
    }

    fn replay_evidence(community_id: CommunityId, consumed: bool) -> ReplayProtectionEvidence {
        ReplayProtectionEvidence::new(
            ReplayChallengeId::from_uuid(Uuid::from_u128(4)),
            community_id,
            TokenId::from_uuid(Uuid::from_u128(1)),
            1_000,
            consumed,
            AggregateVersion::FIRST,
        )
    }

    #[test]
    fn token_admission_only_narrows_scopes() {
        let community_id = community(1);
        let admission = token_evidence(community_id)
            .admit(
                scopes(&["messages:read"]),
                replay_evidence(community_id, false),
                AggregateVersion::FIRST,
                100,
            )
            .expect("narrow scope admission");
        assert_eq!(admission.principal().scopes(), &scopes(&["messages:read"]));
        assert!(admission.replay_evidence().is_consumed());
        assert_eq!(admission.replay_evidence().version().get(), 2);

        assert_eq!(
            token_evidence(community_id).admit(
                scopes(&["administration:write"]),
                replay_evidence(community_id, false),
                AggregateVersion::FIRST,
                100,
            ),
            Err(AdmissionEvidenceError::ScopeEscalation)
        );
    }

    #[test]
    fn consumed_or_cross_tenant_replay_evidence_fails_closed() {
        let community_id = community(1);
        assert_eq!(
            token_evidence(community_id).admit(
                scopes(&["messages:read"]),
                replay_evidence(community_id, true),
                AggregateVersion::FIRST,
                100,
            ),
            Err(AdmissionEvidenceError::ReplayDetected)
        );
        assert_eq!(
            token_evidence(community_id).admit(
                scopes(&["messages:read"]),
                replay_evidence(community(2), false),
                AggregateVersion::FIRST,
                100,
            ),
            Err(AdmissionEvidenceError::TenantMismatch)
        );
    }

    fn invite(
        community_id: CommunityId,
        max_uses: Option<u32>,
        use_count: u32,
        revoked: bool,
    ) -> InviteAdmissionEvidence {
        InviteAdmissionEvidence::new(
            InviteId::from_uuid(Uuid::from_u128(5)),
            community_id,
            max_uses,
            use_count,
            1_000,
            revoked,
            AggregateVersion::FIRST,
        )
        .expect("valid invite")
    }

    #[test]
    fn invite_redemption_consumes_one_use_and_existing_member_retry_is_idempotent() {
        let community_id = community(1);
        let joined = invite(community_id, Some(1), 0, false)
            .redeem(
                community_id,
                principal(6),
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            )
            .expect("first claim");
        assert_eq!(joined.evidence().use_count(), 1);
        assert_eq!(joined.membership().role, MembershipRole::Member);

        let existing = joined.membership();
        let retried = joined
            .evidence()
            .clone()
            .redeem(
                community_id,
                principal(6),
                Some(existing),
                joined.evidence().version(),
                AggregateVersion::FIRST,
                100,
            )
            .expect("existing member retry");
        assert!(matches!(retried, InviteRedemption::AlreadyMember { .. }));
        assert_eq!(retried.evidence().use_count(), 1);
    }

    #[test]
    fn exhausted_and_revoked_invites_fail_closed() {
        let community_id = community(1);
        assert_eq!(
            invite(community_id, Some(1), 1, false).redeem(
                community_id,
                principal(6),
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            ),
            Err(AdmissionEvidenceError::InviteExhausted)
        );
        assert_eq!(
            invite(community_id, None, 20, true).redeem(
                community_id,
                principal(6),
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            ),
            Err(AdmissionEvidenceError::Revoked)
        );
    }

    fn attested_agent(community_id: CommunityId) -> AuthenticatedPrincipal {
        let agent_key = NostrPublicKey::from_bytes([6; 32]);
        let owner_key = NostrPublicKey::from_bytes([7; 32]);
        let profile = IdentityProfile::new(ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(7)),
            community_id,
            author_public_key: agent_key,
            kind: ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(owner_key),
                owner_attestation: Some(OwnerAttestationEvidence {
                    owner_public_key: owner_key,
                    agent_public_key: agent_key,
                    proof_event_id: NostrEventId::from_bytes([8; 32]),
                    exact_conditions: "kind=1".to_owned(),
                    verified_at: 100,
                }),
            }),
            metadata: None,
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("valid agent profile");
        AuthenticatedPrincipal::owner_attested_agent(
            principal(8),
            community_id,
            &profile,
            NostrAuthenticationMethod::Nip42,
            scopes(&["messages:read", "messages:write"]),
        )
        .expect("attested agent")
    }

    #[test]
    fn virtual_agent_membership_is_transient_member_access_without_owner_role_inheritance() {
        let community_id = community(1);
        let owner_membership = CommunityMembership {
            community_id,
            principal_id: principal(9),
            role: MembershipRole::Owner,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        let evidence = VirtualAgentMembershipEvidence::derive(
            &attested_agent(community_id),
            principal(9),
            owner_membership,
            AggregateVersion::FIRST,
        )
        .expect("virtual membership");
        let policy_membership = evidence.policy_membership_snapshot();

        assert_eq!(policy_membership.principal_id, principal(8));
        assert_eq!(policy_membership.role, MembershipRole::Member);
        assert_eq!(evidence.owner_principal_id(), principal(9));
    }

    #[test]
    fn unattested_or_revoked_owner_virtual_membership_fails_closed() {
        let community_id = community(1);
        let unattested = AuthenticatedPrincipal::nostr_identity(
            principal(8),
            community_id,
            NostrPublicKey::from_bytes([6; 32]),
            NostrAuthenticationMethod::Nip42,
            scopes(&["messages:read"]),
        );
        let owner_membership = CommunityMembership {
            community_id,
            principal_id: principal(9),
            role: MembershipRole::Owner,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        assert_eq!(
            VirtualAgentMembershipEvidence::derive(
                &unattested,
                principal(9),
                owner_membership,
                AggregateVersion::FIRST,
            ),
            Err(AdmissionEvidenceError::OwnerAttestationRequired)
        );

        assert_eq!(
            VirtualAgentMembershipEvidence::derive(
                &attested_agent(community_id),
                principal(9),
                CommunityMembership {
                    status: MembershipStatus::Revoked,
                    ..owner_membership
                },
                AggregateVersion::FIRST,
            ),
            Err(AdmissionEvidenceError::OwnerMembershipRequired)
        );
    }
}
