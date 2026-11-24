use std::{error::Error, fmt, num::NonZeroU32};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, InviteId, Membership, MembershipPolicyInput, MembershipRecordFields,
    MembershipRole, MembershipScope, MembershipStatus, PrincipalId, authorize,
};

const MAX_INVITE_USES: u32 = 10_000;
const MIN_INVITE_TTL_MILLIS: u64 = 60_000;
const MAX_INVITE_TTL_MILLIS: u64 = 30 * 24 * 60 * 60 * 1_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InviteTokenHash([u8; 32]);

impl InviteTokenHash {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelInviteTarget {
    Community,
    Channel(AggregateId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelInviteStatus {
    Active,
    Revoked,
    Exhausted,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelInviteRecordFields {
    pub invite_id: InviteId,
    pub community_id: CommunityId,
    pub target: ChannelInviteTarget,
    pub token_hash: InviteTokenHash,
    pub role: MembershipRole,
    pub status: ChannelInviteStatus,
    pub max_uses: Option<NonZeroU32>,
    pub use_count: u32,
    pub expires_at_millis: u64,
    pub created_by_principal_id: PrincipalId,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelInviteCreateFields {
    pub invite_id: InviteId,
    pub community_id: CommunityId,
    pub target: ChannelInviteTarget,
    pub token_hash: InviteTokenHash,
    pub role: MembershipRole,
    pub max_uses: Option<NonZeroU32>,
    pub expires_at_millis: u64,
    pub now_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelInviteCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelInviteRedemption {
    Joined {
        invite: ChannelInvite,
        membership: Membership,
    },
    AlreadyMember {
        invite: ChannelInvite,
        membership: Membership,
    },
}

impl ChannelInviteRedemption {
    pub const fn invite(self) -> ChannelInvite {
        match self {
            Self::Joined { invite, .. } | Self::AlreadyMember { invite, .. } => invite,
        }
    }

    pub const fn membership(self) -> Membership {
        match self {
            Self::Joined { membership, .. } | Self::AlreadyMember { membership, .. } => membership,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelInvite {
    fields: ChannelInviteRecordFields,
}

impl ChannelInvite {
    pub fn from_record(fields: ChannelInviteRecordFields) -> Result<Self, ChannelInviteError> {
        validate_fields(&fields)?;
        Ok(Self { fields })
    }

    pub fn create(
        fields: ChannelInviteCreateFields,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ChannelInviteError> {
        authorize_target(authorization, fields.community_id, fields.target)?;
        let ttl = fields
            .expires_at_millis
            .checked_sub(fields.now_millis)
            .ok_or(ChannelInviteError::InvalidExpiry)?;
        if !(MIN_INVITE_TTL_MILLIS..=MAX_INVITE_TTL_MILLIS).contains(&ttl) {
            return Err(ChannelInviteError::InvalidExpiry);
        }
        let invite = Self {
            fields: ChannelInviteRecordFields {
                invite_id: fields.invite_id,
                community_id: fields.community_id,
                target: fields.target,
                token_hash: fields.token_hash,
                role: fields.role,
                status: ChannelInviteStatus::Active,
                max_uses: fields.max_uses,
                use_count: 0,
                expires_at_millis: fields.expires_at_millis,
                created_by_principal_id: authorization_subject(authorization),
                version: AggregateVersion::FIRST,
            },
        };
        validate_fields(&invite.fields)?;
        Ok(invite)
    }

    pub const fn fields(self) -> ChannelInviteRecordFields {
        self.fields
    }

    pub fn revoke(
        &mut self,
        expected_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ChannelInviteCommandOutcome, ChannelInviteError> {
        authorize_target(authorization, self.fields.community_id, self.fields.target)?;
        self.require_version(expected_version)?;
        if self.fields.status == ChannelInviteStatus::Revoked {
            return Ok(ChannelInviteCommandOutcome::Unchanged);
        }
        self.fields.status = ChannelInviteStatus::Revoked;
        self.advance_version()?;
        Ok(ChannelInviteCommandOutcome::Applied)
    }

    pub fn expire_if_due(
        &mut self,
        expected_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<ChannelInviteCommandOutcome, ChannelInviteError> {
        self.require_version(expected_version)?;
        if self.fields.status != ChannelInviteStatus::Active {
            return Ok(ChannelInviteCommandOutcome::Unchanged);
        }
        if now_millis < self.fields.expires_at_millis {
            return Ok(ChannelInviteCommandOutcome::Unchanged);
        }
        self.fields.status = ChannelInviteStatus::Expired;
        self.advance_version()?;
        Ok(ChannelInviteCommandOutcome::Applied)
    }

    pub fn redeem(
        mut self,
        presented_hash: InviteTokenHash,
        community_id: CommunityId,
        principal_id: PrincipalId,
        existing_membership: Option<MembershipPolicyInput>,
        expected_version: AggregateVersion,
        membership_version: AggregateVersion,
        now_millis: u64,
    ) -> Result<ChannelInviteRedemption, ChannelInviteError> {
        if community_id != self.fields.community_id || presented_hash != self.fields.token_hash {
            return Err(ChannelInviteError::InvalidBearer);
        }
        self.require_version(expected_version)?;
        if now_millis >= self.fields.expires_at_millis {
            return Err(ChannelInviteError::Expired);
        }
        if self.fields.status == ChannelInviteStatus::Revoked {
            return Err(ChannelInviteError::Revoked);
        }
        if let Some(existing) = existing_membership {
            let membership = membership_from_policy(existing, self.fields.target, principal_id)?;
            return Ok(ChannelInviteRedemption::AlreadyMember {
                invite: self,
                membership,
            });
        }
        if self.fields.status == ChannelInviteStatus::Exhausted
            || self
                .fields
                .max_uses
                .is_some_and(|maximum| self.fields.use_count >= maximum.get())
        {
            return Err(ChannelInviteError::Exhausted);
        }
        self.fields.use_count = self
            .fields
            .use_count
            .checked_add(1)
            .ok_or(ChannelInviteError::Exhausted)?;
        if self
            .fields
            .max_uses
            .is_some_and(|maximum| self.fields.use_count >= maximum.get())
        {
            self.fields.status = ChannelInviteStatus::Exhausted;
        }
        self.advance_version()?;
        let membership = Membership::from_record(MembershipRecordFields {
            community_id,
            scope: target_scope(self.fields.target),
            principal_id,
            role: self.fields.role,
            status: MembershipStatus::Active,
            version: membership_version,
            added_by_principal_id: Some(self.fields.created_by_principal_id),
        });
        Ok(ChannelInviteRedemption::Joined {
            invite: self,
            membership,
        })
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), ChannelInviteError> {
        if self.fields.version != expected {
            return Err(ChannelInviteError::StaleVersion);
        }
        Ok(())
    }

    fn advance_version(&mut self) -> Result<(), ChannelInviteError> {
        self.fields.version = self
            .fields
            .version
            .next()
            .ok_or(ChannelInviteError::VersionExhausted)?;
        Ok(())
    }
}

fn validate_fields(fields: &ChannelInviteRecordFields) -> Result<(), ChannelInviteError> {
    if !matches!(fields.role, MembershipRole::Member | MembershipRole::Guest)
        || fields
            .max_uses
            .is_some_and(|maximum| maximum.get() > MAX_INVITE_USES)
        || fields
            .max_uses
            .is_some_and(|maximum| fields.use_count > maximum.get())
    {
        return Err(ChannelInviteError::InvalidState);
    }
    Ok(())
}

fn target_scope(target: ChannelInviteTarget) -> MembershipScope {
    match target {
        ChannelInviteTarget::Community => MembershipScope::Community,
        ChannelInviteTarget::Channel(channel_id) => MembershipScope::Channel(channel_id),
    }
}

fn membership_from_policy(
    input: MembershipPolicyInput,
    target: ChannelInviteTarget,
    principal_id: PrincipalId,
) -> Result<Membership, ChannelInviteError> {
    let membership = match (target, input) {
        (ChannelInviteTarget::Community, MembershipPolicyInput::PersistentCommunity(value))
        | (ChannelInviteTarget::Community, MembershipPolicyInput::VirtualCommunity(value))
            if value.principal_id == principal_id && value.status == MembershipStatus::Active =>
        {
            Membership::from_record(MembershipRecordFields {
                community_id: value.community_id,
                scope: MembershipScope::Community,
                principal_id: value.principal_id,
                role: value.role,
                status: value.status,
                version: value.version,
                added_by_principal_id: None,
            })
        }
        (
            ChannelInviteTarget::Channel(channel_id),
            MembershipPolicyInput::PersistentChannel(value),
        ) if value.channel_id == channel_id
            && value.principal_id == principal_id
            && value.status == MembershipStatus::Active =>
        {
            Membership::from_record(MembershipRecordFields {
                community_id: value.community_id,
                scope: MembershipScope::Channel(channel_id),
                principal_id: value.principal_id,
                role: value.role,
                status: value.status,
                version: value.version,
                added_by_principal_id: None,
            })
        }
        _ => return Err(ChannelInviteError::UnauthorizedRedemption),
    };
    Ok(membership)
}

fn authorize_target(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    target: ChannelInviteTarget,
) -> Result<(), ChannelInviteError> {
    if request.action != AuthorizationAction::Manage
        || request.resource.community_id != community_id
    {
        return Err(ChannelInviteError::AuthorizationShape);
    }
    let matches = match target {
        ChannelInviteTarget::Community => {
            request.resource.kind == AuthorizationResourceKind::Community
                && request.resource.resource_id == AggregateId::from_uuid(community_id.as_uuid())
                && request.resource.channel_id.is_none()
        }
        ChannelInviteTarget::Channel(channel_id) => {
            request.resource.kind == AuthorizationResourceKind::Channel
                && request.resource.resource_id == channel_id
                && request.resource.channel_id == Some(channel_id)
        }
    };
    if !matches {
        return Err(ChannelInviteError::AuthorizationShape);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(ChannelInviteError::Unauthorized(denial)),
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelInviteError {
    InvalidState,
    InvalidExpiry,
    InvalidBearer,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    UnauthorizedRedemption,
    StaleVersion,
    Expired,
    Revoked,
    Exhausted,
    VersionExhausted,
}

impl fmt::Display for ChannelInviteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState | Self::InvalidExpiry => formatter.write_str("invite is invalid"),
            Self::InvalidBearer
            | Self::AuthorizationShape
            | Self::Unauthorized(_)
            | Self::UnauthorizedRedemption => formatter.write_str("invite is not authorized"),
            Self::StaleVersion => formatter.write_str("invite version is stale"),
            Self::Expired => formatter.write_str("invite is expired"),
            Self::Revoked => formatter.write_str("invite is revoked"),
            Self::Exhausted => formatter.write_str("invite is exhausted"),
            Self::VersionExhausted => formatter.write_str("invite version is exhausted"),
        }
    }
}

impl Error for ChannelInviteError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChannelMembership, MembershipPolicyInput};
    use uuid::Uuid;

    fn community() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn channel() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(2))
    }

    fn invite(status: ChannelInviteStatus, max_uses: Option<u32>, use_count: u32) -> ChannelInvite {
        ChannelInvite::from_record(ChannelInviteRecordFields {
            invite_id: InviteId::from_uuid(Uuid::from_u128(3)),
            community_id: community(),
            target: ChannelInviteTarget::Channel(channel()),
            token_hash: InviteTokenHash::from_bytes([7; 32]),
            role: MembershipRole::Guest,
            status,
            max_uses: max_uses.and_then(NonZeroU32::new),
            use_count,
            expires_at_millis: 1_000,
            created_by_principal_id: principal(4),
            version: AggregateVersion::FIRST,
        })
        .expect("invite")
    }

    #[test]
    fn final_slot_exhausts_and_existing_member_replay_does_not_consume() {
        let redemption = invite(ChannelInviteStatus::Active, Some(1), 0)
            .redeem(
                InviteTokenHash::from_bytes([7; 32]),
                community(),
                principal(5),
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            )
            .expect("redeem final slot");
        let joined = redemption.membership();
        let exhausted = redemption.invite();
        assert_eq!(exhausted.fields().status, ChannelInviteStatus::Exhausted);
        assert_eq!(exhausted.fields().use_count, 1);
        let replay = exhausted
            .redeem(
                InviteTokenHash::from_bytes([7; 32]),
                community(),
                principal(5),
                Some(joined.policy_input()),
                AggregateVersion::FIRST.next().expect("second"),
                AggregateVersion::FIRST,
                100,
            )
            .expect("existing-member replay");
        assert!(matches!(
            replay,
            ChannelInviteRedemption::AlreadyMember { .. }
        ));
        assert_eq!(replay.invite().fields().use_count, 1);
    }

    #[test]
    fn expiry_and_revocation_fail_before_membership_or_capacity() {
        for (invite, expected) in [
            (
                invite(ChannelInviteStatus::Active, Some(1), 0),
                ChannelInviteError::Expired,
            ),
            (
                invite(ChannelInviteStatus::Revoked, Some(1), 0),
                ChannelInviteError::Revoked,
            ),
        ] {
            assert_eq!(
                invite.redeem(
                    InviteTokenHash::from_bytes([7; 32]),
                    community(),
                    principal(5),
                    None,
                    AggregateVersion::FIRST,
                    AggregateVersion::FIRST,
                    if expected == ChannelInviteError::Expired {
                        1_000
                    } else {
                        100
                    },
                ),
                Err(expected)
            );
        }
    }

    #[test]
    fn wrong_bearer_tenant_and_foreign_membership_fail_closed() {
        let invite = invite(ChannelInviteStatus::Active, None, 0);
        assert_eq!(
            invite.redeem(
                InviteTokenHash::from_bytes([8; 32]),
                community(),
                principal(5),
                None,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            ),
            Err(ChannelInviteError::InvalidBearer)
        );
        let foreign = MembershipPolicyInput::PersistentChannel(ChannelMembership {
            community_id: community(),
            channel_id: AggregateId::from_uuid(Uuid::from_u128(99)),
            principal_id: principal(5),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        });
        assert_eq!(
            invite.redeem(
                InviteTokenHash::from_bytes([7; 32]),
                community(),
                principal(5),
                Some(foreign),
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                100,
            ),
            Err(ChannelInviteError::UnauthorizedRedemption)
        );
    }
}
