use std::fmt;

use crate::{
    AggregateId, ChannelMembership, CommunityId, CommunityMembership, MembershipStatus,
    PrincipalId, SourceRecordId, SourceSystem,
};

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct NotificationSourceId {
    community_id: CommunityId,
    source_system: SourceSystem,
    source_record_id: SourceRecordId,
}

impl NotificationSourceId {
    pub const fn new(
        community_id: CommunityId,
        source_system: SourceSystem,
        source_record_id: SourceRecordId,
    ) -> Self {
        Self {
            community_id,
            source_system,
            source_record_id,
        }
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn source_system(&self) -> SourceSystem {
        self.source_system
    }

    pub fn source_record_id(&self) -> &SourceRecordId {
        &self.source_record_id
    }
}

impl fmt::Debug for NotificationSourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NotificationSourceId")
            .field("community_id", &self.community_id)
            .field("source_system", &self.source_system)
            .field("source_record_id", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NotificationSurface {
    Native,
    Push,
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct NotificationDeliveryId {
    source: NotificationSourceId,
    recipient_principal_id: PrincipalId,
    surface: NotificationSurface,
}

impl NotificationDeliveryId {
    pub const fn source(&self) -> &NotificationSourceId {
        &self.source
    }

    pub const fn recipient_principal_id(&self) -> PrincipalId {
        self.recipient_principal_id
    }

    pub const fn surface(&self) -> NotificationSurface {
        self.surface
    }
}

impl fmt::Debug for NotificationDeliveryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NotificationDeliveryId")
            .field("source", &self.source)
            .field("recipient_principal_id", &self.recipient_principal_id)
            .field("surface", &self.surface)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotificationReason {
    Mention,
    DirectMessage,
    NeedsAction,
    SubscribedActivity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NotificationMembership {
    pub community: Option<CommunityMembership>,
    pub channel: Option<ChannelMembership>,
}

impl NotificationMembership {
    fn permits(
        self,
        community_id: CommunityId,
        recipient_principal_id: PrincipalId,
        channel_id: Option<AggregateId>,
    ) -> bool {
        let Some(community) = self.community else {
            return false;
        };
        if community.community_id != community_id
            || community.principal_id != recipient_principal_id
            || community.status != MembershipStatus::Active
        {
            return false;
        }

        let Some(channel_id) = channel_id else {
            return true;
        };
        let Some(channel) = self.channel else {
            return false;
        };
        channel.community_id == community_id
            && channel.channel_id == channel_id
            && channel.principal_id == recipient_principal_id
            && channel.status == MembershipStatus::Active
    }

    pub const fn community(community: CommunityMembership) -> Self {
        Self {
            community: Some(community),
            channel: None,
        }
    }

    pub const fn channel(community: CommunityMembership, channel: ChannelMembership) -> Self {
        Self {
            community: Some(community),
            channel: Some(channel),
        }
    }

    pub const fn missing() -> Self {
        Self {
            community: None,
            channel: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotificationPrivacy {
    CommunityVisible,
    Private { recipient_is_participant: bool },
}

impl NotificationPrivacy {
    const fn permits_recipient(self) -> bool {
        match self {
            Self::CommunityVisible => true,
            Self::Private {
                recipient_is_participant,
            } => recipient_is_participant,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotificationReadState {
    Unread,
    Read,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotificationPermission {
    Granted,
    Disabled,
    Denied,
    Revoked,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NotificationDevicePermissions {
    pub native: NotificationPermission,
    pub push: NotificationPermission,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotificationCandidate {
    pub source: NotificationSourceId,
    pub recipient_principal_id: PrincipalId,
    pub author_principal_id: PrincipalId,
    pub channel_id: Option<AggregateId>,
    pub reason: NotificationReason,
    pub membership: NotificationMembership,
    pub privacy: NotificationPrivacy,
    pub read_state: NotificationReadState,
    pub muted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotificationSuppression {
    InactiveMembership,
    UnauthorizedPrivateEvent,
    SelfAuthored,
    AlreadyRead,
    ReadStateUnavailable,
    Muted,
    PermissionDisabled,
    PermissionDenied,
    PermissionRevoked,
    PermissionUnsupported,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NotificationSurfaceDecision {
    Deliver(NotificationDeliveryId),
    Suppress(NotificationSuppression),
}

impl NotificationSurfaceDecision {
    pub fn delivery_id(&self) -> Option<&NotificationDeliveryId> {
        match self {
            Self::Deliver(delivery_id) => Some(delivery_id),
            Self::Suppress(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotificationDecision {
    pub native: NotificationSurfaceDecision,
    pub push: NotificationSurfaceDecision,
}

pub fn decide_notification(
    candidate: &NotificationCandidate,
    permissions: NotificationDevicePermissions,
    already_delivered: impl Fn(&NotificationDeliveryId) -> bool,
) -> NotificationDecision {
    if let Some(suppression) = common_suppression(candidate) {
        return NotificationDecision {
            native: NotificationSurfaceDecision::Suppress(suppression),
            push: NotificationSurfaceDecision::Suppress(suppression),
        };
    }

    NotificationDecision {
        native: decide_surface(
            candidate,
            NotificationSurface::Native,
            permissions.native,
            &already_delivered,
        ),
        push: decide_surface(
            candidate,
            NotificationSurface::Push,
            permissions.push,
            &already_delivered,
        ),
    }
}

fn common_suppression(candidate: &NotificationCandidate) -> Option<NotificationSuppression> {
    if !candidate.membership.permits(
        candidate.source.community_id(),
        candidate.recipient_principal_id,
        candidate.channel_id,
    ) {
        return Some(NotificationSuppression::InactiveMembership);
    }
    if !candidate.privacy.permits_recipient() {
        return Some(NotificationSuppression::UnauthorizedPrivateEvent);
    }
    if candidate.author_principal_id == candidate.recipient_principal_id {
        return Some(NotificationSuppression::SelfAuthored);
    }
    match candidate.read_state {
        NotificationReadState::Read => return Some(NotificationSuppression::AlreadyRead),
        NotificationReadState::Unavailable => {
            return Some(NotificationSuppression::ReadStateUnavailable);
        }
        NotificationReadState::Unread => {}
    }
    if candidate.muted && candidate.reason != NotificationReason::Mention {
        return Some(NotificationSuppression::Muted);
    }
    None
}

fn decide_surface(
    candidate: &NotificationCandidate,
    surface: NotificationSurface,
    permission: NotificationPermission,
    already_delivered: &impl Fn(&NotificationDeliveryId) -> bool,
) -> NotificationSurfaceDecision {
    let suppression = match permission {
        NotificationPermission::Granted => None,
        NotificationPermission::Disabled => Some(NotificationSuppression::PermissionDisabled),
        NotificationPermission::Denied => Some(NotificationSuppression::PermissionDenied),
        NotificationPermission::Revoked => Some(NotificationSuppression::PermissionRevoked),
        NotificationPermission::Unsupported => Some(NotificationSuppression::PermissionUnsupported),
    };
    if let Some(suppression) = suppression {
        return NotificationSurfaceDecision::Suppress(suppression);
    }

    let delivery_id = NotificationDeliveryId {
        source: candidate.source.clone(),
        recipient_principal_id: candidate.recipient_principal_id,
        surface,
    };
    if already_delivered(&delivery_id) {
        NotificationSurfaceDecision::Suppress(NotificationSuppression::Duplicate)
    } else {
        NotificationSurfaceDecision::Deliver(delivery_id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{AggregateVersion, MembershipRole};
    use uuid::Uuid;

    use super::*;

    fn community() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn recipient() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(2))
    }

    fn author() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(3))
    }

    fn channel() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(4))
    }

    fn active_membership() -> NotificationMembership {
        NotificationMembership::channel(
            CommunityMembership {
                community_id: community(),
                principal_id: recipient(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            ChannelMembership {
                community_id: community(),
                channel_id: channel(),
                principal_id: recipient(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        )
    }

    fn candidate() -> NotificationCandidate {
        NotificationCandidate {
            source: NotificationSourceId::new(
                community(),
                SourceSystem::Nostr,
                SourceRecordId::new("event:one").expect("valid source id"),
            ),
            recipient_principal_id: recipient(),
            author_principal_id: author(),
            channel_id: Some(channel()),
            reason: NotificationReason::SubscribedActivity,
            membership: active_membership(),
            privacy: NotificationPrivacy::CommunityVisible,
            read_state: NotificationReadState::Unread,
            muted: false,
        }
    }

    fn granted() -> NotificationDevicePermissions {
        NotificationDevicePermissions {
            native: NotificationPermission::Granted,
            push: NotificationPermission::Granted,
        }
    }

    fn assert_common_suppression(
        candidate: NotificationCandidate,
        expected: NotificationSuppression,
    ) {
        assert_eq!(
            decide_notification(&candidate, granted(), |_| false),
            NotificationDecision {
                native: NotificationSurfaceDecision::Suppress(expected),
                push: NotificationSurfaceDecision::Suppress(expected),
            }
        );
    }

    #[test]
    fn notification_policy_table_covers_self_mute_read_duplicate_revoked_and_private_events() {
        let mut self_authored = candidate();
        self_authored.author_principal_id = recipient();

        let mut muted = candidate();
        muted.muted = true;

        let mut read = candidate();
        read.read_state = NotificationReadState::Read;

        let mut revoked = candidate();
        let mut channel_membership = revoked
            .membership
            .channel
            .expect("active channel membership");
        channel_membership.status = MembershipStatus::Revoked;
        revoked.membership.channel = Some(channel_membership);

        let mut private = candidate();
        private.privacy = NotificationPrivacy::Private {
            recipient_is_participant: false,
        };

        for (name, candidate, expected, native_was_delivered) in [
            (
                "self-authored",
                self_authored,
                NotificationSuppression::SelfAuthored,
                false,
            ),
            ("muted", muted, NotificationSuppression::Muted, false),
            ("read", read, NotificationSuppression::AlreadyRead, false),
            (
                "duplicate",
                candidate(),
                NotificationSuppression::Duplicate,
                true,
            ),
            (
                "revoked",
                revoked,
                NotificationSuppression::InactiveMembership,
                false,
            ),
            (
                "private nonparticipant",
                private,
                NotificationSuppression::UnauthorizedPrivateEvent,
                false,
            ),
        ] {
            let decision = decide_notification(&candidate, granted(), |delivery_id| {
                native_was_delivered && delivery_id.surface() == NotificationSurface::Native
            });
            assert_eq!(
                decision.native,
                NotificationSurfaceDecision::Suppress(expected),
                "{name} native"
            );
            if native_was_delivered {
                assert!(
                    matches!(decision.push, NotificationSurfaceDecision::Deliver(_)),
                    "{name} push"
                );
            } else {
                assert_eq!(
                    decision.push,
                    NotificationSurfaceDecision::Suppress(expected),
                    "{name} push"
                );
            }
        }
    }

    #[test]
    fn notification_policy_allows_mentions_to_override_mute_after_access_checks() {
        let mut mention = candidate();
        mention.reason = NotificationReason::Mention;
        mention.muted = true;
        let decision = decide_notification(&mention, granted(), |_| false);
        assert!(matches!(
            decision.native,
            NotificationSurfaceDecision::Deliver(_)
        ));
        assert!(matches!(
            decision.push,
            NotificationSurfaceDecision::Deliver(_)
        ));

        mention.membership = NotificationMembership::missing();
        assert_common_suppression(mention, NotificationSuppression::InactiveMembership);
    }

    #[test]
    fn notification_policy_deduplicates_stable_source_per_recipient_and_surface() {
        let first = decide_notification(&candidate(), granted(), |_| false);
        let native_id = first
            .native
            .delivery_id()
            .expect("native delivery id")
            .clone();
        let delivered = HashSet::from([native_id.clone()]);

        let duplicate = decide_notification(&candidate(), granted(), |delivery_id| {
            delivered.contains(delivery_id)
        });
        assert_eq!(
            duplicate.native,
            NotificationSurfaceDecision::Suppress(NotificationSuppression::Duplicate)
        );
        assert!(matches!(
            duplicate.push,
            NotificationSurfaceDecision::Deliver(_)
        ));
        assert_eq!(native_id.surface(), NotificationSurface::Native);

        let mut changed_source = candidate();
        changed_source.source = NotificationSourceId::new(
            community(),
            SourceSystem::Nostr,
            SourceRecordId::new("event:two").expect("valid source id"),
        );
        assert!(matches!(
            decide_notification(&changed_source, granted(), |delivery_id| {
                delivered.contains(delivery_id)
            })
            .native,
            NotificationSurfaceDecision::Deliver(_)
        ));
    }

    #[test]
    fn notification_policy_requires_private_participation_and_surface_permission() {
        let mut private = candidate();
        private.reason = NotificationReason::DirectMessage;
        private.privacy = NotificationPrivacy::Private {
            recipient_is_participant: false,
        };
        assert_common_suppression(
            private.clone(),
            NotificationSuppression::UnauthorizedPrivateEvent,
        );

        private.privacy = NotificationPrivacy::Private {
            recipient_is_participant: true,
        };
        let decision = decide_notification(
            &private,
            NotificationDevicePermissions {
                native: NotificationPermission::Granted,
                push: NotificationPermission::Revoked,
            },
            |_| false,
        );
        assert!(matches!(
            decision.native,
            NotificationSurfaceDecision::Deliver(_)
        ));
        assert_eq!(
            decision.push,
            NotificationSurfaceDecision::Suppress(NotificationSuppression::PermissionRevoked)
        );
    }

    #[test]
    fn notification_policy_fails_closed_when_read_state_is_unavailable() {
        let mut unavailable = candidate();
        unavailable.read_state = NotificationReadState::Unavailable;
        assert_common_suppression(unavailable, NotificationSuppression::ReadStateUnavailable);
    }

    #[test]
    fn notification_identifiers_redact_source_records_in_diagnostics() {
        let decision = decide_notification(&candidate(), granted(), |_| false);
        let rendered = format!("{decision:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("event:one"));
    }
}
