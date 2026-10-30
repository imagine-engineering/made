use std::{collections::BTreeMap, error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    AggregateId, AggregateVersion, CommunityId, CommunityMembership, MembershipStatus,
    NostrEventId, NostrPublicKey, PrincipalId,
};

pub const MAX_SIGNED_PRESENCE_TTL_MILLIS: u64 = 180_000;
pub const MAX_ROOM_PRESENCE_TTL_MILLIS: u64 = 180_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PresenceSubject {
    pub community_id: CommunityId,
    pub principal_id: PrincipalId,
    pub nostr_public_key: NostrPublicKey,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresenceStatus {
    Online,
    Away,
    Offline,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct RoomPresenceSourceId(AggregateId);

impl RoomPresenceSourceId {
    pub const fn new(value: AggregateId) -> Self {
        Self(value)
    }

    pub const fn aggregate_id(self) -> AggregateId {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SignedPresenceObservation {
    community_id: CommunityId,
    principal_id: PrincipalId,
    event_id: NostrEventId,
    author: NostrPublicKey,
    status: PresenceStatus,
    observed_at_millis: u64,
    expires_at_millis: Option<u64>,
}

impl SignedPresenceObservation {
    pub fn from_verified_event(
        subject: PresenceSubject,
        event_id: NostrEventId,
        verified_author: NostrPublicKey,
        status: PresenceStatus,
        observed_at_millis: u64,
        expires_at_millis: Option<u64>,
    ) -> Result<Self, PresenceError> {
        if verified_author != subject.nostr_public_key {
            return Err(PresenceError::ForgedSignedObservation);
        }
        validate_expiry(
            status,
            observed_at_millis,
            expires_at_millis,
            Some(MAX_SIGNED_PRESENCE_TTL_MILLIS),
        )?;
        Ok(Self {
            community_id: subject.community_id,
            principal_id: subject.principal_id,
            event_id,
            author: verified_author,
            status,
            observed_at_millis,
            expires_at_millis,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoomPresenceObservation {
    community_id: CommunityId,
    principal_id: PrincipalId,
    source_id: RoomPresenceSourceId,
    sequence: u64,
    status: PresenceStatus,
    observed_at_millis: u64,
    expires_at_millis: Option<u64>,
}

impl RoomPresenceObservation {
    pub fn new(
        community_id: CommunityId,
        principal_id: PrincipalId,
        source_id: RoomPresenceSourceId,
        sequence: u64,
        status: PresenceStatus,
        observed_at_millis: u64,
        expires_at_millis: Option<u64>,
    ) -> Result<Self, PresenceError> {
        validate_expiry(
            status,
            observed_at_millis,
            expires_at_millis,
            Some(MAX_ROOM_PRESENCE_TTL_MILLIS),
        )?;
        Ok(Self {
            community_id,
            principal_id,
            source_id,
            sequence,
            status,
            observed_at_millis,
            expires_at_millis,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresenceMutationOutcome {
    Applied,
    Unchanged,
    IgnoredStale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresenceSources {
    pub signed: bool,
    pub room: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresenceSnapshot {
    pub status: PresenceStatus,
    pub active_sources: PresenceSources,
    pub refresh_at_millis: Option<u64>,
    pub membership_version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SignedPresenceState {
    observation: SignedPresenceObservation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RoomPresenceState {
    observation: RoomPresenceObservation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresenceProjection {
    subject: PresenceSubject,
    membership: CommunityMembership,
    signed: Option<SignedPresenceState>,
    rooms: BTreeMap<RoomPresenceSourceId, RoomPresenceState>,
}

impl PresenceProjection {
    pub fn new(
        subject: PresenceSubject,
        membership: CommunityMembership,
    ) -> Result<Self, PresenceError> {
        validate_membership(subject, membership)?;
        Ok(Self {
            subject,
            membership,
            signed: None,
            rooms: BTreeMap::new(),
        })
    }

    pub const fn subject(&self) -> PresenceSubject {
        self.subject
    }

    pub const fn membership(&self) -> CommunityMembership {
        self.membership
    }

    pub fn reconcile_membership(
        &mut self,
        membership: CommunityMembership,
    ) -> Result<PresenceMutationOutcome, PresenceError> {
        validate_membership(self.subject, membership)?;
        if membership.version < self.membership.version {
            return Ok(PresenceMutationOutcome::IgnoredStale);
        }
        if membership.version == self.membership.version {
            return if membership == self.membership {
                Ok(PresenceMutationOutcome::Unchanged)
            } else {
                Err(PresenceError::ConflictingMembershipVersion)
            };
        }
        if self.membership.status == MembershipStatus::Revoked
            && membership.status != MembershipStatus::Revoked
        {
            return Err(PresenceError::RevokedMembershipIsTerminal);
        }

        self.membership = membership;
        if membership.status != MembershipStatus::Active {
            self.signed = None;
            self.rooms.clear();
        }
        Ok(PresenceMutationOutcome::Applied)
    }

    pub fn apply_signed(
        &mut self,
        observation: SignedPresenceObservation,
    ) -> Result<PresenceMutationOutcome, PresenceError> {
        self.require_active_membership()?;
        if observation.community_id != self.subject.community_id
            || observation.principal_id != self.subject.principal_id
            || observation.author != self.subject.nostr_public_key
        {
            return Err(PresenceError::ObservationScopeMismatch);
        }

        if let Some(current) = self.signed {
            if observation == current.observation {
                return Ok(PresenceMutationOutcome::Unchanged);
            }
            match compare_signed_observations(observation, current.observation) {
                std::cmp::Ordering::Less => return Ok(PresenceMutationOutcome::IgnoredStale),
                std::cmp::Ordering::Equal => {
                    return Err(PresenceError::ConflictingObservationOrder);
                }
                std::cmp::Ordering::Greater => {}
            }
        }

        self.signed = Some(SignedPresenceState { observation });
        Ok(PresenceMutationOutcome::Applied)
    }

    pub fn apply_room(
        &mut self,
        observation: RoomPresenceObservation,
    ) -> Result<PresenceMutationOutcome, PresenceError> {
        self.require_active_membership()?;
        if observation.community_id != self.subject.community_id
            || observation.principal_id != self.subject.principal_id
        {
            return Err(PresenceError::ObservationScopeMismatch);
        }

        if let Some(current) = self.rooms.get(&observation.source_id) {
            if observation == current.observation {
                return Ok(PresenceMutationOutcome::Unchanged);
            }
            if observation.sequence < current.observation.sequence {
                return Ok(PresenceMutationOutcome::IgnoredStale);
            }
            if observation.sequence == current.observation.sequence {
                return Err(PresenceError::ConflictingObservationOrder);
            }
        }

        self.rooms
            .insert(observation.source_id, RoomPresenceState { observation });
        Ok(PresenceMutationOutcome::Applied)
    }

    pub fn snapshot(&self, now_millis: u64) -> PresenceSnapshot {
        if self.membership.status != MembershipStatus::Active {
            return PresenceSnapshot {
                status: PresenceStatus::Offline,
                active_sources: PresenceSources {
                    signed: false,
                    room: false,
                },
                refresh_at_millis: None,
                membership_version: self.membership.version,
            };
        }

        let signed = self
            .signed
            .map(|state| state.observation)
            .filter(|observation| observation_is_active(*observation, now_millis));
        let active_rooms = self
            .rooms
            .values()
            .map(|state| state.observation)
            .filter(|observation| observation_is_active(*observation, now_millis));

        let mut status = signed
            .map(|observation| observation.status)
            .unwrap_or(PresenceStatus::Offline);
        let mut room_source_active = false;
        let mut refresh_at_millis = signed.and_then(|observation| observation.expires_at_millis);
        for observation in active_rooms {
            room_source_active = true;
            status = merge_status(status, observation.status);
            refresh_at_millis = minimum_expiry(refresh_at_millis, observation.expires_at_millis);
        }

        PresenceSnapshot {
            status,
            active_sources: PresenceSources {
                signed: signed.is_some(),
                room: room_source_active,
            },
            refresh_at_millis,
            membership_version: self.membership.version,
        }
    }

    fn require_active_membership(&self) -> Result<(), PresenceError> {
        if self.membership.status == MembershipStatus::Active {
            Ok(())
        } else {
            Err(PresenceError::InactiveMembership)
        }
    }
}

trait PresenceObservation {
    fn status(self) -> PresenceStatus;
    fn expires_at_millis(self) -> Option<u64>;
}

impl PresenceObservation for SignedPresenceObservation {
    fn status(self) -> PresenceStatus {
        self.status
    }

    fn expires_at_millis(self) -> Option<u64> {
        self.expires_at_millis
    }
}

impl PresenceObservation for RoomPresenceObservation {
    fn status(self) -> PresenceStatus {
        self.status
    }

    fn expires_at_millis(self) -> Option<u64> {
        self.expires_at_millis
    }
}

fn validate_membership(
    subject: PresenceSubject,
    membership: CommunityMembership,
) -> Result<(), PresenceError> {
    if membership.community_id == subject.community_id
        && membership.principal_id == subject.principal_id
    {
        Ok(())
    } else {
        Err(PresenceError::MembershipScopeMismatch)
    }
}

fn validate_expiry(
    status: PresenceStatus,
    observed_at_millis: u64,
    expires_at_millis: Option<u64>,
    maximum_ttl_millis: Option<u64>,
) -> Result<(), PresenceError> {
    if status == PresenceStatus::Offline {
        return if expires_at_millis.is_none() {
            Ok(())
        } else {
            Err(PresenceError::OfflineObservationHasExpiry)
        };
    }
    let expires_at_millis = expires_at_millis.ok_or(PresenceError::MissingExpiry)?;
    let ttl_millis = expires_at_millis
        .checked_sub(observed_at_millis)
        .ok_or(PresenceError::InvalidExpiry)?;
    if ttl_millis == 0 {
        return Err(PresenceError::InvalidExpiry);
    }
    if maximum_ttl_millis.is_some_and(|maximum| ttl_millis > maximum) {
        return Err(PresenceError::ExpiryExceedsSourceLimit);
    }
    Ok(())
}

fn compare_signed_observations(
    candidate: SignedPresenceObservation,
    current: SignedPresenceObservation,
) -> std::cmp::Ordering {
    candidate
        .observed_at_millis
        .cmp(&current.observed_at_millis)
        .then_with(|| {
            current
                .event_id
                .as_bytes()
                .cmp(candidate.event_id.as_bytes())
        })
}

fn observation_is_active<T: PresenceObservation + Copy>(observation: T, now_millis: u64) -> bool {
    observation.status() != PresenceStatus::Offline
        && observation
            .expires_at_millis()
            .is_some_and(|expires_at| now_millis < expires_at)
}

fn merge_status(left: PresenceStatus, right: PresenceStatus) -> PresenceStatus {
    match (left, right) {
        (PresenceStatus::Online, _) | (_, PresenceStatus::Online) => PresenceStatus::Online,
        (PresenceStatus::Away, _) | (_, PresenceStatus::Away) => PresenceStatus::Away,
        (PresenceStatus::Offline, PresenceStatus::Offline) => PresenceStatus::Offline,
    }
}

fn minimum_expiry(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresenceError {
    MembershipScopeMismatch,
    InactiveMembership,
    ConflictingMembershipVersion,
    RevokedMembershipIsTerminal,
    ForgedSignedObservation,
    ObservationScopeMismatch,
    MissingExpiry,
    InvalidExpiry,
    ExpiryExceedsSourceLimit,
    OfflineObservationHasExpiry,
    ConflictingObservationOrder,
}

impl fmt::Display for PresenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MembershipScopeMismatch => "presence membership does not match the subject scope",
            Self::InactiveMembership => "presence cannot be updated without active membership",
            Self::ConflictingMembershipVersion => {
                "presence received conflicting membership state at the same version"
            }
            Self::RevokedMembershipIsTerminal => "revoked presence membership is terminal",
            Self::ForgedSignedObservation => {
                "signed presence author does not match the subject identity"
            }
            Self::ObservationScopeMismatch => {
                "presence observation does not match the projection scope"
            }
            Self::MissingExpiry => "active presence observation requires an expiry",
            Self::InvalidExpiry => "presence expiry must be after its observation time",
            Self::ExpiryExceedsSourceLimit => "presence expiry exceeds its source limit",
            Self::OfflineObservationHasExpiry => "offline presence must not carry an expiry",
            Self::ConflictingObservationOrder => {
                "presence observations conflict at the same source order"
            }
        })
    }
}

impl Error for PresenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MembershipRole;
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

    fn public_key(value: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([value; 32])
    }

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn subject() -> PresenceSubject {
        PresenceSubject {
            community_id: community(1),
            principal_id: principal(2),
            nostr_public_key: public_key(3),
        }
    }

    fn membership(status: MembershipStatus, version: u64) -> CommunityMembership {
        CommunityMembership {
            community_id: subject().community_id,
            principal_id: subject().principal_id,
            role: MembershipRole::Member,
            status,
            version: AggregateVersion::new(version).expect("non-zero test version"),
        }
    }

    fn projection() -> PresenceProjection {
        PresenceProjection::new(subject(), membership(MembershipStatus::Active, 1))
            .expect("valid presence projection")
    }

    fn signed(
        status: PresenceStatus,
        observed_at_millis: u64,
        expires_at_millis: Option<u64>,
    ) -> SignedPresenceObservation {
        SignedPresenceObservation::from_verified_event(
            subject(),
            event_id(observed_at_millis as u8),
            subject().nostr_public_key,
            status,
            observed_at_millis,
            expires_at_millis,
        )
        .expect("valid signed presence")
    }

    #[test]
    fn forged_signed_state_is_rejected_without_mutating_projection() {
        let mut projection = projection();
        let error = SignedPresenceObservation::from_verified_event(
            subject(),
            event_id(1),
            public_key(99),
            PresenceStatus::Online,
            1_000,
            Some(181_000),
        )
        .expect_err("mismatched author must be rejected");

        assert_eq!(error, PresenceError::ForgedSignedObservation);
        assert_eq!(projection.snapshot(1_000).status, PresenceStatus::Offline);

        let cross_community = RoomPresenceObservation::new(
            community(99),
            subject().principal_id,
            RoomPresenceSourceId::new(aggregate(1)),
            1,
            PresenceStatus::Online,
            1_000,
            Some(2_000),
        )
        .expect("valid source-local expiry");
        assert_eq!(
            projection.apply_room(cross_community),
            Err(PresenceError::ObservationScopeMismatch)
        );
        assert_eq!(projection.snapshot(1_000).status, PresenceStatus::Offline);
    }

    #[test]
    fn signed_presence_expires_at_the_buzz_ttl_boundary() {
        let mut projection = projection();
        projection
            .apply_signed(signed(PresenceStatus::Online, 1_000, Some(181_000)))
            .expect("apply signed presence");

        assert_eq!(projection.snapshot(180_999).status, PresenceStatus::Online);
        assert_eq!(projection.snapshot(181_000).status, PresenceStatus::Offline);
        assert_eq!(projection.snapshot(181_000).refresh_at_millis, None);
        assert_eq!(
            SignedPresenceObservation::from_verified_event(
                subject(),
                event_id(2),
                subject().nostr_public_key,
                PresenceStatus::Away,
                1_000,
                Some(181_001),
            ),
            Err(PresenceError::ExpiryExceedsSourceLimit)
        );
        assert_eq!(
            RoomPresenceObservation::new(
                subject().community_id,
                subject().principal_id,
                RoomPresenceSourceId::new(aggregate(1)),
                1,
                PresenceStatus::Online,
                1_000,
                Some(181_001),
            ),
            Err(PresenceError::ExpiryExceedsSourceLimit)
        );
    }

    #[test]
    fn multiple_sources_merge_and_expire_independently() {
        let mut projection = projection();
        projection
            .apply_signed(signed(PresenceStatus::Away, 1_000, Some(181_000)))
            .expect("apply signed presence");
        projection
            .apply_room(
                RoomPresenceObservation::new(
                    subject().community_id,
                    subject().principal_id,
                    RoomPresenceSourceId::new(aggregate(4)),
                    1,
                    PresenceStatus::Online,
                    1_000,
                    Some(5_000),
                )
                .expect("valid room presence"),
            )
            .expect("apply room presence");

        assert_eq!(
            projection.snapshot(4_999),
            PresenceSnapshot {
                status: PresenceStatus::Online,
                active_sources: PresenceSources {
                    signed: true,
                    room: true,
                },
                refresh_at_millis: Some(5_000),
                membership_version: AggregateVersion::FIRST,
            }
        );
        assert_eq!(projection.snapshot(5_000).status, PresenceStatus::Away);

        projection
            .apply_signed(signed(PresenceStatus::Offline, 2_000, None))
            .expect("clear only signed presence");
        let room = RoomPresenceObservation::new(
            subject().community_id,
            subject().principal_id,
            RoomPresenceSourceId::new(aggregate(5)),
            1,
            PresenceStatus::Online,
            2_000,
            Some(6_000),
        )
        .expect("valid second room presence");
        projection.apply_room(room).expect("apply second room");
        assert_eq!(projection.snapshot(2_000).status, PresenceStatus::Online);
        assert_eq!(
            projection.snapshot(2_000).active_sources,
            PresenceSources {
                signed: false,
                room: true,
            }
        );
    }

    #[test]
    fn stale_source_updates_cannot_resurrect_cleared_presence() {
        let mut projection = projection();
        projection
            .apply_signed(signed(PresenceStatus::Offline, 2_000, None))
            .expect("apply signed tombstone");
        assert_eq!(
            projection.apply_signed(signed(PresenceStatus::Online, 1_000, Some(2_000))),
            Ok(PresenceMutationOutcome::IgnoredStale)
        );

        let source_id = RoomPresenceSourceId::new(aggregate(4));
        let disconnected = RoomPresenceObservation::new(
            subject().community_id,
            subject().principal_id,
            source_id,
            2,
            PresenceStatus::Offline,
            2_000,
            None,
        )
        .expect("valid disconnect");
        projection
            .apply_room(disconnected)
            .expect("apply disconnect");
        let stale = RoomPresenceObservation::new(
            subject().community_id,
            subject().principal_id,
            source_id,
            1,
            PresenceStatus::Online,
            1_000,
            Some(4_000),
        )
        .expect("valid stale room presence");
        assert_eq!(
            projection.apply_room(stale),
            Ok(PresenceMutationOutcome::IgnoredStale)
        );
        assert_eq!(projection.snapshot(2_000).status, PresenceStatus::Offline);
    }

    #[test]
    fn revoked_membership_clears_presence_and_cannot_be_reactivated() {
        let mut projection = projection();
        projection
            .apply_signed(signed(PresenceStatus::Online, 1_000, Some(181_000)))
            .expect("apply presence");
        assert_eq!(
            projection.reconcile_membership(membership(MembershipStatus::Revoked, 2)),
            Ok(PresenceMutationOutcome::Applied)
        );

        assert_eq!(projection.snapshot(1_000).status, PresenceStatus::Offline);
        assert_eq!(
            projection.apply_signed(signed(PresenceStatus::Online, 2_000, Some(182_000))),
            Err(PresenceError::InactiveMembership)
        );
        assert_eq!(
            projection.reconcile_membership(membership(MembershipStatus::Active, 3)),
            Err(PresenceError::RevokedMembershipIsTerminal)
        );
        assert_eq!(projection.snapshot(2_000).status, PresenceStatus::Offline);
    }
}
