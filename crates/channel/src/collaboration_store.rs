use std::{collections::BTreeMap, error::Error, fmt};

use collaboration_domain::{
    AggregateId, AggregateVersion, Channel as DomainChannel, ChannelLifecycleState,
    ChannelMetadata, ChannelMetadataRecordFields, ChannelRecordFields, ChannelType,
    ChannelVisibility, Community, CommunityId, CommunityRecordFields, Membership,
    MembershipRecordFields, MembershipRole, MembershipScope, MembershipStatus, PrincipalId,
};
use rpc::proto;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CanonicalChannelId {
    community_id: CommunityId,
    channel_id: AggregateId,
}

impl CanonicalChannelId {
    pub const fn new(community_id: CommunityId, channel_id: AggregateId) -> Self {
        Self {
            community_id,
            channel_id,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationSnapshot {
    pub snapshot_version: AggregateVersion,
    pub community: CommunityRecordFields,
    pub channels: Vec<ChannelRecordFields>,
    pub memberships: Vec<MembershipRecordFields>,
    pub metadata: Vec<ChannelMetadataRecordFields>,
}

impl CollaborationSnapshot {
    pub fn from_domain(
        snapshot_version: AggregateVersion,
        community: &Community,
        channels: &[DomainChannel],
        memberships: &[Membership],
        metadata: &[ChannelMetadata],
    ) -> Self {
        Self {
            snapshot_version,
            community: community.fields().clone(),
            channels: channels
                .iter()
                .map(|channel| channel.fields().clone())
                .collect(),
            memberships: memberships
                .iter()
                .copied()
                .map(Membership::fields)
                .collect(),
            metadata: metadata
                .iter()
                .map(|metadata| metadata.fields().clone())
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationChannelProjection {
    id: CanonicalChannelId,
    channel: ChannelRecordFields,
    metadata: Option<ChannelMetadataRecordFields>,
    members: BTreeMap<PrincipalId, MembershipRecordFields>,
}

impl CollaborationChannelProjection {
    pub const fn id(&self) -> CanonicalChannelId {
        self.id
    }

    pub const fn channel(&self) -> &ChannelRecordFields {
        &self.channel
    }

    pub const fn channel_type(&self) -> ChannelType {
        self.channel.channel_type
    }

    pub const fn lifecycle_state(&self) -> ChannelLifecycleState {
        self.channel.lifecycle_state
    }

    pub const fn metadata(&self) -> Option<&ChannelMetadataRecordFields> {
        self.metadata.as_ref()
    }

    pub fn members(&self) -> impl Iterator<Item = &MembershipRecordFields> {
        self.members.values()
    }

    pub fn membership(&self, principal_id: PrincipalId) -> Option<&MembershipRecordFields> {
        self.members.get(&principal_id)
    }

    pub fn role(&self, principal_id: PrincipalId) -> Option<MembershipRole> {
        self.membership(principal_id)
            .filter(|membership| membership.status == MembershipStatus::Active)
            .map(|membership| membership.role)
    }

    pub const fn native_visibility(&self) -> proto::ChannelVisibility {
        match self.channel.visibility {
            ChannelVisibility::Open => proto::ChannelVisibility::Public,
            ChannelVisibility::Private => proto::ChannelVisibility::Members,
        }
    }

    pub fn legacy_role(
        &self,
        principal_id: PrincipalId,
    ) -> Result<Option<proto::ChannelRole>, CollaborationProjectionError> {
        let Some(role) = self.role(principal_id) else {
            return Ok(None);
        };
        match role {
            MembershipRole::Owner | MembershipRole::Admin => Ok(Some(proto::ChannelRole::Admin)),
            MembershipRole::Member => Ok(Some(proto::ChannelRole::Member)),
            MembershipRole::Guest => Ok(Some(proto::ChannelRole::Guest)),
            MembershipRole::Bot => Err(CollaborationProjectionError::LegacyRoleUnsupported {
                channel_id: self.id,
                principal_id,
                role,
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationCommunityProjection {
    snapshot_version: AggregateVersion,
    community: CommunityRecordFields,
    channels: BTreeMap<CanonicalChannelId, CollaborationChannelProjection>,
    members: BTreeMap<PrincipalId, MembershipRecordFields>,
}

impl CollaborationCommunityProjection {
    pub const fn snapshot_version(&self) -> AggregateVersion {
        self.snapshot_version
    }

    pub const fn community(&self) -> &CommunityRecordFields {
        &self.community
    }

    pub fn channels(&self) -> impl Iterator<Item = &CollaborationChannelProjection> {
        self.channels.values()
    }

    pub fn channel(
        &self,
        channel_id: CanonicalChannelId,
    ) -> Option<&CollaborationChannelProjection> {
        self.channels.get(&channel_id)
    }

    pub fn members(&self) -> impl Iterator<Item = &MembershipRecordFields> {
        self.members.values()
    }

    fn build(snapshot: CollaborationSnapshot) -> Result<Self, CollaborationProjectionError> {
        let community_id = snapshot.community.community_id;
        let mut channels = BTreeMap::new();
        for channel in snapshot.channels {
            if channel.community_id != community_id {
                return Err(CollaborationProjectionError::CrossCommunityChannel);
            }
            let id = CanonicalChannelId::new(community_id, channel.channel_id);
            if channels
                .insert(
                    id,
                    CollaborationChannelProjection {
                        id,
                        channel,
                        metadata: None,
                        members: BTreeMap::new(),
                    },
                )
                .is_some()
            {
                return Err(CollaborationProjectionError::DuplicateChannel(id));
            }
        }

        for metadata in snapshot.metadata {
            if metadata.community_id != community_id {
                return Err(CollaborationProjectionError::CrossCommunityMetadata);
            }
            let id = CanonicalChannelId::new(community_id, metadata.channel_id);
            let channel = channels
                .get_mut(&id)
                .ok_or(CollaborationProjectionError::UnknownMetadataChannel(id))?;
            if channel.metadata.replace(metadata).is_some() {
                return Err(CollaborationProjectionError::DuplicateMetadata(id));
            }
        }

        let mut members = BTreeMap::new();
        for membership in snapshot.memberships {
            if membership.community_id != community_id {
                return Err(CollaborationProjectionError::CrossCommunityMembership);
            }
            match membership.scope {
                MembershipScope::Community => {
                    let principal_id = membership.principal_id;
                    if members.insert(principal_id, membership).is_some() {
                        return Err(CollaborationProjectionError::DuplicateCommunityMember(
                            principal_id,
                        ));
                    }
                }
                MembershipScope::Channel(channel_id) => {
                    let id = CanonicalChannelId::new(community_id, channel_id);
                    let channel = channels
                        .get_mut(&id)
                        .ok_or(CollaborationProjectionError::UnknownMembershipChannel(id))?;
                    let principal_id = membership.principal_id;
                    if channel.members.insert(principal_id, membership).is_some() {
                        return Err(CollaborationProjectionError::DuplicateChannelMember {
                            channel_id: id,
                            principal_id,
                        });
                    }
                }
            }
        }

        Ok(Self {
            snapshot_version: snapshot.snapshot_version,
            community: snapshot.community,
            channels,
            members,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationProjectionOutcome {
    Applied,
    Unchanged,
}

#[derive(Default, Debug)]
pub struct CollaborationStore {
    communities: BTreeMap<CommunityId, CollaborationCommunityProjection>,
}

impl CollaborationStore {
    pub fn communities(&self) -> impl Iterator<Item = &CollaborationCommunityProjection> {
        self.communities.values()
    }

    pub fn community(
        &self,
        community_id: CommunityId,
    ) -> Option<&CollaborationCommunityProjection> {
        self.communities.get(&community_id)
    }

    pub fn channel(
        &self,
        channel_id: CanonicalChannelId,
    ) -> Option<&CollaborationChannelProjection> {
        self.community(channel_id.community_id())?
            .channel(channel_id)
    }

    pub fn replace(
        &mut self,
        snapshot: CollaborationSnapshot,
    ) -> Result<CollaborationProjectionOutcome, CollaborationProjectionError> {
        let projection = CollaborationCommunityProjection::build(snapshot)?;
        let community_id = projection.community.community_id;
        if let Some(current) = self.communities.get(&community_id) {
            if projection.snapshot_version < current.snapshot_version {
                return Err(CollaborationProjectionError::StaleSnapshot {
                    expected_after: current.snapshot_version,
                    received: projection.snapshot_version,
                });
            }
            if projection.snapshot_version == current.snapshot_version {
                return if projection == *current {
                    Ok(CollaborationProjectionOutcome::Unchanged)
                } else {
                    Err(CollaborationProjectionError::SnapshotConflict(
                        projection.snapshot_version,
                    ))
                };
            }
        }
        self.communities.insert(community_id, projection);
        Ok(CollaborationProjectionOutcome::Applied)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationProjectionError {
    CrossCommunityChannel,
    CrossCommunityMembership,
    CrossCommunityMetadata,
    DuplicateChannel(CanonicalChannelId),
    DuplicateCommunityMember(PrincipalId),
    DuplicateChannelMember {
        channel_id: CanonicalChannelId,
        principal_id: PrincipalId,
    },
    DuplicateMetadata(CanonicalChannelId),
    UnknownMembershipChannel(CanonicalChannelId),
    UnknownMetadataChannel(CanonicalChannelId),
    StaleSnapshot {
        expected_after: AggregateVersion,
        received: AggregateVersion,
    },
    SnapshotConflict(AggregateVersion),
    LegacyRoleUnsupported {
        channel_id: CanonicalChannelId,
        principal_id: PrincipalId,
        role: MembershipRole,
    },
}

impl fmt::Display for CollaborationProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrossCommunityChannel
            | Self::CrossCommunityMembership
            | Self::CrossCommunityMetadata => {
                formatter.write_str("collaboration projection crosses a community boundary")
            }
            Self::DuplicateChannel(_)
            | Self::DuplicateCommunityMember(_)
            | Self::DuplicateChannelMember { .. }
            | Self::DuplicateMetadata(_) => {
                formatter.write_str("collaboration projection contains a duplicate identity")
            }
            Self::UnknownMembershipChannel(_) | Self::UnknownMetadataChannel(_) => {
                formatter.write_str("collaboration projection references an unknown channel")
            }
            Self::StaleSnapshot { .. } => {
                formatter.write_str("collaboration projection snapshot is stale")
            }
            Self::SnapshotConflict(_) => {
                formatter.write_str("collaboration projection version has conflicting content")
            }
            Self::LegacyRoleUnsupported { .. } => {
                formatter.write_str("canonical role has no safe legacy channel-role projection")
            }
        }
    }
}

impl Error for CollaborationProjectionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use collaboration_domain::{
        ChannelDescription, ChannelExpiration, ChannelMetadataText, ChannelName, CommunityHost,
        CommunityJoinPolicy, CommunityLifecycleState,
    };
    use std::num::NonZeroU32;
    use uuid::Uuid;

    fn community_id(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn community(community_id: CommunityId) -> Community {
        Community::from_record(CommunityRecordFields {
            community_id,
            host: CommunityHost::new("example.test").expect("host"),
            icon: None,
            lifecycle_state: CommunityLifecycleState::Active,
            join_policy: CommunityJoinPolicy::Open,
            version: AggregateVersion::FIRST,
        })
    }

    fn channel(
        community_id: CommunityId,
        channel_id: AggregateId,
        channel_type: ChannelType,
    ) -> DomainChannel {
        let visibility = if matches!(
            channel_type,
            ChannelType::DirectMessage | ChannelType::Workflow | ChannelType::Huddle
        ) {
            ChannelVisibility::Private
        } else {
            ChannelVisibility::Open
        };
        let expiration = if channel_type == ChannelType::Ephemeral {
            Some(
                ChannelExpiration::starting_at(NonZeroU32::new(60).expect("ttl"), 1)
                    .expect("expiration"),
            )
        } else {
            None
        };
        DomainChannel::from_record(ChannelRecordFields {
            community_id,
            channel_id,
            name: ChannelName::new(format!("channel-{channel_id}")).expect("name"),
            channel_type,
            visibility,
            lifecycle_state: ChannelLifecycleState::Active,
            description: Some(ChannelDescription::new("description").expect("description")),
            creator_principal_id: principal_id(100),
            expiration,
            version: AggregateVersion::FIRST,
        })
        .expect("channel")
    }

    fn membership(
        community_id: CommunityId,
        channel_id: AggregateId,
        principal_id: PrincipalId,
        role: MembershipRole,
        status: MembershipStatus,
    ) -> Membership {
        Membership::from_record(MembershipRecordFields {
            community_id,
            scope: MembershipScope::Channel(channel_id),
            principal_id,
            role,
            status,
            version: AggregateVersion::FIRST,
            added_by_principal_id: None,
        })
    }

    #[test]
    fn collaboration_store_preserves_one_canonical_id_and_every_channel_type() {
        let community_id = community_id(1);
        let channel_types = [
            ChannelType::Stream,
            ChannelType::Forum,
            ChannelType::DirectMessage,
            ChannelType::Workflow,
            ChannelType::Ephemeral,
            ChannelType::Huddle,
        ];
        let channels = channel_types
            .into_iter()
            .enumerate()
            .map(|(index, channel_type)| {
                channel(community_id, aggregate_id(index as u128 + 10), channel_type)
            })
            .collect::<Vec<_>>();
        let metadata = ChannelMetadata::from_record(ChannelMetadataRecordFields {
            community_id,
            channel_id: aggregate_id(10),
            topic: Some(ChannelMetadataText::new("topic").expect("topic")),
            canvas: None,
            version: AggregateVersion::FIRST,
            updated_by_principal_id: principal_id(100),
            updated_at_millis: 1,
        });
        let snapshot = CollaborationSnapshot::from_domain(
            AggregateVersion::FIRST,
            &community(community_id),
            &channels,
            &[],
            &[metadata],
        );
        let mut store = CollaborationStore::default();
        assert_eq!(
            store.replace(snapshot.clone()),
            Ok(CollaborationProjectionOutcome::Applied)
        );
        assert_eq!(
            store.replace(snapshot),
            Ok(CollaborationProjectionOutcome::Unchanged)
        );

        let projection = store.community(community_id).expect("community");
        assert_eq!(
            projection
                .channels()
                .map(CollaborationChannelProjection::channel_type)
                .collect::<Vec<_>>(),
            channel_types
        );
        for (index, channel) in projection.channels().enumerate() {
            let expected = CanonicalChannelId::new(community_id, aggregate_id(index as u128 + 10));
            assert_eq!(channel.id(), expected);
            assert_eq!(
                store.channel(expected).map(|channel| channel.id()),
                Some(expected)
            );
        }
        assert_eq!(
            store
                .channel(CanonicalChannelId::new(community_id, aggregate_id(10)))
                .and_then(CollaborationChannelProjection::metadata)
                .and_then(|metadata| metadata.topic.as_ref())
                .map(ChannelMetadataText::as_str),
            Some("topic")
        );
    }

    #[test]
    fn collaboration_store_projects_roles_without_granting_bot_legacy_authority() {
        let community_id = community_id(1);
        let channel_id = aggregate_id(10);
        let roles = [
            MembershipRole::Owner,
            MembershipRole::Admin,
            MembershipRole::Member,
            MembershipRole::Guest,
            MembershipRole::Bot,
        ];
        let memberships = roles
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                membership(
                    community_id,
                    channel_id,
                    principal_id(index as u128 + 20),
                    role,
                    MembershipStatus::Active,
                )
            })
            .chain([membership(
                community_id,
                channel_id,
                principal_id(30),
                MembershipRole::Member,
                MembershipStatus::Revoked,
            )])
            .collect::<Vec<_>>();
        let snapshot = CollaborationSnapshot::from_domain(
            AggregateVersion::FIRST,
            &community(community_id),
            &[channel(community_id, channel_id, ChannelType::Stream)],
            &memberships,
            &[],
        );
        let mut store = CollaborationStore::default();
        store.replace(snapshot).expect("projection");
        let channel = store
            .channel(CanonicalChannelId::new(community_id, channel_id))
            .expect("channel");

        for (index, role) in roles.into_iter().enumerate() {
            assert_eq!(channel.role(principal_id(index as u128 + 20)), Some(role));
        }
        assert_eq!(
            channel.legacy_role(principal_id(20)),
            Ok(Some(proto::ChannelRole::Admin))
        );
        assert_eq!(
            channel.legacy_role(principal_id(22)),
            Ok(Some(proto::ChannelRole::Member))
        );
        assert_eq!(
            channel.legacy_role(principal_id(23)),
            Ok(Some(proto::ChannelRole::Guest))
        );
        assert!(matches!(
            channel.legacy_role(principal_id(24)),
            Err(CollaborationProjectionError::LegacyRoleUnsupported {
                role: MembershipRole::Bot,
                ..
            })
        ));
        assert_eq!(channel.role(principal_id(30)), None);
    }

    #[test]
    fn collaboration_store_rejects_stale_conflicting_and_cross_tenant_snapshots() {
        let community_id = community_id(1);
        let channel_id = aggregate_id(10);
        let base = CollaborationSnapshot::from_domain(
            AggregateVersion::new(2).expect("version"),
            &community(community_id),
            &[channel(community_id, channel_id, ChannelType::Stream)],
            &[],
            &[],
        );
        let mut store = CollaborationStore::default();
        store.replace(base.clone()).expect("projection");

        let mut conflict = base.clone();
        conflict.channels[0].name = ChannelName::new("changed").expect("name");
        assert_eq!(
            store.replace(conflict),
            Err(CollaborationProjectionError::SnapshotConflict(
                AggregateVersion::new(2).expect("version")
            ))
        );
        let mut stale = base.clone();
        stale.snapshot_version = AggregateVersion::FIRST;
        assert!(matches!(
            store.replace(stale),
            Err(CollaborationProjectionError::StaleSnapshot { .. })
        ));

        let mut foreign = base;
        foreign.snapshot_version = AggregateVersion::new(3).expect("version");
        foreign.channels[0].community_id = CommunityId::from_uuid(Uuid::from_u128(2));
        assert_eq!(
            store.replace(foreign),
            Err(CollaborationProjectionError::CrossCommunityChannel)
        );
        assert_eq!(
            store
                .community(community_id)
                .expect("community")
                .snapshot_version(),
            AggregateVersion::new(2).expect("version")
        );
    }
}
