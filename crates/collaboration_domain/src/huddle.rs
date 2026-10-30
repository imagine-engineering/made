use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    num::NonZeroU64,
};

use crate::{AggregateId, AggregateVersion, CommunityId, OperationId, PrincipalId};

pub const MAX_HUDDLE_PARTICIPANTS: usize = 256;
pub const MAX_HUDDLE_EVENTS: usize = 10_000;
pub const MAX_HUDDLE_REACTIONS: usize = 1_024;
pub const MAX_HUDDLE_TRANSCRIPT_REFERENCES: usize = 10_000;
const MAX_HUDDLE_REACTION_BYTES: usize = 256;
const MAX_HUDDLE_REACTION_CHARACTERS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HuddleGeneration(NonZeroU64);

impl HuddleGeneration {
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HuddleIdentity {
    community_id: CommunityId,
    channel_id: AggregateId,
    huddle_id: AggregateId,
    generation: HuddleGeneration,
}

impl HuddleIdentity {
    pub fn new(
        community_id: CommunityId,
        channel_id: AggregateId,
        huddle_id: AggregateId,
        generation: HuddleGeneration,
    ) -> Result<Self, HuddleError> {
        if community_id.as_uuid().is_nil()
            || channel_id.as_uuid().is_nil()
            || huddle_id.as_uuid().is_nil()
        {
            return Err(HuddleError::InvalidIdentity);
        }
        Ok(Self {
            community_id,
            channel_id,
            huddle_id,
            generation,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }

    pub const fn huddle_id(self) -> AggregateId {
        self.huddle_id
    }

    pub const fn generation(self) -> HuddleGeneration {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleParticipantRole {
    Owner,
    Moderator,
    Speaker,
    Listener,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleParticipantPresence {
    Present,
    Left,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleModerationState {
    Unrestricted,
    Muted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HuddleParticipant {
    principal_id: PrincipalId,
    role: HuddleParticipantRole,
    presence: HuddleParticipantPresence,
    moderation: HuddleModerationState,
    joined_at_millis: u64,
    left_at_millis: Option<u64>,
}

impl HuddleParticipant {
    pub const fn principal_id(self) -> PrincipalId {
        self.principal_id
    }

    pub const fn role(self) -> HuddleParticipantRole {
        self.role
    }

    pub const fn presence(self) -> HuddleParticipantPresence {
        self.presence
    }

    pub const fn moderation(self) -> HuddleModerationState {
        self.moderation
    }

    pub const fn joined_at_millis(self) -> u64 {
        self.joined_at_millis
    }

    pub const fn left_at_millis(self) -> Option<u64> {
        self.left_at_millis
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HuddleReactionValue(String);

impl HuddleReactionValue {
    pub fn new(value: impl Into<String>) -> Result<Self, HuddleError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_HUDDLE_REACTION_BYTES
            || value.chars().count() > MAX_HUDDLE_REACTION_CHARACTERS
            || value.chars().any(char::is_control)
        {
            return Err(HuddleError::InvalidReaction);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleReaction {
    pub participant_principal_id: PrincipalId,
    pub value: HuddleReactionValue,
    pub occurred_at_millis: u64,
    pub operation_id: OperationId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HuddleTranscriptSegmentId(AggregateId);

impl HuddleTranscriptSegmentId {
    pub fn new(value: AggregateId) -> Result<Self, HuddleError> {
        if value.as_uuid().is_nil() {
            return Err(HuddleError::InvalidTranscriptReference);
        }
        Ok(Self(value))
    }

    pub const fn aggregate_id(self) -> AggregateId {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HuddleTranscriptReference {
    identity: HuddleIdentity,
    segment_id: HuddleTranscriptSegmentId,
    message_id: AggregateId,
    participant_principal_id: PrincipalId,
    started_at_millis: u64,
    ended_at_millis: u64,
}

impl HuddleTranscriptReference {
    pub fn new(
        identity: HuddleIdentity,
        segment_id: HuddleTranscriptSegmentId,
        message_id: AggregateId,
        participant_principal_id: PrincipalId,
        started_at_millis: u64,
        ended_at_millis: u64,
    ) -> Result<Self, HuddleError> {
        if message_id.as_uuid().is_nil()
            || participant_principal_id.as_uuid().is_nil()
            || started_at_millis == 0
            || ended_at_millis <= started_at_millis
        {
            return Err(HuddleError::InvalidTranscriptReference);
        }
        Ok(Self {
            identity,
            segment_id,
            message_id,
            participant_principal_id,
            started_at_millis,
            ended_at_millis,
        })
    }

    pub const fn identity(self) -> HuddleIdentity {
        self.identity
    }

    pub const fn segment_id(self) -> HuddleTranscriptSegmentId {
        self.segment_id
    }

    pub const fn message_id(self) -> AggregateId {
        self.message_id
    }

    pub const fn participant_principal_id(self) -> PrincipalId {
        self.participant_principal_id
    }

    pub const fn started_at_millis(self) -> u64 {
        self.started_at_millis
    }

    pub const fn ended_at_millis(self) -> u64 {
        self.ended_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleEndReason {
    Explicit,
    Empty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleLifecycleState {
    Active,
    Ended {
        reason: HuddleEndReason,
        ended_at_millis: u64,
        ended_by_principal_id: Option<PrincipalId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HuddleEvent {
    Started {
        identity: HuddleIdentity,
        owner_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    ParticipantJoined {
        participant_principal_id: PrincipalId,
        role: HuddleParticipantRole,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    ParticipantLeft {
        participant_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
        ended_empty: bool,
        resulting_version: AggregateVersion,
    },
    ParticipantRoleChanged {
        actor_principal_id: PrincipalId,
        participant_principal_id: PrincipalId,
        role: HuddleParticipantRole,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    ModerationChanged {
        actor_principal_id: PrincipalId,
        participant_principal_id: PrincipalId,
        state: HuddleModerationState,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    ReactionAdded {
        participant_principal_id: PrincipalId,
        value: HuddleReactionValue,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    TranscriptLinked {
        reference: HuddleTranscriptReference,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
    Ended {
        actor_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
        resulting_version: AggregateVersion,
    },
}

impl HuddleEvent {
    pub const fn operation_id(&self) -> OperationId {
        match self {
            Self::Started { operation_id, .. }
            | Self::ParticipantJoined { operation_id, .. }
            | Self::ParticipantLeft { operation_id, .. }
            | Self::ParticipantRoleChanged { operation_id, .. }
            | Self::ModerationChanged { operation_id, .. }
            | Self::ReactionAdded { operation_id, .. }
            | Self::TranscriptLinked { operation_id, .. }
            | Self::Ended { operation_id, .. } => *operation_id,
        }
    }

    pub const fn occurred_at_millis(&self) -> u64 {
        match self {
            Self::Started {
                occurred_at_millis, ..
            }
            | Self::ParticipantJoined {
                occurred_at_millis, ..
            }
            | Self::ParticipantLeft {
                occurred_at_millis, ..
            }
            | Self::ParticipantRoleChanged {
                occurred_at_millis, ..
            }
            | Self::ModerationChanged {
                occurred_at_millis, ..
            }
            | Self::ReactionAdded {
                occurred_at_millis, ..
            }
            | Self::TranscriptLinked {
                occurred_at_millis, ..
            }
            | Self::Ended {
                occurred_at_millis, ..
            } => *occurred_at_millis,
        }
    }

    pub const fn resulting_version(&self) -> AggregateVersion {
        match self {
            Self::Started {
                resulting_version, ..
            }
            | Self::ParticipantJoined {
                resulting_version, ..
            }
            | Self::ParticipantLeft {
                resulting_version, ..
            }
            | Self::ParticipantRoleChanged {
                resulting_version, ..
            }
            | Self::ModerationChanged {
                resulting_version, ..
            }
            | Self::ReactionAdded {
                resulting_version, ..
            }
            | Self::TranscriptLinked {
                resulting_version, ..
            }
            | Self::Ended {
                resulting_version, ..
            } => *resulting_version,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleRecordFields {
    pub identity: HuddleIdentity,
    pub events: Vec<HuddleEvent>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Huddle {
    fields: HuddleRecordFields,
    owner_principal_id: PrincipalId,
    started_at_millis: u64,
    lifecycle: HuddleLifecycleState,
    participants: BTreeMap<PrincipalId, HuddleParticipant>,
    reactions: Vec<HuddleReaction>,
    transcripts: BTreeMap<HuddleTranscriptSegmentId, HuddleTranscriptReference>,
    operations: BTreeSet<OperationId>,
}

impl Huddle {
    pub fn start(
        identity: HuddleIdentity,
        owner_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<Self, HuddleError> {
        validate_principal(owner_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        let event = HuddleEvent::Started {
            identity,
            owner_principal_id,
            operation_id,
            occurred_at_millis,
            resulting_version: AggregateVersion::FIRST,
        };
        let owner = HuddleParticipant {
            principal_id: owner_principal_id,
            role: HuddleParticipantRole::Owner,
            presence: HuddleParticipantPresence::Present,
            moderation: HuddleModerationState::Unrestricted,
            joined_at_millis: occurred_at_millis,
            left_at_millis: None,
        };
        Ok(Self {
            fields: HuddleRecordFields {
                identity,
                events: vec![event],
                version: AggregateVersion::FIRST,
            },
            owner_principal_id,
            started_at_millis: occurred_at_millis,
            lifecycle: HuddleLifecycleState::Active,
            participants: BTreeMap::from([(owner_principal_id, owner)]),
            reactions: Vec::new(),
            transcripts: BTreeMap::new(),
            operations: BTreeSet::from([operation_id]),
        })
    }

    pub fn from_record(fields: HuddleRecordFields) -> Result<Self, HuddleError> {
        if fields.events.is_empty() || fields.events.len() > MAX_HUDDLE_EVENTS {
            return Err(HuddleError::InvalidRecord);
        }
        let HuddleEvent::Started {
            identity,
            owner_principal_id,
            operation_id,
            occurred_at_millis,
            resulting_version,
        } = &fields.events[0]
        else {
            return Err(HuddleError::InvalidRecord);
        };
        if *identity != fields.identity || *resulting_version != AggregateVersion::FIRST {
            return Err(HuddleError::InvalidRecord);
        }
        let mut huddle = Self::start(
            *identity,
            *owner_principal_id,
            *operation_id,
            *occurred_at_millis,
        )?;
        for event in fields.events.iter().skip(1) {
            huddle.append_replayed(event.clone())?;
        }
        if huddle.fields.version != fields.version {
            return Err(HuddleError::InvalidRecord);
        }
        Ok(huddle)
    }

    pub const fn fields(&self) -> &HuddleRecordFields {
        &self.fields
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.fields.identity
    }

    pub const fn owner_principal_id(&self) -> PrincipalId {
        self.owner_principal_id
    }

    pub const fn started_at_millis(&self) -> u64 {
        self.started_at_millis
    }

    pub const fn lifecycle(&self) -> HuddleLifecycleState {
        self.lifecycle
    }

    pub fn participants(&self) -> impl Iterator<Item = HuddleParticipant> + '_ {
        self.participants.values().copied()
    }

    pub fn participant(&self, principal_id: PrincipalId) -> Option<HuddleParticipant> {
        self.participants.get(&principal_id).copied()
    }

    pub fn reactions(&self) -> &[HuddleReaction] {
        &self.reactions
    }

    pub fn transcript_references(&self) -> impl Iterator<Item = HuddleTranscriptReference> + '_ {
        self.transcripts.values().copied()
    }

    pub fn join(
        &mut self,
        participant_principal_id: PrincipalId,
        role: HuddleParticipantRole,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(participant_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::ParticipantJoined {
                    participant_principal_id: existing_participant,
                    role: existing_role,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_participant == participant_principal_id
                    && *existing_role == role
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_active()?;
        if role == HuddleParticipantRole::Owner
            || participant_principal_id == self.owner_principal_id
        {
            return Err(HuddleError::InvalidRole);
        }
        if let Some(existing) = self.participants.get(&participant_principal_id) {
            if existing.role != role {
                return Err(HuddleError::ParticipantConflict);
            }
            if existing.presence == HuddleParticipantPresence::Present {
                return Ok(HuddleCommandOutcome::Unchanged);
            }
        } else if self.participants.len() >= MAX_HUDDLE_PARTICIPANTS {
            return Err(HuddleError::ParticipantLimitReached);
        }
        let event = HuddleEvent::ParticipantJoined {
            participant_principal_id,
            role,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        };
        self.append_new(event)?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn leave(
        &mut self,
        participant_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(participant_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::ParticipantLeft {
                    participant_principal_id: existing_participant,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_participant == participant_principal_id
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        let Some(participant) = self.participants.get(&participant_principal_id) else {
            return Err(HuddleError::ParticipantNotFound);
        };
        if participant.presence == HuddleParticipantPresence::Left {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_active()?;
        let ended_empty = self
            .participants
            .values()
            .filter(|participant| participant.presence == HuddleParticipantPresence::Present)
            .count()
            == 1;
        let event = HuddleEvent::ParticipantLeft {
            participant_principal_id,
            operation_id,
            occurred_at_millis,
            ended_empty,
            resulting_version: self.next_version()?,
        };
        self.append_new(event)?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn change_role(
        &mut self,
        actor_principal_id: PrincipalId,
        participant_principal_id: PrincipalId,
        role: HuddleParticipantRole,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(actor_principal_id)?;
        validate_principal(participant_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::ParticipantRoleChanged {
                    actor_principal_id: existing_actor,
                    participant_principal_id: existing_participant,
                    role: existing_role,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_actor == actor_principal_id
                    && *existing_participant == participant_principal_id
                    && *existing_role == role
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_active()?;
        self.require_owner(actor_principal_id)?;
        if participant_principal_id == self.owner_principal_id
            || role == HuddleParticipantRole::Owner
        {
            return Err(HuddleError::InvalidRole);
        }
        let participant = self
            .participants
            .get(&participant_principal_id)
            .ok_or(HuddleError::ParticipantNotFound)?;
        if participant.presence != HuddleParticipantPresence::Present {
            return Err(HuddleError::ParticipantNotPresent);
        }
        if participant.role == role {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.append_new(HuddleEvent::ParticipantRoleChanged {
            actor_principal_id,
            participant_principal_id,
            role,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        })?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn set_moderation(
        &mut self,
        actor_principal_id: PrincipalId,
        participant_principal_id: PrincipalId,
        state: HuddleModerationState,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(actor_principal_id)?;
        validate_principal(participant_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::ModerationChanged {
                    actor_principal_id: existing_actor,
                    participant_principal_id: existing_participant,
                    state: existing_state,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_actor == actor_principal_id
                    && *existing_participant == participant_principal_id
                    && *existing_state == state
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_active()?;
        self.require_moderator(actor_principal_id)?;
        if participant_principal_id == self.owner_principal_id {
            return Err(HuddleError::OwnerCannotBeModerated);
        }
        let participant = self
            .participants
            .get(&participant_principal_id)
            .ok_or(HuddleError::ParticipantNotFound)?;
        if participant.presence != HuddleParticipantPresence::Present {
            return Err(HuddleError::ParticipantNotPresent);
        }
        if participant.moderation == state {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.append_new(HuddleEvent::ModerationChanged {
            actor_principal_id,
            participant_principal_id,
            state,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        })?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn react(
        &mut self,
        participant_principal_id: PrincipalId,
        value: HuddleReactionValue,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(participant_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::ReactionAdded {
                    participant_principal_id: existing_participant,
                    value: existing_value,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_participant == participant_principal_id
                    && *existing_value == value
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_active()?;
        self.require_present(participant_principal_id)?;
        if self.reactions.len() >= MAX_HUDDLE_REACTIONS {
            return Err(HuddleError::ReactionLimitReached);
        }
        self.append_new(HuddleEvent::ReactionAdded {
            participant_principal_id,
            value,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        })?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn link_transcript(
        &mut self,
        reference: HuddleTranscriptReference,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::TranscriptLinked {
                    reference: existing_reference,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_reference == reference && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.validate_transcript_reference(reference, occurred_at_millis)?;
        if let Some(existing) = self.transcripts.get(&reference.segment_id()) {
            return if *existing == reference {
                Ok(HuddleCommandOutcome::Unchanged)
            } else {
                Err(HuddleError::TranscriptConflict)
            };
        }
        if self.transcripts.len() >= MAX_HUDDLE_TRANSCRIPT_REFERENCES {
            return Err(HuddleError::TranscriptLimitReached);
        }
        self.append_new(HuddleEvent::TranscriptLinked {
            reference,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        })?;
        Ok(HuddleCommandOutcome::Applied)
    }

    pub fn end(
        &mut self,
        actor_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<HuddleCommandOutcome, HuddleError> {
        validate_principal(actor_principal_id)?;
        validate_operation_time(operation_id, occurred_at_millis)?;
        if self.operation_matches(operation_id, |event| {
            matches!(
                event,
                HuddleEvent::Ended {
                    actor_principal_id: existing_actor,
                    occurred_at_millis: existing_time,
                    ..
                } if *existing_actor == actor_principal_id
                    && *existing_time == occurred_at_millis
            )
        })? {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        if matches!(self.lifecycle, HuddleLifecycleState::Ended { .. }) {
            return Ok(HuddleCommandOutcome::Unchanged);
        }
        self.require_moderator(actor_principal_id)?;
        self.append_new(HuddleEvent::Ended {
            actor_principal_id,
            operation_id,
            occurred_at_millis,
            resulting_version: self.next_version()?,
        })?;
        Ok(HuddleCommandOutcome::Applied)
    }

    fn append_new(&mut self, event: HuddleEvent) -> Result<(), HuddleError> {
        if self.fields.events.len() >= MAX_HUDDLE_EVENTS {
            return Err(HuddleError::EventLimitReached);
        }
        self.append_replayed(event)
    }

    fn append_replayed(&mut self, event: HuddleEvent) -> Result<(), HuddleError> {
        validate_operation_time(event.operation_id(), event.occurred_at_millis())?;
        if event.resulting_version() != self.next_version()?
            || event.occurred_at_millis() < self.last_event_time()
            || self.operations.contains(&event.operation_id())
        {
            return Err(HuddleError::InvalidRecord);
        }
        self.apply_transition(&event)?;
        if !self.operations.insert(event.operation_id()) {
            return Err(HuddleError::InvalidRecord);
        }
        self.fields.version = event.resulting_version();
        self.fields.events.push(event);
        Ok(())
    }

    fn apply_transition(&mut self, event: &HuddleEvent) -> Result<(), HuddleError> {
        match event {
            HuddleEvent::Started { .. } => Err(HuddleError::InvalidRecord),
            HuddleEvent::ParticipantJoined {
                participant_principal_id,
                role,
                occurred_at_millis,
                ..
            } => {
                self.require_active()?;
                if *role == HuddleParticipantRole::Owner
                    || *participant_principal_id == self.owner_principal_id
                {
                    return Err(HuddleError::InvalidRecord);
                }
                let participant_limit_reached = self.participants.len() >= MAX_HUDDLE_PARTICIPANTS;
                match self.participants.get_mut(participant_principal_id) {
                    Some(participant)
                        if participant.presence == HuddleParticipantPresence::Left
                            && participant.role == *role =>
                    {
                        participant.presence = HuddleParticipantPresence::Present;
                        participant.moderation = HuddleModerationState::Unrestricted;
                        participant.joined_at_millis = *occurred_at_millis;
                        participant.left_at_millis = None;
                    }
                    None if !participant_limit_reached => {
                        self.participants.insert(
                            *participant_principal_id,
                            HuddleParticipant {
                                principal_id: *participant_principal_id,
                                role: *role,
                                presence: HuddleParticipantPresence::Present,
                                moderation: HuddleModerationState::Unrestricted,
                                joined_at_millis: *occurred_at_millis,
                                left_at_millis: None,
                            },
                        );
                    }
                    _ => return Err(HuddleError::InvalidRecord),
                }
                Ok(())
            }
            HuddleEvent::ParticipantLeft {
                participant_principal_id,
                occurred_at_millis,
                ended_empty,
                ..
            } => {
                self.require_active()?;
                let present_count = self.present_count();
                let participant = self
                    .participants
                    .get_mut(participant_principal_id)
                    .ok_or(HuddleError::InvalidRecord)?;
                if participant.presence != HuddleParticipantPresence::Present
                    || *ended_empty != (present_count == 1)
                {
                    return Err(HuddleError::InvalidRecord);
                }
                participant.presence = HuddleParticipantPresence::Left;
                participant.moderation = HuddleModerationState::Unrestricted;
                participant.left_at_millis = Some(*occurred_at_millis);
                if *ended_empty {
                    self.lifecycle = HuddleLifecycleState::Ended {
                        reason: HuddleEndReason::Empty,
                        ended_at_millis: *occurred_at_millis,
                        ended_by_principal_id: None,
                    };
                }
                Ok(())
            }
            HuddleEvent::ParticipantRoleChanged {
                actor_principal_id,
                participant_principal_id,
                role,
                ..
            } => {
                self.require_active()?;
                self.require_owner(*actor_principal_id)?;
                if *participant_principal_id == self.owner_principal_id
                    || *role == HuddleParticipantRole::Owner
                {
                    return Err(HuddleError::InvalidRecord);
                }
                let participant = self
                    .participants
                    .get_mut(participant_principal_id)
                    .ok_or(HuddleError::InvalidRecord)?;
                if participant.presence != HuddleParticipantPresence::Present
                    || participant.role == *role
                {
                    return Err(HuddleError::InvalidRecord);
                }
                participant.role = *role;
                Ok(())
            }
            HuddleEvent::ModerationChanged {
                actor_principal_id,
                participant_principal_id,
                state,
                ..
            } => {
                self.require_active()?;
                self.require_moderator(*actor_principal_id)?;
                if *participant_principal_id == self.owner_principal_id {
                    return Err(HuddleError::InvalidRecord);
                }
                let participant = self
                    .participants
                    .get_mut(participant_principal_id)
                    .ok_or(HuddleError::InvalidRecord)?;
                if participant.presence != HuddleParticipantPresence::Present
                    || participant.moderation == *state
                {
                    return Err(HuddleError::InvalidRecord);
                }
                participant.moderation = *state;
                Ok(())
            }
            HuddleEvent::ReactionAdded {
                participant_principal_id,
                value,
                operation_id,
                occurred_at_millis,
                ..
            } => {
                self.require_active()?;
                self.require_present(*participant_principal_id)?;
                HuddleReactionValue::new(value.as_str())?;
                if self.reactions.len() >= MAX_HUDDLE_REACTIONS {
                    return Err(HuddleError::InvalidRecord);
                }
                self.reactions.push(HuddleReaction {
                    participant_principal_id: *participant_principal_id,
                    value: value.clone(),
                    occurred_at_millis: *occurred_at_millis,
                    operation_id: *operation_id,
                });
                Ok(())
            }
            HuddleEvent::TranscriptLinked {
                reference,
                occurred_at_millis,
                ..
            } => {
                self.validate_transcript_reference(*reference, *occurred_at_millis)?;
                if self.transcripts.len() >= MAX_HUDDLE_TRANSCRIPT_REFERENCES
                    || self.transcripts.contains_key(&reference.segment_id())
                {
                    return Err(HuddleError::InvalidRecord);
                }
                self.transcripts.insert(reference.segment_id(), *reference);
                Ok(())
            }
            HuddleEvent::Ended {
                actor_principal_id,
                occurred_at_millis,
                ..
            } => {
                self.require_active()?;
                self.require_moderator(*actor_principal_id)?;
                for participant in self.participants.values_mut() {
                    if participant.presence == HuddleParticipantPresence::Present {
                        participant.presence = HuddleParticipantPresence::Left;
                        participant.moderation = HuddleModerationState::Unrestricted;
                        participant.left_at_millis = Some(*occurred_at_millis);
                    }
                }
                self.lifecycle = HuddleLifecycleState::Ended {
                    reason: HuddleEndReason::Explicit,
                    ended_at_millis: *occurred_at_millis,
                    ended_by_principal_id: Some(*actor_principal_id),
                };
                Ok(())
            }
        }
    }

    fn validate_transcript_reference(
        &self,
        reference: HuddleTranscriptReference,
        linked_at_millis: u64,
    ) -> Result<(), HuddleError> {
        if reference.identity() != self.identity()
            || reference.started_at_millis() < self.started_at_millis
            || reference.ended_at_millis() > linked_at_millis
            || !self
                .participants
                .contains_key(&reference.participant_principal_id())
        {
            return Err(HuddleError::InvalidTranscriptReference);
        }
        if let HuddleLifecycleState::Ended {
            ended_at_millis, ..
        } = self.lifecycle
        {
            if reference.ended_at_millis() > ended_at_millis {
                return Err(HuddleError::InvalidTranscriptReference);
            }
        }
        Ok(())
    }

    fn operation_matches(
        &self,
        operation_id: OperationId,
        matches: impl FnOnce(&HuddleEvent) -> bool,
    ) -> Result<bool, HuddleError> {
        let Some(event) = self
            .fields
            .events
            .iter()
            .find(|event| event.operation_id() == operation_id)
        else {
            return Ok(false);
        };
        if matches(event) {
            Ok(true)
        } else {
            Err(HuddleError::OperationConflict)
        }
    }

    fn require_active(&self) -> Result<(), HuddleError> {
        if self.lifecycle == HuddleLifecycleState::Active {
            Ok(())
        } else {
            Err(HuddleError::Ended)
        }
    }

    fn require_owner(&self, actor_principal_id: PrincipalId) -> Result<(), HuddleError> {
        if actor_principal_id == self.owner_principal_id {
            Ok(())
        } else {
            Err(HuddleError::Unauthorized)
        }
    }

    fn require_moderator(&self, actor_principal_id: PrincipalId) -> Result<(), HuddleError> {
        if actor_principal_id == self.owner_principal_id {
            return Ok(());
        }
        let participant = self
            .participants
            .get(&actor_principal_id)
            .ok_or(HuddleError::Unauthorized)?;
        if participant.presence == HuddleParticipantPresence::Present
            && participant.role == HuddleParticipantRole::Moderator
        {
            Ok(())
        } else {
            Err(HuddleError::Unauthorized)
        }
    }

    fn require_present(&self, principal_id: PrincipalId) -> Result<(), HuddleError> {
        if self
            .participants
            .get(&principal_id)
            .is_some_and(|participant| participant.presence == HuddleParticipantPresence::Present)
        {
            Ok(())
        } else {
            Err(HuddleError::ParticipantNotPresent)
        }
    }

    fn present_count(&self) -> usize {
        self.participants
            .values()
            .filter(|participant| participant.presence == HuddleParticipantPresence::Present)
            .count()
    }

    fn next_version(&self) -> Result<AggregateVersion, HuddleError> {
        self.fields
            .version
            .next()
            .ok_or(HuddleError::VersionExhausted)
    }

    fn last_event_time(&self) -> u64 {
        self.fields
            .events
            .last()
            .map(HuddleEvent::occurred_at_millis)
            .unwrap_or(self.started_at_millis)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleError {
    InvalidIdentity,
    InvalidPrincipal,
    InvalidOperation,
    InvalidTimestamp,
    InvalidRole,
    InvalidReaction,
    InvalidTranscriptReference,
    InvalidRecord,
    Ended,
    Unauthorized,
    ParticipantNotFound,
    ParticipantNotPresent,
    ParticipantConflict,
    ParticipantLimitReached,
    OwnerCannotBeModerated,
    ReactionLimitReached,
    TranscriptConflict,
    TranscriptLimitReached,
    EventLimitReached,
    OperationConflict,
    VersionExhausted,
}

impl fmt::Display for HuddleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidIdentity => "huddle identity is invalid",
            Self::InvalidPrincipal => "huddle principal is invalid",
            Self::InvalidOperation => "huddle operation is invalid",
            Self::InvalidTimestamp => "huddle timestamp is invalid",
            Self::InvalidRole => "huddle participant role is invalid",
            Self::InvalidReaction => "huddle reaction is invalid",
            Self::InvalidTranscriptReference => "huddle transcript reference is invalid",
            Self::InvalidRecord => "huddle record is invalid",
            Self::Ended => "huddle generation has ended",
            Self::Unauthorized => "huddle action is not authorized",
            Self::ParticipantNotFound => "huddle participant is unknown",
            Self::ParticipantNotPresent => "huddle participant is not present",
            Self::ParticipantConflict => "huddle participant state conflicts",
            Self::ParticipantLimitReached => "huddle participant limit is reached",
            Self::OwnerCannotBeModerated => "huddle owner cannot be moderated",
            Self::ReactionLimitReached => "huddle reaction limit is reached",
            Self::TranscriptConflict => "huddle transcript reference conflicts",
            Self::TranscriptLimitReached => "huddle transcript reference limit is reached",
            Self::EventLimitReached => "huddle event limit is reached",
            Self::OperationConflict => "huddle operation was reused with different input",
            Self::VersionExhausted => "huddle version is exhausted",
        })
    }
}

impl Error for HuddleError {}

fn validate_principal(principal_id: PrincipalId) -> Result<(), HuddleError> {
    if principal_id.as_uuid().is_nil() {
        Err(HuddleError::InvalidPrincipal)
    } else {
        Ok(())
    }
}

fn validate_operation_time(
    operation_id: OperationId,
    occurred_at_millis: u64,
) -> Result<(), HuddleError> {
    if operation_id.as_uuid().is_nil() {
        return Err(HuddleError::InvalidOperation);
    }
    if occurred_at_millis == 0 {
        return Err(HuddleError::InvalidTimestamp);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn operation(value: u128) -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(value))
    }

    fn identity(generation: u64) -> HuddleIdentity {
        HuddleIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            aggregate(2),
            aggregate(3),
            HuddleGeneration::new(generation).expect("generation"),
        )
        .expect("identity")
    }

    fn huddle() -> Huddle {
        Huddle::start(identity(1), principal(10), operation(100), 1_000).expect("huddle")
    }

    #[test]
    fn duplicate_join_leave_and_rejoin_are_idempotent() {
        let mut huddle = huddle();
        let participant = principal(11);
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(101),
                1_100,
            ),
            Ok(HuddleCommandOutcome::Applied)
        );
        let joined_version = huddle.fields().version;
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(102),
                1_101,
            ),
            Ok(HuddleCommandOutcome::Unchanged)
        );
        assert_eq!(huddle.fields().version, joined_version);
        assert_eq!(
            huddle.leave(participant, operation(103), 1_200),
            Ok(HuddleCommandOutcome::Applied)
        );
        let left_version = huddle.fields().version;
        assert_eq!(
            huddle.leave(participant, operation(104), 1_201),
            Ok(HuddleCommandOutcome::Unchanged)
        );
        assert_eq!(huddle.fields().version, left_version);
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(105),
                1_300,
            ),
            Ok(HuddleCommandOutcome::Applied)
        );
        assert_eq!(
            huddle
                .participant(participant)
                .map(HuddleParticipant::presence),
            Some(HuddleParticipantPresence::Present)
        );
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(101),
                1_100,
            ),
            Ok(HuddleCommandOutcome::Unchanged)
        );
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Listener,
                operation(101),
                1_100,
            ),
            Err(HuddleError::OperationConflict)
        );
    }

    #[test]
    fn owner_disconnect_preserves_the_roster_until_the_last_participant_leaves() {
        let mut huddle = huddle();
        let owner = huddle.owner_principal_id();
        let participant = principal(11);
        huddle
            .join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(101),
                1_100,
            )
            .expect("participant joins");
        huddle
            .leave(owner, operation(102), 1_200)
            .expect("owner disconnects");
        assert_eq!(huddle.lifecycle(), HuddleLifecycleState::Active);
        assert_eq!(
            huddle.participant(owner).map(HuddleParticipant::presence),
            Some(HuddleParticipantPresence::Left)
        );
        assert_eq!(
            huddle
                .participant(participant)
                .map(HuddleParticipant::presence),
            Some(HuddleParticipantPresence::Present)
        );

        huddle
            .leave(participant, operation(103), 1_300)
            .expect("last participant leaves");
        assert_eq!(
            huddle.lifecycle(),
            HuddleLifecycleState::Ended {
                reason: HuddleEndReason::Empty,
                ended_at_millis: 1_300,
                ended_by_principal_id: None,
            }
        );
        assert_eq!(
            huddle.join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(104),
                1_400,
            ),
            Err(HuddleError::Ended)
        );
    }

    #[test]
    fn explicit_end_is_authorized_terminal_and_idempotent() {
        let mut huddle = huddle();
        let owner = huddle.owner_principal_id();
        let moderator = principal(11);
        let speaker = principal(12);
        huddle
            .join(
                moderator,
                HuddleParticipantRole::Moderator,
                operation(101),
                1_100,
            )
            .expect("moderator joins");
        huddle
            .join(
                speaker,
                HuddleParticipantRole::Speaker,
                operation(102),
                1_200,
            )
            .expect("speaker joins");
        assert_eq!(
            huddle.end(speaker, operation(103), 1_300),
            Err(HuddleError::Unauthorized)
        );
        assert_eq!(
            huddle.end(moderator, operation(104), 1_400),
            Ok(HuddleCommandOutcome::Applied)
        );
        assert_eq!(
            huddle.lifecycle(),
            HuddleLifecycleState::Ended {
                reason: HuddleEndReason::Explicit,
                ended_at_millis: 1_400,
                ended_by_principal_id: Some(moderator),
            }
        );
        assert!(huddle.participants().all(|participant| {
            participant.presence() == HuddleParticipantPresence::Left
                && participant.left_at_millis() == Some(1_400)
        }));
        let ended_version = huddle.fields().version;
        assert_eq!(
            huddle.end(owner, operation(105), 1_500),
            Ok(HuddleCommandOutcome::Unchanged)
        );
        assert_eq!(huddle.fields().version, ended_version);
    }

    #[test]
    fn reactions_roles_and_moderation_share_the_canonical_roster() {
        let mut huddle = huddle();
        let owner = huddle.owner_principal_id();
        let participant = principal(11);
        huddle
            .join(
                participant,
                HuddleParticipantRole::Listener,
                operation(101),
                1_100,
            )
            .expect("listener joins");
        huddle
            .change_role(
                owner,
                participant,
                HuddleParticipantRole::Moderator,
                operation(102),
                1_200,
            )
            .expect("promote moderator");
        huddle
            .set_moderation(
                owner,
                participant,
                HuddleModerationState::Muted,
                operation(103),
                1_300,
            )
            .expect("mute participant");
        huddle
            .react(
                participant,
                HuddleReactionValue::new("🎉").expect("reaction"),
                operation(104),
                1_400,
            )
            .expect("reaction");

        let participant = huddle.participant(participant).expect("participant");
        assert_eq!(participant.role(), HuddleParticipantRole::Moderator);
        assert_eq!(participant.moderation(), HuddleModerationState::Muted);
        assert_eq!(huddle.reactions().len(), 1);
        assert_eq!(huddle.reactions()[0].value.as_str(), "🎉");
        assert_eq!(
            huddle.set_moderation(
                participant.principal_id(),
                owner,
                HuddleModerationState::Muted,
                operation(105),
                1_500,
            ),
            Err(HuddleError::OwnerCannotBeModerated)
        );
    }

    #[test]
    fn transcript_references_are_generation_participant_and_time_bound() {
        let mut huddle = huddle();
        let participant = principal(11);
        huddle
            .join(
                participant,
                HuddleParticipantRole::Speaker,
                operation(101),
                1_100,
            )
            .expect("speaker joins");
        let first = HuddleTranscriptReference::new(
            huddle.identity(),
            HuddleTranscriptSegmentId::new(aggregate(20)).expect("segment"),
            aggregate(21),
            participant,
            1_150,
            1_250,
        )
        .expect("transcript reference");
        assert_eq!(
            huddle.link_transcript(first, operation(102), 1_300),
            Ok(HuddleCommandOutcome::Applied)
        );
        assert_eq!(
            huddle.link_transcript(first, operation(103), 1_301),
            Ok(HuddleCommandOutcome::Unchanged)
        );
        assert_eq!(
            huddle.transcript_references().collect::<Vec<_>>(),
            vec![first]
        );

        huddle
            .end(huddle.owner_principal_id(), operation(104), 1_500)
            .expect("end huddle");
        let final_reference = HuddleTranscriptReference::new(
            huddle.identity(),
            HuddleTranscriptSegmentId::new(aggregate(22)).expect("segment"),
            aggregate(23),
            participant,
            1_300,
            1_450,
        )
        .expect("final reference");
        assert_eq!(
            huddle.link_transcript(final_reference, operation(105), 1_600),
            Ok(HuddleCommandOutcome::Applied)
        );
        let stale_generation = HuddleTranscriptReference::new(
            identity(2),
            HuddleTranscriptSegmentId::new(aggregate(24)).expect("segment"),
            aggregate(25),
            participant,
            1_300,
            1_400,
        )
        .expect("stale generation reference");
        assert_eq!(
            huddle.link_transcript(stale_generation, operation(106), 1_700),
            Err(HuddleError::InvalidTranscriptReference)
        );
        let after_end = HuddleTranscriptReference::new(
            huddle.identity(),
            HuddleTranscriptSegmentId::new(aggregate(26)).expect("segment"),
            aggregate(27),
            participant,
            1_450,
            1_550,
        )
        .expect("late segment shape");
        assert_eq!(
            huddle.link_transcript(after_end, operation(107), 1_700),
            Err(HuddleError::InvalidTranscriptReference)
        );
    }

    #[test]
    fn hydration_rejects_reordered_conflicting_and_cross_generation_history() {
        let mut huddle = huddle();
        huddle
            .join(
                principal(11),
                HuddleParticipantRole::Speaker,
                operation(101),
                1_100,
            )
            .expect("join");
        let record = huddle.fields().clone();
        assert_eq!(Huddle::from_record(record.clone()), Ok(huddle.clone()));

        let mut wrong_version = record.clone();
        wrong_version.version = AggregateVersion::FIRST;
        assert_eq!(
            Huddle::from_record(wrong_version),
            Err(HuddleError::InvalidRecord)
        );

        let mut duplicate_operation = record.clone();
        let HuddleEvent::ParticipantJoined { operation_id, .. } =
            &mut duplicate_operation.events[1]
        else {
            panic!("join event");
        };
        *operation_id = operation(100);
        assert_eq!(
            Huddle::from_record(duplicate_operation),
            Err(HuddleError::InvalidRecord)
        );

        let mut foreign_identity = record;
        foreign_identity.identity = identity(2);
        assert_eq!(
            Huddle::from_record(foreign_identity),
            Err(HuddleError::InvalidRecord)
        );
    }
}
