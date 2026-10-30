use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    num::NonZeroU64,
};

use collaboration_domain::{
    Huddle, HuddleIdentity, HuddleLifecycleState, HuddleParticipantPresence, PrincipalId,
};

pub const MAX_NATIVE_HUDDLE_PARTICIPANTS: usize = 25;
pub const MAX_NATIVE_HUDDLE_RESOURCES: usize = 1_024;
pub const MAX_NATIVE_HUDDLE_RECONNECT_MILLIS: u64 = 90_000;
const MAX_LIVEKIT_IDENTITY_BYTES: usize = 255;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeHuddleRoomName(String);

impl NativeHuddleRoomName {
    pub fn for_huddle(identity: HuddleIdentity) -> Self {
        Self(format!(
            "zed-huddle/{}/{}/{}/{}",
            identity.community_id().as_uuid().simple(),
            identity.channel_id().as_uuid().simple(),
            identity.huddle_id().as_uuid().simple(),
            identity.generation().get(),
        ))
    }

    pub fn from_livekit(value: impl Into<String>) -> Result<Self, NativeHuddleTransportError> {
        let value = value.into();
        validate_livekit_identity(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeHuddleParticipantIdentity(String);

impl NativeHuddleParticipantIdentity {
    pub fn for_participant(identity: HuddleIdentity, principal_id: PrincipalId) -> Self {
        Self(format!(
            "zed-huddle-participant/{}/{}/{}",
            identity.huddle_id().as_uuid().simple(),
            identity.generation().get(),
            principal_id.as_uuid().simple(),
        ))
    }

    pub fn from_livekit(value: impl Into<String>) -> Result<Self, NativeHuddleTransportError> {
        let value = value.into();
        validate_livekit_identity(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn validate_livekit_identity(value: &str) -> Result<(), NativeHuddleTransportError> {
    if value.is_empty()
        || value.len() > MAX_LIVEKIT_IDENTITY_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(NativeHuddleTransportError::InvalidTransportIdentity);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeHuddleConnectRequest {
    room_name: NativeHuddleRoomName,
    participant_identity: NativeHuddleParticipantIdentity,
    attempt_id: NonZeroU64,
}

impl NativeHuddleConnectRequest {
    pub fn room_name(&self) -> &NativeHuddleRoomName {
        &self.room_name
    }

    pub fn participant_identity(&self) -> &NativeHuddleParticipantIdentity {
        &self.participant_identity
    }

    pub const fn attempt_id(&self) -> NonZeroU64 {
        self.attempt_id
    }

    pub fn callback_scope(&self) -> NativeHuddleCallbackScope {
        NativeHuddleCallbackScope {
            room_name: self.room_name.clone(),
            attempt_id: self.attempt_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeHuddleCallbackScope {
    room_name: NativeHuddleRoomName,
    attempt_id: NonZeroU64,
}

impl NativeHuddleCallbackScope {
    pub fn from_livekit(
        room_name: impl Into<String>,
        attempt_id: u64,
    ) -> Result<Self, NativeHuddleTransportError> {
        Ok(Self {
            room_name: NativeHuddleRoomName::from_livekit(room_name)?,
            attempt_id: NonZeroU64::new(attempt_id)
                .ok_or(NativeHuddleTransportError::InvalidCallback)?,
        })
    }

    pub fn room_name(&self) -> &NativeHuddleRoomName {
        &self.room_name
    }

    pub const fn attempt_id(&self) -> NonZeroU64 {
        self.attempt_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeHuddleConnectionState {
    Idle,
    Connecting {
        attempt_id: NonZeroU64,
    },
    Connected {
        attempt_id: NonZeroU64,
    },
    Reconnecting {
        attempt_id: NonZeroU64,
        deadline_millis: u64,
    },
    Ended,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeHuddleParticipantSync {
    present: Vec<PrincipalId>,
    missing: Vec<PrincipalId>,
}

impl NativeHuddleParticipantSync {
    pub fn present(&self) -> &[PrincipalId] {
        &self.present
    }

    pub fn missing(&self) -> &[PrincipalId] {
        &self.missing
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeHuddleParticipantOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeHuddleResourceId(NonZeroU64);

impl NativeHuddleResourceId {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeHuddleResourceKind {
    Room,
    Capture,
    Playback,
    Track,
    Subscription,
    Timer,
    Credential,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("native huddle resource close failed")]
pub struct NativeHuddleResourceCloseError;

pub trait NativeHuddleResource: 'static {
    fn close(&mut self) -> Result<(), NativeHuddleResourceCloseError>;
}

struct RegisteredResource {
    id: NativeHuddleResourceId,
    kind: NativeHuddleResourceKind,
    resource: Box<dyn NativeHuddleResource>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeHuddleCleanupReport {
    attempted: usize,
    closed: usize,
    failures: Vec<NativeHuddleResourceKind>,
}

impl NativeHuddleCleanupReport {
    pub const fn attempted(&self) -> usize {
        self.attempted
    }

    pub const fn closed(&self) -> usize {
        self.closed
    }

    pub fn failures(&self) -> &[NativeHuddleResourceKind] {
        &self.failures
    }
}

pub struct NativeHuddleTransportAdapter {
    identity: HuddleIdentity,
    local_principal_id: PrincipalId,
    room_name: NativeHuddleRoomName,
    local_participant_identity: NativeHuddleParticipantIdentity,
    state: NativeHuddleConnectionState,
    next_attempt_id: u64,
    observed_remote_participants: BTreeSet<PrincipalId>,
    resources: Vec<RegisteredResource>,
}

impl NativeHuddleTransportAdapter {
    pub fn new(
        huddle: &Huddle,
        local_principal_id: PrincipalId,
    ) -> Result<Self, NativeHuddleTransportError> {
        validate_active_huddle(huddle, local_principal_id)?;
        let identity = huddle.identity();
        Ok(Self {
            identity,
            local_principal_id,
            room_name: NativeHuddleRoomName::for_huddle(identity),
            local_participant_identity: NativeHuddleParticipantIdentity::for_participant(
                identity,
                local_principal_id,
            ),
            state: NativeHuddleConnectionState::Idle,
            next_attempt_id: 1,
            observed_remote_participants: BTreeSet::new(),
            resources: Vec::new(),
        })
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn local_principal_id(&self) -> PrincipalId {
        self.local_principal_id
    }

    pub fn room_name(&self) -> &NativeHuddleRoomName {
        &self.room_name
    }

    pub fn local_participant_identity(&self) -> &NativeHuddleParticipantIdentity {
        &self.local_participant_identity
    }

    pub const fn state(&self) -> NativeHuddleConnectionState {
        self.state
    }

    pub fn begin_connect(
        &mut self,
        huddle: &Huddle,
    ) -> Result<NativeHuddleConnectRequest, NativeHuddleTransportError> {
        self.validate_active_snapshot(huddle)?;
        if !matches!(self.state, NativeHuddleConnectionState::Idle) {
            return Err(NativeHuddleTransportError::InvalidState);
        }
        let attempt_id = NonZeroU64::new(self.next_attempt_id)
            .ok_or(NativeHuddleTransportError::AttemptExhausted)?;
        self.next_attempt_id = self
            .next_attempt_id
            .checked_add(1)
            .ok_or(NativeHuddleTransportError::AttemptExhausted)?;
        self.state = NativeHuddleConnectionState::Connecting { attempt_id };
        Ok(NativeHuddleConnectRequest {
            room_name: self.room_name.clone(),
            participant_identity: self.local_participant_identity.clone(),
            attempt_id,
        })
    }

    pub fn finish_connect(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        huddle: &Huddle,
        observed_remote_identities: impl IntoIterator<Item = NativeHuddleParticipantIdentity>,
    ) -> Result<NativeHuddleParticipantSync, NativeHuddleTransportError> {
        self.validate_active_snapshot(huddle)?;
        self.validate_scope(scope)?;
        match self.state {
            NativeHuddleConnectionState::Connecting { attempt_id }
                if attempt_id == scope.attempt_id => {}
            _ => return Err(NativeHuddleTransportError::InvalidState),
        }
        let sync = self.synchronize(huddle, observed_remote_identities)?;
        self.state = NativeHuddleConnectionState::Connected {
            attempt_id: scope.attempt_id,
        };
        Ok(sync)
    }

    pub fn mark_reconnecting(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        now_millis: u64,
        deadline_millis: u64,
    ) -> Result<(), NativeHuddleTransportError> {
        self.validate_scope(scope)?;
        let reconnect_duration = deadline_millis
            .checked_sub(now_millis)
            .ok_or(NativeHuddleTransportError::InvalidReconnectWindow)?;
        if reconnect_duration == 0 || reconnect_duration > MAX_NATIVE_HUDDLE_RECONNECT_MILLIS {
            return Err(NativeHuddleTransportError::InvalidReconnectWindow);
        }
        match self.state {
            NativeHuddleConnectionState::Connected { attempt_id }
                if attempt_id == scope.attempt_id =>
            {
                self.state = NativeHuddleConnectionState::Reconnecting {
                    attempt_id,
                    deadline_millis,
                };
                Ok(())
            }
            NativeHuddleConnectionState::Reconnecting {
                attempt_id,
                deadline_millis: existing_deadline,
            } if attempt_id == scope.attempt_id && existing_deadline == deadline_millis => Ok(()),
            _ => Err(NativeHuddleTransportError::InvalidState),
        }
    }

    pub fn finish_reconnect(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        huddle: &Huddle,
        observed_remote_identities: impl IntoIterator<Item = NativeHuddleParticipantIdentity>,
        now_millis: u64,
    ) -> Result<NativeHuddleParticipantSync, NativeHuddleTransportError> {
        self.validate_active_snapshot(huddle)?;
        self.validate_scope(scope)?;
        let NativeHuddleConnectionState::Reconnecting {
            attempt_id,
            deadline_millis,
        } = self.state
        else {
            return Err(NativeHuddleTransportError::InvalidState);
        };
        if attempt_id != scope.attempt_id {
            return Err(NativeHuddleTransportError::StaleCallback);
        }
        if now_millis >= deadline_millis {
            return Err(NativeHuddleTransportError::ReconnectExpired);
        }
        let sync = self.synchronize(huddle, observed_remote_identities)?;
        self.state = NativeHuddleConnectionState::Connected { attempt_id };
        Ok(sync)
    }

    pub fn expire_reconnect(&mut self, now_millis: u64) -> Option<NativeHuddleCleanupReport> {
        let NativeHuddleConnectionState::Reconnecting {
            deadline_millis, ..
        } = self.state
        else {
            return None;
        };
        if now_millis < deadline_millis {
            return None;
        }
        let report = self.cleanup_resources();
        self.state = NativeHuddleConnectionState::Idle;
        Some(report)
    }

    pub fn reconcile_participants(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        huddle: &Huddle,
        observed_remote_identities: impl IntoIterator<Item = NativeHuddleParticipantIdentity>,
    ) -> Result<NativeHuddleParticipantSync, NativeHuddleTransportError> {
        self.validate_connected_scope(scope)?;
        self.validate_active_snapshot(huddle)?;
        self.synchronize(huddle, observed_remote_identities)
    }

    pub fn participant_connected(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        huddle: &Huddle,
        participant_identity: &NativeHuddleParticipantIdentity,
    ) -> Result<NativeHuddleParticipantOutcome, NativeHuddleTransportError> {
        self.validate_connected_scope(scope)?;
        self.validate_active_snapshot(huddle)?;
        let expected = self.expected_remote_participants(huddle)?;
        let principal_id = expected
            .get(participant_identity)
            .copied()
            .ok_or(NativeHuddleTransportError::StaleCallback)?;
        if self.observed_remote_participants.insert(principal_id) {
            Ok(NativeHuddleParticipantOutcome::Applied)
        } else {
            Ok(NativeHuddleParticipantOutcome::Unchanged)
        }
    }

    pub fn participant_disconnected(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        huddle: &Huddle,
        participant_identity: &NativeHuddleParticipantIdentity,
    ) -> Result<NativeHuddleParticipantOutcome, NativeHuddleTransportError> {
        self.validate_connected_scope(scope)?;
        self.validate_identity(huddle)?;
        let principal_id = huddle
            .participants()
            .filter(|participant| participant.principal_id() != self.local_principal_id)
            .find_map(|participant| {
                (NativeHuddleParticipantIdentity::for_participant(
                    self.identity,
                    participant.principal_id(),
                ) == *participant_identity)
                    .then_some(participant.principal_id())
            })
            .ok_or(NativeHuddleTransportError::StaleCallback)?;
        if self.observed_remote_participants.remove(&principal_id) {
            Ok(NativeHuddleParticipantOutcome::Applied)
        } else {
            Ok(NativeHuddleParticipantOutcome::Unchanged)
        }
    }

    pub fn register_resource(
        &mut self,
        id: NativeHuddleResourceId,
        kind: NativeHuddleResourceKind,
        resource: impl NativeHuddleResource,
    ) -> Result<(), NativeHuddleTransportError> {
        if !matches!(
            self.state,
            NativeHuddleConnectionState::Connected { .. }
                | NativeHuddleConnectionState::Reconnecting { .. }
        ) {
            return Err(NativeHuddleTransportError::InvalidState);
        }
        if self.resources.len() >= MAX_NATIVE_HUDDLE_RESOURCES {
            return Err(NativeHuddleTransportError::ResourceLimitReached);
        }
        if self.resources.iter().any(|existing| existing.id == id) {
            return Err(NativeHuddleTransportError::DuplicateResource);
        }
        self.resources.push(RegisteredResource {
            id,
            kind,
            resource: Box::new(resource),
        });
        Ok(())
    }

    pub fn cancel(&mut self) -> NativeHuddleCleanupReport {
        let report = self.cleanup_resources();
        if self.state != NativeHuddleConnectionState::Ended {
            self.state = NativeHuddleConnectionState::Idle;
        }
        report
    }

    pub fn end(
        &mut self,
        huddle: &Huddle,
    ) -> Result<NativeHuddleCleanupReport, NativeHuddleTransportError> {
        self.validate_identity(huddle)?;
        if !matches!(huddle.lifecycle(), HuddleLifecycleState::Ended { .. }) {
            return Err(NativeHuddleTransportError::HuddleActive);
        }
        if self.state == NativeHuddleConnectionState::Ended {
            return Ok(NativeHuddleCleanupReport::default());
        }
        let report = self.cleanup_resources();
        self.state = NativeHuddleConnectionState::Ended;
        Ok(report)
    }

    fn validate_identity(&self, huddle: &Huddle) -> Result<(), NativeHuddleTransportError> {
        if huddle.identity() != self.identity {
            return Err(NativeHuddleTransportError::WrongHuddle);
        }
        Ok(())
    }

    fn validate_active_snapshot(&self, huddle: &Huddle) -> Result<(), NativeHuddleTransportError> {
        self.validate_identity(huddle)?;
        validate_active_huddle(huddle, self.local_principal_id)
    }

    fn validate_scope(
        &self,
        scope: &NativeHuddleCallbackScope,
    ) -> Result<(), NativeHuddleTransportError> {
        if scope.room_name != self.room_name {
            return Err(NativeHuddleTransportError::StaleCallback);
        }
        Ok(())
    }

    fn validate_connected_scope(
        &self,
        scope: &NativeHuddleCallbackScope,
    ) -> Result<(), NativeHuddleTransportError> {
        self.validate_scope(scope)?;
        match self.state {
            NativeHuddleConnectionState::Connected { attempt_id }
                if attempt_id == scope.attempt_id =>
            {
                Ok(())
            }
            _ => Err(NativeHuddleTransportError::StaleCallback),
        }
    }

    fn synchronize(
        &mut self,
        huddle: &Huddle,
        observed_remote_identities: impl IntoIterator<Item = NativeHuddleParticipantIdentity>,
    ) -> Result<NativeHuddleParticipantSync, NativeHuddleTransportError> {
        let expected = self.expected_remote_participants(huddle)?;
        let mut observed = BTreeSet::new();
        for identity in observed_remote_identities {
            let principal_id = expected
                .get(&identity)
                .copied()
                .ok_or(NativeHuddleTransportError::StaleCallback)?;
            if !observed.insert(principal_id) {
                return Err(NativeHuddleTransportError::DuplicateParticipant);
            }
        }
        let missing = expected
            .values()
            .copied()
            .filter(|principal_id| !observed.contains(principal_id))
            .collect();
        let present = observed.iter().copied().collect();
        self.observed_remote_participants = observed;
        Ok(NativeHuddleParticipantSync { present, missing })
    }

    fn expected_remote_participants(
        &self,
        huddle: &Huddle,
    ) -> Result<BTreeMap<NativeHuddleParticipantIdentity, PrincipalId>, NativeHuddleTransportError>
    {
        let present: Vec<_> = huddle
            .participants()
            .filter(|participant| participant.presence() == HuddleParticipantPresence::Present)
            .collect();
        if present.len() > MAX_NATIVE_HUDDLE_PARTICIPANTS {
            return Err(NativeHuddleTransportError::ParticipantLimitReached);
        }
        Ok(present
            .into_iter()
            .filter(|participant| participant.principal_id() != self.local_principal_id)
            .map(|participant| {
                (
                    NativeHuddleParticipantIdentity::for_participant(
                        self.identity,
                        participant.principal_id(),
                    ),
                    participant.principal_id(),
                )
            })
            .collect())
    }

    fn cleanup_resources(&mut self) -> NativeHuddleCleanupReport {
        self.observed_remote_participants.clear();
        let mut report = NativeHuddleCleanupReport::default();
        while let Some(mut registered) = self.resources.pop() {
            report.attempted += 1;
            match registered.resource.close() {
                Ok(()) => report.closed += 1,
                Err(_) => report.failures.push(registered.kind),
            }
        }
        report
    }
}

impl Drop for NativeHuddleTransportAdapter {
    fn drop(&mut self) {
        let report = self.cleanup_resources();
        for kind in report.failures {
            log::error!("failed to close native huddle {kind:?} resource while dropping adapter");
        }
    }
}

fn validate_active_huddle(
    huddle: &Huddle,
    local_principal_id: PrincipalId,
) -> Result<(), NativeHuddleTransportError> {
    if !matches!(huddle.lifecycle(), HuddleLifecycleState::Active) {
        return Err(NativeHuddleTransportError::HuddleEnded);
    }
    let Some(local_participant) = huddle.participant(local_principal_id) else {
        return Err(NativeHuddleTransportError::LocalParticipantUnavailable);
    };
    if local_participant.presence() != HuddleParticipantPresence::Present {
        return Err(NativeHuddleTransportError::LocalParticipantUnavailable);
    }
    let present_count = huddle
        .participants()
        .filter(|participant| participant.presence() == HuddleParticipantPresence::Present)
        .count();
    if present_count > MAX_NATIVE_HUDDLE_PARTICIPANTS {
        return Err(NativeHuddleTransportError::ParticipantLimitReached);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeHuddleTransportError {
    InvalidTransportIdentity,
    InvalidCallback,
    WrongHuddle,
    HuddleEnded,
    HuddleActive,
    LocalParticipantUnavailable,
    ParticipantLimitReached,
    DuplicateParticipant,
    InvalidState,
    StaleCallback,
    InvalidReconnectWindow,
    ReconnectExpired,
    AttemptExhausted,
    ResourceLimitReached,
    DuplicateResource,
}

impl fmt::Display for NativeHuddleTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidTransportIdentity => "native huddle transport identity is invalid",
            Self::InvalidCallback => "native huddle callback is invalid",
            Self::WrongHuddle => "native huddle scope does not match",
            Self::HuddleEnded => "native huddle is no longer active",
            Self::HuddleActive => "native huddle is still active",
            Self::LocalParticipantUnavailable => "native huddle participant is unavailable",
            Self::ParticipantLimitReached => "native huddle participant limit reached",
            Self::DuplicateParticipant => "native huddle participant was duplicated",
            Self::InvalidState => "native huddle transport state does not allow this operation",
            Self::StaleCallback => "native huddle callback is stale",
            Self::InvalidReconnectWindow => "native huddle reconnect window is invalid",
            Self::ReconnectExpired => "native huddle reconnect window expired",
            Self::AttemptExhausted => "native huddle connection attempts exhausted",
            Self::ResourceLimitReached => "native huddle resource limit reached",
            Self::DuplicateResource => "native huddle resource was duplicated",
        };
        formatter.write_str(message)
    }
}

impl Error for NativeHuddleTransportError {}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use collaboration_domain::{
        AggregateId, CommunityId, HuddleCommandOutcome, HuddleGeneration, HuddleParticipantRole,
        OperationId,
    };

    use super::*;

    struct CloseProbe {
        closes: Arc<AtomicUsize>,
        fail: bool,
    }

    impl NativeHuddleResource for CloseProbe {
        fn close(&mut self) -> Result<(), NativeHuddleResourceCloseError> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(NativeHuddleResourceCloseError)
            } else {
                Ok(())
            }
        }
    }

    fn operation() -> OperationId {
        OperationId::new()
    }

    fn huddle() -> (Huddle, PrincipalId, PrincipalId, PrincipalId) {
        let owner = PrincipalId::new();
        let speaker = PrincipalId::new();
        let listener = PrincipalId::new();
        let identity = HuddleIdentity::new(
            CommunityId::new(),
            AggregateId::new(),
            AggregateId::new(),
            HuddleGeneration::new(7).expect("generation"),
        )
        .expect("identity");
        let mut huddle = Huddle::start(identity, owner, operation(), 1).expect("start huddle");
        assert_eq!(
            huddle.join(speaker, HuddleParticipantRole::Speaker, operation(), 2,),
            Ok(HuddleCommandOutcome::Applied)
        );
        assert_eq!(
            huddle.join(listener, HuddleParticipantRole::Listener, operation(), 3,),
            Ok(HuddleCommandOutcome::Applied)
        );
        (huddle, owner, speaker, listener)
    }

    fn connect(
        adapter: &mut NativeHuddleTransportAdapter,
        huddle: &Huddle,
        observed: Vec<NativeHuddleParticipantIdentity>,
    ) -> NativeHuddleCallbackScope {
        let request = adapter.begin_connect(huddle).expect("begin connect");
        let scope = request.callback_scope();
        adapter
            .finish_connect(&scope, huddle, observed)
            .expect("finish connect");
        scope
    }

    #[test]
    fn connect_binds_exact_room_generation_and_canonical_roster() {
        let (huddle, owner, speaker, listener) = huddle();
        let mut adapter = NativeHuddleTransportAdapter::new(&huddle, owner).expect("adapter");
        let request = adapter.begin_connect(&huddle).expect("begin connect");

        assert_eq!(request.room_name(), adapter.room_name());
        assert_eq!(
            request.participant_identity(),
            adapter.local_participant_identity()
        );
        let speaker_identity =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), speaker);
        let sync = adapter
            .finish_connect(&request.callback_scope(), &huddle, [speaker_identity])
            .expect("finish connect");

        assert_eq!(sync.present(), &[speaker]);
        assert_eq!(sync.missing(), &[listener]);
        assert!(matches!(
            adapter.state(),
            NativeHuddleConnectionState::Connected { attempt_id }
                if attempt_id == request.attempt_id()
        ));

        let wrong_room = NativeHuddleCallbackScope::from_livekit(
            "zed-huddle/wrong-room",
            request.attempt_id().get(),
        )
        .expect("wrong-room callback shape");
        assert_eq!(
            adapter.reconcile_participants(&wrong_room, &huddle, []),
            Err(NativeHuddleTransportError::StaleCallback)
        );
        let stale_attempt = NativeHuddleCallbackScope::from_livekit(
            request.room_name().as_str(),
            request.attempt_id().get() + 1,
        )
        .expect("stale-attempt callback shape");
        assert_eq!(
            adapter.reconcile_participants(&stale_attempt, &huddle, []),
            Err(NativeHuddleTransportError::StaleCallback)
        );

        let foreign = NativeHuddleParticipantIdentity::for_participant(
            HuddleIdentity::new(
                CommunityId::new(),
                AggregateId::new(),
                AggregateId::new(),
                HuddleGeneration::new(8).expect("generation"),
            )
            .expect("identity"),
            speaker,
        );
        assert_eq!(
            adapter.reconcile_participants(&request.callback_scope(), &huddle, [foreign]),
            Err(NativeHuddleTransportError::StaleCallback)
        );
        assert_eq!(sync.present(), &[speaker]);
    }

    #[test]
    fn reconnect_retains_identity_resynchronizes_and_expires_at_the_deadline() {
        let (huddle, owner, speaker, listener) = huddle();
        let mut adapter = NativeHuddleTransportAdapter::new(&huddle, owner).expect("adapter");
        let speaker_identity =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), speaker);
        let listener_identity =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), listener);
        let scope = connect(&mut adapter, &huddle, vec![speaker_identity.clone()]);
        let closes = Arc::new(AtomicUsize::new(0));
        adapter
            .register_resource(
                NativeHuddleResourceId::new(1).expect("resource id"),
                NativeHuddleResourceKind::Room,
                CloseProbe {
                    closes: closes.clone(),
                    fail: false,
                },
            )
            .expect("register room");

        adapter
            .mark_reconnecting(&scope, 1_000, 91_000)
            .expect("start reconnect");
        let sync = adapter
            .finish_reconnect(
                &scope,
                &huddle,
                [speaker_identity.clone(), listener_identity.clone()],
                90_999,
            )
            .expect("finish reconnect");
        let mut expected_present = vec![speaker, listener];
        expected_present.sort_unstable();
        assert_eq!(sync.present(), expected_present);
        assert!(sync.missing().is_empty());
        adapter
            .mark_reconnecting(&scope, 100_000, 190_000)
            .expect("start second reconnect");
        assert_eq!(
            adapter.finish_reconnect(
                &scope,
                &huddle,
                [speaker_identity, listener_identity],
                190_000,
            ),
            Err(NativeHuddleTransportError::ReconnectExpired)
        );
        let cleanup = adapter
            .expire_reconnect(190_000)
            .expect("expired reconnect cleanup");
        assert_eq!(cleanup.attempted(), 1);
        assert_eq!(cleanup.closed(), 1);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert_eq!(adapter.state(), NativeHuddleConnectionState::Idle);

        let retry = adapter.begin_connect(&huddle).expect("retry connect");
        assert_ne!(retry.attempt_id(), scope.attempt_id());
        assert_eq!(retry.room_name(), scope.room_name());
        assert_eq!(
            retry.participant_identity(),
            adapter.local_participant_identity()
        );
    }

    #[test]
    fn participant_callbacks_are_exact_and_idempotent() {
        let (mut huddle, owner, speaker, listener) = huddle();
        let mut adapter = NativeHuddleTransportAdapter::new(&huddle, owner).expect("adapter");
        let scope = connect(&mut adapter, &huddle, vec![]);
        let speaker_identity =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), speaker);

        assert_eq!(
            adapter.participant_connected(&scope, &huddle, &speaker_identity),
            Ok(NativeHuddleParticipantOutcome::Applied)
        );
        assert_eq!(
            adapter.participant_connected(&scope, &huddle, &speaker_identity),
            Ok(NativeHuddleParticipantOutcome::Unchanged)
        );
        assert_eq!(
            huddle.leave(speaker, operation(), 4),
            Ok(HuddleCommandOutcome::Applied)
        );
        assert_eq!(
            adapter.participant_disconnected(&scope, &huddle, &speaker_identity),
            Ok(NativeHuddleParticipantOutcome::Applied)
        );
        assert_eq!(
            adapter.participant_disconnected(&scope, &huddle, &speaker_identity),
            Ok(NativeHuddleParticipantOutcome::Unchanged)
        );

        let unadmitted =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), PrincipalId::new());
        assert_eq!(
            adapter.participant_connected(&scope, &huddle, &unadmitted),
            Err(NativeHuddleTransportError::StaleCallback)
        );
        assert!(huddle.participant(listener).is_some());
    }

    #[test]
    fn canonical_end_closes_every_resource_and_reports_failures() {
        let (mut huddle, owner, speaker, _) = huddle();
        let mut adapter = NativeHuddleTransportAdapter::new(&huddle, owner).expect("adapter");
        let speaker_identity =
            NativeHuddleParticipantIdentity::for_participant(huddle.identity(), speaker);
        connect(&mut adapter, &huddle, vec![speaker_identity]);
        let closes = Arc::new(AtomicUsize::new(0));
        for (id, kind, fail) in [
            (1, NativeHuddleResourceKind::Room, false),
            (2, NativeHuddleResourceKind::Capture, false),
            (3, NativeHuddleResourceKind::Playback, true),
            (4, NativeHuddleResourceKind::Subscription, false),
            (5, NativeHuddleResourceKind::Credential, false),
        ] {
            adapter
                .register_resource(
                    NativeHuddleResourceId::new(id).expect("resource id"),
                    kind,
                    CloseProbe {
                        closes: closes.clone(),
                        fail,
                    },
                )
                .expect("register resource");
        }

        assert_eq!(
            adapter.end(&huddle),
            Err(NativeHuddleTransportError::HuddleActive)
        );
        assert_eq!(closes.load(Ordering::SeqCst), 0);
        assert_eq!(
            huddle.end(owner, operation(), 10),
            Ok(HuddleCommandOutcome::Applied)
        );
        let cleanup = adapter.end(&huddle).expect("end adapter");
        assert_eq!(cleanup.attempted(), 5);
        assert_eq!(cleanup.closed(), 4);
        assert_eq!(cleanup.failures(), &[NativeHuddleResourceKind::Playback]);
        assert_eq!(closes.load(Ordering::SeqCst), 5);
        assert_eq!(adapter.state(), NativeHuddleConnectionState::Ended);
        assert_eq!(
            adapter.end(&huddle),
            Ok(NativeHuddleCleanupReport::default())
        );
    }

    #[test]
    fn cancel_and_drop_release_owned_resources_once() {
        let (huddle, owner, _, _) = huddle();
        let closes = Arc::new(AtomicUsize::new(0));
        {
            let mut adapter = NativeHuddleTransportAdapter::new(&huddle, owner).expect("adapter");
            connect(&mut adapter, &huddle, vec![]);
            adapter
                .register_resource(
                    NativeHuddleResourceId::new(1).expect("resource id"),
                    NativeHuddleResourceKind::Timer,
                    CloseProbe {
                        closes: closes.clone(),
                        fail: false,
                    },
                )
                .expect("register timer");
            assert_eq!(adapter.cancel().closed(), 1);
            assert_eq!(adapter.cancel().attempted(), 0);

            let request = adapter
                .begin_connect(&huddle)
                .expect("reconnect after cancel");
            adapter
                .finish_connect(&request.callback_scope(), &huddle, [])
                .expect("finish reconnect");
            adapter
                .register_resource(
                    NativeHuddleResourceId::new(2).expect("resource id"),
                    NativeHuddleResourceKind::Track,
                    CloseProbe {
                        closes: closes.clone(),
                        fail: false,
                    },
                )
                .expect("register track");
        }
        assert_eq!(closes.load(Ordering::SeqCst), 2);
    }
}
