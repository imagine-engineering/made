use agent_client_protocol::schema::v1 as acp;
use collections::HashMap;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CollaborationSessionScope {
    Channel {
        channel_id: Uuid,
    },
    Thread {
        channel_id: Uuid,
        thread_id: Uuid,
    },
    Job {
        channel_id: Uuid,
        thread_id: Option<Uuid>,
        job_id: Uuid,
    },
}

impl CollaborationSessionScope {
    fn validate(self) -> Result<(), CollaborationSessionError> {
        let valid = match self {
            Self::Channel { channel_id } => !channel_id.is_nil(),
            Self::Thread {
                channel_id,
                thread_id,
            } => !channel_id.is_nil() && !thread_id.is_nil(),
            Self::Job {
                channel_id,
                thread_id,
                job_id,
            } => {
                !channel_id.is_nil()
                    && thread_id.is_none_or(|thread_id| !thread_id.is_nil())
                    && !job_id.is_nil()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(CollaborationSessionError::InvalidIdentity)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CollaborationSessionIdentity {
    community_id: Uuid,
    scope: CollaborationSessionScope,
}

impl CollaborationSessionIdentity {
    pub fn new(
        community_id: Uuid,
        scope: CollaborationSessionScope,
    ) -> Result<Self, CollaborationSessionError> {
        if community_id.is_nil() {
            return Err(CollaborationSessionError::InvalidIdentity);
        }
        scope.validate()?;
        Ok(Self {
            community_id,
            scope,
        })
    }

    pub const fn community_id(self) -> Uuid {
        self.community_id
    }

    pub const fn scope(self) -> CollaborationSessionScope {
        self.scope
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CollaborationExecutorId(Uuid);

impl CollaborationExecutorId {
    pub fn new(value: Uuid) -> Result<Self, CollaborationSessionError> {
        if value.is_nil() {
            Err(CollaborationSessionError::InvalidExecutor)
        } else {
            Ok(Self(value))
        }
    }

    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationSessionLease {
    identity: CollaborationSessionIdentity,
    executor_id: CollaborationExecutorId,
    generation: u64,
}

impl CollaborationSessionLease {
    pub const fn identity(&self) -> CollaborationSessionIdentity {
        self.identity
    }

    pub const fn executor_id(&self) -> CollaborationExecutorId {
        self.executor_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationSessionResolution {
    Create(CollaborationSessionLease),
    Resume {
        lease: CollaborationSessionLease,
        session_id: acp::SessionId,
    },
}

impl CollaborationSessionResolution {
    pub fn lease(&self) -> &CollaborationSessionLease {
        match self {
            Self::Create(lease) | Self::Resume { lease, .. } => lease,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationCancellationAuthorization {
    lease: CollaborationSessionLease,
    session_id: acp::SessionId,
}

impl CollaborationCancellationAuthorization {
    pub fn session_id(&self) -> &acp::SessionId {
        &self.session_id
    }

    pub fn identity(&self) -> CollaborationSessionIdentity {
        self.lease.identity
    }

    pub fn executor_id(&self) -> CollaborationExecutorId {
        self.lease.executor_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CollaborationSessionState {
    Creating,
    Active(acp::SessionId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CollaborationSessionEntry {
    executor_id: CollaborationExecutorId,
    generation: u64,
    state: CollaborationSessionState,
}

#[derive(Default)]
pub struct CollaborationSessionRegistry {
    entries: HashMap<CollaborationSessionIdentity, CollaborationSessionEntry>,
    session_owners: HashMap<acp::SessionId, CollaborationSessionIdentity>,
    next_generation: u64,
}

impl CollaborationSessionRegistry {
    pub fn resolve(
        &mut self,
        identity: CollaborationSessionIdentity,
        executor_id: CollaborationExecutorId,
    ) -> Result<CollaborationSessionResolution, CollaborationSessionError> {
        if let Some(entry) = self.entries.get(&identity) {
            if entry.executor_id != executor_id {
                return Err(CollaborationSessionError::ExecutorAlreadyClaimed);
            }
            let lease = lease(identity, entry);
            return Ok(match &entry.state {
                CollaborationSessionState::Creating => {
                    CollaborationSessionResolution::Create(lease)
                }
                CollaborationSessionState::Active(session_id) => {
                    CollaborationSessionResolution::Resume {
                        lease,
                        session_id: session_id.clone(),
                    }
                }
            });
        }

        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CollaborationSessionError::GenerationExhausted)?;
        self.next_generation = generation;
        let entry = CollaborationSessionEntry {
            executor_id,
            generation,
            state: CollaborationSessionState::Creating,
        };
        let lease = lease(identity, &entry);
        self.entries.insert(identity, entry);
        Ok(CollaborationSessionResolution::Create(lease))
    }

    pub fn activate(
        &mut self,
        lease: &CollaborationSessionLease,
        session_id: acp::SessionId,
    ) -> Result<(), CollaborationSessionError> {
        if session_id.to_string().trim().is_empty() {
            return Err(CollaborationSessionError::InvalidSessionId);
        }
        let entry = self
            .entries
            .get(&lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        validate_lease(entry, lease)?;
        if self
            .session_owners
            .get(&session_id)
            .is_some_and(|owner| *owner != lease.identity)
        {
            return Err(CollaborationSessionError::SessionAlreadyBound);
        }

        let entry = self
            .entries
            .get_mut(&lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        match &entry.state {
            CollaborationSessionState::Creating => {
                entry.state = CollaborationSessionState::Active(session_id.clone());
                self.session_owners.insert(session_id, lease.identity);
                Ok(())
            }
            CollaborationSessionState::Active(active_session_id)
                if active_session_id == &session_id =>
            {
                Ok(())
            }
            CollaborationSessionState::Active(_) => Err(CollaborationSessionError::SessionConflict),
        }
    }

    pub fn abort_creation(
        &mut self,
        lease: &CollaborationSessionLease,
    ) -> Result<(), CollaborationSessionError> {
        let entry = self
            .entries
            .get(&lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        validate_lease(entry, lease)?;
        if !matches!(entry.state, CollaborationSessionState::Creating) {
            return Err(CollaborationSessionError::SessionAlreadyActive);
        }
        self.entries.remove(&lease.identity);
        Ok(())
    }

    pub fn authorize_cancellation(
        &self,
        lease: &CollaborationSessionLease,
    ) -> Result<CollaborationCancellationAuthorization, CollaborationSessionError> {
        let entry = self
            .entries
            .get(&lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        validate_lease(entry, lease)?;
        let CollaborationSessionState::Active(session_id) = &entry.state else {
            return Err(CollaborationSessionError::SessionNotActive);
        };
        Ok(CollaborationCancellationAuthorization {
            lease: lease.clone(),
            session_id: session_id.clone(),
        })
    }

    pub fn complete_cancellation(
        &mut self,
        authorization: &CollaborationCancellationAuthorization,
    ) -> Result<(), CollaborationSessionError> {
        let entry = self
            .entries
            .get(&authorization.lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        validate_lease(entry, &authorization.lease)?;
        match &entry.state {
            CollaborationSessionState::Active(session_id)
                if session_id == &authorization.session_id => {}
            _ => return Err(CollaborationSessionError::LeaseNotCurrent),
        }
        if self.session_owners.get(&authorization.session_id) != Some(&authorization.lease.identity)
        {
            return Err(CollaborationSessionError::SessionConflict);
        }
        self.entries.remove(&authorization.lease.identity);
        self.session_owners.remove(&authorization.session_id);
        Ok(())
    }

    pub fn active_session(
        &self,
        identity: CollaborationSessionIdentity,
    ) -> Option<&acp::SessionId> {
        match &self.entries.get(&identity)?.state {
            CollaborationSessionState::Creating => None,
            CollaborationSessionState::Active(session_id) => Some(session_id),
        }
    }

    pub fn active_session_for_lease(
        &self,
        lease: &CollaborationSessionLease,
    ) -> Result<&acp::SessionId, CollaborationSessionError> {
        let entry = self
            .entries
            .get(&lease.identity)
            .ok_or(CollaborationSessionError::LeaseNotCurrent)?;
        validate_lease(entry, lease)?;
        match &entry.state {
            CollaborationSessionState::Creating => Err(CollaborationSessionError::SessionNotActive),
            CollaborationSessionState::Active(session_id) => Ok(session_id),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn lease(
    identity: CollaborationSessionIdentity,
    entry: &CollaborationSessionEntry,
) -> CollaborationSessionLease {
    CollaborationSessionLease {
        identity,
        executor_id: entry.executor_id,
        generation: entry.generation,
    }
}

fn validate_lease(
    entry: &CollaborationSessionEntry,
    lease: &CollaborationSessionLease,
) -> Result<(), CollaborationSessionError> {
    if entry.executor_id == lease.executor_id && entry.generation == lease.generation {
        Ok(())
    } else {
        Err(CollaborationSessionError::LeaseNotCurrent)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CollaborationSessionError {
    #[error("collaboration session identity is invalid")]
    InvalidIdentity,
    #[error("collaboration session executor is invalid")]
    InvalidExecutor,
    #[error("collaboration session id is invalid")]
    InvalidSessionId,
    #[error("collaboration session generation is exhausted")]
    GenerationExhausted,
    #[error("collaboration session is already claimed by another executor")]
    ExecutorAlreadyClaimed,
    #[error("native ACP session is already bound to another collaboration identity")]
    SessionAlreadyBound,
    #[error("collaboration identity is already bound to another native ACP session")]
    SessionConflict,
    #[error("collaboration session lease is no longer current")]
    LeaseNotCurrent,
    #[error("collaboration session is already active")]
    SessionAlreadyActive,
    #[error("collaboration session is not active")]
    SessionNotActive,
}
