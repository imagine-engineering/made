use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::{AggregateId, AggregateVersion, CommunityId, OperationId, PrincipalId};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(try_from = "JobIdentityFields")]
pub struct JobIdentity {
    community_id: CommunityId,
    job_id: AggregateId,
}

#[derive(Deserialize)]
struct JobIdentityFields {
    community_id: CommunityId,
    job_id: AggregateId,
}

impl TryFrom<JobIdentityFields> for JobIdentity {
    type Error = JobError;

    fn try_from(fields: JobIdentityFields) -> Result<Self, Self::Error> {
        Self::new(fields.community_id, fields.job_id)
    }
}

impl JobIdentity {
    pub fn new(community_id: CommunityId, job_id: AggregateId) -> Result<Self, JobError> {
        if community_id.as_uuid().is_nil() {
            return Err(JobError::InvalidCommunityId);
        }
        if job_id.as_uuid().is_nil() {
            return Err(JobError::InvalidJobId);
        }
        Ok(Self {
            community_id,
            job_id,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn job_id(self) -> AggregateId {
        self.job_id
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobCommandType {
    Request,
    Accept,
    Progress,
    Result,
    Cancel,
    Error,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "command")]
pub enum JobCommandKind {
    Request {
        requester_principal_id: PrincipalId,
        target_executor_principal_id: PrincipalId,
    },
    Accept {
        executor_principal_id: PrincipalId,
    },
    Progress {
        executor_principal_id: PrincipalId,
    },
    Result {
        executor_principal_id: PrincipalId,
    },
    Cancel {
        actor_principal_id: PrincipalId,
    },
    Error {
        actor_principal_id: PrincipalId,
    },
}

impl JobCommandKind {
    pub const fn command_type(self) -> JobCommandType {
        match self {
            Self::Request { .. } => JobCommandType::Request,
            Self::Accept { .. } => JobCommandType::Accept,
            Self::Progress { .. } => JobCommandType::Progress,
            Self::Result { .. } => JobCommandType::Result,
            Self::Cancel { .. } => JobCommandType::Cancel,
            Self::Error { .. } => JobCommandType::Error,
        }
    }

    const fn principal_id(self) -> Option<PrincipalId> {
        match self {
            Self::Request { .. } => None,
            Self::Accept {
                executor_principal_id,
            }
            | Self::Progress {
                executor_principal_id,
            }
            | Self::Result {
                executor_principal_id,
            } => Some(executor_principal_id),
            Self::Cancel { actor_principal_id } | Self::Error { actor_principal_id } => {
                Some(actor_principal_id)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(try_from = "JobCommandFields")]
pub struct JobCommand {
    identity: JobIdentity,
    operation_id: OperationId,
    version: AggregateVersion,
    occurred_at_millis: u64,
    kind: JobCommandKind,
}

#[derive(Deserialize)]
struct JobCommandFields {
    identity: JobIdentity,
    operation_id: OperationId,
    version: AggregateVersion,
    occurred_at_millis: u64,
    kind: JobCommandKind,
}

impl TryFrom<JobCommandFields> for JobCommand {
    type Error = JobError;

    fn try_from(fields: JobCommandFields) -> Result<Self, Self::Error> {
        Self::new(
            fields.identity,
            fields.operation_id,
            fields.version,
            fields.occurred_at_millis,
            fields.kind,
        )
    }
}

impl JobCommand {
    pub fn new(
        identity: JobIdentity,
        operation_id: OperationId,
        version: AggregateVersion,
        occurred_at_millis: u64,
        kind: JobCommandKind,
    ) -> Result<Self, JobError> {
        let command = Self {
            identity,
            operation_id,
            version,
            occurred_at_millis,
            kind,
        };
        command.validate()?;
        Ok(command)
    }

    pub const fn identity(&self) -> JobIdentity {
        self.identity
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }

    pub const fn occurred_at_millis(&self) -> u64 {
        self.occurred_at_millis
    }

    pub const fn kind(&self) -> JobCommandKind {
        self.kind
    }

    fn validate(&self) -> Result<(), JobError> {
        JobIdentity::new(self.identity.community_id, self.identity.job_id)?;
        if self.operation_id.as_uuid().is_nil() {
            return Err(JobError::InvalidOperationId);
        }
        match self.kind {
            JobCommandKind::Request {
                requester_principal_id,
                target_executor_principal_id,
            } => {
                if requester_principal_id.as_uuid().is_nil() {
                    return Err(JobError::InvalidRequesterId);
                }
                if target_executor_principal_id.as_uuid().is_nil() {
                    return Err(JobError::InvalidTargetExecutorId);
                }
            }
            kind if kind
                .principal_id()
                .is_some_and(|principal_id| principal_id.as_uuid().is_nil()) =>
            {
                return Err(JobError::InvalidActorId);
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStateKind {
    Requested,
    Accepted,
    InProgress,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum JobState {
    Requested,
    Accepted {
        executor_principal_id: PrincipalId,
    },
    InProgress {
        executor_principal_id: PrincipalId,
    },
    Completed {
        executor_principal_id: PrincipalId,
    },
    Cancelled {
        executor_principal_id: Option<PrincipalId>,
        cancelled_by_principal_id: PrincipalId,
    },
    Failed {
        executor_principal_id: Option<PrincipalId>,
        reported_by_principal_id: PrincipalId,
    },
}

impl JobState {
    pub const fn kind(self) -> JobStateKind {
        match self {
            Self::Requested => JobStateKind::Requested,
            Self::Accepted { .. } => JobStateKind::Accepted,
            Self::InProgress { .. } => JobStateKind::InProgress,
            Self::Completed { .. } => JobStateKind::Completed,
            Self::Cancelled { .. } => JobStateKind::Cancelled,
            Self::Failed { .. } => JobStateKind::Failed,
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Cancelled { .. } | Self::Failed { .. }
        )
    }

    pub const fn executor_principal_id(self) -> Option<PrincipalId> {
        match self {
            Self::Requested => None,
            Self::Accepted {
                executor_principal_id,
            }
            | Self::InProgress {
                executor_principal_id,
            }
            | Self::Completed {
                executor_principal_id,
            } => Some(executor_principal_id),
            Self::Cancelled {
                executor_principal_id,
                ..
            }
            | Self::Failed {
                executor_principal_id,
                ..
            } => executor_principal_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Job {
    identity: JobIdentity,
    requester_principal_id: PrincipalId,
    target_executor_principal_id: PrincipalId,
    state: JobState,
    version: AggregateVersion,
    requested_at_millis: u64,
    updated_at_millis: u64,
    history: Vec<JobCommand>,
}

impl Job {
    pub fn request(command: JobCommand) -> Result<Self, JobError> {
        command.validate()?;
        if command.version != AggregateVersion::FIRST {
            return Err(JobError::InitialVersionMustBeFirst);
        }
        let JobCommandKind::Request {
            requester_principal_id,
            target_executor_principal_id,
        } = command.kind
        else {
            return Err(JobError::FirstCommandMustBeRequest);
        };
        Ok(Self {
            identity: command.identity,
            requester_principal_id,
            target_executor_principal_id,
            state: JobState::Requested,
            version: command.version,
            requested_at_millis: command.occurred_at_millis,
            updated_at_millis: command.occurred_at_millis,
            history: vec![command],
        })
    }

    pub fn from_history(history: Vec<JobCommand>) -> Result<Self, JobError> {
        let mut commands = history.into_iter();
        let first = commands.next().ok_or(JobError::EmptyHistory)?;
        let mut job = Self::request(first)?;
        for command in commands {
            if job.apply(command)? == JobCommandOutcome::Unchanged {
                return Err(JobError::DuplicateHistoryEntry);
            }
        }
        Ok(job)
    }

    pub const fn identity(&self) -> JobIdentity {
        self.identity
    }

    pub const fn requester_principal_id(&self) -> PrincipalId {
        self.requester_principal_id
    }

    pub const fn target_executor_principal_id(&self) -> PrincipalId {
        self.target_executor_principal_id
    }

    pub const fn state(&self) -> JobState {
        self.state
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }

    pub const fn requested_at_millis(&self) -> u64 {
        self.requested_at_millis
    }

    pub const fn updated_at_millis(&self) -> u64 {
        self.updated_at_millis
    }

    pub fn history(&self) -> &[JobCommand] {
        &self.history
    }

    pub fn apply(&mut self, command: JobCommand) -> Result<JobCommandOutcome, JobError> {
        command.validate()?;
        if command.identity != self.identity {
            return Err(JobError::IdentityMismatch);
        }
        if let Some(existing) = self
            .history
            .iter()
            .find(|existing| existing.operation_id == command.operation_id)
        {
            return if existing == &command {
                Ok(JobCommandOutcome::Unchanged)
            } else {
                Err(JobError::IdempotencyConflict)
            };
        }
        if let Some(existing) = self
            .history
            .iter()
            .find(|existing| existing.version == command.version)
        {
            return if existing == &command {
                Ok(JobCommandOutcome::Unchanged)
            } else {
                Err(JobError::VersionConflict {
                    current: self.version,
                    supplied: command.version,
                })
            };
        }
        let expected = self.version.next().ok_or(JobError::VersionExhausted)?;
        if command.version != expected {
            return Err(JobError::VersionConflict {
                current: self.version,
                supplied: command.version,
            });
        }
        if command.occurred_at_millis < self.updated_at_millis {
            return Err(JobError::TimestampRegression);
        }
        if self.state.is_terminal() {
            return Err(JobError::TerminalState(self.state.kind()));
        }

        let executor_principal_id = self.state.executor_principal_id();
        let state = match (self.state, command.kind) {
            (
                JobState::Requested,
                JobCommandKind::Accept {
                    executor_principal_id,
                },
            ) => {
                self.require_target_executor(executor_principal_id)?;
                JobState::Accepted {
                    executor_principal_id,
                }
            }
            (
                JobState::Accepted { .. } | JobState::InProgress { .. },
                JobCommandKind::Progress {
                    executor_principal_id,
                },
            ) => {
                self.require_current_executor(executor_principal_id)?;
                JobState::InProgress {
                    executor_principal_id,
                }
            }
            (
                JobState::Accepted { .. } | JobState::InProgress { .. },
                JobCommandKind::Result {
                    executor_principal_id,
                },
            ) => {
                self.require_current_executor(executor_principal_id)?;
                JobState::Completed {
                    executor_principal_id,
                }
            }
            (_, JobCommandKind::Cancel { actor_principal_id }) => JobState::Cancelled {
                executor_principal_id,
                cancelled_by_principal_id: actor_principal_id,
            },
            (_, JobCommandKind::Error { actor_principal_id }) => JobState::Failed {
                executor_principal_id,
                reported_by_principal_id: actor_principal_id,
            },
            (state, kind) => {
                return Err(JobError::InvalidTransition {
                    from: state.kind(),
                    command: kind.command_type(),
                });
            }
        };

        self.state = state;
        self.version = command.version;
        self.updated_at_millis = command.occurred_at_millis;
        self.history.push(command);
        Ok(JobCommandOutcome::Applied)
    }

    fn require_target_executor(&self, executor_principal_id: PrincipalId) -> Result<(), JobError> {
        if executor_principal_id != self.target_executor_principal_id {
            return Err(JobError::ExecutorMismatch);
        }
        Ok(())
    }

    fn require_current_executor(&self, executor_principal_id: PrincipalId) -> Result<(), JobError> {
        if self.state.executor_principal_id() != Some(executor_principal_id) {
            return Err(JobError::ExecutorMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobError {
    EmptyHistory,
    DuplicateHistoryEntry,
    InvalidCommunityId,
    InvalidJobId,
    InvalidOperationId,
    InvalidRequesterId,
    InvalidTargetExecutorId,
    InvalidActorId,
    FirstCommandMustBeRequest,
    InitialVersionMustBeFirst,
    IdentityMismatch,
    IdempotencyConflict,
    VersionConflict {
        current: AggregateVersion,
        supplied: AggregateVersion,
    },
    VersionExhausted,
    TimestampRegression,
    ExecutorMismatch,
    InvalidTransition {
        from: JobStateKind,
        command: JobCommandType,
    },
    TerminalState(JobStateKind),
}

impl fmt::Display for JobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHistory => formatter.write_str("job history is empty"),
            Self::DuplicateHistoryEntry => formatter.write_str("job history contains a duplicate"),
            Self::InvalidCommunityId => formatter.write_str("job community ID is nil"),
            Self::InvalidJobId => formatter.write_str("job ID is nil"),
            Self::InvalidOperationId => formatter.write_str("job operation ID is nil"),
            Self::InvalidRequesterId => formatter.write_str("job requester ID is nil"),
            Self::InvalidTargetExecutorId => formatter.write_str("job target executor ID is nil"),
            Self::InvalidActorId => formatter.write_str("job transition actor ID is nil"),
            Self::FirstCommandMustBeRequest => {
                formatter.write_str("job history must begin with a request")
            }
            Self::InitialVersionMustBeFirst => {
                formatter.write_str("job request must use the first aggregate version")
            }
            Self::IdentityMismatch => formatter.write_str("job command identity mismatch"),
            Self::IdempotencyConflict => {
                formatter.write_str("job operation ID was reused for different command bytes")
            }
            Self::VersionConflict { .. } => formatter.write_str("job command version conflict"),
            Self::VersionExhausted => formatter.write_str("job version is exhausted"),
            Self::TimestampRegression => formatter.write_str("job command timestamp regressed"),
            Self::ExecutorMismatch => {
                formatter.write_str("job executor does not match the accepted target")
            }
            Self::InvalidTransition { .. } => formatter.write_str("job transition is not allowed"),
            Self::TerminalState(_) => formatter.write_str("job is already terminal"),
        }
    }
}

impl Error for JobError {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use uuid::Uuid;

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn identity() -> JobIdentity {
        JobIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            AggregateId::from_uuid(Uuid::from_u128(2)),
        )
        .expect("valid job identity")
    }

    fn command(version: u64, kind: JobCommandKind) -> JobCommand {
        JobCommand::new(
            identity(),
            OperationId::from_uuid(Uuid::from_u128(100 + u128::from(version))),
            AggregateVersion::new(version).expect("positive job version"),
            1_000 + version,
            kind,
        )
        .expect("valid job command")
    }

    fn requested_job() -> Job {
        Job::request(command(
            1,
            JobCommandKind::Request {
                requester_principal_id: principal(3),
                target_executor_principal_id: principal(4),
            },
        ))
        .expect("valid requested job")
    }

    fn advance_to(state: JobStateKind) -> Job {
        let mut job = requested_job();
        if state == JobStateKind::Requested {
            return job;
        }
        job.apply(command(
            2,
            JobCommandKind::Accept {
                executor_principal_id: principal(4),
            },
        ))
        .expect("accept job");
        if state == JobStateKind::Accepted {
            return job;
        }
        if state == JobStateKind::InProgress || state == JobStateKind::Completed {
            job.apply(command(
                3,
                JobCommandKind::Progress {
                    executor_principal_id: principal(4),
                },
            ))
            .expect("progress job");
            if state == JobStateKind::InProgress {
                return job;
            }
        }
        let next_version = job.version().next().expect("job version has successor");
        let kind = match state {
            JobStateKind::Completed => JobCommandKind::Result {
                executor_principal_id: principal(4),
            },
            JobStateKind::Cancelled => JobCommandKind::Cancel {
                actor_principal_id: principal(3),
            },
            JobStateKind::Failed => JobCommandKind::Error {
                actor_principal_id: principal(4),
            },
            JobStateKind::Requested | JobStateKind::Accepted | JobStateKind::InProgress => {
                unreachable!("nonterminal states returned above")
            }
        };
        job.apply(command(next_version.get(), kind))
            .expect("complete job state prefix");
        job
    }

    fn state_from_index(index: u8) -> JobStateKind {
        match index {
            0 => JobStateKind::Requested,
            1 => JobStateKind::Accepted,
            2 => JobStateKind::InProgress,
            3 => JobStateKind::Completed,
            4 => JobStateKind::Cancelled,
            _ => JobStateKind::Failed,
        }
    }

    fn kind_from_index(index: u8) -> JobCommandKind {
        match index {
            0 => JobCommandKind::Request {
                requester_principal_id: principal(3),
                target_executor_principal_id: principal(4),
            },
            1 => JobCommandKind::Accept {
                executor_principal_id: principal(4),
            },
            2 => JobCommandKind::Progress {
                executor_principal_id: principal(4),
            },
            3 => JobCommandKind::Result {
                executor_principal_id: principal(4),
            },
            4 => JobCommandKind::Cancel {
                actor_principal_id: principal(3),
            },
            _ => JobCommandKind::Error {
                actor_principal_id: principal(4),
            },
        }
    }

    fn transition_is_legal(state: JobStateKind, command: JobCommandType) -> bool {
        matches!(
            (state, command),
            (
                JobStateKind::Requested,
                JobCommandType::Accept | JobCommandType::Cancel | JobCommandType::Error
            ) | (
                JobStateKind::Accepted | JobStateKind::InProgress,
                JobCommandType::Progress
                    | JobCommandType::Result
                    | JobCommandType::Cancel
                    | JobCommandType::Error
            )
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn property_job_transition_matrix_is_closed(state_index in 0_u8..6, command_index in 0_u8..6) {
            let state = state_from_index(state_index);
            let kind = kind_from_index(command_index);
            let mut job = advance_to(state);
            let next_version = job.version().next().expect("test version has successor");
            let transition = command(next_version.get(), kind);
            let outcome = job.apply(transition.clone());

            if transition_is_legal(state, kind.command_type()) {
                prop_assert_eq!(outcome, Ok(JobCommandOutcome::Applied));
                prop_assert_eq!(job.apply(transition), Ok(JobCommandOutcome::Unchanged));
            } else if matches!(state, JobStateKind::Completed | JobStateKind::Cancelled | JobStateKind::Failed) {
                prop_assert_eq!(outcome, Err(JobError::TerminalState(state)));
            } else {
                let invalid_transition = matches!(outcome, Err(JobError::InvalidTransition { .. }));
                prop_assert!(invalid_transition);
            }
        }

        #[test]
        fn property_terminal_jobs_reject_duplicate_and_out_of_order_updates(
            terminal_index in 3_u8..6,
            command_index in 1_u8..6,
            version_offset in 1_u64..32,
        ) {
            let state = state_from_index(terminal_index);
            let mut job = advance_to(state);
            let terminal = job.history().last().expect("terminal command").clone();
            let before = job.clone();

            prop_assert_eq!(job.apply(terminal), Ok(JobCommandOutcome::Unchanged));
            prop_assert_eq!(&job, &before);

            let stale_or_future_version = if version_offset % 2 == 0 {
                AggregateVersion::FIRST
            } else {
                AggregateVersion::new(job.version().get() + version_offset + 1)
                    .expect("test version is positive")
            };
            let out_of_order = JobCommand::new(
                identity(),
                OperationId::from_uuid(Uuid::from_u128(10_000 + u128::from(version_offset))),
                stale_or_future_version,
                job.updated_at_millis() + 1,
                kind_from_index(command_index),
            ).expect("valid out-of-order command");
            let version_conflict = matches!(
                job.apply(out_of_order),
                Err(JobError::VersionConflict { .. })
            );
            prop_assert!(version_conflict);

            let next = command(
                job.version().next().expect("terminal version has successor").get(),
                kind_from_index(command_index),
            );
            prop_assert_eq!(job.apply(next), Err(JobError::TerminalState(state)));
            prop_assert_eq!(job, before);
        }
    }

    #[test]
    fn job_history_round_trips_and_preserves_executor_continuity() {
        let mut job = advance_to(JobStateKind::InProgress);
        assert_eq!(
            job.apply(command(
                4,
                JobCommandKind::Progress {
                    executor_principal_id: principal(5),
                },
            )),
            Err(JobError::ExecutorMismatch)
        );
        job.apply(command(
            4,
            JobCommandKind::Result {
                executor_principal_id: principal(4),
            },
        ))
        .expect("complete job");

        let encoded = serde_json::to_string(job.history()).expect("serialize job history");
        let history: Vec<JobCommand> =
            serde_json::from_str(&encoded).expect("deserialize job history");
        assert_eq!(Job::from_history(history), Ok(job));
    }

    #[test]
    fn reused_operation_id_with_different_command_is_rejected() {
        let mut job = advance_to(JobStateKind::Accepted);
        let accepted = job.history()[1].clone();
        let conflicting = JobCommand::new(
            identity(),
            accepted.operation_id(),
            accepted.version(),
            accepted.occurred_at_millis(),
            JobCommandKind::Cancel {
                actor_principal_id: principal(3),
            },
        )
        .expect("valid conflicting command");

        assert_eq!(job.apply(conflicting), Err(JobError::IdempotencyConflict));
    }
}
