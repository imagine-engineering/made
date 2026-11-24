use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationRequest, AuthorizationResourceKind, AuthorizationScope,
    CommunityDeletion, CommunityDeletionActiveState, CommunityDeletionAuthorityEvidence,
    CommunityDeletionCompletion, CommunityDeletionFailureReason, CommunityDeletionState,
    CommunityDeletionTransition, MembershipRole, PrincipalId, TenantContext, authorize,
};

use super::executor::{
    COMMUNITY_DELETION_PHASES, CommunityDeletionBackendError, CommunityDeletionExecutionRecord,
    CommunityDeletionExecutorError, CommunityDeletionPhase,
};

const COMMUNITY_MANAGE_SCOPE: &str = "communities:manage";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionOperatorStage {
    Requested,
    Verified,
    Reversible,
    Irreversible,
    Failed,
    Deleted,
    RolledBack,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionOperatorHaltReason {
    AuthorityUnavailable,
    InventoryMismatch,
    DependencyUnavailable,
    FenceLost,
    VerificationFailed,
    ExecutionConflict,
}

impl From<CommunityDeletionFailureReason> for CommunityDeletionOperatorHaltReason {
    fn from(reason: CommunityDeletionFailureReason) -> Self {
        match reason {
            CommunityDeletionFailureReason::AuthorityUnavailable => Self::AuthorityUnavailable,
            CommunityDeletionFailureReason::InventoryMismatch => Self::InventoryMismatch,
            CommunityDeletionFailureReason::DependencyUnavailable => Self::DependencyUnavailable,
            CommunityDeletionFailureReason::FenceLost => Self::FenceLost,
            CommunityDeletionFailureReason::VerificationFailed => Self::VerificationFailed,
            CommunityDeletionFailureReason::ExecutionConflict => Self::ExecutionConflict,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionRecoveryAction {
    None,
    Restore,
    Resume,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityDeletionOperatorStatus {
    stage: CommunityDeletionOperatorStage,
    last_trustworthy_stage: CommunityDeletionOperatorStage,
    completed_phases: u8,
    total_phases: u8,
    next_phase: Option<CommunityDeletionPhase>,
    checkpoint_version: Option<u64>,
    halt_reason: Option<CommunityDeletionOperatorHaltReason>,
    recovery_action: CommunityDeletionRecoveryAction,
}

impl CommunityDeletionOperatorStatus {
    pub const fn stage(self) -> CommunityDeletionOperatorStage {
        self.stage
    }

    pub const fn last_trustworthy_stage(self) -> CommunityDeletionOperatorStage {
        self.last_trustworthy_stage
    }

    pub const fn completed_phases(self) -> u8 {
        self.completed_phases
    }

    pub const fn total_phases(self) -> u8 {
        self.total_phases
    }

    pub const fn next_phase(self) -> Option<CommunityDeletionPhase> {
        self.next_phase
    }

    pub const fn checkpoint_version(self) -> Option<u64> {
        self.checkpoint_version
    }

    pub const fn halt_reason(self) -> Option<CommunityDeletionOperatorHaltReason> {
        self.halt_reason
    }

    pub const fn recovery_action(self) -> CommunityDeletionRecoveryAction {
        self.recovery_action
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionOperatorCommand {
    Status,
    Restore {
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionOperatorOutcome {
    Status(CommunityDeletionOperatorStatus),
    Restored(CommunityDeletionOperatorStatus),
}

#[async_trait]
pub trait CommunityDeletionRecoveryBackend: Send + Sync {
    async fn load_execution(
        &self,
        tenant: &TenantContext,
        deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError>;

    /// Restores the pre-quiesce state, releases an acquired reversible fence and appends the
    /// canonical rollback transition as one durable operation.
    async fn restore_pre_irreversible(
        &self,
        tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
        authority: CommunityDeletionAuthorityEvidence,
    ) -> Result<CommunityDeletion, CommunityDeletionBackendError>;
}

pub struct CommunityDeletionOperatorApi<Backend> {
    backend: Backend,
}

impl<Backend> CommunityDeletionOperatorApi<Backend>
where
    Backend: CommunityDeletionRecoveryBackend,
{
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> Backend {
        self.backend
    }

    pub async fn execute(
        &self,
        authorization: &AuthorizationRequest<'_>,
        deletion_id: AggregateId,
        command: CommunityDeletionOperatorCommand,
    ) -> Result<CommunityDeletionOperatorOutcome, CommunityDeletionRecoveryError> {
        authorize_operator(authorization, command)?;
        if deletion_id.as_uuid().is_nil() {
            return Err(CommunityDeletionRecoveryError::InvalidRequest);
        }
        let execution = self
            .backend
            .load_execution(authorization.tenant, deletion_id)
            .await
            .map_err(CommunityDeletionRecoveryError::from_backend)?;
        execution
            .validate(authorization.tenant, deletion_id)
            .map_err(CommunityDeletionRecoveryError::from_executor)?;

        match command {
            CommunityDeletionOperatorCommand::Status => Ok(
                CommunityDeletionOperatorOutcome::Status(status_from_execution(&execution)?),
            ),
            CommunityDeletionOperatorCommand::Restore {
                expected_version,
                authority,
            } => {
                validate_restore_request(authorization, &execution, expected_version, authority)?;
                let restored = self
                    .backend
                    .restore_pre_irreversible(authorization.tenant, &execution.deletion, authority)
                    .await
                    .map_err(CommunityDeletionRecoveryError::from_backend)?;
                validate_restored(&execution.deletion, authority, &restored)?;
                Ok(CommunityDeletionOperatorOutcome::Restored(
                    status_from_execution(&CommunityDeletionExecutionRecord {
                        deletion: restored,
                        checkpoint: None,
                    })?,
                ))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionRecoveryError {
    AuthorizationDenied,
    InvalidRequest,
    StaleAction,
    RecoveryUnavailable,
    OutcomeUnknown,
    IrreversibleBoundary,
    InvalidBackendResponse,
}

impl CommunityDeletionRecoveryError {
    pub const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::AuthorizationDenied => "community_deletion_operator_denied",
            Self::InvalidRequest => "community_deletion_operator_invalid_request",
            Self::StaleAction => "community_deletion_operator_stale_action",
            Self::RecoveryUnavailable => "community_deletion_operator_unavailable",
            Self::OutcomeUnknown => "community_deletion_operator_outcome_unknown",
            Self::IrreversibleBoundary => "community_deletion_operator_irreversible",
            Self::InvalidBackendResponse => "community_deletion_operator_invalid_response",
        }
    }

    const fn from_backend(error: CommunityDeletionBackendError) -> Self {
        match error {
            CommunityDeletionBackendError::Unavailable => Self::RecoveryUnavailable,
            CommunityDeletionBackendError::StaleCheckpoint => Self::StaleAction,
            CommunityDeletionBackendError::FenceLost => Self::RecoveryUnavailable,
            CommunityDeletionBackendError::InvalidData => Self::InvalidBackendResponse,
            CommunityDeletionBackendError::OutcomeUnknown => Self::OutcomeUnknown,
        }
    }

    const fn from_executor(error: CommunityDeletionExecutorError) -> Self {
        match error {
            CommunityDeletionExecutorError::InvalidInput
            | CommunityDeletionExecutorError::InvalidExecution
            | CommunityDeletionExecutorError::InvalidCheckpoint
            | CommunityDeletionExecutorError::InvalidTransition
            | CommunityDeletionExecutorError::VersionExhausted => Self::InvalidBackendResponse,
            CommunityDeletionExecutorError::Backend(error) => Self::from_backend(error),
        }
    }
}

impl fmt::Display for CommunityDeletionRecoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "community deletion operator request failed ({})",
            self.diagnostic_code()
        )
    }
}

impl Error for CommunityDeletionRecoveryError {}

fn authorize_operator(
    request: &AuthorizationRequest<'_>,
    command: CommunityDeletionOperatorCommand,
) -> Result<(), CommunityDeletionRecoveryError> {
    let scope = AuthorizationScope::new(COMMUNITY_MANAGE_SCOPE)
        .map_err(|_| CommunityDeletionRecoveryError::InvalidRequest)?;
    let action = match command {
        CommunityDeletionOperatorCommand::Status => AuthorizationAction::Read,
        CommunityDeletionOperatorCommand::Restore { .. } => AuthorizationAction::Delete,
    };
    if request.required_scope != &scope
        || request.action != action
        || request.resource.kind != AuthorizationResourceKind::Community
        || request.resource.community_id != request.tenant.community_id()
        || request.resource.resource_id.as_uuid() != request.tenant.community_id().as_uuid()
        || !matches!(authorize(request), AuthorizationDecision::Allowed)
    {
        return Err(CommunityDeletionRecoveryError::AuthorizationDenied);
    }
    let role = request
        .community_membership
        .map(|membership| membership.role)
        .ok_or(CommunityDeletionRecoveryError::AuthorizationDenied)?;
    match (command, role) {
        (
            CommunityDeletionOperatorCommand::Status,
            MembershipRole::Owner | MembershipRole::Admin,
        )
        | (CommunityDeletionOperatorCommand::Restore { .. }, MembershipRole::Owner) => Ok(()),
        _ => Err(CommunityDeletionRecoveryError::AuthorizationDenied),
    }
}

fn validate_restore_request(
    authorization: &AuthorizationRequest<'_>,
    execution: &CommunityDeletionExecutionRecord,
    expected_version: AggregateVersion,
    authority: CommunityDeletionAuthorityEvidence,
) -> Result<(), CommunityDeletionRecoveryError> {
    if execution.checkpoint.is_some()
        || matches!(
            execution.deletion.state(),
            CommunityDeletionState::Irreversible
                | CommunityDeletionState::Failed {
                    failed_from: CommunityDeletionActiveState::Irreversible,
                    ..
                }
                | CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted)
        )
    {
        return Err(CommunityDeletionRecoveryError::IrreversibleBoundary);
    }
    if execution.deletion.fields().version != expected_version {
        return Err(CommunityDeletionRecoveryError::StaleAction);
    }
    if authority.community_archive().community_id != authorization.tenant.community_id()
        || authority.actor_principal_id() != effective_principal_id(authorization)
    {
        return Err(CommunityDeletionRecoveryError::AuthorizationDenied);
    }
    if !matches!(
        execution.deletion.state(),
        CommunityDeletionState::Verified
            | CommunityDeletionState::Reversible
            | CommunityDeletionState::Failed {
                failed_from: CommunityDeletionActiveState::Verified
                    | CommunityDeletionActiveState::Reversible,
                ..
            }
    ) {
        return Err(CommunityDeletionRecoveryError::InvalidRequest);
    }
    Ok(())
}

fn validate_restored(
    expected: &CommunityDeletion,
    authority: CommunityDeletionAuthorityEvidence,
    restored: &CommunityDeletion,
) -> Result<(), CommunityDeletionRecoveryError> {
    let expected_fields = expected.fields();
    let restored_fields = restored.fields();
    if restored.state()
        != CommunityDeletionState::Completed(CommunityDeletionCompletion::RolledBack)
        || restored_fields.community_id != expected_fields.community_id
        || restored_fields.deletion_id != expected_fields.deletion_id
        || restored_fields.version
            != expected_fields
                .version
                .next()
                .ok_or(CommunityDeletionRecoveryError::InvalidBackendResponse)?
        || restored_fields.transitions.len() != expected_fields.transitions.len() + 1
        || !restored_fields
            .transitions
            .starts_with(&expected_fields.transitions)
        || !matches!(
            restored_fields.transitions.last(),
            Some(CommunityDeletionTransition::Completed {
                authority: actual_authority,
                outcome: CommunityDeletionCompletion::RolledBack,
                ..
            }) if *actual_authority == authority
        )
    {
        return Err(CommunityDeletionRecoveryError::InvalidBackendResponse);
    }
    Ok(())
}

fn status_from_execution(
    execution: &CommunityDeletionExecutionRecord,
) -> Result<CommunityDeletionOperatorStatus, CommunityDeletionRecoveryError> {
    let total_phases = u8::try_from(COMMUNITY_DELETION_PHASES.len())
        .map_err(|_| CommunityDeletionRecoveryError::InvalidBackendResponse)?;
    let checkpoint = execution.checkpoint;
    let (stage, last_trustworthy_stage, halt_reason, recovery_action) =
        match execution.deletion.state() {
            CommunityDeletionState::Requested => (
                CommunityDeletionOperatorStage::Requested,
                CommunityDeletionOperatorStage::Requested,
                None,
                CommunityDeletionRecoveryAction::None,
            ),
            CommunityDeletionState::Verified => (
                CommunityDeletionOperatorStage::Verified,
                CommunityDeletionOperatorStage::Verified,
                None,
                CommunityDeletionRecoveryAction::Restore,
            ),
            CommunityDeletionState::Reversible => (
                CommunityDeletionOperatorStage::Reversible,
                CommunityDeletionOperatorStage::Reversible,
                None,
                CommunityDeletionRecoveryAction::Restore,
            ),
            CommunityDeletionState::Irreversible => (
                CommunityDeletionOperatorStage::Irreversible,
                CommunityDeletionOperatorStage::Irreversible,
                None,
                CommunityDeletionRecoveryAction::Resume,
            ),
            CommunityDeletionState::Failed {
                failed_from,
                reason,
            } => {
                let last_trustworthy_stage = stage_from_active(failed_from);
                let recovery_action = match failed_from {
                    CommunityDeletionActiveState::Verified
                    | CommunityDeletionActiveState::Reversible => {
                        CommunityDeletionRecoveryAction::Restore
                    }
                    CommunityDeletionActiveState::Requested
                    | CommunityDeletionActiveState::Irreversible
                        if reason.retryable() =>
                    {
                        CommunityDeletionRecoveryAction::Resume
                    }
                    CommunityDeletionActiveState::Requested
                    | CommunityDeletionActiveState::Irreversible => {
                        CommunityDeletionRecoveryAction::None
                    }
                };
                (
                    CommunityDeletionOperatorStage::Failed,
                    last_trustworthy_stage,
                    Some(reason.into()),
                    recovery_action,
                )
            }
            CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted) => (
                CommunityDeletionOperatorStage::Deleted,
                CommunityDeletionOperatorStage::Deleted,
                None,
                CommunityDeletionRecoveryAction::None,
            ),
            CommunityDeletionState::Completed(CommunityDeletionCompletion::RolledBack) => (
                CommunityDeletionOperatorStage::RolledBack,
                CommunityDeletionOperatorStage::RolledBack,
                None,
                CommunityDeletionRecoveryAction::None,
            ),
        };
    Ok(CommunityDeletionOperatorStatus {
        stage,
        last_trustworthy_stage,
        completed_phases: checkpoint.map_or(0, |checkpoint| checkpoint.completed_phases()),
        total_phases,
        next_phase: checkpoint.and_then(|checkpoint| checkpoint.next_phase()),
        checkpoint_version: checkpoint.map(|checkpoint| checkpoint.checkpoint_version()),
        halt_reason,
        recovery_action,
    })
}

const fn stage_from_active(state: CommunityDeletionActiveState) -> CommunityDeletionOperatorStage {
    match state {
        CommunityDeletionActiveState::Requested => CommunityDeletionOperatorStage::Requested,
        CommunityDeletionActiveState::Verified => CommunityDeletionOperatorStage::Verified,
        CommunityDeletionActiveState::Reversible => CommunityDeletionOperatorStage::Reversible,
        CommunityDeletionActiveState::Irreversible => CommunityDeletionOperatorStage::Irreversible,
    }
}

fn effective_principal_id(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}
