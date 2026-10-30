use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{AggregateVersion, OperationId, ScopedAggregateId, TenantContext};

const MAX_BOUNDARY_LABEL_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CutoverPhase {
    Baseline,
    NativePresentation,
    Foundations,
    CommunicationReadShadow,
    CommunicationWriteCutover,
    ProjectGitAgentIntegration,
    WorkflowInfrastructureCutover,
    ClientDeploymentMigration,
    Retirement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoverAuthority {
    Legacy,
    Canonical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CutoverCursor {
    sequence: u64,
    token_hash: [u8; 32],
}

impl CutoverCursor {
    pub const fn initial() -> Self {
        Self {
            sequence: 0,
            token_hash: [0; 32],
        }
    }

    pub fn new(sequence: u64, token_hash: [u8; 32]) -> Result<Self, CutoverCheckpointError> {
        if sequence == 0 || token_hash == [0; 32] {
            return Err(CutoverCheckpointError::InvalidInput);
        }
        Ok(Self {
            sequence,
            token_hash,
        })
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    pub const fn token_hash(self) -> [u8; 32] {
        self.token_hash
    }

    fn follows(self, previous: Self) -> bool {
        self.sequence > previous.sequence
            || (self.sequence == previous.sequence && self.token_hash == previous.token_hash)
    }

    fn valid(self) -> bool {
        (self.sequence == 0 && self.token_hash == [0; 32])
            || (self.sequence > 0 && self.token_hash != [0; 32])
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CutoverIntegrity {
    source_hash: Option<[u8; 32]>,
    target_hash: Option<[u8; 32]>,
}

impl CutoverIntegrity {
    pub fn new(
        source_hash: Option<[u8; 32]>,
        target_hash: Option<[u8; 32]>,
    ) -> Result<Self, CutoverCheckpointError> {
        let integrity = Self {
            source_hash,
            target_hash,
        };
        if !integrity.valid() {
            return Err(CutoverCheckpointError::InvalidInput);
        }
        Ok(integrity)
    }

    pub const fn source_hash(self) -> Option<[u8; 32]> {
        self.source_hash
    }

    pub const fn target_hash(self) -> Option<[u8; 32]> {
        self.target_hash
    }

    fn verified(self) -> bool {
        matches!((self.source_hash, self.target_hash), (Some(source), Some(target)) if source == target)
    }

    fn follows(self, previous: Self) -> bool {
        (previous.source_hash.is_none() || self.source_hash == previous.source_hash)
            && (previous.target_hash.is_none() || self.target_hash == previous.target_hash)
    }

    fn valid(self) -> bool {
        !self.source_hash.is_some_and(|hash| hash == [0; 32])
            && !self.target_hash.is_some_and(|hash| hash == [0; 32])
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CutoverGateEvidence {
    qualification_report_hash: Option<[u8; 32]>,
    shadow_report_hash: Option<[u8; 32]>,
    rollback_snapshot_hash: Option<[u8; 32]>,
    write_window_approval_hash: Option<[u8; 32]>,
}

impl CutoverGateEvidence {
    pub fn new(
        qualification_report_hash: Option<[u8; 32]>,
        shadow_report_hash: Option<[u8; 32]>,
        rollback_snapshot_hash: Option<[u8; 32]>,
        write_window_approval_hash: Option<[u8; 32]>,
    ) -> Result<Self, CutoverCheckpointError> {
        let evidence = Self {
            qualification_report_hash,
            shadow_report_hash,
            rollback_snapshot_hash,
            write_window_approval_hash,
        };
        if !evidence.valid() {
            return Err(CutoverCheckpointError::InvalidInput);
        }
        Ok(evidence)
    }

    pub const fn qualification_report_hash(self) -> Option<[u8; 32]> {
        self.qualification_report_hash
    }

    pub const fn shadow_report_hash(self) -> Option<[u8; 32]> {
        self.shadow_report_hash
    }

    pub const fn rollback_snapshot_hash(self) -> Option<[u8; 32]> {
        self.rollback_snapshot_hash
    }

    pub const fn write_window_approval_hash(self) -> Option<[u8; 32]> {
        self.write_window_approval_hash
    }

    fn complete(self) -> bool {
        self.qualification_report_hash.is_some()
            && self.shadow_report_hash.is_some()
            && self.rollback_snapshot_hash.is_some()
            && self.write_window_approval_hash.is_some()
    }

    fn follows(self, previous: Self) -> bool {
        (previous.qualification_report_hash.is_none()
            || self.qualification_report_hash == previous.qualification_report_hash)
            && (previous.shadow_report_hash.is_none()
                || self.shadow_report_hash == previous.shadow_report_hash)
            && (previous.rollback_snapshot_hash.is_none()
                || self.rollback_snapshot_hash == previous.rollback_snapshot_hash)
            && (previous.write_window_approval_hash.is_none()
                || self.write_window_approval_hash == previous.write_window_approval_hash)
    }

    fn valid(self) -> bool {
        !self
            .qualification_report_hash
            .is_some_and(|hash| hash == [0; 32])
            && !self.shadow_report_hash.is_some_and(|hash| hash == [0; 32])
            && !self
                .rollback_snapshot_hash
                .is_some_and(|hash| hash == [0; 32])
            && !self
                .write_window_approval_hash
                .is_some_and(|hash| hash == [0; 32])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverReversibleBoundary {
    label: String,
    checkpoint_version: AggregateVersion,
    phase: CutoverPhase,
    authority: CutoverAuthority,
    source_cursor: CutoverCursor,
    target_cursor: CutoverCursor,
    integrity: CutoverIntegrity,
}

impl CutoverReversibleBoundary {
    fn from_checkpoint(
        checkpoint: &CutoverCheckpoint,
        label: String,
    ) -> Result<Self, CutoverCheckpointError> {
        validate_label(&label)?;
        Ok(Self {
            label,
            checkpoint_version: checkpoint.version,
            phase: checkpoint.phase,
            authority: checkpoint.authority,
            source_cursor: checkpoint.source_cursor,
            target_cursor: checkpoint.target_cursor,
            integrity: checkpoint.integrity,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn checkpoint_version(&self) -> AggregateVersion {
        self.checkpoint_version
    }

    pub const fn phase(&self) -> CutoverPhase {
        self.phase
    }

    pub const fn authority(&self) -> CutoverAuthority {
        self.authority
    }

    pub const fn source_cursor(&self) -> CutoverCursor {
        self.source_cursor
    }

    pub const fn target_cursor(&self) -> CutoverCursor {
        self.target_cursor
    }

    pub const fn integrity(&self) -> CutoverIntegrity {
        self.integrity
    }

    fn valid_for(&self, checkpoint: &CutoverCheckpoint) -> bool {
        validate_label(&self.label).is_ok()
            && self.authority == CutoverAuthority::Legacy
            && self.checkpoint_version < checkpoint.version
            && self.phase <= checkpoint.phase
            && self.source_cursor.valid()
            && self.target_cursor.valid()
            && self.integrity.valid()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverTransitionReceipt {
    operation_id: OperationId,
    expected_version: AggregateVersion,
    boundary_label: Option<String>,
}

impl CutoverTransitionReceipt {
    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn expected_version(&self) -> AggregateVersion {
        self.expected_version
    }

    pub fn boundary_label(&self) -> Option<&str> {
        self.boundary_label.as_deref()
    }

    fn valid_for(&self, version: AggregateVersion) -> bool {
        !self.operation_id.as_uuid().is_nil()
            && self.expected_version.next() == Some(version)
            && self
                .boundary_label
                .as_deref()
                .is_none_or(|label| validate_label(label).is_ok())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverCheckpointFields {
    pub aggregate: ScopedAggregateId,
    pub version: AggregateVersion,
    pub phase: CutoverPhase,
    pub authority: CutoverAuthority,
    pub source_cursor: CutoverCursor,
    pub target_cursor: CutoverCursor,
    pub integrity: CutoverIntegrity,
    pub gates: CutoverGateEvidence,
    pub last_reversible_boundary: Option<CutoverReversibleBoundary>,
    pub last_transition: Option<CutoverTransitionReceipt>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverCheckpoint {
    aggregate: ScopedAggregateId,
    version: AggregateVersion,
    phase: CutoverPhase,
    authority: CutoverAuthority,
    source_cursor: CutoverCursor,
    target_cursor: CutoverCursor,
    integrity: CutoverIntegrity,
    gates: CutoverGateEvidence,
    last_reversible_boundary: Option<CutoverReversibleBoundary>,
    last_transition: Option<CutoverTransitionReceipt>,
}

impl CutoverCheckpoint {
    pub fn new(
        aggregate: ScopedAggregateId,
        phase: CutoverPhase,
    ) -> Result<Self, CutoverCheckpointError> {
        let checkpoint = Self {
            aggregate,
            version: AggregateVersion::FIRST,
            phase,
            authority: CutoverAuthority::Legacy,
            source_cursor: CutoverCursor::initial(),
            target_cursor: CutoverCursor::initial(),
            integrity: CutoverIntegrity::default(),
            gates: CutoverGateEvidence::default(),
            last_reversible_boundary: None,
            last_transition: None,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn from_record(fields: CutoverCheckpointFields) -> Result<Self, CutoverCheckpointError> {
        let checkpoint = Self {
            aggregate: fields.aggregate,
            version: fields.version,
            phase: fields.phase,
            authority: fields.authority,
            source_cursor: fields.source_cursor,
            target_cursor: fields.target_cursor,
            integrity: fields.integrity,
            gates: fields.gates,
            last_reversible_boundary: fields.last_reversible_boundary,
            last_transition: fields.last_transition,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn record(&self) -> CutoverCheckpointFields {
        CutoverCheckpointFields {
            aggregate: self.aggregate,
            version: self.version,
            phase: self.phase,
            authority: self.authority,
            source_cursor: self.source_cursor,
            target_cursor: self.target_cursor,
            integrity: self.integrity,
            gates: self.gates,
            last_reversible_boundary: self.last_reversible_boundary.clone(),
            last_transition: self.last_transition.clone(),
        }
    }

    pub fn transition(
        &self,
        transition: &CutoverTransition,
    ) -> Result<CutoverTransitionOutcome, CutoverCheckpointError> {
        transition.validate()?;
        if self.exact_retry(transition) {
            return Ok(CutoverTransitionOutcome::AlreadyApplied(self.clone()));
        }
        if self
            .last_transition
            .as_ref()
            .is_some_and(|receipt| receipt.operation_id == transition.operation_id)
        {
            return Err(CutoverCheckpointError::OperationConflict);
        }
        if transition.expected_version != self.version {
            return Err(CutoverCheckpointError::StaleCheckpoint);
        }
        if transition.phase < self.phase {
            return Err(CutoverCheckpointError::PhaseRegression);
        }
        if !transition.source_cursor.follows(self.source_cursor)
            || !transition.target_cursor.follows(self.target_cursor)
            || !transition.integrity.follows(self.integrity)
            || !transition.gates.follows(self.gates)
        {
            return Err(CutoverCheckpointError::ProgressRegression);
        }
        if self.authority == CutoverAuthority::Canonical
            && transition.authority != CutoverAuthority::Canonical
        {
            return Err(CutoverCheckpointError::AuthorityRegression);
        }

        let authority_advances = self.authority == CutoverAuthority::Legacy
            && transition.authority == CutoverAuthority::Canonical;
        if authority_advances {
            if transition.phase < CutoverPhase::CommunicationWriteCutover
                || transition.source_cursor != transition.target_cursor
                || !transition.integrity.verified()
            {
                return Err(CutoverCheckpointError::IntegrityGateMissing);
            }
            if !transition.gates.complete() {
                return Err(CutoverCheckpointError::AuthorityGateMissing);
            }
            if transition.reversible_boundary_label.is_none() {
                return Err(CutoverCheckpointError::ReversibleBoundaryMissing);
            }
        } else if transition.reversible_boundary_label.is_some() {
            return Err(CutoverCheckpointError::InvalidInput);
        }

        let state_changes = transition.phase != self.phase
            || transition.authority != self.authority
            || transition.source_cursor != self.source_cursor
            || transition.target_cursor != self.target_cursor
            || transition.integrity != self.integrity
            || transition.gates != self.gates;
        if !state_changes {
            return Err(CutoverCheckpointError::NoProgress);
        }

        let version = self
            .version
            .next()
            .ok_or(CutoverCheckpointError::VersionExhausted)?;
        let boundary_label = transition.reversible_boundary_label.clone();
        let last_reversible_boundary = match boundary_label.as_ref() {
            Some(label) => Some(CutoverReversibleBoundary::from_checkpoint(
                self,
                label.clone(),
            )?),
            None => self.last_reversible_boundary.clone(),
        };
        let next = Self {
            aggregate: self.aggregate,
            version,
            phase: transition.phase,
            authority: transition.authority,
            source_cursor: transition.source_cursor,
            target_cursor: transition.target_cursor,
            integrity: transition.integrity,
            gates: transition.gates,
            last_reversible_boundary,
            last_transition: Some(CutoverTransitionReceipt {
                operation_id: transition.operation_id,
                expected_version: transition.expected_version,
                boundary_label,
            }),
        };
        next.validate()?;
        Ok(CutoverTransitionOutcome::Advanced(next))
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn version(&self) -> AggregateVersion {
        self.version
    }

    pub const fn phase(&self) -> CutoverPhase {
        self.phase
    }

    pub const fn authority(&self) -> CutoverAuthority {
        self.authority
    }

    pub const fn source_cursor(&self) -> CutoverCursor {
        self.source_cursor
    }

    pub const fn target_cursor(&self) -> CutoverCursor {
        self.target_cursor
    }

    pub const fn integrity(&self) -> CutoverIntegrity {
        self.integrity
    }

    pub const fn gates(&self) -> CutoverGateEvidence {
        self.gates
    }

    pub fn last_reversible_boundary(&self) -> Option<&CutoverReversibleBoundary> {
        self.last_reversible_boundary.as_ref()
    }

    pub fn last_transition(&self) -> Option<&CutoverTransitionReceipt> {
        self.last_transition.as_ref()
    }

    fn exact_retry(&self, transition: &CutoverTransition) -> bool {
        self.last_transition.as_ref().is_some_and(|receipt| {
            receipt.operation_id == transition.operation_id
                && receipt.expected_version == transition.expected_version
                && receipt.boundary_label == transition.reversible_boundary_label
                && self.phase == transition.phase
                && self.authority == transition.authority
                && self.source_cursor == transition.source_cursor
                && self.target_cursor == transition.target_cursor
                && self.integrity == transition.integrity
                && self.gates == transition.gates
        })
    }

    fn validate(&self) -> Result<(), CutoverCheckpointError> {
        if self.aggregate.community_id().as_uuid().is_nil()
            || self.aggregate.aggregate_id().as_uuid().is_nil()
            || !self.source_cursor.valid()
            || !self.target_cursor.valid()
            || !self.integrity.valid()
            || !self.gates.valid()
            || (self.version == AggregateVersion::FIRST) != self.last_transition.is_none()
            || self
                .last_transition
                .as_ref()
                .is_some_and(|receipt| !receipt.valid_for(self.version))
            || self
                .last_reversible_boundary
                .as_ref()
                .is_some_and(|boundary| !boundary.valid_for(self))
        {
            return Err(CutoverCheckpointError::InvalidRecord);
        }
        if self.authority == CutoverAuthority::Canonical
            && (self.phase < CutoverPhase::CommunicationWriteCutover
                || self.source_cursor != self.target_cursor
                || !self.integrity.verified()
                || !self.gates.complete()
                || self.last_reversible_boundary.is_none())
        {
            return Err(CutoverCheckpointError::InvalidRecord);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverTransition {
    pub operation_id: OperationId,
    pub expected_version: AggregateVersion,
    pub phase: CutoverPhase,
    pub authority: CutoverAuthority,
    pub source_cursor: CutoverCursor,
    pub target_cursor: CutoverCursor,
    pub integrity: CutoverIntegrity,
    pub gates: CutoverGateEvidence,
    pub reversible_boundary_label: Option<String>,
}

impl CutoverTransition {
    fn validate(&self) -> Result<(), CutoverCheckpointError> {
        if self.operation_id.as_uuid().is_nil()
            || self
                .reversible_boundary_label
                .as_deref()
                .is_some_and(|label| validate_label(label).is_err())
            || !self.source_cursor.valid()
            || !self.target_cursor.valid()
            || !self.integrity.valid()
            || !self.gates.valid()
        {
            return Err(CutoverCheckpointError::InvalidInput);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CutoverTransitionOutcome {
    Advanced(CutoverCheckpoint),
    AlreadyApplied(CutoverCheckpoint),
}

impl CutoverTransitionOutcome {
    pub fn checkpoint(&self) -> &CutoverCheckpoint {
        match self {
            Self::Advanced(checkpoint) | Self::AlreadyApplied(checkpoint) => checkpoint,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoverCheckpointCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoverCheckpointStoreError {
    Unavailable,
    StaleCheckpoint,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for CutoverCheckpointStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "cutover checkpoint storage is unavailable",
            Self::StaleCheckpoint => "cutover checkpoint storage rejected a stale version",
            Self::InvalidData => "cutover checkpoint storage returned invalid data",
            Self::OutcomeUnknown => "cutover checkpoint storage outcome is unknown",
        })
    }
}

impl Error for CutoverCheckpointStoreError {}

#[async_trait]
pub trait CutoverCheckpointStore: Send + Sync {
    /// Creates one typed record. `AlreadyCommitted` is valid only for the exact same record.
    async fn create(
        &self,
        tenant: &TenantContext,
        checkpoint: CutoverCheckpoint,
    ) -> Result<CutoverCheckpointCommitOutcome, CutoverCheckpointStoreError>;

    async fn load(
        &self,
        tenant: &TenantContext,
        aggregate: ScopedAggregateId,
    ) -> Result<Option<CutoverCheckpoint>, CutoverCheckpointStoreError>;

    /// Persists `next` only when `expected` is still current. `AlreadyCommitted` is valid only
    /// for the exact same typed record; an unobservable receipt must remain reloadable.
    async fn compare_and_set(
        &self,
        tenant: &TenantContext,
        expected: CutoverCheckpoint,
        next: CutoverCheckpoint,
    ) -> Result<CutoverCheckpointCommitOutcome, CutoverCheckpointStoreError>;
}

pub struct CutoverCheckpointRepository<Store> {
    store: Store,
}

impl<Store> CutoverCheckpointRepository<Store>
where
    Store: CutoverCheckpointStore,
{
    pub const fn new(store: Store) -> Self {
        Self { store }
    }

    pub fn into_store(self) -> Store {
        self.store
    }

    pub async fn create(
        &self,
        tenant: &TenantContext,
        checkpoint: CutoverCheckpoint,
    ) -> Result<CutoverCheckpointCommitOutcome, CutoverCheckpointError> {
        validate_tenant(tenant, checkpoint.aggregate)?;
        checkpoint.validate()?;
        self.store
            .create(tenant, checkpoint)
            .await
            .map_err(CutoverCheckpointError::from_store)
    }

    pub async fn resume(
        &self,
        tenant: &TenantContext,
        aggregate: ScopedAggregateId,
        transition: &CutoverTransition,
    ) -> Result<CutoverTransitionOutcome, CutoverCheckpointError> {
        validate_tenant(tenant, aggregate)?;
        let current = self
            .store
            .load(tenant, aggregate)
            .await
            .map_err(CutoverCheckpointError::from_store)?
            .ok_or(CutoverCheckpointError::NotFound)?;
        validate_tenant(tenant, current.aggregate)?;
        if current.aggregate != aggregate {
            return Err(CutoverCheckpointError::InvalidBackendResponse);
        }
        current.validate()?;

        let outcome = current.transition(transition)?;
        let CutoverTransitionOutcome::Advanced(next) = outcome else {
            return Ok(outcome);
        };
        match self
            .store
            .compare_and_set(tenant, current, next.clone())
            .await
        {
            Ok(CutoverCheckpointCommitOutcome::Committed) => {
                Ok(CutoverTransitionOutcome::Advanced(next))
            }
            Ok(CutoverCheckpointCommitOutcome::AlreadyCommitted) => {
                Ok(CutoverTransitionOutcome::AlreadyApplied(next))
            }
            Err(CutoverCheckpointStoreError::OutcomeUnknown) => {
                let reloaded = self
                    .store
                    .load(tenant, aggregate)
                    .await
                    .map_err(CutoverCheckpointError::from_store)?;
                if reloaded.as_ref() == Some(&next) {
                    Ok(CutoverTransitionOutcome::AlreadyApplied(next))
                } else {
                    Err(CutoverCheckpointError::OutcomeUnknown)
                }
            }
            Err(CutoverCheckpointStoreError::StaleCheckpoint) => {
                let reloaded = self
                    .store
                    .load(tenant, aggregate)
                    .await
                    .map_err(CutoverCheckpointError::from_store)?;
                if reloaded.as_ref() == Some(&next) {
                    Ok(CutoverTransitionOutcome::AlreadyApplied(next))
                } else {
                    Err(CutoverCheckpointError::StaleCheckpoint)
                }
            }
            Err(error) => Err(CutoverCheckpointError::from_store(error)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoverCheckpointError {
    InvalidInput,
    InvalidRecord,
    TenantBoundaryViolation,
    NotFound,
    StaleCheckpoint,
    OperationConflict,
    NoProgress,
    PhaseRegression,
    ProgressRegression,
    AuthorityRegression,
    IntegrityGateMissing,
    AuthorityGateMissing,
    ReversibleBoundaryMissing,
    VersionExhausted,
    StorageUnavailable,
    OutcomeUnknown,
    InvalidBackendResponse,
}

impl CutoverCheckpointError {
    const fn from_store(error: CutoverCheckpointStoreError) -> Self {
        match error {
            CutoverCheckpointStoreError::Unavailable => Self::StorageUnavailable,
            CutoverCheckpointStoreError::StaleCheckpoint => Self::StaleCheckpoint,
            CutoverCheckpointStoreError::InvalidData => Self::InvalidBackendResponse,
            CutoverCheckpointStoreError::OutcomeUnknown => Self::OutcomeUnknown,
        }
    }
}

impl fmt::Display for CutoverCheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "cutover checkpoint input is invalid",
            Self::InvalidRecord => "cutover checkpoint record is invalid",
            Self::TenantBoundaryViolation => "cutover checkpoint crossed its tenant boundary",
            Self::NotFound => "cutover checkpoint does not exist",
            Self::StaleCheckpoint => "cutover checkpoint version is stale",
            Self::OperationConflict => "cutover operation was reused for different input",
            Self::NoProgress => "cutover checkpoint transition makes no progress",
            Self::PhaseRegression => "cutover phase regressed",
            Self::ProgressRegression => "cutover cursor, hash, or gate evidence regressed",
            Self::AuthorityRegression => "cutover authority cannot regress through resume",
            Self::IntegrityGateMissing => "cutover integrity evidence is incomplete or divergent",
            Self::AuthorityGateMissing => "cutover authority gate evidence is incomplete",
            Self::ReversibleBoundaryMissing => "cutover last reversible boundary is missing",
            Self::VersionExhausted => "cutover checkpoint version is exhausted",
            Self::StorageUnavailable => "cutover checkpoint storage is unavailable",
            Self::OutcomeUnknown => "cutover checkpoint outcome remains unknown",
            Self::InvalidBackendResponse => "cutover checkpoint storage returned invalid data",
        })
    }
}

impl Error for CutoverCheckpointError {}

fn validate_tenant(
    tenant: &TenantContext,
    aggregate: ScopedAggregateId,
) -> Result<(), CutoverCheckpointError> {
    if tenant.community_id() != aggregate.community_id() {
        return Err(CutoverCheckpointError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_label(label: &str) -> Result<(), CutoverCheckpointError> {
    if label.is_empty()
        || label.len() > MAX_BOUNDARY_LABEL_BYTES
        || label.trim() != label
        || label.chars().any(char::is_control)
    {
        return Err(CutoverCheckpointError::InvalidInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use collaboration_domain::{AggregateId, AggregateType, CommunityId, TrustedTenantRoute};
    use uuid::Uuid;

    use super::*;

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(community_id: CommunityId) -> ScopedAggregateId {
        ScopedAggregateId::new(
            community_id,
            AggregateType::Conversation,
            AggregateId::from_uuid(Uuid::from_u128(100)),
        )
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "cutover-checkpoint")
                    .expect("trusted tenant route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn cursor(sequence: u64) -> CutoverCursor {
        CutoverCursor::new(sequence, hash(u8::try_from(sequence).unwrap_or(255))).expect("cursor")
    }

    fn complete_gates() -> CutoverGateEvidence {
        CutoverGateEvidence::new(
            Some(hash(10)),
            Some(hash(11)),
            Some(hash(12)),
            Some(hash(13)),
        )
        .expect("gate evidence")
    }

    fn matching_integrity() -> CutoverIntegrity {
        CutoverIntegrity::new(Some(hash(20)), Some(hash(20))).expect("integrity")
    }

    fn canonical_transition(checkpoint: &CutoverCheckpoint, operation: u128) -> CutoverTransition {
        CutoverTransition {
            operation_id: OperationId::from_uuid(Uuid::from_u128(operation)),
            expected_version: checkpoint.version(),
            phase: CutoverPhase::CommunicationWriteCutover,
            authority: CutoverAuthority::Canonical,
            source_cursor: cursor(5),
            target_cursor: cursor(5),
            integrity: matching_integrity(),
            gates: complete_gates(),
            reversible_boundary_label: Some("before-conversation-authority".to_string()),
        }
    }

    #[test]
    fn authority_advancement_requires_hashes_gates_and_reversible_boundary() {
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community(1)),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");

        let mut transition = canonical_transition(&checkpoint, 1);
        transition.integrity = CutoverIntegrity::default();
        assert_eq!(
            checkpoint.transition(&transition),
            Err(CutoverCheckpointError::IntegrityGateMissing)
        );

        transition.integrity = matching_integrity();
        transition.gates = CutoverGateEvidence::default();
        assert_eq!(
            checkpoint.transition(&transition),
            Err(CutoverCheckpointError::AuthorityGateMissing)
        );

        transition.gates = complete_gates();
        transition.reversible_boundary_label = None;
        assert_eq!(
            checkpoint.transition(&transition),
            Err(CutoverCheckpointError::ReversibleBoundaryMissing)
        );
    }

    #[test]
    fn authority_advancement_rejects_cursor_and_hash_divergence() {
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community(1)),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let mut transition = canonical_transition(&checkpoint, 1);
        transition.target_cursor = cursor(4);
        assert_eq!(
            checkpoint.transition(&transition),
            Err(CutoverCheckpointError::IntegrityGateMissing)
        );

        transition.target_cursor = cursor(5);
        transition.integrity =
            CutoverIntegrity::new(Some(hash(20)), Some(hash(21))).expect("integrity");
        assert_eq!(
            checkpoint.transition(&transition),
            Err(CutoverCheckpointError::IntegrityGateMissing)
        );
    }

    #[test]
    fn exact_transition_retry_is_idempotent_and_operation_reuse_conflicts() {
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community(1)),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let transition = canonical_transition(&checkpoint, 1);
        let CutoverTransitionOutcome::Advanced(advanced) = checkpoint
            .transition(&transition)
            .expect("authority transition")
        else {
            panic!("first transition must advance");
        };
        let boundary = advanced
            .last_reversible_boundary()
            .expect("reversible boundary");
        assert_eq!(boundary.authority(), CutoverAuthority::Legacy);
        assert_eq!(boundary.checkpoint_version(), AggregateVersion::FIRST);
        assert_eq!(advanced.authority(), CutoverAuthority::Canonical);
        assert_eq!(
            CutoverCheckpoint::from_record(advanced.record()).expect("stored record"),
            advanced
        );

        assert!(matches!(
            advanced.transition(&transition).expect("exact retry"),
            CutoverTransitionOutcome::AlreadyApplied(checkpoint) if checkpoint == advanced
        ));

        let mut changed = transition;
        changed.target_cursor = cursor(6);
        assert_eq!(
            advanced.transition(&changed),
            Err(CutoverCheckpointError::OperationConflict)
        );
    }

    #[derive(Default)]
    struct TestStore {
        checkpoint: Mutex<Option<CutoverCheckpoint>>,
        commits: Mutex<u64>,
    }

    impl TestStore {
        fn commits(&self) -> u64 {
            *self.commits.lock().expect("commit lock")
        }
    }

    #[async_trait]
    impl CutoverCheckpointStore for TestStore {
        async fn create(
            &self,
            _tenant: &TenantContext,
            checkpoint: CutoverCheckpoint,
        ) -> Result<CutoverCheckpointCommitOutcome, CutoverCheckpointStoreError> {
            let mut stored = self
                .checkpoint
                .lock()
                .map_err(|_| CutoverCheckpointStoreError::Unavailable)?;
            match stored.as_ref() {
                None => {
                    *stored = Some(checkpoint);
                    Ok(CutoverCheckpointCommitOutcome::Committed)
                }
                Some(current) if current == &checkpoint => {
                    Ok(CutoverCheckpointCommitOutcome::AlreadyCommitted)
                }
                Some(_) => Err(CutoverCheckpointStoreError::StaleCheckpoint),
            }
        }

        async fn load(
            &self,
            _tenant: &TenantContext,
            _aggregate: ScopedAggregateId,
        ) -> Result<Option<CutoverCheckpoint>, CutoverCheckpointStoreError> {
            self.checkpoint
                .lock()
                .map(|checkpoint| checkpoint.clone())
                .map_err(|_| CutoverCheckpointStoreError::Unavailable)
        }

        async fn compare_and_set(
            &self,
            _tenant: &TenantContext,
            expected: CutoverCheckpoint,
            next: CutoverCheckpoint,
        ) -> Result<CutoverCheckpointCommitOutcome, CutoverCheckpointStoreError> {
            let mut stored = self
                .checkpoint
                .lock()
                .map_err(|_| CutoverCheckpointStoreError::Unavailable)?;
            if stored.as_ref() == Some(&next) {
                return Ok(CutoverCheckpointCommitOutcome::AlreadyCommitted);
            }
            if stored.as_ref() != Some(&expected) {
                return Err(CutoverCheckpointStoreError::StaleCheckpoint);
            }
            *stored = Some(next);
            let mut commits = self
                .commits
                .lock()
                .map_err(|_| CutoverCheckpointStoreError::Unavailable)?;
            *commits = commits
                .checked_add(1)
                .ok_or(CutoverCheckpointStoreError::InvalidData)?;
            Ok(CutoverCheckpointCommitOutcome::Committed)
        }
    }

    #[tokio::test]
    async fn repository_persists_once_and_resumes_exactly() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let aggregate = aggregate(community_id);
        let checkpoint = CutoverCheckpoint::new(aggregate, CutoverPhase::CommunicationReadShadow)
            .expect("checkpoint");
        let transition = canonical_transition(&checkpoint, 1);
        let repository = CutoverCheckpointRepository::new(TestStore::default());
        assert_eq!(
            repository
                .create(&tenant, checkpoint.clone())
                .await
                .expect("create"),
            CutoverCheckpointCommitOutcome::Committed
        );

        let first = repository
            .resume(&tenant, aggregate, &transition)
            .await
            .expect("resume");
        assert!(matches!(first, CutoverTransitionOutcome::Advanced(_)));
        let second = repository
            .resume(&tenant, aggregate, &transition)
            .await
            .expect("retry");
        assert!(matches!(
            second,
            CutoverTransitionOutcome::AlreadyApplied(_)
        ));
        assert_eq!(repository.store.commits(), 1);
    }

    #[tokio::test]
    async fn repository_rejects_cross_tenant_before_storage() {
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community(1)),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let repository = CutoverCheckpointRepository::new(TestStore::default());
        assert_eq!(
            repository.create(&tenant(community(2)), checkpoint).await,
            Err(CutoverCheckpointError::TenantBoundaryViolation)
        );
        assert_eq!(repository.store.commits(), 0);
    }
}
