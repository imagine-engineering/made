use std::{error::Error, fmt, num::NonZeroU64};

use collaboration_domain::{
    AggregateId, Huddle, HuddleIdentity, HuddleLifecycleState, HuddleParticipantPresence,
    HuddleTranscriptReference, HuddleTranscriptSegmentId, OperationId, PrincipalId,
};

pub const MAX_TRANSCRIPT_TEXT_BYTES: usize = 16 * 1024;
pub const MAX_TRANSCRIPT_TEXT_CHARACTERS: usize = 4_096;
const MAX_TRANSCRIPT_PROVENANCE_BYTES: usize = 128;

#[derive(Clone, Eq, PartialEq)]
pub struct TranscriptText(String);

impl TranscriptText {
    pub fn new(value: impl Into<String>) -> Result<Self, TranscriptProjectionError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.len() > MAX_TRANSCRIPT_TEXT_BYTES
            || value.chars().count() > MAX_TRANSCRIPT_TEXT_CHARACTERS
            || value.chars().any(|character| character == '\0')
        {
            return Err(TranscriptProjectionError::InvalidText);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for TranscriptText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TranscriptText")
            .field("bytes", &self.0.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptSourceProvenance {
    model_id: String,
    model_revision: String,
    source_track_digest: [u8; 32],
    transcription_generation: NonZeroU64,
}

impl TranscriptSourceProvenance {
    pub fn new(
        model_id: impl Into<String>,
        model_revision: impl Into<String>,
        source_track_digest: [u8; 32],
        transcription_generation: NonZeroU64,
    ) -> Result<Self, TranscriptProjectionError> {
        let model_id = model_id.into();
        let model_revision = model_revision.into();
        if !valid_provenance_value(&model_id)
            || !valid_provenance_value(&model_revision)
            || source_track_digest.iter().all(|byte| *byte == 0)
        {
            return Err(TranscriptProjectionError::InvalidProvenance);
        }
        Ok(Self {
            model_id,
            model_revision,
            source_track_digest,
            transcription_generation,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    pub const fn source_track_digest(&self) -> [u8; 32] {
        self.source_track_digest
    }

    pub const fn transcription_generation(&self) -> NonZeroU64 {
        self.transcription_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptSegmentPhase {
    Partial,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptRetention {
    Retain,
    ExpireAt(NonZeroU64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptProjectionAuthorization {
    consent_version: NonZeroU64,
    retention: TranscriptRetention,
}

impl TranscriptProjectionAuthorization {
    pub const fn new(consent_version: NonZeroU64, retention: TranscriptRetention) -> Self {
        Self {
            consent_version,
            retention,
        }
    }

    pub const fn consent_version(self) -> NonZeroU64 {
        self.consent_version
    }

    pub const fn retention(self) -> TranscriptRetention {
        self.retention
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptProjectionAuthorizationRequest {
    huddle_identity: HuddleIdentity,
    participant_principal_id: PrincipalId,
    message_id: AggregateId,
}

impl TranscriptProjectionAuthorizationRequest {
    pub const fn huddle_identity(self) -> HuddleIdentity {
        self.huddle_identity
    }

    pub const fn participant_principal_id(self) -> PrincipalId {
        self.participant_principal_id
    }

    pub const fn message_id(self) -> AggregateId {
        self.message_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptProjectionAuthorizationError {
    Denied,
    Unavailable,
}

pub trait TranscriptProjectionAuthorizer {
    fn authorize(
        &mut self,
        request: TranscriptProjectionAuthorizationRequest,
    ) -> Result<TranscriptProjectionAuthorization, TranscriptProjectionAuthorizationError>;
}

#[derive(Clone, Eq, PartialEq)]
pub struct TranscriptSegmentInput {
    identity: HuddleIdentity,
    segment_id: HuddleTranscriptSegmentId,
    message_id: AggregateId,
    participant_principal_id: PrincipalId,
    source_revision: NonZeroU64,
    phase: TranscriptSegmentPhase,
    text: TranscriptText,
    started_at_millis: u64,
    ended_at_millis: u64,
    observed_at_millis: u64,
    operation_id: OperationId,
    provenance: TranscriptSourceProvenance,
}

impl TranscriptSegmentInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: HuddleIdentity,
        segment_id: HuddleTranscriptSegmentId,
        message_id: AggregateId,
        participant_principal_id: PrincipalId,
        source_revision: NonZeroU64,
        phase: TranscriptSegmentPhase,
        text: TranscriptText,
        started_at_millis: u64,
        ended_at_millis: u64,
        observed_at_millis: u64,
        operation_id: OperationId,
        provenance: TranscriptSourceProvenance,
    ) -> Result<Self, TranscriptProjectionError> {
        if message_id.as_uuid().is_nil()
            || participant_principal_id.as_uuid().is_nil()
            || operation_id.as_uuid().is_nil()
            || started_at_millis == 0
            || ended_at_millis <= started_at_millis
            || observed_at_millis < ended_at_millis
        {
            return Err(TranscriptProjectionError::InvalidSegment);
        }
        Ok(Self {
            identity,
            segment_id,
            message_id,
            participant_principal_id,
            source_revision,
            phase,
            text,
            started_at_millis,
            ended_at_millis,
            observed_at_millis,
            operation_id,
            provenance,
        })
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn segment_id(&self) -> HuddleTranscriptSegmentId {
        self.segment_id
    }

    pub const fn message_id(&self) -> AggregateId {
        self.message_id
    }

    pub const fn participant_principal_id(&self) -> PrincipalId {
        self.participant_principal_id
    }

    pub const fn source_revision(&self) -> NonZeroU64 {
        self.source_revision
    }

    pub const fn phase(&self) -> TranscriptSegmentPhase {
        self.phase
    }

    pub fn text(&self) -> &TranscriptText {
        &self.text
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub fn provenance(&self) -> &TranscriptSourceProvenance {
        &self.provenance
    }
}

impl fmt::Debug for TranscriptSegmentInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TranscriptSegmentInput")
            .field("identity", &self.identity)
            .field("segment_id", &self.segment_id)
            .field("message_id", &self.message_id)
            .field("participant_principal_id", &self.participant_principal_id)
            .field("source_revision", &self.source_revision)
            .field("phase", &self.phase)
            .field("text", &self.text)
            .field("started_at_millis", &self.started_at_millis)
            .field("ended_at_millis", &self.ended_at_millis)
            .field("observed_at_millis", &self.observed_at_millis)
            .field("operation_id", &self.operation_id)
            .field("provenance", &self.provenance)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptChannelRecordState {
    Partial,
    Final,
    Redacted,
    Expired,
}

#[derive(Clone, Eq, PartialEq)]
pub struct TranscriptChannelRecord {
    identity: HuddleIdentity,
    segment_id: HuddleTranscriptSegmentId,
    message_id: AggregateId,
    participant_principal_id: PrincipalId,
    source_revision: NonZeroU64,
    projection_version: NonZeroU64,
    state: TranscriptChannelRecordState,
    content: Option<TranscriptText>,
    started_at_millis: u64,
    ended_at_millis: u64,
    observed_at_millis: u64,
    last_operation_id: OperationId,
    provenance: TranscriptSourceProvenance,
    consent_version: NonZeroU64,
    retention: TranscriptRetention,
}

impl TranscriptChannelRecord {
    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn segment_id(&self) -> HuddleTranscriptSegmentId {
        self.segment_id
    }

    pub const fn message_id(&self) -> AggregateId {
        self.message_id
    }

    pub const fn participant_principal_id(&self) -> PrincipalId {
        self.participant_principal_id
    }

    pub const fn source_revision(&self) -> NonZeroU64 {
        self.source_revision
    }

    pub const fn projection_version(&self) -> NonZeroU64 {
        self.projection_version
    }

    pub const fn state(&self) -> TranscriptChannelRecordState {
        self.state
    }

    pub const fn content(&self) -> Option<&TranscriptText> {
        self.content.as_ref()
    }

    pub const fn started_at_millis(&self) -> u64 {
        self.started_at_millis
    }

    pub const fn ended_at_millis(&self) -> u64 {
        self.ended_at_millis
    }

    pub const fn observed_at_millis(&self) -> u64 {
        self.observed_at_millis
    }

    pub const fn last_operation_id(&self) -> OperationId {
        self.last_operation_id
    }

    pub fn provenance(&self) -> &TranscriptSourceProvenance {
        &self.provenance
    }

    pub const fn consent_version(&self) -> NonZeroU64 {
        self.consent_version
    }

    pub const fn retention(&self) -> TranscriptRetention {
        self.retention
    }
}

impl fmt::Debug for TranscriptChannelRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TranscriptChannelRecord")
            .field("identity", &self.identity)
            .field("segment_id", &self.segment_id)
            .field("message_id", &self.message_id)
            .field("participant_principal_id", &self.participant_principal_id)
            .field("source_revision", &self.source_revision)
            .field("projection_version", &self.projection_version)
            .field("state", &self.state)
            .field("content", &self.content)
            .field("started_at_millis", &self.started_at_millis)
            .field("ended_at_millis", &self.ended_at_millis)
            .field("observed_at_millis", &self.observed_at_millis)
            .field("last_operation_id", &self.last_operation_id)
            .field("provenance", &self.provenance)
            .field("consent_version", &self.consent_version)
            .field("retention", &self.retention)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TranscriptChannelRecordKey {
    identity: HuddleIdentity,
    segment_id: HuddleTranscriptSegmentId,
}

impl TranscriptChannelRecordKey {
    pub const fn new(identity: HuddleIdentity, segment_id: HuddleTranscriptSegmentId) -> Self {
        Self {
            identity,
            segment_id,
        }
    }

    pub const fn identity(self) -> HuddleIdentity {
        self.identity
    }

    pub const fn segment_id(self) -> HuddleTranscriptSegmentId {
        self.segment_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptChannelStoreOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptChannelStoreError {
    Unavailable,
    Conflict,
    UnknownOutcome,
}

pub trait TranscriptChannelStore {
    fn load(
        &mut self,
        key: TranscriptChannelRecordKey,
    ) -> Result<Option<TranscriptChannelRecord>, TranscriptChannelStoreError>;

    fn apply(
        &mut self,
        expected_projection_version: Option<NonZeroU64>,
        record: TranscriptChannelRecord,
        final_reference: Option<HuddleTranscriptReference>,
    ) -> Result<TranscriptChannelStoreOutcome, TranscriptChannelStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptProjectionOutcome {
    Applied,
    Unchanged,
    NotDue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptProjectionFunction {
    Project,
    Redact,
    Expire,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptProjectionVisibleFailure {
    function: TranscriptProjectionFunction,
    error: TranscriptProjectionError,
    retryable: bool,
}

impl TranscriptProjectionVisibleFailure {
    pub const fn function(self) -> TranscriptProjectionFunction {
        self.function
    }

    pub const fn error(self) -> TranscriptProjectionError {
        self.error
    }

    pub const fn retryable(self) -> bool {
        self.retryable
    }
}

#[derive(Default)]
pub struct TranscriptChannelProjector {
    last_failure: Option<TranscriptProjectionVisibleFailure>,
}

impl TranscriptChannelProjector {
    pub fn project_segment(
        &mut self,
        huddle: &Huddle,
        input: TranscriptSegmentInput,
        authorizer: &mut impl TranscriptProjectionAuthorizer,
        store: &mut impl TranscriptChannelStore,
    ) -> Result<TranscriptProjectionOutcome, TranscriptProjectionError> {
        self.validate_huddle(huddle, &input)?;
        let authorization = authorizer
            .authorize(TranscriptProjectionAuthorizationRequest {
                huddle_identity: input.identity,
                participant_principal_id: input.participant_principal_id,
                message_id: input.message_id,
            })
            .map_err(|error| self.authorization_failed(error))?;
        if retention_expired(authorization.retention, input.observed_at_millis) {
            return self.fail(
                TranscriptProjectionFunction::Project,
                TranscriptProjectionError::RetentionExpired,
                false,
            );
        }
        let key = TranscriptChannelRecordKey::new(input.identity, input.segment_id);
        let existing = store
            .load(key)
            .map_err(|error| self.store_failed(TranscriptProjectionFunction::Project, error))?;
        let (record, expected_projection_version) =
            self.next_record(existing.as_ref(), &input, authorization)?;
        if existing.as_ref() == Some(&record) {
            self.last_failure = None;
            return Ok(TranscriptProjectionOutcome::Unchanged);
        }
        let final_reference = if input.phase == TranscriptSegmentPhase::Final {
            Some(self.validate_final_reference(huddle, &input)?)
        } else {
            None
        };
        let outcome = store
            .apply(expected_projection_version, record, final_reference)
            .map_err(|error| self.store_failed(TranscriptProjectionFunction::Project, error))?;
        self.last_failure = None;
        Ok(map_store_outcome(outcome))
    }

    pub fn redact(
        &mut self,
        key: TranscriptChannelRecordKey,
        operation_id: OperationId,
        observed_at_millis: u64,
        store: &mut impl TranscriptChannelStore,
    ) -> Result<TranscriptProjectionOutcome, TranscriptProjectionError> {
        self.remove_content(
            TranscriptProjectionFunction::Redact,
            key,
            operation_id,
            observed_at_millis,
            None,
            TranscriptChannelRecordState::Redacted,
            store,
        )
    }

    pub fn expire(
        &mut self,
        key: TranscriptChannelRecordKey,
        operation_id: OperationId,
        observed_at_millis: u64,
        store: &mut impl TranscriptChannelStore,
    ) -> Result<TranscriptProjectionOutcome, TranscriptProjectionError> {
        self.remove_content(
            TranscriptProjectionFunction::Expire,
            key,
            operation_id,
            observed_at_millis,
            Some(observed_at_millis),
            TranscriptChannelRecordState::Expired,
            store,
        )
    }

    pub const fn last_failure(&self) -> Option<TranscriptProjectionVisibleFailure> {
        self.last_failure
    }

    fn next_record(
        &mut self,
        existing: Option<&TranscriptChannelRecord>,
        input: &TranscriptSegmentInput,
        authorization: TranscriptProjectionAuthorization,
    ) -> Result<(TranscriptChannelRecord, Option<NonZeroU64>), TranscriptProjectionError> {
        let (projection_version, expected_projection_version, retention) = match existing {
            Some(record) => {
                if record.last_operation_id == input.operation_id {
                    if record_matches_input(record, input) {
                        return Ok((record.clone(), Some(record.projection_version)));
                    }
                    return self.fail(
                        TranscriptProjectionFunction::Project,
                        TranscriptProjectionError::OperationConflict,
                        false,
                    );
                }
                if let Err(error) = validate_progression(record, input) {
                    return self.fail(TranscriptProjectionFunction::Project, error, false);
                }
                let projection_version = match next_version(record.projection_version) {
                    Ok(version) => version,
                    Err(error) => {
                        return self.fail(TranscriptProjectionFunction::Project, error, false);
                    }
                };
                (
                    projection_version,
                    Some(record.projection_version),
                    tighten_retention(record.retention, authorization.retention),
                )
            }
            None => {
                if input.source_revision.get() != 1 {
                    return self.fail(
                        TranscriptProjectionFunction::Project,
                        TranscriptProjectionError::StaleRevision,
                        false,
                    );
                }
                (NonZeroU64::MIN, None, authorization.retention)
            }
        };
        Ok((
            TranscriptChannelRecord {
                identity: input.identity,
                segment_id: input.segment_id,
                message_id: input.message_id,
                participant_principal_id: input.participant_principal_id,
                source_revision: input.source_revision,
                projection_version,
                state: match input.phase {
                    TranscriptSegmentPhase::Partial => TranscriptChannelRecordState::Partial,
                    TranscriptSegmentPhase::Final => TranscriptChannelRecordState::Final,
                },
                content: Some(input.text.clone()),
                started_at_millis: input.started_at_millis,
                ended_at_millis: input.ended_at_millis,
                observed_at_millis: input.observed_at_millis,
                last_operation_id: input.operation_id,
                provenance: input.provenance.clone(),
                consent_version: authorization.consent_version,
                retention,
            },
            expected_projection_version,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn remove_content(
        &mut self,
        function: TranscriptProjectionFunction,
        key: TranscriptChannelRecordKey,
        operation_id: OperationId,
        observed_at_millis: u64,
        expiry_check_at_millis: Option<u64>,
        state: TranscriptChannelRecordState,
        store: &mut impl TranscriptChannelStore,
    ) -> Result<TranscriptProjectionOutcome, TranscriptProjectionError> {
        if operation_id.as_uuid().is_nil() || observed_at_millis == 0 {
            return self.fail(function, TranscriptProjectionError::InvalidSegment, false);
        }
        let Some(mut record) = store
            .load(key)
            .map_err(|error| self.store_failed(function, error))?
        else {
            return self.fail(function, TranscriptProjectionError::MissingSegment, false);
        };
        if record.last_operation_id == operation_id {
            self.last_failure = None;
            return Ok(TranscriptProjectionOutcome::Unchanged);
        }
        if record.state == TranscriptChannelRecordState::Expired
            || record.state == TranscriptChannelRecordState::Redacted
        {
            self.last_failure = None;
            return Ok(TranscriptProjectionOutcome::Unchanged);
        }
        if expiry_check_at_millis.is_some_and(|now| !retention_expired(record.retention, now)) {
            self.last_failure = None;
            return Ok(TranscriptProjectionOutcome::NotDue);
        }
        let expected_projection_version = record.projection_version;
        record.projection_version = match next_version(record.projection_version) {
            Ok(version) => version,
            Err(error) => return self.fail(function, error, false),
        };
        record.state = state;
        record.content = None;
        record.observed_at_millis = observed_at_millis;
        record.last_operation_id = operation_id;
        let outcome = store
            .apply(Some(expected_projection_version), record, None)
            .map_err(|error| self.store_failed(function, error))?;
        self.last_failure = None;
        Ok(map_store_outcome(outcome))
    }

    fn validate_huddle(
        &mut self,
        huddle: &Huddle,
        input: &TranscriptSegmentInput,
    ) -> Result<(), TranscriptProjectionError> {
        let participant = huddle.participant(input.participant_principal_id);
        let error = if huddle.identity() != input.identity {
            Some(TranscriptProjectionError::WrongHuddle)
        } else if participant.is_none()
            || (input.phase == TranscriptSegmentPhase::Partial
                && (!matches!(huddle.lifecycle(), HuddleLifecycleState::Active)
                    || participant.is_none_or(|participant| {
                        participant.presence() != HuddleParticipantPresence::Present
                    })))
        {
            Some(TranscriptProjectionError::ParticipantUnavailable)
        } else {
            None
        };
        match error {
            Some(error) => self.fail(TranscriptProjectionFunction::Project, error, false),
            None => Ok(()),
        }
    }

    fn validate_final_reference(
        &mut self,
        huddle: &Huddle,
        input: &TranscriptSegmentInput,
    ) -> Result<HuddleTranscriptReference, TranscriptProjectionError> {
        let reference = match HuddleTranscriptReference::new(
            input.identity,
            input.segment_id,
            input.message_id,
            input.participant_principal_id,
            input.started_at_millis,
            input.ended_at_millis,
        ) {
            Ok(reference) => reference,
            Err(_) => {
                return self.fail(
                    TranscriptProjectionFunction::Project,
                    TranscriptProjectionError::InvalidSegment,
                    false,
                );
            }
        };
        let mut validation_huddle = huddle.clone();
        if validation_huddle
            .link_transcript(reference, input.operation_id, input.observed_at_millis)
            .is_err()
        {
            return self.fail(
                TranscriptProjectionFunction::Project,
                TranscriptProjectionError::InvalidSegment,
                false,
            );
        }
        Ok(reference)
    }

    fn authorization_failed(
        &mut self,
        error: TranscriptProjectionAuthorizationError,
    ) -> TranscriptProjectionError {
        let (error, retryable) = match error {
            TranscriptProjectionAuthorizationError::Denied => {
                (TranscriptProjectionError::NotAuthorized, false)
            }
            TranscriptProjectionAuthorizationError::Unavailable => {
                (TranscriptProjectionError::AuthorizationUnavailable, true)
            }
        };
        self.last_failure = Some(TranscriptProjectionVisibleFailure {
            function: TranscriptProjectionFunction::Project,
            error,
            retryable,
        });
        error
    }

    fn store_failed(
        &mut self,
        function: TranscriptProjectionFunction,
        error: TranscriptChannelStoreError,
    ) -> TranscriptProjectionError {
        let error = match error {
            TranscriptChannelStoreError::Unavailable => TranscriptProjectionError::StoreUnavailable,
            TranscriptChannelStoreError::Conflict => TranscriptProjectionError::StoreConflict,
            TranscriptChannelStoreError::UnknownOutcome => {
                TranscriptProjectionError::UnknownOutcome
            }
        };
        self.last_failure = Some(TranscriptProjectionVisibleFailure {
            function,
            error,
            retryable: true,
        });
        error
    }

    fn fail<T>(
        &mut self,
        function: TranscriptProjectionFunction,
        error: TranscriptProjectionError,
        retryable: bool,
    ) -> Result<T, TranscriptProjectionError> {
        self.last_failure = Some(TranscriptProjectionVisibleFailure {
            function,
            error,
            retryable,
        });
        Err(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptProjectionError {
    InvalidText,
    InvalidProvenance,
    InvalidSegment,
    WrongHuddle,
    ParticipantUnavailable,
    NotAuthorized,
    AuthorizationUnavailable,
    RetentionExpired,
    MissingSegment,
    StaleRevision,
    SegmentConflict,
    OperationConflict,
    TerminalSegment,
    VersionExhausted,
    StoreUnavailable,
    StoreConflict,
    UnknownOutcome,
}

impl fmt::Display for TranscriptProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidText => "transcript text is invalid or exceeds its bound",
            Self::InvalidProvenance => "transcript provenance is invalid",
            Self::InvalidSegment => "transcript segment is invalid",
            Self::WrongHuddle => "transcript belongs to another huddle",
            Self::ParticipantUnavailable => "transcript participant is unavailable",
            Self::NotAuthorized => "transcript projection is not authorized",
            Self::AuthorizationUnavailable => "transcript authorization is unavailable",
            Self::RetentionExpired => "transcript retention has expired",
            Self::MissingSegment => "transcript segment does not exist",
            Self::StaleRevision => "transcript revision is stale or discontinuous",
            Self::SegmentConflict => "transcript identity or provenance conflicts",
            Self::OperationConflict => "transcript operation was reused with different input",
            Self::TerminalSegment => "transcript segment is terminal",
            Self::VersionExhausted => "transcript projection version exhausted",
            Self::StoreUnavailable => "transcript channel store is unavailable",
            Self::StoreConflict => "transcript channel store rejected a stale write",
            Self::UnknownOutcome => "transcript channel write outcome is unknown",
        };
        formatter.write_str(message)
    }
}

impl Error for TranscriptProjectionError {}

fn validate_progression(
    existing: &TranscriptChannelRecord,
    input: &TranscriptSegmentInput,
) -> Result<(), TranscriptProjectionError> {
    if matches!(
        existing.state,
        TranscriptChannelRecordState::Final
            | TranscriptChannelRecordState::Redacted
            | TranscriptChannelRecordState::Expired
    ) {
        return Err(TranscriptProjectionError::TerminalSegment);
    }
    if existing.identity != input.identity
        || existing.segment_id != input.segment_id
        || existing.message_id != input.message_id
        || existing.participant_principal_id != input.participant_principal_id
        || existing.started_at_millis != input.started_at_millis
        || existing.provenance != input.provenance
    {
        return Err(TranscriptProjectionError::SegmentConflict);
    }
    if input.source_revision.get() != existing.source_revision.get().saturating_add(1)
        || input.ended_at_millis < existing.ended_at_millis
        || input.observed_at_millis < existing.observed_at_millis
    {
        return Err(TranscriptProjectionError::StaleRevision);
    }
    Ok(())
}

fn record_matches_input(record: &TranscriptChannelRecord, input: &TranscriptSegmentInput) -> bool {
    record.identity == input.identity
        && record.segment_id == input.segment_id
        && record.message_id == input.message_id
        && record.participant_principal_id == input.participant_principal_id
        && record.source_revision == input.source_revision
        && record.state
            == match input.phase {
                TranscriptSegmentPhase::Partial => TranscriptChannelRecordState::Partial,
                TranscriptSegmentPhase::Final => TranscriptChannelRecordState::Final,
            }
        && record.content.as_ref() == Some(&input.text)
        && record.started_at_millis == input.started_at_millis
        && record.ended_at_millis == input.ended_at_millis
        && record.observed_at_millis == input.observed_at_millis
        && record.provenance == input.provenance
}

fn next_version(version: NonZeroU64) -> Result<NonZeroU64, TranscriptProjectionError> {
    version
        .get()
        .checked_add(1)
        .and_then(NonZeroU64::new)
        .ok_or(TranscriptProjectionError::VersionExhausted)
}

fn tighten_retention(
    existing: TranscriptRetention,
    current: TranscriptRetention,
) -> TranscriptRetention {
    match (existing, current) {
        (TranscriptRetention::Retain, retention) | (retention, TranscriptRetention::Retain) => {
            retention
        }
        (TranscriptRetention::ExpireAt(first), TranscriptRetention::ExpireAt(second)) => {
            TranscriptRetention::ExpireAt(first.min(second))
        }
    }
}

fn retention_expired(retention: TranscriptRetention, now_millis: u64) -> bool {
    matches!(retention, TranscriptRetention::ExpireAt(expires_at) if now_millis >= expires_at.get())
}

fn valid_provenance_value(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TRANSCRIPT_PROVENANCE_BYTES
        && !value.chars().any(char::is_control)
}

fn map_store_outcome(outcome: TranscriptChannelStoreOutcome) -> TranscriptProjectionOutcome {
    match outcome {
        TranscriptChannelStoreOutcome::Applied => TranscriptProjectionOutcome::Applied,
        TranscriptChannelStoreOutcome::Unchanged => TranscriptProjectionOutcome::Unchanged,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use collaboration_domain::{CommunityId, HuddleGeneration, HuddleParticipantRole};

    use super::*;

    struct AllowPolicy {
        authorization: TranscriptProjectionAuthorization,
    }

    impl TranscriptProjectionAuthorizer for AllowPolicy {
        fn authorize(
            &mut self,
            _request: TranscriptProjectionAuthorizationRequest,
        ) -> Result<TranscriptProjectionAuthorization, TranscriptProjectionAuthorizationError>
        {
            Ok(self.authorization)
        }
    }

    struct DenyPolicy;

    impl TranscriptProjectionAuthorizer for DenyPolicy {
        fn authorize(
            &mut self,
            _request: TranscriptProjectionAuthorizationRequest,
        ) -> Result<TranscriptProjectionAuthorization, TranscriptProjectionAuthorizationError>
        {
            Err(TranscriptProjectionAuthorizationError::Denied)
        }
    }

    #[derive(Default)]
    struct MemoryStore {
        records: HashMap<TranscriptChannelRecordKey, TranscriptChannelRecord>,
        references: Vec<HuddleTranscriptReference>,
        fail_next: Option<TranscriptChannelStoreError>,
        apply_count: usize,
    }

    impl TranscriptChannelStore for MemoryStore {
        fn load(
            &mut self,
            key: TranscriptChannelRecordKey,
        ) -> Result<Option<TranscriptChannelRecord>, TranscriptChannelStoreError> {
            Ok(self.records.get(&key).cloned())
        }

        fn apply(
            &mut self,
            expected_projection_version: Option<NonZeroU64>,
            record: TranscriptChannelRecord,
            final_reference: Option<HuddleTranscriptReference>,
        ) -> Result<TranscriptChannelStoreOutcome, TranscriptChannelStoreError> {
            if let Some(error) = self.fail_next.take() {
                return Err(error);
            }
            let key = TranscriptChannelRecordKey::new(record.identity, record.segment_id);
            let existing = self.records.get(&key);
            if existing
                .is_some_and(|existing| existing.last_operation_id == record.last_operation_id)
            {
                return Ok(TranscriptChannelStoreOutcome::Unchanged);
            }
            if existing.map(TranscriptChannelRecord::projection_version)
                != expected_projection_version
            {
                return Err(TranscriptChannelStoreError::Conflict);
            }
            self.records.insert(key, record);
            if let Some(reference) = final_reference {
                self.references.push(reference);
            }
            self.apply_count += 1;
            Ok(TranscriptChannelStoreOutcome::Applied)
        }
    }

    struct Fixture {
        huddle: Huddle,
        speaker: PrincipalId,
        segment_id: HuddleTranscriptSegmentId,
        message_id: AggregateId,
        provenance: TranscriptSourceProvenance,
    }

    fn fixture() -> Fixture {
        let owner = PrincipalId::new();
        let speaker = PrincipalId::new();
        let identity = HuddleIdentity::new(
            CommunityId::new(),
            AggregateId::new(),
            AggregateId::new(),
            HuddleGeneration::new(4).expect("generation"),
        )
        .expect("identity");
        let mut huddle =
            Huddle::start(identity, owner, OperationId::new(), 1_000).expect("start huddle");
        huddle
            .join(
                speaker,
                HuddleParticipantRole::Speaker,
                OperationId::new(),
                1_100,
            )
            .expect("join speaker");
        Fixture {
            huddle,
            speaker,
            segment_id: HuddleTranscriptSegmentId::new(AggregateId::new()).expect("segment"),
            message_id: AggregateId::new(),
            provenance: TranscriptSourceProvenance::new(
                "whisper-small",
                "revision-7",
                [7; 32],
                NonZeroU64::new(9).expect("transcription generation"),
            )
            .expect("provenance"),
        }
    }

    fn input(
        fixture: &Fixture,
        revision: u64,
        phase: TranscriptSegmentPhase,
        text: &str,
        ended_at_millis: u64,
        operation_id: OperationId,
    ) -> TranscriptSegmentInput {
        TranscriptSegmentInput::new(
            fixture.huddle.identity(),
            fixture.segment_id,
            fixture.message_id,
            fixture.speaker,
            NonZeroU64::new(revision).expect("revision"),
            phase,
            TranscriptText::new(text).expect("text"),
            1_200,
            ended_at_millis,
            ended_at_millis + 10,
            operation_id,
            fixture.provenance.clone(),
        )
        .expect("input")
    }

    fn policy(expiry: u64) -> AllowPolicy {
        AllowPolicy {
            authorization: TranscriptProjectionAuthorization::new(
                NonZeroU64::new(3).expect("consent version"),
                TranscriptRetention::ExpireAt(NonZeroU64::new(expiry).expect("expiry")),
            ),
        }
    }

    #[test]
    fn partial_updates_finalize_one_channel_record_with_huddle_provenance() {
        let fixture = fixture();
        let mut projector = TranscriptChannelProjector::default();
        let mut policy = policy(10_000);
        let mut store = MemoryStore::default();

        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    1,
                    TranscriptSegmentPhase::Partial,
                    "one private",
                    1_300,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            ),
            Ok(TranscriptProjectionOutcome::Applied)
        );
        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    2,
                    TranscriptSegmentPhase::Partial,
                    "one private thought",
                    1_400,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            ),
            Ok(TranscriptProjectionOutcome::Applied)
        );
        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    3,
                    TranscriptSegmentPhase::Final,
                    "one private thought finalized",
                    1_500,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            ),
            Ok(TranscriptProjectionOutcome::Applied)
        );

        let key = TranscriptChannelRecordKey::new(fixture.huddle.identity(), fixture.segment_id);
        let record = store.records.get(&key).expect("record");
        assert_eq!(record.state(), TranscriptChannelRecordState::Final);
        assert_eq!(record.source_revision().get(), 3);
        assert_eq!(record.projection_version().get(), 3);
        assert_eq!(record.participant_principal_id(), fixture.speaker);
        assert_eq!(store.references.len(), 1);
        assert_eq!(store.references[0].message_id(), fixture.message_id);
    }

    #[test]
    fn failed_projection_is_visible_and_exact_retry_recovers() {
        let fixture = fixture();
        let mut projector = TranscriptChannelProjector::default();
        let mut policy = policy(10_000);
        let mut store = MemoryStore {
            fail_next: Some(TranscriptChannelStoreError::Unavailable),
            ..MemoryStore::default()
        };
        let operation_id = OperationId::new();
        let segment = input(
            &fixture,
            1,
            TranscriptSegmentPhase::Final,
            "retry me",
            1_400,
            operation_id,
        );

        assert_eq!(
            projector.project_segment(&fixture.huddle, segment.clone(), &mut policy, &mut store,),
            Err(TranscriptProjectionError::StoreUnavailable)
        );
        assert_eq!(
            projector.last_failure(),
            Some(TranscriptProjectionVisibleFailure {
                function: TranscriptProjectionFunction::Project,
                error: TranscriptProjectionError::StoreUnavailable,
                retryable: true,
            })
        );
        assert_eq!(store.apply_count, 0);

        assert_eq!(
            projector.project_segment(&fixture.huddle, segment, &mut policy, &mut store,),
            Ok(TranscriptProjectionOutcome::Applied)
        );
        assert!(projector.last_failure().is_none());
        assert_eq!(store.apply_count, 1);
    }

    #[test]
    fn redaction_removes_content_and_diagnostics_never_reveal_text() {
        let fixture = fixture();
        let mut projector = TranscriptChannelProjector::default();
        let mut policy = policy(10_000);
        let mut store = MemoryStore::default();
        projector
            .project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    1,
                    TranscriptSegmentPhase::Final,
                    "secret spoken phrase",
                    1_400,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            )
            .expect("project");
        let key = TranscriptChannelRecordKey::new(fixture.huddle.identity(), fixture.segment_id);
        assert_eq!(
            projector.redact(key, OperationId::new(), 1_500, &mut store),
            Ok(TranscriptProjectionOutcome::Applied)
        );
        let record = store.records.get(&key).expect("record");
        assert_eq!(record.state(), TranscriptChannelRecordState::Redacted);
        assert!(record.content().is_none());
        assert!(!format!("{record:?}").contains("secret spoken phrase"));
    }

    #[test]
    fn retention_expiry_is_exact_idempotent_and_removes_content() {
        let fixture = fixture();
        let mut projector = TranscriptChannelProjector::default();
        let mut policy = policy(2_000);
        let mut store = MemoryStore::default();
        projector
            .project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    1,
                    TranscriptSegmentPhase::Final,
                    "expires exactly",
                    1_400,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            )
            .expect("project");
        let key = TranscriptChannelRecordKey::new(fixture.huddle.identity(), fixture.segment_id);
        assert_eq!(
            projector.expire(key, OperationId::new(), 1_999, &mut store),
            Ok(TranscriptProjectionOutcome::NotDue)
        );
        let expiry_operation = OperationId::new();
        assert_eq!(
            projector.expire(key, expiry_operation, 2_000, &mut store),
            Ok(TranscriptProjectionOutcome::Applied)
        );
        assert_eq!(
            projector.expire(key, expiry_operation, 2_001, &mut store),
            Ok(TranscriptProjectionOutcome::Unchanged)
        );
        let record = store.records.get(&key).expect("record");
        assert_eq!(record.state(), TranscriptChannelRecordState::Expired);
        assert!(record.content().is_none());
    }

    #[test]
    fn stale_partial_and_post_final_updates_cannot_replace_trustworthy_text() {
        let fixture = fixture();
        let mut projector = TranscriptChannelProjector::default();
        let mut policy = policy(10_000);
        let mut store = MemoryStore::default();
        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    1,
                    TranscriptSegmentPhase::Partial,
                    "unauthorized partial",
                    1_250,
                    OperationId::new(),
                ),
                &mut DenyPolicy,
                &mut store,
            ),
            Err(TranscriptProjectionError::NotAuthorized)
        );
        assert!(store.records.is_empty());
        projector
            .project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    1,
                    TranscriptSegmentPhase::Partial,
                    "trusted partial",
                    1_300,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            )
            .expect("partial");
        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    3,
                    TranscriptSegmentPhase::Partial,
                    "skipped revision",
                    1_350,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            ),
            Err(TranscriptProjectionError::StaleRevision)
        );
        projector
            .project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    2,
                    TranscriptSegmentPhase::Final,
                    "trusted final",
                    1_400,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            )
            .expect("final");
        assert_eq!(
            projector.project_segment(
                &fixture.huddle,
                input(
                    &fixture,
                    3,
                    TranscriptSegmentPhase::Partial,
                    "late replacement",
                    1_500,
                    OperationId::new(),
                ),
                &mut policy,
                &mut store,
            ),
            Err(TranscriptProjectionError::TerminalSegment)
        );
        let key = TranscriptChannelRecordKey::new(fixture.huddle.identity(), fixture.segment_id);
        assert_eq!(
            store
                .records
                .get(&key)
                .and_then(TranscriptChannelRecord::content)
                .map(TranscriptText::as_str),
            Some("trusted final")
        );
    }
}
