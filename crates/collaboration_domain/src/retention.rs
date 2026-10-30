use std::{error::Error, fmt, num::NonZeroU16, num::NonZeroU64};

use crate::{
    AggregateId, AggregateVersion, CommunityArchivePolicyState, CommunityArchiveSnapshot,
    CommunityId,
};

pub const MAX_RETENTION_KIND_RULES: usize = 128;
pub const CURRENT_RETENTION_POLICY_SCHEMA_VERSION: RetentionPolicySchemaVersion =
    RetentionPolicySchemaVersion(NonZeroU16::MIN);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RetentionPolicySchemaVersion(NonZeroU16);

impl RetentionPolicySchemaVersion {
    pub const fn new(value: u16) -> Option<Self> {
        match NonZeroU16::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RetentionPersistenceClass {
    Durable,
    Ephemeral,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RetentionEventKind {
    value: u16,
    persistence: RetentionPersistenceClass,
}

impl RetentionEventKind {
    pub fn from_registry(
        value: u16,
        persistence: RetentionPersistenceClass,
    ) -> Result<Self, RetentionError> {
        let is_ephemeral = (20_000..=29_999).contains(&value);
        if is_ephemeral != (persistence == RetentionPersistenceClass::Ephemeral) {
            return Err(RetentionError::InvalidEventKind);
        }
        Ok(Self { value, persistence })
    }

    pub const fn value(self) -> u16 {
        self.value
    }

    pub const fn persistence(self) -> RetentionPersistenceClass {
        self.persistence
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RetentionTtl(NonZeroU64);

impl RetentionTtl {
    pub const fn from_millis(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn as_millis(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveRetentionRule {
    Preserve,
    FollowCommunityPolicy,
    DeleteOnArchive,
    ExpireAfter(RetentionTtl),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionKindRule {
    pub event_kind: RetentionEventKind,
    pub ttl: Option<RetentionTtl>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityRetentionPolicyFields {
    pub community_id: CommunityId,
    pub schema_version: RetentionPolicySchemaVersion,
    pub version: AggregateVersion,
    pub default_ttl: Option<RetentionTtl>,
    pub archive_rule: ArchiveRetentionRule,
    pub kind_rules: Vec<RetentionKindRule>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityRetentionPolicy {
    fields: CommunityRetentionPolicyFields,
}

impl CommunityRetentionPolicy {
    pub fn from_record(fields: CommunityRetentionPolicyFields) -> Result<Self, RetentionError> {
        if fields.community_id.as_uuid().is_nil()
            || fields.kind_rules.len() > MAX_RETENTION_KIND_RULES
            || fields
                .kind_rules
                .iter()
                .any(|rule| rule.event_kind.persistence() == RetentionPersistenceClass::Ephemeral)
        {
            return Err(RetentionError::InvalidPolicy);
        }
        for (index, rule) in fields.kind_rules.iter().enumerate() {
            if fields.kind_rules[..index]
                .iter()
                .any(|existing| existing.event_kind.value() == rule.event_kind.value())
            {
                return Err(RetentionError::InvalidPolicy);
            }
        }
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &CommunityRetentionPolicyFields {
        &self.fields
    }

    fn ttl_for(&self, event_kind: RetentionEventKind) -> Option<RetentionTtl> {
        self.fields
            .kind_rules
            .iter()
            .find(|rule| rule.event_kind.value() == event_kind.value())
            .map_or(self.fields.default_ttl, |rule| rule.ttl)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionSnapshot<T> {
    Absent,
    Current(T),
    Unavailable,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegalHoldScope {
    Community,
    Record(AggregateId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegalHoldState {
    Active,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegalHoldSnapshot {
    pub community_id: CommunityId,
    pub scope: LegalHoldScope,
    pub state: LegalHoldState,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionArchiveSnapshot {
    pub archive: CommunityArchiveSnapshot,
    pub archived_at_millis: Option<u64>,
}

impl RetentionArchiveSnapshot {
    fn validate(self) -> Result<(), RetentionError> {
        match (self.archive.state, self.archived_at_millis) {
            (CommunityArchivePolicyState::Active, None)
            | (CommunityArchivePolicyState::Archived, Some(_)) => Ok(()),
            _ => Err(RetentionError::InvalidArchive),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionRecord {
    pub community_id: CommunityId,
    pub record_id: AggregateId,
    pub event_kind: RetentionEventKind,
    pub retention_started_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionVisibility {
    Live,
    ArchiveOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionReason {
    Permanent,
    CommunityPolicy,
    LegalHold,
    CommunityArchive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDisposition {
    DoNotPersist,
    Retain {
        visibility: RetentionVisibility,
        reason: RetentionReason,
    },
    DeleteAt {
        visibility: RetentionVisibility,
        reason: RetentionReason,
        expires_at_millis: u64,
    },
    DeleteNow {
        reason: RetentionReason,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PolicyFence {
    schema_version: RetentionPolicySchemaVersion,
    version: AggregateVersion,
    ttl: Option<RetentionTtl>,
    archive_rule: ArchiveRetentionRule,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HoldFence {
    version: AggregateVersion,
    scope: LegalHoldScope,
    state: LegalHoldState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArchiveFence {
    version: AggregateVersion,
    state: CommunityArchivePolicyState,
    archived_at_millis: Option<u64>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct RetentionDecision {
    record: RetentionRecord,
    evaluated_at_millis: u64,
    policy_fence: Option<PolicyFence>,
    hold_fence: Option<HoldFence>,
    archive_fence: Option<ArchiveFence>,
    disposition: RetentionDisposition,
}

impl RetentionDecision {
    pub const fn community_id(&self) -> CommunityId {
        self.record.community_id
    }

    pub const fn record_id(&self) -> AggregateId {
        self.record.record_id
    }

    pub const fn event_kind(&self) -> RetentionEventKind {
        self.record.event_kind
    }

    pub const fn evaluated_at_millis(&self) -> u64 {
        self.evaluated_at_millis
    }

    pub const fn policy_version(&self) -> Option<AggregateVersion> {
        match self.policy_fence {
            Some(fence) => Some(fence.version),
            None => None,
        }
    }

    pub const fn hold_version(&self) -> Option<AggregateVersion> {
        match self.hold_fence {
            Some(fence) => Some(fence.version),
            None => None,
        }
    }

    pub const fn archive_version(&self) -> Option<AggregateVersion> {
        match self.archive_fence {
            Some(fence) => Some(fence.version),
            None => None,
        }
    }

    pub const fn disposition(&self) -> RetentionDisposition {
        self.disposition
    }
}

impl fmt::Debug for RetentionDecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetentionDecision")
            .field("community_id", &self.record.community_id)
            .field("record_id", &"<redacted>")
            .field("event_kind", &self.record.event_kind)
            .field("evaluated_at_millis", &self.evaluated_at_millis)
            .field("policy_version", &self.policy_version())
            .field("hold_version", &self.hold_version())
            .field("archive_version", &self.archive_version())
            .field("disposition", &self.disposition)
            .finish()
    }
}

pub struct RetentionRequest<'request> {
    pub record: RetentionRecord,
    pub policy: RetentionSnapshot<&'request CommunityRetentionPolicy>,
    pub legal_hold: RetentionSnapshot<LegalHoldSnapshot>,
    pub community_archive: RetentionSnapshot<RetentionArchiveSnapshot>,
    pub now_millis: u64,
    pub previous_decision: Option<&'request RetentionDecision>,
}

impl fmt::Debug for RetentionRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetentionRequest")
            .field("community_id", &self.record.community_id)
            .field("record_id", &"<redacted>")
            .field("event_kind", &self.record.event_kind)
            .field("now_millis", &self.now_millis)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionResolution {
    Evaluated(RetentionDecision),
    Unchanged(RetentionDecision),
}

impl RetentionResolution {
    pub const fn decision(&self) -> &RetentionDecision {
        match self {
            Self::Evaluated(decision) | Self::Unchanged(decision) => decision,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionError {
    InvalidEventKind,
    InvalidRecord,
    InvalidPolicy,
    InvalidArchive,
    InvalidHold,
    TenantMismatch,
    AuthorityUnavailable,
    AuthorityAmbiguous,
    UnsupportedPolicyVersion,
    InvalidDeadline,
    StaleAuthority,
    ClockRegression,
    RetryConflict,
}

impl fmt::Display for RetentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidEventKind => "retention event kind is invalid",
            Self::InvalidRecord => "retention record is invalid",
            Self::InvalidPolicy => "retention policy is invalid",
            Self::InvalidArchive => "retention archive state is invalid",
            Self::InvalidHold => "retention hold is invalid",
            Self::TenantMismatch => "retention tenant does not match",
            Self::AuthorityUnavailable => "retention authority is unavailable",
            Self::AuthorityAmbiguous => "retention authority is ambiguous",
            Self::UnsupportedPolicyVersion => "retention policy version is unsupported",
            Self::InvalidDeadline => "retention deadline is invalid",
            Self::StaleAuthority => "retention authority is stale",
            Self::ClockRegression => "retention clock regressed",
            Self::RetryConflict => "retention retry conflicts with prior evaluation",
        };
        formatter.write_str(message)
    }
}

impl Error for RetentionError {}

pub fn resolve_retention(
    request: &RetentionRequest<'_>,
) -> Result<RetentionResolution, RetentionError> {
    validate_record(request.record)?;

    let policy = resolve_snapshot(request.policy)?;
    if let Some(policy) = policy {
        if policy.fields.community_id != request.record.community_id {
            return Err(RetentionError::TenantMismatch);
        }
        if policy.fields.schema_version != CURRENT_RETENTION_POLICY_SCHEMA_VERSION {
            return Err(RetentionError::UnsupportedPolicyVersion);
        }
    }
    let hold = resolve_snapshot(request.legal_hold)?;
    if let Some(hold) = hold {
        validate_hold(request.record, hold)?;
    }
    let archive = resolve_snapshot(request.community_archive)?;
    if let Some(archive) = archive {
        archive.validate()?;
        if archive.archive.community_id != request.record.community_id {
            return Err(RetentionError::TenantMismatch);
        }
        if archive
            .archived_at_millis
            .is_some_and(|archived_at_millis| archived_at_millis > request.now_millis)
        {
            return Err(RetentionError::InvalidArchive);
        }
    }

    if request.record.event_kind.persistence() == RetentionPersistenceClass::Ephemeral {
        if hold.is_some_and(|hold| hold.state == LegalHoldState::Active) {
            return Err(RetentionError::InvalidHold);
        }
        return finish_decision(
            request,
            policy_fence(policy, request.record.event_kind),
            hold.map(hold_fence),
            archive.map(archive_fence),
            RetentionDisposition::DoNotPersist,
        );
    }

    let visibility = archive.map_or(RetentionVisibility::Live, |archive| {
        if archive.archive.state == CommunityArchivePolicyState::Archived {
            RetentionVisibility::ArchiveOnly
        } else {
            RetentionVisibility::Live
        }
    });

    let disposition = if hold.is_some_and(|hold| hold.state == LegalHoldState::Active) {
        RetentionDisposition::Retain {
            visibility,
            reason: RetentionReason::LegalHold,
        }
    } else if let Some(archive) =
        archive.filter(|archive| archive.archive.state == CommunityArchivePolicyState::Archived)
    {
        resolve_archived_disposition(request, policy, archive, visibility)?
    } else {
        resolve_policy_disposition(request, policy, visibility)?
    };

    finish_decision(
        request,
        policy_fence(policy, request.record.event_kind),
        hold.map(hold_fence),
        archive.map(archive_fence),
        disposition,
    )
}

fn validate_record(record: RetentionRecord) -> Result<(), RetentionError> {
    if record.community_id.as_uuid().is_nil() || record.record_id.as_uuid().is_nil() {
        return Err(RetentionError::InvalidRecord);
    }
    Ok(())
}

fn validate_hold(record: RetentionRecord, hold: LegalHoldSnapshot) -> Result<(), RetentionError> {
    if hold.community_id != record.community_id {
        return Err(RetentionError::TenantMismatch);
    }
    if let LegalHoldScope::Record(record_id) = hold.scope
        && record_id != record.record_id
    {
        return Err(RetentionError::InvalidHold);
    }
    Ok(())
}

fn resolve_snapshot<T>(snapshot: RetentionSnapshot<T>) -> Result<Option<T>, RetentionError> {
    match snapshot {
        RetentionSnapshot::Absent => Ok(None),
        RetentionSnapshot::Current(value) => Ok(Some(value)),
        RetentionSnapshot::Unavailable => Err(RetentionError::AuthorityUnavailable),
        RetentionSnapshot::Ambiguous => Err(RetentionError::AuthorityAmbiguous),
    }
}

fn resolve_archived_disposition(
    request: &RetentionRequest<'_>,
    policy: Option<&CommunityRetentionPolicy>,
    archive: RetentionArchiveSnapshot,
    visibility: RetentionVisibility,
) -> Result<RetentionDisposition, RetentionError> {
    let archive_rule = policy.map_or(ArchiveRetentionRule::Preserve, |policy| {
        policy.fields.archive_rule
    });
    match archive_rule {
        ArchiveRetentionRule::Preserve => Ok(RetentionDisposition::Retain {
            visibility,
            reason: RetentionReason::CommunityArchive,
        }),
        ArchiveRetentionRule::FollowCommunityPolicy => {
            resolve_policy_disposition(request, policy, visibility)
        }
        ArchiveRetentionRule::DeleteOnArchive => Ok(RetentionDisposition::DeleteNow {
            reason: RetentionReason::CommunityArchive,
        }),
        ArchiveRetentionRule::ExpireAfter(ttl) => deadline_disposition(
            archive
                .archived_at_millis
                .ok_or(RetentionError::InvalidArchive)?,
            ttl,
            request.now_millis,
            visibility,
            RetentionReason::CommunityArchive,
        ),
    }
}

fn resolve_policy_disposition(
    request: &RetentionRequest<'_>,
    policy: Option<&CommunityRetentionPolicy>,
    visibility: RetentionVisibility,
) -> Result<RetentionDisposition, RetentionError> {
    let Some(ttl) = policy.and_then(|policy| policy.ttl_for(request.record.event_kind)) else {
        return Ok(RetentionDisposition::Retain {
            visibility,
            reason: RetentionReason::Permanent,
        });
    };
    deadline_disposition(
        request.record.retention_started_at_millis,
        ttl,
        request.now_millis,
        visibility,
        RetentionReason::CommunityPolicy,
    )
}

fn deadline_disposition(
    started_at_millis: u64,
    ttl: RetentionTtl,
    now_millis: u64,
    visibility: RetentionVisibility,
    reason: RetentionReason,
) -> Result<RetentionDisposition, RetentionError> {
    let expires_at_millis = started_at_millis
        .checked_add(ttl.as_millis())
        .ok_or(RetentionError::InvalidDeadline)?;
    if now_millis >= expires_at_millis {
        Ok(RetentionDisposition::DeleteNow { reason })
    } else {
        Ok(RetentionDisposition::DeleteAt {
            visibility,
            reason,
            expires_at_millis,
        })
    }
}

fn policy_fence(
    policy: Option<&CommunityRetentionPolicy>,
    event_kind: RetentionEventKind,
) -> Option<PolicyFence> {
    policy.map(|policy| PolicyFence {
        schema_version: policy.fields.schema_version,
        version: policy.fields.version,
        ttl: policy.ttl_for(event_kind),
        archive_rule: policy.fields.archive_rule,
    })
}

const fn hold_fence(hold: LegalHoldSnapshot) -> HoldFence {
    HoldFence {
        version: hold.version,
        scope: hold.scope,
        state: hold.state,
    }
}

const fn archive_fence(archive: RetentionArchiveSnapshot) -> ArchiveFence {
    ArchiveFence {
        version: archive.archive.version,
        state: archive.archive.state,
        archived_at_millis: archive.archived_at_millis,
    }
}

fn finish_decision(
    request: &RetentionRequest<'_>,
    policy_fence: Option<PolicyFence>,
    hold_fence: Option<HoldFence>,
    archive_fence: Option<ArchiveFence>,
    disposition: RetentionDisposition,
) -> Result<RetentionResolution, RetentionError> {
    let decision = RetentionDecision {
        record: request.record,
        evaluated_at_millis: request.now_millis,
        policy_fence,
        hold_fence,
        archive_fence,
        disposition,
    };
    let Some(previous) = request.previous_decision else {
        return Ok(RetentionResolution::Evaluated(decision));
    };
    if previous.record != decision.record {
        return Err(RetentionError::RetryConflict);
    }
    if previous.evaluated_at_millis > decision.evaluated_at_millis {
        return Err(RetentionError::ClockRegression);
    }
    validate_fence_progress(previous.policy_fence, decision.policy_fence)?;
    validate_fence_progress(previous.hold_fence, decision.hold_fence)?;
    validate_fence_progress(previous.archive_fence, decision.archive_fence)?;
    if previous.evaluated_at_millis == decision.evaluated_at_millis {
        if previous == &decision {
            return Ok(RetentionResolution::Unchanged(decision));
        }
        return Err(RetentionError::RetryConflict);
    }
    Ok(RetentionResolution::Evaluated(decision))
}

trait VersionedFence: Copy + Eq {
    fn version(self) -> AggregateVersion;
}

impl VersionedFence for PolicyFence {
    fn version(self) -> AggregateVersion {
        self.version
    }
}

impl VersionedFence for HoldFence {
    fn version(self) -> AggregateVersion {
        self.version
    }
}

impl VersionedFence for ArchiveFence {
    fn version(self) -> AggregateVersion {
        self.version
    }
}

fn validate_fence_progress<T: VersionedFence>(
    previous: Option<T>,
    current: Option<T>,
) -> Result<(), RetentionError> {
    match (previous, current) {
        (None, _) => Ok(()),
        (Some(_), None) => Err(RetentionError::StaleAuthority),
        (Some(previous), Some(current)) if current.version() < previous.version() => {
            Err(RetentionError::StaleAuthority)
        }
        (Some(previous), Some(current))
            if current.version() == previous.version() && current != previous =>
        {
            Err(RetentionError::RetryConflict)
        }
        (Some(_), Some(_)) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn record(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn durable_kind(value: u16) -> RetentionEventKind {
        RetentionEventKind::from_registry(value, RetentionPersistenceClass::Durable)
            .expect("durable kind")
    }

    fn ttl(value: u64) -> RetentionTtl {
        RetentionTtl::from_millis(value).expect("retention ttl")
    }

    fn make_policy(
        community_id: CommunityId,
        schema_version: u16,
        version: AggregateVersion,
        default_ttl: Option<RetentionTtl>,
        archive_rule: ArchiveRetentionRule,
        kind_rules: Vec<RetentionKindRule>,
    ) -> CommunityRetentionPolicy {
        CommunityRetentionPolicy::from_record(CommunityRetentionPolicyFields {
            community_id,
            schema_version: RetentionPolicySchemaVersion::new(schema_version)
                .expect("schema version"),
            version,
            default_ttl,
            archive_rule,
            kind_rules,
        })
        .expect("policy")
    }

    fn retention_record(
        community_id: CommunityId,
        event_kind: RetentionEventKind,
    ) -> RetentionRecord {
        RetentionRecord {
            community_id,
            record_id: record(10),
            event_kind,
            retention_started_at_millis: 1_000,
        }
    }

    fn archive(
        community_id: CommunityId,
        state: CommunityArchivePolicyState,
        archived_at_millis: Option<u64>,
        version: AggregateVersion,
    ) -> RetentionArchiveSnapshot {
        RetentionArchiveSnapshot {
            archive: CommunityArchiveSnapshot {
                community_id,
                state,
                version,
            },
            archived_at_millis,
        }
    }

    fn request<'a>(
        record: RetentionRecord,
        policy: RetentionSnapshot<&'a CommunityRetentionPolicy>,
        legal_hold: RetentionSnapshot<LegalHoldSnapshot>,
        community_archive: RetentionSnapshot<RetentionArchiveSnapshot>,
        now_millis: u64,
        previous_decision: Option<&'a RetentionDecision>,
    ) -> RetentionRequest<'a> {
        RetentionRequest {
            record,
            policy,
            legal_hold,
            community_archive,
            now_millis,
            previous_decision,
        }
    }

    #[test]
    fn ttl_uses_authoritative_anchor_and_kind_override_at_exact_boundary() {
        let community_id = community(1);
        let event_kind = durable_kind(1);
        let permanent_kind = durable_kind(3);
        let policy = make_policy(
            community_id,
            1,
            AggregateVersion::FIRST,
            Some(ttl(1_000)),
            ArchiveRetentionRule::FollowCommunityPolicy,
            vec![RetentionKindRule {
                event_kind: permanent_kind,
                ttl: None,
            }],
        );
        let before = resolve_retention(&request(
            retention_record(community_id, event_kind),
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            1_999,
            None,
        ))
        .expect("scheduled retention");
        assert_eq!(
            before.decision().disposition(),
            RetentionDisposition::DeleteAt {
                visibility: RetentionVisibility::Live,
                reason: RetentionReason::CommunityPolicy,
                expires_at_millis: 2_000,
            }
        );
        let due = resolve_retention(&request(
            retention_record(community_id, event_kind),
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            2_000,
            None,
        ))
        .expect("due retention");
        assert_eq!(
            due.decision().disposition(),
            RetentionDisposition::DeleteNow {
                reason: RetentionReason::CommunityPolicy
            }
        );
        let permanent = resolve_retention(&request(
            retention_record(community_id, permanent_kind),
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            10_000,
            None,
        ))
        .expect("permanent override");
        assert_eq!(
            permanent.decision().disposition(),
            RetentionDisposition::Retain {
                visibility: RetentionVisibility::Live,
                reason: RetentionReason::Permanent,
            }
        );
    }

    #[test]
    fn archive_rules_hide_retained_records_and_expire_from_archive_time() {
        let community_id = community(1);
        let policy = make_policy(
            community_id,
            1,
            AggregateVersion::FIRST,
            None,
            ArchiveRetentionRule::ExpireAfter(ttl(2_000)),
            Vec::new(),
        );
        let archive = archive(
            community_id,
            CommunityArchivePolicyState::Archived,
            Some(5_000),
            AggregateVersion::FIRST,
        );
        let before = resolve_retention(&request(
            retention_record(community_id, durable_kind(1)),
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Current(archive),
            6_999,
            None,
        ))
        .expect("archived schedule");
        assert_eq!(
            before.decision().disposition(),
            RetentionDisposition::DeleteAt {
                visibility: RetentionVisibility::ArchiveOnly,
                reason: RetentionReason::CommunityArchive,
                expires_at_millis: 7_000,
            }
        );
        let due = resolve_retention(&request(
            retention_record(community_id, durable_kind(1)),
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Current(archive),
            7_000,
            None,
        ))
        .expect("archived expiry");
        assert_eq!(
            due.decision().disposition(),
            RetentionDisposition::DeleteNow {
                reason: RetentionReason::CommunityArchive
            }
        );
    }

    #[test]
    fn legal_hold_overrides_due_ttl_and_archive_without_restoring_visibility() {
        let community_id = community(1);
        let event_record = retention_record(community_id, durable_kind(1));
        let policy = make_policy(
            community_id,
            1,
            AggregateVersion::FIRST,
            Some(ttl(1)),
            ArchiveRetentionRule::DeleteOnArchive,
            Vec::new(),
        );
        let archive = archive(
            community_id,
            CommunityArchivePolicyState::Archived,
            Some(1_500),
            AggregateVersion::FIRST,
        );
        let hold = LegalHoldSnapshot {
            community_id,
            scope: LegalHoldScope::Record(event_record.record_id),
            state: LegalHoldState::Active,
            version: AggregateVersion::FIRST,
        };
        let held = resolve_retention(&request(
            event_record,
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Current(hold),
            RetentionSnapshot::Current(archive),
            10_000,
            None,
        ))
        .expect("legal hold");
        assert_eq!(
            held.decision().disposition(),
            RetentionDisposition::Retain {
                visibility: RetentionVisibility::ArchiveOnly,
                reason: RetentionReason::LegalHold,
            }
        );

        let released = LegalHoldSnapshot {
            state: LegalHoldState::Released,
            version: AggregateVersion::FIRST.next().expect("next hold version"),
            ..hold
        };
        let unheld = resolve_retention(&request(
            event_record,
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Current(released),
            RetentionSnapshot::Current(archive),
            10_001,
            Some(held.decision()),
        ))
        .expect("released hold");
        assert_eq!(
            unheld.decision().disposition(),
            RetentionDisposition::DeleteNow {
                reason: RetentionReason::CommunityArchive
            }
        );
    }

    #[test]
    fn ephemeral_and_mixed_version_inputs_fail_closed() {
        let community_id = community(1);
        let ephemeral_kind =
            RetentionEventKind::from_registry(24_200, RetentionPersistenceClass::Ephemeral)
                .expect("ephemeral kind");
        let future_policy = make_policy(
            community_id,
            2,
            AggregateVersion::FIRST,
            Some(ttl(1)),
            ArchiveRetentionRule::DeleteOnArchive,
            Vec::new(),
        );
        assert_eq!(
            resolve_retention(&request(
                retention_record(community_id, durable_kind(1)),
                RetentionSnapshot::Current(&future_policy),
                RetentionSnapshot::Absent,
                RetentionSnapshot::Absent,
                10_000,
                None,
            )),
            Err(RetentionError::UnsupportedPolicyVersion)
        );
        assert_eq!(
            resolve_retention(&request(
                retention_record(community_id, durable_kind(1)),
                RetentionSnapshot::Unavailable,
                RetentionSnapshot::Absent,
                RetentionSnapshot::Absent,
                10_000,
                None,
            )),
            Err(RetentionError::AuthorityUnavailable)
        );
        assert_eq!(
            resolve_retention(&request(
                retention_record(community_id, durable_kind(1)),
                RetentionSnapshot::Absent,
                RetentionSnapshot::Absent,
                RetentionSnapshot::Current(archive(
                    community_id,
                    CommunityArchivePolicyState::Archived,
                    Some(10_001),
                    AggregateVersion::FIRST,
                )),
                10_000,
                None,
            )),
            Err(RetentionError::InvalidArchive)
        );
        let ephemeral = resolve_retention(&request(
            retention_record(community_id, ephemeral_kind),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            10_000,
            None,
        ))
        .expect("ephemeral disposition");
        assert_eq!(
            ephemeral.decision().disposition(),
            RetentionDisposition::DoNotPersist
        );
        assert_eq!(
            RetentionEventKind::from_registry(24_200, RetentionPersistenceClass::Durable),
            Err(RetentionError::InvalidEventKind)
        );
    }

    #[test]
    fn exact_retry_is_unchanged_while_time_and_authority_advance_monotonically() {
        let community_id = community(1);
        let event_record = retention_record(community_id, durable_kind(1));
        let policy = make_policy(
            community_id,
            1,
            AggregateVersion::FIRST,
            Some(ttl(1_000)),
            ArchiveRetentionRule::FollowCommunityPolicy,
            Vec::new(),
        );
        let first = resolve_retention(&request(
            event_record,
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            1_500,
            None,
        ))
        .expect("first evaluation");
        let retry = resolve_retention(&request(
            event_record,
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            1_500,
            Some(first.decision()),
        ))
        .expect("exact retry");
        assert!(matches!(retry, RetentionResolution::Unchanged(_)));

        let due = resolve_retention(&request(
            event_record,
            RetentionSnapshot::Current(&policy),
            RetentionSnapshot::Absent,
            RetentionSnapshot::Absent,
            2_000,
            Some(first.decision()),
        ))
        .expect("time advancement");
        assert_eq!(
            due.decision().disposition(),
            RetentionDisposition::DeleteNow {
                reason: RetentionReason::CommunityPolicy
            }
        );

        let changed_without_version = make_policy(
            community_id,
            1,
            AggregateVersion::FIRST,
            Some(ttl(2_000)),
            ArchiveRetentionRule::FollowCommunityPolicy,
            Vec::new(),
        );
        assert_eq!(
            resolve_retention(&request(
                event_record,
                RetentionSnapshot::Current(&changed_without_version),
                RetentionSnapshot::Absent,
                RetentionSnapshot::Absent,
                1_600,
                Some(first.decision()),
            )),
            Err(RetentionError::RetryConflict)
        );
        assert!(!format!("{:?}", first.decision()).contains(&record(10).to_string()));
    }
}
