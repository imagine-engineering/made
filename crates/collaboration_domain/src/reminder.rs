use std::{cmp::Ordering, error::Error, fmt};

use crate::{CommunityId, NostrEventId, PrincipalId};

const MAX_SAFE_TIMESTAMP_SECONDS: u64 = 9_007_199_254_740_991;
const MIN_TERMINAL_RETENTION_SECONDS: u64 = 30 * 24 * 60 * 60;
const MAX_TERMINAL_RETENTION_SECONDS: u64 = 90 * 24 * 60 * 60;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReminderId(String);

impl ReminderId {
    pub fn new(value: impl Into<String>) -> Result<Self, ReminderError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ReminderError::InvalidIdentity);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ReminderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ReminderId(<redacted>)")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReminderScope {
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
}

impl ReminderScope {
    pub const fn new(community_id: CommunityId, owner_principal_id: PrincipalId) -> Self {
        Self {
            community_id,
            owner_principal_id,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn owner_principal_id(self) -> PrincipalId {
        self.owner_principal_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReminderHead {
    event_id: NostrEventId,
    created_at_seconds: u64,
}

impl ReminderHead {
    pub fn new(event_id: NostrEventId, created_at_seconds: u64) -> Result<Self, ReminderError> {
        if event_id.as_bytes().iter().all(|byte| *byte == 0)
            || created_at_seconds > MAX_SAFE_TIMESTAMP_SECONDS
        {
            return Err(ReminderError::InvalidHead);
        }
        Ok(Self {
            event_id,
            created_at_seconds,
        })
    }

    pub const fn event_id(self) -> NostrEventId {
        self.event_id
    }

    pub const fn created_at_seconds(self) -> u64 {
        self.created_at_seconds
    }

    fn replacement_order(self, current: Self) -> Ordering {
        self.created_at_seconds
            .cmp(&current.created_at_seconds)
            .then_with(|| current.event_id.cmp(&self.event_id))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ReminderTarget {
    event_id: Option<NostrEventId>,
    address: Option<String>,
}

impl ReminderTarget {
    pub fn new(
        event_id: Option<NostrEventId>,
        address: Option<String>,
    ) -> Result<Self, ReminderError> {
        if event_id.is_none() && address.is_none() {
            return Err(ReminderError::InvalidTarget);
        }
        if event_id.is_some_and(|event_id| event_id.as_bytes().iter().all(|byte| *byte == 0))
            || address.as_ref().is_some_and(String::is_empty)
        {
            return Err(ReminderError::InvalidTarget);
        }
        Ok(Self { event_id, address })
    }

    pub const fn event_id(&self) -> Option<NostrEventId> {
        self.event_id
    }

    pub fn address(&self) -> Option<&str> {
        self.address.as_deref()
    }
}

impl fmt::Debug for ReminderTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReminderTarget")
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ReminderContent {
    target: Option<ReminderTarget>,
    note: Option<String>,
}

impl ReminderContent {
    pub const fn new(target: Option<ReminderTarget>, note: Option<String>) -> Self {
        Self { target, note }
    }

    pub const fn target(&self) -> Option<&ReminderTarget> {
        self.target.as_ref()
    }

    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    fn is_actionable(&self) -> bool {
        self.target.is_some() || self.note.as_ref().is_some_and(|note| !note.is_empty())
    }
}

impl fmt::Debug for ReminderContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReminderContent")
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderLifecycle {
    Pending { not_before_seconds: u64 },
    Done,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderDismissal {
    Done,
    Cancelled,
}

impl From<ReminderDismissal> for ReminderLifecycle {
    fn from(value: ReminderDismissal) -> Self {
        match value {
            ReminderDismissal::Done => Self::Done,
            ReminderDismissal::Cancelled => Self::Cancelled,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderTargetStatus {
    Visible,
    TemporarilyUnavailable,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderHandledReason {
    Delivered,
    ReminderExpired,
    TargetExpired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReminderHandled {
    pub event_id: NostrEventId,
    pub handled_at_seconds: u64,
    pub reason: ReminderHandledReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReminderDueOutcome {
    NotDue,
    Inactive,
    TargetUnavailable,
    Due,
    AlreadyHandled,
    ReminderExpired,
    TargetExpired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderRetention {
    Retained,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Eq, PartialEq)]
pub struct OwnerReminderReplica {
    scope: ReminderScope,
    reminder_id: ReminderId,
    head: ReminderHead,
    content: ReminderContent,
    lifecycle: ReminderLifecycle,
    expiration_seconds: Option<u64>,
}

impl OwnerReminderReplica {
    pub fn new(
        scope: ReminderScope,
        decrypted_for: PrincipalId,
        reminder_id: ReminderId,
        head: ReminderHead,
        content: ReminderContent,
        lifecycle: ReminderLifecycle,
        expiration_seconds: Option<u64>,
    ) -> Result<Self, ReminderError> {
        if decrypted_for != scope.owner_principal_id {
            return Err(ReminderError::OwnerMismatch);
        }
        validate_replica(
            scope,
            &reminder_id,
            head,
            &content,
            lifecycle,
            expiration_seconds,
        )?;
        Ok(Self {
            scope,
            reminder_id,
            head,
            content,
            lifecycle,
            expiration_seconds,
        })
    }

    pub const fn scope(&self) -> ReminderScope {
        self.scope
    }

    pub const fn head(&self) -> ReminderHead {
        self.head
    }
}

impl fmt::Debug for OwnerReminderReplica {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnerReminderReplica")
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ReminderRecordFields {
    pub scope: ReminderScope,
    pub reminder_id: ReminderId,
    pub head: ReminderHead,
    pub content: ReminderContent,
    pub lifecycle: ReminderLifecycle,
    pub expiration_seconds: Option<u64>,
    pub handled: Option<ReminderHandled>,
}

impl fmt::Debug for ReminderRecordFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReminderRecordFields")
            .field("handled", &self.handled.is_some())
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Reminder {
    fields: ReminderRecordFields,
}

impl Reminder {
    pub fn create(
        scope: ReminderScope,
        decrypted_for: PrincipalId,
        reminder_id: ReminderId,
        head: ReminderHead,
        content: ReminderContent,
        not_before_seconds: u64,
    ) -> Result<Self, ReminderError> {
        let replica = OwnerReminderReplica::new(
            scope,
            decrypted_for,
            reminder_id,
            head,
            content,
            ReminderLifecycle::Pending { not_before_seconds },
            None,
        )?;
        Self::from_replica(replica)
    }

    pub fn from_replica(replica: OwnerReminderReplica) -> Result<Self, ReminderError> {
        Self::from_record(ReminderRecordFields {
            scope: replica.scope,
            reminder_id: replica.reminder_id,
            head: replica.head,
            content: replica.content,
            lifecycle: replica.lifecycle,
            expiration_seconds: replica.expiration_seconds,
            handled: None,
        })
    }

    pub fn from_record(fields: ReminderRecordFields) -> Result<Self, ReminderError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn scope(&self) -> ReminderScope {
        self.fields.scope
    }

    pub fn record(&self, viewer: PrincipalId) -> Result<ReminderRecordFields, ReminderError> {
        self.require_owner(viewer)?;
        Ok(self.fields.clone())
    }

    pub fn lifecycle(&self, viewer: PrincipalId) -> Result<ReminderLifecycle, ReminderError> {
        self.require_owner(viewer)?;
        Ok(self.fields.lifecycle)
    }

    pub fn content(&self, viewer: PrincipalId) -> Result<&ReminderContent, ReminderError> {
        self.require_owner(viewer)?;
        Ok(&self.fields.content)
    }

    pub fn update(
        &mut self,
        owner: PrincipalId,
        head: ReminderHead,
        content: ReminderContent,
        not_before_seconds: u64,
    ) -> Result<ReminderCommandOutcome, ReminderError> {
        self.require_owner(owner)?;
        if !matches!(self.fields.lifecycle, ReminderLifecycle::Pending { .. }) {
            return Err(ReminderError::Terminal);
        }
        let replica = OwnerReminderReplica::new(
            self.fields.scope,
            owner,
            self.fields.reminder_id.clone(),
            head,
            content,
            ReminderLifecycle::Pending { not_before_seconds },
            None,
        )?;
        self.apply_replica(replica)
    }

    pub fn dismiss(
        &mut self,
        owner: PrincipalId,
        head: ReminderHead,
        dismissal: ReminderDismissal,
        expiration_seconds: u64,
    ) -> Result<ReminderCommandOutcome, ReminderError> {
        self.require_owner(owner)?;
        validate_terminal_retention(head.created_at_seconds, expiration_seconds)?;
        let replica = OwnerReminderReplica::new(
            self.fields.scope,
            owner,
            self.fields.reminder_id.clone(),
            head,
            self.fields.content.clone(),
            dismissal.into(),
            Some(expiration_seconds),
        )?;
        if !matches!(self.fields.lifecycle, ReminderLifecycle::Pending { .. }) {
            return if self.matches_replica(&replica) {
                Ok(ReminderCommandOutcome::Unchanged)
            } else {
                Err(ReminderError::Terminal)
            };
        }
        self.apply_replica(replica)
    }

    pub fn merge_owner_replica(
        &mut self,
        replica: OwnerReminderReplica,
    ) -> Result<ReminderCommandOutcome, ReminderError> {
        if replica.scope != self.fields.scope || replica.reminder_id != self.fields.reminder_id {
            return Err(ReminderError::ScopeMismatch);
        }
        self.apply_replica(replica)
    }

    pub fn poll_due(
        &mut self,
        owner: PrincipalId,
        now_seconds: u64,
        target_status: Option<ReminderTargetStatus>,
    ) -> Result<ReminderDueOutcome, ReminderError> {
        self.require_owner(owner)?;
        if now_seconds > MAX_SAFE_TIMESTAMP_SECONDS {
            return Err(ReminderError::InvalidClock);
        }
        let ReminderLifecycle::Pending { not_before_seconds } = self.fields.lifecycle else {
            return Ok(ReminderDueOutcome::Inactive);
        };
        if self
            .fields
            .handled
            .is_some_and(|handled| handled.event_id == self.fields.head.event_id)
        {
            return Ok(ReminderDueOutcome::AlreadyHandled);
        }
        if self
            .fields
            .expiration_seconds
            .is_some_and(|expiration| now_seconds >= expiration)
        {
            self.mark_handled(now_seconds, ReminderHandledReason::ReminderExpired);
            return Ok(ReminderDueOutcome::ReminderExpired);
        }
        if now_seconds < not_before_seconds {
            return Ok(ReminderDueOutcome::NotDue);
        }
        match (self.fields.content.target.is_some(), target_status) {
            (true, None) => return Err(ReminderError::TargetStatusRequired),
            (false, Some(_)) => return Err(ReminderError::UnexpectedTargetStatus),
            (true, Some(ReminderTargetStatus::TemporarilyUnavailable)) => {
                return Ok(ReminderDueOutcome::TargetUnavailable);
            }
            (true, Some(ReminderTargetStatus::Expired)) => {
                self.mark_handled(now_seconds, ReminderHandledReason::TargetExpired);
                return Ok(ReminderDueOutcome::TargetExpired);
            }
            (true, Some(ReminderTargetStatus::Visible)) | (false, None) => {}
        }
        self.mark_handled(now_seconds, ReminderHandledReason::Delivered);
        Ok(ReminderDueOutcome::Due)
    }

    pub fn retention(
        &self,
        viewer: PrincipalId,
        now_seconds: u64,
    ) -> Result<ReminderRetention, ReminderError> {
        self.require_owner(viewer)?;
        Ok(
            if self
                .fields
                .expiration_seconds
                .is_some_and(|expiration| now_seconds >= expiration)
            {
                ReminderRetention::Expired
            } else {
                ReminderRetention::Retained
            },
        )
    }

    fn apply_replica(
        &mut self,
        replica: OwnerReminderReplica,
    ) -> Result<ReminderCommandOutcome, ReminderError> {
        if replica.head.event_id == self.fields.head.event_id {
            return if self.matches_replica(&replica) {
                Ok(ReminderCommandOutcome::Unchanged)
            } else {
                Err(ReminderError::ConflictingReplica)
            };
        }
        if replica.head.replacement_order(self.fields.head) != Ordering::Greater {
            return Ok(ReminderCommandOutcome::Unchanged);
        }
        self.fields.head = replica.head;
        self.fields.content = replica.content;
        self.fields.lifecycle = replica.lifecycle;
        self.fields.expiration_seconds = replica.expiration_seconds;
        self.fields.handled = None;
        Ok(ReminderCommandOutcome::Applied)
    }

    fn matches_replica(&self, replica: &OwnerReminderReplica) -> bool {
        self.fields.scope == replica.scope
            && self.fields.reminder_id == replica.reminder_id
            && self.fields.head == replica.head
            && self.fields.content == replica.content
            && self.fields.lifecycle == replica.lifecycle
            && self.fields.expiration_seconds == replica.expiration_seconds
    }

    fn mark_handled(&mut self, handled_at_seconds: u64, reason: ReminderHandledReason) {
        self.fields.handled = Some(ReminderHandled {
            event_id: self.fields.head.event_id,
            handled_at_seconds,
            reason,
        });
    }

    fn require_owner(&self, viewer: PrincipalId) -> Result<(), ReminderError> {
        if viewer != self.fields.scope.owner_principal_id {
            return Err(ReminderError::OwnerMismatch);
        }
        Ok(())
    }
}

impl fmt::Debug for Reminder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Reminder")
            .field("handled", &self.fields.handled.is_some())
            .field("content", &"<redacted>")
            .finish()
    }
}

fn validate_replica(
    scope: ReminderScope,
    reminder_id: &ReminderId,
    head: ReminderHead,
    content: &ReminderContent,
    lifecycle: ReminderLifecycle,
    expiration_seconds: Option<u64>,
) -> Result<(), ReminderError> {
    if scope.community_id.as_uuid().is_nil()
        || scope.owner_principal_id.as_uuid().is_nil()
        || reminder_id.as_str().is_empty()
        || head.event_id.as_bytes().iter().all(|byte| *byte == 0)
        || head.created_at_seconds > MAX_SAFE_TIMESTAMP_SECONDS
    {
        return Err(ReminderError::InvalidIdentity);
    }
    if expiration_seconds.is_some_and(|expiration| expiration > MAX_SAFE_TIMESTAMP_SECONDS) {
        return Err(ReminderError::InvalidExpiration);
    }
    if let ReminderLifecycle::Pending { not_before_seconds } = lifecycle {
        if !content.is_actionable() {
            return Err(ReminderError::InvalidContent);
        }
        if not_before_seconds > MAX_SAFE_TIMESTAMP_SECONDS {
            return Err(ReminderError::InvalidSchedule);
        }
        if expiration_seconds.is_some_and(|expiration| expiration <= not_before_seconds) {
            return Err(ReminderError::InvalidExpiration);
        }
    }
    Ok(())
}

fn validate_record(fields: &ReminderRecordFields) -> Result<(), ReminderError> {
    validate_replica(
        fields.scope,
        &fields.reminder_id,
        fields.head,
        &fields.content,
        fields.lifecycle,
        fields.expiration_seconds,
    )?;
    if !matches!(fields.lifecycle, ReminderLifecycle::Pending { .. }) && fields.handled.is_some() {
        return Err(ReminderError::InvalidRecord);
    }
    if let Some(handled) = fields.handled {
        let ReminderLifecycle::Pending { not_before_seconds } = fields.lifecycle else {
            return Err(ReminderError::InvalidRecord);
        };
        if handled.event_id != fields.head.event_id
            || handled.handled_at_seconds < not_before_seconds
            || handled.handled_at_seconds > MAX_SAFE_TIMESTAMP_SECONDS
        {
            return Err(ReminderError::InvalidRecord);
        }
        match handled.reason {
            ReminderHandledReason::Delivered
                if fields
                    .expiration_seconds
                    .is_some_and(|expiration| handled.handled_at_seconds >= expiration) =>
            {
                return Err(ReminderError::InvalidRecord);
            }
            ReminderHandledReason::ReminderExpired
                if !fields
                    .expiration_seconds
                    .is_some_and(|expiration| handled.handled_at_seconds >= expiration) =>
            {
                return Err(ReminderError::InvalidRecord);
            }
            ReminderHandledReason::TargetExpired if fields.content.target.is_none() => {
                return Err(ReminderError::InvalidRecord);
            }
            ReminderHandledReason::TargetExpired
                if fields
                    .expiration_seconds
                    .is_some_and(|expiration| handled.handled_at_seconds >= expiration) =>
            {
                return Err(ReminderError::InvalidRecord);
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_terminal_retention(
    completed_at_seconds: u64,
    expiration_seconds: u64,
) -> Result<(), ReminderError> {
    let retention = expiration_seconds
        .checked_sub(completed_at_seconds)
        .ok_or(ReminderError::InvalidExpiration)?;
    if !(MIN_TERMINAL_RETENTION_SECONDS..=MAX_TERMINAL_RETENTION_SECONDS).contains(&retention)
        || expiration_seconds > MAX_SAFE_TIMESTAMP_SECONDS
    {
        return Err(ReminderError::InvalidExpiration);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderError {
    InvalidIdentity,
    InvalidHead,
    InvalidTarget,
    InvalidContent,
    InvalidSchedule,
    InvalidExpiration,
    InvalidRecord,
    OwnerMismatch,
    ScopeMismatch,
    ConflictingReplica,
    Terminal,
    TargetStatusRequired,
    UnexpectedTargetStatus,
    InvalidClock,
}

impl fmt::Display for ReminderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity | Self::InvalidHead | Self::InvalidRecord => {
                formatter.write_str("reminder record is invalid")
            }
            Self::InvalidTarget | Self::InvalidContent => {
                formatter.write_str("reminder content is invalid")
            }
            Self::InvalidSchedule => formatter.write_str("reminder schedule is invalid"),
            Self::InvalidExpiration => formatter.write_str("reminder expiration is invalid"),
            Self::OwnerMismatch | Self::ScopeMismatch => {
                formatter.write_str("reminder owner scope does not match")
            }
            Self::ConflictingReplica => formatter.write_str("reminder replica conflicts"),
            Self::Terminal => formatter.write_str("terminal reminder cannot be reused"),
            Self::TargetStatusRequired | Self::UnexpectedTargetStatus => {
                formatter.write_str("reminder target status does not match content")
            }
            Self::InvalidClock => formatter.write_str("reminder clock is invalid"),
        }
    }
}

impl Error for ReminderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const DUE_SECONDS: u64 = 10_000;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn scope() -> ReminderScope {
        ReminderScope::new(community(1), principal(2))
    }

    fn reminder_id() -> ReminderId {
        ReminderId::new("a3f8c2e1b4d79600e5d2f1a8c3b6094d").expect("valid reminder id")
    }

    fn head(value: u8, created_at_seconds: u64) -> ReminderHead {
        ReminderHead::new(NostrEventId::from_bytes([value; 32]), created_at_seconds)
            .expect("valid reminder head")
    }

    fn note(value: &str) -> ReminderContent {
        ReminderContent::new(None, Some(value.to_owned()))
    }

    fn targeted(value: u8, note: &str) -> ReminderContent {
        ReminderContent::new(
            Some(
                ReminderTarget::new(Some(NostrEventId::from_bytes([value; 32])), None)
                    .expect("valid target"),
            ),
            Some(note.to_owned()),
        )
    }

    fn reminder_fixture(content: ReminderContent) -> Reminder {
        Reminder::create(
            scope(),
            principal(2),
            reminder_id(),
            head(10, 100),
            content,
            DUE_SECONDS,
        )
        .expect("valid reminder")
    }

    #[test]
    fn create_update_and_dismiss_follow_replacement_and_retention_rules() {
        let mut reminder = reminder_fixture(note("review release"));
        assert_eq!(
            reminder.update(principal(2), head(11, 101), note("review later"), 20_000),
            Ok(ReminderCommandOutcome::Applied)
        );
        assert_eq!(
            reminder.update(principal(2), head(12, 99), note("stale"), 30_000),
            Ok(ReminderCommandOutcome::Unchanged)
        );
        assert_eq!(
            reminder
                .content(principal(2))
                .expect("owner content")
                .note(),
            Some("review later")
        );

        let completed_at = 25_000;
        let expiration = completed_at + MIN_TERMINAL_RETENTION_SECONDS;
        assert_eq!(
            reminder.dismiss(
                principal(2),
                head(13, completed_at),
                ReminderDismissal::Done,
                expiration,
            ),
            Ok(ReminderCommandOutcome::Applied)
        );
        assert_eq!(
            reminder.lifecycle(principal(2)),
            Ok(ReminderLifecycle::Done)
        );
        assert_eq!(
            reminder.retention(principal(2), expiration - 1),
            Ok(ReminderRetention::Retained)
        );
        assert_eq!(
            reminder.retention(principal(2), expiration),
            Ok(ReminderRetention::Expired)
        );
        assert_eq!(
            reminder.update(
                principal(2),
                head(14, completed_at + 1),
                note("reopen"),
                40_000
            ),
            Err(ReminderError::Terminal)
        );

        let mut cancelled = reminder_fixture(note("cancel me"));
        let cancelled_at = 30_000;
        assert_eq!(
            cancelled.dismiss(
                principal(2),
                head(15, cancelled_at),
                ReminderDismissal::Cancelled,
                cancelled_at + MAX_TERMINAL_RETENTION_SECONDS,
            ),
            Ok(ReminderCommandOutcome::Applied)
        );
        assert_eq!(
            cancelled.lifecycle(principal(2)),
            Ok(ReminderLifecycle::Cancelled)
        );
    }

    #[test]
    fn due_delivery_uses_the_strict_local_clock_boundary() {
        let mut reminder = reminder_fixture(note("clock boundary"));

        assert_eq!(
            reminder.poll_due(principal(2), DUE_SECONDS - 1, None),
            Ok(ReminderDueOutcome::NotDue)
        );
        assert_eq!(
            reminder.poll_due(principal(2), DUE_SECONDS, None),
            Ok(ReminderDueOutcome::Due)
        );
        assert_eq!(
            reminder.poll_due(principal(2), MAX_SAFE_TIMESTAMP_SECONDS + 1, None),
            Err(ReminderError::InvalidClock)
        );
    }

    #[test]
    fn restart_recovers_an_overdue_reminder_and_deduplicates_delivery() {
        let reminder = reminder_fixture(note("recover after sleep"));
        let before_due = reminder.record(principal(2)).expect("persist reminder");
        let mut restarted = Reminder::from_record(before_due).expect("restart reminder");

        assert_eq!(
            restarted.poll_due(principal(2), DUE_SECONDS + 60, None),
            Ok(ReminderDueOutcome::Due)
        );
        let after_delivery = restarted.record(principal(2)).expect("persist delivery");
        let mut restarted_again = Reminder::from_record(after_delivery).expect("second restart");
        assert_eq!(
            restarted_again.poll_due(principal(2), DUE_SECONDS + 120, None),
            Ok(ReminderDueOutcome::AlreadyHandled)
        );

        let mut corrupt = restarted_again
            .record(principal(2))
            .expect("persist handled reminder");
        corrupt.handled = corrupt.handled.map(|handled| ReminderHandled {
            reason: ReminderHandledReason::ReminderExpired,
            ..handled
        });
        assert_eq!(
            Reminder::from_record(corrupt),
            Err(ReminderError::InvalidRecord)
        );
    }

    #[test]
    fn expired_targets_are_suppressed_but_transient_targets_can_retry() {
        let mut reminder = reminder_fixture(targeted(20, "private target"));
        assert_eq!(
            reminder.poll_due(
                principal(2),
                DUE_SECONDS,
                Some(ReminderTargetStatus::TemporarilyUnavailable),
            ),
            Ok(ReminderDueOutcome::TargetUnavailable)
        );
        assert_eq!(
            reminder.poll_due(
                principal(2),
                DUE_SECONDS + 1,
                Some(ReminderTargetStatus::Expired),
            ),
            Ok(ReminderDueOutcome::TargetExpired)
        );
        assert_eq!(
            reminder.poll_due(
                principal(2),
                DUE_SECONDS + 2,
                Some(ReminderTargetStatus::Visible),
            ),
            Ok(ReminderDueOutcome::AlreadyHandled)
        );

        assert_eq!(
            reminder.update(
                principal(2),
                head(21, 101),
                targeted(22, "replacement target"),
                DUE_SECONDS,
            ),
            Ok(ReminderCommandOutcome::Applied)
        );
        assert_eq!(
            reminder.poll_due(
                principal(2),
                DUE_SECONDS + 3,
                Some(ReminderTargetStatus::Visible),
            ),
            Ok(ReminderDueOutcome::Due)
        );
    }

    #[test]
    fn expired_pending_reminders_never_deliver_or_reenter_retention() {
        let expiration = DUE_SECONDS + 10;
        let replica = OwnerReminderReplica::new(
            scope(),
            principal(2),
            reminder_id(),
            head(23, 100),
            note("expires without delivery"),
            ReminderLifecycle::Pending {
                not_before_seconds: DUE_SECONDS,
            },
            Some(expiration),
        )
        .expect("valid expiring replica");
        let mut reminder = Reminder::from_replica(replica).expect("valid expiring reminder");

        assert_eq!(
            reminder.poll_due(principal(2), expiration, None),
            Ok(ReminderDueOutcome::ReminderExpired)
        );
        assert_eq!(
            reminder.retention(principal(2), expiration),
            Ok(ReminderRetention::Expired)
        );
        assert_eq!(
            reminder.poll_due(principal(2), expiration + 1, None),
            Ok(ReminderDueOutcome::AlreadyHandled)
        );
    }

    #[test]
    fn owner_scope_and_head_order_converge_without_exposing_private_content() {
        let mut reminder = reminder_fixture(note("private reminder note"));
        assert_eq!(
            OwnerReminderReplica::new(
                scope(),
                principal(3),
                reminder_id(),
                head(30, 200),
                note("foreign owner"),
                ReminderLifecycle::Pending {
                    not_before_seconds: 30_000,
                },
                None,
            ),
            Err(ReminderError::OwnerMismatch)
        );

        let foreign_scope = ReminderScope::new(community(4), principal(2));
        let foreign = OwnerReminderReplica::new(
            foreign_scope,
            principal(2),
            reminder_id(),
            head(31, 200),
            note("foreign community"),
            ReminderLifecycle::Pending {
                not_before_seconds: 30_000,
            },
            None,
        )
        .expect("valid foreign replica");
        assert_eq!(
            reminder.merge_owner_replica(foreign),
            Err(ReminderError::ScopeMismatch)
        );

        let later_tie_winner = OwnerReminderReplica::new(
            scope(),
            principal(2),
            reminder_id(),
            head(1, 200),
            note("tie winner"),
            ReminderLifecycle::Pending {
                not_before_seconds: 30_000,
            },
            None,
        )
        .expect("valid winning replica");
        assert_eq!(
            reminder.merge_owner_replica(later_tie_winner),
            Ok(ReminderCommandOutcome::Applied)
        );
        assert_eq!(
            reminder
                .content(principal(2))
                .expect("owner content")
                .note(),
            Some("tie winner")
        );
        assert_eq!(
            reminder.content(principal(3)),
            Err(ReminderError::OwnerMismatch)
        );

        let debug = format!("{reminder:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("private reminder note"));
        assert!(!debug.contains("tie winner"));
        assert!(!debug.contains("30000"));
    }
}
