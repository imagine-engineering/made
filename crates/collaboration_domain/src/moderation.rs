use std::{error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    CommunityId, CommunityMembership, MembershipRole, OperationId, PrincipalId, authorize,
};

const MODERATION_REPORT_SCOPE: &str = "moderation:report";
const MODERATION_MANAGE_SCOPE: &str = "moderation:manage";
const PERSONAL_MUTE_SCOPE: &str = "moderation:mute";
const MAX_REPORT_CONTEXT_BYTES: usize = 4_096;
const MAX_RESTRICTION_TRANSITIONS: usize = 1_000;
const MAX_PERSONAL_MUTE_TRANSITIONS: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModerationCommandSource {
    pub operation_id: OperationId,
    pub occurred_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationReportReason {
    Spam,
    Profanity,
    IllegalContent,
    Nudity,
    Malware,
    Impersonation,
    Other,
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationReportContext(String);

impl ModerationReportContext {
    pub fn new(value: impl Into<String>) -> Result<Self, ModerationError> {
        let value = value.into();
        let value = value.trim();
        if value.is_empty() || value.len() > MAX_REPORT_CONTEXT_BYTES {
            return Err(ModerationError::InvalidReportContext);
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModerationReportContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ModerationReportContext([REDACTED])")
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ModerationReportTarget {
    Event(crate::NostrEventId),
    Principal(PrincipalId),
    BlobSha256([u8; 32]),
}

impl fmt::Debug for ModerationReportTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Event(_) => formatter.write_str("Event([REDACTED])"),
            Self::Principal(_) => formatter.write_str("Principal([REDACTED])"),
            Self::BlobSha256(_) => formatter.write_str("BlobSha256([REDACTED])"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationResolution {
    Dismissed,
    ContentRemoved,
    MemberRemoved,
    TimedOut,
    Banned,
    Escalated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModerationResolutionRecord {
    pub resolution: ModerationResolution,
    pub actor_principal_id: PrincipalId,
    pub source: ModerationCommandSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationReportState {
    Open,
    Resolved(ModerationResolutionRecord),
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationReportRecordFields {
    pub report_id: AggregateId,
    pub community_id: CommunityId,
    pub reporter_principal_id: PrincipalId,
    pub target: ModerationReportTarget,
    pub reason: ModerationReportReason,
    pub private_context: Option<ModerationReportContext>,
    pub filed_source: ModerationCommandSource,
    pub state: ModerationReportState,
    pub version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationReport {
    fields: ModerationReportRecordFields,
}

impl ModerationReport {
    pub fn file(
        report_id: AggregateId,
        community_id: CommunityId,
        target: ModerationReportTarget,
        reason: ModerationReportReason,
        private_context: Option<ModerationReportContext>,
        source: ModerationCommandSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ModerationError> {
        validate_member_authorization(
            community_id,
            MODERATION_REPORT_SCOPE,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
            authorization,
        )?;
        validate_aggregate_id(report_id)?;
        validate_report_target(target)?;
        validate_source(source)?;
        Ok(Self {
            fields: ModerationReportRecordFields {
                report_id,
                community_id,
                reporter_principal_id: authorization_subject(authorization),
                target,
                reason,
                private_context,
                filed_source: source,
                state: ModerationReportState::Open,
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn from_record(fields: ModerationReportRecordFields) -> Result<Self, ModerationError> {
        validate_aggregate_id(fields.report_id)?;
        validate_community_and_principal(fields.community_id, fields.reporter_principal_id)?;
        validate_report_target(fields.target)?;
        validate_source(fields.filed_source)?;
        match fields.state {
            ModerationReportState::Open => {
                if fields.version != AggregateVersion::FIRST {
                    return Err(ModerationError::InvalidRecord);
                }
            }
            ModerationReportState::Resolved(resolution) => {
                validate_principal_id(resolution.actor_principal_id)?;
                validate_source(resolution.source)?;
                if resolution.source.occurred_at_millis < fields.filed_source.occurred_at_millis
                    || resolution.source.operation_id == fields.filed_source.operation_id
                    || fields.version.get() != 2
                {
                    return Err(ModerationError::InvalidRecord);
                }
            }
        }
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &ModerationReportRecordFields {
        &self.fields
    }

    pub fn resolve(
        &mut self,
        expected_version: AggregateVersion,
        resolution: ModerationResolution,
        source: ModerationCommandSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        validate_administration_authorization(self.fields.community_id, authorization)?;
        validate_source(source)?;
        let actor_principal_id = authorization_subject(authorization);
        if let ModerationReportState::Resolved(existing) = self.fields.state {
            if existing
                == (ModerationResolutionRecord {
                    resolution,
                    actor_principal_id,
                    source,
                })
            {
                return Ok(ModerationCommandOutcome::Unchanged);
            }
            if existing.source.operation_id == source.operation_id {
                return Err(ModerationError::ConflictingOperation);
            }
            return Err(ModerationError::InvalidTransition);
        }
        self.require_version(expected_version)?;
        if source.operation_id == self.fields.filed_source.operation_id {
            return Err(ModerationError::ConflictingOperation);
        }
        if source.occurred_at_millis < self.fields.filed_source.occurred_at_millis {
            return Err(ModerationError::InvalidTimestamp);
        }
        self.fields.state = ModerationReportState::Resolved(ModerationResolutionRecord {
            resolution,
            actor_principal_id,
            source,
        });
        self.advance_version()?;
        Ok(ModerationCommandOutcome::Applied)
    }

    fn require_version(&self, expected_version: AggregateVersion) -> Result<(), ModerationError> {
        if self.fields.version != expected_version {
            return Err(ModerationError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn advance_version(&mut self) -> Result<(), ModerationError> {
        self.fields.version = self
            .fields
            .version
            .next()
            .ok_or(ModerationError::VersionExhausted)?;
        Ok(())
    }
}

impl fmt::Debug for ModerationReportRecordFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModerationReportRecordFields")
            .field("community_id", &self.community_id)
            .field("reason", &self.reason)
            .field(
                "state",
                &match self.state {
                    ModerationReportState::Open => "open",
                    ModerationReportState::Resolved(_) => "resolved",
                },
            )
            .field("version", &self.version)
            .field("private_report", &"[REDACTED]")
            .finish()
    }
}

impl fmt::Debug for ModerationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fields.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BanState {
    None,
    Active {
        expires_at_millis: Option<u64>,
        actor_principal_id: PrincipalId,
        source: ModerationCommandSource,
    },
}

impl BanState {
    pub const fn is_active_at(self, now_millis: u64) -> bool {
        match self {
            Self::None => false,
            Self::Active {
                expires_at_millis, ..
            } => match expires_at_millis {
                Some(expires_at_millis) => expires_at_millis > now_millis,
                None => true,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeoutState {
    None,
    Active {
        expires_at_millis: u64,
        actor_principal_id: PrincipalId,
        source: ModerationCommandSource,
    },
}

impl TimeoutState {
    pub const fn is_active_at(self, now_millis: u64) -> bool {
        match self {
            Self::None => false,
            Self::Active {
                expires_at_millis, ..
            } => expires_at_millis > now_millis,
        }
    }

    pub const fn active_expiry_at(self, now_millis: u64) -> Option<u64> {
        match self {
            Self::Active {
                expires_at_millis, ..
            } if expires_at_millis > now_millis => Some(expires_at_millis),
            Self::None | Self::Active { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestrictionTransitionKind {
    ApplyBan { expires_at_millis: Option<u64> },
    LiftBan,
    ApplyTimeout { expires_at_millis: u64 },
    LiftTimeout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RestrictionTransition {
    pub kind: RestrictionTransitionKind,
    pub actor_principal_id: PrincipalId,
    pub source: ModerationCommandSource,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationRestrictionRecordFields {
    pub community_id: CommunityId,
    pub target_principal_id: PrincipalId,
    pub ban: BanState,
    pub timeout: TimeoutState,
    pub transitions: Vec<RestrictionTransition>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModerationTargetSnapshot {
    community_id: CommunityId,
    principal_id: PrincipalId,
    membership: Option<CommunityMembership>,
    current_membership_version: Option<AggregateVersion>,
}

impl ModerationTargetSnapshot {
    pub fn new(
        community_id: CommunityId,
        principal_id: PrincipalId,
        membership: Option<CommunityMembership>,
        current_membership_version: Option<AggregateVersion>,
    ) -> Result<Self, ModerationError> {
        validate_community_and_principal(community_id, principal_id)?;
        match (membership, current_membership_version) {
            (Some(membership), Some(version)) => {
                if membership.community_id != community_id
                    || membership.principal_id != principal_id
                {
                    return Err(ModerationError::TenantMismatch);
                }
                if membership.version != version {
                    return Err(ModerationError::StaleTarget);
                }
            }
            (None, None) => {}
            _ => return Err(ModerationError::StaleTarget),
        }
        Ok(Self {
            community_id,
            principal_id,
            membership,
            current_membership_version,
        })
    }

    pub const fn current_membership_version(self) -> Option<AggregateVersion> {
        self.current_membership_version
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationRestriction {
    fields: ModerationRestrictionRecordFields,
}

impl fmt::Debug for ModerationRestrictionRecordFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModerationRestrictionRecordFields")
            .field("community_id", &self.community_id)
            .field(
                "ban",
                &match self.ban {
                    BanState::None => "none",
                    BanState::Active { .. } => "active",
                },
            )
            .field(
                "timeout",
                &match self.timeout {
                    TimeoutState::None => "none",
                    TimeoutState::Active { .. } => "active",
                },
            )
            .field("transition_count", &self.transitions.len())
            .field("version", &self.version)
            .field("restricted_identity", &"[REDACTED]")
            .finish()
    }
}

impl fmt::Debug for ModerationRestriction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fields.fmt(formatter)
    }
}

impl ModerationRestriction {
    pub fn new(
        community_id: CommunityId,
        target_principal_id: PrincipalId,
    ) -> Result<Self, ModerationError> {
        validate_community_and_principal(community_id, target_principal_id)?;
        Ok(Self {
            fields: ModerationRestrictionRecordFields {
                community_id,
                target_principal_id,
                ban: BanState::None,
                timeout: TimeoutState::None,
                transitions: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn from_record(fields: ModerationRestrictionRecordFields) -> Result<Self, ModerationError> {
        let reconstructed = Self::from_history(
            fields.community_id,
            fields.target_principal_id,
            &fields.transitions,
        )?;
        if reconstructed.fields.ban != fields.ban
            || reconstructed.fields.timeout != fields.timeout
            || reconstructed.fields.version != fields.version
        {
            return Err(ModerationError::InvalidRecord);
        }
        Ok(Self { fields })
    }

    pub fn from_history(
        community_id: CommunityId,
        target_principal_id: PrincipalId,
        transitions: &[RestrictionTransition],
    ) -> Result<Self, ModerationError> {
        Self::from_history_with_observer(community_id, target_principal_id, transitions, |_, _| {
            Ok(())
        })
    }

    pub fn from_history_with_observer(
        community_id: CommunityId,
        target_principal_id: PrincipalId,
        transitions: &[RestrictionTransition],
        mut observer: impl FnMut(
            &ModerationRestrictionRecordFields,
            &RestrictionTransition,
        ) -> Result<(), ModerationError>,
    ) -> Result<Self, ModerationError> {
        validate_community_and_principal(community_id, target_principal_id)?;
        if transitions.len() > MAX_RESTRICTION_TRANSITIONS {
            return Err(ModerationError::TooManyTransitions);
        }
        let mut reconstructed = Self::new(community_id, target_principal_id)?;
        for transition in transitions {
            reconstructed.replay_transition(*transition)?;
            observer(reconstructed.fields(), transition)?;
        }
        Ok(reconstructed)
    }

    pub const fn fields(&self) -> &ModerationRestrictionRecordFields {
        &self.fields
    }

    pub fn apply_ban(
        &mut self,
        expected_version: AggregateVersion,
        expires_at_millis: Option<u64>,
        source: ModerationCommandSource,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let target = ModerationTargetSnapshot::new(
            self.fields.community_id,
            self.fields.target_principal_id,
            Some(target_membership),
            Some(current_target_membership_version),
        )?;
        self.apply_ban_to_subject(
            expected_version,
            expires_at_millis,
            source,
            target,
            authorization,
        )
    }

    pub fn apply_ban_to_subject(
        &mut self,
        expected_version: AggregateVersion,
        expires_at_millis: Option<u64>,
        source: ModerationCommandSource,
        target: ModerationTargetSnapshot,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let actor_principal_id = self.authorize_transition(target, true, authorization)?;
        if expires_at_millis.is_some_and(|expiry| expiry <= source.occurred_at_millis) {
            return Err(ModerationError::InvalidExpiry);
        }
        self.apply_transition(
            expected_version,
            RestrictionTransitionKind::ApplyBan { expires_at_millis },
            actor_principal_id,
            source,
        )
    }

    pub fn lift_ban(
        &mut self,
        expected_version: AggregateVersion,
        source: ModerationCommandSource,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let target = ModerationTargetSnapshot::new(
            self.fields.community_id,
            self.fields.target_principal_id,
            Some(target_membership),
            Some(current_target_membership_version),
        )?;
        self.lift_ban_to_subject(expected_version, source, target, authorization)
    }

    pub fn lift_ban_to_subject(
        &mut self,
        expected_version: AggregateVersion,
        source: ModerationCommandSource,
        target: ModerationTargetSnapshot,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let actor_principal_id = self.authorize_transition(target, false, authorization)?;
        validate_source(source)?;
        self.require_version(expected_version)?;
        if matches!(self.fields.ban, BanState::None) {
            return Ok(ModerationCommandOutcome::Unchanged);
        }
        self.apply_transition(
            expected_version,
            RestrictionTransitionKind::LiftBan,
            actor_principal_id,
            source,
        )
    }

    pub fn apply_timeout(
        &mut self,
        expected_version: AggregateVersion,
        expires_at_millis: u64,
        source: ModerationCommandSource,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let target = ModerationTargetSnapshot::new(
            self.fields.community_id,
            self.fields.target_principal_id,
            Some(target_membership),
            Some(current_target_membership_version),
        )?;
        self.apply_timeout_to_subject(
            expected_version,
            expires_at_millis,
            source,
            target,
            authorization,
        )
    }

    pub fn apply_timeout_to_subject(
        &mut self,
        expected_version: AggregateVersion,
        expires_at_millis: u64,
        source: ModerationCommandSource,
        target: ModerationTargetSnapshot,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let actor_principal_id = self.authorize_transition(target, true, authorization)?;
        if expires_at_millis <= source.occurred_at_millis {
            return Err(ModerationError::InvalidExpiry);
        }
        self.apply_transition(
            expected_version,
            RestrictionTransitionKind::ApplyTimeout { expires_at_millis },
            actor_principal_id,
            source,
        )
    }

    pub fn lift_timeout(
        &mut self,
        expected_version: AggregateVersion,
        source: ModerationCommandSource,
        target_membership: CommunityMembership,
        current_target_membership_version: AggregateVersion,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let target = ModerationTargetSnapshot::new(
            self.fields.community_id,
            self.fields.target_principal_id,
            Some(target_membership),
            Some(current_target_membership_version),
        )?;
        self.lift_timeout_to_subject(expected_version, source, target, authorization)
    }

    pub fn lift_timeout_to_subject(
        &mut self,
        expected_version: AggregateVersion,
        source: ModerationCommandSource,
        target: ModerationTargetSnapshot,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        let actor_principal_id = self.authorize_transition(target, false, authorization)?;
        validate_source(source)?;
        self.require_version(expected_version)?;
        if !self.fields.timeout.is_active_at(source.occurred_at_millis) {
            return Ok(ModerationCommandOutcome::Unchanged);
        }
        self.apply_transition(
            expected_version,
            RestrictionTransitionKind::LiftTimeout,
            actor_principal_id,
            source,
        )
    }

    fn authorize_transition(
        &self,
        target: ModerationTargetSnapshot,
        applying: bool,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<PrincipalId, ModerationError> {
        if authorization.required_scope.as_str() == "messages:write" {
            if authorization.resource.kind != AuthorizationResourceKind::Administration
                || authorization.resource.resource_id.as_uuid()
                    != self.fields.community_id.as_uuid()
                || authorization.resource.channel_id.is_some()
                || authorization.action != AuthorizationAction::Manage
                || authorization.tenant.community_id() != self.fields.community_id
            {
                return Err(ModerationError::AuthorizationShape);
            }
            match authorize(authorization) {
                AuthorizationDecision::Allowed => {}
                AuthorizationDecision::Denied(denial) => {
                    return Err(ModerationError::Unauthorized(denial));
                }
            }
            if authorization.community_membership.is_none_or(|membership| {
                !matches!(
                    membership.role,
                    MembershipRole::Owner | MembershipRole::Admin
                )
            }) {
                return Err(ModerationError::Unauthorized(
                    AuthorizationDenial::InsufficientRole,
                ));
            }
        } else {
            validate_administration_authorization(self.fields.community_id, authorization)?;
        }
        if target.community_id != self.fields.community_id
            || target.principal_id != self.fields.target_principal_id
        {
            return Err(ModerationError::TenantMismatch);
        }
        let actor_role = authorization
            .community_membership
            .map(|membership| membership.role)
            .ok_or(ModerationError::Unauthorized(
                AuthorizationDenial::MissingMembership,
            ))?;
        if applying
            && actor_role == MembershipRole::Admin
            && target.membership.is_some_and(|membership| {
                matches!(
                    membership.role,
                    MembershipRole::Owner | MembershipRole::Admin
                )
            })
        {
            return Err(ModerationError::ProtectedTarget);
        }
        Ok(authorization_subject(authorization))
    }

    fn require_version(&self, expected_version: AggregateVersion) -> Result<(), ModerationError> {
        if self.fields.version != expected_version {
            return Err(ModerationError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn apply_transition(
        &mut self,
        expected_version: AggregateVersion,
        kind: RestrictionTransitionKind,
        actor_principal_id: PrincipalId,
        source: ModerationCommandSource,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        validate_source(source)?;
        if let Some(existing) = self
            .fields
            .transitions
            .iter()
            .find(|transition| transition.source.operation_id == source.operation_id)
        {
            if existing.kind == kind
                && existing.actor_principal_id == actor_principal_id
                && existing.source == source
            {
                return Ok(ModerationCommandOutcome::Unchanged);
            }
            return Err(ModerationError::ConflictingOperation);
        }
        if self.fields.version != expected_version {
            return Err(ModerationError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        if self.fields.transitions.len() >= MAX_RESTRICTION_TRANSITIONS {
            return Err(ModerationError::TooManyTransitions);
        }
        if self.fields.transitions.last().is_some_and(|transition| {
            source.occurred_at_millis < transition.source.occurred_at_millis
        }) {
            return Err(ModerationError::InvalidTimestamp);
        }
        let resulting_version = self
            .fields
            .version
            .next()
            .ok_or(ModerationError::VersionExhausted)?;
        self.set_restriction_state(kind, actor_principal_id, source);
        self.fields.transitions.push(RestrictionTransition {
            kind,
            actor_principal_id,
            source,
            resulting_version,
        });
        self.fields.version = resulting_version;
        Ok(ModerationCommandOutcome::Applied)
    }

    fn replay_transition(
        &mut self,
        transition: RestrictionTransition,
    ) -> Result<(), ModerationError> {
        validate_principal_id(transition.actor_principal_id)?;
        validate_source(transition.source)?;
        if self
            .fields
            .transitions
            .iter()
            .any(|existing| existing.source.operation_id == transition.source.operation_id)
            || self.fields.transitions.last().is_some_and(|previous| {
                transition.source.occurred_at_millis < previous.source.occurred_at_millis
            })
            || !transition.resulting_version.follows(self.fields.version)
        {
            return Err(ModerationError::InvalidRecord);
        }
        match transition.kind {
            RestrictionTransitionKind::ApplyBan {
                expires_at_millis: Some(expiry),
            } if expiry <= transition.source.occurred_at_millis => {
                return Err(ModerationError::InvalidRecord);
            }
            RestrictionTransitionKind::ApplyTimeout { expires_at_millis }
                if expires_at_millis <= transition.source.occurred_at_millis =>
            {
                return Err(ModerationError::InvalidRecord);
            }
            RestrictionTransitionKind::LiftBan if matches!(self.fields.ban, BanState::None) => {
                return Err(ModerationError::InvalidRecord);
            }
            RestrictionTransitionKind::LiftTimeout
                if !self
                    .fields
                    .timeout
                    .is_active_at(transition.source.occurred_at_millis) =>
            {
                return Err(ModerationError::InvalidRecord);
            }
            RestrictionTransitionKind::ApplyBan { .. }
            | RestrictionTransitionKind::LiftBan
            | RestrictionTransitionKind::ApplyTimeout { .. }
            | RestrictionTransitionKind::LiftTimeout => {}
        }
        self.set_restriction_state(
            transition.kind,
            transition.actor_principal_id,
            transition.source,
        );
        self.fields.transitions.push(transition);
        self.fields.version = transition.resulting_version;
        Ok(())
    }

    fn set_restriction_state(
        &mut self,
        kind: RestrictionTransitionKind,
        actor_principal_id: PrincipalId,
        source: ModerationCommandSource,
    ) {
        match kind {
            RestrictionTransitionKind::ApplyBan { expires_at_millis } => {
                self.fields.ban = BanState::Active {
                    expires_at_millis,
                    actor_principal_id,
                    source,
                };
            }
            RestrictionTransitionKind::LiftBan => self.fields.ban = BanState::None,
            RestrictionTransitionKind::ApplyTimeout { expires_at_millis } => {
                self.fields.timeout = TimeoutState::Active {
                    expires_at_millis,
                    actor_principal_id,
                    source,
                };
            }
            RestrictionTransitionKind::LiftTimeout => self.fields.timeout = TimeoutState::None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersonalMuteState {
    Unmuted,
    Muted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersonalMuteTransition {
    pub state: PersonalMuteState,
    pub source: ModerationCommandSource,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct PersonalMuteRecordFields {
    pub community_id: CommunityId,
    pub owner_principal_id: PrincipalId,
    pub muted_principal_id: PrincipalId,
    pub state: PersonalMuteState,
    pub transitions: Vec<PersonalMuteTransition>,
    pub version: AggregateVersion,
}

#[derive(Clone, Eq, PartialEq)]
pub struct PersonalMute {
    fields: PersonalMuteRecordFields,
}

impl fmt::Debug for PersonalMuteRecordFields {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PersonalMuteRecordFields")
            .field("community_id", &self.community_id)
            .field("state", &self.state)
            .field("transition_count", &self.transitions.len())
            .field("version", &self.version)
            .field("personal_identities", &"[REDACTED]")
            .finish()
    }
}

impl fmt::Debug for PersonalMute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fields.fmt(formatter)
    }
}

impl PersonalMute {
    pub fn new(
        community_id: CommunityId,
        muted_principal_id: PrincipalId,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ModerationError> {
        validate_member_authorization(
            community_id,
            PERSONAL_MUTE_SCOPE,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
            authorization,
        )?;
        validate_principal_id(muted_principal_id)?;
        let owner_principal_id = authorization_subject(authorization);
        if owner_principal_id == muted_principal_id {
            return Err(ModerationError::SelfMute);
        }
        Ok(Self {
            fields: PersonalMuteRecordFields {
                community_id,
                owner_principal_id,
                muted_principal_id,
                state: PersonalMuteState::Unmuted,
                transitions: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub const fn fields(&self) -> &PersonalMuteRecordFields {
        &self.fields
    }

    pub fn from_record(fields: PersonalMuteRecordFields) -> Result<Self, ModerationError> {
        validate_community_and_principal(fields.community_id, fields.owner_principal_id)?;
        validate_principal_id(fields.muted_principal_id)?;
        if fields.owner_principal_id == fields.muted_principal_id
            || fields.transitions.len() > MAX_PERSONAL_MUTE_TRANSITIONS
        {
            return Err(ModerationError::InvalidRecord);
        }
        let mut state = PersonalMuteState::Unmuted;
        let mut version = AggregateVersion::FIRST;
        let mut previous_timestamp = None;
        for (index, transition) in fields.transitions.iter().enumerate() {
            validate_source(transition.source)?;
            if transition.state == state
                || previous_timestamp
                    .is_some_and(|timestamp| transition.source.occurred_at_millis < timestamp)
                || fields.transitions[..index]
                    .iter()
                    .any(|existing| existing.source.operation_id == transition.source.operation_id)
                || !transition.resulting_version.follows(version)
            {
                return Err(ModerationError::InvalidRecord);
            }
            state = transition.state;
            version = transition.resulting_version;
            previous_timestamp = Some(transition.source.occurred_at_millis);
        }
        if fields.state != state || fields.version != version {
            return Err(ModerationError::InvalidRecord);
        }
        Ok(Self { fields })
    }

    pub fn set_state(
        &mut self,
        expected_version: AggregateVersion,
        state: PersonalMuteState,
        source: ModerationCommandSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ModerationCommandOutcome, ModerationError> {
        validate_member_authorization(
            self.fields.community_id,
            PERSONAL_MUTE_SCOPE,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
            authorization,
        )?;
        if authorization_subject(authorization) != self.fields.owner_principal_id {
            return Err(ModerationError::PersonalMuteOwnerMismatch);
        }
        validate_source(source)?;
        if let Some(existing) = self
            .fields
            .transitions
            .iter()
            .find(|transition| transition.source.operation_id == source.operation_id)
        {
            if existing.state == state && existing.source == source {
                return Ok(ModerationCommandOutcome::Unchanged);
            }
            return Err(ModerationError::ConflictingOperation);
        }
        if self.fields.version != expected_version {
            return Err(ModerationError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        if self.fields.state == state {
            return Ok(ModerationCommandOutcome::Unchanged);
        }
        if self.fields.transitions.len() >= MAX_PERSONAL_MUTE_TRANSITIONS {
            return Err(ModerationError::TooManyTransitions);
        }
        if self.fields.transitions.last().is_some_and(|transition| {
            source.occurred_at_millis < transition.source.occurred_at_millis
        }) {
            return Err(ModerationError::InvalidTimestamp);
        }
        let resulting_version = self
            .fields
            .version
            .next()
            .ok_or(ModerationError::VersionExhausted)?;
        self.fields.transitions.push(PersonalMuteTransition {
            state,
            source,
            resulting_version,
        });
        self.fields.state = state;
        self.fields.version = resulting_version;
        Ok(ModerationCommandOutcome::Applied)
    }
}

fn validate_member_authorization(
    community_id: CommunityId,
    required_scope: &str,
    action: AuthorizationAction,
    resource_kind: AuthorizationResourceKind,
    request: &AuthorizationRequest<'_>,
) -> Result<(), ModerationError> {
    if request.required_scope.as_str() != required_scope
        || request.action != action
        || request.resource.community_id != community_id
        || request.resource.kind != resource_kind
        || request.resource.resource_id != AggregateId::from_uuid(community_id.as_uuid())
        || request.resource.channel_id.is_some()
        || request.resource.owner_principal_id.is_some()
    {
        return Err(ModerationError::AuthorizationShape);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(ModerationError::Unauthorized(denial)),
    }
}

fn validate_administration_authorization(
    community_id: CommunityId,
    request: &AuthorizationRequest<'_>,
) -> Result<(), ModerationError> {
    validate_member_authorization(
        community_id,
        MODERATION_MANAGE_SCOPE,
        AuthorizationAction::Manage,
        AuthorizationResourceKind::Administration,
        request,
    )?;
    let role = request
        .community_membership
        .map(|membership| membership.role)
        .ok_or(ModerationError::Unauthorized(
            AuthorizationDenial::MissingMembership,
        ))?;
    if !matches!(role, MembershipRole::Owner | MembershipRole::Admin) {
        return Err(ModerationError::Unauthorized(
            AuthorizationDenial::InsufficientRole,
        ));
    }
    Ok(())
}

fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

fn validate_source(source: ModerationCommandSource) -> Result<(), ModerationError> {
    if source.operation_id.as_uuid().is_nil() {
        return Err(ModerationError::InvalidOperationId);
    }
    Ok(())
}

fn validate_aggregate_id(value: AggregateId) -> Result<(), ModerationError> {
    if value.as_uuid().is_nil() {
        return Err(ModerationError::InvalidIdentity);
    }
    Ok(())
}

fn validate_principal_id(value: PrincipalId) -> Result<(), ModerationError> {
    if value.as_uuid().is_nil() {
        return Err(ModerationError::InvalidIdentity);
    }
    Ok(())
}

fn validate_community_and_principal(
    community_id: CommunityId,
    principal_id: PrincipalId,
) -> Result<(), ModerationError> {
    if community_id.as_uuid().is_nil() || principal_id.as_uuid().is_nil() {
        return Err(ModerationError::InvalidIdentity);
    }
    Ok(())
}

fn validate_report_target(target: ModerationReportTarget) -> Result<(), ModerationError> {
    match target {
        ModerationReportTarget::Event(event_id) if event_id.as_bytes() == &[0; 32] => {
            Err(ModerationError::InvalidIdentity)
        }
        ModerationReportTarget::Principal(principal_id) => validate_principal_id(principal_id),
        ModerationReportTarget::BlobSha256(digest) if digest == [0; 32] => {
            Err(ModerationError::InvalidIdentity)
        }
        ModerationReportTarget::Event(_) | ModerationReportTarget::BlobSha256(_) => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationError {
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    TenantMismatch,
    InvalidIdentity,
    InvalidOperationId,
    InvalidReportContext,
    InvalidTimestamp,
    InvalidExpiry,
    InvalidTransition,
    ConflictingOperation,
    ProtectedTarget,
    SelfRestriction,
    SelfMute,
    PersonalMuteOwnerMismatch,
    StaleTarget,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    TooManyTransitions,
    VersionExhausted,
    InvalidRecord,
}

impl fmt::Display for ModerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorizationShape | Self::Unauthorized(_) | Self::TenantMismatch => {
                formatter.write_str("moderation command is not authorized")
            }
            Self::InvalidIdentity => formatter.write_str("moderation identity is invalid"),
            Self::InvalidOperationId => formatter.write_str("moderation operation ID is invalid"),
            Self::InvalidReportContext => {
                formatter.write_str("moderation report context is invalid")
            }
            Self::InvalidTimestamp => formatter.write_str("moderation timestamp is invalid"),
            Self::InvalidExpiry => formatter.write_str("moderation expiry is invalid"),
            Self::InvalidTransition => formatter.write_str("moderation transition is invalid"),
            Self::ConflictingOperation => formatter.write_str("moderation operation conflicts"),
            Self::ProtectedTarget => formatter.write_str("moderation target is protected"),
            Self::SelfRestriction => {
                formatter.write_str("moderation self-restriction is forbidden")
            }
            Self::SelfMute => formatter.write_str("personal self-mute is forbidden"),
            Self::PersonalMuteOwnerMismatch => {
                formatter.write_str("personal mute owner does not match")
            }
            Self::StaleTarget => formatter.write_str("moderation target membership is stale"),
            Self::StaleVersion { .. } => formatter.write_str("moderation version is stale"),
            Self::TooManyTransitions => {
                formatter.write_str("moderation transition history is full")
            }
            Self::VersionExhausted => formatter.write_str("moderation version is exhausted"),
            Self::InvalidRecord => formatter.write_str("moderation record is invalid"),
        }
    }
}

impl Error for ModerationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, MembershipStatus,
        PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "moderation-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn scope(value: &str) -> AuthorizationScope {
        AuthorizationScope::new(value).expect("scope")
    }

    fn principal(
        community_id: CommunityId,
        principal_id: PrincipalId,
        required_scope: &AuthorizationScope,
    ) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id,
            community_id,
            ServiceAccountId::new(principal_id.as_uuid().as_u128() as u64),
            PrincipalScopes::new([required_scope.clone()]).expect("scopes"),
        )
    }

    fn membership(
        community_id: CommunityId,
        principal_id: PrincipalId,
        role: MembershipRole,
        version: AggregateVersion,
    ) -> CommunityMembership {
        CommunityMembership {
            community_id,
            principal_id,
            role,
            status: MembershipStatus::Active,
            version,
        }
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        required_scope: &'a AuthorizationScope,
        role: MembershipRole,
        membership_version: AggregateVersion,
        current_membership_version: AggregateVersion,
        manage: bool,
    ) -> AuthorizationRequest<'a> {
        let community_id = tenant.community_id();
        AuthorizationRequest {
            tenant,
            principal,
            required_scope,
            action: if manage {
                AuthorizationAction::Manage
            } else {
                AuthorizationAction::Write
            },
            resource: AuthorizationResource {
                community_id,
                kind: if manage {
                    AuthorizationResourceKind::Administration
                } else {
                    AuthorizationResourceKind::Community
                },
                resource_id: AggregateId::from_uuid(community_id.as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version,
            community_membership: Some(membership(
                community_id,
                principal.principal_id(),
                role,
                membership_version,
            )),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn source(value: u128, occurred_at_millis: u64) -> ModerationCommandSource {
        ModerationCommandSource {
            operation_id: OperationId::from_uuid(Uuid::from_u128(value)),
            occurred_at_millis,
        }
    }

    #[test]
    fn member_files_private_report_and_admin_resolves_it_once() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let report_scope = scope(MODERATION_REPORT_SCOPE);
        let reporter = principal(community_id, principal_id(10), &report_scope);
        let report_request = request(
            &tenant,
            &reporter,
            &report_scope,
            MembershipRole::Member,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            false,
        );
        let mut report = ModerationReport::file(
            AggregateId::from_uuid(Uuid::from_u128(30)),
            community_id,
            ModerationReportTarget::Event(crate::NostrEventId::from_bytes([4; 32])),
            ModerationReportReason::Spam,
            Some(ModerationReportContext::new("private evidence").expect("context")),
            source(40, 100),
            &report_request,
        )
        .expect("report");
        assert_eq!(report.fields.reporter_principal_id, principal_id(10));
        assert_eq!(report.fields.state, ModerationReportState::Open);
        let report_debug = format!("{report:?}");
        assert!(report_debug.contains("[REDACTED]"));
        assert!(!report_debug.contains(&principal_id(10).as_uuid().to_string()));
        assert!(!report_debug.contains("[4, 4, 4"));
        assert!(!report_debug.contains("private evidence"));

        let manage_scope = scope(MODERATION_MANAGE_SCOPE);
        let administrator = principal(community_id, principal_id(20), &manage_scope);
        let manage_request = request(
            &tenant,
            &administrator,
            &manage_scope,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        let resolution_source = source(41, 110);
        assert_eq!(
            report.resolve(
                AggregateVersion::FIRST,
                ModerationResolution::ContentRemoved,
                resolution_source,
                &manage_request,
            ),
            Ok(ModerationCommandOutcome::Applied)
        );
        assert_eq!(
            report.resolve(
                AggregateVersion::FIRST,
                ModerationResolution::ContentRemoved,
                resolution_source,
                &manage_request,
            ),
            Ok(ModerationCommandOutcome::Unchanged)
        );
        assert!(matches!(
            report.fields.state,
            ModerationReportState::Resolved(ModerationResolutionRecord {
                resolution: ModerationResolution::ContentRemoved,
                actor_principal_id,
                ..
            }) if actor_principal_id == principal_id(20)
        ));
        assert_eq!(
            ModerationReport::from_record(report.fields().clone()),
            Ok(report)
        );
    }

    #[test]
    fn timeout_expires_without_hiding_an_independent_ban() {
        let community_id = community(1);
        let target_principal_id = principal_id(30);
        let target = membership(
            community_id,
            target_principal_id,
            MembershipRole::Member,
            AggregateVersion::FIRST,
        );
        let tenant = tenant(community_id);
        let manage_scope = scope(MODERATION_MANAGE_SCOPE);
        let owner = principal(community_id, principal_id(20), &manage_scope);
        let manage_request = request(
            &tenant,
            &owner,
            &manage_scope,
            MembershipRole::Owner,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        let mut restriction =
            ModerationRestriction::new(community_id, target_principal_id).expect("restriction");
        assert_eq!(
            restriction.apply_timeout(
                AggregateVersion::FIRST,
                200,
                source(50, 100),
                target,
                AggregateVersion::FIRST,
                &manage_request,
            ),
            Ok(ModerationCommandOutcome::Applied)
        );
        assert_eq!(
            restriction.apply_ban(
                AggregateVersion::new(2).expect("version"),
                None,
                source(51, 110),
                target,
                AggregateVersion::FIRST,
                &manage_request,
            ),
            Ok(ModerationCommandOutcome::Applied)
        );
        assert!(restriction.fields.timeout.is_active_at(199));
        assert!(!restriction.fields.timeout.is_active_at(200));
        assert!(restriction.fields.ban.is_active_at(200));
        assert_eq!(restriction.fields.timeout.active_expiry_at(199), Some(200));
        assert_eq!(restriction.fields.timeout.active_expiry_at(200), None);
        assert_eq!(
            ModerationRestriction::from_record(restriction.fields.clone()),
            Ok(restriction)
        );
    }

    #[test]
    fn administrator_cannot_restrict_an_owner_or_peer_administrator() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(MODERATION_MANAGE_SCOPE);
        let administrator = principal(community_id, principal_id(20), &manage_scope);
        let manage_request = request(
            &tenant,
            &administrator,
            &manage_scope,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        for (target_principal_id, role) in [
            (principal_id(30), MembershipRole::Owner),
            (principal_id(31), MembershipRole::Admin),
        ] {
            let mut restriction =
                ModerationRestriction::new(community_id, target_principal_id).expect("restriction");
            assert_eq!(
                restriction.apply_ban(
                    AggregateVersion::FIRST,
                    None,
                    source(60 + target_principal_id.as_uuid().as_u128(), 100),
                    membership(
                        community_id,
                        target_principal_id,
                        role,
                        AggregateVersion::FIRST,
                    ),
                    AggregateVersion::FIRST,
                    &manage_request,
                ),
                Err(ModerationError::ProtectedTarget)
            );
        }
    }

    #[test]
    fn owner_restrictions_cover_absent_self_and_protected_targets_and_admin_lifts() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(MODERATION_MANAGE_SCOPE);
        let owner = principal(community_id, principal_id(20), &manage_scope);
        let owner_request = request(
            &tenant,
            &owner,
            &manage_scope,
            MembershipRole::Owner,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        for (target_principal_id, target_role) in [
            (principal_id(20), Some(MembershipRole::Owner)),
            (principal_id(30), Some(MembershipRole::Owner)),
            (principal_id(31), Some(MembershipRole::Admin)),
            (principal_id(32), None),
        ] {
            let target_membership = target_role.map(|role| {
                membership(
                    community_id,
                    target_principal_id,
                    role,
                    AggregateVersion::FIRST,
                )
            });
            let target = ModerationTargetSnapshot::new(
                community_id,
                target_principal_id,
                target_membership,
                target_membership.map(|membership| membership.version),
            )
            .expect("exact optional target");
            assert_eq!(
                target.current_membership_version(),
                target_membership.map(|membership| membership.version)
            );
            let mut restriction =
                ModerationRestriction::new(community_id, target_principal_id).expect("restriction");
            assert_eq!(
                restriction.apply_ban_to_subject(
                    AggregateVersion::FIRST,
                    Some(150),
                    source(80 + target_principal_id.as_uuid().as_u128(), 100),
                    target,
                    &owner_request,
                ),
                Ok(ModerationCommandOutcome::Applied),
                "owner may restrict absent, self and protected targets"
            );
            let admin = principal(community_id, principal_id(40), &manage_scope);
            let admin_request = request(
                &tenant,
                &admin,
                &manage_scope,
                MembershipRole::Admin,
                AggregateVersion::FIRST,
                AggregateVersion::FIRST,
                true,
            );
            assert_eq!(
                restriction.lift_ban_to_subject(
                    AggregateVersion::new(2).expect("version"),
                    source(130 + target_principal_id.as_uuid().as_u128(), 200),
                    target,
                    &admin_request,
                ),
                Ok(ModerationCommandOutcome::Applied),
                "admin may clear stored ban after expiry without target-role guard"
            );
            assert_eq!(restriction.fields().ban, BanState::None);
            assert_eq!(
                ModerationRestriction::from_record(restriction.fields().clone()),
                Ok(restriction)
            );
        }
        let absent = ModerationTargetSnapshot::new(community_id, principal_id(90), None, None)
            .expect("absent");
        let mut restriction =
            ModerationRestriction::new(community_id, principal_id(90)).expect("restriction");
        let admin = principal(community_id, principal_id(40), &manage_scope);
        let admin_request = request(
            &tenant,
            &admin,
            &manage_scope,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        assert_eq!(
            restriction.apply_timeout_to_subject(
                AggregateVersion::FIRST,
                300,
                source(200, 100),
                absent,
                &admin_request,
            ),
            Ok(ModerationCommandOutcome::Applied)
        );
        assert_eq!(
            ModerationTargetSnapshot::new(
                community_id,
                principal_id(90),
                None,
                Some(AggregateVersion::FIRST)
            ),
            Err(ModerationError::StaleTarget)
        );
        assert_eq!(
            ModerationTargetSnapshot::new(
                community_id,
                principal_id(90),
                Some(membership(
                    community_id,
                    principal_id(91),
                    MembershipRole::Member,
                    AggregateVersion::FIRST
                )),
                Some(AggregateVersion::FIRST)
            ),
            Err(ModerationError::TenantMismatch)
        );
    }

    #[test]
    fn personal_mute_is_owned_by_the_member_and_does_not_create_a_restriction() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let mute_scope = scope(PERSONAL_MUTE_SCOPE);
        let member = principal(community_id, principal_id(10), &mute_scope);
        let mute_request = request(
            &tenant,
            &member,
            &mute_scope,
            MembershipRole::Member,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            false,
        );
        let mut personal_mute = PersonalMute::new(community_id, principal_id(30), &mute_request)
            .expect("personal mute");
        assert_eq!(
            personal_mute.set_state(
                AggregateVersion::FIRST,
                PersonalMuteState::Muted,
                source(70, 100),
                &mute_request,
            ),
            Ok(ModerationCommandOutcome::Applied)
        );
        assert_eq!(personal_mute.fields.state, PersonalMuteState::Muted);
        assert_eq!(
            PersonalMute::from_record(personal_mute.fields.clone()),
            Ok(personal_mute)
        );

        let separate_restriction =
            ModerationRestriction::new(community_id, principal_id(30)).expect("restriction");
        assert_eq!(separate_restriction.fields.ban, BanState::None);
        assert_eq!(separate_restriction.fields.timeout, TimeoutState::None);
    }

    #[test]
    fn stale_actor_and_stale_target_memberships_fail_closed() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let manage_scope = scope(MODERATION_MANAGE_SCOPE);
        let administrator = principal(community_id, principal_id(20), &manage_scope);
        let stale_actor_request = request(
            &tenant,
            &administrator,
            &manage_scope,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
            AggregateVersion::new(2).expect("version"),
            true,
        );
        let target_principal_id = principal_id(30);
        let mut restriction =
            ModerationRestriction::new(community_id, target_principal_id).expect("restriction");
        let target = membership(
            community_id,
            target_principal_id,
            MembershipRole::Member,
            AggregateVersion::FIRST,
        );
        assert_eq!(
            restriction.apply_timeout(
                AggregateVersion::FIRST,
                200,
                source(80, 100),
                target,
                AggregateVersion::FIRST,
                &stale_actor_request,
            ),
            Err(ModerationError::Unauthorized(
                AuthorizationDenial::StaleMembership
            ))
        );

        let current_actor_request = request(
            &tenant,
            &administrator,
            &manage_scope,
            MembershipRole::Admin,
            AggregateVersion::FIRST,
            AggregateVersion::FIRST,
            true,
        );
        assert_eq!(
            restriction.apply_timeout(
                AggregateVersion::FIRST,
                200,
                source(81, 100),
                target,
                AggregateVersion::new(2).expect("version"),
                &current_actor_request,
            ),
            Err(ModerationError::StaleTarget)
        );
        assert!(restriction.fields.transitions.is_empty());
    }
    #[test]
    fn restriction_history_hydration_preserves_independent_state_and_rejects_gaps() {
        let community = community(1);
        let target = principal_id(30);
        let transitions = [
            RestrictionTransition {
                kind: RestrictionTransitionKind::ApplyBan {
                    expires_at_millis: None,
                },
                actor_principal_id: principal_id(20),
                source: source(90, 100),
                resulting_version: AggregateVersion::new(2).expect("version"),
            },
            RestrictionTransition {
                kind: RestrictionTransitionKind::ApplyTimeout {
                    expires_at_millis: 300,
                },
                actor_principal_id: principal_id(21),
                source: source(91, 200),
                resulting_version: AggregateVersion::new(3).expect("version"),
            },
        ];
        let hydrated =
            ModerationRestriction::from_history(community, target, &transitions).expect("history");
        assert_eq!(
            ModerationRestriction::from_record(hydrated.fields().clone()),
            Ok(hydrated.clone())
        );
        assert!(hydrated.fields().ban.is_active_at(301));
        assert!(!hydrated.fields().timeout.is_active_at(301));
        assert!(
            matches!(hydrated.fields().ban, BanState::Active { actor_principal_id, .. } if actor_principal_id == principal_id(20))
        );
        assert!(
            matches!(hydrated.fields().timeout, TimeoutState::Active { actor_principal_id, .. } if actor_principal_id == principal_id(21))
        );
        assert_eq!(
            ModerationRestriction::from_history(community, target, &transitions[1..]),
            Err(ModerationError::InvalidRecord)
        );
        let mut reused = transitions;
        reused[1].source.operation_id = reused[0].source.operation_id;
        assert_eq!(
            ModerationRestriction::from_history(community, target, &reused),
            Err(ModerationError::InvalidRecord)
        );
        let mut regressed = transitions;
        regressed[1].source.occurred_at_millis = 99;
        assert_eq!(
            ModerationRestriction::from_history(community, target, &regressed),
            Err(ModerationError::InvalidRecord)
        );
        let mut contradictory = hydrated.fields().clone();
        contradictory.ban = BanState::None;
        assert_eq!(
            ModerationRestriction::from_record(contradictory),
            Err(ModerationError::InvalidRecord)
        );
    }
    #[test]
    fn restriction_history_observer_cannot_bypass_replay_or_hide_its_failure() {
        let transitions = [RestrictionTransition {
            kind: RestrictionTransitionKind::ApplyBan {
                expires_at_millis: None,
            },
            actor_principal_id: principal_id(20),
            source: source(95, 100),
            resulting_version: AggregateVersion::new(2).expect("version"),
        }];
        let mut observed = 0;
        let hydrated = ModerationRestriction::from_history_with_observer(
            community(1),
            principal_id(30),
            &transitions,
            |fields, transition| {
                observed += 1;
                assert_eq!(fields.version, transition.resulting_version);
                assert_eq!(fields.transitions.last(), Some(transition));
                assert!(fields.ban.is_active_at(100));
                Ok(())
            },
        )
        .expect("observed history");
        assert_eq!(observed, 1);
        assert_eq!(hydrated.fields().version.get(), 2);
        assert_eq!(
            ModerationRestriction::from_history_with_observer(
                community(1),
                principal_id(30),
                &transitions,
                |_, _| Err(ModerationError::InvalidRecord)
            ),
            Err(ModerationError::InvalidRecord)
        );
        let mut invalid = transitions;
        invalid[0].resulting_version = AggregateVersion::new(4).expect("version");
        observed = 0;
        assert_eq!(
            ModerationRestriction::from_history_with_observer(
                community(1),
                principal_id(30),
                &invalid,
                |_, _| {
                    observed += 1;
                    Ok(())
                }
            ),
            Err(ModerationError::InvalidRecord)
        );
        assert_eq!(observed, 0);
    }
}
