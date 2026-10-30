use std::{collections::BTreeSet, error::Error, fmt};

use crate::message::{
    authorization_subject, authorize_message_command, validate_authenticated_author,
};
use crate::{
    AggregateId, AggregateVersion, AuthorizationAction, AuthorizationRequest, Channel,
    ChannelLifecycleState, CommunityId, MessageAuthor, MessageContent, MessageError, MessageSource,
    NostrEventId, OperationId, PrincipalId,
};

const MAX_CLOCK_SKEW_MILLIS: u64 = 300_000;
const MAX_SCHEDULE_HORIZON_MILLIS: u64 = 31_536_000_000;
const MAX_LEASE_MILLIS: u64 = 300_000;
const MAX_EXECUTION_ATTEMPTS: u16 = 32;
const MAX_AUTHORED_MUTATIONS: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DueClaim {
    pub claim_id: OperationId,
    pub claimed_at_millis: u64,
    pub lease_expires_at_millis: u64,
    pub attempt: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduledMessageState {
    Pending,
    Claimed(DueClaim),
    Cancelled,
    Executed {
        claim_id: OperationId,
        execution_attempt: u16,
        published_message_id: AggregateId,
        published_event_id: NostrEventId,
        executed_at_millis: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleMutationKind {
    Update {
        content: MessageContent,
        scheduled_for_millis: u64,
    },
    Cancel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduleMutation {
    pub source: MessageSource,
    pub actor_principal_id: PrincipalId,
    pub kind: ScheduleMutationKind,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledMessageRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub schedule_id: AggregateId,
    pub author: MessageAuthor,
    pub initial_content: MessageContent,
    pub initial_scheduled_for_millis: u64,
    pub content: MessageContent,
    pub scheduled_for_millis: u64,
    pub source: MessageSource,
    pub authored_mutations: Vec<ScheduleMutation>,
    pub state: ScheduledMessageState,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledMessageCreateFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub schedule_id: AggregateId,
    pub author: MessageAuthor,
    pub content: MessageContent,
    pub scheduled_for_millis: u64,
    pub source: MessageSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduleCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledMessage {
    fields: ScheduledMessageRecordFields,
}

impl ScheduledMessage {
    pub fn create(
        fields: ScheduledMessageCreateFields,
        channel: &Channel,
        now_millis: u64,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ScheduleError> {
        validate_identity(fields.schedule_id, fields.channel_id, fields.source)?;
        fields.author.validate().map_err(ScheduleError::Message)?;
        validate_schedule_time(fields.scheduled_for_millis, now_millis)?;
        if channel.fields().community_id != fields.community_id
            || channel.fields().channel_id != fields.channel_id
        {
            return Err(ScheduleError::ChannelMismatch);
        }
        if channel.fields().lifecycle_state != ChannelLifecycleState::Active {
            return Err(ScheduleError::ChannelUnavailable);
        }
        authorize_schedule(
            authorization,
            fields.community_id,
            fields.channel_id,
            fields.schedule_id,
            fields.author,
        )?;
        validate_authenticated_author(fields.author, authorization)
            .map_err(ScheduleError::Message)?;
        Ok(Self {
            fields: ScheduledMessageRecordFields {
                community_id: fields.community_id,
                channel_id: fields.channel_id,
                schedule_id: fields.schedule_id,
                author: fields.author,
                initial_content: fields.content.clone(),
                initial_scheduled_for_millis: fields.scheduled_for_millis,
                content: fields.content,
                scheduled_for_millis: fields.scheduled_for_millis,
                source: fields.source,
                authored_mutations: Vec::new(),
                state: ScheduledMessageState::Pending,
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub fn from_record(fields: ScheduledMessageRecordFields) -> Result<Self, ScheduleError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &ScheduledMessageRecordFields {
        &self.fields
    }

    pub fn update(
        &mut self,
        expected_version: AggregateVersion,
        content: MessageContent,
        scheduled_for_millis: u64,
        source: MessageSource,
        now_millis: u64,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ScheduleCommandOutcome, ScheduleError> {
        self.authorize_author(authorization)?;
        source
            .validate()
            .map_err(|_| ScheduleError::InvalidSource)?;
        if self.has_source(source.event_id) {
            return Ok(ScheduleCommandOutcome::Unchanged);
        }
        self.require_pending()?;
        validate_schedule_time(scheduled_for_millis, now_millis)?;
        self.require_authored_mutation(source, expected_version)?;
        let next_version = self.next_version()?;
        self.fields.content = content.clone();
        self.fields.scheduled_for_millis = scheduled_for_millis;
        self.fields.authored_mutations.push(ScheduleMutation {
            source,
            actor_principal_id: authorization_subject(authorization),
            kind: ScheduleMutationKind::Update {
                content,
                scheduled_for_millis,
            },
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(ScheduleCommandOutcome::Applied)
    }

    pub fn cancel(
        &mut self,
        expected_version: AggregateVersion,
        source: MessageSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ScheduleCommandOutcome, ScheduleError> {
        self.authorize_author(authorization)?;
        source
            .validate()
            .map_err(|_| ScheduleError::InvalidSource)?;
        if self.has_source(source.event_id) {
            return Ok(ScheduleCommandOutcome::Unchanged);
        }
        if self.fields.state == ScheduledMessageState::Cancelled {
            return Ok(ScheduleCommandOutcome::Unchanged);
        }
        self.require_pending()?;
        self.require_authored_mutation(source, expected_version)?;
        let next_version = self.next_version()?;
        self.fields.state = ScheduledMessageState::Cancelled;
        self.fields.authored_mutations.push(ScheduleMutation {
            source,
            actor_principal_id: authorization_subject(authorization),
            kind: ScheduleMutationKind::Cancel,
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(ScheduleCommandOutcome::Applied)
    }

    pub fn claim_due(
        &mut self,
        expected_version: AggregateVersion,
        claim_id: OperationId,
        now_millis: u64,
        lease_millis: u64,
    ) -> Result<ScheduleCommandOutcome, ScheduleError> {
        if claim_id.as_uuid().is_nil() || lease_millis == 0 || lease_millis > MAX_LEASE_MILLIS {
            return Err(ScheduleError::InvalidClaim);
        }
        match self.fields.state {
            ScheduledMessageState::Cancelled | ScheduledMessageState::Executed { .. } => {
                return Ok(ScheduleCommandOutcome::Unchanged);
            }
            ScheduledMessageState::Claimed(claim) if claim.claim_id == claim_id => {
                return Ok(ScheduleCommandOutcome::Unchanged);
            }
            ScheduledMessageState::Claimed(claim) if claim.lease_expires_at_millis > now_millis => {
                return Err(ScheduleError::LeaseHeld);
            }
            _ => {}
        }
        if now_millis.saturating_add(MAX_CLOCK_SKEW_MILLIS) < self.fields.scheduled_for_millis {
            return Err(ScheduleError::NotDue);
        }
        self.require_version(expected_version)?;
        let attempt = match self.fields.state {
            ScheduledMessageState::Claimed(claim) => claim
                .attempt
                .checked_add(1)
                .ok_or(ScheduleError::RecoveryExhausted)?,
            ScheduledMessageState::Pending => 1,
            _ => return Ok(ScheduleCommandOutcome::Unchanged),
        };
        if attempt > MAX_EXECUTION_ATTEMPTS {
            return Err(ScheduleError::RecoveryExhausted);
        }
        let lease_expires_at_millis = now_millis
            .checked_add(lease_millis)
            .ok_or(ScheduleError::InvalidClaim)?;
        self.fields.state = ScheduledMessageState::Claimed(DueClaim {
            claim_id,
            claimed_at_millis: now_millis,
            lease_expires_at_millis,
            attempt,
        });
        self.fields.version = self.next_version()?;
        Ok(ScheduleCommandOutcome::Applied)
    }

    pub fn complete_due(
        &mut self,
        expected_version: AggregateVersion,
        claim_id: OperationId,
        published_message_id: AggregateId,
        published_event_id: NostrEventId,
        now_millis: u64,
    ) -> Result<ScheduleCommandOutcome, ScheduleError> {
        if let ScheduledMessageState::Executed {
            claim_id: existing_claim_id,
            published_message_id: existing_message_id,
            published_event_id: existing_event_id,
            ..
        } = self.fields.state
        {
            return if existing_claim_id == claim_id
                && existing_message_id == published_message_id
                && existing_event_id == published_event_id
            {
                Ok(ScheduleCommandOutcome::Unchanged)
            } else {
                Err(ScheduleError::AlreadyExecuted)
            };
        }
        let ScheduledMessageState::Claimed(claim) = self.fields.state else {
            return Err(ScheduleError::ClaimMismatch);
        };
        if claim.claim_id != claim_id
            || published_message_id.as_uuid().is_nil()
            || published_event_id.as_bytes().iter().all(|byte| *byte == 0)
        {
            return Err(ScheduleError::ClaimMismatch);
        }
        if claim.lease_expires_at_millis <= now_millis {
            return Err(ScheduleError::LeaseExpired);
        }
        self.require_version(expected_version)?;
        self.fields.state = ScheduledMessageState::Executed {
            claim_id,
            execution_attempt: claim.attempt,
            published_message_id,
            published_event_id,
            executed_at_millis: now_millis,
        };
        self.fields.version = self.next_version()?;
        Ok(ScheduleCommandOutcome::Applied)
    }

    fn authorize_author(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<(), ScheduleError> {
        authorize_schedule(
            authorization,
            self.fields.community_id,
            self.fields.channel_id,
            self.fields.schedule_id,
            self.fields.author,
        )
    }

    fn require_pending(&self) -> Result<(), ScheduleError> {
        match self.fields.state {
            ScheduledMessageState::Pending => Ok(()),
            ScheduledMessageState::Claimed(_) => Err(ScheduleError::InFlight),
            ScheduledMessageState::Cancelled => Err(ScheduleError::Cancelled),
            ScheduledMessageState::Executed { .. } => Err(ScheduleError::AlreadyExecuted),
        }
    }

    fn require_authored_mutation(
        &self,
        source: MessageSource,
        expected_version: AggregateVersion,
    ) -> Result<(), ScheduleError> {
        if self.fields.authored_mutations.len() >= MAX_AUTHORED_MUTATIONS {
            return Err(ScheduleError::TooManyMutations);
        }
        self.require_version(expected_version)?;
        let previous_source = self
            .fields
            .authored_mutations
            .last()
            .map_or(self.fields.source, |mutation| mutation.source);
        if source.event_created_at < previous_source.event_created_at {
            return Err(ScheduleError::InvalidTimestamp);
        }
        Ok(())
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), ScheduleError> {
        if self.fields.version != expected {
            return Err(ScheduleError::StaleVersion {
                expected,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn next_version(&self) -> Result<AggregateVersion, ScheduleError> {
        self.fields
            .version
            .next()
            .ok_or(ScheduleError::VersionExhausted)
    }

    fn has_source(&self, event_id: NostrEventId) -> bool {
        self.fields.source.event_id == event_id
            || self
                .fields
                .authored_mutations
                .iter()
                .any(|mutation| mutation.source.event_id == event_id)
    }
}

fn authorize_schedule(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    channel_id: AggregateId,
    schedule_id: AggregateId,
    author: MessageAuthor,
) -> Result<(), ScheduleError> {
    let actor = authorization_subject(request);
    if actor != author.principal_id() && author.owner_principal_id() != Some(actor) {
        return Err(ScheduleError::ActorNotAuthor);
    }
    authorize_message_command(
        request,
        community_id,
        channel_id,
        schedule_id,
        author.principal_id(),
        AuthorizationAction::Write,
    )
    .map_err(ScheduleError::Message)
}

fn validate_identity(
    schedule_id: AggregateId,
    channel_id: AggregateId,
    source: MessageSource,
) -> Result<(), ScheduleError> {
    if schedule_id.as_uuid().is_nil() || channel_id.as_uuid().is_nil() {
        return Err(ScheduleError::InvalidIdentity);
    }
    source.validate().map_err(|_| ScheduleError::InvalidSource)
}

fn validate_schedule_time(scheduled_for_millis: u64, now_millis: u64) -> Result<(), ScheduleError> {
    if scheduled_for_millis.saturating_add(MAX_CLOCK_SKEW_MILLIS) < now_millis
        || scheduled_for_millis > now_millis.saturating_add(MAX_SCHEDULE_HORIZON_MILLIS)
    {
        return Err(ScheduleError::InvalidScheduleTime);
    }
    Ok(())
}

fn validate_record(fields: &ScheduledMessageRecordFields) -> Result<(), ScheduleError> {
    validate_identity(fields.schedule_id, fields.channel_id, fields.source)?;
    fields
        .author
        .validate()
        .map_err(|_| ScheduleError::InvalidRecord)?;
    if fields.community_id.as_uuid().is_nil()
        || fields.author.principal_id().as_uuid().is_nil()
        || fields.authored_mutations.len() > MAX_AUTHORED_MUTATIONS
    {
        return Err(ScheduleError::InvalidRecord);
    }
    let mut content = fields.initial_content.clone();
    let mut scheduled_for_millis = fields.initial_scheduled_for_millis;
    let mut previous_source = fields.source;
    let mut previous_version = AggregateVersion::FIRST;
    let mut cancelled = false;
    let mut sources = BTreeSet::from([fields.source.event_id]);
    for mutation in &fields.authored_mutations {
        mutation
            .source
            .validate()
            .map_err(|_| ScheduleError::InvalidRecord)?;
        if cancelled
            || mutation.actor_principal_id.as_uuid().is_nil()
            || (mutation.actor_principal_id != fields.author.principal_id()
                && fields.author.owner_principal_id() != Some(mutation.actor_principal_id))
            || mutation.source.event_created_at < previous_source.event_created_at
            || !sources.insert(mutation.source.event_id)
            || !mutation.resulting_version.follows(previous_version)
        {
            return Err(ScheduleError::InvalidRecord);
        }
        match &mutation.kind {
            ScheduleMutationKind::Update {
                content: updated_content,
                scheduled_for_millis: updated_time,
            } => {
                content = updated_content.clone();
                scheduled_for_millis = *updated_time;
            }
            ScheduleMutationKind::Cancel => cancelled = true,
        }
        previous_source = mutation.source;
        previous_version = mutation.resulting_version;
    }
    if fields.content != content || fields.scheduled_for_millis != scheduled_for_millis {
        return Err(ScheduleError::InvalidRecord);
    }
    if cancelled != (fields.state == ScheduledMessageState::Cancelled) {
        return Err(ScheduleError::InvalidRecord);
    }
    let expected_version = match fields.state {
        ScheduledMessageState::Pending | ScheduledMessageState::Cancelled => previous_version,
        ScheduledMessageState::Claimed(claim)
            if claim.claim_id.as_uuid().is_nil()
                || claim.attempt == 0
                || claim.attempt > MAX_EXECUTION_ATTEMPTS
                || claim.lease_expires_at_millis <= claim.claimed_at_millis
                || claim.lease_expires_at_millis - claim.claimed_at_millis > MAX_LEASE_MILLIS =>
        {
            return Err(ScheduleError::InvalidRecord);
        }
        ScheduledMessageState::Claimed(claim) => {
            advance_version(previous_version, u64::from(claim.attempt))?
        }
        ScheduledMessageState::Executed {
            claim_id,
            execution_attempt,
            published_message_id,
            published_event_id,
            ..
        } if claim_id.as_uuid().is_nil()
            || execution_attempt == 0
            || execution_attempt > MAX_EXECUTION_ATTEMPTS
            || published_message_id.as_uuid().is_nil()
            || published_event_id.as_bytes().iter().all(|byte| *byte == 0) =>
        {
            return Err(ScheduleError::InvalidRecord);
        }
        ScheduledMessageState::Executed {
            execution_attempt, ..
        } => advance_version(previous_version, u64::from(execution_attempt) + 1)?,
    };
    if fields.version != expected_version {
        return Err(ScheduleError::InvalidRecord);
    }
    Ok(())
}

fn advance_version(
    mut version: AggregateVersion,
    steps: u64,
) -> Result<AggregateVersion, ScheduleError> {
    for _ in 0..steps {
        version = version.next().ok_or(ScheduleError::InvalidRecord)?;
    }
    Ok(version)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    InvalidIdentity,
    InvalidSource,
    InvalidScheduleTime,
    InvalidClaim,
    InvalidRecord,
    ChannelMismatch,
    ChannelUnavailable,
    ActorNotAuthor,
    Message(MessageError),
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTimestamp,
    NotDue,
    LeaseHeld,
    LeaseExpired,
    ClaimMismatch,
    InFlight,
    Cancelled,
    AlreadyExecuted,
    RecoveryExhausted,
    TooManyMutations,
    VersionExhausted,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity | Self::InvalidSource | Self::InvalidRecord => {
                formatter.write_str("scheduled message record is invalid")
            }
            Self::InvalidScheduleTime => formatter.write_str("scheduled message time is invalid"),
            Self::InvalidClaim | Self::ClaimMismatch => {
                formatter.write_str("scheduled message claim is invalid")
            }
            Self::ChannelMismatch | Self::ChannelUnavailable => {
                formatter.write_str("scheduled message channel is unavailable")
            }
            Self::ActorNotAuthor | Self::Message(_) => {
                formatter.write_str("scheduled message command is not authorized")
            }
            Self::StaleVersion { .. } => formatter.write_str("scheduled message version is stale"),
            Self::InvalidTimestamp => {
                formatter.write_str("scheduled message source timestamp is invalid")
            }
            Self::NotDue => formatter.write_str("scheduled message is not due"),
            Self::LeaseHeld | Self::InFlight => {
                formatter.write_str("scheduled message execution is in flight")
            }
            Self::LeaseExpired => formatter.write_str("scheduled message execution lease expired"),
            Self::Cancelled => formatter.write_str("scheduled message is cancelled"),
            Self::AlreadyExecuted => formatter.write_str("scheduled message is already executed"),
            Self::RecoveryExhausted => {
                formatter.write_str("scheduled message recovery is exhausted")
            }
            Self::TooManyMutations => formatter.write_str("scheduled message history is full"),
            Self::VersionExhausted => formatter.write_str("scheduled message version is exhausted"),
        }
    }
}

impl Error for ScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationResourceKind,
        AuthorizationScope, ChannelMembership, ChannelName, ChannelRecordFields, ChannelType,
        ChannelVisibility, CommunityMembership, MembershipRole, MembershipStatus, PrincipalScopes,
        ServiceAccountId, TenantContext, TrustedTenantRoute,
    };
    use uuid::Uuid;

    const NOW_MILLIS: u64 = 1_000_000;
    const DUE_MILLIS: u64 = 1_100_000;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn operation_id(value: u128) -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn source(value: u8, event_created_at: u64) -> MessageSource {
        MessageSource {
            event_id: NostrEventId::from_bytes([value; 32]),
            event_created_at,
        }
    }

    fn version(value: u64) -> AggregateVersion {
        AggregateVersion::new(value).expect("nonzero version")
    }

    fn channel() -> Channel {
        Channel::from_record(ChannelRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            name: ChannelName::new("general").expect("valid channel name"),
            channel_type: ChannelType::Stream,
            visibility: ChannelVisibility::Open,
            lifecycle_state: ChannelLifecycleState::Active,
            description: None,
            creator_principal_id: principal_id(3),
            expiration: None,
            version: AggregateVersion::FIRST,
        })
        .expect("valid channel")
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id(), "schedule-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("messages:write").expect("valid scope")
    }

    fn principal(id: PrincipalId, scope: &AuthorizationScope) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            id,
            community_id(),
            ServiceAccountId::new(id.as_uuid().as_u128() as u64),
            PrincipalScopes::new([scope.clone()]).expect("valid scopes"),
        )
    }

    fn request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Conversation,
                resource_id: aggregate_id(4),
                owner_principal_id: Some(principal_id(3)),
                channel_id: Some(aggregate_id(2)),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id: aggregate_id(2),
                principal_id: principal.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: NOW_MILLIS,
        }
    }

    fn create_fields(scheduled_for_millis: u64) -> ScheduledMessageCreateFields {
        ScheduledMessageCreateFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            schedule_id: aggregate_id(4),
            author: MessageAuthor::principal(principal_id(3)),
            content: MessageContent::new("original").expect("valid content"),
            scheduled_for_millis,
            source: source(1, 10),
        }
    }

    fn schedule(authorization: &AuthorizationRequest<'_>) -> ScheduledMessage {
        ScheduledMessage::create(
            create_fields(DUE_MILLIS),
            &channel(),
            NOW_MILLIS,
            authorization,
        )
        .expect("authorized schedule")
    }

    #[test]
    fn author_updates_cancels_and_retries_while_other_actor_is_denied() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let author_request = request(&tenant, &author, &scope);
        let mut schedule = schedule(&author_request);

        assert_eq!(
            schedule.update(
                AggregateVersion::FIRST,
                MessageContent::new("edited").expect("valid content"),
                DUE_MILLIS + 1_000,
                source(2, 11),
                NOW_MILLIS,
                &author_request,
            ),
            Ok(ScheduleCommandOutcome::Applied)
        );
        let edited = schedule.clone();
        assert_eq!(
            schedule.update(
                AggregateVersion::FIRST,
                MessageContent::new("ignored retry").expect("valid content"),
                DUE_MILLIS,
                source(2, 11),
                NOW_MILLIS,
                &author_request,
            ),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, edited);

        let other = principal(principal_id(5), &scope);
        let other_request = request(&tenant, &other, &scope);
        assert_eq!(
            schedule.update(
                version(2),
                MessageContent::new("denied").expect("valid content"),
                DUE_MILLIS,
                source(3, 12),
                NOW_MILLIS,
                &other_request,
            ),
            Err(ScheduleError::ActorNotAuthor)
        );
        assert_eq!(schedule, edited);

        assert_eq!(
            schedule.cancel(version(2), source(4, 13), &author_request),
            Ok(ScheduleCommandOutcome::Applied)
        );
        let cancelled = schedule.clone();
        assert_eq!(
            schedule.cancel(version(2), source(4, 13), &author_request),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, cancelled);
        assert_eq!(schedule.fields().state, ScheduledMessageState::Cancelled);
        assert_eq!(
            schedule.claim_due(version(3), operation_id(10), DUE_MILLIS, 10_000),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, cancelled);
    }

    #[test]
    fn due_claim_and_completion_are_exactly_once_and_clock_skew_bounded() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &author, &scope);
        let mut schedule = schedule(&authorization);
        let too_early = DUE_MILLIS - MAX_CLOCK_SKEW_MILLIS - 1;
        assert_eq!(
            schedule.claim_due(AggregateVersion::FIRST, operation_id(10), too_early, 10_000,),
            Err(ScheduleError::NotDue)
        );
        assert_eq!(
            schedule.claim_due(
                AggregateVersion::FIRST,
                operation_id(10),
                DUE_MILLIS - MAX_CLOCK_SKEW_MILLIS,
                10_000,
            ),
            Ok(ScheduleCommandOutcome::Applied)
        );
        let claimed = schedule.clone();
        assert_eq!(
            schedule.claim_due(
                AggregateVersion::FIRST,
                operation_id(10),
                DUE_MILLIS,
                10_000,
            ),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, claimed);

        let completion_time = DUE_MILLIS - MAX_CLOCK_SKEW_MILLIS + 1_000;
        assert_eq!(
            schedule.complete_due(
                version(2),
                operation_id(10),
                aggregate_id(20),
                source(8, 20).event_id,
                completion_time,
            ),
            Ok(ScheduleCommandOutcome::Applied)
        );
        let executed = schedule.clone();
        assert_eq!(
            schedule.complete_due(
                version(2),
                operation_id(10),
                aggregate_id(20),
                source(8, 20).event_id,
                completion_time,
            ),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, executed);
        assert_eq!(
            schedule.complete_due(
                version(3),
                operation_id(11),
                aggregate_id(20),
                source(8, 20).event_id,
                completion_time,
            ),
            Err(ScheduleError::AlreadyExecuted)
        );
        assert_eq!(
            schedule.claim_due(version(3), operation_id(12), DUE_MILLIS, 10_000),
            Ok(ScheduleCommandOutcome::Unchanged)
        );
        assert_eq!(schedule, executed);
    }

    #[test]
    fn expired_claim_recovers_after_restart_and_old_worker_loses_authority() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &author, &scope);
        let mut schedule = schedule(&authorization);
        assert_eq!(
            schedule.claim_due(
                AggregateVersion::FIRST,
                operation_id(10),
                DUE_MILLIS,
                10_000,
            ),
            Ok(ScheduleCommandOutcome::Applied)
        );
        let mut recovered =
            ScheduledMessage::from_record(schedule.fields().clone()).expect("restart hydration");
        assert_eq!(
            recovered.claim_due(version(2), operation_id(11), DUE_MILLIS + 9_999, 10_000),
            Err(ScheduleError::LeaseHeld)
        );
        assert_eq!(
            recovered.complete_due(
                version(2),
                operation_id(10),
                aggregate_id(20),
                source(8, 20).event_id,
                DUE_MILLIS + 10_000,
            ),
            Err(ScheduleError::LeaseExpired)
        );
        assert_eq!(
            recovered.claim_due(version(2), operation_id(11), DUE_MILLIS + 10_000, 10_000),
            Ok(ScheduleCommandOutcome::Applied)
        );
        assert!(matches!(
            recovered.fields().state,
            ScheduledMessageState::Claimed(DueClaim { attempt: 2, .. })
        ));
        let mut recovered_again = ScheduledMessage::from_record(recovered.fields().clone())
            .expect("second restart hydration");
        assert_eq!(
            recovered_again.complete_due(
                version(3),
                operation_id(11),
                aggregate_id(20),
                source(8, 20).event_id,
                DUE_MILLIS + 15_000,
            ),
            Ok(ScheduleCommandOutcome::Applied)
        );
        ScheduledMessage::from_record(recovered_again.fields().clone())
            .expect("executed restart hydration");
        let mut invalid = recovered_again.fields().clone();
        invalid.version = version(3);
        assert_eq!(
            ScheduledMessage::from_record(invalid),
            Err(ScheduleError::InvalidRecord)
        );
    }

    #[test]
    fn schedule_time_accepts_only_bounded_skew_and_horizon() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &author, &scope);
        assert!(
            ScheduledMessage::create(
                create_fields(NOW_MILLIS - MAX_CLOCK_SKEW_MILLIS),
                &channel(),
                NOW_MILLIS,
                &authorization,
            )
            .is_ok()
        );
        assert_eq!(
            ScheduledMessage::create(
                create_fields(NOW_MILLIS - MAX_CLOCK_SKEW_MILLIS - 1),
                &channel(),
                NOW_MILLIS,
                &authorization,
            ),
            Err(ScheduleError::InvalidScheduleTime)
        );
        assert!(
            ScheduledMessage::create(
                create_fields(NOW_MILLIS + MAX_SCHEDULE_HORIZON_MILLIS),
                &channel(),
                NOW_MILLIS,
                &authorization,
            )
            .is_ok()
        );
        assert_eq!(
            ScheduledMessage::create(
                create_fields(NOW_MILLIS + MAX_SCHEDULE_HORIZON_MILLIS + 1),
                &channel(),
                NOW_MILLIS,
                &authorization,
            ),
            Err(ScheduleError::InvalidScheduleTime)
        );
    }
}
