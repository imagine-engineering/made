use std::{collections::BTreeSet, error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind,
    Channel, ChannelLifecycleState, CommunityId, NostrEventId, PrincipalId,
    VirtualAgentMembershipEvidence, authorize,
};

const MAX_CONTENT_BYTES: usize = 65_536;
const MAX_MUTATIONS: usize = 10_000;
const MAX_REASON_CODE_BYTES: usize = 64;
const MAX_PUBLIC_REASON_BYTES: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageContent(String);

impl MessageContent {
    pub fn new(value: impl Into<String>) -> Result<Self, MessageError> {
        let value = value.into();
        if value.len() > MAX_CONTENT_BYTES {
            return Err(MessageError::ContentTooLarge);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageAuthor {
    Principal(PrincipalId),
    OwnerAttestedAgent {
        agent_principal_id: PrincipalId,
        owner_principal_id: PrincipalId,
        proof_event_id: NostrEventId,
    },
}

impl MessageAuthor {
    pub const fn principal(principal_id: PrincipalId) -> Self {
        Self::Principal(principal_id)
    }

    pub const fn from_virtual_agent(evidence: VirtualAgentMembershipEvidence) -> Self {
        Self::OwnerAttestedAgent {
            agent_principal_id: evidence.policy_membership_snapshot().principal_id,
            owner_principal_id: evidence.owner_principal_id(),
            proof_event_id: evidence.proof_event_id(),
        }
    }

    pub const fn principal_id(self) -> PrincipalId {
        match self {
            Self::Principal(principal_id) => principal_id,
            Self::OwnerAttestedAgent {
                agent_principal_id, ..
            } => agent_principal_id,
        }
    }

    pub const fn owner_principal_id(self) -> Option<PrincipalId> {
        match self {
            Self::Principal(_) => None,
            Self::OwnerAttestedAgent {
                owner_principal_id, ..
            } => Some(owner_principal_id),
        }
    }

    pub(crate) fn validate(self) -> Result<(), MessageError> {
        match self {
            Self::Principal(principal_id) if !principal_id.as_uuid().is_nil() => Ok(()),
            Self::OwnerAttestedAgent {
                agent_principal_id,
                owner_principal_id,
                proof_event_id,
            } if !agent_principal_id.as_uuid().is_nil()
                && !owner_principal_id.as_uuid().is_nil()
                && agent_principal_id != owner_principal_id
                && proof_event_id.as_bytes().iter().any(|byte| *byte != 0) =>
            {
                Ok(())
            }
            _ => Err(MessageError::InvalidAuthor),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessageSource {
    pub event_id: NostrEventId,
    pub event_created_at: u64,
}

impl MessageSource {
    pub(crate) fn validate(self) -> Result<(), MessageError> {
        if self.event_id.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(MessageError::InvalidSource);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageDeleteMetadata {
    action_id: Option<AggregateId>,
    reason_code: Option<String>,
    public_reason: Option<String>,
}

impl MessageDeleteMetadata {
    pub fn new(
        action_id: Option<AggregateId>,
        reason_code: Option<String>,
        public_reason: Option<String>,
    ) -> Result<Self, MessageError> {
        let metadata = Self {
            action_id,
            reason_code,
            public_reason,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    pub const fn is_empty(&self) -> bool {
        self.action_id.is_none() && self.reason_code.is_none() && self.public_reason.is_none()
    }

    pub const fn action_id(&self) -> Option<AggregateId> {
        self.action_id
    }

    pub fn reason_code(&self) -> Option<&str> {
        self.reason_code.as_deref()
    }

    pub fn public_reason(&self) -> Option<&str> {
        self.public_reason.as_deref()
    }

    fn validate(&self) -> Result<(), MessageError> {
        if self
            .action_id
            .is_some_and(|action_id| action_id.as_uuid().is_nil())
            || self.reason_code.as_ref().is_some_and(|reason| {
                reason.is_empty()
                    || reason.len() > MAX_REASON_CODE_BYTES
                    || reason.chars().any(char::is_control)
            })
            || self
                .public_reason
                .as_ref()
                .is_some_and(|reason| reason.len() > MAX_PUBLIC_REASON_BYTES)
        {
            return Err(MessageError::InvalidModerationMetadata);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageLifecycleState {
    Active,
    Edited,
    Deleted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageMutationKind {
    Edit,
    Delete {
        moderated: bool,
        metadata: Option<MessageDeleteMetadata>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageMutation {
    pub source: MessageSource,
    pub actor_principal_id: PrincipalId,
    pub kind: MessageMutationKind,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub message_id: AggregateId,
    pub author: MessageAuthor,
    pub content: MessageContent,
    pub lifecycle_state: MessageLifecycleState,
    pub source: MessageSource,
    pub current_source: MessageSource,
    pub mutations: Vec<MessageMutation>,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageCreateFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub message_id: AggregateId,
    pub author: MessageAuthor,
    pub content: MessageContent,
    pub source: MessageSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    fields: MessageRecordFields,
}

impl Message {
    pub fn from_record(fields: MessageRecordFields) -> Result<Self, MessageError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub fn create(
        fields: MessageCreateFields,
        channel: &Channel,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, MessageError> {
        validate_identity(fields.message_id, fields.channel_id, fields.source)?;
        fields.author.validate()?;
        if channel.fields().community_id != fields.community_id
            || channel.fields().channel_id != fields.channel_id
        {
            return Err(MessageError::ChannelMismatch);
        }
        if channel.fields().lifecycle_state != ChannelLifecycleState::Active {
            return Err(MessageError::ChannelUnavailable);
        }
        authorize_message_command(
            authorization,
            fields.community_id,
            fields.channel_id,
            fields.message_id,
            fields.author.principal_id(),
            AuthorizationAction::Write,
        )?;
        validate_authenticated_author(fields.author, authorization)?;
        Ok(Self {
            fields: MessageRecordFields {
                community_id: fields.community_id,
                channel_id: fields.channel_id,
                message_id: fields.message_id,
                author: fields.author,
                content: fields.content,
                lifecycle_state: MessageLifecycleState::Active,
                source: fields.source,
                current_source: fields.source,
                mutations: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        })
    }

    pub const fn fields(&self) -> &MessageRecordFields {
        &self.fields
    }

    pub fn visible_content(&self) -> Option<&MessageContent> {
        (self.fields.lifecycle_state != MessageLifecycleState::Deleted)
            .then_some(&self.fields.content)
    }

    pub fn edit(
        &mut self,
        expected_version: AggregateVersion,
        content: MessageContent,
        source: MessageSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MessageCommandOutcome, MessageError> {
        source.validate()?;
        let actor = authorization_subject(authorization);
        if !self.is_author_or_owner(actor) {
            return Err(MessageError::ActorNotAuthor);
        }
        authorize_message_command(
            authorization,
            self.fields.community_id,
            self.fields.channel_id,
            self.fields.message_id,
            self.fields.author.principal_id(),
            AuthorizationAction::Write,
        )?;
        if self.has_source(source.event_id) {
            return Ok(MessageCommandOutcome::Unchanged);
        }
        self.require_mutation_capacity()?;
        self.require_source_time(source)?;
        self.require_not_deleted()?;
        self.require_version(expected_version)?;
        let next_version = self.next_version()?;
        self.fields.content = content;
        self.fields.lifecycle_state = MessageLifecycleState::Edited;
        self.fields.current_source = source;
        self.fields.mutations.push(MessageMutation {
            source,
            actor_principal_id: actor,
            kind: MessageMutationKind::Edit,
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(MessageCommandOutcome::Applied)
    }

    pub fn delete(
        &mut self,
        expected_version: AggregateVersion,
        source: MessageSource,
        metadata: Option<MessageDeleteMetadata>,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MessageCommandOutcome, MessageError> {
        source.validate()?;
        if let Some(metadata) = &metadata {
            metadata.validate()?;
        }
        let actor = authorization_subject(authorization);
        let self_delete = self.is_author_or_owner(actor)
            && metadata
                .as_ref()
                .is_none_or(MessageDeleteMetadata::is_empty);
        authorize_message_command(
            authorization,
            self.fields.community_id,
            self.fields.channel_id,
            self.fields.message_id,
            self.fields.author.principal_id(),
            if self_delete {
                AuthorizationAction::Write
            } else {
                AuthorizationAction::Manage
            },
        )?;
        if self.has_source(source.event_id)
            || self.fields.lifecycle_state == MessageLifecycleState::Deleted
        {
            return Ok(MessageCommandOutcome::Unchanged);
        }
        self.require_mutation_capacity()?;
        self.require_source_time(source)?;
        self.require_version(expected_version)?;
        let next_version = self.next_version()?;
        self.fields.lifecycle_state = MessageLifecycleState::Deleted;
        self.fields.current_source = source;
        self.fields.mutations.push(MessageMutation {
            source,
            actor_principal_id: actor,
            kind: MessageMutationKind::Delete {
                moderated: !self_delete,
                metadata: metadata.filter(|metadata| !metadata.is_empty()),
            },
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(MessageCommandOutcome::Applied)
    }

    fn is_author_or_owner(&self, principal_id: PrincipalId) -> bool {
        principal_id == self.fields.author.principal_id()
            || self.fields.author.owner_principal_id() == Some(principal_id)
    }

    fn has_source(&self, event_id: NostrEventId) -> bool {
        self.fields.source.event_id == event_id
            || self
                .fields
                .mutations
                .iter()
                .any(|mutation| mutation.source.event_id == event_id)
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), MessageError> {
        if self.fields.version != expected {
            return Err(MessageError::StaleVersion {
                expected,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_not_deleted(&self) -> Result<(), MessageError> {
        if self.fields.lifecycle_state == MessageLifecycleState::Deleted {
            return Err(MessageError::Deleted);
        }
        Ok(())
    }

    fn require_source_time(&self, source: MessageSource) -> Result<(), MessageError> {
        if source.event_created_at < self.fields.current_source.event_created_at {
            return Err(MessageError::InvalidTimestamp);
        }
        Ok(())
    }

    fn require_mutation_capacity(&self) -> Result<(), MessageError> {
        if self.fields.mutations.len() >= MAX_MUTATIONS {
            return Err(MessageError::TooManyMutations);
        }
        Ok(())
    }

    fn next_version(&self) -> Result<AggregateVersion, MessageError> {
        self.fields
            .version
            .next()
            .ok_or(MessageError::VersionExhausted)
    }
}

pub(crate) fn authorize_message_command(
    request: &AuthorizationRequest<'_>,
    community_id: CommunityId,
    channel_id: AggregateId,
    message_id: AggregateId,
    author_principal_id: PrincipalId,
    action: AuthorizationAction,
) -> Result<(), MessageError> {
    if request.action != action
        || request.resource.community_id != community_id
        || request.resource.kind != AuthorizationResourceKind::Conversation
        || request.resource.resource_id != message_id
        || request.resource.owner_principal_id != Some(author_principal_id)
        || request.resource.channel_id != Some(channel_id)
    {
        return Err(MessageError::AuthorizationShape);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(MessageError::Unauthorized(denial)),
    }
}

pub(crate) fn validate_authenticated_author(
    author: MessageAuthor,
    request: &AuthorizationRequest<'_>,
) -> Result<(), MessageError> {
    if authorization_subject(request) != author.principal_id() {
        return Err(MessageError::AuthorMismatch);
    }
    match (author, request.principal.kind()) {
        (
            MessageAuthor::OwnerAttestedAgent { proof_event_id, .. },
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                proof_event_id: authenticated_proof_event_id,
                ..
            },
        ) if proof_event_id == *authenticated_proof_event_id => Ok(()),
        (MessageAuthor::OwnerAttestedAgent { .. }, _)
        | (_, AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }) => {
            Err(MessageError::AuthorEvidenceRequired)
        }
        _ => Ok(()),
    }
}

pub(crate) fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

fn validate_identity(
    message_id: AggregateId,
    channel_id: AggregateId,
    source: MessageSource,
) -> Result<(), MessageError> {
    if message_id.as_uuid().is_nil() || channel_id.as_uuid().is_nil() {
        return Err(MessageError::InvalidMessageId);
    }
    source.validate()
}

fn validate_record(fields: &MessageRecordFields) -> Result<(), MessageError> {
    validate_identity(fields.message_id, fields.channel_id, fields.source)?;
    fields.author.validate()?;
    if fields.mutations.len() > MAX_MUTATIONS {
        return Err(MessageError::TooManyMutations);
    }
    let mut sources = BTreeSet::from([fields.source.event_id]);
    let mut previous_source = fields.source;
    let mut previous_version = AggregateVersion::FIRST;
    let mut derived_lifecycle = MessageLifecycleState::Active;
    for mutation in &fields.mutations {
        mutation.source.validate()?;
        if mutation.actor_principal_id.as_uuid().is_nil()
            || derived_lifecycle == MessageLifecycleState::Deleted
        {
            return Err(MessageError::InvalidHistory);
        }
        let actor_is_author_or_owner = mutation.actor_principal_id == fields.author.principal_id()
            || fields.author.owner_principal_id() == Some(mutation.actor_principal_id);
        derived_lifecycle = match &mutation.kind {
            MessageMutationKind::Edit if actor_is_author_or_owner => MessageLifecycleState::Edited,
            MessageMutationKind::Edit => return Err(MessageError::InvalidHistory),
            MessageMutationKind::Delete {
                moderated: false,
                metadata: None,
            } if actor_is_author_or_owner => MessageLifecycleState::Deleted,
            MessageMutationKind::Delete {
                moderated: true,
                metadata,
            } => {
                if let Some(metadata) = metadata {
                    metadata.validate()?;
                    if metadata.is_empty() {
                        return Err(MessageError::InvalidHistory);
                    }
                }
                MessageLifecycleState::Deleted
            }
            MessageMutationKind::Delete { .. } => return Err(MessageError::InvalidHistory),
        };
        if mutation.source.event_created_at < previous_source.event_created_at {
            return Err(MessageError::InvalidTimestamp);
        }
        if !sources.insert(mutation.source.event_id) {
            return Err(MessageError::DuplicateSource);
        }
        if !mutation.resulting_version.follows(previous_version) {
            return Err(MessageError::InvalidHistory);
        }
        previous_source = mutation.source;
        previous_version = mutation.resulting_version;
    }
    if fields.current_source != previous_source || fields.version != previous_version {
        return Err(MessageError::InvalidHistory);
    }
    if fields.lifecycle_state != derived_lifecycle {
        return Err(MessageError::InvalidHistory);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageError {
    InvalidMessageId,
    InvalidAuthor,
    InvalidSource,
    ContentTooLarge,
    InvalidModerationMetadata,
    ChannelMismatch,
    ChannelUnavailable,
    AuthorMismatch,
    AuthorEvidenceRequired,
    ActorNotAuthor,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTimestamp,
    Deleted,
    DuplicateSource,
    InvalidHistory,
    TooManyMutations,
    VersionExhausted,
}

impl fmt::Display for MessageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMessageId
            | Self::InvalidAuthor
            | Self::InvalidSource
            | Self::InvalidHistory => {
                formatter.write_str("message identity or source history is invalid")
            }
            Self::ContentTooLarge => formatter.write_str("message content is too large"),
            Self::InvalidModerationMetadata => {
                formatter.write_str("message moderation metadata is invalid")
            }
            Self::ChannelMismatch | Self::ChannelUnavailable => {
                formatter.write_str("message channel is unavailable")
            }
            Self::AuthorMismatch
            | Self::AuthorEvidenceRequired
            | Self::ActorNotAuthor
            | Self::AuthorizationShape
            | Self::Unauthorized(_) => formatter.write_str("message command is not authorized"),
            Self::StaleVersion { .. } => formatter.write_str("message version is stale"),
            Self::InvalidTimestamp => formatter.write_str("message source timestamp is invalid"),
            Self::Deleted => formatter.write_str("message is deleted"),
            Self::DuplicateSource => formatter.write_str("message source event is duplicated"),
            Self::TooManyMutations => formatter.write_str("message mutation history is full"),
            Self::VersionExhausted => formatter.write_str("message version is exhausted"),
        }
    }
}

impl Error for MessageError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope, ChannelMembership,
        ChannelName, ChannelRecordFields, ChannelType, ChannelVisibility, CommunityMembership,
        MembershipRole, MembershipStatus, PrincipalScopes, ServiceAccountId, TenantContext,
        TrustedTenantRoute,
    };
    use uuid::Uuid;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
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
                TrustedTenantRoute::from_listener(community_id(), "message-test")
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
        author_principal_id: PrincipalId,
        action: AuthorizationAction,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Conversation,
                resource_id: aggregate_id(4),
                owner_principal_id: Some(author_principal_id),
                channel_id: Some(aggregate_id(2)),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal.principal_id(),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id: aggregate_id(2),
                principal_id: principal.principal_id(),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }

    fn create_message(
        channel: &Channel,
        authorization: &AuthorizationRequest<'_>,
        author_principal_id: PrincipalId,
    ) -> Message {
        Message::create(
            MessageCreateFields {
                community_id: community_id(),
                channel_id: aggregate_id(2),
                message_id: aggregate_id(4),
                author: MessageAuthor::principal(author_principal_id),
                content: MessageContent::new("original").expect("valid content"),
                source: source(1, 10),
            },
            channel,
            authorization,
        )
        .expect("authorized message")
    }

    #[test]
    fn author_edits_are_versioned_stale_safe_and_retry_idempotent() {
        let tenant = tenant();
        let scope = scope();
        let author_id = principal_id(3);
        let author = principal(author_id, &scope);
        let author_request = request(
            &tenant,
            &author,
            &scope,
            author_id,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let mut message = create_message(&channel(), &author_request, author_id);

        assert_eq!(
            message.edit(
                AggregateVersion::FIRST,
                MessageContent::new("edited").expect("valid content"),
                source(2, 11),
                &author_request,
            ),
            Ok(MessageCommandOutcome::Applied)
        );
        assert_eq!(
            message.visible_content().map(MessageContent::as_str),
            Some("edited")
        );
        let edited = message.clone();

        assert_eq!(
            message.edit(
                AggregateVersion::FIRST,
                MessageContent::new("retry payload is ignored").expect("valid content"),
                source(2, 11),
                &author_request,
            ),
            Ok(MessageCommandOutcome::Unchanged)
        );
        assert_eq!(message, edited);

        assert_eq!(
            message.edit(
                AggregateVersion::FIRST,
                MessageContent::new("stale").expect("valid content"),
                source(3, 12),
                &author_request,
            ),
            Err(MessageError::StaleVersion {
                expected: AggregateVersion::FIRST,
                actual: AggregateVersion::new(2).expect("valid version"),
            })
        );
        assert_eq!(message, edited);

        let other = principal(principal_id(5), &scope);
        let other_request = request(
            &tenant,
            &other,
            &scope,
            author_id,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        assert_eq!(
            message.edit(
                AggregateVersion::new(2).expect("valid version"),
                MessageContent::new("unauthorized").expect("valid content"),
                source(4, 13),
                &other_request,
            ),
            Err(MessageError::ActorNotAuthor)
        );
        assert_eq!(message, edited);
    }

    #[test]
    fn author_delete_hides_content_and_authenticated_retry_is_idempotent() {
        let tenant = tenant();
        let scope = scope();
        let author_id = principal_id(3);
        let author = principal(author_id, &scope);
        let author_request = request(
            &tenant,
            &author,
            &scope,
            author_id,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let mut message = create_message(&channel(), &author_request, author_id);

        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(2, 11),
                None,
                &author_request,
            ),
            Ok(MessageCommandOutcome::Applied)
        );
        assert_eq!(message.visible_content(), None);
        assert_eq!(
            message.fields().lifecycle_state,
            MessageLifecycleState::Deleted
        );
        let deleted = message.clone();

        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(2, 11),
                None,
                &author_request,
            ),
            Ok(MessageCommandOutcome::Unchanged)
        );
        assert_eq!(message, deleted);
    }

    #[test]
    fn moderator_delete_requires_manage_role_and_preserves_reason() {
        let tenant = tenant();
        let scope = scope();
        let author_id = principal_id(3);
        let author = principal(author_id, &scope);
        let author_request = request(
            &tenant,
            &author,
            &scope,
            author_id,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let mut message = create_message(&channel(), &author_request, author_id);
        let metadata = MessageDeleteMetadata::new(
            Some(aggregate_id(8)),
            Some("policy_violation".to_owned()),
            Some("Removed by a moderator".to_owned()),
        )
        .expect("valid moderation metadata");

        let member = principal(principal_id(5), &scope);
        let member_request = request(
            &tenant,
            &member,
            &scope,
            author_id,
            AuthorizationAction::Manage,
            MembershipRole::Member,
        );
        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(2, 11),
                Some(metadata.clone()),
                &member_request,
            ),
            Err(MessageError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );
        assert_eq!(
            message.visible_content().map(MessageContent::as_str),
            Some("original")
        );

        let moderator = principal(principal_id(6), &scope);
        let moderator_request = request(
            &tenant,
            &moderator,
            &scope,
            author_id,
            AuthorizationAction::Manage,
            MembershipRole::Admin,
        );
        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(2, 11),
                Some(metadata),
                &moderator_request,
            ),
            Ok(MessageCommandOutcome::Applied)
        );
        assert_eq!(message.visible_content(), None);
        let Some(MessageMutation {
            actor_principal_id,
            kind:
                MessageMutationKind::Delete {
                    moderated,
                    metadata: Some(metadata),
                },
            ..
        }) = message.fields().mutations.last()
        else {
            panic!("moderated delete mutation must be retained");
        };
        assert_eq!(*actor_principal_id, principal_id(6));
        assert!(*moderated);
        assert_eq!(metadata.action_id(), Some(aggregate_id(8)));
        assert_eq!(metadata.reason_code(), Some("policy_violation"));
        assert_eq!(metadata.public_reason(), Some("Removed by a moderator"));

        let deleted = message.clone();
        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(2, 11),
                None,
                &member_request,
            ),
            Err(MessageError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        );
        assert_eq!(message, deleted);
    }
}
