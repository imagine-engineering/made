use std::{collections::BTreeMap, collections::BTreeSet, error::Error, fmt};

use crate::message::{authorization_subject, authorize_message_command};
use crate::{
    AggregateId, AggregateVersion, AuthorizationAction, AuthorizationRequest, CommunityId, Message,
    MessageError, MessageLifecycleState, MessageSource, NostrEventId, PrincipalId,
};

const MAX_STANDARD_REACTION_CHARACTERS: usize = 64;
const MAX_CUSTOM_EMOJI_SHORTCODE_BYTES: usize = 64;
const MAX_REACTION_MUTATIONS: usize = 10_000;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReactionValue(String);

impl ReactionValue {
    pub fn new(value: impl Into<String>) -> Result<Self, ReactionError> {
        let value = value.into();
        let character_count = value.chars().count();
        if value.is_empty() || value.chars().any(char::is_control) {
            return Err(ReactionError::InvalidValue);
        }
        if character_count <= MAX_STANDARD_REACTION_CHARACTERS {
            return Ok(Self(value));
        }
        let Some(shortcode) = value
            .strip_prefix(':')
            .and_then(|value| value.strip_suffix(':'))
        else {
            return Err(ReactionError::InvalidValue);
        };
        if shortcode.is_empty()
            || shortcode.len() > MAX_CUSTOM_EMOJI_SHORTCODE_BYTES
            || !shortcode.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            })
            || shortcode != shortcode.to_ascii_lowercase()
        {
            return Err(ReactionError::InvalidValue);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReactionMutationKind {
    Add,
    Remove { added_event_id: NostrEventId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReactionMutation {
    pub source: MessageSource,
    pub actor_principal_id: PrincipalId,
    pub value: ReactionValue,
    pub kind: ReactionMutationKind,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReactionRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub message_id: AggregateId,
    pub target_message_event_id: NostrEventId,
    pub mutations: Vec<ReactionMutation>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReactionCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveReaction {
    pub actor_principal_id: PrincipalId,
    pub added_source: MessageSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReactionGroup {
    pub value: ReactionValue,
    pub reactions: Vec<ActiveReaction>,
}

impl ReactionGroup {
    pub fn count(&self) -> usize {
        self.reactions.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReactionSet {
    fields: ReactionRecordFields,
}

impl ReactionSet {
    pub fn for_message(message: &Message) -> Self {
        Self {
            fields: ReactionRecordFields {
                community_id: message.fields().community_id,
                channel_id: message.fields().channel_id,
                message_id: message.fields().message_id,
                target_message_event_id: message.fields().source.event_id,
                mutations: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        }
    }

    pub fn from_record(fields: ReactionRecordFields) -> Result<Self, ReactionError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &ReactionRecordFields {
        &self.fields
    }

    pub fn add(
        &mut self,
        expected_version: AggregateVersion,
        value: ReactionValue,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ReactionCommandOutcome, ReactionError> {
        source
            .validate()
            .map_err(|_| ReactionError::InvalidSource)?;
        self.require_target(message)?;
        let actor = self.authorize(message, authorization)?;
        if self.has_source(source.event_id) {
            return Ok(ReactionCommandOutcome::Unchanged);
        }
        if message.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Err(ReactionError::TargetDeleted);
        }
        let active = self.active_state()?;
        if active.contains_key(&(value.clone(), actor)) {
            return Ok(ReactionCommandOutcome::Unchanged);
        }
        self.require_mutation(source, expected_version)?;
        self.push_mutation(actor, value, source, ReactionMutationKind::Add)?;
        Ok(ReactionCommandOutcome::Applied)
    }

    pub fn remove(
        &mut self,
        expected_version: AggregateVersion,
        value: ReactionValue,
        added_event_id: NostrEventId,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<ReactionCommandOutcome, ReactionError> {
        source
            .validate()
            .map_err(|_| ReactionError::InvalidSource)?;
        self.require_target(message)?;
        let actor = self.authorize(message, authorization)?;
        if self.has_source(source.event_id) {
            return Ok(ReactionCommandOutcome::Unchanged);
        }
        let active = self.active_state()?;
        let Some(active_source) = active.get(&(value.clone(), actor)) else {
            return Ok(ReactionCommandOutcome::Unchanged);
        };
        if active_source.event_id != added_event_id {
            return Err(ReactionError::AddedEventMismatch);
        }
        self.require_mutation(source, expected_version)?;
        self.push_mutation(
            actor,
            value,
            source,
            ReactionMutationKind::Remove { added_event_id },
        )?;
        Ok(ReactionCommandOutcome::Applied)
    }

    pub fn active_groups(&self, message: &Message) -> Result<Vec<ReactionGroup>, ReactionError> {
        self.require_target(message)?;
        if message.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Ok(Vec::new());
        }
        let mut groups = BTreeMap::<ReactionValue, Vec<ActiveReaction>>::new();
        for ((value, actor_principal_id), added_source) in self.active_state()? {
            groups.entry(value).or_default().push(ActiveReaction {
                actor_principal_id,
                added_source,
            });
        }
        Ok(groups
            .into_iter()
            .map(|(value, reactions)| ReactionGroup { value, reactions })
            .collect())
    }

    fn authorize(
        &self,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<PrincipalId, ReactionError> {
        authorize_message_command(
            authorization,
            message.fields().community_id,
            message.fields().channel_id,
            message.fields().message_id,
            message.fields().author.principal_id(),
            AuthorizationAction::Write,
        )
        .map_err(ReactionError::Message)?;
        Ok(authorization_subject(authorization))
    }

    fn require_target(&self, message: &Message) -> Result<(), ReactionError> {
        if self.fields.community_id != message.fields().community_id
            || self.fields.channel_id != message.fields().channel_id
            || self.fields.message_id != message.fields().message_id
            || self.fields.target_message_event_id != message.fields().source.event_id
        {
            return Err(ReactionError::TargetMismatch);
        }
        Ok(())
    }

    fn has_source(&self, event_id: NostrEventId) -> bool {
        self.fields
            .mutations
            .iter()
            .any(|mutation| mutation.source.event_id == event_id)
    }

    fn require_mutation(
        &self,
        source: MessageSource,
        expected_version: AggregateVersion,
    ) -> Result<(), ReactionError> {
        if self.fields.mutations.len() >= MAX_REACTION_MUTATIONS {
            return Err(ReactionError::TooManyMutations);
        }
        if self.fields.version != expected_version {
            return Err(ReactionError::StaleVersion {
                expected: expected_version,
                actual: self.fields.version,
            });
        }
        if self
            .fields
            .mutations
            .last()
            .is_some_and(|mutation| source.event_created_at < mutation.source.event_created_at)
        {
            return Err(ReactionError::InvalidTimestamp);
        }
        Ok(())
    }

    fn push_mutation(
        &mut self,
        actor_principal_id: PrincipalId,
        value: ReactionValue,
        source: MessageSource,
        kind: ReactionMutationKind,
    ) -> Result<(), ReactionError> {
        let next_version = self
            .fields
            .version
            .next()
            .ok_or(ReactionError::VersionExhausted)?;
        self.fields.mutations.push(ReactionMutation {
            source,
            actor_principal_id,
            value,
            kind,
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(())
    }

    fn active_state(
        &self,
    ) -> Result<BTreeMap<(ReactionValue, PrincipalId), MessageSource>, ReactionError> {
        fold_mutations(&self.fields.mutations)
    }
}

fn validate_record(fields: &ReactionRecordFields) -> Result<(), ReactionError> {
    if fields.community_id.as_uuid().is_nil()
        || fields.channel_id.as_uuid().is_nil()
        || fields.message_id.as_uuid().is_nil()
        || fields
            .target_message_event_id
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
        || fields.mutations.len() > MAX_REACTION_MUTATIONS
    {
        return Err(ReactionError::InvalidRecord);
    }
    let mut sources = BTreeSet::new();
    let mut previous_source = None;
    let mut previous_version = AggregateVersion::FIRST;
    for mutation in &fields.mutations {
        mutation
            .source
            .validate()
            .map_err(|_| ReactionError::InvalidRecord)?;
        if mutation.actor_principal_id.as_uuid().is_nil()
            || mutation.source.event_id == fields.target_message_event_id
            || previous_source.is_some_and(|previous: MessageSource| {
                mutation.source.event_created_at < previous.event_created_at
            })
            || !sources.insert(mutation.source.event_id)
            || !mutation.resulting_version.follows(previous_version)
        {
            return Err(ReactionError::InvalidRecord);
        }
        previous_source = Some(mutation.source);
        previous_version = mutation.resulting_version;
    }
    if fields.version != previous_version {
        return Err(ReactionError::InvalidRecord);
    }
    fold_mutations(&fields.mutations)?;
    Ok(())
}

fn fold_mutations(
    mutations: &[ReactionMutation],
) -> Result<BTreeMap<(ReactionValue, PrincipalId), MessageSource>, ReactionError> {
    let mut active = BTreeMap::new();
    for mutation in mutations {
        let key = (mutation.value.clone(), mutation.actor_principal_id);
        match mutation.kind {
            ReactionMutationKind::Add => {
                if active.insert(key, mutation.source).is_some() {
                    return Err(ReactionError::InvalidRecord);
                }
            }
            ReactionMutationKind::Remove { added_event_id } => {
                let Some(added_source) = active.remove(&key) else {
                    return Err(ReactionError::InvalidRecord);
                };
                if added_source.event_id != added_event_id {
                    return Err(ReactionError::InvalidRecord);
                }
            }
        }
    }
    Ok(active)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReactionError {
    InvalidValue,
    InvalidSource,
    InvalidRecord,
    TargetMismatch,
    TargetDeleted,
    AddedEventMismatch,
    Message(MessageError),
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTimestamp,
    TooManyMutations,
    VersionExhausted,
}

impl fmt::Display for ReactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue => formatter.write_str("reaction value is invalid"),
            Self::InvalidSource => formatter.write_str("reaction source is invalid"),
            Self::InvalidRecord => formatter.write_str("reaction history is invalid"),
            Self::TargetMismatch | Self::AddedEventMismatch => {
                formatter.write_str("reaction target is invalid")
            }
            Self::TargetDeleted => formatter.write_str("reaction target is deleted"),
            Self::Message(_) => formatter.write_str("reaction command is not authorized"),
            Self::StaleVersion { .. } => formatter.write_str("reaction version is stale"),
            Self::InvalidTimestamp => formatter.write_str("reaction timestamp is invalid"),
            Self::TooManyMutations => formatter.write_str("reaction history is full"),
            Self::VersionExhausted => formatter.write_str("reaction version is exhausted"),
        }
    }
}

impl Error for ReactionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationDenial, AuthorizationResource,
        AuthorizationResourceKind, AuthorizationScope, Channel, ChannelLifecycleState,
        ChannelMembership, ChannelName, ChannelRecordFields, ChannelType, ChannelVisibility,
        CommunityMembership, MembershipRole, MembershipStatus, MessageAuthor, MessageContent,
        MessageCreateFields, MessageDeleteMetadata, PrincipalScopes, ServiceAccountId,
        TenantContext, TrustedTenantRoute,
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
                TrustedTenantRoute::from_listener(community_id(), "reaction-test")
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
        action: AuthorizationAction,
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
            now_millis: 100,
        }
    }

    fn message(authorization: &AuthorizationRequest<'_>) -> Message {
        Message::create(
            MessageCreateFields {
                community_id: community_id(),
                channel_id: aggregate_id(2),
                message_id: aggregate_id(4),
                author: MessageAuthor::principal(principal_id(3)),
                content: MessageContent::new("message").expect("valid content"),
                source: source(1, 10),
            },
            &channel(),
            authorization,
        )
        .expect("authorized message")
    }

    fn version(value: u64) -> AggregateVersion {
        AggregateVersion::new(value).expect("nonzero version")
    }

    #[test]
    fn add_remove_reactivate_and_duplicate_delivery_are_deterministic() {
        let tenant = tenant();
        let scope = scope();
        let actor = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &actor, &scope, AuthorizationAction::Write);
        let message = message(&authorization);
        let mut reactions = ReactionSet::for_message(&message);
        let value = ReactionValue::new("👍").expect("valid reaction");

        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                value.clone(),
                source(2, 11),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Applied)
        );
        assert_eq!(
            reactions.active_groups(&message).expect("groups")[0].count(),
            1
        );
        let added = reactions.clone();

        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                value.clone(),
                source(2, 11),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Unchanged)
        );
        assert_eq!(
            reactions.add(
                version(2),
                value.clone(),
                source(3, 12),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Unchanged)
        );
        assert_eq!(reactions, added);

        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                ReactionValue::new("✅").expect("valid reaction"),
                source(6, 12),
                &message,
                &authorization,
            ),
            Err(ReactionError::StaleVersion {
                expected: AggregateVersion::FIRST,
                actual: version(2),
            })
        );
        assert_eq!(reactions, added);

        assert_eq!(
            reactions.remove(
                version(2),
                value.clone(),
                source(2, 11).event_id,
                source(4, 13),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Applied)
        );
        assert!(
            reactions
                .active_groups(&message)
                .expect("groups")
                .is_empty()
        );
        let removed = reactions.clone();
        assert_eq!(
            reactions.remove(
                version(2),
                value.clone(),
                source(2, 11).event_id,
                source(4, 13),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Unchanged)
        );
        assert_eq!(reactions, removed);

        assert_eq!(
            reactions.add(version(3), value, source(5, 14), &message, &authorization,),
            Ok(ReactionCommandOutcome::Applied)
        );
        assert_eq!(
            reactions.active_groups(&message).expect("groups")[0].count(),
            1
        );
    }

    #[test]
    fn wrapped_sixty_four_byte_custom_emoji_is_preserved() {
        let tenant = tenant();
        let scope = scope();
        let actor = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &actor, &scope, AuthorizationAction::Write);
        let message = message(&authorization);
        let mut reactions = ReactionSet::for_message(&message);
        let custom = format!(":{}:", "a".repeat(64));
        let value = ReactionValue::new(custom.clone()).expect("valid long custom emoji");

        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                value,
                source(2, 11),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Applied)
        );
        let groups = reactions.active_groups(&message).expect("groups");
        assert_eq!(groups[0].value.as_str(), custom);
        assert!(ReactionValue::new("x".repeat(65)).is_err());
        assert!(ReactionValue::new(format!(":{}:", "a".repeat(65))).is_err());
        assert!(ReactionValue::new(format!(":{}:", "A".repeat(64))).is_err());
    }

    #[test]
    fn deleted_target_hides_history_and_rejects_new_reactions() {
        let tenant = tenant();
        let scope = scope();
        let actor = principal(principal_id(3), &scope);
        let authorization = request(&tenant, &actor, &scope, AuthorizationAction::Write);
        let mut message = message(&authorization);
        let mut reactions = ReactionSet::for_message(&message);
        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                ReactionValue::new("👀").expect("valid reaction"),
                source(2, 11),
                &message,
                &authorization,
            ),
            Ok(ReactionCommandOutcome::Applied)
        );

        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(9, 20),
                None::<MessageDeleteMetadata>,
                &authorization,
            ),
            Ok(crate::MessageCommandOutcome::Applied)
        );
        assert!(
            reactions
                .active_groups(&message)
                .expect("groups")
                .is_empty()
        );
        assert_eq!(reactions.fields().mutations.len(), 1);
        let before_rejected_add = reactions.clone();
        assert_eq!(
            reactions.add(
                version(2),
                ReactionValue::new("✅").expect("valid reaction"),
                source(3, 12),
                &message,
                &authorization,
            ),
            Err(ReactionError::TargetDeleted)
        );
        assert_eq!(reactions, before_rejected_add);
    }

    #[test]
    fn missing_channel_membership_fails_closed_without_mutation() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let author_request = request(&tenant, &author, &scope, AuthorizationAction::Write);
        let message = message(&author_request);
        let outsider = principal(principal_id(5), &scope);
        let mut outsider_request = request(&tenant, &outsider, &scope, AuthorizationAction::Write);
        outsider_request.channel_membership = None;
        let mut reactions = ReactionSet::for_message(&message);
        let unchanged = reactions.clone();

        assert_eq!(
            reactions.add(
                AggregateVersion::FIRST,
                ReactionValue::new("👀").expect("valid reaction"),
                source(2, 11),
                &message,
                &outsider_request,
            ),
            Err(ReactionError::Message(MessageError::Unauthorized(
                AuthorizationDenial::MissingChannelMembership
            )))
        );
        assert_eq!(reactions, unchanged);
    }
}
