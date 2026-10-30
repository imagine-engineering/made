use std::{collections::BTreeMap, collections::BTreeSet, error::Error, fmt};

use crate::message::{authorization_subject, authorize_message_command};
use crate::{
    AggregateId, AggregateVersion, AuthorizationAction, AuthorizationRequest, CommunityId, Message,
    MessageError, MessageLifecycleState, MessageSource, NostrEventId, PrincipalId,
};

const MAX_MARKER_MUTATIONS: usize = 10_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MarkerMutationKind {
    Pin,
    Unpin { pinned_event_id: NostrEventId },
    Bookmark,
    Unbookmark { bookmarked_event_id: NostrEventId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkerMutation {
    pub source: MessageSource,
    pub actor_principal_id: PrincipalId,
    pub kind: MarkerMutationKind,
    pub resulting_version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkerRecordFields {
    pub community_id: CommunityId,
    pub channel_id: AggregateId,
    pub message_id: AggregateId,
    pub target_message_event_id: NostrEventId,
    pub mutations: Vec<MarkerMutation>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarkerCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarkerView {
    pub pinned: bool,
    pub bookmarked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageMarkers {
    fields: MarkerRecordFields,
}

impl MessageMarkers {
    pub fn for_message(message: &Message) -> Self {
        Self {
            fields: MarkerRecordFields {
                community_id: message.fields().community_id,
                channel_id: message.fields().channel_id,
                message_id: message.fields().message_id,
                target_message_event_id: message.fields().source.event_id,
                mutations: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        }
    }

    pub fn from_record(fields: MarkerRecordFields) -> Result<Self, MarkerError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &MarkerRecordFields {
        &self.fields
    }

    pub fn pin(
        &mut self,
        expected_version: AggregateVersion,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MarkerCommandOutcome, MarkerError> {
        self.prepare(source, message)?;
        let actor = self.authorize(message, authorization, AuthorizationAction::Manage)?;
        if self.has_source(source.event_id) {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        if message.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Err(MarkerError::TargetDeleted);
        }
        if self.active_state()?.pin.is_some() {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        self.require_mutation(source, expected_version)?;
        self.push(actor, source, MarkerMutationKind::Pin)?;
        Ok(MarkerCommandOutcome::Applied)
    }

    pub fn unpin(
        &mut self,
        expected_version: AggregateVersion,
        pinned_event_id: NostrEventId,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MarkerCommandOutcome, MarkerError> {
        self.prepare(source, message)?;
        let actor = self.authorize(message, authorization, AuthorizationAction::Manage)?;
        if self.has_source(source.event_id) {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        let Some(active_pin) = self.active_state()?.pin else {
            return Ok(MarkerCommandOutcome::Unchanged);
        };
        if active_pin.source.event_id != pinned_event_id {
            return Err(MarkerError::MarkerEventMismatch);
        }
        self.require_mutation(source, expected_version)?;
        self.push(actor, source, MarkerMutationKind::Unpin { pinned_event_id })?;
        Ok(MarkerCommandOutcome::Applied)
    }

    pub fn bookmark(
        &mut self,
        expected_version: AggregateVersion,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MarkerCommandOutcome, MarkerError> {
        self.prepare(source, message)?;
        let actor = self.authorize(message, authorization, AuthorizationAction::Write)?;
        if self.has_source(source.event_id) {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        if message.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Err(MarkerError::TargetDeleted);
        }
        if self.active_state()?.bookmarks.contains_key(&actor) {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        self.require_mutation(source, expected_version)?;
        self.push(actor, source, MarkerMutationKind::Bookmark)?;
        Ok(MarkerCommandOutcome::Applied)
    }

    pub fn unbookmark(
        &mut self,
        expected_version: AggregateVersion,
        bookmarked_event_id: NostrEventId,
        source: MessageSource,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MarkerCommandOutcome, MarkerError> {
        self.prepare(source, message)?;
        let actor = self.authorize(message, authorization, AuthorizationAction::Write)?;
        if self.has_source(source.event_id) {
            return Ok(MarkerCommandOutcome::Unchanged);
        }
        let state = self.active_state()?;
        let Some(active_bookmark) = state.bookmarks.get(&actor) else {
            return Ok(MarkerCommandOutcome::Unchanged);
        };
        if active_bookmark.source.event_id != bookmarked_event_id {
            return Err(MarkerError::MarkerEventMismatch);
        }
        self.require_mutation(source, expected_version)?;
        self.push(
            actor,
            source,
            MarkerMutationKind::Unbookmark {
                bookmarked_event_id,
            },
        )?;
        Ok(MarkerCommandOutcome::Applied)
    }

    pub fn view_for(
        &self,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<MarkerView, MarkerError> {
        self.require_target(message)?;
        let viewer = self.authorize(message, authorization, AuthorizationAction::Read)?;
        if message.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Ok(MarkerView {
                pinned: false,
                bookmarked: false,
            });
        }
        let state = self.active_state()?;
        Ok(MarkerView {
            pinned: state.pin.is_some(),
            bookmarked: state.bookmarks.contains_key(&viewer),
        })
    }

    fn prepare(&self, source: MessageSource, message: &Message) -> Result<(), MarkerError> {
        source.validate().map_err(|_| MarkerError::InvalidSource)?;
        self.require_target(message)
    }

    fn authorize(
        &self,
        message: &Message,
        authorization: &AuthorizationRequest<'_>,
        action: AuthorizationAction,
    ) -> Result<PrincipalId, MarkerError> {
        authorize_message_command(
            authorization,
            message.fields().community_id,
            message.fields().channel_id,
            message.fields().message_id,
            message.fields().author.principal_id(),
            action,
        )
        .map_err(MarkerError::Message)?;
        Ok(authorization_subject(authorization))
    }

    fn require_target(&self, message: &Message) -> Result<(), MarkerError> {
        if self.fields.community_id != message.fields().community_id
            || self.fields.channel_id != message.fields().channel_id
            || self.fields.message_id != message.fields().message_id
            || self.fields.target_message_event_id != message.fields().source.event_id
        {
            return Err(MarkerError::TargetMismatch);
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
    ) -> Result<(), MarkerError> {
        if self.fields.mutations.len() >= MAX_MARKER_MUTATIONS {
            return Err(MarkerError::TooManyMutations);
        }
        if self.fields.version != expected_version {
            return Err(MarkerError::StaleVersion {
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
            return Err(MarkerError::InvalidTimestamp);
        }
        Ok(())
    }

    fn push(
        &mut self,
        actor_principal_id: PrincipalId,
        source: MessageSource,
        kind: MarkerMutationKind,
    ) -> Result<(), MarkerError> {
        let next_version = self
            .fields
            .version
            .next()
            .ok_or(MarkerError::VersionExhausted)?;
        self.fields.mutations.push(MarkerMutation {
            source,
            actor_principal_id,
            kind,
            resulting_version: next_version,
        });
        self.fields.version = next_version;
        Ok(())
    }

    fn active_state(&self) -> Result<ActiveMarkerState, MarkerError> {
        fold_mutations(&self.fields.mutations)
    }
}

#[derive(Clone, Copy)]
struct ActiveMarker {
    source: MessageSource,
}

struct ActiveMarkerState {
    pin: Option<ActiveMarker>,
    bookmarks: BTreeMap<PrincipalId, ActiveMarker>,
}

fn fold_mutations(mutations: &[MarkerMutation]) -> Result<ActiveMarkerState, MarkerError> {
    let mut state = ActiveMarkerState {
        pin: None,
        bookmarks: BTreeMap::new(),
    };
    for mutation in mutations {
        match mutation.kind {
            MarkerMutationKind::Pin => {
                if state
                    .pin
                    .replace(ActiveMarker {
                        source: mutation.source,
                    })
                    .is_some()
                {
                    return Err(MarkerError::InvalidRecord);
                }
            }
            MarkerMutationKind::Unpin { pinned_event_id } => {
                let Some(pin) = state.pin.take() else {
                    return Err(MarkerError::InvalidRecord);
                };
                if pin.source.event_id != pinned_event_id {
                    return Err(MarkerError::InvalidRecord);
                }
            }
            MarkerMutationKind::Bookmark => {
                if state
                    .bookmarks
                    .insert(
                        mutation.actor_principal_id,
                        ActiveMarker {
                            source: mutation.source,
                        },
                    )
                    .is_some()
                {
                    return Err(MarkerError::InvalidRecord);
                }
            }
            MarkerMutationKind::Unbookmark {
                bookmarked_event_id,
            } => {
                let Some(bookmark) = state.bookmarks.remove(&mutation.actor_principal_id) else {
                    return Err(MarkerError::InvalidRecord);
                };
                if bookmark.source.event_id != bookmarked_event_id {
                    return Err(MarkerError::InvalidRecord);
                }
            }
        }
    }
    Ok(state)
}

fn validate_record(fields: &MarkerRecordFields) -> Result<(), MarkerError> {
    if fields.community_id.as_uuid().is_nil()
        || fields.channel_id.as_uuid().is_nil()
        || fields.message_id.as_uuid().is_nil()
        || fields
            .target_message_event_id
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
        || fields.mutations.len() > MAX_MARKER_MUTATIONS
    {
        return Err(MarkerError::InvalidRecord);
    }
    let mut sources = BTreeSet::new();
    let mut previous_source = None;
    let mut previous_version = AggregateVersion::FIRST;
    for mutation in &fields.mutations {
        mutation
            .source
            .validate()
            .map_err(|_| MarkerError::InvalidRecord)?;
        if mutation.actor_principal_id.as_uuid().is_nil()
            || mutation.source.event_id == fields.target_message_event_id
            || previous_source.is_some_and(|previous: MessageSource| {
                mutation.source.event_created_at < previous.event_created_at
            })
            || !sources.insert(mutation.source.event_id)
            || !mutation.resulting_version.follows(previous_version)
        {
            return Err(MarkerError::InvalidRecord);
        }
        previous_source = Some(mutation.source);
        previous_version = mutation.resulting_version;
    }
    if fields.version != previous_version {
        return Err(MarkerError::InvalidRecord);
    }
    fold_mutations(&fields.mutations)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarkerError {
    InvalidSource,
    InvalidRecord,
    TargetMismatch,
    TargetDeleted,
    MarkerEventMismatch,
    Message(MessageError),
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    InvalidTimestamp,
    TooManyMutations,
    VersionExhausted,
}

impl fmt::Display for MarkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource => formatter.write_str("message marker source is invalid"),
            Self::InvalidRecord => formatter.write_str("message marker history is invalid"),
            Self::TargetMismatch | Self::MarkerEventMismatch => {
                formatter.write_str("message marker target is invalid")
            }
            Self::TargetDeleted => formatter.write_str("message marker target is deleted"),
            Self::Message(_) => formatter.write_str("message marker command is not authorized"),
            Self::StaleVersion { .. } => formatter.write_str("message marker version is stale"),
            Self::InvalidTimestamp => formatter.write_str("message marker timestamp is invalid"),
            Self::TooManyMutations => formatter.write_str("message marker history is full"),
            Self::VersionExhausted => formatter.write_str("message marker version is exhausted"),
        }
    }
}

impl Error for MarkerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthenticatedPrincipal, AuthorizationDenial, AuthorizationResource,
        AuthorizationResourceKind, AuthorizationScope, Channel, ChannelLifecycleState,
        ChannelMembership, ChannelName, ChannelRecordFields, ChannelType, ChannelVisibility,
        CommunityMembership, MarkerCommandOutcome, MembershipRole, MembershipStatus, MessageAuthor,
        MessageCommandOutcome, MessageContent, MessageCreateFields, MessageDeleteMetadata,
        PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
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
                TrustedTenantRoute::from_listener(community_id(), "marker-test")
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
                owner_principal_id: Some(principal_id(3)),
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

    #[test]
    fn pin_and_unpin_require_manage_role_and_retry_idempotently() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let author_write = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let message = message(&author_write);
        let mut markers = MessageMarkers::for_message(&message);
        let member_manage = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Manage,
            MembershipRole::Member,
        );

        assert_eq!(
            markers.pin(
                AggregateVersion::FIRST,
                source(2, 11),
                &message,
                &member_manage,
            ),
            Err(MarkerError::Message(MessageError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            )))
        );
        let moderator = principal(principal_id(6), &scope);
        let moderator_manage = request(
            &tenant,
            &moderator,
            &scope,
            AuthorizationAction::Manage,
            MembershipRole::Admin,
        );
        assert_eq!(
            markers.pin(
                AggregateVersion::FIRST,
                source(2, 11),
                &message,
                &moderator_manage,
            ),
            Ok(MarkerCommandOutcome::Applied)
        );
        let author_read = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Read,
            MembershipRole::Member,
        );
        assert_eq!(
            markers.view_for(&message, &author_read),
            Ok(MarkerView {
                pinned: true,
                bookmarked: false,
            })
        );
        let pinned = markers.clone();
        assert_eq!(
            markers.pin(
                AggregateVersion::FIRST,
                source(2, 11),
                &message,
                &moderator_manage,
            ),
            Ok(MarkerCommandOutcome::Unchanged)
        );
        assert_eq!(markers, pinned);
        assert_eq!(
            markers.unpin(
                version(2),
                source(2, 11).event_id,
                source(3, 12),
                &message,
                &moderator_manage,
            ),
            Ok(MarkerCommandOutcome::Applied)
        );
        let unpinned = markers.clone();
        assert_eq!(
            markers.unpin(
                version(2),
                source(2, 11).event_id,
                source(3, 12),
                &message,
                &moderator_manage,
            ),
            Ok(MarkerCommandOutcome::Unchanged)
        );
        assert_eq!(markers, unpinned);
        assert_eq!(
            markers.view_for(&message, &author_read),
            Ok(MarkerView {
                pinned: false,
                bookmarked: false,
            })
        );
    }

    #[test]
    fn bookmarks_are_viewer_private_and_remove_independently() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let author_write = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let message = message(&author_write);
        let mut markers = MessageMarkers::for_message(&message);
        assert_eq!(
            markers.bookmark(
                AggregateVersion::FIRST,
                source(2, 11),
                &message,
                &author_write,
            ),
            Ok(MarkerCommandOutcome::Applied)
        );

        let other = principal(principal_id(5), &scope);
        let other_write = request(
            &tenant,
            &other,
            &scope,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        assert_eq!(
            markers.bookmark(version(2), source(3, 12), &message, &other_write),
            Ok(MarkerCommandOutcome::Applied)
        );
        let author_read = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Read,
            MembershipRole::Member,
        );
        let other_read = request(
            &tenant,
            &other,
            &scope,
            AuthorizationAction::Read,
            MembershipRole::Member,
        );
        let viewer = principal(principal_id(7), &scope);
        let viewer_read = request(
            &tenant,
            &viewer,
            &scope,
            AuthorizationAction::Read,
            MembershipRole::Member,
        );
        assert!(
            markers
                .view_for(&message, &author_read)
                .expect("view")
                .bookmarked
        );
        assert!(
            markers
                .view_for(&message, &other_read)
                .expect("view")
                .bookmarked
        );
        assert!(
            !markers
                .view_for(&message, &viewer_read)
                .expect("view")
                .bookmarked
        );

        assert_eq!(
            markers.unbookmark(
                version(3),
                source(2, 11).event_id,
                source(4, 13),
                &message,
                &author_write,
            ),
            Ok(MarkerCommandOutcome::Applied)
        );
        assert!(
            !markers
                .view_for(&message, &author_read)
                .expect("view")
                .bookmarked
        );
        assert!(
            markers
                .view_for(&message, &other_read)
                .expect("view")
                .bookmarked
        );
    }

    #[test]
    fn deleted_target_hides_markers_rejects_adds_and_retains_history() {
        let tenant = tenant();
        let scope = scope();
        let author = principal(principal_id(3), &scope);
        let author_write = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Write,
            MembershipRole::Member,
        );
        let mut message = message(&author_write);
        let mut markers = MessageMarkers::for_message(&message);
        assert_eq!(
            markers.bookmark(
                AggregateVersion::FIRST,
                source(2, 11),
                &message,
                &author_write,
            ),
            Ok(MarkerCommandOutcome::Applied)
        );
        assert_eq!(
            message.delete(
                AggregateVersion::FIRST,
                source(9, 20),
                None::<MessageDeleteMetadata>,
                &author_write,
            ),
            Ok(MessageCommandOutcome::Applied)
        );
        let author_read = request(
            &tenant,
            &author,
            &scope,
            AuthorizationAction::Read,
            MembershipRole::Member,
        );
        assert_eq!(
            markers.view_for(&message, &author_read),
            Ok(MarkerView {
                pinned: false,
                bookmarked: false,
            })
        );
        let retained = markers.clone();
        assert_eq!(
            markers.bookmark(version(2), source(3, 12), &message, &author_write),
            Err(MarkerError::TargetDeleted)
        );
        assert_eq!(markers, retained);
        assert_eq!(markers.fields().mutations.len(), 1);
    }
}
