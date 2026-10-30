use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::message::{authorization_subject, authorize_message_command};
use crate::{
    AggregateId, AuthorizationAction, AuthorizationDecision, AuthorizationDenial,
    AuthorizationRequest, AuthorizationResourceKind, Channel, ChannelLifecycleState, ChannelType,
    CommunityId, Message, MessageError, MessageLifecycleState, MessageSource, NostrEventId,
    NostrPublicKey, PrincipalId, ThreadCursor, ThreadError, ThreadEvent, ThreadGraph,
    ThreadReference, ThreadSummary, authorize,
};

pub const MAX_FORUM_MESSAGES: usize = 100_000;
pub const MAX_FORUM_VOTES: usize = 100_000;
pub const MAX_FORUM_POST_PAGE_ROWS: usize = 200;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForumMessageInput<'a> {
    pub message: &'a Message,
    pub author_public_key: NostrPublicKey,
    pub reference: ThreadReference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForumVoteDirection {
    Up,
    Down,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumVote {
    community_id: CommunityId,
    channel_id: AggregateId,
    target_message_id: AggregateId,
    target_event_id: NostrEventId,
    voter_principal_id: PrincipalId,
    direction: ForumVoteDirection,
    source: MessageSource,
}

impl ForumVote {
    pub fn cast(
        channel: &Channel,
        target: &Message,
        direction: ForumVoteDirection,
        source: MessageSource,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<Self, ForumError> {
        require_writable_forum(channel)?;
        source
            .validate()
            .map_err(|_| ForumError::InvalidVoteSource)?;
        if target.fields().community_id != channel.fields().community_id
            || target.fields().channel_id != channel.fields().channel_id
        {
            return Err(ForumError::MessageScopeMismatch);
        }
        if target.fields().lifecycle_state == MessageLifecycleState::Deleted {
            return Err(ForumError::VoteTargetDeleted);
        }
        authorize_message_command(
            authorization,
            target.fields().community_id,
            target.fields().channel_id,
            target.fields().message_id,
            target.fields().author.principal_id(),
            AuthorizationAction::Write,
        )
        .map_err(ForumError::Message)?;
        Ok(Self {
            community_id: target.fields().community_id,
            channel_id: target.fields().channel_id,
            target_message_id: target.fields().message_id,
            target_event_id: target.fields().source.event_id,
            voter_principal_id: authorization_subject(authorization),
            direction,
            source,
        })
    }

    pub const fn target_event_id(&self) -> NostrEventId {
        self.target_event_id
    }

    pub const fn voter_principal_id(&self) -> PrincipalId {
        self.voter_principal_id
    }

    pub const fn direction(&self) -> ForumVoteDirection {
        self.direction
    }

    pub const fn source(&self) -> MessageSource {
        self.source
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForumVoteSummary {
    pub upvotes: u64,
    pub downvotes: u64,
    pub score: i64,
    pub viewer_vote: Option<ForumVoteDirection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForumPostCursor {
    pub created_at: u64,
    pub event_id: NostrEventId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumPost<'a> {
    pub message: &'a Message,
    pub thread_summary: Option<ThreadSummary>,
    pub votes: ForumVoteSummary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumComment<'a> {
    pub message: &'a Message,
    pub parent_event_id: NostrEventId,
    pub root_event_id: NostrEventId,
    pub depth: u16,
    pub votes: ForumVoteSummary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumPostPage<'a> {
    pub posts: Vec<ForumPost<'a>>,
    pub has_more: bool,
    pub next_cursor: Option<ForumPostCursor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumThreadPage<'a> {
    pub post: ForumPost<'a>,
    pub comments: Vec<ForumComment<'a>>,
    pub total_comments: u64,
    pub has_more: bool,
    pub next_cursor: Option<ThreadCursor>,
}

pub struct ForumProjection<'a> {
    community_id: CommunityId,
    channel_id: AggregateId,
    archived: bool,
    viewer_principal_id: PrincipalId,
    messages: BTreeMap<NostrEventId, &'a Message>,
    thread_graph: ThreadGraph,
    votes: BTreeMap<NostrEventId, ForumVoteSummary>,
}

impl<'a> ForumProjection<'a> {
    pub fn build(
        channel: &Channel,
        authorization: &AuthorizationRequest<'_>,
        messages: impl IntoIterator<Item = ForumMessageInput<'a>>,
        votes: impl IntoIterator<Item = ForumVote>,
    ) -> Result<Self, ForumError> {
        require_readable_forum(channel, authorization)?;
        let community_id = channel.fields().community_id;
        let channel_id = channel.fields().channel_id;
        let viewer_principal_id = authorization_subject(authorization);

        let mut message_ids = BTreeSet::new();
        let mut messages_by_event = BTreeMap::new();
        let mut thread_events = Vec::new();
        for (index, input) in messages.into_iter().enumerate() {
            if index >= MAX_FORUM_MESSAGES {
                return Err(ForumError::TooManyMessages);
            }
            let fields = input.message.fields();
            if fields.community_id != community_id || fields.channel_id != channel_id {
                return Err(ForumError::MessageScopeMismatch);
            }
            if !message_ids.insert(fields.message_id) {
                return Err(ForumError::DuplicateMessageId(fields.message_id));
            }
            if messages_by_event
                .insert(fields.source.event_id, input.message)
                .is_some()
            {
                return Err(ForumError::DuplicateEventId(fields.source.event_id));
            }
            thread_events.push(ThreadEvent {
                event_id: fields.source.event_id,
                channel_id,
                author: input.author_public_key,
                created_at: fields.source.event_created_at,
                reference: input.reference,
                broadcast: false,
                deleted: fields.lifecycle_state == MessageLifecycleState::Deleted,
            });
        }
        let thread_graph = ThreadGraph::build(thread_events).map_err(ForumError::Thread)?;
        validate_forum_threads(&thread_graph)?;
        let votes = project_votes(
            community_id,
            channel_id,
            viewer_principal_id,
            &messages_by_event,
            votes,
        )?;

        Ok(Self {
            community_id,
            channel_id,
            archived: channel.fields().lifecycle_state == ChannelLifecycleState::Archived,
            viewer_principal_id,
            messages: messages_by_event,
            thread_graph,
            votes,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(&self) -> AggregateId {
        self.channel_id
    }

    pub const fn viewer_principal_id(&self) -> PrincipalId {
        self.viewer_principal_id
    }

    pub const fn is_archived(&self) -> bool {
        self.archived
    }

    pub fn posts(
        &self,
        cursor: Option<ForumPostCursor>,
        requested_limit: usize,
    ) -> Result<ForumPostPage<'a>, ForumError> {
        let limit = requested_limit.clamp(1, MAX_FORUM_POST_PAGE_ROWS);
        let mut roots = self
            .thread_graph
            .nodes()
            .filter(|node| {
                matches!(node.event.reference, ThreadReference::TopLevel) && !node.event.deleted
            })
            .filter(|node| cursor.is_none_or(|cursor| post_follows_cursor(*node, cursor)))
            .collect::<Vec<_>>();
        roots.sort_by(|left, right| {
            right
                .event
                .created_at
                .cmp(&left.event.created_at)
                .then_with(|| left.event.event_id.cmp(&right.event.event_id))
        });
        let has_more = roots.len() > limit;
        roots.truncate(limit);
        let next_cursor = has_more.then(|| {
            roots
                .last()
                .map(|node| ForumPostCursor {
                    created_at: node.event.created_at,
                    event_id: node.event.event_id,
                })
                .ok_or(ForumError::ProjectionInvariant)
        });
        let next_cursor = next_cursor.transpose()?;
        let posts = roots
            .into_iter()
            .map(|node| self.post(node.event.event_id))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ForumPostPage {
            posts,
            has_more,
            next_cursor,
        })
    }

    pub fn thread(
        &self,
        root_event_id: NostrEventId,
        cursor: Option<ThreadCursor>,
        requested_limit: usize,
    ) -> Result<ForumThreadPage<'a>, ForumError> {
        let root = self
            .thread_graph
            .node(root_event_id)
            .ok_or(ForumError::MissingPost(root_event_id))?;
        if !matches!(root.event.reference, ThreadReference::TopLevel) {
            return Err(ForumError::NotForumPost(root_event_id));
        }
        if root.event.deleted {
            return Err(ForumError::PostDeleted(root_event_id));
        }
        let total_comments = self
            .thread_graph
            .summary(root_event_id)
            .map_err(ForumError::Thread)?
            .map_or(0, |summary| summary.descendant_count);
        let page = self
            .thread_graph
            .page(root_event_id, cursor, requested_limit, None)
            .map_err(ForumError::Thread)?;
        let comments = page
            .replies
            .into_iter()
            .map(|node| {
                let parent_event_id = node
                    .parent_event_id
                    .ok_or(ForumError::ProjectionInvariant)?;
                let message = self
                    .messages
                    .get(&node.event.event_id)
                    .copied()
                    .ok_or(ForumError::ProjectionInvariant)?;
                Ok(ForumComment {
                    message,
                    parent_event_id,
                    root_event_id: node.root_event_id,
                    depth: node.depth,
                    votes: self.vote_summary(node.event.event_id),
                })
            })
            .collect::<Result<Vec<_>, ForumError>>()?;
        Ok(ForumThreadPage {
            post: self.post(root_event_id)?,
            comments,
            total_comments,
            has_more: page.has_more,
            next_cursor: page.next_cursor,
        })
    }

    fn post(&self, event_id: NostrEventId) -> Result<ForumPost<'a>, ForumError> {
        let message = self
            .messages
            .get(&event_id)
            .copied()
            .ok_or(ForumError::ProjectionInvariant)?;
        let thread_summary = self
            .thread_graph
            .summary(event_id)
            .map_err(ForumError::Thread)?;
        Ok(ForumPost {
            message,
            thread_summary,
            votes: self.vote_summary(event_id),
        })
    }

    fn vote_summary(&self, event_id: NostrEventId) -> ForumVoteSummary {
        self.votes.get(&event_id).copied().unwrap_or_default()
    }
}

fn require_readable_forum(
    channel: &Channel,
    authorization: &AuthorizationRequest<'_>,
) -> Result<(), ForumError> {
    require_forum_channel(channel)?;
    if matches!(
        channel.fields().lifecycle_state,
        ChannelLifecycleState::Deleted | ChannelLifecycleState::Expired
    ) {
        return Err(ForumError::ChannelUnavailable);
    }
    if authorization.action != AuthorizationAction::Read
        || authorization.resource.community_id != channel.fields().community_id
        || authorization.resource.kind != AuthorizationResourceKind::Channel
        || authorization.resource.resource_id != channel.fields().channel_id
        || authorization.resource.channel_id != Some(channel.fields().channel_id)
    {
        return Err(ForumError::AuthorizationShape);
    }
    match authorize(authorization) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(ForumError::Unauthorized(denial)),
    }
}

fn require_writable_forum(channel: &Channel) -> Result<(), ForumError> {
    require_forum_channel(channel)?;
    if channel.fields().lifecycle_state != ChannelLifecycleState::Active {
        return Err(ForumError::ChannelUnavailable);
    }
    Ok(())
}

fn require_forum_channel(channel: &Channel) -> Result<(), ForumError> {
    if channel.fields().channel_type != ChannelType::Forum {
        return Err(ForumError::NotForumChannel);
    }
    Ok(())
}

fn validate_forum_threads(graph: &ThreadGraph) -> Result<(), ForumError> {
    for node in graph.nodes() {
        match node.event.reference {
            ThreadReference::TopLevel if node.depth == 0 => {}
            ThreadReference::Reply { .. } if node.depth > 0 => {
                let root = graph
                    .node(node.root_event_id)
                    .ok_or(ForumError::ProjectionInvariant)?;
                if !matches!(root.event.reference, ThreadReference::TopLevel) {
                    return Err(ForumError::NotForumPost(node.root_event_id));
                }
            }
            _ => return Err(ForumError::InvalidForumReference(node.event.event_id)),
        }
    }
    Ok(())
}

fn project_votes(
    community_id: CommunityId,
    channel_id: AggregateId,
    viewer_principal_id: PrincipalId,
    messages: &BTreeMap<NostrEventId, &Message>,
    votes: impl IntoIterator<Item = ForumVote>,
) -> Result<BTreeMap<NostrEventId, ForumVoteSummary>, ForumError> {
    let mut votes_by_source = BTreeMap::new();
    for (index, vote) in votes.into_iter().enumerate() {
        if index >= MAX_FORUM_VOTES {
            return Err(ForumError::TooManyVotes);
        }
        if vote.community_id != community_id || vote.channel_id != channel_id {
            return Err(ForumError::VoteScopeMismatch);
        }
        let target = messages
            .get(&vote.target_event_id)
            .copied()
            .ok_or(ForumError::MissingVoteTarget(vote.target_event_id))?;
        if target.fields().message_id != vote.target_message_id {
            return Err(ForumError::VoteTargetMismatch);
        }
        match votes_by_source.entry(vote.source.event_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(vote);
            }
            std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &vote => {}
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(ForumError::ConflictingVoteSource);
            }
        }
    }

    let mut current = BTreeMap::<(NostrEventId, PrincipalId), ForumVote>::new();
    for vote in votes_by_source.into_values() {
        let key = (vote.target_event_id, vote.voter_principal_id);
        match current.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(vote);
            }
            std::collections::btree_map::Entry::Occupied(mut entry)
                if vote_replaces(&vote, entry.get()) =>
            {
                entry.insert(vote);
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }

    let mut summaries = BTreeMap::<NostrEventId, ForumVoteSummary>::new();
    for vote in current.into_values() {
        let target = messages
            .get(&vote.target_event_id)
            .copied()
            .ok_or(ForumError::ProjectionInvariant)?;
        if target.fields().lifecycle_state == MessageLifecycleState::Deleted {
            continue;
        }
        let summary = summaries.entry(vote.target_event_id).or_default();
        match vote.direction {
            ForumVoteDirection::Up => summary.upvotes += 1,
            ForumVoteDirection::Down => summary.downvotes += 1,
        }
        if vote.voter_principal_id == viewer_principal_id {
            summary.viewer_vote = Some(vote.direction);
        }
    }
    for summary in summaries.values_mut() {
        summary.score = i64::try_from(summary.upvotes)
            .and_then(|upvotes| {
                i64::try_from(summary.downvotes).map(|downvotes| upvotes - downvotes)
            })
            .map_err(|_| ForumError::ProjectionInvariant)?;
    }
    Ok(summaries)
}

fn vote_replaces(candidate: &ForumVote, current: &ForumVote) -> bool {
    candidate.source.event_created_at > current.source.event_created_at
        || (candidate.source.event_created_at == current.source.event_created_at
            && candidate.source.event_id < current.source.event_id)
}

fn post_follows_cursor(node: crate::ThreadNode, cursor: ForumPostCursor) -> bool {
    node.event.created_at < cursor.created_at
        || (node.event.created_at == cursor.created_at && node.event.event_id > cursor.event_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForumError {
    NotForumChannel,
    ChannelUnavailable,
    AuthorizationShape,
    Unauthorized(AuthorizationDenial),
    Message(MessageError),
    Thread(ThreadError),
    TooManyMessages,
    TooManyVotes,
    MessageScopeMismatch,
    DuplicateMessageId(AggregateId),
    DuplicateEventId(NostrEventId),
    InvalidForumReference(NostrEventId),
    InvalidVoteSource,
    VoteTargetDeleted,
    VoteScopeMismatch,
    MissingVoteTarget(NostrEventId),
    VoteTargetMismatch,
    ConflictingVoteSource,
    MissingPost(NostrEventId),
    NotForumPost(NostrEventId),
    PostDeleted(NostrEventId),
    ProjectionInvariant,
}

impl fmt::Display for ForumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotForumChannel => formatter.write_str("channel is not a forum"),
            Self::ChannelUnavailable => formatter.write_str("forum channel is unavailable"),
            Self::AuthorizationShape | Self::Unauthorized(_) | Self::Message(_) => {
                formatter.write_str("forum operation is not authorized")
            }
            Self::Thread(_)
            | Self::MessageScopeMismatch
            | Self::DuplicateMessageId(_)
            | Self::DuplicateEventId(_)
            | Self::InvalidForumReference(_)
            | Self::ProjectionInvariant => formatter.write_str("forum message graph is invalid"),
            Self::TooManyMessages => formatter.write_str("too many forum messages"),
            Self::TooManyVotes => formatter.write_str("too many forum votes"),
            Self::InvalidVoteSource
            | Self::VoteTargetDeleted
            | Self::VoteScopeMismatch
            | Self::MissingVoteTarget(_)
            | Self::VoteTargetMismatch
            | Self::ConflictingVoteSource => formatter.write_str("forum vote is invalid"),
            Self::MissingPost(_) | Self::NotForumPost(_) | Self::PostDeleted(_) => {
                formatter.write_str("forum post is unavailable")
            }
        }
    }
}

impl Error for ForumError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AggregateVersion, AuthenticatedPrincipal, AuthorizationResource, AuthorizationScope,
        ChannelMembership, ChannelName, ChannelRecordFields, ChannelVisibility,
        CommunityMembership, MembershipRole, MembershipStatus, MessageAuthor, MessageContent,
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

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn source(value: u8, created_at: u64) -> MessageSource {
        MessageSource {
            event_id: event_id(value),
            event_created_at: created_at,
        }
    }

    fn channel(lifecycle_state: ChannelLifecycleState) -> Channel {
        Channel::from_record(ChannelRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            name: ChannelName::new("discussions").expect("valid channel name"),
            channel_type: ChannelType::Forum,
            visibility: ChannelVisibility::Open,
            lifecycle_state,
            description: None,
            creator_principal_id: principal_id(3),
            expiration: None,
            version: AggregateVersion::FIRST,
        })
        .expect("valid forum channel")
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id(), "forum-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("forum:access").expect("valid scope")
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
        resource_kind: AuthorizationResourceKind,
        resource_id: AggregateId,
        owner_principal_id: Option<PrincipalId>,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: resource_kind,
                resource_id,
                owner_principal_id,
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

    fn message(message_value: u128, event_value: u8, created_at: u64) -> Message {
        Message::from_record(crate::MessageRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            message_id: aggregate_id(message_value),
            author: MessageAuthor::principal(principal_id(10 + message_value)),
            content: MessageContent::new(format!("forum message {message_value}"))
                .expect("valid content"),
            lifecycle_state: MessageLifecycleState::Active,
            source: source(event_value, created_at),
            current_source: source(event_value, created_at),
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("valid message")
    }

    fn read_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        request(
            tenant,
            principal,
            scope,
            AuthorizationAction::Read,
            AuthorizationResourceKind::Channel,
            aggregate_id(2),
            None,
            role,
        )
    }

    fn vote_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
        target: &Message,
        role: MembershipRole,
    ) -> AuthorizationRequest<'a> {
        request(
            tenant,
            principal,
            scope,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Conversation,
            target.fields().message_id,
            Some(target.fields().author.principal_id()),
            role,
        )
    }

    fn input(message: &Message, reference: ThreadReference) -> ForumMessageInput<'_> {
        let key_byte = message.fields().source.event_id.as_bytes()[0];
        ForumMessageInput {
            message,
            author_public_key: NostrPublicKey::from_bytes([key_byte; 32]),
            reference,
        }
    }

    #[test]
    fn latest_authorized_vote_replaces_prior_direction_with_stable_ties() {
        let active_channel = channel(ChannelLifecycleState::Active);
        let root = message(100, 1, 10);
        let tenant = tenant();
        let scope = scope();
        let viewer = principal(principal_id(20), &scope);
        let other = principal(principal_id(21), &scope);
        let viewer_vote = vote_request(&tenant, &viewer, &scope, &root, MembershipRole::Member);
        let other_vote = vote_request(&tenant, &other, &scope, &root, MembershipRole::Member);
        let old_up = ForumVote::cast(
            &active_channel,
            &root,
            ForumVoteDirection::Up,
            source(20, 20),
            &viewer_vote,
        )
        .expect("authorized vote");
        let newest_down = ForumVote::cast(
            &active_channel,
            &root,
            ForumVoteDirection::Down,
            source(22, 21),
            &viewer_vote,
        )
        .expect("authorized replacement vote");
        let tied_up = ForumVote::cast(
            &active_channel,
            &root,
            ForumVoteDirection::Up,
            source(21, 21),
            &viewer_vote,
        )
        .expect("authorized tied vote");
        let other_down = ForumVote::cast(
            &active_channel,
            &root,
            ForumVoteDirection::Down,
            source(23, 22),
            &other_vote,
        )
        .expect("authorized second voter");
        let viewer_read = read_request(&tenant, &viewer, &scope, MembershipRole::Member);
        let projection = ForumProjection::build(
            &active_channel,
            &viewer_read,
            [input(&root, ThreadReference::TopLevel)],
            [old_up, newest_down, tied_up.clone(), tied_up, other_down],
        )
        .expect("valid forum projection");
        let page = projection.posts(None, 10).expect("post page");

        assert_eq!(page.posts.len(), 1);
        assert_eq!(
            page.posts[0].votes,
            ForumVoteSummary {
                upvotes: 1,
                downvotes: 1,
                score: 0,
                viewer_vote: Some(ForumVoteDirection::Up),
            }
        );

        let bot = principal(principal_id(22), &scope);
        let bot_vote = vote_request(&tenant, &bot, &scope, &root, MembershipRole::Bot);
        assert!(matches!(
            ForumVote::cast(
                &active_channel,
                &root,
                ForumVoteDirection::Up,
                source(24, 23),
                &bot_vote,
            ),
            Err(ForumError::Message(MessageError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            )))
        ));
    }

    #[test]
    fn deleted_comments_preserve_links_and_exact_thread_continuation() {
        let channel = channel(ChannelLifecycleState::Active);
        let root = message(100, 1, 10);
        let mut deleted_comment = message(101, 2, 20);
        let nested_comment = message(102, 3, 21);
        let direct_comment = message(103, 4, 20);
        let tenant = tenant();
        let scope = scope();
        let author = principal(deleted_comment.fields().author.principal_id(), &scope);
        let delete_request = vote_request(
            &tenant,
            &author,
            &scope,
            &deleted_comment,
            MembershipRole::Member,
        );
        deleted_comment
            .delete(
                AggregateVersion::FIRST,
                source(30, 30),
                None,
                &delete_request,
            )
            .expect("author deletes comment");
        let viewer = principal(principal_id(20), &scope);
        let viewer_read = read_request(&tenant, &viewer, &scope, MembershipRole::Member);
        let projection = ForumProjection::build(
            &channel,
            &viewer_read,
            [
                input(&root, ThreadReference::TopLevel),
                input(
                    &deleted_comment,
                    ThreadReference::Reply {
                        parent_event_id: event_id(1),
                        root_event_id: Some(event_id(1)),
                    },
                ),
                input(
                    &nested_comment,
                    ThreadReference::Reply {
                        parent_event_id: event_id(2),
                        root_event_id: Some(event_id(1)),
                    },
                ),
                input(
                    &direct_comment,
                    ThreadReference::Reply {
                        parent_event_id: event_id(1),
                        root_event_id: Some(event_id(1)),
                    },
                ),
            ],
            [],
        )
        .expect("valid forum projection");

        let first = projection
            .thread(event_id(1), None, 1)
            .expect("first comment page");
        assert!(first.has_more);
        assert_eq!(first.total_comments, 2);
        assert_eq!(
            first.comments[0].message.fields().source.event_id,
            event_id(4)
        );
        let second = projection
            .thread(event_id(1), first.next_cursor, 1)
            .expect("second comment page");
        assert!(!second.has_more);
        assert_eq!(second.comments.len(), 1);
        assert_eq!(
            second.comments[0].message.fields().source.event_id,
            event_id(3)
        );
        assert_eq!(second.comments[0].parent_event_id, event_id(2));
        assert_eq!(second.comments[0].root_event_id, event_id(1));
        assert_eq!(second.comments[0].depth, 2);
    }

    #[test]
    fn post_pagination_and_channel_visibility_are_stable() {
        let active_channel = channel(ChannelLifecycleState::Active);
        let first = message(100, 1, 100);
        let second = message(101, 2, 100);
        let third = message(102, 3, 100);
        let newer = message(103, 4, 101);
        let tenant = tenant();
        let scope = scope();
        let viewer = principal(principal_id(20), &scope);
        let viewer_read = read_request(&tenant, &viewer, &scope, MembershipRole::Member);
        let projection = ForumProjection::build(
            &active_channel,
            &viewer_read,
            [
                input(&third, ThreadReference::TopLevel),
                input(&first, ThreadReference::TopLevel),
                input(&second, ThreadReference::TopLevel),
            ],
            [],
        )
        .expect("valid forum projection");
        let page = projection.posts(None, 2).expect("first post page");
        assert_eq!(
            page.posts
                .iter()
                .map(|post| post.message.fields().source.event_id)
                .collect::<Vec<_>>(),
            vec![event_id(1), event_id(2)]
        );

        let with_concurrent_newer = ForumProjection::build(
            &active_channel,
            &viewer_read,
            [
                input(&newer, ThreadReference::TopLevel),
                input(&third, ThreadReference::TopLevel),
                input(&first, ThreadReference::TopLevel),
                input(&second, ThreadReference::TopLevel),
            ],
            [],
        )
        .expect("updated forum projection");
        let continuation = with_concurrent_newer
            .posts(page.next_cursor, 2)
            .expect("stable continuation");
        assert_eq!(continuation.posts.len(), 1);
        assert_eq!(
            continuation.posts[0].message.fields().source.event_id,
            event_id(3)
        );

        let bot = principal(principal_id(21), &scope);
        let denied_read = read_request(&tenant, &bot, &scope, MembershipRole::Bot);
        assert!(matches!(
            ForumProjection::build(
                &active_channel,
                &denied_read,
                [input(&first, ThreadReference::TopLevel)],
                [],
            ),
            Err(ForumError::Unauthorized(
                AuthorizationDenial::InsufficientRole
            ))
        ));

        let archived = channel(ChannelLifecycleState::Archived);
        let archived_projection = ForumProjection::build(
            &archived,
            &viewer_read,
            [input(&first, ThreadReference::TopLevel)],
            [],
        )
        .expect("archived forum remains readable");
        assert!(archived_projection.is_archived());
        let vote = vote_request(&tenant, &viewer, &scope, &first, MembershipRole::Member);
        assert_eq!(
            ForumVote::cast(
                &archived,
                &first,
                ForumVoteDirection::Up,
                source(40, 40),
                &vote,
            ),
            Err(ForumError::ChannelUnavailable)
        );
    }
}
