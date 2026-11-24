use std::collections::VecDeque;

use agent_client_protocol::schema::v1 as acp;
use collaboration_domain::{
    AggregateId, AuthenticatedPrincipalKind, AuthorizationAction, AuthorizationDecision,
    AuthorizationRequest, AuthorizationResourceKind, Message, MessageAuthor, NostrEventId,
    PrincipalId, authorize,
};
use collections::{HashMap, HashSet};

use crate::collaboration_session::{
    CollaborationSessionError, CollaborationSessionIdentity, CollaborationSessionLease,
    CollaborationSessionRegistry, CollaborationSessionScope,
};

const MAX_TEAM_MENTION_MEMBERS: usize = 256;
const MAX_PROCESSED_MENTIONS: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedCollaborationMentionTarget {
    Direct(PrincipalId),
    Team {
        team_id: AggregateId,
        members: Vec<PrincipalId>,
    },
}

impl ResolvedCollaborationMentionTarget {
    pub const fn direct(principal_id: PrincipalId) -> Self {
        Self::Direct(principal_id)
    }

    pub fn team(
        team_id: AggregateId,
        members: impl IntoIterator<Item = PrincipalId>,
    ) -> Result<Self, CollaborationMentionError> {
        let members = members.into_iter().collect::<Vec<_>>();
        let unique_members = members.iter().copied().collect::<HashSet<_>>();
        if team_id.as_uuid().is_nil()
            || members.is_empty()
            || members.len() > MAX_TEAM_MENTION_MEMBERS
            || unique_members.len() != members.len()
            || members
                .iter()
                .any(|principal_id| principal_id.as_uuid().is_nil())
        {
            return Err(CollaborationMentionError::InvalidMentionTarget);
        }
        Ok(Self::Team { team_id, members })
    }

    fn addresses(&self, principal_id: PrincipalId) -> bool {
        match self {
            Self::Direct(target) => *target == principal_id,
            Self::Team { members, .. } => members.contains(&principal_id),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RoutedMentionId {
    identity: CollaborationSessionIdentity,
    event_id: NostrEventId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActiveMention {
    event_id: NostrEventId,
    generation: u64,
}

#[derive(Clone, Debug)]
pub struct CollaborationPromptDispatch {
    identity: CollaborationSessionIdentity,
    event_id: NostrEventId,
    generation: u64,
    request: acp::PromptRequest,
}

impl CollaborationPromptDispatch {
    pub fn request(&self) -> &acp::PromptRequest {
        &self.request
    }

    pub fn identity(&self) -> CollaborationSessionIdentity {
        self.identity
    }

    pub const fn event_id(&self) -> NostrEventId {
        self.event_id
    }
}

pub struct CollaborationMentionRouter {
    agent_principal_id: PrincipalId,
    active_mentions: HashMap<CollaborationSessionIdentity, ActiveMention>,
    processed_mentions: HashSet<RoutedMentionId>,
    processed_order: VecDeque<RoutedMentionId>,
    next_generation: u64,
}

impl CollaborationMentionRouter {
    pub fn new(agent_principal_id: PrincipalId) -> Result<Self, CollaborationMentionError> {
        if agent_principal_id.as_uuid().is_nil() {
            return Err(CollaborationMentionError::InvalidAgent);
        }
        Ok(Self {
            agent_principal_id,
            active_mentions: HashMap::default(),
            processed_mentions: HashSet::default(),
            processed_order: VecDeque::new(),
            next_generation: 0,
        })
    }

    pub fn begin_prompt(
        &mut self,
        message: &Message,
        target: &ResolvedCollaborationMentionTarget,
        authorization: &AuthorizationRequest<'_>,
        sessions: &CollaborationSessionRegistry,
        lease: &CollaborationSessionLease,
    ) -> Result<CollaborationPromptDispatch, CollaborationMentionError> {
        authorize_message(message, authorization)?;
        if !target.addresses(self.agent_principal_id) {
            return Err(CollaborationMentionError::UnsupportedMention);
        }

        let fields = message.fields();
        if !identity_matches_message(lease.identity(), message) {
            return Err(CollaborationMentionError::SessionMismatch);
        }
        let content = message
            .visible_content()
            .ok_or(CollaborationMentionError::MessageUnavailable)?
            .as_str();
        if content.trim().is_empty() {
            return Err(CollaborationMentionError::EmptyPrompt);
        }
        let session_id = sessions.active_session_for_lease(lease)?.clone();
        let routed_id = RoutedMentionId {
            identity: lease.identity(),
            event_id: fields.source.event_id,
        };
        if self.processed_mentions.contains(&routed_id)
            || self
                .active_mentions
                .get(&routed_id.identity)
                .is_some_and(|active| active.event_id == routed_id.event_id)
        {
            return Err(CollaborationMentionError::DuplicateEvent);
        }
        if self.active_mentions.contains_key(&routed_id.identity) {
            return Err(CollaborationMentionError::BusySession);
        }

        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CollaborationMentionError::GenerationExhausted)?;
        self.next_generation = generation;
        self.active_mentions.insert(
            routed_id.identity,
            ActiveMention {
                event_id: routed_id.event_id,
                generation,
            },
        );
        Ok(CollaborationPromptDispatch {
            identity: routed_id.identity,
            event_id: routed_id.event_id,
            generation,
            request: acp::PromptRequest::new(session_id, vec![content.into()]),
        })
    }

    pub fn complete_prompt(
        &mut self,
        dispatch: &CollaborationPromptDispatch,
    ) -> Result<(), CollaborationMentionError> {
        self.remove_current(dispatch)?;
        let routed_id = RoutedMentionId {
            identity: dispatch.identity,
            event_id: dispatch.event_id,
        };
        if self.processed_mentions.insert(routed_id.clone()) {
            self.processed_order.push_back(routed_id);
        }
        while self.processed_order.len() > MAX_PROCESSED_MENTIONS {
            if let Some(expired) = self.processed_order.pop_front() {
                self.processed_mentions.remove(&expired);
            }
        }
        Ok(())
    }

    pub fn abort_prompt(
        &mut self,
        dispatch: &CollaborationPromptDispatch,
    ) -> Result<(), CollaborationMentionError> {
        self.remove_current(dispatch)
    }

    fn remove_current(
        &mut self,
        dispatch: &CollaborationPromptDispatch,
    ) -> Result<(), CollaborationMentionError> {
        let Some(active) = self.active_mentions.get(&dispatch.identity) else {
            return Err(CollaborationMentionError::DispatchNotCurrent);
        };
        if active.event_id != dispatch.event_id || active.generation != dispatch.generation {
            return Err(CollaborationMentionError::DispatchNotCurrent);
        }
        self.active_mentions.remove(&dispatch.identity);
        Ok(())
    }
}

fn authorize_message(
    message: &Message,
    request: &AuthorizationRequest<'_>,
) -> Result<(), CollaborationMentionError> {
    let fields = message.fields();
    let subject = match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    };
    if request.action != AuthorizationAction::Write
        || request.resource.community_id != fields.community_id
        || request.resource.kind != AuthorizationResourceKind::Conversation
        || request.resource.resource_id != fields.message_id
        || request.resource.owner_principal_id != Some(fields.author.principal_id())
        || request.resource.channel_id != Some(fields.channel_id)
        || subject != fields.author.principal_id()
    {
        return Err(CollaborationMentionError::Unauthorized);
    }
    match (fields.author, request.principal.kind()) {
        (
            MessageAuthor::OwnerAttestedAgent { proof_event_id, .. },
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                proof_event_id: authenticated_proof_event_id,
                ..
            },
        ) if proof_event_id == *authenticated_proof_event_id => {}
        (MessageAuthor::OwnerAttestedAgent { .. }, _)
        | (_, AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }) => {
            return Err(CollaborationMentionError::Unauthorized);
        }
        _ => {}
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(_) => Err(CollaborationMentionError::Unauthorized),
    }
}

fn identity_matches_message(identity: CollaborationSessionIdentity, message: &Message) -> bool {
    let fields = message.fields();
    let channel_id = match identity.scope() {
        CollaborationSessionScope::Channel { channel_id }
        | CollaborationSessionScope::Thread { channel_id, .. }
        | CollaborationSessionScope::Job { channel_id, .. } => channel_id,
    };
    identity.community_id() == fields.community_id.as_uuid()
        && channel_id == fields.channel_id.as_uuid()
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CollaborationMentionError {
    #[error("collaboration agent identity is invalid")]
    InvalidAgent,
    #[error("resolved collaboration mention target is invalid")]
    InvalidMentionTarget,
    #[error("collaboration mention is not addressed to this agent")]
    UnsupportedMention,
    #[error("collaboration mention actor is not authorized")]
    Unauthorized,
    #[error("collaboration mention does not match the native session scope")]
    SessionMismatch,
    #[error("collaboration mention message is unavailable")]
    MessageUnavailable,
    #[error("collaboration mention prompt is empty")]
    EmptyPrompt,
    #[error("collaboration mention event was already routed")]
    DuplicateEvent,
    #[error("collaboration session already has an active prompt")]
    BusySession,
    #[error("collaboration prompt generation is exhausted")]
    GenerationExhausted,
    #[error("collaboration prompt dispatch is no longer current")]
    DispatchNotCurrent,
    #[error(transparent)]
    Session(#[from] CollaborationSessionError),
}
