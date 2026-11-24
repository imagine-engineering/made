use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{AggregateId, NostrEventId, NostrPublicKey};

pub const MAX_THREAD_DEPTH: u16 = 100;
pub const MAX_THREAD_PAGE_ROWS: usize = 200;
pub const MAX_AUXILIARY_EVENTS_PER_HOP: usize = 1_000;
pub const MAX_THREAD_SUMMARY_PARTICIPANTS: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadReference {
    TopLevel,
    Reply {
        parent_event_id: NostrEventId,
        root_event_id: Option<NostrEventId>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadEvent {
    pub event_id: NostrEventId,
    pub channel_id: AggregateId,
    pub author: NostrPublicKey,
    pub created_at: u64,
    pub reference: ThreadReference,
    pub broadcast: bool,
    pub deleted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadNode {
    pub event: ThreadEvent,
    pub parent_event_id: Option<NostrEventId>,
    pub root_event_id: NostrEventId,
    pub depth: u16,
}

impl ThreadNode {
    pub const fn is_top_level(self) -> bool {
        self.depth == 0 || (self.depth == 1 && self.event.broadcast)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ThreadCursor {
    pub created_at: u64,
    pub event_id: NostrEventId,
}

impl ThreadCursor {
    const fn for_node(node: ThreadNode) -> Self {
        Self {
            created_at: node.event.created_at,
            event_id: node.event.event_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadPage {
    pub replies: Vec<ThreadNode>,
    pub has_more: bool,
    pub next_cursor: Option<ThreadCursor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadSummary {
    pub reply_count: u64,
    pub descendant_count: u64,
    pub last_reply_at: Option<u64>,
    pub participants: Vec<NostrPublicKey>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxiliaryEventKind {
    Reaction,
    Deletion,
    AdministrativeDeletion,
    Edit,
}

impl AuxiliaryEventKind {
    const fn is_deletion(self) -> bool {
        matches!(self, Self::Deletion | Self::AdministrativeDeletion)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuxiliaryEvent {
    pub event_id: NostrEventId,
    pub channel_id: AggregateId,
    pub target_event_id: NostrEventId,
    pub created_at: u64,
    pub kind: AuxiliaryEventKind,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuxiliaryClosure {
    pub first_hop: Vec<AuxiliaryEvent>,
    pub second_hop: Vec<AuxiliaryEvent>,
}

impl AuxiliaryClosure {
    pub fn events(&self) -> impl Iterator<Item = &AuxiliaryEvent> {
        self.first_hop.iter().chain(&self.second_hop)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadGraph {
    nodes: BTreeMap<NostrEventId, ThreadNode>,
    children: BTreeMap<NostrEventId, Vec<NostrEventId>>,
}

impl ThreadGraph {
    pub fn build(events: impl IntoIterator<Item = ThreadEvent>) -> Result<Self, ThreadError> {
        let mut events_by_id = BTreeMap::new();
        for event in events {
            validate_event(event)?;
            if events_by_id.insert(event.event_id, event).is_some() {
                return Err(ThreadError::DuplicateEvent(event.event_id));
            }
        }

        let mut graph = Self {
            nodes: BTreeMap::new(),
            children: BTreeMap::new(),
        };
        let mut visiting = BTreeSet::new();
        for event_id in events_by_id.keys().copied() {
            graph.resolve(event_id, &events_by_id, &mut visiting)?;
        }
        for children in graph.children.values_mut() {
            children.sort_by_key(|event_id| {
                let node = graph.nodes.get(event_id).copied();
                node.map(ThreadCursor::for_node)
            });
        }
        Ok(graph)
    }

    pub fn node(&self, event_id: NostrEventId) -> Option<ThreadNode> {
        self.nodes.get(&event_id).copied()
    }

    pub fn nodes(&self) -> impl Iterator<Item = ThreadNode> + '_ {
        self.nodes.values().copied()
    }

    pub fn summary(
        &self,
        row_event_id: NostrEventId,
    ) -> Result<Option<ThreadSummary>, ThreadError> {
        self.require_node(row_event_id)?;
        let direct_reply_count = self
            .children
            .get(&row_event_id)
            .into_iter()
            .flatten()
            .filter(|event_id| {
                self.nodes
                    .get(event_id)
                    .is_some_and(|node| !node.event.deleted)
            })
            .count();
        if direct_reply_count == 0 {
            return Ok(None);
        }

        let descendants = self.active_descendants(row_event_id);
        let mut participants = BTreeMap::<NostrPublicKey, ThreadCursor>::new();
        for node in &descendants {
            let cursor = ThreadCursor::for_node(*node);
            participants
                .entry(node.event.author)
                .and_modify(|latest| *latest = (*latest).max(cursor))
                .or_insert(cursor);
        }
        let mut participants = participants.into_iter().collect::<Vec<_>>();
        participants.sort_by_key(|(author, latest)| (Reverse(*latest), *author));
        participants.truncate(MAX_THREAD_SUMMARY_PARTICIPANTS);

        Ok(Some(ThreadSummary {
            reply_count: direct_reply_count as u64,
            descendant_count: descendants.len() as u64,
            last_reply_at: descendants.iter().map(|node| node.event.created_at).max(),
            participants: participants.into_iter().map(|(author, _)| author).collect(),
        }))
    }

    pub fn page(
        &self,
        row_event_id: NostrEventId,
        cursor: Option<ThreadCursor>,
        requested_limit: usize,
        depth_limit: Option<u16>,
    ) -> Result<ThreadPage, ThreadError> {
        let row = self.require_node(row_event_id)?;
        let limit = requested_limit.clamp(1, MAX_THREAD_PAGE_ROWS);
        let maximum_depth = depth_limit
            .map(|depth| row.depth.saturating_add(depth.min(MAX_THREAD_DEPTH)))
            .unwrap_or(MAX_THREAD_DEPTH);
        let mut replies = self
            .active_descendants(row_event_id)
            .into_iter()
            .filter(|node| node.depth <= maximum_depth)
            .filter(|node| cursor.is_none_or(|cursor| ThreadCursor::for_node(*node) > cursor))
            .collect::<Vec<_>>();
        replies.sort_by_key(|node| ThreadCursor::for_node(*node));

        let has_more = replies.len() > limit;
        replies.truncate(limit);
        let next_cursor = has_more
            .then(|| replies.last().copied().map(ThreadCursor::for_node))
            .flatten();
        Ok(ThreadPage {
            replies,
            has_more,
            next_cursor,
        })
    }

    pub fn auxiliary_closure(
        &self,
        row_event_ids: impl IntoIterator<Item = NostrEventId>,
        events: impl IntoIterator<Item = AuxiliaryEvent>,
    ) -> Result<AuxiliaryClosure, ThreadError> {
        let row_event_ids = row_event_ids.into_iter().collect::<BTreeSet<_>>();
        if row_event_ids.is_empty() {
            return Ok(AuxiliaryClosure::default());
        }
        let mut channels = BTreeSet::new();
        for row_event_id in &row_event_ids {
            channels.insert(self.require_node(*row_event_id)?.event.channel_id);
        }
        if channels.len() != 1 {
            return Err(ThreadError::MixedChannelRows);
        }
        let channel_id = channels
            .first()
            .copied()
            .ok_or(ThreadError::MixedChannelRows)?;

        let mut events_by_id = BTreeMap::new();
        for event in events {
            validate_auxiliary_event(event)?;
            if events_by_id.insert(event.event_id, event).is_some() {
                return Err(ThreadError::DuplicateAuxiliaryEvent(event.event_id));
            }
        }

        let mut first_hop = events_by_id
            .values()
            .copied()
            .filter(|event| row_event_ids.contains(&event.target_event_id))
            .map(|event| require_auxiliary_channel(event, channel_id))
            .collect::<Result<Vec<_>, _>>()?;
        first_hop.sort_by_key(auxiliary_sort_key);
        first_hop.truncate(MAX_AUXILIARY_EVENTS_PER_HOP);

        let first_hop_ids = first_hop
            .iter()
            .map(|event| event.event_id)
            .collect::<BTreeSet<_>>();
        let mut second_hop = events_by_id
            .values()
            .copied()
            .filter(|event| {
                event.kind.is_deletion()
                    && first_hop_ids.contains(&event.target_event_id)
                    && !first_hop_ids.contains(&event.event_id)
            })
            .map(|event| require_auxiliary_channel(event, channel_id))
            .collect::<Result<Vec<_>, _>>()?;
        second_hop.sort_by_key(auxiliary_sort_key);
        second_hop.truncate(MAX_AUXILIARY_EVENTS_PER_HOP);

        Ok(AuxiliaryClosure {
            first_hop,
            second_hop,
        })
    }

    fn resolve(
        &mut self,
        event_id: NostrEventId,
        events: &BTreeMap<NostrEventId, ThreadEvent>,
        visiting: &mut BTreeSet<NostrEventId>,
    ) -> Result<ThreadNode, ThreadError> {
        if let Some(node) = self.nodes.get(&event_id) {
            return Ok(*node);
        }
        if !visiting.insert(event_id) {
            return Err(ThreadError::ReferenceCycle(event_id));
        }
        let event = events
            .get(&event_id)
            .copied()
            .ok_or(ThreadError::MissingEvent(event_id))?;
        let node = match event.reference {
            ThreadReference::TopLevel => ThreadNode {
                event,
                parent_event_id: None,
                root_event_id: event_id,
                depth: 0,
            },
            ThreadReference::Reply {
                parent_event_id,
                root_event_id,
            } => {
                let parent = self.resolve(parent_event_id, events, visiting)?;
                if parent.event.channel_id != event.channel_id {
                    return Err(ThreadError::ParentChannelMismatch {
                        event_id,
                        parent_event_id,
                    });
                }
                if root_event_id.is_some_and(|root_event_id| root_event_id != parent.root_event_id)
                {
                    return Err(ThreadError::RootMismatch {
                        event_id,
                        expected_root_event_id: parent.root_event_id,
                    });
                }
                let depth = parent
                    .depth
                    .checked_add(1)
                    .filter(|depth| *depth <= MAX_THREAD_DEPTH)
                    .ok_or(ThreadError::DepthExceeded(event_id))?;
                self.children
                    .entry(parent_event_id)
                    .or_default()
                    .push(event_id);
                ThreadNode {
                    event,
                    parent_event_id: Some(parent_event_id),
                    root_event_id: parent.root_event_id,
                    depth,
                }
            }
        };
        visiting.remove(&event_id);
        self.nodes.insert(event_id, node);
        Ok(node)
    }

    fn require_node(&self, event_id: NostrEventId) -> Result<ThreadNode, ThreadError> {
        self.node(event_id)
            .ok_or(ThreadError::MissingEvent(event_id))
    }

    fn active_descendants(&self, event_id: NostrEventId) -> Vec<ThreadNode> {
        let mut pending = self.children.get(&event_id).cloned().unwrap_or_default();
        let mut descendants = Vec::new();
        while let Some(event_id) = pending.pop() {
            let Some(node) = self.nodes.get(&event_id).copied() else {
                continue;
            };
            if !node.event.deleted {
                descendants.push(node);
            }
            if let Some(children) = self.children.get(&event_id) {
                pending.extend(children.iter().copied());
            }
        }
        descendants
    }
}

fn validate_event(event: ThreadEvent) -> Result<(), ThreadError> {
    if event.event_id.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(ThreadError::InvalidEventId);
    }
    if event.channel_id.as_uuid().is_nil() {
        return Err(ThreadError::InvalidChannelId);
    }
    if event.author.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(ThreadError::InvalidAuthor);
    }
    Ok(())
}

fn validate_auxiliary_event(event: AuxiliaryEvent) -> Result<(), ThreadError> {
    if event.event_id.as_bytes().iter().all(|byte| *byte == 0)
        || event
            .target_event_id
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
    {
        return Err(ThreadError::InvalidAuxiliaryReference);
    }
    if event.channel_id.as_uuid().is_nil() {
        return Err(ThreadError::InvalidChannelId);
    }
    Ok(())
}

fn require_auxiliary_channel(
    event: AuxiliaryEvent,
    channel_id: AggregateId,
) -> Result<AuxiliaryEvent, ThreadError> {
    if event.channel_id != channel_id {
        return Err(ThreadError::AuxiliaryChannelMismatch(event.event_id));
    }
    Ok(event)
}

fn auxiliary_sort_key(event: &AuxiliaryEvent) -> (u64, NostrEventId) {
    (event.created_at, event.event_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadError {
    InvalidEventId,
    InvalidChannelId,
    InvalidAuthor,
    InvalidAuxiliaryReference,
    DuplicateEvent(NostrEventId),
    DuplicateAuxiliaryEvent(NostrEventId),
    MissingEvent(NostrEventId),
    ReferenceCycle(NostrEventId),
    ParentChannelMismatch {
        event_id: NostrEventId,
        parent_event_id: NostrEventId,
    },
    RootMismatch {
        event_id: NostrEventId,
        expected_root_event_id: NostrEventId,
    },
    DepthExceeded(NostrEventId),
    MixedChannelRows,
    AuxiliaryChannelMismatch(NostrEventId),
}

impl fmt::Display for ThreadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEventId => formatter.write_str("thread event ID is invalid"),
            Self::InvalidChannelId => formatter.write_str("thread channel ID is invalid"),
            Self::InvalidAuthor => formatter.write_str("thread author is invalid"),
            Self::InvalidAuxiliaryReference => {
                formatter.write_str("auxiliary event reference is invalid")
            }
            Self::DuplicateEvent(_) => formatter.write_str("thread event ID is duplicated"),
            Self::DuplicateAuxiliaryEvent(_) => {
                formatter.write_str("auxiliary event ID is duplicated")
            }
            Self::MissingEvent(_) => formatter.write_str("thread event is missing"),
            Self::ReferenceCycle(_) => formatter.write_str("thread references form a cycle"),
            Self::ParentChannelMismatch { .. } => {
                formatter.write_str("reply parent belongs to another channel")
            }
            Self::RootMismatch { .. } => {
                formatter.write_str("reply root does not match its parent ancestry")
            }
            Self::DepthExceeded(_) => formatter.write_str("thread depth exceeds the protocol cap"),
            Self::MixedChannelRows => {
                formatter.write_str("window rows belong to multiple channels")
            }
            Self::AuxiliaryChannelMismatch(_) => {
                formatter.write_str("auxiliary event belongs to another channel")
            }
        }
    }
}

impl Error for ThreadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn event_id(value: u16) -> NostrEventId {
        let mut bytes = [0; 32];
        bytes[30..].copy_from_slice(&value.to_be_bytes());
        NostrEventId::from_bytes(bytes)
    }

    fn channel_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn author(value: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([value; 32])
    }

    fn root(value: u16, created_at: u64) -> ThreadEvent {
        ThreadEvent {
            event_id: event_id(value),
            channel_id: channel_id(1),
            author: author(1),
            created_at,
            reference: ThreadReference::TopLevel,
            broadcast: false,
            deleted: false,
        }
    }

    fn reply(
        value: u16,
        parent: u16,
        root: Option<u16>,
        depth_time: u64,
        author_value: u8,
    ) -> ThreadEvent {
        ThreadEvent {
            event_id: event_id(value),
            channel_id: channel_id(1),
            author: author(author_value),
            created_at: depth_time,
            reference: ThreadReference::Reply {
                parent_event_id: event_id(parent),
                root_event_id: root.map(event_id),
            },
            broadcast: false,
            deleted: false,
        }
    }

    fn auxiliary(
        value: u16,
        target: u16,
        created_at: u64,
        kind: AuxiliaryEventKind,
    ) -> AuxiliaryEvent {
        AuxiliaryEvent {
            event_id: event_id(value),
            channel_id: channel_id(1),
            target_event_id: event_id(target),
            created_at,
            kind,
        }
    }

    #[test]
    fn deep_reply_fixture_has_stable_ancestry_summaries_and_continuation() {
        let mut broadcast = reply(2, 1, Some(1), 20, 2);
        broadcast.broadcast = true;
        let graph = ThreadGraph::build([
            reply(5, 4, Some(1), 22, 3),
            reply(3, 2, Some(1), 20, 2),
            root(1, 10),
            reply(4, 3, Some(1), 21, 3),
            broadcast,
        ])
        .expect("valid deep thread");

        assert_eq!(graph.node(event_id(5)).map(|node| node.depth), Some(4));
        assert_eq!(
            graph.node(event_id(5)).map(|node| node.root_event_id),
            Some(event_id(1))
        );
        assert!(
            graph
                .node(event_id(2))
                .is_some_and(ThreadNode::is_top_level)
        );
        let summary = graph
            .summary(event_id(1))
            .expect("known root")
            .expect("root has replies");
        assert_eq!(summary.reply_count, 1);
        assert_eq!(summary.descendant_count, 4);
        assert_eq!(summary.last_reply_at, Some(22));
        assert_eq!(summary.participants, vec![author(3), author(2)]);

        let first = graph
            .page(event_id(1), None, 2, None)
            .expect("first thread page");
        assert!(first.has_more);
        assert_eq!(
            first
                .replies
                .iter()
                .map(|node| node.event.event_id)
                .collect::<Vec<_>>(),
            vec![event_id(2), event_id(3)]
        );
        let second = graph
            .page(event_id(1), first.next_cursor, 2, None)
            .expect("second thread page");
        assert!(!second.has_more);
        assert_eq!(
            second
                .replies
                .iter()
                .map(|node| node.event.event_id)
                .collect::<Vec<_>>(),
            vec![event_id(4), event_id(5)]
        );
    }

    #[test]
    fn deleted_root_fixture_preserves_ancestry_and_omits_deleted_replies() {
        let mut deleted_root = root(1, 10);
        deleted_root.deleted = true;
        let mut deleted_reply = reply(2, 1, Some(1), 20, 2);
        deleted_reply.deleted = true;
        let graph = ThreadGraph::build([
            deleted_root,
            deleted_reply,
            reply(3, 2, Some(1), 21, 3),
            reply(4, 1, Some(1), 22, 4),
        ])
        .expect("deleted ancestors remain structural");

        assert_eq!(graph.node(event_id(3)).map(|node| node.depth), Some(2));
        let summary = graph
            .summary(event_id(1))
            .expect("known deleted root")
            .expect("active direct reply remains");
        assert_eq!(summary.reply_count, 1);
        assert_eq!(summary.descendant_count, 2);
        assert_eq!(summary.participants, vec![author(4), author(3)]);
        assert_eq!(
            graph
                .page(event_id(1), None, 20, None)
                .expect("deleted-root page")
                .replies
                .iter()
                .map(|node| node.event.event_id)
                .collect::<Vec<_>>(),
            vec![event_id(3), event_id(4)]
        );
    }

    #[test]
    fn auxiliary_closure_fixture_is_two_hop_unique_stable_and_bounded() {
        let graph = ThreadGraph::build([root(1, 10), root(2, 11)]).expect("valid rows");
        let mut events = (0..1_002)
            .map(|offset| {
                auxiliary(
                    100 + offset,
                    1,
                    u64::from(offset),
                    AuxiliaryEventKind::Reaction,
                )
            })
            .collect::<Vec<_>>();
        events.extend([
            auxiliary(1_500, 100, 2_000, AuxiliaryEventKind::Deletion),
            auxiliary(1_501, 1_500, 2_001, AuxiliaryEventKind::Deletion),
            auxiliary(1_502, 100, 2_002, AuxiliaryEventKind::Edit),
            auxiliary(1_503, 2, 2_003, AuxiliaryEventKind::Edit),
        ]);

        let closure = graph
            .auxiliary_closure([event_id(1)], events)
            .expect("valid closure");
        assert_eq!(closure.first_hop.len(), MAX_AUXILIARY_EVENTS_PER_HOP);
        assert_eq!(
            closure.first_hop.first().map(|event| event.event_id),
            Some(event_id(100))
        );
        assert_eq!(closure.second_hop.len(), 1);
        assert_eq!(closure.second_hop[0].event_id, event_id(1_500));
        assert_eq!(closure.events().count(), 1_001);
    }

    #[test]
    fn malformed_reference_fixture_rejects_missing_cross_channel_root_cycle_and_depth() {
        assert_eq!(
            ThreadGraph::build([reply(2, 99, Some(1), 20, 2)]),
            Err(ThreadError::MissingEvent(event_id(99)))
        );

        let mut cross_channel = reply(2, 1, Some(1), 20, 2);
        cross_channel.channel_id = channel_id(2);
        assert!(matches!(
            ThreadGraph::build([root(1, 10), cross_channel]),
            Err(ThreadError::ParentChannelMismatch { .. })
        ));

        assert!(matches!(
            ThreadGraph::build([root(1, 10), reply(2, 1, Some(9), 20, 2)]),
            Err(ThreadError::RootMismatch { .. })
        ));

        assert!(matches!(
            ThreadGraph::build([reply(1, 2, Some(1), 10, 1), reply(2, 1, Some(1), 11, 2),]),
            Err(ThreadError::ReferenceCycle(_))
        ));

        let mut too_deep = vec![root(1, 1)];
        for value in 2..=102 {
            too_deep.push(reply(value, value - 1, Some(1), u64::from(value), 2));
        }
        assert!(matches!(
            ThreadGraph::build(too_deep),
            Err(ThreadError::DepthExceeded(_))
        ));
    }
}
