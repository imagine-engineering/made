use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::{Mutex, MutexGuard},
};

use collaboration_domain::{
    AggregateId, AuthenticatedPrincipal, AuthenticatedPrincipalKind, AuthorizationAction,
    AuthorizationDecision, AuthorizationRequest, AuthorizationResourceKind, CommunityId,
    PrincipalId, TenantContext, authorize,
};
use nostr_compat::{
    EventId, PublicKey, SignedEvent, TimestampPolicy, generated_kinds::KIND_TYPING_INDICATOR,
    verify_signed_event,
};
use uuid::Uuid;

pub const TYPING_INDICATOR_TTL_MILLIS: u64 = 60_000;
pub const TYPING_RATE_TOKENS_PER_SECOND: u64 = 2;
pub const TYPING_RATE_BURST: u64 = 10;
pub const MAX_TYPING_CONNECTIONS: usize = 10_000;
pub const MAX_RETAINED_TYPING_ENTRIES: usize = 65_536;
pub const MAX_TRACKED_TYPING_PRINCIPALS: usize = 65_536;

const MAX_TYPING_EVENT_FUTURE_SECONDS: u64 = 15;
const RETAINED_ORDER_TTL_MILLIS: u64 = TYPING_INDICATOR_TTL_MILLIS * 2;
const MILLI_TOKENS_PER_TOKEN: u64 = 1_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ConnectionKey {
    community_id: CommunityId,
    connection_id: Uuid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConnectionState {
    principal_id: PrincipalId,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TypingEntryKey {
    community_id: CommunityId,
    channel_id: AggregateId,
    principal_id: PrincipalId,
    connection_id: Uuid,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TypingEntry {
    event_id: EventId,
    event_created_at: u64,
    active_until_millis: u64,
    retain_until_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RateLimitKey {
    community_id: CommunityId,
    principal_id: PrincipalId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TokenBucket {
    milli_tokens: u64,
    last_refill_millis: u64,
    last_publication_millis: u64,
}

impl TokenBucket {
    const MAX_MILLI_TOKENS: u64 = TYPING_RATE_BURST * MILLI_TOKENS_PER_TOKEN;
    const MILLI_TOKENS_PER_MILLISECOND: u64 = TYPING_RATE_TOKENS_PER_SECOND;

    const fn full(now_millis: u64) -> Self {
        Self {
            milli_tokens: Self::MAX_MILLI_TOKENS,
            last_refill_millis: now_millis,
            last_publication_millis: now_millis,
        }
    }

    fn consume(&mut self, now_millis: u64) -> Result<(), u64> {
        let elapsed_millis = now_millis.saturating_sub(self.last_refill_millis);
        let replenished = elapsed_millis.saturating_mul(Self::MILLI_TOKENS_PER_MILLISECOND);
        self.milli_tokens = self
            .milli_tokens
            .saturating_add(replenished)
            .min(Self::MAX_MILLI_TOKENS);
        self.last_refill_millis = self.last_refill_millis.max(now_millis);
        self.last_publication_millis = self.last_publication_millis.max(now_millis);
        if self.milli_tokens >= MILLI_TOKENS_PER_TOKEN {
            self.milli_tokens -= MILLI_TOKENS_PER_TOKEN;
            return Ok(());
        }

        let missing = MILLI_TOKENS_PER_TOKEN - self.milli_tokens;
        let retry_after_millis = missing.div_ceil(Self::MILLI_TOKENS_PER_MILLISECOND);
        Err(retry_after_millis)
    }
}

#[derive(Default)]
struct TypingState {
    connections: HashMap<ConnectionKey, ConnectionState>,
    entries: HashMap<TypingEntryKey, TypingEntry>,
    rate_limits: HashMap<RateLimitKey, TokenBucket>,
}

impl TypingState {
    fn prune(&mut self, now_millis: u64) {
        self.entries
            .retain(|_, entry| now_millis < entry.retain_until_millis);
        self.rate_limits.retain(|_, bucket| {
            now_millis.saturating_sub(bucket.last_publication_millis) < RETAINED_ORDER_TTL_MILLIS
        });
    }

    fn remove_connection_entries(&mut self, key: ConnectionKey, generation: u64) {
        self.entries.retain(|entry, _| {
            entry.community_id != key.community_id
                || entry.connection_id != key.connection_id
                || entry.generation != generation
        });
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypingConnectionToken {
    community_id: CommunityId,
    principal_id: PrincipalId,
    connection_id: Uuid,
    generation: u64,
}

impl TypingConnectionToken {
    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn principal_id(self) -> PrincipalId {
        self.principal_id
    }

    pub const fn connection_id(self) -> Uuid {
        self.connection_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypingPublicationOutcome {
    Applied,
    Duplicate,
    IgnoredStale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypingDisconnectOutcome {
    Removed,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypingParticipant {
    pub principal_id: PrincipalId,
    pub expires_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypingStoreMetrics {
    pub connections: usize,
    pub active_entries: usize,
    pub retained_entries: usize,
    pub tracked_rate_limit_principals: usize,
}

pub struct TypingIndicatorStore {
    state: Mutex<TypingState>,
}

impl Default for TypingIndicatorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TypingIndicatorStore {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(TypingState::default()),
        }
    }

    pub fn register_connection(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        connection_id: Uuid,
        generation: u64,
        now_millis: u64,
    ) -> Result<TypingConnectionToken, TypingError> {
        if tenant.community_id() != principal.community_id()
            || connection_id.is_nil()
            || generation == 0
        {
            return Err(TypingError::InvalidRequest);
        }
        let key = ConnectionKey {
            community_id: tenant.community_id(),
            connection_id,
        };
        let mut state = self.lock_state()?;
        state.prune(now_millis);
        if let Some(current) = state.connections.get(&key).copied() {
            if current.principal_id != principal.principal_id() {
                return Err(TypingError::StaleConnection);
            }
            if generation < current.generation {
                return Err(TypingError::StaleConnection);
            }
            if generation > current.generation {
                state.remove_connection_entries(key, current.generation);
                state.connections.insert(
                    key,
                    ConnectionState {
                        principal_id: current.principal_id,
                        generation,
                    },
                );
            }
        } else {
            if state.connections.len() >= MAX_TYPING_CONNECTIONS {
                return Err(TypingError::CapacityExceeded);
            }
            state.connections.insert(
                key,
                ConnectionState {
                    principal_id: principal.principal_id(),
                    generation,
                },
            );
        }
        Ok(TypingConnectionToken {
            community_id: tenant.community_id(),
            principal_id: principal.principal_id(),
            connection_id,
            generation,
        })
    }

    pub fn publish(
        &self,
        token: TypingConnectionToken,
        authorization: &AuthorizationRequest<'_>,
        signed_event: &SignedEvent,
        now_millis: u64,
    ) -> Result<TypingPublicationOutcome, TypingError> {
        let channel_id = validate_publication(token, authorization, signed_event, now_millis)?;
        let active_until_millis = now_millis
            .checked_add(TYPING_INDICATOR_TTL_MILLIS)
            .ok_or(TypingError::InvalidRequest)?;
        let retain_until_millis = now_millis
            .checked_add(RETAINED_ORDER_TTL_MILLIS)
            .ok_or(TypingError::InvalidRequest)?;
        let connection_key = ConnectionKey {
            community_id: token.community_id,
            connection_id: token.connection_id,
        };
        let entry_key = TypingEntryKey {
            community_id: token.community_id,
            channel_id,
            principal_id: token.principal_id,
            connection_id: token.connection_id,
            generation: token.generation,
        };
        let candidate = TypingEntry {
            event_id: signed_event.claimed_id,
            event_created_at: signed_event.event.created_at,
            active_until_millis,
            retain_until_millis,
        };

        let mut state = self.lock_state()?;
        state.prune(now_millis);
        let Some(connection) = state.connections.get(&connection_key) else {
            return Err(TypingError::StaleConnection);
        };
        if connection.principal_id != token.principal_id
            || connection.generation != token.generation
        {
            return Err(TypingError::StaleConnection);
        }
        if state
            .entries
            .get(&entry_key)
            .is_some_and(|current| current.event_id == candidate.event_id)
        {
            return Ok(TypingPublicationOutcome::Duplicate);
        }
        if !state.entries.contains_key(&entry_key)
            && state.entries.len() >= MAX_RETAINED_TYPING_ENTRIES
        {
            return Err(TypingError::CapacityExceeded);
        }

        let rate_limit_key = RateLimitKey {
            community_id: token.community_id,
            principal_id: token.principal_id,
        };
        if !state.rate_limits.contains_key(&rate_limit_key)
            && state.rate_limits.len() >= MAX_TRACKED_TYPING_PRINCIPALS
        {
            return Err(TypingError::CapacityExceeded);
        }
        let bucket = state
            .rate_limits
            .entry(rate_limit_key)
            .or_insert_with(|| TokenBucket::full(now_millis));
        if let Err(retry_after_millis) = bucket.consume(now_millis) {
            return Err(TypingError::RateLimited { retry_after_millis });
        }

        if let Some(current) = state.entries.get(&entry_key)
            && compare_events(candidate, *current) != Ordering::Greater
        {
            return Ok(TypingPublicationOutcome::IgnoredStale);
        }
        state.entries.insert(entry_key, candidate);
        Ok(TypingPublicationOutcome::Applied)
    }

    pub fn active_for_channel(
        &self,
        authorization: &AuthorizationRequest<'_>,
        channel_id: AggregateId,
        now_millis: u64,
    ) -> Result<Vec<TypingParticipant>, TypingError> {
        validate_channel_authorization(authorization, channel_id, AuthorizationAction::Read)?;
        let mut state = self.lock_state()?;
        state.prune(now_millis);
        let mut participants = BTreeMap::<PrincipalId, u64>::new();
        for (key, entry) in &state.entries {
            if key.community_id == authorization.tenant.community_id()
                && key.channel_id == channel_id
                && now_millis < entry.active_until_millis
            {
                participants
                    .entry(key.principal_id)
                    .and_modify(|expires_at| {
                        *expires_at = (*expires_at).max(entry.active_until_millis)
                    })
                    .or_insert(entry.active_until_millis);
            }
        }
        Ok(participants
            .into_iter()
            .map(|(principal_id, expires_at_millis)| TypingParticipant {
                principal_id,
                expires_at_millis,
            })
            .collect())
    }

    pub fn disconnect(
        &self,
        token: TypingConnectionToken,
    ) -> Result<TypingDisconnectOutcome, TypingError> {
        let key = ConnectionKey {
            community_id: token.community_id,
            connection_id: token.connection_id,
        };
        let mut state = self.lock_state()?;
        let Some(current) = state.connections.get(&key).copied() else {
            return Ok(TypingDisconnectOutcome::Stale);
        };
        if current.principal_id != token.principal_id || current.generation != token.generation {
            return Ok(TypingDisconnectOutcome::Stale);
        }
        state.connections.remove(&key);
        state.remove_connection_entries(key, token.generation);
        Ok(TypingDisconnectOutcome::Removed)
    }

    pub fn metrics(&self, now_millis: u64) -> Result<TypingStoreMetrics, TypingError> {
        let mut state = self.lock_state()?;
        state.prune(now_millis);
        Ok(TypingStoreMetrics {
            connections: state.connections.len(),
            active_entries: state
                .entries
                .values()
                .filter(|entry| now_millis < entry.active_until_millis)
                .count(),
            retained_entries: state.entries.len(),
            tracked_rate_limit_principals: state.rate_limits.len(),
        })
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, TypingState>, TypingError> {
        self.state.lock().map_err(|_| TypingError::Unavailable)
    }
}

fn validate_publication(
    token: TypingConnectionToken,
    authorization: &AuthorizationRequest<'_>,
    signed_event: &SignedEvent,
    now_millis: u64,
) -> Result<AggregateId, TypingError> {
    if token.community_id != authorization.tenant.community_id()
        || token.principal_id != authorization.principal.principal_id()
    {
        return Err(TypingError::TenantMismatch);
    }
    let channel_id = parse_channel_id(signed_event)?;
    validate_channel_authorization(authorization, channel_id, AuthorizationAction::Write)?;
    if !principal_matches_author(authorization.principal, signed_event.event.public_key) {
        return Err(TypingError::Unauthorized);
    }
    let now_seconds = now_millis / 1_000;
    verify_signed_event(
        signed_event,
        TimestampPolicy::Bounded {
            now: now_seconds,
            max_past_seconds: TYPING_INDICATOR_TTL_MILLIS / 1_000,
            max_future_seconds: MAX_TYPING_EVENT_FUTURE_SECONDS,
        },
    )
    .map_err(|_| TypingError::InvalidEvent)?;
    Ok(channel_id)
}

fn validate_channel_authorization(
    authorization: &AuthorizationRequest<'_>,
    channel_id: AggregateId,
    action: AuthorizationAction,
) -> Result<(), TypingError> {
    if authorization.action != action
        || authorization.delegation.is_some()
        || authorization.resource.kind != AuthorizationResourceKind::Channel
        || authorization.resource.resource_id != channel_id
        || authorization.resource.channel_id != Some(channel_id)
    {
        return Err(TypingError::Unauthorized);
    }
    match authorize(authorization) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(_) => Err(TypingError::Unauthorized),
    }
}

fn parse_channel_id(signed_event: &SignedEvent) -> Result<AggregateId, TypingError> {
    if u32::from(signed_event.event.kind) != KIND_TYPING_INDICATOR {
        return Err(TypingError::InvalidEvent);
    }
    let mut channel_tags = signed_event
        .event
        .tags
        .iter()
        .filter(|tag| tag.first().is_some_and(|name| name == "h"));
    let tag = channel_tags.next().ok_or(TypingError::InvalidEvent)?;
    if channel_tags.next().is_some() || tag.len() != 2 {
        return Err(TypingError::InvalidEvent);
    }
    let channel_id = tag
        .get(1)
        .and_then(|value| Uuid::parse_str(value).ok())
        .filter(|value| !value.is_nil())
        .ok_or(TypingError::InvalidEvent)?;
    Ok(AggregateId::from_uuid(channel_id))
}

fn principal_matches_author(principal: &AuthenticatedPrincipal, author: PublicKey) -> bool {
    let expected = match principal.kind() {
        AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } => *public_key.as_bytes(),
        AuthenticatedPrincipalKind::OwnerAttestedAgent {
            agent_public_key, ..
        } => *agent_public_key.as_bytes(),
        AuthenticatedPrincipalKind::SimAccount { .. }
        | AuthenticatedPrincipalKind::ScopedToken { .. }
        | AuthenticatedPrincipalKind::Service { .. } => return false,
    };
    expected == *author.as_bytes()
}

fn compare_events(candidate: TypingEntry, current: TypingEntry) -> Ordering {
    candidate
        .event_created_at
        .cmp(&current.event_created_at)
        .then_with(|| {
            current
                .event_id
                .as_bytes()
                .cmp(candidate.event_id.as_bytes())
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TypingError {
    #[error("typing request is invalid")]
    InvalidRequest,
    #[error("typing event is invalid")]
    InvalidEvent,
    #[error("typing request is unauthorized")]
    Unauthorized,
    #[error("typing request crossed its tenant boundary")]
    TenantMismatch,
    #[error("typing connection generation is stale")]
    StaleConnection,
    #[error("typing publication rate limit exceeded; retry after {retry_after_millis}ms")]
    RateLimited { retry_after_millis: u64 },
    #[error("typing state capacity was exceeded")]
    CapacityExceeded,
    #[error("typing state is unavailable")]
    Unavailable,
}
