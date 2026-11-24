use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fmt,
};

use async_trait::async_trait;
use collaboration_domain::{AuthenticatedPrincipal, TenantContext};
use nostr_compat::{
    PublicKey,
    filter::{EventFilter, HexPrefix},
};
use serde_json::{Map, Value};
use uuid::Uuid;

pub use super::MAX_NOSTR_FRAME_BYTES;
pub const MAX_SUBSCRIPTION_ID_BYTES: usize = 256;
pub const MAX_FILTERS_PER_REQUEST: usize = 10;
pub const MAX_FILTER_LIMIT: u32 = 1_000;
pub const MAX_ACTIVE_SUBSCRIPTIONS: usize = 1_024;
pub const MAX_HISTORICAL_EVENTS_PER_REQUEST: usize =
    MAX_FILTERS_PER_REQUEST * MAX_FILTER_LIMIT as usize;
const MAX_SEARCH_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SubscriptionId(String);

impl SubscriptionId {
    pub fn new(value: impl Into<String>) -> Result<Self, NostrSubscriptionError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_SUBSCRIPTION_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(NostrSubscriptionError::InvalidSubscriptionId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrSubscriptionFilter {
    event_filter: EventFilter,
    limit: u32,
    search: Option<String>,
}

impl NostrSubscriptionFilter {
    pub const fn event_filter(&self) -> &EventFilter {
        &self.event_filter
    }

    pub const fn limit(&self) -> u32 {
        self.limit
    }

    pub fn search(&self) -> Option<&str> {
        self.search.as_deref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NostrSubscriptionFrame {
    Req {
        subscription_id: SubscriptionId,
        filters: Vec<NostrSubscriptionFilter>,
    },
    Count {
        subscription_id: SubscriptionId,
        filters: Vec<NostrSubscriptionFilter>,
    },
    Close {
        subscription_id: SubscriptionId,
    },
}

impl NostrSubscriptionFrame {
    pub fn parse(raw: &str) -> Result<Self, NostrSubscriptionError> {
        if raw.len() > MAX_NOSTR_FRAME_BYTES {
            return Err(NostrSubscriptionError::FrameTooLarge);
        }
        let Value::Array(parts) =
            serde_json::from_str(raw).map_err(|_| NostrSubscriptionError::InvalidFrame)?
        else {
            return Err(NostrSubscriptionError::InvalidFrame);
        };
        let Some(frame_kind) = parts.first().and_then(Value::as_str) else {
            return Err(NostrSubscriptionError::InvalidFrame);
        };
        match frame_kind {
            "REQ" | "COUNT" => {
                let subscription_id = parse_subscription_id(&parts)?;
                let filter_values = parts.get(2..).ok_or(NostrSubscriptionError::InvalidFrame)?;
                if filter_values.len() > MAX_FILTERS_PER_REQUEST {
                    return Err(NostrSubscriptionError::TooManyFilters);
                }
                let filters = filter_values
                    .iter()
                    .map(parse_filter)
                    .collect::<Result<Vec<_>, _>>()?;
                if frame_kind == "REQ" {
                    Ok(Self::Req {
                        subscription_id,
                        filters,
                    })
                } else {
                    Ok(Self::Count {
                        subscription_id,
                        filters,
                    })
                }
            }
            "CLOSE" if parts.len() == 2 => Ok(Self::Close {
                subscription_id: parse_subscription_id(&parts)?,
            }),
            _ => Err(NostrSubscriptionError::InvalidFrame),
        }
    }
}

fn parse_subscription_id(parts: &[Value]) -> Result<SubscriptionId, NostrSubscriptionError> {
    let value = parts
        .get(1)
        .and_then(Value::as_str)
        .ok_or(NostrSubscriptionError::InvalidSubscriptionId)?;
    SubscriptionId::new(value)
}

fn parse_filter(value: &Value) -> Result<NostrSubscriptionFilter, NostrSubscriptionError> {
    let object = value
        .as_object()
        .ok_or(NostrSubscriptionError::InvalidFilter)?;
    let ids = parse_string_array(object, "ids")?
        .into_iter()
        .map(|value| {
            HexPrefix::new("ids", value).map_err(|_| NostrSubscriptionError::InvalidFilter)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let authors = parse_string_array(object, "authors")?
        .into_iter()
        .map(|value| PublicKey::from_hex(&value).map_err(|_| NostrSubscriptionError::InvalidFilter))
        .collect::<Result<Vec<_>, _>>()?;
    let kinds = parse_u16_array(object, "kinds")?;
    let since = parse_optional_u64(object, "since")?;
    let until = parse_optional_u64(object, "until")?;
    let limit = parse_optional_u64(object, "limit")?
        .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
        .unwrap_or(100)
        .min(MAX_FILTER_LIMIT);
    let search = match object.get("search") {
        Some(Value::String(value))
            if !value.is_empty()
                && value.len() <= MAX_SEARCH_BYTES
                && !value.chars().any(char::is_control) =>
        {
            Some(value.clone())
        }
        Some(_) => return Err(NostrSubscriptionError::InvalidFilter),
        None => None,
    };
    let mut generic_tags = BTreeMap::new();
    for (key, value) in object {
        if matches!(
            key.as_str(),
            "ids" | "authors" | "kinds" | "since" | "until" | "limit" | "search"
        ) {
            continue;
        }
        let Some(tag_name) = key.strip_prefix('#') else {
            return Err(NostrSubscriptionError::InvalidFilter);
        };
        let mut characters = tag_name.chars();
        let Some(tag) = characters.next() else {
            return Err(NostrSubscriptionError::InvalidFilter);
        };
        if characters.next().is_some() || !tag.is_ascii_alphabetic() {
            return Err(NostrSubscriptionError::InvalidFilter);
        }
        let values = parse_value_string_array(value)?;
        if generic_tags.insert(tag, values).is_some() {
            return Err(NostrSubscriptionError::InvalidFilter);
        }
    }
    let event_filter = EventFilter {
        ids,
        authors,
        kinds,
        since,
        until,
        generic_tags,
    };
    event_filter
        .validate()
        .map_err(|_| NostrSubscriptionError::InvalidFilter)?;
    Ok(NostrSubscriptionFilter {
        event_filter,
        limit,
        search,
    })
}

fn parse_string_array(
    object: &Map<String, Value>,
    key: &'static str,
) -> Result<Vec<String>, NostrSubscriptionError> {
    match object.get(key) {
        Some(value) => parse_value_string_array(value),
        None => Ok(Vec::new()),
    }
}

fn parse_value_string_array(value: &Value) -> Result<Vec<String>, NostrSubscriptionError> {
    let values = value
        .as_array()
        .ok_or(NostrSubscriptionError::InvalidFilter)?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or(NostrSubscriptionError::InvalidFilter)
        })
        .collect()
}

fn parse_u16_array(
    object: &Map<String, Value>,
    key: &'static str,
) -> Result<Vec<u16>, NostrSubscriptionError> {
    let Some(value) = object.get(key) else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or(NostrSubscriptionError::InvalidFilter)?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u16::try_from(value).ok())
                .ok_or(NostrSubscriptionError::InvalidFilter)
        })
        .collect()
}

fn parse_optional_u64(
    object: &Map<String, Value>,
    key: &'static str,
) -> Result<Option<u64>, NostrSubscriptionError> {
    object
        .get(key)
        .map(|value| value.as_u64().ok_or(NostrSubscriptionError::InvalidFilter))
        .transpose()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrStoredEvent {
    wire_event: Value,
}

impl NostrStoredEvent {
    pub fn new(wire_event: Value) -> Result<Self, NostrSubscriptionError> {
        if !wire_event.is_object() {
            return Err(NostrSubscriptionError::InvalidStoredEvent);
        }
        Ok(Self { wire_event })
    }

    pub const fn wire_event(&self) -> &Value {
        &self.wire_event
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SubscriptionResourceToken(u64);

impl SubscriptionResourceToken {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrSubscriptionServiceError {
    Unavailable,
    Denied,
    LimitExceeded,
}

#[async_trait]
pub trait NostrSubscriptionQuery: Send + Sync {
    async fn historical(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        filters: &[NostrSubscriptionFilter],
    ) -> Result<Vec<NostrStoredEvent>, NostrSubscriptionServiceError>;

    async fn count(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        filters: &[NostrSubscriptionFilter],
    ) -> Result<u64, NostrSubscriptionServiceError>;
}

#[async_trait]
pub trait NostrSubscriptionResources: Send + Sync {
    async fn activate(
        &self,
        tenant: &TenantContext,
        connection_id: Uuid,
        subscription_id: &SubscriptionId,
        filters: &[NostrSubscriptionFilter],
    ) -> Result<SubscriptionResourceToken, NostrSubscriptionServiceError>;

    async fn release(
        &self,
        token: SubscriptionResourceToken,
    ) -> Result<(), NostrSubscriptionServiceError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrSubscriptionFailure {
    QueryUnavailable,
    QueryDenied,
    QueryLimitExceeded,
    ResourceUnavailable,
    CleanupFailed,
    TooManySubscriptions,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrSubscriptionOutcome {
    frames: Vec<String>,
    failure: Option<NostrSubscriptionFailure>,
}

impl NostrSubscriptionOutcome {
    fn success(frames: Vec<String>) -> Self {
        Self {
            frames,
            failure: None,
        }
    }

    fn failure(frame: String, failure: NostrSubscriptionFailure) -> Self {
        Self {
            frames: vec![frame],
            failure: Some(failure),
        }
    }

    pub fn frames(&self) -> &[String] {
        &self.frames
    }

    pub const fn failure_reason(&self) -> Option<NostrSubscriptionFailure> {
        self.failure
    }
}

struct ActiveSubscription {
    resource_token: SubscriptionResourceToken,
}

pub struct NostrSubscriptionSession<Q, R> {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    connection_id: Uuid,
    query: Q,
    resources: R,
    active: HashMap<SubscriptionId, ActiveSubscription>,
}

impl<Q, R> NostrSubscriptionSession<Q, R>
where
    Q: NostrSubscriptionQuery,
    R: NostrSubscriptionResources,
{
    pub fn new(
        tenant: TenantContext,
        principal: AuthenticatedPrincipal,
        connection_id: Uuid,
        query: Q,
        resources: R,
    ) -> Result<Self, NostrSubscriptionError> {
        if principal.community_id() != tenant.community_id() {
            return Err(NostrSubscriptionError::TenantMismatch);
        }
        Ok(Self {
            tenant,
            principal,
            connection_id,
            query,
            resources,
            active: HashMap::new(),
        })
    }

    pub fn active_subscription_count(&self) -> usize {
        self.active.len()
    }

    pub async fn handle_frame(
        &mut self,
        raw: &str,
    ) -> Result<NostrSubscriptionOutcome, NostrSubscriptionError> {
        match NostrSubscriptionFrame::parse(raw)? {
            NostrSubscriptionFrame::Req {
                subscription_id,
                filters,
            } => self.handle_req(subscription_id, filters).await,
            NostrSubscriptionFrame::Count {
                subscription_id,
                filters,
            } => self.handle_count(subscription_id, filters).await,
            NostrSubscriptionFrame::Close { subscription_id } => {
                Ok(self.handle_close(subscription_id).await)
            }
        }
    }

    async fn handle_req(
        &mut self,
        subscription_id: SubscriptionId,
        filters: Vec<NostrSubscriptionFilter>,
    ) -> Result<NostrSubscriptionOutcome, NostrSubscriptionError> {
        if !self.active.contains_key(&subscription_id)
            && self.active.len() >= MAX_ACTIVE_SUBSCRIPTIONS
        {
            return Ok(NostrSubscriptionOutcome::failure(
                closed_frame(&subscription_id, "error: too many subscriptions"),
                NostrSubscriptionFailure::TooManySubscriptions,
            ));
        }
        let resource_token = match self
            .resources
            .activate(&self.tenant, self.connection_id, &subscription_id, &filters)
            .await
        {
            Ok(token) => token,
            Err(_) => {
                return Ok(NostrSubscriptionOutcome::failure(
                    closed_frame(&subscription_id, "error: subscription unavailable"),
                    NostrSubscriptionFailure::ResourceUnavailable,
                ));
            }
        };
        if let Some(replaced) = self.active.insert(
            subscription_id.clone(),
            ActiveSubscription { resource_token },
        ) && self
            .resources
            .release(replaced.resource_token)
            .await
            .is_err()
        {
            return Ok(self
                .fail_active(
                    &subscription_id,
                    "error: subscription cleanup failed",
                    NostrSubscriptionFailure::CleanupFailed,
                )
                .await);
        }

        let historical = match self
            .query
            .historical(&self.tenant, &self.principal, &filters)
            .await
        {
            Ok(events) if events.len() <= MAX_HISTORICAL_EVENTS_PER_REQUEST => events,
            Ok(_) => {
                return Ok(self
                    .fail_active(
                        &subscription_id,
                        "restricted: query exceeds result limit",
                        NostrSubscriptionFailure::QueryLimitExceeded,
                    )
                    .await);
            }
            Err(error) => {
                let (reason, failure) = query_failure(error);
                return Ok(self.fail_active(&subscription_id, reason, failure).await);
            }
        };
        let mut frames = historical
            .iter()
            .map(|event| event_frame(&subscription_id, event.wire_event()))
            .collect::<Vec<_>>();
        frames.push(eose_frame(&subscription_id));
        Ok(NostrSubscriptionOutcome::success(frames))
    }

    async fn fail_active(
        &mut self,
        subscription_id: &SubscriptionId,
        reason: &'static str,
        failure: NostrSubscriptionFailure,
    ) -> NostrSubscriptionOutcome {
        let cleanup_failed = match self.active.remove(subscription_id) {
            Some(active) => self.resources.release(active.resource_token).await.is_err(),
            None => false,
        };
        NostrSubscriptionOutcome::failure(
            closed_frame(subscription_id, reason),
            if cleanup_failed {
                NostrSubscriptionFailure::CleanupFailed
            } else {
                failure
            },
        )
    }

    async fn handle_count(
        &self,
        subscription_id: SubscriptionId,
        filters: Vec<NostrSubscriptionFilter>,
    ) -> Result<NostrSubscriptionOutcome, NostrSubscriptionError> {
        match self
            .query
            .count(&self.tenant, &self.principal, &filters)
            .await
        {
            Ok(count) => Ok(NostrSubscriptionOutcome::success(vec![count_frame(
                &subscription_id,
                count,
            )])),
            Err(error) => {
                let (reason, failure) = query_failure(error);
                Ok(NostrSubscriptionOutcome::failure(
                    closed_frame(&subscription_id, reason),
                    failure,
                ))
            }
        }
    }

    async fn handle_close(&mut self, subscription_id: SubscriptionId) -> NostrSubscriptionOutcome {
        let failure = match self.active.remove(&subscription_id) {
            Some(active) if self.resources.release(active.resource_token).await.is_err() => {
                Some(NostrSubscriptionFailure::CleanupFailed)
            }
            _ => None,
        };
        NostrSubscriptionOutcome {
            frames: vec![closed_frame(&subscription_id, "")],
            failure,
        }
    }

    pub async fn cancel(&mut self) -> Result<(), NostrSubscriptionError> {
        let active = self
            .active
            .drain()
            .map(|(_, active)| active)
            .collect::<Vec<_>>();
        let mut cleanup_failed = false;
        for subscription in active {
            if self
                .resources
                .release(subscription.resource_token)
                .await
                .is_err()
            {
                cleanup_failed = true;
            }
        }
        if cleanup_failed {
            Err(NostrSubscriptionError::CleanupFailed)
        } else {
            Ok(())
        }
    }
}

fn query_failure(error: NostrSubscriptionServiceError) -> (&'static str, NostrSubscriptionFailure) {
    match error {
        NostrSubscriptionServiceError::Unavailable => (
            "error: query unavailable",
            NostrSubscriptionFailure::QueryUnavailable,
        ),
        NostrSubscriptionServiceError::Denied => (
            "restricted: query denied",
            NostrSubscriptionFailure::QueryDenied,
        ),
        NostrSubscriptionServiceError::LimitExceeded => (
            "restricted: query exceeds result limit",
            NostrSubscriptionFailure::QueryLimitExceeded,
        ),
    }
}

fn event_frame(subscription_id: &SubscriptionId, event: &Value) -> String {
    serde_json::json!(["EVENT", subscription_id.as_str(), event]).to_string()
}

fn eose_frame(subscription_id: &SubscriptionId) -> String {
    serde_json::json!(["EOSE", subscription_id.as_str()]).to_string()
}

fn closed_frame(subscription_id: &SubscriptionId, reason: &str) -> String {
    serde_json::json!(["CLOSED", subscription_id.as_str(), reason]).to_string()
}

fn count_frame(subscription_id: &SubscriptionId, count: u64) -> String {
    serde_json::json!(["COUNT", subscription_id.as_str(), {"count": count}]).to_string()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrSubscriptionError {
    FrameTooLarge,
    InvalidFrame,
    InvalidSubscriptionId,
    TooManyFilters,
    InvalidFilter,
    InvalidStoredEvent,
    TenantMismatch,
    CleanupFailed,
}

impl fmt::Display for NostrSubscriptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FrameTooLarge => "Nostr frame exceeds the configured limit",
            Self::InvalidFrame => "Nostr subscription frame is invalid",
            Self::InvalidSubscriptionId => "Nostr subscription identifier is invalid",
            Self::TooManyFilters => "Nostr subscription contains too many filters",
            Self::InvalidFilter => "Nostr subscription filter is invalid",
            Self::InvalidStoredEvent => "Nostr stored event is invalid",
            Self::TenantMismatch => "Nostr subscription tenant does not match",
            Self::CleanupFailed => "Nostr subscription cleanup failed",
        })
    }
}

impl Error for NostrSubscriptionError {}
