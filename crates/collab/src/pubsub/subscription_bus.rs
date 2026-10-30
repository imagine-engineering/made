use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, Weak},
};

use async_trait::async_trait;
use collaboration_domain::{CommunityId, SourceSystem, TenantContext};
use tokio::sync::mpsc;

use super::envelope::{
    FanoutAdmission, FanoutEnvelope, FanoutEnvelopeError, LocalFanoutDeduplicator,
};

pub const MAX_BUS_SUBSCRIPTIONS: usize = 4_096;
pub const MAX_SUBSCRIPTION_QUEUE: usize = 1_024;
pub const MAX_REPLAY_BATCH: usize = 1_000;

#[async_trait]
pub trait FanoutReplayStore: Send + Sync {
    async fn load_after(
        &self,
        tenant: &TenantContext,
        topic: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<FanoutEnvelope>, SubscriptionBusError>;
}

#[async_trait]
pub trait FanoutTransport: Send + Sync {
    async fn publish(&self, encoded_envelope: Vec<u8>) -> Result<(), SubscriptionBusError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubscriptionEvent {
    Envelope(FanoutEnvelope),
    Shutdown,
}

struct Subscriber {
    topic: String,
    cursor: u64,
    sender: mpsc::Sender<SubscriptionEvent>,
    initializing: bool,
    buffered: Vec<FanoutEnvelope>,
}

struct BusState {
    next_subscription_id: u64,
    shutdown: bool,
    subscribers: HashMap<u64, Subscriber>,
    deduplicator: LocalFanoutDeduplicator,
}

struct SubscriptionBusInner {
    community_id: CommunityId,
    state: Mutex<BusState>,
}

#[derive(Clone)]
pub struct SubscriptionBus {
    inner: Arc<SubscriptionBusInner>,
}

impl SubscriptionBus {
    pub fn new(
        community_id: CommunityId,
        deduplication_capacity: usize,
    ) -> Result<Self, SubscriptionBusError> {
        let deduplicator = LocalFanoutDeduplicator::new(community_id, deduplication_capacity)?;
        Ok(Self {
            inner: Arc::new(SubscriptionBusInner {
                community_id,
                state: Mutex::new(BusState {
                    next_subscription_id: 1,
                    shutdown: false,
                    subscribers: HashMap::new(),
                    deduplicator,
                }),
            }),
        })
    }

    pub async fn subscribe<R>(
        &self,
        tenant: &TenantContext,
        topic: impl Into<String>,
        after_sequence: u64,
        queue_capacity: usize,
        replay_limit: usize,
        replay_store: &R,
    ) -> Result<FanoutSubscription, SubscriptionBusError>
    where
        R: FanoutReplayStore,
    {
        self.validate_tenant(tenant)?;
        let topic = topic.into();
        if !valid_topic(&topic)
            || queue_capacity == 0
            || queue_capacity > MAX_SUBSCRIPTION_QUEUE
            || replay_limit == 0
            || replay_limit > MAX_REPLAY_BATCH
            || replay_limit > queue_capacity
        {
            return Err(SubscriptionBusError::InvalidRequest);
        }
        let (sender, receiver) = mpsc::channel(queue_capacity);
        let subscription_id = {
            let mut state = self.lock_state()?;
            if state.shutdown {
                return Err(SubscriptionBusError::Shutdown);
            }
            if state.subscribers.len() >= MAX_BUS_SUBSCRIPTIONS {
                return Err(SubscriptionBusError::CapacityExceeded);
            }
            let subscription_id = state.next_subscription_id;
            state.next_subscription_id = state
                .next_subscription_id
                .checked_add(1)
                .ok_or(SubscriptionBusError::CapacityExceeded)?;
            state.subscribers.insert(
                subscription_id,
                Subscriber {
                    topic: topic.clone(),
                    cursor: after_sequence,
                    sender,
                    initializing: true,
                    buffered: Vec::new(),
                },
            );
            subscription_id
        };

        let subscription = FanoutSubscription {
            subscription_id,
            receiver,
            bus: Arc::downgrade(&self.inner),
        };
        let replay = replay_store
            .load_after(tenant, &topic, after_sequence, replay_limit)
            .await?;
        self.finish_subscription(tenant, subscription_id, &topic, after_sequence, replay)?;
        Ok(subscription)
    }

    pub async fn publish_authoritative<T>(
        &self,
        tenant: &TenantContext,
        envelope: FanoutEnvelope,
        transport: &T,
    ) -> Result<FanoutAdmission, SubscriptionBusError>
    where
        T: FanoutTransport,
    {
        let admission = self.deliver_live(tenant, &envelope)?;
        transport.publish(envelope.encode()?).await?;
        Ok(admission)
    }

    pub fn receive_remote(
        &self,
        tenant: &TenantContext,
        encoded_envelope: &[u8],
    ) -> Result<FanoutAdmission, SubscriptionBusError> {
        let envelope = FanoutEnvelope::decode(encoded_envelope)?;
        self.deliver_live(tenant, &envelope)
    }

    pub fn shutdown(&self) -> Result<(), SubscriptionBusError> {
        let subscribers = {
            let mut state = self.lock_state()?;
            if state.shutdown {
                return Ok(());
            }
            state.shutdown = true;
            std::mem::take(&mut state.subscribers)
        };
        for subscriber in subscribers.into_values() {
            if let Err(error) = subscriber.sender.try_send(SubscriptionEvent::Shutdown) {
                log::debug!("could not enqueue subscription shutdown marker: {error}");
            }
        }
        Ok(())
    }

    pub fn subscription_count(&self) -> Result<usize, SubscriptionBusError> {
        Ok(self.lock_state()?.subscribers.len())
    }

    fn finish_subscription(
        &self,
        tenant: &TenantContext,
        subscription_id: u64,
        topic: &str,
        after_sequence: u64,
        replay: Vec<FanoutEnvelope>,
    ) -> Result<(), SubscriptionBusError> {
        self.validate_tenant(tenant)?;
        let mut state = self.lock_state()?;
        let subscriber = state
            .subscribers
            .get_mut(&subscription_id)
            .ok_or(SubscriptionBusError::Cancelled)?;
        let buffered = std::mem::take(&mut subscriber.buffered);
        let mut candidates = replay;
        candidates.extend(buffered);
        candidates.sort_unstable_by_key(FanoutEnvelope::outbox_sequence);
        let mut source_keys = HashSet::new();
        let mut cursor = after_sequence;
        for envelope in candidates {
            validate_delivery(self.inner.community_id, topic, after_sequence, &envelope)?;
            if !source_keys.insert(source_key(&envelope)?) {
                cursor = cursor.max(envelope.outbox_sequence());
                continue;
            }
            if envelope.outbox_sequence() <= cursor {
                return Err(SubscriptionBusError::InvalidRequest);
            }
            cursor = envelope.outbox_sequence();
            subscriber
                .sender
                .try_send(SubscriptionEvent::Envelope(envelope))
                .map_err(|_| SubscriptionBusError::Backpressure)?;
        }
        subscriber.cursor = cursor;
        subscriber.initializing = false;
        Ok(())
    }

    fn deliver_live(
        &self,
        tenant: &TenantContext,
        envelope: &FanoutEnvelope,
    ) -> Result<FanoutAdmission, SubscriptionBusError> {
        self.validate_tenant(tenant)?;
        let mut state = self.lock_state()?;
        if state.shutdown {
            return Err(SubscriptionBusError::Shutdown);
        }
        let admission = state.deduplicator.admit(tenant, envelope)?;
        if admission == FanoutAdmission::Duplicate {
            return Ok(admission);
        }
        let mut remove = Vec::new();
        for (subscription_id, subscriber) in &mut state.subscribers {
            if subscriber.topic != envelope.topic()
                || envelope.outbox_sequence() <= subscriber.cursor
            {
                continue;
            }
            if subscriber.initializing {
                if subscriber.buffered.len() >= MAX_SUBSCRIPTION_QUEUE {
                    remove.push(*subscription_id);
                } else {
                    subscriber.buffered.push(envelope.clone());
                }
                continue;
            }
            match subscriber
                .sender
                .try_send(SubscriptionEvent::Envelope(envelope.clone()))
            {
                Ok(()) => subscriber.cursor = envelope.outbox_sequence(),
                Err(_) => remove.push(*subscription_id),
            }
        }
        for subscription_id in remove {
            state.subscribers.remove(&subscription_id);
        }
        Ok(admission)
    }

    fn validate_tenant(&self, tenant: &TenantContext) -> Result<(), SubscriptionBusError> {
        if tenant.community_id() != self.inner.community_id {
            return Err(SubscriptionBusError::TenantMismatch);
        }
        Ok(())
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, BusState>, SubscriptionBusError> {
        self.inner
            .state
            .lock()
            .map_err(|_| SubscriptionBusError::Unavailable)
    }
}

pub struct FanoutSubscription {
    subscription_id: u64,
    receiver: mpsc::Receiver<SubscriptionEvent>,
    bus: Weak<SubscriptionBusInner>,
}

impl FanoutSubscription {
    pub async fn recv(&mut self) -> Option<SubscriptionEvent> {
        self.receiver.recv().await
    }

    // Use only when envelopes are wakeups for a separate canonical replay reader.
    // Coalescing here must never advance that reader's delivery cursor.
    pub async fn recv_hint(&mut self) -> Option<SubscriptionEvent> {
        let mut event = self.receiver.recv().await?;
        for _ in 0..MAX_SUBSCRIPTION_QUEUE {
            if event == SubscriptionEvent::Shutdown {
                return Some(event);
            }
            match self.receiver.try_recv() {
                Ok(next) => event = next,
                Err(_) => return Some(event),
            }
        }
        Some(event)
    }

    pub fn cancel(mut self) {
        self.remove_from_bus();
        self.receiver.close();
    }

    fn remove_from_bus(&mut self) {
        if let Some(bus) = self.bus.upgrade()
            && let Ok(mut state) = bus.state.lock()
        {
            state.subscribers.remove(&self.subscription_id);
        }
    }
}

impl Drop for FanoutSubscription {
    fn drop(&mut self) {
        self.remove_from_bus();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SubscriptionBusError {
    #[error("subscription bus request is invalid")]
    InvalidRequest,
    #[error("subscription bus crossed its tenant boundary")]
    TenantMismatch,
    #[error("subscription bus capacity was exceeded")]
    CapacityExceeded,
    #[error("subscription bus subscriber could not keep up")]
    Backpressure,
    #[error("subscription was cancelled during replay")]
    Cancelled,
    #[error("subscription bus is shut down")]
    Shutdown,
    #[error("subscription bus dependency is unavailable")]
    Unavailable,
    #[error(transparent)]
    Envelope(#[from] FanoutEnvelopeError),
}

fn validate_delivery(
    community_id: CommunityId,
    topic: &str,
    cursor: u64,
    envelope: &FanoutEnvelope,
) -> Result<(), SubscriptionBusError> {
    if envelope.community_id() != community_id
        || envelope.topic() != topic
        || envelope.outbox_sequence() <= cursor
    {
        return Err(SubscriptionBusError::InvalidRequest);
    }
    Ok(())
}

fn source_key(
    envelope: &FanoutEnvelope,
) -> Result<(SourceSystem, String, String), SubscriptionBusError> {
    Ok((
        envelope.provenance().source_system,
        envelope.provenance().source_record_id.as_str().to_owned(),
        envelope
            .provenance()
            .source_version
            .clone()
            .ok_or(SubscriptionBusError::InvalidRequest)?,
    ))
}

fn valid_topic(topic: &str) -> bool {
    !topic.is_empty()
        && topic.len() <= 128
        && topic.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b':' | b'_' | b'-')
        })
}
