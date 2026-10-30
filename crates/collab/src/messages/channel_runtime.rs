use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context as _, ensure};

use async_trait::async_trait;
use collaboration_domain::{AggregateId, CommunityId, OperationId, TenantContext};
use futures::{
    FutureExt as _,
    future::{AbortHandle, AbortRegistration, Abortable},
};
use rpc::{ConnectionId, Peer, proto};
use tokio::sync::watch;

use crate::{
    executor::Executor,
    messages::{
        channel_admission::AuthorizedChannel,
        channel_mutation::{MessageOutboxPayload, channel_topic},
        channel_service::{CanonicalMessageService, HydratedMessage, HydratedPage},
    },
    pubsub::{
        envelope::FanoutEnvelope,
        redis::RedisFanoutTransport,
        subscription_bus::{
            FanoutReplayStore as _, FanoutTransport, SubscriptionBus, SubscriptionBusError,
            SubscriptionEvent,
        },
    },
};

const DEDUPLICATION_CAPACITY: usize = 65_536;
const SUBSCRIPTION_QUEUE_CAPACITY: usize = 512;
const REPLAY_BATCH: usize = 512;
const REPLAY_POLL_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone)]
struct RegisteredCommunityBus {
    tenant: TenantContext,
    bus: SubscriptionBus,
}

#[derive(Default)]
struct RedisReceiver {
    abort_handle: parking_lot::Mutex<Option<AbortHandle>>,
}

impl RedisReceiver {
    fn start(&self) -> Option<AbortRegistration> {
        let mut current = self.abort_handle.lock();
        if current.is_some() {
            return None;
        }
        let (handle, registration) = AbortHandle::new_pair();
        *current = Some(handle);
        Some(registration)
    }

    fn stop(&self) {
        let handle = self.abort_handle.lock().take();
        if let Some(handle) = handle {
            handle.abort();
        }
    }
}

impl Drop for RedisReceiver {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct CanonicalMessageRuntime {
    database: Arc<crate::db::Database>,
    service: Arc<CanonicalMessageService>,
    subscriptions: Arc<parking_lot::Mutex<SubscriptionTasks>>,
    redis: Option<RedisFanoutTransport>,
    redis_receiver: RedisReceiver,
    executor: Executor,
}

impl CanonicalMessageRuntime {
    pub fn new(
        database: Arc<crate::db::Database>,
        service: CanonicalMessageService,
        redis_url: Option<&str>,
        executor: Executor,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            database,
            service: Arc::new(service),
            subscriptions: Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default())),
            redis: redis_url.map(RedisFanoutTransport::new).transpose()?,
            redis_receiver: RedisReceiver::default(),
            executor,
        })
    }

    pub fn service(&self) -> &CanonicalMessageService {
        &self.service
    }

    pub fn start_redis_receiver(&self) {
        let registration = {
            let mut subscriptions = self.subscriptions.lock();
            subscriptions.running = true;
            self.redis
                .as_ref()
                .and_then(|_| self.redis_receiver.start())
        };
        let Some(redis) = self.redis.clone() else {
            return;
        };
        let Some(registration) = registration else {
            return;
        };
        let subscriptions = Arc::clone(&self.subscriptions);
        let service = Arc::clone(&self.service);
        let database = Arc::clone(&self.database);
        let communities = self.subscriptions.lock().redis_communities.subscribe();
        let executor = self.executor.clone();
        self.executor.spawn_detached(Abortable::new(async move {
            loop {
                let subscriptions = Arc::clone(&subscriptions);
                let service = Arc::clone(&service);
                let database = Arc::clone(&database);
                let result = redis
                    .subscribe(communities.clone(), move |encoded| {
                        let subscriptions = Arc::clone(&subscriptions);
                        let service = Arc::clone(&service);
                        let database = Arc::clone(&database);
                        async move {
                            let Ok(envelope) = FanoutEnvelope::decode(&encoded) else {
                                log::warn!("discarding invalid Redis collaboration envelope");
                                return;
                            };
                            let registered = subscriptions.lock().buses.get(&envelope.community_id()).cloned();
                            let Some(registered) = registered
                            else {
                                return;
                            };
                            if let Err(error) = database
                                .run_on_database_runtime(service.replay().validate_notification(
                                    &registered.tenant,
                                    &envelope,
                                ))
                                .await
                            {
                                log::warn!("could not validate Redis collaboration notification: {error}");
                                return;
                            }
                            if let Err(error) = registered.bus.receive_remote(&registered.tenant, &encoded) {
                                log::warn!("could not admit Redis collaboration envelope: {error}");
                            }
                        }
                    })
                    .await;
                if let Err(error) = result {
                    log::warn!("Redis collaboration notifications unavailable; PostgreSQL replay remains authoritative: {error}");
                }
                executor.sleep(std::time::Duration::from_secs(1)).await;
            }
        }, registration).map(|_| ()));
    }

    pub fn shutdown(&self) {
        let handles = {
            let mut subscriptions = self.subscriptions.lock();
            self.redis_receiver.stop();
            subscriptions.stop()
        };
        for handle in handles {
            handle.abort();
        }
    }

    pub async fn publish_committed(
        &self,
        authorization: &AuthorizedChannel,
        operation_id: OperationId,
    ) -> Result<FanoutEnvelope, SubscriptionBusError> {
        let envelope = self
            .service
            .replay()
            .envelope_for_operation(&authorization.tenant, operation_id)
            .await?;
        let bus = {
            let subscriptions = self.subscriptions.lock();
            if !subscriptions.running {
                return Err(SubscriptionBusError::Unavailable);
            }
            subscriptions
                .buses
                .get(&authorization.tenant.community_id())
                .cloned()
        };
        if let Some(registered) = bus {
            registered
                .bus
                .publish_authoritative(&authorization.tenant, envelope.clone(), &LocalTransport)
                .await?;
        }
        if let Some(redis) = &self.redis
            && let Err(error) = redis.publish(envelope.encode()?).await
        {
            log::warn!(
                "Redis collaboration notification failed; clients will recover from PostgreSQL: {error}"
            );
        }
        Ok(envelope)
    }

    pub async fn subscribe(
        &self,
        authorization: AuthorizedChannel,
        connection_id: ConnectionId,
        after_sequence: u64,
        peer: Arc<Peer>,
    ) -> Result<(), SubscriptionBusError> {
        let key = (
            connection_id,
            authorization.tenant.community_id(),
            authorization.channel_id,
        );
        let (reservation, replay_registration) =
            SubscriptionReservation::new(Arc::clone(&self.subscriptions), key)?;
        let bus = {
            let mut subscriptions = self.subscriptions.lock();
            reservation.validate_connection(&subscriptions, &peer)?;
            subscriptions.bus_for(&authorization.tenant)?
        };
        let subscription = Abortable::new(
            bus.subscribe(
                &authorization.tenant,
                channel_topic(authorization.channel_id),
                after_sequence,
                SUBSCRIPTION_QUEUE_CAPACITY,
                REPLAY_BATCH,
                self.service.replay(),
            ),
            replay_registration,
        )
        .await
        .map_err(|_| SubscriptionBusError::Unavailable)??;
        let service = Arc::clone(&self.service);
        let database = Arc::clone(&self.database);
        let abort_registration = reservation.activate()?;
        self.executor.spawn_detached(
            Abortable::new(
                MessageSubscription {
                    database,
                    service,
                    authorization,
                    connection_id,
                    peer,
                    executor: self.executor.clone(),
                    delivered_cursor: after_sequence,
                }
                .run(subscription),
                abort_registration,
            )
            .map(move |_| drop(reservation)),
        );
        Ok(())
    }

    pub fn close(
        &self,
        connection_id: ConnectionId,
        community_id: CommunityId,
        channel_id: AggregateId,
    ) {
        let task = self
            .subscriptions
            .lock()
            .remove((connection_id, community_id, channel_id));
        if let Some(task) = task {
            task.abort_handle.abort();
        }
    }

    pub fn close_connection(&self, connection_id: ConnectionId) {
        let tasks = {
            let mut subscriptions = self.subscriptions.lock();
            let keys = subscriptions
                .entries
                .keys()
                .filter(|key| key.0 == connection_id)
                .copied()
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|key| subscriptions.remove(key))
                .collect::<Vec<_>>()
        };
        for task in tasks {
            task.abort_handle.abort();
        }
    }
}

impl Drop for CanonicalMessageRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

type SubscriptionKey = (ConnectionId, CommunityId, AggregateId);

const MAX_CONNECTION_SUBSCRIPTIONS: usize = 1_024;
const MAX_RUNTIME_SUBSCRIPTIONS: usize = crate::pubsub::subscription_bus::MAX_BUS_SUBSCRIPTIONS;

struct SubscriptionTask {
    token: Arc<()>,
    abort_handle: AbortHandle,
}

struct SubscriptionTasks {
    running: bool,
    entries: HashMap<SubscriptionKey, SubscriptionTask>,
    buses: HashMap<CommunityId, RegisteredCommunityBus>,
    redis_communities: watch::Sender<Vec<CommunityId>>,
}

impl Default for SubscriptionTasks {
    fn default() -> Self {
        Self {
            running: true,
            entries: HashMap::new(),
            buses: HashMap::new(),
            redis_communities: watch::channel(Vec::new()).0,
        }
    }
}

impl SubscriptionTasks {
    fn bus_for(&mut self, tenant: &TenantContext) -> Result<SubscriptionBus, SubscriptionBusError> {
        let community_id = tenant.community_id();
        if !self.running || !self.entries.keys().any(|key| key.1 == community_id) {
            return Err(SubscriptionBusError::Unavailable);
        }
        if let Some(registered) = self.buses.get(&community_id) {
            return Ok(registered.bus.clone());
        }
        let bus = SubscriptionBus::new(community_id, DEDUPLICATION_CAPACITY)?;
        self.buses.insert(
            community_id,
            RegisteredCommunityBus {
                tenant: tenant.clone(),
                bus: bus.clone(),
            },
        );
        self.update_redis_communities();
        Ok(bus)
    }

    fn remove(&mut self, key: SubscriptionKey) -> Option<SubscriptionTask> {
        let removed = self.entries.remove(&key);
        if !self.entries.keys().any(|entry| entry.1 == key.1) {
            if self.buses.remove(&key.1).is_some() {
                self.update_redis_communities();
            }
        }
        removed
    }

    fn is_current(&self, key: SubscriptionKey, token: &Arc<()>) -> bool {
        self.running
            && self
                .entries
                .get(&key)
                .is_some_and(|task| Arc::ptr_eq(&task.token, token))
    }

    fn stop(&mut self) -> Vec<AbortHandle> {
        self.running = false;
        self.buses.clear();
        self.update_redis_communities();
        self.entries
            .drain()
            .map(|(_, task)| task.abort_handle)
            .collect()
    }

    fn update_redis_communities(&self) {
        let mut communities = self.buses.keys().copied().collect::<Vec<_>>();
        communities.sort_unstable_by_key(|community| community.as_uuid());
        self.redis_communities.send_replace(communities);
    }
}

struct SubscriptionReservation {
    tasks: Arc<parking_lot::Mutex<SubscriptionTasks>>,
    key: SubscriptionKey,
    token: Arc<()>,
}

impl SubscriptionReservation {
    fn validate_connection(
        &self,
        tasks: &SubscriptionTasks,
        peer: &Peer,
    ) -> Result<(), SubscriptionBusError> {
        // The caller holds the registry lock. Connection loss removes the peer connection
        // before taking that lock, preventing late authorization from reopening a stream.
        if !peer.has_connection(self.key.0) || !tasks.is_current(self.key, &self.token) {
            return Err(SubscriptionBusError::Unavailable);
        }
        Ok(())
    }

    fn new(
        tasks: Arc<parking_lot::Mutex<SubscriptionTasks>>,
        key: SubscriptionKey,
    ) -> Result<(Self, AbortRegistration), SubscriptionBusError> {
        let token = Arc::new(());
        let (abort_handle, registration) = AbortHandle::new_pair();
        let previous = {
            let mut state = tasks.lock();
            if !state.running {
                return Err(SubscriptionBusError::Unavailable);
            }
            if !state.entries.contains_key(&key)
                && (state.entries.len() >= MAX_RUNTIME_SUBSCRIPTIONS
                    || state
                        .entries
                        .keys()
                        .filter(|existing| existing.0 == key.0)
                        .count()
                        >= MAX_CONNECTION_SUBSCRIPTIONS)
            {
                return Err(SubscriptionBusError::CapacityExceeded);
            }
            state.entries.insert(
                key,
                SubscriptionTask {
                    token: Arc::clone(&token),
                    abort_handle,
                },
            )
        };
        if let Some(previous) = previous {
            previous.abort_handle.abort();
        }
        Ok((Self { tasks, key, token }, registration))
    }

    fn activate(&self) -> Result<AbortRegistration, SubscriptionBusError> {
        let mut tasks = self.tasks.lock();
        if !tasks.is_current(self.key, &self.token) {
            return Err(SubscriptionBusError::Unavailable);
        }
        let task = tasks
            .entries
            .get_mut(&self.key)
            .ok_or(SubscriptionBusError::Unavailable)?;
        let (handle, registration) = AbortHandle::new_pair();
        task.abort_handle = handle;
        Ok(registration)
    }
}

impl Drop for SubscriptionReservation {
    fn drop(&mut self) {
        let removed = {
            let mut tasks = self.tasks.lock();
            if tasks
                .entries
                .get(&self.key)
                .is_some_and(|task| Arc::ptr_eq(&task.token, &self.token))
            {
                tasks.remove(self.key)
            } else {
                None
            }
        };
        if let Some(task) = removed {
            task.abort_handle.abort();
        }
    }
}

struct MessageSubscription {
    database: Arc<crate::db::Database>,
    service: Arc<CanonicalMessageService>,
    authorization: AuthorizedChannel,
    connection_id: ConnectionId,
    peer: Arc<Peer>,
    executor: Executor,
    delivered_cursor: u64,
}

enum ReplayProgress {
    More,
    CaughtUp,
    Closed,
    Denied,
}

impl MessageSubscription {
    async fn run(mut self, mut subscription: crate::pubsub::subscription_bus::FanoutSubscription) {
        let mut notifications_open = true;
        let mut last_status = None;
        loop {
            match self.replay_batch().await {
                Ok(ReplayProgress::More) => continue,
                Ok(ReplayProgress::Closed) => break,
                Ok(ReplayProgress::Denied) => {
                    self.send_status(
                        proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorDenied,
                    )
                    .await;
                    break;
                }
                Ok(ReplayProgress::CaughtUp) => {
                    if last_status
                        != Some(proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone)
                    {
                        if !self
                            .send_status(
                                proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
                            )
                            .await
                        {
                            break;
                        }
                        last_status = Some(
                            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorNone,
                        );
                    }
                }
                Err(error) => {
                    log::warn!(
                        "could not replay collaboration messages; retaining delivery cursor: {error:#}"
                    );
                    if last_status
                        != Some(
                            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying,
                        )
                    {
                        if !self.send_status(proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying).await {
                            break;
                        }
                        last_status = Some(
                            proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying,
                        );
                    }
                    self.executor.sleep(REPLAY_POLL_INTERVAL).await;
                    continue;
                }
            }
            if notifications_open {
                futures::select_biased! {
                    event = subscription.recv_hint().fuse() => match event {
                        Some(SubscriptionEvent::Shutdown) => break,
                        Some(SubscriptionEvent::Envelope(_)) => {},
                        None => {
                            // A full hint queue may be evicted. Its cursor never
                            // acknowledges delivery; canonical polling still recovers.
                            notifications_open = false;
                            log::debug!("collaboration hints closed; continuing PostgreSQL replay");
                        }
                    },
                    _ = self.executor.sleep(REPLAY_POLL_INTERVAL).fuse() => {},
                }
            } else {
                self.executor.sleep(REPLAY_POLL_INTERVAL).await;
            }
        }
        subscription.cancel();
    }

    async fn send_status(&self, error_code: proto::CollaborativeMessageErrorCode) -> bool {
        let update = proto::CollaborativeMessageStreamUpdate {
            contract_version: 1,
            community_id: self
                .authorization
                .tenant
                .community_id()
                .as_uuid()
                .as_bytes()
                .to_vec(),
            channel_id: self.authorization.channel_id.as_uuid().as_bytes().to_vec(),
            status: Some(proto::CollaborativeMessageStreamStatus {
                error_code: error_code.into(),
                retry_after_millis: if error_code
                    == proto::CollaborativeMessageErrorCode::CollaborativeMessageErrorRetrying
                {
                    1000
                } else {
                    0
                },
            }),
            ..Default::default()
        };
        match self
            .peer
            .send_with_backpressure(self.connection_id, update)
            .await
        {
            Ok(()) => true,
            Err(error) => {
                log::debug!("collaboration status subscriber disconnected: {error}");
                false
            }
        }
    }

    async fn replay_batch(&mut self) -> anyhow::Result<ReplayProgress> {
        if !self
            .database
            .run_on_database_runtime(self.service.authorization_is_current(&self.authorization))
            .await?
        {
            return Ok(ReplayProgress::Denied);
        }
        let topic = channel_topic(self.authorization.channel_id);
        let envelopes = self
            .database
            .run_on_database_runtime(self.service.replay().load_after(
                &self.authorization.tenant,
                &topic,
                self.delivered_cursor,
                REPLAY_BATCH,
            ))
            .await?;
        let has_more = envelopes.len() == REPLAY_BATCH;
        for envelope in envelopes {
            ensure!(
                envelope.outbox_sequence() > self.delivered_cursor,
                "canonical replay did not advance"
            );
            if !self
                .database
                .run_on_database_runtime(self.service.authorization_is_current(&self.authorization))
                .await?
            {
                return Ok(ReplayProgress::Denied);
            }
            let payload = self
                .database
                .run_on_database_runtime(
                    self.service
                        .replay()
                        .payload(&self.authorization.tenant, envelope.outbox_sequence()),
                )
                .await
                .context("could not load committed collaboration outbox record")?;
            let payload: MessageOutboxPayload = serde_json::from_slice(&payload)
                .context("invalid committed collaboration message payload")?;
            ensure!(
                payload.channel_id == self.authorization.channel_id,
                "replay payload crossed channel boundary"
            );
            let mut message = self
                .database
                .run_on_database_runtime(
                    self.service
                        .message(&self.authorization, payload.message_id),
                )
                .await
                .context("could not hydrate collaboration message update")?;
            ensure!(
                message.is_some()
                    || payload.kind
                        == crate::messages::channel_mutation::MessageOperationKind::Acknowledge,
                "committed message projection is unavailable"
            );
            if let Some(message) = &mut message {
                // A receipt identifies the replayed command, while hydration can
                // already contain a later edit/reaction and retains its version.
                message.accepted_operation_id = Some(payload.operation_id);
            }
            if let Err(error) = self
                .peer
                .send_with_backpressure(
                    self.connection_id,
                    proto::CollaborativeMessageStreamUpdate {
                        contract_version: 1,
                        status: None,
                        community_id: self
                            .authorization
                            .tenant
                            .community_id()
                            .as_uuid()
                            .as_bytes()
                            .to_vec(),
                        channel_id: self.authorization.channel_id.as_uuid().as_bytes().to_vec(),
                        outbox_sequence: envelope.outbox_sequence(),
                        message: message.map(message_to_proto),
                        operation_kind: operation_kind_to_proto(payload.kind).into(),
                        actor_principal_id: payload
                            .actor_principal_id
                            .map_or_else(Vec::new, |principal_id| {
                                principal_id.as_uuid().as_bytes().to_vec()
                            }),
                        acknowledged_outbox_sequence: payload
                            .acknowledged_outbox_sequence
                            .unwrap_or_default(),
                    },
                )
                .await
            {
                log::debug!("collaboration message subscriber disconnected: {error}");
                return Ok(ReplayProgress::Closed);
            }
            self.delivered_cursor = envelope.outbox_sequence();
        }
        Ok(if has_more {
            ReplayProgress::More
        } else {
            ReplayProgress::CaughtUp
        })
    }
}

fn operation_kind_to_proto(
    kind: crate::messages::channel_mutation::MessageOperationKind,
) -> proto::CollaborativeMessageOperationKind {
    use crate::messages::channel_mutation::MessageOperationKind;
    match kind {
        MessageOperationKind::Create => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageCreate
        }
        MessageOperationKind::Edit => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageEdit
        }
        MessageOperationKind::Delete => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageDelete
        }
        MessageOperationKind::ReactionAdd => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageReactionAdd
        }
        MessageOperationKind::ReactionRemove => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageReactionRemove
        }
        MessageOperationKind::Acknowledge => {
            proto::CollaborativeMessageOperationKind::CollaborativeMessageAcknowledge
        }
    }
}

pub fn page_to_proto(page: HydratedPage) -> proto::CollaborativeMessagePage {
    proto::CollaborativeMessagePage {
        messages: page.messages.into_iter().map(message_to_proto).collect(),
        next_cursor: page
            .next_cursor
            .map(|cursor| proto::CollaborativeMessageCursor {
                snapshot_micros: page.snapshot.as_micros(),
                message_created_at: cursor.message_created_at,
                source_event_id: cursor.source_event_id.as_bytes().to_vec(),
                outbox_sequence: page.authoritative_outbox_cursor,
            }),
        done: !page.has_more,
        authoritative_outbox_cursor: page.authoritative_outbox_cursor,
    }
}

pub fn message_to_proto(message: HydratedMessage) -> proto::CollaborativeMessageRecord {
    proto::CollaborativeMessageRecord {
        community_id: message.community_id.as_uuid().as_bytes().to_vec(),
        channel_id: message.channel_id.as_uuid().as_bytes().to_vec(),
        message_id: message.message_id.as_uuid().as_bytes().to_vec(),
        source_event_id: message.source_event_id.as_bytes().to_vec(),
        current_event_id: message.current_event_id.as_bytes().to_vec(),
        author_principal_id: message.author_principal_id.as_uuid().as_bytes().to_vec(),
        author_display_name: message.author_display_name,
        author_avatar_url: message.author_avatar_url,
        body: message.body,
        created_at: message.created_at,
        version: message.version,
        edited: message.edited,
        deleted: message.deleted,
        reply_to_event_id: message
            .reply_to_event_id
            .map_or_else(Vec::new, |event_id| event_id.as_bytes().to_vec()),
        reactions: message
            .reactions
            .into_iter()
            .map(|reaction| proto::CollaborativeMessageReaction {
                value: reaction.value,
                actor_principal_id: reaction.actor_principal_id.as_uuid().as_bytes().to_vec(),
                source_event_id: reaction.source_event_id.as_bytes().to_vec(),
            })
            .collect(),
        accepted_operation_id: message
            .accepted_operation_id
            .map_or_else(String::new, |operation_id| operation_id.to_string()),
        outbox_sequence: message.outbox_sequence,
        reaction_version: message.reaction_version,
    }
}

struct LocalTransport;

#[async_trait]
impl FanoutTransport for LocalTransport {
    async fn publish(&self, _encoded_envelope: Vec<u8>) -> Result<(), SubscriptionBusError> {
        Ok(())
    }
}

#[cfg(test)]
mod subscription_tests {
    use super::*;
    use futures::future;

    fn key(community_id: CommunityId) -> SubscriptionKey {
        (
            ConnectionId { owner_id: 7, id: 1 },
            community_id,
            AggregateId::from_uuid(uuid::Uuid::from_u128(9)),
        )
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        crate::tenant_admission::bind_rpc_tenant(
            Some(
                collaboration_domain::TrustedTenantRoute::from_listener(
                    community_id,
                    "runtime-lifecycle",
                )
                .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    #[test]
    fn community_bus_requires_admission_and_releases_after_last_subscription() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let tenant = tenant(CommunityId::new());
        let communities = tasks.lock().redis_communities.subscribe();
        assert!(matches!(
            tasks.lock().bus_for(&tenant),
            Err(SubscriptionBusError::Unavailable)
        ));
        assert!(tasks.lock().buses.is_empty());
        assert!(communities.borrow().is_empty());
        let first_key = key(tenant.community_id());
        let (first, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), first_key).expect("first admission");
        let mut second_key = first_key;
        second_key.2 = AggregateId::new();
        let (second, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), second_key).expect("second admission");
        tasks.lock().bus_for(&tenant).expect("admitted bus");
        assert_eq!(*communities.borrow(), vec![tenant.community_id()]);
        drop(first);
        assert_eq!(tasks.lock().buses.len(), 1);
        assert_eq!(*communities.borrow(), vec![tenant.community_id()]);
        drop(second);
        assert!(tasks.lock().buses.is_empty());
        assert!(tasks.lock().entries.is_empty());
        assert!(communities.borrow().is_empty());
    }

    #[test]
    fn stale_reservation_cannot_remove_replacement_bus_after_explicit_close() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let tenant = tenant(CommunityId::new());
        let key = key(tenant.community_id());
        let (old, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), key).expect("old admission");
        tasks.lock().bus_for(&tenant).expect("old bus");
        tasks
            .lock()
            .remove(key)
            .expect("explicit close")
            .abort_handle
            .abort();
        assert!(tasks.lock().buses.is_empty());
        let (replacement, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), key).expect("new admission");
        tasks.lock().bus_for(&tenant).expect("replacement bus");
        drop(old);
        assert_eq!(tasks.lock().buses.len(), 1);
        assert!(replacement.activate().is_ok());
        for handle in tasks.lock().stop() {
            handle.abort();
        }
        assert!(tasks.lock().buses.is_empty());
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn pending_replays_share_the_connection_budget_across_communities() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let mut reservations = Vec::new();
        for _ in 0..MAX_CONNECTION_SUBSCRIPTIONS {
            let (reservation, _) =
                SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new()))
                    .expect("within connection budget");
            reservations.push(reservation);
        }
        assert!(matches!(
            SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new())),
            Err(SubscriptionBusError::CapacityExceeded)
        ));
        let old = reservations.pop().expect("reserved subscription");
        let (replacement, _) = SubscriptionReservation::new(Arc::clone(&tasks), old.key)
            .expect("replacement does not consume another slot");
        drop(old);
        assert_eq!(tasks.lock().entries.len(), MAX_CONNECTION_SUBSCRIPTIONS);
        drop(replacement);
        assert!(SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new())).is_ok());
        drop(reservations);
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn runtime_budget_includes_pending_replays_on_different_connections() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let community_id = CommunityId::new();
        let mut reservations = Vec::new();
        for index in 0..MAX_RUNTIME_SUBSCRIPTIONS {
            let mut key = key(community_id);
            key.0.id = u32::try_from(index).expect("bounded connection identifier");
            let (reservation, _) = SubscriptionReservation::new(Arc::clone(&tasks), key)
                .expect("within runtime budget");
            reservations.push(reservation);
        }
        let mut extra = key(CommunityId::new());
        extra.0.owner_id = 8;
        assert!(matches!(
            SubscriptionReservation::new(Arc::clone(&tasks), extra),
            Err(SubscriptionBusError::CapacityExceeded)
        ));
        let old = reservations.pop().expect("reserved subscription");
        let (replacement, _) = SubscriptionReservation::new(Arc::clone(&tasks), old.key)
            .expect("replacement at capacity");
        drop(old);
        assert!(replacement.activate().is_ok());
        assert_eq!(tasks.lock().entries.len(), MAX_RUNTIME_SUBSCRIPTIONS);
        drop(replacement);
        assert!(SubscriptionReservation::new(Arc::clone(&tasks), extra).is_ok());
        drop(reservations);
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn absent_connection_cannot_finish_subscription_admission() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let (reservation, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new()))
                .expect("pending authorization");
        let peer = Peer::new(7);
        assert_eq!(
            reservation.validate_connection(&tasks.lock(), &peer),
            Err(SubscriptionBusError::Unavailable)
        );
        drop(reservation);
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn replacement_cancels_replay_without_stale_cleanup_removing_its_successor() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let key = key(CommunityId::new());
        let (first, registration) =
            SubscriptionReservation::new(Arc::clone(&tasks), key).expect("first");
        let (second, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), key).expect("replacement");
        assert!(matches!(
            Abortable::new(future::pending::<()>(), registration).now_or_never(),
            Some(Err(_))
        ));
        assert!(matches!(
            first.activate(),
            Err(SubscriptionBusError::Unavailable)
        ));
        drop(first);
        assert!(second.activate().is_ok());
        assert_eq!(tasks.lock().entries.len(), 1);
        drop(second);
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn dropping_pending_admission_cancels_replay_and_releases_its_entry() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let (reservation, registration) =
            SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new()))
                .expect("reservation");
        drop(reservation);
        assert!(matches!(
            Abortable::new(future::pending::<()>(), registration).now_or_never(),
            Some(Err(_))
        ));
        assert!(tasks.lock().entries.is_empty());
    }

    #[test]
    fn shutdown_rejects_late_admission_even_after_restart() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let key = key(CommunityId::new());
        let (old, _) =
            SubscriptionReservation::new(Arc::clone(&tasks), key).expect("old reservation");
        let active = old.activate().expect("live stream");
        let handles = tasks.lock().stop();
        for handle in handles {
            handle.abort();
        }
        assert!(matches!(
            Abortable::new(future::pending::<()>(), active).now_or_never(),
            Some(Err(_))
        ));
        assert!(SubscriptionReservation::new(Arc::clone(&tasks), key).is_err());
        tasks.lock().running = true;
        let (new, _) = SubscriptionReservation::new(Arc::clone(&tasks), key).expect("new epoch");
        assert!(matches!(
            old.activate(),
            Err(SubscriptionBusError::Unavailable)
        ));
        drop(old);
        assert!(new.activate().is_ok());
    }

    #[test]
    fn same_channel_identifier_in_two_communities_has_independent_lifetimes() {
        let tasks = Arc::new(parking_lot::Mutex::new(SubscriptionTasks::default()));
        let (first, _) = SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new()))
            .expect("first community");
        let (second, _) = SubscriptionReservation::new(Arc::clone(&tasks), key(CommunityId::new()))
            .expect("second community");
        assert_eq!(tasks.lock().entries.len(), 2);
        drop(first);
        assert!(second.activate().is_ok());
        assert_eq!(tasks.lock().entries.len(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future;

    #[test]
    fn redis_receiver_starts_once_and_stops_a_pending_receive() {
        let receiver = RedisReceiver::default();
        let registration = receiver.start().expect("first receiver starts");
        assert!(receiver.start().is_none());
        let task = Abortable::new(future::pending::<()>(), registration);
        futures::pin_mut!(task);
        assert!(task.as_mut().now_or_never().is_none());
        receiver.stop();
        assert!(matches!(task.now_or_never(), Some(Err(_))));
    }

    #[test]
    fn redis_receiver_stop_before_spawn_cancels_work_and_allows_restart() {
        let receiver = RedisReceiver::default();
        let registration = receiver.start().expect("first receiver starts");
        let ran = std::cell::Cell::new(false);
        let task = Abortable::new(async { ran.set(true) }, registration);
        receiver.stop();
        let restarted = receiver.start().expect("receiver restarts after stop");
        assert!(matches!(task.now_or_never(), Some(Err(_))));
        assert!(!ran.get());
        assert_eq!(
            Abortable::new(future::ready(()), restarted).now_or_never(),
            Some(Ok(()))
        );
    }

    #[test]
    fn dropping_redis_receiver_cancels_detached_work() {
        let receiver = RedisReceiver::default();
        let registration = receiver.start().expect("receiver starts");
        let task = Abortable::new(future::pending::<()>(), registration);
        futures::pin_mut!(task);
        assert!(task.as_mut().now_or_never().is_none());
        drop(receiver);
        assert!(matches!(task.now_or_never(), Some(Err(_))));
    }
}
