use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use async_trait::async_trait;
use collab::{
    pubsub::{
        envelope::{FanoutAdmission, FanoutEnvelope},
        subscription_bus::{
            FanoutReplayStore, FanoutTransport, SubscriptionBus, SubscriptionBusError,
            SubscriptionEvent,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    CommunityId, Provenance, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use futures::FutureExt as _;
use sha2::{Digest, Sha256};
use uuid::Uuid;

fn tenant() -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "subscription-bus")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn envelope(tenant: &TenantContext, sequence: u64) -> FanoutEnvelope {
    FanoutEnvelope::new(
        tenant.community_id(),
        sequence,
        "conversation.activity",
        Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new(format!("activity:{sequence}")).expect("source ID"),
            1_900_000_000_000 + sequence,
        )
        .with_source_version(sequence.to_string()),
        Sha256::digest(format!("payload:{sequence}").as_bytes()).into(),
    )
    .expect("envelope")
}

#[derive(Default)]
struct TestReplayStore {
    envelopes: Mutex<Vec<FanoutEnvelope>>,
}

impl TestReplayStore {
    fn replace(&self, envelopes: Vec<FanoutEnvelope>) {
        *self.envelopes.lock().expect("replay lock") = envelopes;
    }
}

#[async_trait]
impl FanoutReplayStore for TestReplayStore {
    async fn load_after(
        &self,
        tenant: &TenantContext,
        topic: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<FanoutEnvelope>, SubscriptionBusError> {
        let mut envelopes = self
            .envelopes
            .lock()
            .map_err(|_| SubscriptionBusError::Unavailable)?
            .iter()
            .filter(|envelope| {
                envelope.community_id() == tenant.community_id()
                    && envelope.topic() == topic
                    && envelope.outbox_sequence() > after_sequence
            })
            .cloned()
            .collect::<Vec<_>>();
        envelopes.sort_unstable_by_key(FanoutEnvelope::outbox_sequence);
        envelopes.truncate(limit);
        Ok(envelopes)
    }
}

struct TestTransport {
    published: Mutex<Vec<Vec<u8>>>,
    available: AtomicBool,
}

struct PendingReplayStore;

#[async_trait]
impl FanoutReplayStore for PendingReplayStore {
    async fn load_after(
        &self,
        _: &TenantContext,
        _: &str,
        _: u64,
        _: usize,
    ) -> Result<Vec<FanoutEnvelope>, SubscriptionBusError> {
        std::future::pending().await
    }
}

#[test]
fn cancellation_during_replay_releases_the_bus_subscription() {
    let tenant = tenant();
    let bus = SubscriptionBus::new(tenant.community_id(), 16).expect("bus");
    let mut pending = Box::pin(bus.subscribe(
        &tenant,
        "conversation.activity",
        0,
        16,
        16,
        &PendingReplayStore,
    ));
    assert!(pending.as_mut().now_or_never().is_none());
    assert_eq!(bus.subscription_count().expect("registered replay"), 1);
    drop(pending);
    assert_eq!(bus.subscription_count().expect("cancelled replay"), 0);
}

impl Default for TestTransport {
    fn default() -> Self {
        Self {
            published: Mutex::new(Vec::new()),
            available: AtomicBool::new(true),
        }
    }
}

impl TestTransport {
    fn drain(&self) -> Vec<Vec<u8>> {
        std::mem::take(&mut *self.published.lock().expect("transport lock"))
    }

    fn set_available(&self, available: bool) {
        self.available.store(available, Ordering::SeqCst);
    }
}

#[async_trait]
impl FanoutTransport for TestTransport {
    async fn publish(&self, encoded_envelope: Vec<u8>) -> Result<(), SubscriptionBusError> {
        if !self.available.load(Ordering::SeqCst) {
            return Err(SubscriptionBusError::Unavailable);
        }
        self.published
            .lock()
            .map_err(|_| SubscriptionBusError::Unavailable)?
            .push(encoded_envelope);
        Ok(())
    }
}

async fn next_sequence(
    subscription: &mut collab::pubsub::subscription_bus::FanoutSubscription,
) -> u64 {
    match subscription.recv().await.expect("subscription event") {
        SubscriptionEvent::Envelope(envelope) => envelope.outbox_sequence(),
        SubscriptionEvent::Shutdown => panic!("unexpected shutdown"),
    }
}

#[tokio::test]
async fn collaboration_subscription_bus_orders_two_replicas_replays_and_cleans_up() {
    let tenant = tenant();
    let replay = Arc::new(TestReplayStore::default());
    replay.replace(vec![envelope(&tenant, 1), envelope(&tenant, 2)]);
    let transport = Arc::new(TestTransport::default());
    let first_bus = SubscriptionBus::new(tenant.community_id(), 64).expect("first bus");
    let second_bus = SubscriptionBus::new(tenant.community_id(), 64).expect("second bus");
    let mut first = first_bus
        .subscribe(&tenant, "conversation.activity", 0, 16, 16, replay.as_ref())
        .await
        .expect("first subscription");
    let mut second = second_bus
        .subscribe(&tenant, "conversation.activity", 0, 16, 16, replay.as_ref())
        .await
        .expect("second subscription");
    assert_eq!(
        (
            next_sequence(&mut first).await,
            next_sequence(&mut first).await
        ),
        (1, 2)
    );
    assert_eq!(
        (
            next_sequence(&mut second).await,
            next_sequence(&mut second).await
        ),
        (1, 2)
    );

    let third = envelope(&tenant, 3);
    assert_eq!(
        first_bus
            .publish_authoritative(&tenant, third.clone(), transport.as_ref())
            .await,
        Ok(FanoutAdmission::New)
    );
    assert_eq!(next_sequence(&mut first).await, 3);
    let encoded = transport.drain();
    assert_eq!(encoded.len(), 1);
    assert_eq!(
        second_bus.receive_remote(&tenant, &encoded[0]),
        Ok(FanoutAdmission::New)
    );
    assert_eq!(next_sequence(&mut second).await, 3);
    assert_eq!(
        first_bus.receive_remote(&tenant, &encoded[0]),
        Ok(FanoutAdmission::Duplicate),
        "the publishing replica suppresses its Redis echo"
    );

    drop(second);
    replay.replace(vec![
        envelope(&tenant, 1),
        envelope(&tenant, 2),
        third,
        envelope(&tenant, 4),
    ]);
    let mut reconnected = second_bus
        .subscribe(&tenant, "conversation.activity", 2, 16, 16, replay.as_ref())
        .await
        .expect("reconnected subscription");
    assert_eq!(
        (
            next_sequence(&mut reconnected).await,
            next_sequence(&mut reconnected).await
        ),
        (3, 4)
    );
    reconnected.cancel();
    assert_eq!(second_bus.subscription_count(), Ok(0));

    first_bus.shutdown().expect("shutdown first bus");
    assert_eq!(first.recv().await, Some(SubscriptionEvent::Shutdown));
    assert_eq!(first_bus.subscription_count(), Ok(0));
    assert!(matches!(
        first_bus
            .subscribe(&tenant, "conversation.activity", 0, 1, 1, replay.as_ref())
            .await,
        Err(SubscriptionBusError::Shutdown)
    ));
}

#[tokio::test]
async fn collaboration_subscription_bus_rejects_foreign_replay_and_backpressure() {
    let tenant = tenant();
    let foreign_tenant = bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(
                CommunityId::from_uuid(Uuid::from_u128(2)),
                "foreign-subscription-bus",
            )
            .expect("trusted route"),
        ),
        &[],
    )
    .expect("foreign tenant");
    let replay = TestReplayStore::default();
    replay.replace(vec![envelope(&tenant, 1), envelope(&tenant, 2)]);
    let bus = SubscriptionBus::new(tenant.community_id(), 8).expect("bus");
    assert!(matches!(
        bus.subscribe(&foreign_tenant, "conversation.activity", 0, 2, 2, &replay,)
            .await,
        Err(SubscriptionBusError::TenantMismatch)
    ));
    assert_eq!(bus.subscription_count(), Ok(0));
    replay.replace(Vec::new());
    let mut slow = bus
        .subscribe(&tenant, "conversation.activity", 0, 1, 1, &replay)
        .await
        .expect("slow subscription");
    let transport = TestTransport::default();
    bus.publish_authoritative(&tenant, envelope(&tenant, 1), &transport)
        .await
        .expect("first live delivery");
    bus.publish_authoritative(&tenant, envelope(&tenant, 2), &transport)
        .await
        .expect("second live delivery removes the full subscriber");
    assert_eq!(bus.subscription_count(), Ok(0));
    assert_eq!(next_sequence(&mut slow).await, 1);
    assert_eq!(slow.recv().await, None);

    let retry_bus = SubscriptionBus::new(tenant.community_id(), 8).expect("retry bus");
    let retry_transport = TestTransport::default();
    retry_transport.set_available(false);
    let retry_envelope = envelope(&tenant, 3);
    assert_eq!(
        retry_bus
            .publish_authoritative(&tenant, retry_envelope.clone(), &retry_transport)
            .await,
        Err(SubscriptionBusError::Unavailable)
    );
    retry_transport.set_available(true);
    assert_eq!(
        retry_bus
            .publish_authoritative(&tenant, retry_envelope, &retry_transport)
            .await,
        Ok(FanoutAdmission::Duplicate)
    );
    assert_eq!(retry_transport.drain().len(), 1);
}

#[tokio::test]
async fn coalesced_hints_preserve_replay_and_shutdown() {
    let tenant = tenant();
    let bus = SubscriptionBus::new(tenant.community_id(), 16).expect("bus");
    let replay = TestReplayStore::default();
    replay.replace(
        (1..=3)
            .map(|sequence| envelope(&tenant, sequence))
            .collect(),
    );
    let mut hints = bus
        .subscribe(&tenant, "conversation.activity", 0, 4, 3, &replay)
        .await
        .expect("hint subscription");
    assert_eq!(
        hints.recv_hint().await,
        Some(SubscriptionEvent::Envelope(envelope(&tenant, 3)))
    );
    assert!(hints.recv_hint().now_or_never().is_none());
    let canonical = replay
        .load_after(&tenant, "conversation.activity", 0, 4)
        .await
        .expect("canonical replay");
    assert_eq!(
        canonical
            .iter()
            .map(FanoutEnvelope::outbox_sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    bus.receive_remote(&tenant, &envelope(&tenant, 4).encode().expect("encode"))
        .expect("new hint");
    bus.shutdown().expect("shutdown");
    assert_eq!(hints.recv_hint().await, Some(SubscriptionEvent::Shutdown));
    assert_eq!(hints.recv_hint().await, None);
}

#[tokio::test]
async fn overflowed_hint_receiver_closes_without_changing_replay() {
    let tenant = tenant();
    let bus = SubscriptionBus::new(tenant.community_id(), 16).expect("bus");
    let replay = TestReplayStore::default();
    replay.replace(vec![envelope(&tenant, 1)]);
    let mut hints = bus
        .subscribe(&tenant, "conversation.activity", 0, 1, 1, &replay)
        .await
        .expect("full initial hint queue");
    bus.receive_remote(&tenant, &envelope(&tenant, 2).encode().expect("encode"))
        .expect("overflow hint");
    assert_eq!(bus.subscription_count(), Ok(0));
    assert_eq!(
        hints.recv_hint().await,
        Some(SubscriptionEvent::Envelope(envelope(&tenant, 1)))
    );
    assert_eq!(hints.recv_hint().await, None);
    replay.replace(vec![envelope(&tenant, 1), envelope(&tenant, 2)]);
    let canonical = replay
        .load_after(&tenant, "conversation.activity", 0, 2)
        .await
        .expect("fallback replay");
    assert_eq!(
        canonical
            .iter()
            .map(FanoutEnvelope::outbox_sequence)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}
