use std::{
    collections::BTreeMap,
    error::Error,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::IntoResponse,
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures::{SinkExt as _, StreamExt as _};
use nostr::PublicKey as NostrPublicKey;
use nostr_compat::{
    CanonicalEvent, EventId, PublicKey,
    pairing::{
        MAX_NIP_AB_CIPHERTEXT_BYTES, NIP_AB_SESSION_MILLIS, PairingEncryptedFrame, PairingFrameId,
    },
    verification::{EventSignature, SignedEvent, TimestampPolicy, verify_signed_event},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

const PAIRING_EVENT_KIND: u16 = 24_134;
const MAX_CONNECTIONS: usize = 128;
const MAX_SUBSCRIPTION_ID_BYTES: usize = 64;
const MAX_CONNECTION_FRAMES: u16 = 8;
const MAX_CONNECTION_MESSAGES: u16 = 32;
const MAX_RECIPIENT_FRAMES: u16 = 12;
const MAX_REPLAY_IDS: usize = 1_024;
const OUTBOUND_QUEUE_CAPACITY: usize = 8;
const MAX_WEBSOCKET_MESSAGE_BYTES: usize = 96 * 1_024;

#[derive(Clone)]
struct AppState {
    relay: Arc<Mutex<PairRelay>>,
    next_connection_id: Arc<AtomicU64>,
}

struct PairRelay {
    connections: BTreeMap<u64, ConnectionState>,
    replay_ids: BTreeMap<PairingFrameId, u64>,
    recipient_deliveries: BTreeMap<PublicKey, RecipientDeliveryState>,
}

struct ConnectionState {
    expires_at_millis: u64,
    attempted_frames: u16,
    handled_messages: u16,
    subscription: Option<SubscriptionState>,
    sender: mpsc::Sender<String>,
}

struct SubscriptionState {
    subscription_id: String,
    recipient: PublicKey,
}

struct RecipientDeliveryState {
    delivered_frames: u16,
    expires_at_millis: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RelaySnapshot {
    connections: usize,
    subscriptions: usize,
    queued_frames: usize,
    replay_ids: usize,
    durable_records: usize,
    private_key_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RelayError {
    Capacity,
    DuplicateConnection,
    UnknownConnection,
    Expired,
    InvalidSubscription,
    AmbiguousRecipient,
    InvalidEvent,
    Replay,
    FrameBudget,
    QueueFull,
}

impl PairRelay {
    fn new() -> Self {
        Self {
            connections: BTreeMap::new(),
            replay_ids: BTreeMap::new(),
            recipient_deliveries: BTreeMap::new(),
        }
    }

    fn connect(
        &mut self,
        connection_id: u64,
        sender: mpsc::Sender<String>,
        now_millis: u64,
    ) -> Result<(), RelayError> {
        self.reap(now_millis);
        if connection_id == 0 || now_millis == 0 {
            return Err(RelayError::UnknownConnection);
        }
        if self.connections.contains_key(&connection_id) {
            return Err(RelayError::DuplicateConnection);
        }
        if self.connections.len() >= MAX_CONNECTIONS {
            return Err(RelayError::Capacity);
        }
        let expires_at_millis = now_millis
            .checked_add(NIP_AB_SESSION_MILLIS)
            .ok_or(RelayError::Expired)?;
        self.connections.insert(
            connection_id,
            ConnectionState {
                expires_at_millis,
                attempted_frames: 0,
                handled_messages: 0,
                subscription: None,
                sender,
            },
        );
        Ok(())
    }

    fn subscribe(
        &mut self,
        connection_id: u64,
        subscription_id: String,
        recipient: PublicKey,
        now_millis: u64,
    ) -> Result<(), RelayError> {
        self.reap(now_millis);
        if subscription_id.is_empty()
            || subscription_id.len() > MAX_SUBSCRIPTION_ID_BYTES
            || subscription_id.chars().any(char::is_control)
            || self.connections.iter().any(|(existing_id, connection)| {
                *existing_id != connection_id
                    && connection
                        .subscription
                        .as_ref()
                        .is_some_and(|subscription| subscription.recipient == recipient)
            })
        {
            return Err(RelayError::InvalidSubscription);
        }
        let connection = self
            .connections
            .get_mut(&connection_id)
            .ok_or(RelayError::UnknownConnection)?;
        connection.subscription = Some(SubscriptionState {
            subscription_id,
            recipient,
        });
        Ok(())
    }

    fn publish(
        &mut self,
        connection_id: u64,
        event: WireSignedEvent,
        now_millis: u64,
    ) -> Result<(), RelayError> {
        self.reap(now_millis);
        let connection = self
            .connections
            .get_mut(&connection_id)
            .ok_or(RelayError::UnknownConnection)?;
        if connection.attempted_frames >= MAX_CONNECTION_FRAMES {
            return Err(RelayError::FrameBudget);
        }
        connection.attempted_frames += 1;
        let (recipient, frame_id) = validate_pairing_event(&event, now_millis)?;
        if self.replay_ids.contains_key(&frame_id) {
            return Err(RelayError::Replay);
        }
        if self.replay_ids.len() >= MAX_REPLAY_IDS {
            return Err(RelayError::Capacity);
        }
        let matches: Vec<_> = self
            .connections
            .values()
            .filter_map(|connection| {
                connection
                    .subscription
                    .as_ref()
                    .filter(|subscription| subscription.recipient == recipient)
                    .map(|subscription| {
                        (
                            subscription.subscription_id.clone(),
                            connection.sender.clone(),
                            connection.expires_at_millis,
                        )
                    })
            })
            .collect();
        let [(subscription_id, sender, recipient_expiry)] = matches.as_slice() else {
            return Err(RelayError::AmbiguousRecipient);
        };
        let recipient_state = self
            .recipient_deliveries
            .get(&recipient)
            .map(|state| state.delivered_frames)
            .unwrap_or(0);
        if recipient_state >= MAX_RECIPIENT_FRAMES {
            return Err(RelayError::FrameBudget);
        }
        let outgoing = serde_json::to_string(&("EVENT", subscription_id, &event))
            .map_err(|_| RelayError::InvalidEvent)?;
        sender
            .try_send(outgoing)
            .map_err(|_| RelayError::QueueFull)?;
        let replay_expiry = now_millis
            .checked_add(NIP_AB_SESSION_MILLIS)
            .ok_or(RelayError::Expired)?;
        self.replay_ids.insert(frame_id, replay_expiry);
        self.recipient_deliveries.insert(
            recipient,
            RecipientDeliveryState {
                delivered_frames: recipient_state + 1,
                expires_at_millis: (*recipient_expiry).min(replay_expiry),
            },
        );
        Ok(())
    }

    fn admit_message(&mut self, connection_id: u64, now_millis: u64) -> Result<(), RelayError> {
        self.reap(now_millis);
        let connection = self
            .connections
            .get_mut(&connection_id)
            .ok_or(RelayError::UnknownConnection)?;
        if connection.handled_messages >= MAX_CONNECTION_MESSAGES {
            return Err(RelayError::FrameBudget);
        }
        connection.handled_messages += 1;
        Ok(())
    }

    fn disconnect(&mut self, connection_id: u64) -> Result<(), RelayError> {
        self.connections
            .remove(&connection_id)
            .ok_or(RelayError::UnknownConnection)?;
        Ok(())
    }

    fn is_connected(&mut self, connection_id: u64, now_millis: u64) -> bool {
        self.reap(now_millis);
        self.connections.contains_key(&connection_id)
    }

    #[cfg(test)]
    fn snapshot(&self) -> RelaySnapshot {
        RelaySnapshot {
            connections: self.connections.len(),
            subscriptions: self
                .connections
                .values()
                .filter(|connection| connection.subscription.is_some())
                .count(),
            queued_frames: self
                .connections
                .values()
                .map(|connection| connection.sender.max_capacity() - connection.sender.capacity())
                .sum(),
            replay_ids: self.replay_ids.len(),
            durable_records: 0,
            private_key_bytes: 0,
        }
    }

    fn reap(&mut self, now_millis: u64) {
        self.connections
            .retain(|_, connection| connection.expires_at_millis > now_millis);
        self.replay_ids
            .retain(|_, expires_at_millis| *expires_at_millis > now_millis);
        self.recipient_deliveries
            .retain(|_, state| state.expires_at_millis > now_millis);
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WireSignedEvent {
    id: String,
    pubkey: String,
    created_at: u64,
    kind: u16,
    tags: Vec<Vec<String>>,
    content: String,
    sig: String,
}

fn validate_pairing_event(
    wire: &WireSignedEvent,
    now_millis: u64,
) -> Result<(PublicKey, PairingFrameId), RelayError> {
    if wire.kind != PAIRING_EVENT_KIND
        || wire.content.is_empty()
        || wire.content.len() > MAX_NIP_AB_CIPHERTEXT_BYTES
        || wire.tags.len() != 1
        || wire.tags[0].len() != 2
        || wire.tags[0][0] != "p"
    {
        return Err(RelayError::InvalidEvent);
    }
    let decoded = BASE64_STANDARD
        .decode(&wire.content)
        .map_err(|_| RelayError::InvalidEvent)?;
    if decoded.len() < 99 || decoded.first() != Some(&2) {
        return Err(RelayError::InvalidEvent);
    }
    let recipient = PublicKey::from_hex(&wire.tags[0][1]).map_err(|_| RelayError::InvalidEvent)?;
    let event = CanonicalEvent::new(
        PublicKey::from_hex(&wire.pubkey).map_err(|_| RelayError::InvalidEvent)?,
        wire.created_at,
        wire.kind,
        wire.tags.clone(),
        wire.content.clone(),
    );
    let signed_event = SignedEvent {
        claimed_id: EventId::from_hex(&wire.id).map_err(|_| RelayError::InvalidEvent)?,
        event,
        signature: EventSignature::from_hex(&wire.sig).map_err(|_| RelayError::InvalidEvent)?,
    };
    let now_seconds = now_millis / 1_000;
    verify_signed_event(
        &signed_event,
        TimestampPolicy::Bounded {
            now: now_seconds,
            max_past_seconds: NIP_AB_SESSION_MILLIS / 1_000,
            max_future_seconds: 5,
        },
    )
    .map_err(|_| RelayError::InvalidEvent)?;
    let frame_id = PairingFrameId::new(*signed_event.claimed_id.as_bytes())
        .map_err(|_| RelayError::InvalidEvent)?;
    PairingEncryptedFrame::from_transport(
        frame_id,
        NostrPublicKey::from_slice(signed_event.event.public_key.as_bytes())
            .map_err(|_| RelayError::InvalidEvent)?,
        NostrPublicKey::from_slice(recipient.as_bytes()).map_err(|_| RelayError::InvalidEvent)?,
        wire.created_at
            .checked_mul(1_000)
            .ok_or(RelayError::InvalidEvent)?,
        wire.content.clone(),
    )
    .map_err(|_| RelayError::InvalidEvent)?;
    Ok((recipient, frame_id))
}

async fn pair_upgrade(
    State(state): State<AppState>,
    upgrade: WebSocketUpgrade,
) -> impl IntoResponse {
    upgrade
        .max_frame_size(MAX_WEBSOCKET_MESSAGE_BYTES)
        .max_message_size(MAX_WEBSOCKET_MESSAGE_BYTES)
        .on_upgrade(move |socket| serve_connection(socket, state))
}

async fn serve_connection(socket: WebSocket, state: AppState) {
    let connection_id = state.next_connection_id.fetch_add(1, Ordering::Relaxed);
    let (sender, mut receiver) = mpsc::channel(OUTBOUND_QUEUE_CAPACITY);
    let now_millis = current_time_millis();
    let connected = state
        .relay
        .lock()
        .ok()
        .and_then(|mut relay| relay.connect(connection_id, sender, now_millis).ok())
        .is_some();
    if !connected {
        return;
    }
    let (mut websocket_sender, mut websocket_receiver) = socket.split();
    let mut expiry_tick = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        tokio::select! {
            outbound = receiver.recv() => {
                let Some(outbound) = outbound else { break };
                if websocket_sender.send(Message::Text(outbound)).await.is_err() {
                    break;
                }
            }
            inbound = websocket_receiver.next() => {
                let Some(Ok(inbound)) = inbound else { break };
                match inbound {
                    Message::Text(text) => {
                        if let Some(response) = handle_client_message(&state, connection_id, &text) {
                            if websocket_sender.send(Message::Text(response)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        if websocket_sender.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    Message::Binary(_) | Message::Pong(_) => {}
                }
            }
            _ = expiry_tick.tick() => {
                let active = state.relay.lock().ok().is_some_and(|mut relay| {
                    relay.is_connected(connection_id, current_time_millis())
                });
                if !active { break; }
            }
        }
    }
    if let Ok(mut relay) = state.relay.lock() {
        match relay.disconnect(connection_id) {
            Ok(()) | Err(RelayError::UnknownConnection) => {}
            Err(error) => eprintln!("pair relay disconnect failed: {error:?}"),
        }
    }
}

fn handle_client_message(state: &AppState, connection_id: u64, text: &str) -> Option<String> {
    if text.len() > MAX_WEBSOCKET_MESSAGE_BYTES {
        return Some(notice("message too large"));
    }
    let admitted = state.relay.lock().ok().and_then(|mut relay| {
        relay
            .admit_message(connection_id, current_time_millis())
            .ok()
    });
    if admitted.is_none() {
        return Some(notice("connection message budget exhausted"));
    }
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(_) => return Some(notice("invalid message")),
    };
    let Some(parts) = value.as_array() else {
        return Some(notice("invalid message"));
    };
    match parts.first().and_then(Value::as_str) {
        Some("REQ") => {
            let subscription_id = parts.get(1).and_then(Value::as_str).unwrap_or_default();
            let recipient = parts.get(2).and_then(parse_subscription_filter);
            let Some(recipient) = recipient else {
                return Some(closed(subscription_id, "invalid pairing subscription"));
            };
            let result = state.relay.lock().ok().and_then(|mut relay| {
                relay
                    .subscribe(
                        connection_id,
                        subscription_id.to_owned(),
                        recipient,
                        current_time_millis(),
                    )
                    .ok()
            });
            Some(if result.is_some() {
                serde_json::to_string(&("EOSE", subscription_id))
                    .unwrap_or_else(|_| notice("codec failure"))
            } else {
                closed(subscription_id, "pairing subscription rejected")
            })
        }
        Some("EVENT") => {
            let event: WireSignedEvent = match parts
                .get(1)
                .cloned()
                .and_then(|event| serde_json::from_value(event).ok())
            {
                Some(event) => event,
                None => return Some(notice("invalid pairing event")),
            };
            let event_id = event.id.clone();
            let result = state.relay.lock().ok().and_then(|mut relay| {
                relay
                    .publish(connection_id, event, current_time_millis())
                    .ok()
            });
            Some(ok(&event_id, result.is_some()))
        }
        Some("CLOSE") => {
            if let Ok(mut relay) = state.relay.lock()
                && let Some(connection) = relay.connections.get_mut(&connection_id)
            {
                connection.subscription = None;
            }
            None
        }
        _ => Some(notice("unsupported message")),
    }
}

fn parse_subscription_filter(value: &Value) -> Option<PublicKey> {
    let filter = value.as_object()?;
    if filter
        .keys()
        .any(|key| !matches!(key.as_str(), "#p" | "kinds"))
        || filter.get("kinds").is_some_and(|kinds| {
            kinds.as_array().is_none_or(|kinds| {
                kinds.len() != 1 || kinds[0].as_u64() != Some(u64::from(PAIRING_EVENT_KIND))
            })
        })
    {
        return None;
    }
    let recipients = filter.get("#p")?.as_array()?;
    if recipients.len() != 1 {
        return None;
    }
    PublicKey::from_hex(recipients[0].as_str()?).ok()
}

fn ok(event_id: &str, accepted: bool) -> String {
    serde_json::to_string(&(
        "OK",
        event_id,
        accepted,
        if accepted { "" } else { "rejected" },
    ))
    .unwrap_or_else(|_| "[\"NOTICE\",\"codec failure\"]".into())
}

fn closed(subscription_id: &str, reason: &str) -> String {
    serde_json::to_string(&("CLOSED", subscription_id, reason))
        .unwrap_or_else(|_| "[\"NOTICE\",\"codec failure\"]".into())
}

fn notice(reason: &str) -> String {
    serde_json::to_string(&("NOTICE", reason))
        .unwrap_or_else(|_| "[\"NOTICE\",\"codec failure\"]".into())
}

fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(1)
}

pub fn router() -> Router {
    let state = AppState {
        relay: Arc::new(Mutex::new(PairRelay::new())),
        next_connection_id: Arc::new(AtomicU64::new(1)),
    };
    Router::new()
        .route("/pair", get(pair_upgrade))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let address: SocketAddr = std::env::var("PAIR_RELAY_BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:5000".into())
        .parse()?;
    axum::Server::bind(&address)
        .serve(router().into_make_service())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use nostr_compat::pairing::{PairingRelayUrl, PairingSession};
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey as SecpSecretKey};

    use super::*;

    const SOURCE_SECRET: [u8; 32] = [0x11; 32];
    const TARGET_SECRET: [u8; 32] = [0x22; 32];
    const SESSION_SECRET: [u8; 32] = [0x33; 32];

    fn channel() -> (mpsc::Sender<String>, mpsc::Receiver<String>) {
        mpsc::channel(OUTBOUND_QUEUE_CAPACITY)
    }

    fn signed_pairing_event(frame_value: u8, created_at_millis: u64) -> WireSignedEvent {
        let (_source, qr) = PairingSession::new_source(
            SOURCE_SECRET,
            SESSION_SECRET,
            vec![PairingRelayUrl::parse("wss://pair.example.test").expect("relay")],
            created_at_millis,
        )
        .expect("source");
        let target =
            PairingSession::new_target(&qr, TARGET_SECRET, created_at_millis).expect("target");
        let frame = target
            .offer(
                PairingFrameId::new([frame_value; 32]).expect("frame id"),
                created_at_millis,
            )
            .expect("offer");
        let public_key =
            PublicKey::from_hex(&frame.sender_public_key().to_hex()).expect("public key");
        let recipient = frame.recipient_public_key().to_hex();
        let event = CanonicalEvent::new(
            public_key,
            created_at_millis / 1_000,
            PAIRING_EVENT_KIND,
            vec![vec!["p".into(), recipient]],
            frame.ciphertext().to_owned(),
        );
        let event_id = event.event_id().expect("event id");
        let secret = SecpSecretKey::from_slice(&TARGET_SECRET).expect("target secret");
        let signature = Secp256k1::new().sign_schnorr_no_aux_rand(
            &Message::from_digest(*event_id.as_bytes()),
            &Keypair::from_secret_key(&Secp256k1::new(), &secret),
        );
        WireSignedEvent {
            id: event_id.to_hex(),
            pubkey: event.public_key.to_hex(),
            created_at: event.created_at,
            kind: event.kind,
            tags: event.tags,
            content: event.content,
            sig: signature.to_string(),
        }
    }

    #[test]
    fn expiry_removes_connections_subscriptions_queues_and_replay_state() {
        let mut relay = PairRelay::new();
        let (recipient_sender, mut recipient_receiver) = channel();
        let (publisher_sender, _publisher_receiver) = channel();
        relay
            .connect(1, recipient_sender, 10_000)
            .expect("recipient");
        relay
            .connect(2, publisher_sender, 10_000)
            .expect("publisher");
        let event = signed_pairing_event(1, 10_000);
        let recipient = PublicKey::from_hex(&event.tags[0][1]).expect("recipient key");
        relay
            .subscribe(1, "pair".into(), recipient, 10_000)
            .expect("subscribe");
        relay.publish(2, event, 10_000).expect("publish");
        assert!(recipient_receiver.try_recv().is_ok());
        relay.reap(130_000);
        assert_eq!(
            relay.snapshot(),
            RelaySnapshot {
                connections: 0,
                subscriptions: 0,
                queued_frames: 0,
                replay_ids: 0,
                durable_records: 0,
                private_key_bytes: 0,
            }
        );
    }

    #[test]
    fn connection_capacity_fails_closed() {
        let mut relay = PairRelay::new();
        let mut receivers = Vec::new();
        for connection_id in 1..=MAX_CONNECTIONS as u64 {
            let (sender, receiver) = channel();
            relay
                .connect(connection_id, sender, 10_000)
                .expect("within capacity");
            receivers.push(receiver);
        }
        let (overflow_sender, _overflow_receiver) = channel();
        assert_eq!(
            relay.connect(MAX_CONNECTIONS as u64 + 1, overflow_sender, 10_000),
            Err(RelayError::Capacity)
        );
        assert_eq!(relay.snapshot().connections, MAX_CONNECTIONS);
        assert_eq!(receivers.len(), MAX_CONNECTIONS);
    }

    #[test]
    fn replay_is_rejected_without_a_second_delivery() {
        let mut relay = PairRelay::new();
        let (recipient_sender, mut recipient_receiver) = channel();
        let (publisher_sender, _publisher_receiver) = channel();
        relay
            .connect(1, recipient_sender, 10_000)
            .expect("recipient");
        relay
            .connect(2, publisher_sender, 10_000)
            .expect("publisher");
        let event = signed_pairing_event(1, 10_000);
        let recipient = PublicKey::from_hex(&event.tags[0][1]).expect("recipient key");
        relay
            .subscribe(1, "pair".into(), recipient, 10_000)
            .expect("subscribe");
        relay
            .publish(2, event.clone(), 10_000)
            .expect("first publish");
        assert_eq!(relay.publish(2, event, 10_001), Err(RelayError::Replay));
        assert!(recipient_receiver.try_recv().is_ok());
        assert!(recipient_receiver.try_recv().is_err());
    }

    #[test]
    fn full_recipient_queue_rejects_without_reserving_replay_id() {
        let mut relay = PairRelay::new();
        let (recipient_sender, _recipient_receiver) = channel();
        relay
            .connect(1, recipient_sender, 10_000)
            .expect("recipient");
        let first_event = signed_pairing_event(1, 10_000);
        let recipient = PublicKey::from_hex(&first_event.tags[0][1]).expect("recipient key");
        relay
            .subscribe(1, "pair".into(), recipient, 10_000)
            .expect("subscribe");
        for offset in 0..OUTBOUND_QUEUE_CAPACITY {
            let connection_id = u64::try_from(offset + 2).expect("bounded connection id");
            let (publisher_sender, _publisher_receiver) = channel();
            relay
                .connect(connection_id, publisher_sender, 10_000)
                .expect("publisher");
            relay
                .publish(
                    connection_id,
                    signed_pairing_event(u8::try_from(offset + 1).expect("bounded frame"), 10_000),
                    10_000,
                )
                .expect("queue within capacity");
        }
        let overflow_connection =
            u64::try_from(OUTBOUND_QUEUE_CAPACITY + 2).expect("bounded overflow connection");
        let (publisher_sender, _publisher_receiver) = channel();
        relay
            .connect(overflow_connection, publisher_sender, 10_000)
            .expect("overflow publisher");
        assert_eq!(
            relay.publish(
                overflow_connection,
                signed_pairing_event(99, 10_000),
                10_000,
            ),
            Err(RelayError::QueueFull)
        );
        assert_eq!(relay.snapshot().replay_ids, OUTBOUND_QUEUE_CAPACITY);
    }

    #[test]
    fn disconnect_removes_subscription_and_queued_ciphertext() {
        let mut relay = PairRelay::new();
        let (recipient_sender, recipient_receiver) = channel();
        relay
            .connect(1, recipient_sender, 10_000)
            .expect("recipient");
        let event = signed_pairing_event(1, 10_000);
        let recipient = PublicKey::from_hex(&event.tags[0][1]).expect("recipient key");
        relay
            .subscribe(1, "pair".into(), recipient, 10_000)
            .expect("subscribe");
        relay.disconnect(1).expect("disconnect");
        drop(recipient_receiver);
        assert_eq!(relay.snapshot().connections, 0);
        assert_eq!(relay.snapshot().subscriptions, 0);
        let (publisher_sender, _publisher_receiver) = channel();
        relay
            .connect(2, publisher_sender, 10_000)
            .expect("publisher");
        assert_eq!(
            relay.publish(2, event, 10_000),
            Err(RelayError::AmbiguousRecipient)
        );
    }

    #[test]
    fn relay_state_contains_no_durable_or_private_key_material() {
        let mut relay = PairRelay::new();
        let (sender, _receiver) = channel();
        relay.connect(1, sender, 10_000).expect("connection");
        let snapshot = relay.snapshot();
        assert_eq!(snapshot.durable_records, 0);
        assert_eq!(snapshot.private_key_bytes, 0);
        assert_eq!(
            std::mem::size_of::<PairRelay>(),
            std::mem::size_of_val(&relay)
        );
    }
}
