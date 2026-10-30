use std::{
    collections::BTreeMap,
    future::Future,
    net::TcpListener,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use async_tungstenite::{
    WebSocketStream,
    tokio::{ConnectStream, connect_async},
    tungstenite::Message as WebSocketMessage,
};
use credentials_provider::CredentialsProvider;
use futures::StreamExt as _;
use gpui::{AsyncApp, TestAppContext};
use nostr::PublicKey as NostrPublicKey;
use nostr_compat::{
    CanonicalEvent, PublicKey,
    pairing::{
        NIP_AB_SESSION_MILLIS, PairingAbortReason, PairingEncryptedFrame, PairingError,
        PairingFrameId, PairingPayload, PairingPayloadType, PairingQr, PairingRelayUrl,
        PairingSession, PairingSessionState,
    },
};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;
use zed_credentials_provider::pairing::{
    PairedIdentityImportRequest, PairingCredentialError, PairingImportDisposition,
    import_paired_identity,
};
use zeroize::Zeroizing;

const PAIRING_EVENT_KIND: u16 = 24_134;
const CLIENT_FIXTURES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/clients/manifest.json"
));
const ZED_MANIFEST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../zed/Cargo.toml"));

type PairingSocket = WebSocketStream<ConnectStream>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PairingClient {
    Desktop,
    Mobile,
    Cli,
}

impl PairingClient {
    const ALL: [Self; 3] = [Self::Desktop, Self::Mobile, Self::Cli];

    const fn label(self) -> &'static str {
        match self {
            Self::Desktop => "zed-desktop",
            Self::Mobile => "buzz-mobile",
            Self::Cli => "buzz-cli",
        }
    }
}

#[derive(Deserialize)]
struct FrozenClientManifest {
    clients: BTreeMap<String, FrozenClient>,
    contracts: Vec<FrozenClientContract>,
}

#[derive(Deserialize)]
struct FrozenClient {
    version: String,
}

#[derive(Deserialize)]
struct FrozenClientContract {
    id: String,
    input: Value,
    expected_output: Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InteropObservation {
    sender: PairingClient,
    receiver: PairingClient,
    sender_version: String,
    receiver_version: String,
    sas: String,
    imported_public_key: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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

struct TestRelay {
    websocket_url: String,
    task: tokio::task::JoinHandle<()>,
}

impl TestRelay {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind pairing relay");
        listener
            .set_nonblocking(true)
            .expect("nonblocking pairing relay listener");
        let address = listener.local_addr().expect("pairing relay address");
        let server = axum::Server::from_tcp(listener)
            .expect("pairing relay server")
            .serve(pair_relay::router().into_make_service());
        let task = tokio::spawn(async move {
            server.await.expect("pairing relay run");
        });
        Self {
            websocket_url: format!("ws://{address}/pair"),
            task,
        }
    }
}

impl Drop for TestRelay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn current_second_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_millis();
    u64::try_from(millis).expect("current time fits u64") / 1_000 * 1_000
}

fn frame_id(value: u8) -> PairingFrameId {
    PairingFrameId::new([value; 32]).expect("nonzero frame ID")
}

fn public_key(secret: [u8; 32]) -> [u8; 32] {
    let secret = SecretKey::from_slice(&secret).expect("fixture secret");
    let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
    XOnlyPublicKey::from_keypair(&keypair).0.serialize()
}

fn nsec_payload(secret: [u8; 32]) -> PairingPayload {
    let encoded =
        bech32::encode::<bech32::Bech32>(bech32::Hrp::parse("nsec").expect("nsec HRP"), &secret)
            .expect("fixture nsec");
    PairingPayload::new(PairingPayloadType::Nsec, Zeroizing::new(encoded))
        .expect("valid pairing payload")
}

fn frozen_versions() -> BTreeMap<PairingClient, String> {
    let manifest: FrozenClientManifest =
        serde_json::from_str(CLIENT_FIXTURES).expect("frozen client manifest");
    let mobile_contract = manifest
        .contracts
        .iter()
        .find(|contract| contract.id == "CLIENT-MOBILE-009")
        .expect("frozen mobile NIP-AB contract");
    assert_eq!(mobile_contract.input["protocol"], "nostrpair");
    assert_eq!(mobile_contract.input["uri_version"], "omitted");
    assert_eq!(mobile_contract.expected_output["selected_version"], 1);
    assert_eq!(mobile_contract.expected_output["accepted"], true);

    let zed: toml::Value = ZED_MANIFEST.parse().expect("Zed package manifest");
    BTreeMap::from([
        (
            PairingClient::Desktop,
            zed["package"]["version"]
                .as_str()
                .expect("Zed package version")
                .to_owned(),
        ),
        (
            PairingClient::Mobile,
            manifest.clients["buzz-mobile"].version.clone(),
        ),
        (
            PairingClient::Cli,
            manifest.clients["buzz-cli"].version.clone(),
        ),
    ])
}

fn supported_matrix() -> Vec<(PairingClient, PairingClient)> {
    PairingClient::ALL
        .into_iter()
        .flat_map(|sender| {
            PairingClient::ALL
                .into_iter()
                .filter(move |receiver| *receiver != sender)
                .map(move |receiver| (sender, receiver))
        })
        .collect()
}

async fn connect_and_subscribe(
    websocket_url: &str,
    subscription_id: &str,
    public_key: NostrPublicKey,
) -> PairingSocket {
    let (mut socket, _) = connect_async(websocket_url)
        .await
        .expect("connect pairing relay");
    socket
        .send(WebSocketMessage::Text(
            json!([
                "REQ",
                subscription_id,
                {"kinds": [PAIRING_EVENT_KIND], "#p": [public_key.to_hex()]}
            ])
            .to_string()
            .into(),
        ))
        .await
        .expect("send pairing subscription");
    let eose = next_json(&mut socket).await;
    assert_eq!(eose, json!(["EOSE", subscription_id]));
    socket
}

async fn next_json(socket: &mut PairingSocket) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("pairing relay response timeout")
        .expect("pairing relay closed")
        .expect("pairing relay message");
    let WebSocketMessage::Text(text) = message else {
        panic!("expected pairing relay text frame")
    };
    serde_json::from_str(text.as_str()).expect("pairing relay JSON")
}

fn signed_event(frame: &PairingEncryptedFrame, sender_secret: [u8; 32]) -> WireSignedEvent {
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(frame.sender_public_key().to_bytes()),
        frame.created_at_millis() / 1_000,
        PAIRING_EVENT_KIND,
        vec![vec!["p".to_owned(), frame.recipient_public_key().to_hex()]],
        frame.ciphertext().to_owned(),
    );
    let event_id = event.event_id().expect("pairing event ID");
    let secret = SecretKey::from_slice(&sender_secret).expect("sender secret");
    let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
    assert_eq!(
        XOnlyPublicKey::from_keypair(&keypair).0.serialize(),
        frame.sender_public_key().to_bytes(),
        "event signer must own the frame sender key"
    );
    let signature = Secp256k1::new()
        .sign_schnorr_no_aux_rand(&Message::from_digest(*event_id.as_bytes()), &keypair);
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

fn transported_frame(event: &WireSignedEvent) -> PairingEncryptedFrame {
    let frame_id_bytes: [u8; 32] = hex::decode(&event.id)
        .expect("event ID hex")
        .try_into()
        .expect("event ID size");
    let sender = NostrPublicKey::from_hex(&event.pubkey).expect("sender public key");
    let recipient = NostrPublicKey::from_hex(&event.tags[0][1]).expect("recipient public key");
    PairingEncryptedFrame::from_transport(
        PairingFrameId::new(frame_id_bytes).expect("event frame ID"),
        sender,
        recipient,
        event.created_at * 1_000,
        event.content.clone(),
    )
    .expect("relay-admitted pairing frame")
}

async fn publish(
    sender: &mut PairingSocket,
    receiver: &mut PairingSocket,
    event: &WireSignedEvent,
) -> PairingEncryptedFrame {
    sender
        .send(WebSocketMessage::Text(
            json!(["EVENT", event]).to_string().into(),
        ))
        .await
        .expect("publish pairing event");
    let acknowledgement = next_json(sender).await;
    assert_eq!(acknowledgement[0], "OK");
    assert_eq!(acknowledgement[1], event.id);
    assert_eq!(acknowledgement[2], true);
    let delivery = next_json(receiver).await;
    assert_eq!(delivery[0], "EVENT");
    let delivered: WireSignedEvent =
        serde_json::from_value(delivery[2].clone()).expect("delivered pairing event");
    assert_eq!(&delivered, event);
    transported_frame(&delivered)
}

async fn run_success_case(
    sender: PairingClient,
    receiver: PairingClient,
    case_index: u8,
    versions: &BTreeMap<PairingClient, String>,
) -> InteropObservation {
    let relay = TestRelay::start();
    let now_millis = current_second_millis();
    let source_secret = [case_index.saturating_mul(2).saturating_add(1); 32];
    let target_secret = [case_index.saturating_mul(2).saturating_add(2); 32];
    let session_secret = [case_index.saturating_add(64); 32];
    let identity_secret = [case_index.saturating_add(96); 32];
    let relay_url = PairingRelayUrl::parse(relay.websocket_url.clone()).expect("relay URL");
    let (mut source, qr) =
        PairingSession::new_source(source_secret, session_secret, vec![relay_url], now_millis)
            .expect("source session");
    let encoded_qr = qr.encode().expect("QR encoding");
    let parsed_qr = PairingQr::parse(&encoded_qr).expect("frozen client QR parsing");
    assert_eq!(parsed_qr.version(), 1);
    let mut target =
        PairingSession::new_target(&parsed_qr, target_secret, now_millis).expect("target session");
    let mut source_socket =
        connect_and_subscribe(&relay.websocket_url, "source", source.local_public_key()).await;
    let mut target_socket =
        connect_and_subscribe(&relay.websocket_url, "target", target.local_public_key()).await;

    let offer = target.offer(frame_id(1), now_millis).expect("target offer");
    let offer = publish(
        &mut target_socket,
        &mut source_socket,
        &signed_event(&offer, target_secret),
    )
    .await;
    let source_sas = source
        .receive_offer(&offer, now_millis)
        .expect("source receives offer");
    assert_eq!(target.sas().as_deref(), Some(source_sas.as_str()));

    let confirmation = source
        .confirm_source_sas(frame_id(2), now_millis)
        .expect("source confirms SAS");
    let confirmation = publish(
        &mut source_socket,
        &mut target_socket,
        &signed_event(&confirmation, source_secret),
    )
    .await;
    target
        .receive_sas_confirm(&confirmation, now_millis)
        .expect("target verifies SAS transcript");
    target
        .confirm_target_sas(now_millis)
        .expect("target confirms SAS");

    let payload = nsec_payload(identity_secret);
    let encrypted_payload = source
        .send_payload(frame_id(3), &payload, now_millis)
        .expect("source sends identity");
    let encrypted_payload = publish(
        &mut source_socket,
        &mut target_socket,
        &signed_event(&encrypted_payload, source_secret),
    )
    .await;
    let received = target
        .receive_payload(&encrypted_payload, now_millis)
        .expect("target receives identity");
    assert_eq!(received.payload_type(), PairingPayloadType::Nsec);
    assert_eq!(received.secret(), payload.secret());

    let completion = target
        .complete_target(frame_id(4), now_millis)
        .expect("target completes import");
    let completion = publish(
        &mut target_socket,
        &mut source_socket,
        &signed_event(&completion, target_secret),
    )
    .await;
    source
        .receive_complete(&completion, now_millis)
        .expect("source receives verified completion");
    assert_eq!(source.state(), PairingSessionState::Completed);
    assert_eq!(target.state(), PairingSessionState::Completed);

    InteropObservation {
        sender,
        receiver,
        sender_version: versions[&sender].clone(),
        receiver_version: versions[&receiver].clone(),
        sas: source_sas,
        imported_public_key: public_key(identity_secret),
    }
}

#[tokio::test]
async fn desktop_mobile_and_cli_pairing_matrix_completes_verified_transfer() {
    let versions = frozen_versions();
    let matrix = supported_matrix();
    assert_eq!(matrix.len(), 6);

    let mut observations = Vec::new();
    for (index, (sender, receiver)) in matrix.iter().copied().enumerate() {
        observations.push(
            run_success_case(
                sender,
                receiver,
                u8::try_from(index + 1).expect("bounded case index"),
                &versions,
            )
            .await,
        );
    }

    assert_eq!(
        observations
            .iter()
            .map(|observation| (observation.sender, observation.receiver))
            .collect::<Vec<_>>(),
        matrix
    );
    assert!(observations.iter().all(|observation| {
        !observation.sender.label().is_empty()
            && !observation.receiver.label().is_empty()
            && !observation.sender_version.is_empty()
            && !observation.receiver_version.is_empty()
            && observation.sas.len() == 6
            && observation.sas.bytes().all(|byte| byte.is_ascii_digit())
            && observation.imported_public_key != [0; 32]
    }));
}

#[tokio::test]
async fn pairing_relay_rejects_replay_and_delivers_cancel_once() {
    let relay = TestRelay::start();
    let now_millis = current_second_millis();
    let source_secret = [17; 32];
    let target_secret = [18; 32];
    let (mut source, qr) = PairingSession::new_source(
        source_secret,
        [19; 32],
        vec![PairingRelayUrl::parse(relay.websocket_url.clone()).expect("relay")],
        now_millis,
    )
    .expect("source");
    let mut target = PairingSession::new_target(&qr, target_secret, now_millis).expect("target");
    let mut source_socket =
        connect_and_subscribe(&relay.websocket_url, "source", source.local_public_key()).await;
    let mut target_socket =
        connect_and_subscribe(&relay.websocket_url, "target", target.local_public_key()).await;

    let offer = target.offer(frame_id(10), now_millis).expect("offer");
    let wire_offer = signed_event(&offer, target_secret);
    let delivered = publish(&mut target_socket, &mut source_socket, &wire_offer).await;
    source
        .receive_offer(&delivered, now_millis)
        .expect("first offer");

    target_socket
        .send(WebSocketMessage::Text(
            json!(["EVENT", &wire_offer]).to_string().into(),
        ))
        .await
        .expect("replay offer");
    let replay = next_json(&mut target_socket).await;
    assert_eq!(replay[0], "OK");
    assert_eq!(replay[1], wire_offer.id);
    assert_eq!(replay[2], false);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), source_socket.next())
            .await
            .is_err(),
        "replayed event must not be delivered"
    );

    let abort = source
        .abort(frame_id(11), PairingAbortReason::UserDenied, now_millis)
        .expect("cancel frame");
    let abort = publish(
        &mut source_socket,
        &mut target_socket,
        &signed_event(&abort, source_secret),
    )
    .await;
    assert_eq!(
        target
            .receive_abort(&abort, now_millis)
            .expect("target receives cancellation"),
        PairingAbortReason::UserDenied
    );
    assert_eq!(source.state(), PairingSessionState::Aborted);
    assert_eq!(target.state(), PairingSessionState::Aborted);
    assert_eq!(
        target.receive_abort(&abort, now_millis),
        Err(PairingError::Replay)
    );
}

#[test]
fn frozen_clients_fail_closed_on_expiry_and_corrupt_qr() {
    let versions = frozen_versions();
    assert_eq!(versions.len(), PairingClient::ALL.len());
    let started_at = 10_000;
    let (source, qr) = PairingSession::new_source(
        [21; 32],
        [22; 32],
        vec![PairingRelayUrl::parse("wss://pair.example.test").expect("relay")],
        started_at,
    )
    .expect("source");
    let target = PairingSession::new_target(&qr, [23; 32], started_at).expect("target");
    assert!(matches!(
        target.offer(frame_id(20), started_at + NIP_AB_SESSION_MILLIS),
        Err(PairingError::Expired)
    ));
    assert_eq!(source.state(), PairingSessionState::WaitingOffer);
    assert_eq!(target.state(), PairingSessionState::AwaitingSasConfirm);

    let encoded = qr.encode().expect("QR");
    let corrupt = encoded.replacen("secret=", "secret=00", 1);
    assert!(matches!(
        PairingQr::parse(&corrupt),
        Err(PairingError::InvalidQr)
    ));
    let unsupported = encoded.replace("v=1", "v=2");
    assert!(matches!(
        PairingQr::parse(&unsupported),
        Err(PairingError::UnsupportedVersion)
    ));
}

#[derive(Default)]
struct RecoveringCredentialsProvider {
    entries: Mutex<BTreeMap<String, (String, Vec<u8>)>>,
    interrupt_next_write: AtomicBool,
}

impl RecoveringCredentialsProvider {
    fn interrupt_next_write(&self) {
        self.interrupt_next_write.store(true, Ordering::SeqCst);
    }

    fn insert(&self, identifier: &str, username: &str, secret: Vec<u8>) {
        self.entries
            .lock()
            .expect("credential entries")
            .insert(identifier.to_owned(), (username.to_owned(), secret));
    }

    fn entries(&self) -> BTreeMap<String, (String, Vec<u8>)> {
        self.entries.lock().expect("credential entries").clone()
    }
}

impl CredentialsProvider for RecoveringCredentialsProvider {
    fn read_credentials<'a>(
        &'a self,
        url: &'a str,
        _cx: &'a AsyncApp,
    ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
        Box::pin(async move {
            Ok(self
                .entries
                .lock()
                .map_err(|_| anyhow!("credential store unavailable"))?
                .get(url)
                .cloned())
        })
    }

    fn write_credentials<'a>(
        &'a self,
        url: &'a str,
        username: &'a str,
        password: &'a [u8],
        _cx: &'a AsyncApp,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
        Box::pin(async move {
            self.entries
                .lock()
                .map_err(|_| anyhow!("credential store unavailable"))?
                .insert(url.to_owned(), (username.to_owned(), password.to_vec()));
            if self.interrupt_next_write.swap(false, Ordering::SeqCst) {
                return Err(anyhow!("write receipt interrupted"));
            }
            Ok(())
        })
    }

    fn delete_credentials<'a>(
        &'a self,
        url: &'a str,
        _cx: &'a AsyncApp,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
        Box::pin(async move {
            self.entries
                .lock()
                .map_err(|_| anyhow!("credential store unavailable"))?
                .remove(url);
            Ok(())
        })
    }
}

fn import_request() -> PairedIdentityImportRequest {
    PairedIdentityImportRequest {
        community_id: collaboration_domain::CommunityId::from_uuid(Uuid::from_u128(1)),
        service_account_id: collaboration_domain::ServiceAccountId::new(2),
        profile_id: collaboration_domain::ProfileId::from_uuid(Uuid::from_u128(3)),
    }
}

#[gpui::test]
async fn interrupted_verified_import_cleans_up_and_recovers_exactly(cx: &mut TestAppContext) {
    const PRIOR_IDENTIFIER: &str = "zed-nostr://credential/v1/prior";
    let provider = Arc::new(RecoveringCredentialsProvider::default());
    provider.insert(PRIOR_IDENTIFIER, "prior", vec![44; 32]);
    provider.interrupt_next_write();
    let payload = Arc::new(nsec_payload([31; 32]));

    let first = cx
        .spawn({
            let provider = provider.clone();
            let payload = payload.clone();
            move |cx| async move {
                import_paired_identity(provider.as_ref(), &import_request(), payload.as_ref(), &cx)
                    .await
            }
        })
        .await;
    assert_eq!(
        first,
        Err(PairingCredentialError::ProtectedStorageUnavailable)
    );
    assert_eq!(
        provider.entries(),
        BTreeMap::from([(
            PRIOR_IDENTIFIER.to_owned(),
            ("prior".to_owned(), vec![44; 32])
        )])
    );

    let recovered = cx
        .spawn({
            let provider = provider.clone();
            let payload = payload.clone();
            move |cx| async move {
                import_paired_identity(provider.as_ref(), &import_request(), payload.as_ref(), &cx)
                    .await
            }
        })
        .await
        .expect("verified import retry");
    assert_eq!(recovered.disposition, PairingImportDisposition::Imported);
    assert_eq!(recovered.public_key.as_bytes(), &public_key([31; 32]));
    let stored = provider
        .entries()
        .remove(&recovered.credential_identifier)
        .expect("canonical imported credential");
    assert_eq!(stored.0, hex::encode(public_key([31; 32])));
    assert_eq!(stored.1, vec![31; 32]);
    assert_eq!(payload.payload_type(), PairingPayloadType::Nsec);
}
