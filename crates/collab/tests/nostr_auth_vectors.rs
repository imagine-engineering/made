use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use collab::{
    nostr::auth::{
        NIP42_AUTH_TIMEOUT_SECONDS, NostrAuthChallenge, NostrAuthProtocolDisposition,
        NostrAuthReplayStore, NostrAuthenticationError, NostrAuthenticationInfrastructureError,
        NostrAuthenticationState, NostrConnectionAuthenticator, NostrPrincipalResolutionError,
        NostrPrincipalResolver, ReplayClaim, VerifiedNip42Identity,
    },
    nostr::{
        http::{Nip98ReplayClaim, Nip98ReplayStore},
        replay::RedisAuthenticationReplayStore,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AuthenticatedPrincipal, AuthorizationScope, CommunityId, NostrAuthenticationMethod,
    NostrPublicKey, PrincipalId, PrincipalScopes, TrustedTenantRoute,
};
use nostr_compat::{
    CanonicalEvent, EventSignature, PublicKey, SignedEvent, generated_kinds::KIND_AUTH,
};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    net::TcpListener,
};
use uuid::Uuid;

const RELAY_URL: &str = "wss://relay.example.com";
const NOW: u64 = 1_900_000_000;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> collaboration_domain::TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "nostr-auth-vectors")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn challenge() -> NostrAuthChallenge {
    NostrAuthChallenge::parse("ab".repeat(32)).expect("challenge")
}

fn signed_auth_event(
    secret_byte: u8,
    challenge: &NostrAuthChallenge,
    relay_url: &str,
    created_at: u64,
) -> SignedEvent {
    let secret = SecretKey::from_slice(&[secret_byte; 32]).expect("fixture secret");
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        created_at,
        KIND_AUTH as u16,
        vec![
            vec!["relay".to_owned(), relay_url.to_owned()],
            vec!["challenge".to_owned(), challenge.as_str().to_owned()],
        ],
        String::new(),
    );
    let claimed_id = event.event_id().expect("event id");
    let signature =
        secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

#[derive(Clone, Default)]
struct ReplayStore {
    state: Arc<ReplayState>,
}

#[derive(Default)]
struct ReplayState {
    claimed: Mutex<BTreeSet<(CommunityId, nostr_compat::EventId)>>,
    attempts: AtomicUsize,
    last_expiration: std::sync::atomic::AtomicU64,
}

#[async_trait]
impl NostrAuthReplayStore for ReplayStore {
    async fn claim(
        &self,
        community_id: CommunityId,
        event_id: nostr_compat::EventId,
        expires_at_seconds: u64,
    ) -> Result<ReplayClaim, NostrAuthenticationInfrastructureError> {
        self.state.attempts.fetch_add(1, Ordering::SeqCst);
        self.state
            .last_expiration
            .store(expires_at_seconds, Ordering::SeqCst);
        let inserted = self
            .state
            .claimed
            .lock()
            .expect("replay store lock")
            .insert((community_id, event_id));
        Ok(if inserted {
            ReplayClaim::Claimed
        } else {
            ReplayClaim::Replay
        })
    }
}

#[derive(Clone, Copy)]
enum ResolutionMode {
    Allow,
    WrongTenant,
    Revoked,
    Pending,
}

#[derive(Clone, Copy)]
struct PrincipalResolver {
    mode: ResolutionMode,
}

#[async_trait]
impl NostrPrincipalResolver for PrincipalResolver {
    async fn resolve(
        &self,
        tenant: &collaboration_domain::TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        if matches!(self.mode, ResolutionMode::Pending) {
            return std::future::pending().await;
        }
        if matches!(self.mode, ResolutionMode::Revoked) {
            return Err(NostrPrincipalResolutionError::Denied);
        }
        let community_id = if matches!(self.mode, ResolutionMode::WrongTenant) {
            community(2)
        } else {
            tenant.community_id()
        };
        Ok(AuthenticatedPrincipal::nostr_identity(
            principal(3),
            community_id,
            NostrPublicKey::from_bytes(*identity.public_key().as_bytes()),
            NostrAuthenticationMethod::Nip42,
            PrincipalScopes::new([
                AuthorizationScope::new("events:read").expect("read scope"),
                AuthorizationScope::new("events:write").expect("write scope"),
            ])
            .expect("scopes"),
        ))
    }
}

fn authenticator(
    replay_store: ReplayStore,
    mode: ResolutionMode,
) -> NostrConnectionAuthenticator<ReplayStore, PrincipalResolver> {
    NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW,
        replay_store,
        PrincipalResolver { mode },
    )
    .expect("authenticator")
}

struct PendingReplayStore;

#[async_trait]
impl NostrAuthReplayStore for PendingReplayStore {
    async fn claim(
        &self,
        _community_id: CommunityId,
        _event_id: nostr_compat::EventId,
        _expires_at_seconds: u64,
    ) -> Result<ReplayClaim, NostrAuthenticationInfrastructureError> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn nostr_auth_vectors_timeout_pending_replay_and_resolution() {
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    let mut replay_pending = NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW - 4,
        PendingReplayStore,
        PrincipalResolver {
            mode: ResolutionMode::Allow,
        },
    )
    .expect("authenticator");
    let mut resolution_pending = NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW - 4,
        ReplayStore::default(),
        PrincipalResolver {
            mode: ResolutionMode::Pending,
        },
    )
    .expect("authenticator");
    let replay_result = tokio::time::timeout(
        Duration::from_secs(2),
        replay_pending.authenticate(&event, NOW),
    )
    .await
    .expect("pending replay must respect the connection deadline");
    assert_eq!(
        replay_result.expect_err("timeout"),
        NostrAuthenticationError::TimedOut
    );
    assert_eq!(*replay_pending.state(), NostrAuthenticationState::TimedOut);
    let resolution_result = tokio::time::timeout(
        Duration::from_secs(2),
        resolution_pending.authenticate(&event, NOW),
    )
    .await
    .expect("pending resolution must respect the connection deadline");
    assert_eq!(
        resolution_result.expect_err("timeout"),
        NostrAuthenticationError::TimedOut
    );
    assert_eq!(
        *resolution_pending.state(),
        NostrAuthenticationState::TimedOut
    );
}

#[tokio::test]
async fn nostr_auth_vectors_accept_success_and_reject_reauthentication() {
    let replay_store = ReplayStore::default();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    let mut authenticator = authenticator(replay_store.clone(), ResolutionMode::Allow);
    assert_eq!(
        authenticator.challenge_frame().as_deref(),
        Some(r#"["AUTH","abababababababababababababababababababababababababababababababab"]"#)
    );

    let authenticated = authenticator
        .authenticate(&event, NOW)
        .await
        .expect("valid NIP-42 event");
    assert_eq!(authenticated.community_id(), community(1));
    assert_eq!(authenticated.principal_id(), principal(3));
    assert!(matches!(
        authenticator.state(),
        NostrAuthenticationState::Authenticated(_)
    ));
    let error = authenticator
        .authenticate(&event, NOW)
        .await
        .expect_err("reauthentication is rejected");
    assert_eq!(error, NostrAuthenticationError::AlreadyAuthenticated);
    assert_eq!(
        error.protocol_disposition(),
        NostrAuthProtocolDisposition::Reject {
            reason: "auth-required: already authenticated",
            close_connection: false,
        }
    );
    assert_eq!(replay_store.state.attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn nostr_auth_vectors_enforce_connection_timeout_before_crypto_or_replay() {
    let replay_store = ReplayStore::default();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    let mut authenticator = authenticator(replay_store.clone(), ResolutionMode::Allow);

    let error = authenticator
        .authenticate(&event, NOW + NIP42_AUTH_TIMEOUT_SECONDS)
        .await
        .expect_err("deadline is terminal");
    assert_eq!(error, NostrAuthenticationError::TimedOut);
    assert_eq!(
        error.protocol_disposition(),
        NostrAuthProtocolDisposition::CloseWithoutResponse
    );
    assert_eq!(authenticator.state(), &NostrAuthenticationState::TimedOut);
    assert_eq!(replay_store.state.attempts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn nostr_auth_vectors_reject_a_tenant_scoped_event_replay() {
    let replay_store = ReplayStore::default();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    authenticator(replay_store.clone(), ResolutionMode::Allow)
        .authenticate(&event, NOW)
        .await
        .expect("first authentication");
    let mut replayed = authenticator(replay_store, ResolutionMode::Allow);

    assert_eq!(
        replayed.authenticate(&event, NOW).await,
        Err(NostrAuthenticationError::Replay)
    );
    assert_eq!(replayed.state(), &NostrAuthenticationState::Failed);
}

#[tokio::test]
async fn nostr_auth_vectors_reject_wrong_tenant_identity_resolution() {
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    let mut authenticator = authenticator(ReplayStore::default(), ResolutionMode::WrongTenant);

    assert_eq!(
        authenticator.authenticate(&event, NOW).await,
        Err(NostrAuthenticationError::IdentityDenied)
    );
    assert_eq!(authenticator.state(), &NostrAuthenticationState::Failed);
}

#[tokio::test]
async fn nostr_auth_vectors_reject_revoked_keys() {
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    let mut authenticator = authenticator(ReplayStore::default(), ResolutionMode::Revoked);

    assert_eq!(
        authenticator.authenticate(&event, NOW).await,
        Err(NostrAuthenticationError::IdentityDenied)
    );
    assert_eq!(authenticator.state(), &NostrAuthenticationState::Failed);
}

#[tokio::test]
async fn nostr_auth_vectors_reject_wrong_challenge_and_relay_without_replay_claims() {
    for event in [
        signed_auth_event(
            1,
            &NostrAuthChallenge::parse("cd".repeat(32)).expect("other challenge"),
            RELAY_URL,
            NOW,
        ),
        signed_auth_event(1, &challenge(), "wss://other.example.com", NOW),
    ] {
        let replay_store = ReplayStore::default();
        let mut authenticator = authenticator(replay_store.clone(), ResolutionMode::Allow);
        assert!(matches!(
            authenticator.authenticate(&event, NOW).await,
            Err(NostrAuthenticationError::ChallengeMismatch
                | NostrAuthenticationError::RelayMismatch)
        ));
        assert_eq!(replay_store.state.attempts.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn nostr_auth_vectors_fence_future_events_through_their_validity_window() {
    for created_at in [NOW - 60, NOW, NOW + 60] {
        let replay_store = ReplayStore::default();
        let mut authenticator = authenticator(replay_store.clone(), ResolutionMode::Allow);
        let event = signed_auth_event(1, &challenge(), RELAY_URL, created_at);
        authenticator
            .authenticate(&event, NOW)
            .await
            .expect("fresh event");
        assert_eq!(
            replay_store.state.last_expiration.load(Ordering::SeqCst),
            (NOW + 60).max(created_at + 60)
        );
    }
}

fn current_seconds() -> anyhow::Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

async fn read_redis_command(
    reader: &mut BufReader<tokio::net::TcpStream>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let count = line
        .trim()
        .strip_prefix('*')
        .ok_or_else(|| anyhow::anyhow!("command array"))?
        .parse::<usize>()?;
    anyhow::ensure!(count <= 8, "command bound");
    let mut command = Vec::new();
    for _ in 0..count {
        line.clear();
        reader.read_line(&mut line).await?;
        let length = line
            .trim()
            .strip_prefix('$')
            .ok_or_else(|| anyhow::anyhow!("command bulk"))?
            .parse::<usize>()?;
        anyhow::ensure!(length <= 1024, "argument bound");
        let mut value = vec![0; length];
        reader.read_exact(&mut value).await?;
        let mut terminator = [0; 2];
        reader.read_exact(&mut terminator).await?;
        anyhow::ensure!(terminator == *b"\r\n", "argument terminator");
        command.push(value);
    }
    Ok(command)
}

fn replay_key(community: CommunityId, event: nostr_compat::EventId) -> String {
    format!(
        "zed.collaboration.auth.v1:{}:{}",
        community.as_uuid(),
        hex::encode(event.as_bytes())
    )
}

#[tokio::test]
async fn nostr_replay_shared_backend_uses_atomic_tenant_bound_claims() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("redis://{}", listener.local_addr()?);
    let store = RedisAuthenticationReplayStore::new(&endpoint)?;
    let community = CommunityId::new();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW).claimed_id;
    let key = replay_key(community, event);
    let server = tokio::spawn(async move {
        for response in [b"+OK\r\n".as_slice(), b"$-1\r\n".as_slice()] {
            let (stream, _) = listener.accept().await?;
            let mut reader = BufReader::new(stream);
            let command = read_redis_command(&mut reader).await?;
            assert_eq!(command.len(), 6);
            assert_eq!(
                &command[..5],
                &[
                    b"SET".to_vec(),
                    key.as_bytes().to_vec(),
                    b"1".to_vec(),
                    b"NX".to_vec(),
                    b"EX".to_vec()
                ]
            );
            let lifetime = std::str::from_utf8(&command[5])?.parse::<u64>()?;
            assert!((59..=61).contains(&lifetime));
            reader.get_mut().write_all(response).await?;
        }
        anyhow::Ok(())
    });
    let expires = current_seconds()? + 60;
    assert_eq!(
        NostrAuthReplayStore::claim(&store, community, event, expires)
            .await
            .expect("claim fence"),
        ReplayClaim::Claimed
    );
    assert_eq!(
        Nip98ReplayStore::claim(&store, &tenant(community), event, expires)
            .await
            .expect("shared fence"),
        Nip98ReplayClaim::Replay
    );
    tokio::time::timeout(Duration::from_secs(2), server).await???;
    if let Ok(endpoint) = std::env::var("COLLAB_TEST_REDIS_URL") {
        verify_disposable_replay_store(&endpoint).await?;
        println!(
            "disposable Redis authentication replay: atomic race, tenant/protocol/database isolation and TTL verified"
        );
    }
    Ok(())
}

#[tokio::test]
async fn nostr_replay_rejects_expired_or_unbounded_fences_before_io() -> anyhow::Result<()> {
    let store = RedisAuthenticationReplayStore::new("redis://127.0.0.1:1")?;
    let community = CommunityId::new();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW).claimed_id;
    let now = current_seconds()?;
    for expiration in [0, now - 1, now, now + 121, u64::MAX] {
        assert_eq!(
            NostrAuthReplayStore::claim(&store, community, event, expiration).await,
            Err(NostrAuthenticationInfrastructureError::Unavailable)
        );
    }
    Ok(())
}

#[tokio::test]
async fn nostr_replay_selects_configured_database_and_denies_non_ok_authentication()
-> anyhow::Result<()> {
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW).claimed_id;
    for authentication in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let endpoint = if authentication {
            format!("redis://account:fixture@{address}/2")
        } else {
            format!("redis://{address}/2")
        };
        let store = RedisAuthenticationReplayStore::new(&endpoint)?;
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await?;
            let mut reader = BufReader::new(stream);
            if authentication {
                assert_eq!(
                    read_redis_command(&mut reader).await?,
                    vec![b"AUTH".to_vec(), b"account".to_vec(), b"fixture".to_vec()]
                );
                reader.get_mut().write_all(b"+DENIED\r\n").await?;
                let mut byte = [0];
                assert_eq!(
                    reader.read(&mut byte).await?,
                    0,
                    "denied auth must close before SELECT or SET"
                );
            } else {
                assert_eq!(
                    read_redis_command(&mut reader).await?,
                    vec![b"SELECT".to_vec(), b"2".to_vec()]
                );
                reader.get_mut().write_all(b"+OK\r\n").await?;
                assert_eq!(
                    read_redis_command(&mut reader)
                        .await?
                        .first()
                        .map(Vec::as_slice),
                    Some(b"SET".as_slice())
                );
                reader.get_mut().write_all(b"+OK\r\n").await?;
            }
            anyhow::Ok(())
        });
        let result =
            NostrAuthReplayStore::claim(&store, CommunityId::new(), event, current_seconds()? + 60)
                .await;
        assert_eq!(
            result,
            if authentication {
                Err(NostrAuthenticationInfrastructureError::Unavailable)
            } else {
                Ok(ReplayClaim::Claimed)
            }
        );
        tokio::time::timeout(Duration::from_secs(2), server).await???;
    }
    Ok(())
}

#[tokio::test]
async fn nostr_replay_rejects_oversized_unterminated_reply() -> anyhow::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let store =
        RedisAuthenticationReplayStore::new(&format!("redis://{}", listener.local_addr()?))?;
    let (release, released) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await?;
        let mut reader = BufReader::new(stream);
        read_redis_command(&mut reader).await?;
        reader
            .get_mut()
            .write_all(&vec![b'+'; 64 * 1024 + 1])
            .await?;
        released.await?;
        anyhow::Ok(())
    });
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW).claimed_id;
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        NostrAuthReplayStore::claim(&store, CommunityId::new(), event, current_seconds()? + 60),
    )
    .await;
    release
        .send(())
        .map_err(|_| anyhow::anyhow!("response server stopped"))?;
    server.await??;
    assert_eq!(
        result?,
        Err(NostrAuthenticationInfrastructureError::Unavailable)
    );
    Ok(())
}

fn redis_request(parts: &[&[u8]]) -> Vec<u8> {
    let mut request = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        request.extend(format!("${}\r\n", part.len()).bytes());
        request.extend_from_slice(part);
        request.extend_from_slice(b"\r\n");
    }
    request
}

async fn observe_replay_ttl(endpoint: &str, key: &str) -> anyhow::Result<i64> {
    let endpoint = url::Url::parse(endpoint)?;
    anyhow::ensure!(
        endpoint.host_str() == Some("127.0.0.1") && endpoint.password().is_none(),
        "live observer requires the disposable loopback Redis service"
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        let stream =
            tokio::net::TcpStream::connect(("127.0.0.1", endpoint.port().unwrap_or(6379))).await?;
        let mut reader = BufReader::new(stream);
        let database = endpoint.path().trim_matches('/');
        let mut line = String::new();
        if !database.is_empty() && database != "0" {
            reader
                .get_mut()
                .write_all(&redis_request(&[b"SELECT", database.as_bytes()]))
                .await?;
            reader.read_line(&mut line).await?;
            anyhow::ensure!(line == "+OK\r\n", "observer database selection");
            line.clear();
        }
        reader
            .get_mut()
            .write_all(&redis_request(&[b"TTL", key.as_bytes()]))
            .await?;
        reader.read_line(&mut line).await?;
        let ttl = line
            .trim()
            .strip_prefix(':')
            .ok_or_else(|| anyhow::anyhow!("TTL integer"))?
            .parse::<i64>()?;
        anyhow::Ok(ttl)
    })
    .await?
}

async fn verify_disposable_replay_store(endpoint: &str) -> anyhow::Result<()> {
    let first = RedisAuthenticationReplayStore::new(endpoint)?;
    let second = RedisAuthenticationReplayStore::new(endpoint)?;
    let community = CommunityId::new();
    let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW).claimed_id;
    let expires = current_seconds()? + 60;
    let mut claims = Vec::new();
    for index in 0..32 {
        let store = if index % 2 == 0 {
            first.clone()
        } else {
            second.clone()
        };
        claims.push(tokio::spawn(async move {
            NostrAuthReplayStore::claim(&store, community, event, expires).await
        }));
    }
    let claims = tokio::time::timeout(
        Duration::from_secs(3),
        futures::future::try_join_all(claims),
    )
    .await??;
    assert_eq!(
        claims
            .iter()
            .filter(|claim| **claim == Ok(ReplayClaim::Claimed))
            .count(),
        1
    );
    assert_eq!(
        claims
            .iter()
            .filter(|claim| **claim == Ok(ReplayClaim::Replay))
            .count(),
        31
    );
    assert_eq!(
        Nip98ReplayStore::claim(&second, &tenant(community), event, expires).await,
        Ok(Nip98ReplayClaim::Replay)
    );
    assert_eq!(
        Nip98ReplayStore::claim(&second, &tenant(CommunityId::new()), event, expires).await,
        Ok(Nip98ReplayClaim::Claimed)
    );
    let mut selected_endpoint = url::Url::parse(endpoint)?;
    selected_endpoint.set_path("/2");
    let selected_store = RedisAuthenticationReplayStore::new(selected_endpoint.as_str())?;
    assert_eq!(
        NostrAuthReplayStore::claim(&selected_store, community, event, expires).await,
        Ok(ReplayClaim::Claimed)
    );
    let key = replay_key(community, event);
    for endpoint in [endpoint, selected_endpoint.as_str()] {
        let remaining = expires.saturating_sub(current_seconds()?) as i64;
        let observed = observe_replay_ttl(endpoint, &key).await?;
        assert!(
            observed >= remaining - 1 && observed <= remaining + 2,
            "actual TTL {observed} must cover requested remaining lifetime {remaining}"
        );
        println!("Observed live replay TTL: {observed} seconds (remaining validity: {remaining})");
    }
    Ok(())
}

fn receipt_auth_event(agent: u8, tags: Vec<Vec<String>>, created_at: u64) -> SignedEvent {
    let mut signed = signed_auth_event(agent, &challenge(), RELAY_URL, created_at);
    signed.event.tags.extend(tags);
    signed.claimed_id = signed.event.event_id().expect("receipt event ID");
    let keypair = Keypair::from_secret_key(
        &Secp256k1::new(),
        &SecretKey::from_slice(&[agent; 32]).expect("fixture agent key"),
    );
    signed.signature = EventSignature::from_hex(
        &Secp256k1::new()
            .sign_schnorr_no_aux_rand(
                &Message::from_digest(*signed.claimed_id.as_bytes()),
                &keypair,
            )
            .to_string(),
    )
    .expect("actual receipt outer signature");
    signed
}

fn receipt_owner_tag(agent: u8, owner: u8, conditions: &str) -> Vec<String> {
    use sha2::{Digest as _, Sha256};
    let agent = signed_auth_event(agent, &challenge(), RELAY_URL, NOW)
        .event
        .public_key;
    let keypair = Keypair::from_secret_key(
        &Secp256k1::new(),
        &SecretKey::from_slice(&[owner; 32]).expect("fixture owner key"),
    );
    let (owner, _) = XOnlyPublicKey::from_keypair(&keypair);
    let preimage = format!("nostr:agent-auth:{}:{conditions}", agent.to_hex());
    let mut digest = [0; 32];
    digest.copy_from_slice(&Sha256::digest(preimage.as_bytes()));
    let signature =
        Secp256k1::new().sign_schnorr_no_aux_rand(&Message::from_digest(digest), &keypair);
    vec![
        "auth".into(),
        hex::encode(owner.serialize()),
        conditions.to_owned(),
        signature.to_string(),
    ]
}

#[derive(Clone, Default)]
struct ReceiptResolver {
    received: Arc<Mutex<Vec<VerifiedNip42Identity>>>,
}

#[async_trait]
impl NostrPrincipalResolver for ReceiptResolver {
    async fn resolve(
        &self,
        tenant: &collaboration_domain::TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        use collaboration_domain::{
            AgentProfile, AggregateVersion, IdentityProfile, NostrEventId,
            OwnerAttestationEvidence, ProfileId, ProfileKind, ProfileRecordFields,
        };
        let identity = identity.for_tenant(tenant)?;
        self.received
            .lock()
            .expect("receipt lock")
            .push(identity.clone());
        let Some(attestation) = identity.owner_attestation() else {
            return Ok(AuthenticatedPrincipal::nostr_identity(
                principal(3),
                tenant.community_id(),
                identity.public_key(),
                NostrAuthenticationMethod::Nip42,
                PrincipalScopes::default(),
            ));
        };
        let profile = IdentityProfile::new(ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(4)),
            community_id: tenant.community_id(),
            author_public_key: identity.public_key(),
            kind: ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(attestation.owner_public_key()),
                owner_attestation: Some(OwnerAttestationEvidence {
                    owner_public_key: attestation.owner_public_key(),
                    agent_public_key: identity.public_key(),
                    proof_event_id: NostrEventId::from_bytes(*identity.event_id().as_bytes()),
                    exact_conditions: attestation.exact_conditions().to_owned(),
                    verified_at: identity.verified_at_seconds(),
                }),
            }),
            metadata: None,
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("fixture profile from actual receipt; not persisted registration or membership");
        AuthenticatedPrincipal::owner_attested_agent(
            principal(3),
            tenant.community_id(),
            &profile,
            NostrAuthenticationMethod::Nip42,
            PrincipalScopes::default(),
        )
        .map_err(|_| NostrPrincipalResolutionError::Denied)
    }
}

fn receipt_authenticator(
    replay: ReplayStore,
    resolver: ReceiptResolver,
) -> NostrConnectionAuthenticator<ReplayStore, ReceiptResolver> {
    NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW,
        replay,
        resolver,
    )
    .expect("receipt authenticator")
}

#[tokio::test]
async fn nostr_auth_receipt_retains_exact_signed_owner_sources_and_tenant() {
    let conditions = "created_at>1899999999&created_at<1900000001";
    let tag = receipt_owner_tag(3, 7, conditions);
    let signed = receipt_auth_event(3, vec![tag.clone()], NOW);
    let replay = ReplayStore::default();
    let resolver = ReceiptResolver::default();
    let mut authenticator = receipt_authenticator(replay.clone(), resolver.clone());
    let principal = authenticator
        .authenticate(&signed, NOW)
        .await
        .expect("actual signed AUTH and owner proof");
    assert!(
        principal.scopes().is_empty(),
        "fixture must not manufacture membership or scopes"
    );
    let receipts = resolver.received.lock().expect("received lock");
    let receipt = receipts
        .first()
        .expect("verified receipt must reach existing resolver");
    assert_eq!(receipt.community_id(), community(1));
    assert_eq!(receipt.verified_at_seconds(), NOW);
    assert_eq!(receipt.signed_authentication_event(), &signed);
    assert_eq!(receipt.event_id(), signed.claimed_id);
    let persistence =
        collab::db::collaboration::persistence_policy::EventPersistencePolicy::evaluate(
            signed.event.kind,
            collab::db::collaboration::persistence_policy::PrivacyAdmission::author(),
        )
        .expect("canonical AUTH policy");
    assert_eq!(
        persistence.durability(),
        collab::db::collaboration::persistence_policy::EventDurability::TransientOnly,
        "receipt retention must not redefine AUTH as durable event storage"
    );
    let owner = receipt.owner_attestation().expect("actual owner proof");
    assert_eq!(owner.exact_conditions(), conditions);
    assert_eq!(
        owner.attestation().to_tag(),
        tag,
        "receipt must retain the exact owner signature and signed conditions"
    );
    owner
        .attestation()
        .verify_for_membership(&signed.event.public_key, signed.event.created_at)
        .expect("retained original must independently verify");
    assert!(receipt.for_tenant(&tenant(community(1))).is_ok());
    assert_eq!(
        receipt.for_tenant(&tenant(community(2))),
        Err(NostrPrincipalResolutionError::Denied)
    );
    let debug = format!("{receipt:?}");
    assert!(
        !debug.contains(challenge().as_str()),
        "receipt Debug must not disclose challenge"
    );
    assert!(
        !debug.contains(&signed.signature.to_hex()),
        "receipt Debug must not disclose the complete signed AUTH credential"
    );
    assert_eq!(replay.state.attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn nostr_auth_receipt_plain_key_success_does_not_establish_human_or_membership() {
    let resolver = ReceiptResolver::default();
    let mut authenticator = receipt_authenticator(ReplayStore::default(), resolver.clone());
    let signed = signed_auth_event(3, &challenge(), RELAY_URL, NOW);
    let principal = authenticator
        .authenticate(&signed, NOW)
        .await
        .expect("plain signed key authentication");
    assert!(matches!(
        principal.kind(),
        collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { .. }
    ));
    assert!(principal.scopes().is_empty());
    let receipts = resolver.received.lock().expect("received lock");
    let receipt = receipts.first().expect("verified receipt");
    assert!(
        receipt.owner_attestation().is_none(),
        "plain key proof must not fabricate owner evidence or Human classification"
    );
    assert_eq!(receipt.signed_authentication_event(), &signed);
}

#[tokio::test]
async fn nostr_auth_receipt_invalid_optional_owner_evidence_preserves_exact_key_authentication() {
    let mut invalid_signature = receipt_owner_tag(3, 7, "");
    if let Some(signature) = invalid_signature.get_mut(3) {
        *signature = "00".repeat(64);
    }
    let duplicate = receipt_owner_tag(3, 7, "");
    for tags in [
        vec![invalid_signature],
        vec![vec!["auth".into()]],
        vec![receipt_owner_tag(3, 3, "")],
        vec![receipt_owner_tag(3, 7, "created_at<1900000000")],
        vec![duplicate.clone(), duplicate],
    ] {
        let resolver = ReceiptResolver::default();
        let replay = ReplayStore::default();
        let mut authenticator = receipt_authenticator(replay.clone(), resolver.clone());
        let signed = receipt_auth_event(3, tags, NOW);
        let authenticated = authenticator
            .authenticate(&signed, NOW)
            .await
            .expect("invalid optional owner evidence rejected genuine key authentication");
        assert!(matches!(authenticated.kind(),
            collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { public_key, .. }
            if *public_key == NostrPublicKey::from_bytes(*signed.event.public_key.as_bytes())));
        assert!(authenticated.scopes().is_empty());
        let receipts = resolver.received.lock().expect("received lock");
        assert_eq!(receipts.len(), 1);
        let receipt = receipts.first().expect("actual outer signature receipt");
        assert!(receipt.owner_attestation().is_none());
        assert_eq!(receipt.signed_authentication_event(), &signed);
        assert_eq!(replay.state.attempts.load(Ordering::SeqCst), 1);
    }
    let resolver = ReceiptResolver::default();
    let replay = ReplayStore::default();
    let mut authenticator = receipt_authenticator(replay.clone(), resolver.clone());
    let mut tampered = receipt_auth_event(3, vec![receipt_owner_tag(3, 7, "")], NOW);
    tampered.event.content = "changed after signature".into();
    assert_eq!(
        authenticator.authenticate(&tampered, NOW).await,
        Err(NostrAuthenticationError::InvalidEvent)
    );
    assert!(resolver.received.lock().expect("received lock").is_empty());
    assert_eq!(replay.state.attempts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn nostr_auth_invalid_optional_owner_evidence_cannot_bypass_principal_admission() {
    for mode in [ResolutionMode::WrongTenant, ResolutionMode::Revoked] {
        let mut connection = authenticator(ReplayStore::default(), mode);
        let signed = receipt_auth_event(3, vec![vec!["auth".into()]], NOW);
        assert_eq!(
            connection.authenticate(&signed, NOW).await,
            Err(NostrAuthenticationError::IdentityDenied)
        );
        assert!(matches!(
            connection.state(),
            NostrAuthenticationState::Failed
        ));
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let mut connection = NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW,
        ReplayStore::default(),
        StructuralDenialResolver {
            error: NostrPrincipalResolutionError::Banned,
            calls: calls.clone(),
        },
    )
    .expect("authenticator");
    let signed = receipt_auth_event(3, vec![vec!["auth".into()]], NOW);
    assert_eq!(
        connection.authenticate(&signed, NOW).await,
        Err(NostrAuthenticationError::Banned)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn nostr_auth_receipt_preserves_replay_and_freshness_gates() {
    let resolver = ReceiptResolver::default();
    let replay = ReplayStore::default();
    let signed = receipt_auth_event(3, vec![receipt_owner_tag(3, 7, "")], NOW);
    receipt_authenticator(replay.clone(), resolver.clone())
        .authenticate(&signed, NOW)
        .await
        .expect("initial owner AUTH");
    let mut repeated = receipt_authenticator(replay, resolver.clone());
    assert_eq!(
        repeated.authenticate(&signed, NOW).await,
        Err(NostrAuthenticationError::Replay)
    );
    assert_eq!(resolver.received.lock().expect("received lock").len(), 1);
    for created_at in [NOW - 61, NOW + 61] {
        let resolver = ReceiptResolver::default();
        let replay = ReplayStore::default();
        let mut authenticator = receipt_authenticator(replay.clone(), resolver.clone());
        let signed = receipt_auth_event(3, vec![receipt_owner_tag(3, 7, "")], created_at);
        assert_eq!(
            authenticator.authenticate(&signed, NOW).await,
            Err(NostrAuthenticationError::InvalidEvent)
        );
        assert!(resolver.received.lock().expect("received lock").is_empty());
        assert_eq!(replay.state.attempts.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn nostr_auth_vectors_valid_owner_proof_can_retain_exact_key_without_delegation() {
    let signed = receipt_auth_event(3, vec![receipt_owner_tag(3, 7, "kind=0")], NOW);
    let mut connection = authenticator(ReplayStore::default(), ResolutionMode::Allow);
    let principal = connection
        .authenticate(&signed, NOW)
        .await
        .expect("valid key AUTH survives declined owner admission");
    assert!(
        matches!(principal.kind(), collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { public_key, .. }
        if *public_key == NostrPublicKey::from_bytes(*signed.event.public_key.as_bytes()))
    );
    assert_eq!(principal.community_id(), community(1));
}

#[derive(Clone)]
struct StructuralDenialResolver {
    error: NostrPrincipalResolutionError,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl NostrPrincipalResolver for StructuralDenialResolver {
    async fn resolve(
        &self,
        tenant: &collaboration_domain::TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        identity.for_tenant(tenant)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(self.error)
    }
}

#[tokio::test]
async fn nostr_auth_structural_denial_has_exact_distinct_protocol_disposition() {
    for (resolution_error, authentication_error, reason) in [
        (
            NostrPrincipalResolutionError::Banned,
            NostrAuthenticationError::Banned,
            "blocked: you are banned from this community",
        ),
        (
            NostrPrincipalResolutionError::Unavailable,
            NostrAuthenticationError::InfrastructureUnavailable,
            "error: internal",
        ),
        (
            NostrPrincipalResolutionError::UnrepresentedSubject,
            NostrAuthenticationError::UnrepresentedSubject,
            "error: restriction target is not represented",
        ),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut authenticator = NostrConnectionAuthenticator::new(
            tenant(community(1)),
            RELAY_URL,
            challenge(),
            NOW,
            ReplayStore::default(),
            StructuralDenialResolver {
                error: resolution_error,
                calls: calls.clone(),
            },
        )
        .expect("authenticator");
        let event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
        assert_eq!(
            authenticator.authenticate(&event, NOW).await,
            Err(authentication_error)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            authentication_error.protocol_disposition(),
            NostrAuthProtocolDisposition::Reject {
                reason,
                close_connection: true
            }
        );
        assert!(matches!(
            authenticator.state(),
            NostrAuthenticationState::Failed
        ));
        assert_eq!(
            authenticator.authenticate(&event, NOW).await,
            Err(NostrAuthenticationError::AuthenticationAlreadyFailed)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let mut authenticator = NostrConnectionAuthenticator::new(
        tenant(community(1)),
        RELAY_URL,
        challenge(),
        NOW,
        ReplayStore::default(),
        StructuralDenialResolver {
            error: NostrPrincipalResolutionError::Banned,
            calls: calls.clone(),
        },
    )
    .expect("authenticator");
    let mut event = signed_auth_event(1, &challenge(), RELAY_URL, NOW);
    event.event.content = "tampered".to_owned();
    assert_eq!(
        authenticator.authenticate(&event, NOW).await,
        Err(NostrAuthenticationError::InvalidEvent)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
