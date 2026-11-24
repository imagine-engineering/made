use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use axum::{
    body::{Body, HttpBody as _},
    http::{Request, StatusCode},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use collab::nostr::http::{
    Nip05Identity, Nip98AuthenticationError, Nip98Authenticator, Nip98PrincipalResolver,
    Nip98ReplayClaim, Nip98ReplayStore, NostrDeploymentHostResolver, NostrHttpConfiguration,
    NostrHttpDirectory, NostrHttpDirectoryError, NostrHttpHostResolver, NostrPublicHttpState,
    NostrTenantRelayMetadata, VerifiedNip98Identity, router,
};
use collaboration_domain::{
    AuthenticatedPrincipal, AuthorizationScope, CommunityId, NostrAuthenticationMethod,
    NostrPublicKey, PrincipalId, PrincipalScopes, TenantContext, TrustedTenantRoute,
};
use nostr_compat::{CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use tower::ServiceExt as _;
use uuid::Uuid;

const NOW: u64 = 1_900_000_000;
const RELAY_HOST: &str = "a.example.test";

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId, host: &str) -> TenantContext {
    TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(community_id, host).expect("direct host")),
        &[],
    )
    .expect("tenant")
}

#[derive(Default)]
struct Directory {
    icons: BTreeMap<CommunityId, String>,
    identities: BTreeMap<(CommunityId, String), PublicKey>,
    lookups: Mutex<Vec<(CommunityId, String)>>,
    fail_metadata: AtomicBool,
    fail_identity: AtomicBool,
}

#[async_trait]
impl NostrHttpDirectory for Directory {
    async fn relay_metadata(
        &self,
        tenant: &TenantContext,
    ) -> Result<NostrTenantRelayMetadata, NostrHttpDirectoryError> {
        if self.fail_metadata.load(Ordering::SeqCst) {
            return Err(NostrHttpDirectoryError::Unavailable);
        }
        NostrTenantRelayMetadata::new(self.icons.get(&tenant.community_id()).cloned())
            .map_err(|_| NostrHttpDirectoryError::Unavailable)
    }

    async fn resolve_nip05(
        &self,
        tenant: &TenantContext,
        canonical_name: &str,
    ) -> Result<Option<Nip05Identity>, NostrHttpDirectoryError> {
        if self.fail_identity.load(Ordering::SeqCst) {
            return Err(NostrHttpDirectoryError::Unavailable);
        }
        self.lookups
            .lock()
            .expect("directory lookup lock")
            .push((tenant.community_id(), canonical_name.to_owned()));
        Ok(self
            .identities
            .get(&(tenant.community_id(), canonical_name.to_owned()))
            .copied()
            .map(Nip05Identity::new))
    }
}

fn public_state(directory: Arc<Directory>) -> Arc<NostrPublicHttpState> {
    let resolver = NostrDeploymentHostResolver::new([
        (
            community(1),
            url::Url::parse("https://a.example.test").expect("public URL"),
        ),
        (
            community(2),
            url::Url::parse("https://b.example.test").expect("public URL"),
        ),
    ])
    .expect("trusted deployment routes");
    Arc::new(NostrPublicHttpState::new(
        NostrHttpConfiguration::buzz_compatible(
            "0.1.0",
            Some(PublicKey::from_bytes([3; 32])),
            true,
        )
        .expect("HTTP configuration"),
        Arc::new(resolver),
        directory,
    ))
}

#[tokio::test]
async fn nostr_http_deployment_routes_preserve_exact_trusted_authorities() -> anyhow::Result<()> {
    let resolver = NostrDeploymentHostResolver::new([
        (community(1), url::Url::parse("https://a.example.test")?),
        (community(2), url::Url::parse("wss://b.example.test:8443")?),
        (community(3), url::Url::parse("http://[::1]:8080")?),
    ])?;
    for (host, expected) in [
        ("a.example.test", community(1)),
        ("a.example.test:443", community(1)),
        ("b.example.test:8443", community(2)),
        ("[::1]:8080", community(3)),
    ] {
        let admitted = resolver
            .resolve_host(host)
            .await
            .map_err(|error| anyhow::anyhow!("trusted route failed: {error:?}"))?;
        assert_eq!(admitted.community_id(), expected);
        assert_eq!(
            admitted.route_source(),
            collaboration_domain::TrustedTenantRouteSource::DirectHost
        );
    }
    for host in [
        "missing.example.test",
        "a.example.test:8443",
        "b.example.test",
        "[::1]",
        "a.example.test@attacker.example",
        "A.example.test",
    ] {
        assert!(matches!(
            resolver.resolve_host(host).await,
            Err(NostrHttpDirectoryError::Denied)
        ));
    }
    Ok(())
}

#[test]
fn nostr_http_deployment_routes_reject_ambiguous_or_unsafe_configuration() -> anyhow::Result<()> {
    assert!(NostrDeploymentHostResolver::new([]).is_err());
    for endpoint in [
        "http://a.example.test",
        "ws://a.example.test",
        "ftp://a.example.test",
        "https://name:secret@a.example.test",
        "https://a.example.test/info",
        "https://a.example.test?community=other",
        "https://a.example.test#other",
    ] {
        assert!(
            NostrDeploymentHostResolver::new([(community(1), url::Url::parse(endpoint)?)]).is_err()
        );
    }
    assert!(
        NostrDeploymentHostResolver::new([
            (community(1), url::Url::parse("https://a.example.test")?),
            (community(2), url::Url::parse("wss://a.example.test:443")?),
        ])
        .is_err()
    );
    let routes = (0..257)
        .map(|index| {
            url::Url::parse(&format!("https://host-{index}.example.test"))
                .map(|url| (community(1), url))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert!(NostrDeploymentHostResolver::new(routes).is_err());
    Ok(())
}

async fn response_json(response: axum::response::Response) -> Value {
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    while let Some(chunk) = body.data().await {
        bytes.extend_from_slice(&chunk.expect("response chunk"));
    }
    serde_json::from_slice(&bytes).expect("JSON response")
}

#[tokio::test]
async fn nostr_http_nip11_binds_host_and_redacts_unmapped_metadata() {
    let mut directory = Directory::default();
    directory
        .icons
        .insert(community(1), "https://a.example.test/workspace.png".into());
    let directory = Arc::new(directory);
    let app = router(public_state(Arc::clone(&directory)));

    let mapped = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header("host", RELAY_HOST)
                .header("accept", "application/nostr+json")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("mapped response");
    assert_eq!(mapped.status(), StatusCode::OK);
    let mapped = response_json(mapped).await;
    assert_eq!(mapped["icon"], "https://a.example.test/workspace.png");
    assert_eq!(mapped["limitation"]["max_subscriptions"], 1024);
    assert_eq!(mapped["limitation"]["max_filters"], 10);
    assert_eq!(mapped["limitation"]["max_limit"], 1000);
    assert_eq!(mapped["limitation"]["auth_required"], true);
    assert!(mapped.get("community_id").is_none());
    assert!(mapped.get("members").is_none());

    directory.fail_metadata.store(true, Ordering::SeqCst);
    let unavailable = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/info")
                .header("host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("unavailable metadata response");
    assert!(response_json(unavailable).await.get("icon").is_none());

    let unmapped = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/info")
                .header("host", "unknown.example.test")
                .header("x-forwarded-host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("unmapped response");
    assert_eq!(unmapped.status(), StatusCode::OK);
    let unmapped = response_json(unmapped).await;
    assert!(unmapped.get("icon").is_none());
    assert_eq!(unmapped["name"], "Zed Collaborative Relay");

    let not_negotiated = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header("host", RELAY_HOST)
                .header("accept", "text/html")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("HTML response");
    assert_eq!(not_negotiated.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn nostr_http_nip05_is_tenant_scoped_and_returns_empty_on_miss() {
    let public_key = PublicKey::from_bytes([4; 32]);
    let mut directory = Directory::default();
    directory
        .identities
        .insert((community(1), "alice".into()), public_key);
    let directory = Arc::new(directory);
    let app = router(public_state(Arc::clone(&directory)));

    let found = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/.well-known/nostr.json?name=Alice")
                .header("host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("NIP-05 response");
    assert_eq!(found.status(), StatusCode::OK);
    assert_eq!(
        found
            .headers()
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok()),
        Some("*")
    );
    let found = response_json(found).await;
    assert_eq!(found["names"]["alice"], public_key.to_hex());
    assert_eq!(
        found["relays"][public_key.to_hex()][0],
        "wss://a.example.test"
    );

    directory.fail_identity.store(true, Ordering::SeqCst);
    let unavailable = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/.well-known/nostr.json?name=alice")
                .header("host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("unavailable directory response");
    assert_eq!(
        response_json(unavailable).await,
        json!({"names": {}, "relays": {}})
    );
    directory.fail_identity.store(false, Ordering::SeqCst);

    let foreign = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/.well-known/nostr.json?name=alice")
                .header("host", "b.example.test")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("foreign response");
    let foreign = response_json(foreign).await;
    assert_eq!(foreign, json!({"names": {}, "relays": {}}));

    let malformed = app
        .oneshot(
            Request::builder()
                .uri("/.well-known/nostr.json?name=../admin")
                .header("host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("malformed response");
    assert_eq!(
        response_json(malformed).await,
        json!({"names": {}, "relays": {}})
    );
    assert_eq!(
        directory.lookups.lock().expect("lookup lock").as_slice(),
        &[
            (community(1), "alice".into()),
            (community(2), "alice".into()),
            (community(1), "../admin".into()),
        ]
    );
}

#[tokio::test]
async fn nostr_http_rejects_ambiguous_host_before_identity_lookup() {
    let mut directory = Directory::default();
    directory.identities.insert(
        (community(1), "alice".into()),
        PublicKey::from_bytes([4; 32]),
    );
    let directory = Arc::new(directory);
    let app = router(public_state(Arc::clone(&directory)));
    for hosts in [
        vec![RELAY_HOST, "b.example.test"],
        vec![RELAY_HOST, RELAY_HOST],
        vec!["user@a.example.test"],
        vec!["a.example.test:invalid"],
        vec!["a.example.test:65536"],
        vec!["a.example.test:"],
    ] {
        let mut request = Request::builder().uri("/.well-known/nostr.json?name=alice");
        for host in &hosts {
            request = request.header("host", *host);
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(
            response_json(response).await,
            json!({"names": {}, "relays": {}}),
            "{hosts:?}"
        );
    }
    assert!(directory.lookups.lock().expect("lookup lock").is_empty());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/.well-known/nostr.json?name=alice")
                .header("host", "A.EXAMPLE.TEST")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("valid response");
    assert_eq!(
        response_json(response).await["names"]["alice"],
        PublicKey::from_bytes([4; 32]).to_hex()
    );
    assert_eq!(directory.lookups.lock().expect("lookup lock").len(), 1);
}

fn signed_nip98_event(
    secret_byte: u8,
    url: &str,
    method: &str,
    body: &[u8],
    include_payload: bool,
    created_at: u64,
) -> (SignedEvent, String) {
    let secret = SecretKey::from_slice(&[secret_byte; 32]).expect("fixture secret");
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let mut tags = vec![
        vec!["u".into(), url.into()],
        vec!["method".into(), method.into()],
    ];
    if include_payload {
        tags.push(vec!["payload".into(), hex::encode(Sha256::digest(body))]);
    }
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        created_at,
        27_235,
        tags,
        String::new(),
    );
    let claimed_id = event.event_id().expect("event id");
    let signature =
        secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    let signed_event = SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    };
    let wire = json!({
        "id": signed_event.claimed_id.to_hex(),
        "pubkey": signed_event.event.public_key.to_hex(),
        "created_at": signed_event.event.created_at,
        "kind": signed_event.event.kind,
        "tags": signed_event.event.tags,
        "content": signed_event.event.content,
        "sig": signed_event.signature.to_hex(),
    });
    let authorization = format!(
        "Nostr {}",
        BASE64.encode(serde_json::to_vec(&wire).expect("wire event"))
    );
    (signed_event, authorization)
}

#[derive(Default)]
struct ReplayState {
    claimed: Mutex<BTreeSet<(CommunityId, EventId)>>,
    fail: AtomicUsize,
}

#[derive(Clone, Default)]
struct ReplayStore(Arc<ReplayState>);

#[async_trait]
impl Nip98ReplayStore for ReplayStore {
    async fn claim(
        &self,
        tenant: &TenantContext,
        event_id: EventId,
        _expires_at_seconds: u64,
    ) -> Result<Nip98ReplayClaim, NostrHttpDirectoryError> {
        if self.0.fail.load(Ordering::SeqCst) != 0 {
            return Err(NostrHttpDirectoryError::Unavailable);
        }
        let inserted = self
            .0
            .claimed
            .lock()
            .expect("replay lock")
            .insert((tenant.community_id(), event_id));
        Ok(if inserted {
            Nip98ReplayClaim::Claimed
        } else {
            Nip98ReplayClaim::Replay
        })
    }
}

#[derive(Clone, Copy)]
enum PrincipalMode {
    Allow,
    WrongTenant,
    WrongKey,
    WrongAuthenticationMethod,
    Revoked,
}

struct PrincipalResolver {
    mode: PrincipalMode,
    calls: AtomicUsize,
    receipts: Mutex<Vec<VerifiedNip98Identity>>,
}

#[async_trait]
impl Nip98PrincipalResolver for PrincipalResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip98Identity,
    ) -> Result<AuthenticatedPrincipal, NostrHttpDirectoryError> {
        identity.for_tenant(tenant)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.receipts
            .lock()
            .expect("receipt lock")
            .push(identity.clone());
        if matches!(self.mode, PrincipalMode::Revoked) {
            return Err(NostrHttpDirectoryError::Denied);
        }
        Ok(AuthenticatedPrincipal::nostr_identity(
            principal(9),
            if matches!(self.mode, PrincipalMode::WrongTenant) {
                community(2)
            } else {
                tenant.community_id()
            },
            if matches!(self.mode, PrincipalMode::WrongKey) {
                NostrPublicKey::from_bytes([0; 32])
            } else {
                identity.public_key()
            },
            if matches!(self.mode, PrincipalMode::WrongAuthenticationMethod) {
                NostrAuthenticationMethod::Nip42
            } else {
                NostrAuthenticationMethod::Nip98
            },
            PrincipalScopes::new([
                AuthorizationScope::new("events:read").expect("read scope"),
                AuthorizationScope::new("events:write").expect("write scope"),
            ])
            .expect("principal scopes"),
        ))
    }
}

fn authenticator(
    replay_store: ReplayStore,
    mode: PrincipalMode,
) -> (Nip98Authenticator, Arc<PrincipalResolver>) {
    let resolver = Arc::new(PrincipalResolver {
        mode,
        calls: AtomicUsize::new(0),
        receipts: Mutex::new(Vec::new()),
    });
    (
        Nip98Authenticator::new(true, Arc::new(replay_store), resolver.clone()),
        resolver,
    )
}

#[tokio::test]
async fn nostr_http_nip98_accepts_signature_payload_and_rejects_replay() {
    let body = br#"{"content":"hello"}"#;
    let (event, authorization) =
        signed_nip98_event(1, "https://a.example.test/events", "POST", body, true, NOW);
    let replay_store = ReplayStore::default();
    let (authenticator, resolver) = authenticator(replay_store.clone(), PrincipalMode::Allow);
    let tenant = tenant(community(1), RELAY_HOST);

    let principal = authenticator
        .authenticate(&tenant, &authorization, "POST", "/events", body, true, NOW)
        .await
        .expect("valid NIP-98");
    assert_eq!(principal.community_id(), community(1));
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
    {
        let receipts = resolver.receipts.lock().expect("receipt lock");
        let receipt = receipts.first().expect("actual verified receipt");
        assert_eq!(receipt.signed_authentication_event(), &event);
        assert_eq!(receipt.event_id(), event.claimed_id);
        assert_eq!(receipt.verified_at_seconds(), NOW);
        assert_eq!(receipt.request_url(), "https://a.example.test/events");
        assert_eq!(receipt.request_method(), "POST");
        assert!(receipt.request_payload_verified());
        let expected_hash: [u8; 32] = Sha256::digest(body).into();
        assert_eq!(receipt.request_body_sha256(), &expected_hash);
        assert_eq!(format!("{receipt:?}"), "VerifiedNip98Identity([redacted])");
        assert_eq!(
            receipt
                .for_tenant(&self::tenant(community(2), RELAY_HOST))
                .map(|_| ()),
            Err(NostrHttpDirectoryError::Denied),
        );
    }
    assert!(
        replay_store
            .0
            .claimed
            .lock()
            .expect("replay lock")
            .contains(&(community(1), event.claimed_id))
    );
    assert_eq!(
        authenticator
            .authenticate(&tenant, &authorization, "POST", "/events", body, true, NOW,)
            .await,
        Err(Nip98AuthenticationError::Replay)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn nostr_http_nip98_rejects_host_method_payload_expiry_and_identity_failures() {
    let body = b"body";
    let (_, authorization) =
        signed_nip98_event(2, "https://a.example.test/events", "POST", body, true, NOW);
    let (host_authenticator, resolver) =
        authenticator(ReplayStore::default(), PrincipalMode::Allow);
    assert_eq!(
        host_authenticator
            .authenticate(
                &tenant(community(2), "b.example.test"),
                &authorization,
                "POST",
                "/events",
                body,
                true,
                NOW,
            )
            .await,
        Err(Nip98AuthenticationError::TenantBinding)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);

    let encoded = authorization.strip_prefix("Nostr ").expect("Nostr scheme");
    let mut invalid_event: Value =
        serde_json::from_slice(&BASE64.decode(encoded).expect("encoded NIP-98 event"))
            .expect("wire event");
    invalid_event["sig"] = Value::String("00".repeat(64));
    let invalid_signature = format!(
        "Nostr {}",
        BASE64.encode(serde_json::to_vec(&invalid_event).expect("invalid wire event"))
    );
    let (signature_authenticator, resolver) =
        authenticator(ReplayStore::default(), PrincipalMode::Allow);
    assert_eq!(
        signature_authenticator
            .authenticate(
                &tenant(community(1), RELAY_HOST),
                &invalid_signature,
                "POST",
                "/events",
                body,
                true,
                NOW,
            )
            .await,
        Err(Nip98AuthenticationError::Invalid)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);

    for (method, path, presented_body, now, expected) in [
        (
            "POST",
            "/other",
            body.as_slice(),
            NOW,
            Nip98AuthenticationError::TenantBinding,
        ),
        (
            "GET",
            "/events",
            body.as_slice(),
            NOW,
            Nip98AuthenticationError::Invalid,
        ),
        (
            "POST",
            "/events",
            b"wrong".as_slice(),
            NOW,
            Nip98AuthenticationError::Invalid,
        ),
        (
            "POST",
            "/events",
            body.as_slice(),
            NOW + 61,
            Nip98AuthenticationError::Invalid,
        ),
    ] {
        let (authenticator, resolver) = authenticator(ReplayStore::default(), PrincipalMode::Allow);
        assert_eq!(
            authenticator
                .authenticate(
                    &tenant(community(1), RELAY_HOST),
                    &authorization,
                    method,
                    path,
                    presented_body,
                    true,
                    now,
                )
                .await,
            Err(expected)
        );
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);
    }

    for mode in [
        PrincipalMode::WrongTenant,
        PrincipalMode::WrongKey,
        PrincipalMode::WrongAuthenticationMethod,
        PrincipalMode::Revoked,
    ] {
        let (authenticator, _) = authenticator(ReplayStore::default(), mode);
        assert!(matches!(
            authenticator
                .authenticate(
                    &tenant(community(1), RELAY_HOST),
                    &authorization,
                    "POST",
                    "/events",
                    body,
                    true,
                    NOW,
                )
                .await,
            Err(Nip98AuthenticationError::TenantBinding | Nip98AuthenticationError::Denied)
        ));
    }

    let replay_store = ReplayStore::default();
    replay_store.0.fail.store(1, Ordering::SeqCst);
    let (authenticator, resolver) = authenticator(replay_store, PrincipalMode::Allow);
    assert_eq!(
        authenticator
            .authenticate(
                &tenant(community(1), RELAY_HOST),
                &authorization,
                "POST",
                "/events",
                body,
                true,
                NOW,
            )
            .await,
        Err(Nip98AuthenticationError::Unavailable)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn nostr_http_nip98_receipt_accepts_read_request_without_payload() {
    let (event, authorization) = signed_nip98_event(
        3,
        "https://a.example.test/events?limit=5",
        "GET",
        b"",
        false,
        NOW,
    );
    let (authenticator, resolver) = authenticator(ReplayStore::default(), PrincipalMode::Allow);
    let tenant = tenant(community(1), RELAY_HOST);
    authenticator
        .authenticate(
            &tenant,
            &authorization,
            "GET",
            "/events?limit=5",
            b"",
            false,
            NOW,
        )
        .await
        .expect("valid signed read request");
    let receipts = resolver.receipts.lock().expect("receipt lock");
    let receipt = receipts.first().expect("actual receipt");
    assert_eq!(receipt.signed_authentication_event(), &event);
    assert_eq!(receipt.request_method(), "GET");
    assert!(!receipt.request_payload_verified());
    assert_eq!(
        receipt.request_url(),
        "https://a.example.test/events?limit=5"
    );
    let expected_hash: [u8; 32] = Sha256::digest(b"").into();
    assert_eq!(receipt.request_body_sha256(), &expected_hash);
}

#[tokio::test]
async fn nostr_http_nip98_receipt_never_admits_unbounded_body_or_required_missing_payload() {
    let (_, authorization) =
        signed_nip98_event(4, "https://a.example.test/events", "POST", b"", false, NOW);
    let (authenticator, resolver) = authenticator(ReplayStore::default(), PrincipalMode::Allow);
    let tenant = tenant(community(1), RELAY_HOST);
    assert_eq!(
        authenticator
            .authenticate(&tenant, &authorization, "POST", "/events", b"", true, NOW,)
            .await,
        Err(Nip98AuthenticationError::Invalid)
    );
    let oversized_body = vec![0; 1024 * 1024 + 1];
    assert_eq!(
        authenticator
            .authenticate(
                &tenant,
                &authorization,
                "POST",
                "/events",
                &oversized_body,
                true,
                NOW,
            )
            .await,
        Err(Nip98AuthenticationError::Malformed)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);
    assert!(resolver.receipts.lock().expect("receipt lock").is_empty());
}

#[tokio::test]
async fn profile_http_quota_rejects_invalid_shared_counter_responses() -> anyhow::Result<()> {
    use collab::nostr::replay::{HttpQuotaAdmissionError, RedisHttpAdmissionQuota};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    for response in [
        b"*2\r\n:0\r\n:60\r\n".as_slice(),
        b"*2\r\n:1\r\n:-1\r\n",
        b"*2\r\n:1\r\n:61\r\n",
        b"*2\r\n:bad\r\n:60\r\n",
        b"-ERR unavailable\r\n",
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let quota =
            RedisHttpAdmissionQuota::new(&format!("redis://{}", listener.local_addr()?), 1)?;
        let server = tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                let (mut stream, _) = listener.accept().await?;
                let mut bytes = Vec::new();
                tokio::time::timeout(std::time::Duration::from_secs(5), async {
                    while !bytes.windows(4).any(|value| value == b"EVAL") {
                        let mut chunk = [0u8; 128];
                        let length = stream.read(&mut chunk).await?;
                        anyhow::ensure!(
                            length > 0 && bytes.len() < 4096,
                            "quota did not issue bounded actual atomic Redis command"
                        );
                        bytes.extend_from_slice(&chunk[..length]);
                    }
                    anyhow::Ok(())
                })
                .await??;
                stream.write_all(response).await?;
                anyhow::Ok(())
            })
            .await
            .map_err(|_| anyhow::anyhow!("owned quota response accept/read/write timed out"))?
        });
        let result = quota
            .check(
                &tenant(community(9100), RELAY_HOST),
                PublicKey::from_bytes([3; 32]),
            )
            .await;
        server.await??;
        anyhow::ensure!(
            result == Err(HttpQuotaAdmissionError::Unavailable),
            "quota malformed shared result allowed admission"
        );
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    drop(listener);
    let quota = RedisHttpAdmissionQuota::new(&format!("redis://{address}"), 0)?;
    anyhow::ensure!(
        quota
            .check(
                &tenant(community(9101), RELAY_HOST),
                PublicKey::from_bytes([4; 32])
            )
            .await
            == Err(HttpQuotaAdmissionError::Unavailable),
        "unavailable quota used local fallback"
    );
    Ok(())
}

#[tokio::test]
async fn nostr_http_metadata_disabled_preserves_status_root() -> anyhow::Result<()> {
    use axum::body::HttpBody as _;
    use collab::nostr::http::metadata_http_root_response;
    for accept in [None, Some("application/nostr+json"), Some("text/plain")] {
        let mut headers = axum::http::HeaderMap::new();
        if let Some(accept) = accept {
            headers.insert("accept", accept.parse()?);
        }
        let response =
            metadata_http_root_response(None, &headers, "retained root status".to_owned()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        let mut bytes = Vec::new();
        while let Some(chunk) = body.data().await {
            bytes.extend_from_slice(&chunk?);
        }
        assert_eq!(bytes, b"retained root status");
    }
    Ok(())
}

#[tokio::test]
async fn nostr_http_nip05_preserves_buzz_unicode_and_bounded_name_grammar() {
    let mut directory = Directory::default();
    let key = PublicKey::from_bytes([23; 32]);
    let long_name = "a".repeat(255);
    for name in ["i̇é", "with space", long_name.as_str()] {
        directory
            .identities
            .insert((community(1), name.to_owned()), key);
    }
    let directory = Arc::new(directory);
    let app = router(public_state(directory.clone()));
    for (requested, canonical) in [
        ("İÉ", "i̇é"),
        ("with space", "with space"),
        (long_name.as_str(), long_name.as_str()),
    ] {
        let query = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("name", requested)
            .finish();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/.well-known/nostr.json?{query}"))
                    .header("Host", RELAY_HOST)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(
            response_json(response).await["names"][canonical],
            key.to_hex()
        );
    }
    let before = directory.lookups.lock().expect("lookups").len();
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("name", &"a".repeat(256))
        .finish();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/.well-known/nostr.json?{query}"))
                .header("Host", RELAY_HOST)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(
        response_json(response).await,
        json!({"names":{},"relays":{}})
    );
    assert_eq!(directory.lookups.lock().expect("lookups").len(), before);
}

#[test]
fn closed_nip42_configuration_is_explicit_and_fail_closed() -> anyhow::Result<()> {
    let mut config: collab::Config = serde_json::from_value(json!({
        "http_port":0,"database_url":"","database_max_connections":1,
        "zed_environment":"test","zed_cloud_internal_api_key":"fixture-only"
    }))?;
    assert!(
        collab::nostr::http::configured_profile_http_state(
            &config,
            sea_orm::DatabaseConnection::Disconnected
        )?
        .is_none()
    );
    config.collaboration_nostr_closed_websocket = Some(true);
    let error = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::Disconnected,
    )
    .err()
    .ok_or_else(|| anyhow::anyhow!("socket without routes must deny"))?;
    assert!(
        error
            .to_string()
            .contains("requires configured Nostr routes")
    );
    config.collaboration_nostr_routes=Some(json!([{"community_id":community(1).as_uuid(),"public_url":"wss://a.example.test","allowlist_enabled":true}]).to_string());
    let error = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::Disconnected,
    )
    .err()
    .ok_or_else(|| anyhow::anyhow!("unsupported policy configuration must deny"))?;
    assert!(error.to_string().contains("unknown field"));
    config.collaboration_nostr_routes = Some(
        json!([{"community_id":community(1).as_uuid(),"public_url":"wss://a.example.test"}])
            .to_string(),
    );
    let error = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::Disconnected,
    )
    .err()
    .ok_or_else(|| anyhow::anyhow!("socket without actual Redis must deny"))?;
    assert!(error.to_string().contains("requires shared Redis"));
    Ok(())
}
