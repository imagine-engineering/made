use collab::{
    db::collaboration::{
        event_repository::{
            EventRepository, EventStoreOutcome, EventVerificationState, VerifiedEventRecord,
        },
        persistence_policy::{
            EventDurability, EventPersistencePolicy, EventSearchScope, PrivacyAdmission,
        },
    },
    presence::typing::{
        TYPING_INDICATOR_TTL_MILLIS, TypingDisconnectOutcome, TypingError, TypingIndicatorStore,
        TypingPublicationOutcome,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    NostrAuthenticationMethod, NostrPublicKey, PrincipalId, PrincipalScopes, TenantContext,
    TrustedTenantRoute,
};
use nostr_compat::{
    CanonicalEvent, EventSignature, PublicKey, SignedEvent, TimestampPolicy,
    generated_kinds::KIND_TYPING_INDICATOR,
};
use sea_orm::{DatabaseBackend, MockDatabase};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use uuid::Uuid;

const NOW_MILLIS: u64 = 1_900_000_000_000;

struct AccessFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    required_scope: AuthorizationScope,
    community_membership: CommunityMembership,
    channel_membership: ChannelMembership,
    channel_id: AggregateId,
}

impl AccessFixture {
    fn new(community_value: u128, principal_value: u128, secret_value: u8) -> Self {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(community_value));
        let principal_id = PrincipalId::from_uuid(Uuid::from_u128(principal_value));
        let channel_id = AggregateId::from_uuid(Uuid::from_u128(30));
        let tenant = bind_rpc_tenant(
            Some(
                TrustedTenantRoute::from_listener(community_id, "typing-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant");
        let required_scope = AuthorizationScope::new("channels:access").expect("scope");
        let scopes = PrincipalScopes::new([required_scope.clone()]).expect("principal scopes");
        let public_key = signing_public_key(secret_value);
        let principal = AuthenticatedPrincipal::nostr_identity(
            principal_id,
            community_id,
            NostrPublicKey::from_bytes(*public_key.as_bytes()),
            NostrAuthenticationMethod::Nip42,
            scopes,
        );
        Self {
            tenant,
            principal,
            required_scope,
            community_membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            channel_membership: ChannelMembership {
                community_id,
                channel_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            channel_id,
        }
    }

    fn authorization(
        &self,
        action: AuthorizationAction,
        channel_status: MembershipStatus,
        now_millis: u64,
    ) -> AuthorizationRequest<'_> {
        let mut channel_membership = self.channel_membership;
        channel_membership.status = channel_status;
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.required_scope,
            action,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: self.channel_id,
                owner_principal_id: None,
                channel_id: Some(self.channel_id),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.community_membership),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(channel_membership),
            delegation: None,
            now_millis,
        }
    }
}

fn signing_keypair(secret_value: u8) -> Keypair {
    let secret = SecretKey::from_slice(&[secret_value; 32]).expect("fixture secret");
    Keypair::from_secret_key(&Secp256k1::new(), &secret)
}

fn signing_public_key(secret_value: u8) -> PublicKey {
    let (public_key, _) = XOnlyPublicKey::from_keypair(&signing_keypair(secret_value));
    PublicKey::from_bytes(public_key.serialize())
}

fn signed_typing_event(
    secret_value: u8,
    channel_id: AggregateId,
    created_at: u64,
    content: impl Into<String>,
) -> SignedEvent {
    let keypair = signing_keypair(secret_value);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        created_at,
        u16::try_from(KIND_TYPING_INDICATOR).expect("typing kind fits u16"),
        vec![vec!["h".into(), channel_id.to_string()]],
        content.into(),
    );
    let claimed_id = event.event_id().expect("event ID");
    let signature = Secp256k1::new()
        .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

#[test]
fn typing_rate_limit_is_per_principal_with_burst_ten_and_two_per_second_refill() {
    let store = TypingIndicatorStore::new();
    let access = AccessFixture::new(1, 2, 11);
    let token = store
        .register_connection(
            &access.tenant,
            &access.principal,
            Uuid::from_u128(40),
            1,
            NOW_MILLIS,
        )
        .expect("register connection");
    let authorization = access.authorization(
        AuthorizationAction::Write,
        MembershipStatus::Active,
        NOW_MILLIS,
    );

    for sequence in 0..10 {
        let event = signed_typing_event(
            11,
            access.channel_id,
            NOW_MILLIS / 1_000,
            format!("typing-{sequence}"),
        );
        store
            .publish(token, &authorization, &event, NOW_MILLIS)
            .expect("burst publication admitted");
    }
    let limited = signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "typing-limited");
    assert_eq!(
        store.publish(token, &authorization, &limited, NOW_MILLIS),
        Err(TypingError::RateLimited {
            retry_after_millis: 500,
        })
    );

    let refilled =
        signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "typing-refilled");
    assert!(
        store
            .publish(token, &authorization, &refilled, NOW_MILLIS + 500)
            .is_ok()
    );
}

#[test]
fn typing_rejects_forged_and_inactive_channel_publications_without_state() {
    let store = TypingIndicatorStore::new();
    let access = AccessFixture::new(1, 2, 11);
    let token = store
        .register_connection(
            &access.tenant,
            &access.principal,
            Uuid::from_u128(40),
            1,
            NOW_MILLIS,
        )
        .expect("register connection");
    let active = access.authorization(
        AuthorizationAction::Write,
        MembershipStatus::Active,
        NOW_MILLIS,
    );
    let forged = signed_typing_event(12, access.channel_id, NOW_MILLIS / 1_000, "forged");
    assert_eq!(
        store.publish(token, &active, &forged, NOW_MILLIS),
        Err(TypingError::Unauthorized)
    );

    let valid = signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "valid");
    let revoked = access.authorization(
        AuthorizationAction::Write,
        MembershipStatus::Revoked,
        NOW_MILLIS,
    );
    assert_eq!(
        store.publish(token, &revoked, &valid, NOW_MILLIS),
        Err(TypingError::Unauthorized)
    );
    let mut tampered = valid;
    tampered.event.content = "tampered after signing".into();
    assert_eq!(
        store.publish(token, &active, &tampered, NOW_MILLIS),
        Err(TypingError::InvalidEvent)
    );
    assert_eq!(
        store.metrics(NOW_MILLIS).expect("metrics").active_entries,
        0
    );
}

#[test]
fn typing_sources_remain_independent_across_two_devices() {
    let store = TypingIndicatorStore::new();
    let access = AccessFixture::new(1, 2, 11);
    let first_connection_id = Uuid::from_u128(40);
    let second_connection_id = Uuid::from_u128(41);
    let first = store
        .register_connection(
            &access.tenant,
            &access.principal,
            first_connection_id,
            1,
            NOW_MILLIS,
        )
        .expect("register first device");
    let second = store
        .register_connection(
            &access.tenant,
            &access.principal,
            second_connection_id,
            1,
            NOW_MILLIS,
        )
        .expect("register second device");
    let write = access.authorization(
        AuthorizationAction::Write,
        MembershipStatus::Active,
        NOW_MILLIS,
    );
    store
        .publish(
            first,
            &write,
            &signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "first"),
            NOW_MILLIS,
        )
        .expect("publish first device");
    store
        .publish(
            second,
            &write,
            &signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "second"),
            NOW_MILLIS,
        )
        .expect("publish second device");

    let first_reconnected = store
        .register_connection(
            &access.tenant,
            &access.principal,
            first_connection_id,
            2,
            NOW_MILLIS + 1,
        )
        .expect("reconnect first device");
    let read = access.authorization(
        AuthorizationAction::Read,
        MembershipStatus::Active,
        NOW_MILLIS + 1,
    );
    assert_eq!(
        store
            .active_for_channel(&read, access.channel_id, NOW_MILLIS + 1)
            .expect("multi-device snapshot")
            .len(),
        1
    );
    assert_eq!(
        store
            .disconnect(first_reconnected)
            .expect("disconnect first device"),
        TypingDisconnectOutcome::Removed
    );
    assert_eq!(
        store
            .active_for_channel(&read, access.channel_id, NOW_MILLIS + 1)
            .expect("second device remains")
            .len(),
        1
    );
    assert_eq!(
        store.disconnect(second).expect("disconnect second device"),
        TypingDisconnectOutcome::Removed
    );
    assert!(
        store
            .active_for_channel(&read, access.channel_id, NOW_MILLIS + 1)
            .expect("all devices disconnected")
            .is_empty()
    );
}

#[test]
fn typing_expires_rejects_replay_and_reconnects_with_a_fresh_generation() {
    let store = TypingIndicatorStore::new();
    let access = AccessFixture::new(1, 2, 11);
    let connection_id = Uuid::from_u128(40);
    let first_token = store
        .register_connection(
            &access.tenant,
            &access.principal,
            connection_id,
            1,
            NOW_MILLIS,
        )
        .expect("register first generation");
    let write = access.authorization(
        AuthorizationAction::Write,
        MembershipStatus::Active,
        NOW_MILLIS,
    );
    let event = signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "typing");
    assert_eq!(
        store.publish(first_token, &write, &event, NOW_MILLIS),
        Ok(TypingPublicationOutcome::Applied)
    );
    let read = access.authorization(
        AuthorizationAction::Read,
        MembershipStatus::Active,
        NOW_MILLIS,
    );
    assert_eq!(
        store
            .active_for_channel(
                &read,
                access.channel_id,
                NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS - 1,
            )
            .expect("typing snapshot")
            .len(),
        1
    );
    assert!(
        store
            .active_for_channel(
                &read,
                access.channel_id,
                NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
            )
            .expect("expired snapshot")
            .is_empty()
    );
    assert_eq!(
        store.publish(
            first_token,
            &write,
            &event,
            NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
        ),
        Ok(TypingPublicationOutcome::Duplicate)
    );

    let second_token = store
        .register_connection(
            &access.tenant,
            &access.principal,
            connection_id,
            2,
            NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
        )
        .expect("register reconnect generation");
    let fresh_event = signed_typing_event(
        11,
        access.channel_id,
        NOW_MILLIS / 1_000 + 60,
        "fresh typing",
    );
    assert_eq!(
        store.publish(
            first_token,
            &write,
            &fresh_event,
            NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
        ),
        Err(TypingError::StaleConnection)
    );
    assert_eq!(
        store.publish(
            second_token,
            &write,
            &fresh_event,
            NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
        ),
        Ok(TypingPublicationOutcome::Applied)
    );
    assert_eq!(
        store.disconnect(first_token).expect("stale disconnect"),
        TypingDisconnectOutcome::Stale
    );
    assert_eq!(
        store.disconnect(second_token).expect("current disconnect"),
        TypingDisconnectOutcome::Removed
    );
    assert!(
        store
            .active_for_channel(
                &read,
                access.channel_id,
                NOW_MILLIS + TYPING_INDICATOR_TTL_MILLIS,
            )
            .expect("disconnected snapshot")
            .is_empty()
    );
}

#[tokio::test]
async fn typing_events_produce_zero_durable_rows() {
    let access = AccessFixture::new(1, 2, 11);
    let event = signed_typing_event(11, access.channel_id, NOW_MILLIS / 1_000, "ephemeral");
    let record = VerifiedEventRecord::new(
        access.tenant.community_id(),
        event,
        EventVerificationState::Live,
        NOW_MILLIS,
        TimestampPolicy::Bounded {
            now: NOW_MILLIS / 1_000,
            max_past_seconds: 0,
            max_future_seconds: 0,
        },
    )
    .expect("verified typing event");
    let decision = EventPersistencePolicy::evaluate(
        record.signed_event().event.kind,
        PrivacyAdmission::community(),
    )
    .expect("typing persistence classification");
    assert_eq!(decision.durability(), EventDurability::TransientOnly);
    assert_eq!(decision.search_scope(), EventSearchScope::Excluded);
    let repository =
        EventRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("Postgres repository");

    assert_eq!(
        repository
            .store(&access.tenant, &record, decision)
            .await
            .expect("ephemeral disposition"),
        EventStoreOutcome::EphemeralNotPersisted
    );
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}
