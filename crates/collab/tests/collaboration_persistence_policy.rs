use collab::{
    db::collaboration::{
        event_repository::{
            EventRepository, EventRepositoryError, EventStoreOutcome, EventVerificationState,
            VerifiedEventRecord,
        },
        persistence_policy::{
            EventDurability, EventPersistencePolicy, EventSearchScope, PersistencePolicyError,
            PrivacyAdmission, SearchAudience,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use nostr_compat::{
    CanonicalEvent, EventSignature, PublicKey, SignedEvent, TimestampPolicy,
    generated_kinds::{
        KIND_AGENT_TURN_METRIC, KIND_GIFT_WRAP, KIND_MEDIA_UPLOAD, KIND_PERSONA,
        KIND_PRESENCE_UPDATE, KIND_PRIVATE_MANAGED_AGENT, KIND_TEXT_NOTE,
    },
};
use sea_orm::{DatabaseBackend, MockDatabase};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use uuid::Uuid;

fn tenant() -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "persistence-policy")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn signed_event(kind: u16, content: &str) -> SignedEvent {
    let secret = SecretKey::from_slice(&[11; 32]).expect("fixture secret");
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        1_900_000_000,
        kind,
        Vec::new(),
        content.to_owned(),
    );
    let claimed_id = event.event_id().expect("event ID");
    let signature =
        secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

fn verified_record(tenant: &TenantContext, kind: u16, content: &str) -> VerifiedEventRecord {
    VerifiedEventRecord::new(
        tenant.community_id(),
        signed_event(kind, content),
        EventVerificationState::Historical,
        1_900_000_000_000,
        TimestampPolicy::Historical,
    )
    .expect("verified record")
}

#[tokio::test]
async fn collaboration_persistence_policy_keeps_ephemeral_events_out_of_sql_and_search() {
    let tenant = tenant();
    let record = verified_record(
        &tenant,
        u16::try_from(KIND_PRESENCE_UPDATE).expect("u16 kind"),
        "private-presence-marker",
    );
    let decision = EventPersistencePolicy::evaluate(
        record.signed_event().event.kind,
        PrivacyAdmission::community(),
    )
    .expect("ephemeral policy");
    assert_eq!(decision.durability(), EventDurability::TransientOnly);
    assert_eq!(decision.search_scope(), EventSearchScope::Excluded);
    assert!(!decision.allows_search_for(SearchAudience::Community));
    assert!(!decision.allows_search_for(SearchAudience::AuthorizedRestricted));
    let repository =
        EventRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("Postgres repository");

    assert_eq!(
        repository
            .store(&tenant, &record, decision)
            .await
            .expect("transient disposition"),
        EventStoreOutcome::EphemeralNotPersisted
    );
    let transaction_log = repository.into_connection().into_transaction_log();
    assert!(transaction_log.is_empty(), "{transaction_log:#?}");
    let log = format!("{transaction_log:#?}");
    assert!(!log.contains("private-presence-marker"));
}

#[tokio::test]
async fn collaboration_persistence_policy_rejects_private_kind_decision_substitution_before_sql() {
    let tenant = tenant();
    let record = verified_record(
        &tenant,
        u16::try_from(KIND_PRIVATE_MANAGED_AGENT).expect("u16 kind"),
        "private-agent-marker",
    );
    assert_eq!(
        EventPersistencePolicy::evaluate(
            record.signed_event().event.kind,
            PrivacyAdmission::community(),
        ),
        Err(PersistencePolicyError::PrivacyDenied)
    );
    let substituted = EventPersistencePolicy::evaluate(
        u16::try_from(KIND_TEXT_NOTE).expect("u16 kind"),
        PrivacyAdmission::community(),
    )
    .expect("public text policy");
    let repository =
        EventRepository::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("Postgres repository");

    assert!(matches!(
        repository.store(&tenant, &record, substituted).await,
        Err(EventRepositoryError::PersistencePolicy(
            PersistencePolicyError::DecisionMismatch
        ))
    ));
    let transaction_log = repository.into_connection().into_transaction_log();
    assert!(transaction_log.is_empty(), "{transaction_log:#?}");
    let log = format!("{transaction_log:#?}");
    assert!(!log.contains("private-agent-marker"));
    let outward_error = PersistencePolicyError::PrivacyDenied.to_string();
    assert!(!outward_error.contains(&KIND_PRIVATE_MANAGED_AGENT.to_string()));
    assert!(!outward_error.contains("private-agent-marker"));
}

#[test]
fn collaboration_persistence_policy_requires_every_overlapping_privacy_gate() {
    let metric_kind = u16::try_from(KIND_AGENT_TURN_METRIC).expect("u16 kind");
    assert_eq!(
        EventPersistencePolicy::evaluate(metric_kind, PrivacyAdmission::recipient()),
        Err(PersistencePolicyError::PrivacyDenied)
    );
    let metric = EventPersistencePolicy::evaluate(
        metric_kind,
        PrivacyAdmission::recipient().with_result_reader(),
    )
    .expect("recipient and result admission");
    assert_eq!(metric.durability(), EventDurability::Durable);
    assert_eq!(
        metric.search_scope(),
        EventSearchScope::AuthorizedRestricted
    );
    assert!(!metric.allows_search_for(SearchAudience::Community));
    assert!(metric.allows_search_for(SearchAudience::AuthorizedRestricted));

    let persona_kind = u16::try_from(KIND_PERSONA).expect("u16 kind");
    assert_eq!(
        EventPersistencePolicy::evaluate(persona_kind, PrivacyAdmission::community()),
        Err(PersistencePolicyError::PrivacyDenied)
    );
    assert!(
        EventPersistencePolicy::evaluate(
            persona_kind,
            PrivacyAdmission::community().with_explicit_share(),
        )
        .is_ok()
    );
    assert!(EventPersistencePolicy::evaluate(persona_kind, PrivacyAdmission::author()).is_ok());

    let gift_wrap = EventPersistencePolicy::evaluate(
        u16::try_from(KIND_GIFT_WRAP).expect("u16 kind"),
        PrivacyAdmission::recipient(),
    )
    .expect("recipient-scoped envelope");
    assert_eq!(
        gift_wrap.search_scope(),
        EventSearchScope::AuthorizedRestricted
    );
}

#[test]
fn collaboration_persistence_policy_rejects_unclassified_and_internal_kinds() {
    assert_eq!(
        EventPersistencePolicy::evaluate(u16::MAX, PrivacyAdmission::community()),
        Err(PersistencePolicyError::UnclassifiedKind)
    );
    assert!(EventPersistencePolicy::evaluate(41, PrivacyAdmission::community()).is_ok());
    assert_eq!(
        EventPersistencePolicy::evaluate(
            u16::try_from(KIND_MEDIA_UPLOAD).expect("u16 kind"),
            PrivacyAdmission::community(),
        ),
        Err(PersistencePolicyError::NonRelayKind)
    );
}
