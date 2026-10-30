use std::collections::{BTreeMap, BTreeSet};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use collab::messages::dm_visibility::{DmVisibilityAccess, DmVisibilityRepository};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, PrincipalId,
    PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use nostr_compat::dm::{
    DirectMessageIndexing, GiftWrapEnvelope, Nip44Ciphertext, authorize_gift_wrap_filters,
};
use nostr_compat::filter::EventFilter;
use nostr_compat::{EventSignature, PublicKey, SignedEvent};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey};
use uuid::Uuid;

const GIFT_WRAP_KIND: u16 = 1059;
const FIXED_WAKE_BODY: &str =
    r#"{"aps":{"alert":{"body":"Reconnect to your relay now"},"mutable-content":1}}"#;
const SEARCH_MIGRATION: &str =
    include_str!("../migrations/20260820000500_collaboration_search.up.sql");

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TraceCommunity {
    Alpha,
    Beta,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TraceRole {
    Participant,
    Nonparticipant,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PrivacySeam {
    Event,
    Filter,
    Count,
    Search,
    Notification,
    Logs,
}

impl PrivacySeam {
    const ALL: [Self; 6] = [
        Self::Event,
        Self::Filter,
        Self::Count,
        Self::Search,
        Self::Notification,
        Self::Logs,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Availability {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PrivacyOutcome {
    Resource {
        existing: Availability,
        missing: Availability,
    },
    Count {
        value: usize,
        identifiers: Vec<String>,
    },
    Search {
        hits: Vec<String>,
    },
    Notification {
        body: Option<String>,
    },
    Logs {
        output: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrivacyObservation {
    community: TraceCommunity,
    role: TraceRole,
    seam: PrivacySeam,
    outcome: PrivacyOutcome,
}

#[derive(Clone, Copy)]
struct Actor {
    principal_id: PrincipalId,
    public_key: PublicKey,
    service_account_id: ServiceAccountId,
}

struct CommunityFixture {
    trace_community: TraceCommunity,
    community_id: CommunityId,
    participant: Actor,
    nonparticipant: Actor,
    signed_event: SignedEvent,
    plaintext_marker: &'static str,
    ciphertext: String,
    hidden_dm_ids: Vec<AggregateId>,
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn keypair(last_byte: u8) -> Keypair {
    let mut bytes = [0; 32];
    bytes[31] = last_byte;
    let secret = SecretKey::from_slice(&bytes).expect("fixture secret key");
    Keypair::from_secret_key(&Secp256k1::new(), &secret)
}

fn public_key(keypair: &Keypair) -> PublicKey {
    PublicKey::from_bytes(keypair.x_only_public_key().0.serialize())
}

fn actor(principal_value: u128, secret_last_byte: u8, service_account_id: u64) -> Actor {
    Actor {
        principal_id: principal(principal_value),
        public_key: public_key(&keypair(secret_last_byte)),
        service_account_id: ServiceAccountId::new(service_account_id),
    }
}

fn signed_gift_wrap(
    recipient: PublicKey,
    plaintext_marker: &'static str,
    created_at: u64,
) -> (SignedEvent, String) {
    let ephemeral_keypair = keypair(1);
    let mut ciphertext_bytes = vec![0; 99];
    ciphertext_bytes[0] = 2;
    let marker_end = 1 + plaintext_marker.len();
    ciphertext_bytes[1..marker_end].copy_from_slice(plaintext_marker.as_bytes());
    let ciphertext = STANDARD.encode(ciphertext_bytes);
    let envelope = GiftWrapEnvelope::new(
        public_key(&ephemeral_keypair),
        created_at,
        recipient,
        Nip44Ciphertext::parse(ciphertext.clone()).expect("fixture NIP-44 ciphertext"),
    );
    let event = envelope.into_canonical_event();
    let claimed_id = event.event_id().expect("fixture event id");
    let signature = Secp256k1::new().sign_schnorr_no_aux_rand(
        &Message::from_digest(*claimed_id.as_bytes()),
        &ephemeral_keypair,
    );
    (
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("fixture signature"),
        },
        ciphertext,
    )
}

fn community_fixture(
    trace_community: TraceCommunity,
    community_id: CommunityId,
    participant: Actor,
    nonparticipant: Actor,
    plaintext_marker: &'static str,
    created_at: u64,
    hidden_dm_ids: Vec<AggregateId>,
) -> CommunityFixture {
    let (signed_event, ciphertext) =
        signed_gift_wrap(participant.public_key, plaintext_marker, created_at);
    CommunityFixture {
        trace_community,
        community_id,
        participant,
        nonparticipant,
        signed_event,
        plaintext_marker,
        ciphertext,
        hidden_dm_ids,
    }
}

fn availability<T, E>(result: &Result<T, E>) -> Availability {
    if result.is_ok() {
        Availability::Available
    } else {
        Availability::Unavailable
    }
}

fn gift_wrap_filter(recipient: PublicKey) -> EventFilter {
    EventFilter {
        kinds: vec![GIFT_WRAP_KIND],
        generic_tags: BTreeMap::from([('p', vec![recipient.to_hex()])]),
        ..EventFilter::default()
    }
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "dm-privacy-conformance")
                .expect("trusted fixture route"),
        ),
        &[],
    )
    .expect("fixture tenant")
}

fn visibility_scope() -> AuthorizationScope {
    AuthorizationScope::new("collaboration:dms:visibility").expect("fixture scope")
}

fn snapshot_row(channel_id: AggregateId) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([("channel_id".to_owned(), channel_id.as_uuid().into())])
}

fn tenant_exec_result() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

async fn observe_count(
    fixture: &CommunityFixture,
    actor: Actor,
    role: TraceRole,
) -> (PrivacyObservation, String) {
    let tenant = tenant(fixture.community_id);
    let scope = visibility_scope();
    let principal = AuthenticatedPrincipal::zed_account(
        actor.principal_id,
        fixture.community_id,
        actor.service_account_id,
        PrincipalScopes::new([scope.clone()]).expect("fixture principal scopes"),
    );
    let membership = CommunityMembership {
        community_id: fixture.community_id,
        principal_id: actor.principal_id,
        role: MembershipRole::Member,
        status: MembershipStatus::Active,
        version: AggregateVersion::FIRST,
    };
    let authorization = AuthorizationRequest {
        tenant: &tenant,
        principal: &principal,
        required_scope: &scope,
        action: AuthorizationAction::Read,
        resource: AuthorizationResource {
            community_id: fixture.community_id,
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(fixture.community_id.as_uuid()),
            owner_principal_id: Some(actor.principal_id),
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(membership),
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: 1_900_000_000_000,
    };
    let rows = if role == TraceRole::Participant {
        fixture
            .hidden_dm_ids
            .iter()
            .copied()
            .map(snapshot_row)
            .collect()
    } else {
        Vec::new()
    };
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([tenant_exec_result()])
        .append_query_results([rows])
        .into_connection();
    let repository = DmVisibilityRepository::new(database).expect("fixture repository");
    let snapshot = repository
        .snapshot(DmVisibilityAccess {
            authorization: &authorization,
        })
        .await
        .expect("authorized viewer snapshot");
    let identifiers = snapshot
        .hidden_dm_ids()
        .iter()
        .map(|id| id.as_uuid().to_string())
        .collect();
    let observation = PrivacyObservation {
        community: fixture.trace_community,
        role,
        seam: PrivacySeam::Count,
        outcome: PrivacyOutcome::Count {
            value: snapshot.hidden_count(),
            identifiers,
        },
    };
    let database_log = format!("{:#?}", repository.into_connection().into_transaction_log());
    (observation, database_log)
}

async fn observe_fixture(fixture: &CommunityFixture) -> Vec<PrivacyObservation> {
    let participant_envelope =
        GiftWrapEnvelope::parse_signed_event(&fixture.signed_event, fixture.participant.public_key)
            .expect("participant gift wrap");
    let missing_recipient = public_key(&keypair(4));
    let search_excluded = participant_envelope.indexing() == DirectMessageIndexing::Excluded
        && !SEARCH_MIGRATION.contains("1059");
    let mut observations = Vec::new();

    for (role, actor) in [
        (TraceRole::Participant, fixture.participant),
        (TraceRole::Nonparticipant, fixture.nonparticipant),
    ] {
        let parsed = GiftWrapEnvelope::parse_signed_event(&fixture.signed_event, actor.public_key);
        observations.push(PrivacyObservation {
            community: fixture.trace_community,
            role,
            seam: PrivacySeam::Event,
            outcome: PrivacyOutcome::Resource {
                existing: availability(&parsed),
                missing: Availability::Unavailable,
            },
        });

        let existing_filter = gift_wrap_filter(fixture.participant.public_key);
        let missing_filter = gift_wrap_filter(missing_recipient);
        observations.push(PrivacyObservation {
            community: fixture.trace_community,
            role,
            seam: PrivacySeam::Filter,
            outcome: PrivacyOutcome::Resource {
                existing: availability(&authorize_gift_wrap_filters(
                    std::slice::from_ref(&existing_filter),
                    actor.public_key,
                )),
                missing: availability(&authorize_gift_wrap_filters(
                    std::slice::from_ref(&missing_filter),
                    actor.public_key,
                )),
            },
        });

        let (count_observation, database_log) = observe_count(fixture, actor, role).await;
        observations.push(count_observation);
        observations.push(PrivacyObservation {
            community: fixture.trace_community,
            role,
            seam: PrivacySeam::Search,
            outcome: PrivacyOutcome::Search {
                hits: if search_excluded {
                    Vec::new()
                } else {
                    vec!["gift-wrap-indexing-regression".to_owned()]
                },
            },
        });
        observations.push(PrivacyObservation {
            community: fixture.trace_community,
            role,
            seam: PrivacySeam::Notification,
            outcome: PrivacyOutcome::Notification {
                body: (role == TraceRole::Participant).then(|| FIXED_WAKE_BODY.to_owned()),
            },
        });

        let codec_log = match parsed {
            Ok(envelope) => format!("{envelope:?} {:?}", envelope.ciphertext()),
            Err(_) => "direct-message unavailable".to_owned(),
        };
        observations.push(PrivacyObservation {
            community: fixture.trace_community,
            role,
            seam: PrivacySeam::Logs,
            outcome: PrivacyOutcome::Logs {
                output: format!("{codec_log}\n{database_log}"),
            },
        });
    }
    observations
}

fn visible_strings(outcome: &PrivacyOutcome) -> Vec<&str> {
    match outcome {
        PrivacyOutcome::Resource { .. } => Vec::new(),
        PrivacyOutcome::Count { identifiers, .. } => {
            identifiers.iter().map(String::as_str).collect()
        }
        PrivacyOutcome::Search { hits } => hits.iter().map(String::as_str).collect(),
        PrivacyOutcome::Notification { body } => body.iter().map(String::as_str).collect(),
        PrivacyOutcome::Logs { output } => vec![output],
    }
}

fn audit_privacy_trace(
    observations: &[PrivacyObservation],
    expected_participant_ids: &BTreeMap<TraceCommunity, Vec<String>>,
    sensitive_tokens: &[String],
) -> Result<(), String> {
    let expected_coverage = [TraceCommunity::Alpha, TraceCommunity::Beta]
        .into_iter()
        .flat_map(|community| {
            [TraceRole::Participant, TraceRole::Nonparticipant]
                .into_iter()
                .flat_map(move |role| {
                    PrivacySeam::ALL
                        .into_iter()
                        .map(move |seam| (community, role, seam))
                })
        })
        .collect::<BTreeSet<_>>();
    let actual_coverage = observations
        .iter()
        .map(|observation| (observation.community, observation.role, observation.seam))
        .collect::<BTreeSet<_>>();
    if actual_coverage != expected_coverage || observations.len() != expected_coverage.len() {
        return Err("trace must cover every DM privacy seam exactly once".to_owned());
    }

    for observation in observations {
        for visible in visible_strings(&observation.outcome) {
            if sensitive_tokens
                .iter()
                .any(|token| !token.is_empty() && visible.contains(token))
            {
                return Err(format!(
                    "sensitive DM data crossed the {:?} seam",
                    observation.seam
                ));
            }
        }

        match (observation.seam, &observation.outcome) {
            (
                PrivacySeam::Event | PrivacySeam::Filter,
                PrivacyOutcome::Resource { existing, missing },
            ) => {
                let expected_existing = if observation.role == TraceRole::Participant {
                    Availability::Available
                } else {
                    Availability::Unavailable
                };
                if *existing != expected_existing || *missing != Availability::Unavailable {
                    return Err(format!(
                        "DM existence leaked through the {:?} seam",
                        observation.seam
                    ));
                }
            }
            (PrivacySeam::Count, PrivacyOutcome::Count { value, identifiers }) => {
                let expected = if observation.role == TraceRole::Participant {
                    expected_participant_ids
                        .get(&observation.community)
                        .ok_or_else(|| "unknown trace community".to_owned())?
                        .as_slice()
                } else {
                    &[]
                };
                if *value != expected.len() || identifiers != expected {
                    return Err(
                        "DM identifiers or counts crossed a participant boundary".to_owned()
                    );
                }
            }
            (PrivacySeam::Search, PrivacyOutcome::Search { hits }) => {
                if !hits.is_empty() {
                    return Err("DM content or existence entered search".to_owned());
                }
            }
            (PrivacySeam::Notification, PrivacyOutcome::Notification { body }) => {
                let expected =
                    (observation.role == TraceRole::Participant).then_some(FIXED_WAKE_BODY);
                if body.as_deref() != expected {
                    return Err("DM notification was not recipient-only and wake-only".to_owned());
                }
            }
            (PrivacySeam::Logs, PrivacyOutcome::Logs { output }) => {
                if observation.role == TraceRole::Participant && !output.contains("<redacted>") {
                    return Err("participant DM logs did not prove ciphertext redaction".to_owned());
                }
                if !output.contains("membership.community_id = $1")
                    || !output.contains("membership.principal_id = $2")
                    || output.to_ascii_lowercase().contains("count(")
                {
                    return Err(
                        "DM visibility reads are not owner-scoped before counting".to_owned()
                    );
                }
            }
            _ => return Err("trace outcome does not match its privacy seam".to_owned()),
        }
    }
    Ok(())
}

fn expected_ids(fixtures: &[CommunityFixture]) -> BTreeMap<TraceCommunity, Vec<String>> {
    fixtures
        .iter()
        .map(|fixture| {
            (
                fixture.trace_community,
                fixture
                    .hidden_dm_ids
                    .iter()
                    .map(|id| id.as_uuid().to_string())
                    .collect(),
            )
        })
        .collect()
}

fn sensitive_tokens(fixtures: &[CommunityFixture]) -> Vec<String> {
    fixtures
        .iter()
        .flat_map(|fixture| {
            [
                fixture.plaintext_marker.to_owned(),
                fixture.ciphertext.clone(),
                fixture.signed_event.claimed_id.to_hex(),
            ]
        })
        .collect()
}

#[tokio::test]
async fn two_users_and_two_communities_report_no_dm_privacy_leak() {
    let alice = actor(101, 2, 201);
    let bob = actor(102, 3, 202);
    let fixtures = vec![
        community_fixture(
            TraceCommunity::Alpha,
            community(1),
            alice,
            bob,
            "alpha private plaintext",
            1_756_000_001,
            vec![aggregate(1001)],
        ),
        community_fixture(
            TraceCommunity::Beta,
            community(2),
            bob,
            alice,
            "beta private plaintext",
            1_756_000_002,
            vec![aggregate(2001), aggregate(2002)],
        ),
    ];
    let mut observations = Vec::new();
    for fixture in &fixtures {
        observations.extend(observe_fixture(fixture).await);
    }

    audit_privacy_trace(
        &observations,
        &expected_ids(&fixtures),
        &sensitive_tokens(&fixtures),
    )
    .expect("independent DM privacy audit");

    let participant_wakes = observations
        .iter()
        .filter_map(|observation| {
            if observation.role == TraceRole::Participant
                && observation.seam == PrivacySeam::Notification
            {
                let PrivacyOutcome::Notification { body } = &observation.outcome else {
                    return None;
                };
                body.as_deref()
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(participant_wakes, vec![FIXED_WAKE_BODY, FIXED_WAKE_BODY]);
}

fn safe_checker_trace() -> (
    Vec<PrivacyObservation>,
    BTreeMap<TraceCommunity, Vec<String>>,
    Vec<String>,
) {
    let expected_ids = BTreeMap::from([
        (TraceCommunity::Alpha, vec!["alpha-dm".to_owned()]),
        (
            TraceCommunity::Beta,
            vec!["beta-dm-1".to_owned(), "beta-dm-2".to_owned()],
        ),
    ]);
    let mut observations = Vec::new();
    for community in [TraceCommunity::Alpha, TraceCommunity::Beta] {
        for role in [TraceRole::Participant, TraceRole::Nonparticipant] {
            for seam in PrivacySeam::ALL {
                let outcome = match seam {
                    PrivacySeam::Event | PrivacySeam::Filter => PrivacyOutcome::Resource {
                        existing: if role == TraceRole::Participant {
                            Availability::Available
                        } else {
                            Availability::Unavailable
                        },
                        missing: Availability::Unavailable,
                    },
                    PrivacySeam::Count => PrivacyOutcome::Count {
                        value: if role == TraceRole::Participant {
                            expected_ids[&community].len()
                        } else {
                            0
                        },
                        identifiers: if role == TraceRole::Participant {
                            expected_ids[&community].clone()
                        } else {
                            Vec::new()
                        },
                    },
                    PrivacySeam::Search => PrivacyOutcome::Search { hits: Vec::new() },
                    PrivacySeam::Notification => PrivacyOutcome::Notification {
                        body: (role == TraceRole::Participant).then(|| FIXED_WAKE_BODY.to_owned()),
                    },
                    PrivacySeam::Logs => PrivacyOutcome::Logs {
                        output: if role == TraceRole::Participant {
                            "ciphertext: <redacted>; membership.community_id = $1; membership.principal_id = $2"
                                .to_owned()
                        } else {
                            "direct-message unavailable; membership.community_id = $1; membership.principal_id = $2"
                                .to_owned()
                        },
                    },
                };
                observations.push(PrivacyObservation {
                    community,
                    role,
                    seam,
                    outcome,
                });
            }
        }
    }
    (
        observations,
        expected_ids,
        vec!["private-marker".to_owned()],
    )
}

fn find_observation(
    observations: &mut [PrivacyObservation],
    community: TraceCommunity,
    role: TraceRole,
    seam: PrivacySeam,
) -> &mut PrivacyObservation {
    observations
        .iter_mut()
        .find(|observation| {
            observation.community == community
                && observation.role == role
                && observation.seam == seam
        })
        .expect("checker fixture observation")
}

#[test]
fn independent_checker_rejects_every_dm_leak_class() {
    let (safe, expected_ids, sensitive_tokens) = safe_checker_trace();
    audit_privacy_trace(&safe, &expected_ids, &sensitive_tokens).expect("safe checker fixture");

    let mut plaintext_leak = safe.clone();
    find_observation(
        &mut plaintext_leak,
        TraceCommunity::Alpha,
        TraceRole::Participant,
        PrivacySeam::Logs,
    )
    .outcome = PrivacyOutcome::Logs {
        output: "ciphertext: <redacted>; private-marker".to_owned(),
    };
    assert!(audit_privacy_trace(&plaintext_leak, &expected_ids, &sensitive_tokens).is_err());

    let mut existence_leak = safe.clone();
    find_observation(
        &mut existence_leak,
        TraceCommunity::Alpha,
        TraceRole::Nonparticipant,
        PrivacySeam::Event,
    )
    .outcome = PrivacyOutcome::Resource {
        existing: Availability::Available,
        missing: Availability::Unavailable,
    };
    assert!(audit_privacy_trace(&existence_leak, &expected_ids, &sensitive_tokens).is_err());

    let mut count_leak = safe.clone();
    find_observation(
        &mut count_leak,
        TraceCommunity::Alpha,
        TraceRole::Nonparticipant,
        PrivacySeam::Count,
    )
    .outcome = PrivacyOutcome::Count {
        value: 1,
        identifiers: vec!["alpha-dm".to_owned()],
    };
    assert!(audit_privacy_trace(&count_leak, &expected_ids, &sensitive_tokens).is_err());

    let mut search_leak = safe.clone();
    find_observation(
        &mut search_leak,
        TraceCommunity::Beta,
        TraceRole::Nonparticipant,
        PrivacySeam::Search,
    )
    .outcome = PrivacyOutcome::Search {
        hits: vec!["private DM exists".to_owned()],
    };
    assert!(audit_privacy_trace(&search_leak, &expected_ids, &sensitive_tokens).is_err());

    let mut notification_leak = safe;
    find_observation(
        &mut notification_leak,
        TraceCommunity::Beta,
        TraceRole::Nonparticipant,
        PrivacySeam::Notification,
    )
    .outcome = PrivacyOutcome::Notification {
        body: Some(FIXED_WAKE_BODY.to_owned()),
    };
    assert!(audit_privacy_trace(&notification_leak, &expected_ids, &sensitive_tokens).is_err());
}

#[test]
fn independent_checker_rejects_missing_or_duplicate_seam_coverage() {
    let (mut observations, expected_ids, sensitive_tokens) = safe_checker_trace();
    observations.pop();
    assert!(audit_privacy_trace(&observations, &expected_ids, &sensitive_tokens).is_err());

    let (mut observations, expected_ids, sensitive_tokens) = safe_checker_trace();
    observations.push(observations[0].clone());
    assert!(audit_privacy_trace(&observations, &expected_ids, &sensitive_tokens).is_err());
}
