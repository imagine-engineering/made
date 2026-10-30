use nostr_compat::generated_kinds::{EVENT_KINDS, PrivacyGates};
use nostr_compat::head::{HeadCandidate, select_head};
use nostr_compat::{
    CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent, TimestampPolicy,
    VerificationError, verify_signed_event,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
));
const MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/manifest.json"
));
const WIRE_TRACES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/wire-traces.json"
));
const PROTOCOL_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/catalogs/protocol.csv"
));

struct NipFixture {
    id: &'static str,
    document_file: &'static str,
    codec_source: &'static str,
    golden_test: &'static str,
    malformed_test: &'static str,
}

macro_rules! nip_fixture {
    ($id:literal, $file:literal, $source:expr, $golden:literal, $malformed:literal) => {
        NipFixture {
            id: $id,
            document_file: $file,
            codec_source: $source,
            golden_test: $golden,
            malformed_test: $malformed,
        }
    };
}

const IDENTITY_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/identity.rs"
));
const AGENT_CONFIG_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/agent_config.rs"
));
const AGENT_ACTIVITY_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/agent_activity.rs"
));
const COMMUNICATION_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/communication.rs"
));
const PROJECT_WORKFLOW_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/project_workflow.rs"
));
const PUSH_LEASE_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/buzz_nips/push_lease.rs"
));

const NIP_FIXTURES: &[NipFixture] = &[
    nip_fixture!(
        "NIP-AA",
        "NIP-AA.md",
        IDENTITY_SOURCE,
        "agent_authentication_round_trips_and_rejects_duplicate_credentials",
        "agent_authentication_round_trips_and_rejects_duplicate_credentials"
    ),
    nip_fixture!(
        "NIP-AE",
        "NIP-AE.md",
        AGENT_ACTIVITY_SOURCE,
        "engram_coordinate_and_encrypted_vector_match_nip_ae",
        "engram_body_rejects_duplicate_members_and_wrong_coordinate"
    ),
    nip_fixture!(
        "NIP-AM",
        "NIP-AM.md",
        AGENT_ACTIVITY_SOURCE,
        "turn_metric_enforces_owner_only_envelope_and_payload_rules",
        "turn_metric_enforces_owner_only_envelope_and_payload_rules"
    ),
    nip_fixture!(
        "NIP-AO",
        "NIP-AO.md",
        AGENT_ACTIVITY_SOURCE,
        "observer_frames_enforce_direction_privacy_and_redacted_payloads",
        "observer_unknown_frames_are_ignored_and_malformed_directions_fail"
    ),
    nip_fixture!(
        "NIP-AP",
        "NIP-AP.md",
        AGENT_CONFIG_SOURCE,
        "persona_and_team_privacy_envelopes_round_trip_and_fail_closed",
        "persona_and_team_privacy_envelopes_round_trip_and_fail_closed"
    ),
    nip_fixture!(
        "NIP-CW",
        "NIP-CW.md",
        COMMUNICATION_SOURCE,
        "channel_window_cursor_and_overlays_are_request_bound",
        "channel_window_cursor_and_overlays_are_request_bound"
    ),
    nip_fixture!(
        "NIP-DV",
        "NIP-DV.md",
        COMMUNICATION_SOURCE,
        "dm_visibility_is_relay_signed_owner_only_and_set_valued",
        "dm_visibility_is_relay_signed_owner_only_and_set_valued"
    ),
    nip_fixture!(
        "NIP-ER",
        "NIP-ER.md",
        COMMUNICATION_SOURCE,
        "reminder_envelope_and_plaintext_enforce_schedule_and_privacy",
        "reminder_envelope_and_plaintext_enforce_schedule_and_privacy"
    ),
    nip_fixture!(
        "NIP-GS",
        "NIP-GS.md",
        PROJECT_WORKFLOW_SOURCE,
        "git_signature_matches_published_hash_armor_and_signature_vector",
        "git_owner_attestation_vector_is_bound_and_reported_separately"
    ),
    nip_fixture!(
        "NIP-IA",
        "NIP-IA.md",
        IDENTITY_SOURCE,
        "archive_deltas_and_snapshot_match_nip_ia_vectors",
        "archive_codec_rejects_invalid_action_specific_and_consent_shapes"
    ),
    nip_fixture!(
        "NIP-MP",
        "NIP-MP.md",
        PROJECT_WORKFLOW_SOURCE,
        "project_codec_matches_every_nip_mp_ingest_fixture",
        "project_coordinates_preserve_colons_and_never_confer_repository_authority"
    ),
    nip_fixture!(
        "NIP-OA",
        "NIP-OA.md",
        IDENTITY_SOURCE,
        "owner_attestation_vector_round_trips_and_preserves_context_rules",
        "owner_attestation_vector_round_trips_and_preserves_context_rules"
    ),
    nip_fixture!(
        "NIP-PL",
        "NIP-PL.md",
        PUSH_LEASE_SOURCE,
        "active_lease_round_trips_narrow_filters_and_suppression",
        "plaintext_rejects_duplicate_unknown_and_cross_user_filter_state"
    ),
    nip_fixture!(
        "NIP-PMA",
        "NIP-PMA.md",
        AGENT_CONFIG_SOURCE,
        "private_envelope_enforces_cas_predecessors_and_signed_owner",
        "private_payload_rejects_versions_duplicates_and_invalid_tombstones"
    ),
    nip_fixture!(
        "NIP-RS",
        "NIP-RS.md",
        COMMUNICATION_SOURCE,
        "read_state_preserves_last_frontier_and_canonical_override_shapes",
        "read_state_merge_is_monotone_and_counters_never_wrap"
    ),
    nip_fixture!(
        "NIP-WP",
        "NIP-WP.md",
        PROJECT_WORKFLOW_SOURCE,
        "workspace_profile_codec_validates_image_sinks_and_clear_shape",
        "workspace_profile_codec_validates_image_sinks_and_clear_shape"
    ),
];

#[test]
fn buzz_nips_catalog_registers_every_document_and_vector_pair() {
    let expected = BTreeSet::from([
        "NIP-AA", "NIP-AE", "NIP-AM", "NIP-AO", "NIP-AP", "NIP-CW", "NIP-DV", "NIP-ER", "NIP-GS",
        "NIP-IA", "NIP-MP", "NIP-OA", "NIP-PL", "NIP-PMA", "NIP-RS", "NIP-WP",
    ]);
    let registered = NIP_FIXTURES
        .iter()
        .map(|fixture| fixture.id)
        .collect::<BTreeSet<_>>();
    assert_eq!(registered, expected);

    for fixture in NIP_FIXTURES {
        let document_prefix = format!(
            "DOC-{}-MD,custom_nip_document,{},,{},projects/buzz/docs/nips/{},",
            fixture.id, fixture.document_file, fixture.id, fixture.document_file
        );
        assert!(
            PROTOCOL_CATALOG
                .lines()
                .any(|line| line.starts_with(&document_prefix)),
            "{} has no pinned protocol document entry",
            fixture.id
        );
        for (class, test_name) in [
            ("golden", fixture.golden_test),
            ("malformed", fixture.malformed_test),
        ] {
            assert!(
                fixture.codec_source.contains(&format!("fn {test_name}(")),
                "{} has no registered {class} vector {test_name}",
                fixture.id
            );
        }
    }
}

#[test]
fn buzz_nips_frozen_event_manifest_matches_public_verification() {
    let manifest = json(MANIFEST);
    let events = json(EVENTS);
    assert_eq!(manifest["events_sha256"], sha256(EVENTS));

    for case in manifest["event_cases"].as_array().expect("event cases") {
        let name = case["event"].as_str().expect("event name");
        let result = signed_event(&events["events"][name])
            .and_then(|event| verify_fixture(&event).map_err(classify_verification));
        match case["expected"].as_str() {
            Some("accept") => assert!(result.is_ok(), "{}: {result:?}", case["id"]),
            Some("reject") => assert_eq!(
                result.expect_err("malformed fixture must fail"),
                case["expected_error"].as_str().expect("expected error"),
                "{}",
                case["id"]
            ),
            expected => panic!("unsupported event expectation {expected:?}"),
        }
    }
}

#[test]
fn buzz_nips_frozen_head_privacy_and_version_cases_match_registry() {
    let manifest = json(MANIFEST);
    let events = json(EVENTS);

    for case in manifest["replaceable_cases"]
        .as_array()
        .expect("replaceable cases")
    {
        let names = case["events"].as_array().expect("head events");
        let parsed = names
            .iter()
            .map(|name| signed_event(&events["events"][name.as_str().expect("event name")]))
            .collect::<Result<Vec<_>, _>>()
            .expect("valid head fixtures");
        let candidates = parsed
            .iter()
            .map(|event| HeadCandidate {
                id: event.claimed_id,
                event: &event.event,
            })
            .collect::<Vec<_>>();
        let winner = select_head(&candidates)
            .expect("one replacement coordinate")
            .expect("non-empty head");
        let expected =
            signed_event(&events["events"][case["winner"].as_str().expect("winner fixture")])
                .expect("valid winner");
        assert_eq!(winner.id, expected.claimed_id, "{}", case["id"]);
    }

    for case in manifest["privacy_cases"].as_array().expect("privacy cases") {
        let event = signed_event(&events["events"][case["event"].as_str().expect("event fixture")])
            .expect("valid privacy event");
        let reader =
            PublicKey::from_hex(case["reader"].as_str().expect("reader")).expect("valid reader");
        assert_eq!(
            registry_visible_to(&event.event, reader),
            case["visible"].as_bool().expect("visibility"),
            "{}",
            case["id"]
        );
    }

    for case in manifest["mixed_version_cases"]
        .as_array()
        .expect("mixed-version cases")
    {
        let expected_kinds = case["kinds"].as_array().expect("expected kinds");
        for (name, expected_kind) in case["events"]
            .as_array()
            .expect("mixed events")
            .iter()
            .zip(expected_kinds)
        {
            let event = signed_event(&events["events"][name.as_str().expect("event name")])
                .expect("valid mixed-version event");
            verify_fixture(&event).expect("mixed-version signature");
            assert_eq!(
                u64::from(event.event.kind),
                expected_kind.as_u64().expect("kind")
            );
        }
    }
}

#[test]
fn buzz_nips_frozen_wire_and_relay_artifacts_are_closed_and_immutable() {
    let manifest = json(MANIFEST);
    let events = json(EVENTS);
    let wire_traces = json(WIRE_TRACES);
    assert_eq!(manifest["wire_traces_sha256"], sha256(WIRE_TRACES));

    let known_events = events["events"].as_object().expect("event corpus");
    for trace in wire_traces["traces"].as_array().expect("wire traces") {
        assert!(
            trace["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("PROTO-WIRE-"))
        );
        for frame in trace["frames"].as_array().expect("frames") {
            let message = frame["message"].as_array().expect("wire message");
            for value in message {
                if let Some(name) = value.get("$event").and_then(Value::as_str) {
                    assert!(
                        known_events.contains_key(name),
                        "unknown event fixture {name}"
                    );
                }
                if let Some(name) = value.get("$event_id").and_then(Value::as_str) {
                    assert!(
                        known_events.contains_key(name),
                        "unknown event-id fixture {name}"
                    );
                }
            }
        }
    }

    for case in manifest["relay_trace_cases"]
        .as_array()
        .expect("relay trace cases")
    {
        let contents = relay_trace(case["file"].as_str().expect("relay filename"));
        assert_eq!(sha256(contents), case["sha256"], "{}", case["id"]);
        for line in contents.lines() {
            let record = json(line);
            assert_eq!(record["schema_version"], 1, "{}", case["id"]);
        }
    }
}

fn json(contents: &str) -> Value {
    serde_json::from_str(contents).expect("valid frozen JSON fixture")
}

fn sha256(contents: &str) -> String {
    hex::encode(Sha256::digest(contents.as_bytes()))
}

fn signed_event(value: &Value) -> Result<SignedEvent, &'static str> {
    let claimed_id = value["id"]
        .as_str()
        .ok_or("invalid_id")
        .and_then(|value| EventId::from_hex(value).map_err(|_| "invalid_id"))?;
    let public_key = value["pubkey"]
        .as_str()
        .ok_or("invalid_pubkey")
        .and_then(|value| PublicKey::from_hex(value).map_err(|_| "invalid_pubkey"))?;
    let created_at = value["created_at"].as_u64().ok_or("invalid_created_at")?;
    let kind = value["kind"]
        .as_u64()
        .and_then(|kind| u16::try_from(kind).ok())
        .ok_or("invalid_kind")?;
    let tags = serde_json::from_value(value["tags"].clone()).map_err(|_| "invalid_tags")?;
    let content = value["content"]
        .as_str()
        .ok_or("invalid_content")?
        .to_owned();
    let signature = value["sig"]
        .as_str()
        .ok_or("invalid_signature")
        .and_then(|value| EventSignature::from_hex(value).map_err(|_| "invalid_signature"))?;
    Ok(SignedEvent {
        claimed_id,
        event: CanonicalEvent::new(public_key, created_at, kind, tags, content),
        signature,
    })
}

fn verify_fixture(event: &SignedEvent) -> Result<(), VerificationError> {
    verify_signed_event(event, TimestampPolicy::Historical)
}

fn classify_verification(error: VerificationError) -> &'static str {
    match error {
        VerificationError::InvalidEventId { .. } => "invalid_id",
        VerificationError::InvalidPublicKey => "invalid_pubkey",
        VerificationError::MalformedSignature | VerificationError::InvalidSignature => {
            "invalid_signature"
        }
        _ => "invalid_event",
    }
}

fn registry_visible_to(event: &CanonicalEvent, reader: PublicKey) -> bool {
    let metadata = EVENT_KINDS
        .iter()
        .find(|metadata| metadata.value == u32::from(event.kind))
        .expect("fixture kind is registered");
    if metadata.privacy.is_community_visible() || event.public_key == reader {
        return true;
    }
    if metadata.privacy.contains(PrivacyGates::AUTHOR_ONLY) {
        return false;
    }
    if metadata.privacy.contains(PrivacyGates::RECIPIENT_GATED)
        && event.tags.iter().any(|tag| {
            tag.first().map(String::as_str) == Some("p")
                && tag.get(1).is_some_and(|value| value == &reader.to_hex())
        })
    {
        return true;
    }
    metadata.privacy.contains(PrivacyGates::AUTHOR_OR_SHARED)
        && event.tags.iter().any(|tag| {
            tag.first().map(String::as_str) == Some("shared")
                && tag.get(1).map(String::as_str) == Some("true")
        })
}

fn relay_trace(filename: &str) -> &'static str {
    match filename {
        "relay-good.jsonl" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.agents/specs/collaborative-workspace/fixtures/protocol/relay-good.jsonl"
        )),
        "relay-bad-host-channel.jsonl" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.agents/specs/collaborative-workspace/fixtures/protocol/relay-bad-host-channel.jsonl"
        )),
        "relay-bad-foreign-row.jsonl" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.agents/specs/collaborative-workspace/fixtures/protocol/relay-bad-foreign-row.jsonl"
        )),
        "relay-bad-coverage.jsonl" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.agents/specs/collaborative-workspace/fixtures/protocol/relay-bad-coverage.jsonl"
        )),
        _ => panic!("unregistered relay fixture {filename}"),
    }
}
