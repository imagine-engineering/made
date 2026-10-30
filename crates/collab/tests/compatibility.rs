use std::sync::Arc;

use axum::{
    Json,
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
};
use collab::compatibility::{
    CompatibilityAccess, CompatibilityOutcome, CompatibilityPolicy, CompatibilityReason,
    CompatibilityRequest, NostrCompatibilityFrame, RequestedProtocol, http_negotiate, http_router,
    negotiate_nostr,
};
use semver::Version;
use tower::ServiceExt as _;

fn request(
    client_id: &str,
    client_version: &str,
    access: CompatibilityAccess,
    protocols: &[(&str, u32)],
    features: &[&str],
) -> CompatibilityRequest {
    CompatibilityRequest::new(
        client_id,
        client_version,
        access,
        protocols
            .iter()
            .map(|(id, version)| RequestedProtocol::new(*id, *version))
            .collect(),
        features.iter().map(|value| (*value).to_owned()).collect(),
    )
}

#[tokio::test]
async fn supported_http_client_receives_write_admission() {
    let policy = Arc::new(CompatibilityPolicy::current());
    let request = request(
        "buzz-mobile",
        "0.0.0+1",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1), ("nip44-payload", 2)],
        &["channels", "direct-messages"],
    );

    let (status, Json(response)) =
        http_negotiate(State(Arc::clone(&policy)), Json(request.clone())).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response.outcome, CompatibilityOutcome::Supported);
    assert_eq!(response.error, None);
    assert_eq!(response.reason, None);
    assert_eq!(response.minimum_client_version.as_deref(), Some("0.0.0+1"));
    assert_eq!(response.schema.current_version, "20261026000100");
    assert_eq!(response.schema.minimum_version, "20261026000100");
    assert_eq!(response.schema.maximum_version, "20261026000100");
    let admission = policy.admit_write(&request).expect("write admission");
    assert_eq!(admission.policy_version(), 1);
    assert_eq!(admission.client_id(), "buzz-mobile");

    let route_response = http_router(policy)
        .oneshot(
            Request::post("/v1/collaboration/compatibility")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request).expect("serialize compatibility request"),
                ))
                .expect("build compatibility request"),
        )
        .await
        .expect("route compatibility request");
    assert_eq!(route_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn read_only_feature_cannot_produce_write_admission() {
    let policy = Arc::new(CompatibilityPolicy::current());
    let request = request(
        "buzz-web",
        "0.1.0",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["repository-browse"],
    );

    let (status, Json(response)) =
        http_negotiate(State(Arc::clone(&policy)), Json(request.clone())).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response.outcome, CompatibilityOutcome::ReadOnly);
    assert_eq!(response.reason, Some(CompatibilityReason::ReadOnlyFeature));
    assert!(policy.admit_write(&request).is_err());
}

#[tokio::test]
async fn incompatible_client_version_gets_upgrade_before_write() {
    let policy = Arc::new(CompatibilityPolicy::current());
    let request = request(
        "buzz-mobile",
        "0.0.0+2",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["channels"],
    );

    let (status, Json(response)) =
        http_negotiate(State(Arc::clone(&policy)), Json(request.clone())).await;

    assert_eq!(status, StatusCode::UPGRADE_REQUIRED);
    assert_eq!(response.outcome, CompatibilityOutcome::UpgradeRequired);
    assert_eq!(response.error.as_deref(), Some("upgrade_required"));
    assert_eq!(
        response.reason,
        Some(CompatibilityReason::ClientVersionUnsupported)
    );
    assert_eq!(response.maximum_client_version.as_deref(), Some("0.0.0+1"));
    assert!(policy.admit_write(&request).is_err());
}

#[tokio::test]
async fn unknown_feature_is_an_upgrade_error_not_canonical_state() {
    let policy = Arc::new(CompatibilityPolicy::current());
    let request = request(
        "zed-desktop",
        "1.16.2",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["future-canonical-writer"],
    );

    let (status, Json(response)) =
        http_negotiate(State(Arc::clone(&policy)), Json(request.clone())).await;

    assert_eq!(status, StatusCode::UPGRADE_REQUIRED);
    assert_eq!(response.reason, Some(CompatibilityReason::UnknownFeature));
    assert!(response.selected_features.is_empty());
    assert!(policy.admit_write(&request).is_err());
}

#[test]
fn nostr_negotiation_uses_closed_supported_and_upgrade_frames() {
    let policy = CompatibilityPolicy::current();
    let supported = request(
        "buzz-desktop",
        "0.5.11",
        CompatibilityAccess::Write,
        &[("nostr-ingress", 1)],
        &["messages"],
    );
    assert!(matches!(
        negotiate_nostr(&policy, &supported),
        NostrCompatibilityFrame::Ok { .. }
    ));

    let unsupported = request(
        "buzz-desktop",
        "0.5.10",
        CompatibilityAccess::Write,
        &[("nostr-ingress", 1)],
        &["messages"],
    );
    let NostrCompatibilityFrame::Closed { reason, response } =
        negotiate_nostr(&policy, &unsupported)
    else {
        panic!("unsupported Nostr client must close")
    };
    assert_eq!(reason, "upgrade-required: client-version-unsupported");
    assert_eq!(response.outcome, CompatibilityOutcome::UpgradeRequired);
}

#[test]
fn invalid_runtime_schema_and_malformed_requests_fail_closed() {
    let incompatible_policy =
        CompatibilityPolicy::for_runtime(Version::new(0, 44, 0), 20_260_824_000_500);
    let valid = request(
        "zed-desktop",
        "1.16.2",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["channels"],
    );
    assert_eq!(
        incompatible_policy.negotiate(&valid).reason,
        Some(CompatibilityReason::SchemaVersionUnsupported)
    );

    let malformed = CompatibilityRequest::new(
        "zed-desktop",
        "1.16.2",
        CompatibilityAccess::Write,
        vec![
            RequestedProtocol::new("collaboration-http", 1),
            RequestedProtocol::new("collaboration-http", 1),
        ],
        vec!["channels".to_owned()],
    );
    let response = CompatibilityPolicy::current().negotiate(&malformed);
    assert_eq!(response.http_status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.reason, Some(CompatibilityReason::InvalidRequest));

    let oversized_identity = CompatibilityRequest::new(
        "z".repeat(65),
        "1.16.2",
        CompatibilityAccess::Write,
        vec![RequestedProtocol::new("collaboration-http", 1)],
        vec!["channels".to_owned()],
    );
    let response = CompatibilityPolicy::current().negotiate(&oversized_identity);
    assert_eq!(response.client_id, "unknown");
    assert_eq!(response.reason, Some(CompatibilityReason::InvalidRequest));
}

#[test]
fn compatibility_requires_the_packaged_message_transport_schema() {
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../deploy/collaboration/migrations/manifest.json"
    ))
    .expect("migration manifest");
    let request = request(
        "zed-desktop",
        "1.16.2",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["channels"],
    );
    let response = CompatibilityPolicy::current().negotiate(&request);
    assert_eq!(
        Some(response.schema.current_version.as_str()),
        manifest["schemaCeiling"].as_str()
    );
    for schema_version in [
        20_260_825_000_100,
        20_260_930_000_100,
        20_261_010_000_200,
        20_261_010_000_300,
        20_261_010_000_400,
    ] {
        let previous = CompatibilityPolicy::for_runtime(Version::new(0, 44, 0), schema_version);
        assert_eq!(
            previous.negotiate(&request).reason,
            Some(CompatibilityReason::SchemaVersionUnsupported)
        );
        assert!(previous.admit_write(&request).is_err());
    }
}

#[test]
fn transport_schema_without_room_authorization_generation_rejects_writes() {
    let policy = CompatibilityPolicy::for_runtime(Version::new(0, 44, 0), 20_260_826_000_100);
    let request = request(
        "zed-desktop",
        "1.16.2",
        CompatibilityAccess::Write,
        &[("collaboration-http", 1)],
        &["channels"],
    );
    let response = policy.negotiate(&request);
    assert_eq!(response.http_status(), StatusCode::UPGRADE_REQUIRED);
    assert_eq!(
        response.reason,
        Some(CompatibilityReason::SchemaVersionUnsupported)
    );
    assert!(policy.admit_write(&request).is_err());
}
