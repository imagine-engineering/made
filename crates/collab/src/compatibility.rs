use std::{collections::BTreeSet, sync::Arc};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::post,
};
use semver::Version;
use serde::{Deserialize, Serialize};

pub const COMPATIBILITY_POLICY_VERSION: u32 = 1;
pub const COLLABORATION_HTTP_PROTOCOL_VERSION: u32 = 1;

const MAX_NEGOTIATION_BODY_BYTES: usize = 16 * 1024;
const MAX_IDENTIFIER_BYTES: usize = 64;
const MAX_VERSION_BYTES: usize = 32;
const MAX_PROTOCOLS: usize = 16;
const MAX_FEATURES: usize = 32;
const COLLAB_SERVICE_MINIMUM_VERSION: &str = "0.44.0";
const COLLAB_SERVICE_MAXIMUM_VERSION: &str = "0.44.0";
const COLLAB_SCHEMA_ID: &str = "canonical-collaboration-postgres";
const COLLAB_SCHEMA_MINIMUM_VERSION: u64 = 20_261_026_000_100;
const COLLAB_SCHEMA_MAXIMUM_VERSION: u64 = 20_261_026_000_100;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityAccess {
    Read,
    Write,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedProtocol {
    pub id: String,
    pub version: u32,
}

impl RequestedProtocol {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: id.into(),
            version,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityRequest {
    pub client_id: String,
    pub client_version: String,
    pub access: CompatibilityAccess,
    pub protocols: Vec<RequestedProtocol>,
    pub features: Vec<String>,
}

impl CompatibilityRequest {
    pub fn new(
        client_id: impl Into<String>,
        client_version: impl Into<String>,
        access: CompatibilityAccess,
        protocols: Vec<RequestedProtocol>,
        features: Vec<String>,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            client_version: client_version.into(),
            access,
            protocols,
            features,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityOutcome {
    Supported,
    ReadOnly,
    UpgradeRequired,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityReason {
    InvalidRequest,
    UnknownClient,
    ClientVersionUnsupported,
    ServiceVersionUnsupported,
    SchemaVersionUnsupported,
    ProtocolUnsupported,
    UnknownFeature,
    FeatureUnavailable,
    ReadOnlyFeature,
}

impl CompatibilityReason {
    const fn text(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid-request",
            Self::UnknownClient => "unknown-client",
            Self::ClientVersionUnsupported => "client-version-unsupported",
            Self::ServiceVersionUnsupported => "service-version-unsupported",
            Self::SchemaVersionUnsupported => "schema-version-unsupported",
            Self::ProtocolUnsupported => "protocol-unsupported",
            Self::UnknownFeature => "unknown-feature",
            Self::FeatureUnavailable => "feature-unavailable",
            Self::ReadOnlyFeature => "read-only-feature",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SchemaCompatibility {
    pub id: String,
    pub current_version: String,
    pub minimum_version: String,
    pub maximum_version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompatibilityResponse {
    pub policy_version: u32,
    pub outcome: CompatibilityOutcome,
    pub error: Option<String>,
    pub reason: Option<CompatibilityReason>,
    pub client_id: String,
    pub minimum_client_version: Option<String>,
    pub maximum_client_version: Option<String>,
    pub service_minimum_version: String,
    pub service_maximum_version: String,
    pub accepted_protocols: Vec<RequestedProtocol>,
    pub selected_features: Vec<String>,
    pub schema: SchemaCompatibility,
    pub retryable: bool,
}

impl CompatibilityResponse {
    pub fn http_status(&self) -> StatusCode {
        match (self.outcome, self.reason) {
            (CompatibilityOutcome::Supported | CompatibilityOutcome::ReadOnly, _) => StatusCode::OK,
            (CompatibilityOutcome::UpgradeRequired, Some(CompatibilityReason::InvalidRequest)) => {
                StatusCode::BAD_REQUEST
            }
            (CompatibilityOutcome::UpgradeRequired, _) => StatusCode::UPGRADE_REQUIRED,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityWriteAdmission {
    policy_version: u32,
    client_id: String,
}

impl CompatibilityWriteAdmission {
    pub const fn policy_version(&self) -> u32 {
        self.policy_version
    }

    pub fn client_id(&self) -> &str {
        &self.client_id
    }
}

#[derive(Clone, Debug)]
pub struct CompatibilityPolicy {
    service_version: Version,
    schema_version: u64,
}

impl Default for CompatibilityPolicy {
    fn default() -> Self {
        Self::current()
    }
}

impl CompatibilityPolicy {
    pub fn current() -> Self {
        Self {
            service_version: Version::new(0, 44, 0),
            schema_version: COLLAB_SCHEMA_MAXIMUM_VERSION,
        }
    }

    pub fn for_runtime(service_version: Version, schema_version: u64) -> Self {
        Self {
            service_version,
            schema_version,
        }
    }

    pub fn negotiate(&self, request: &CompatibilityRequest) -> CompatibilityResponse {
        let contract = client_contract(&request.client_id);
        if !valid_request_shape(request) {
            return self.response(
                request,
                contract,
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::InvalidRequest),
                Vec::new(),
            );
        }
        let Some(contract) = contract else {
            return self.response(
                request,
                None,
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::UnknownClient),
                Vec::new(),
            );
        };
        if self.service_version != Version::new(0, 44, 0) {
            return self.response(
                request,
                Some(contract),
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::ServiceVersionUnsupported),
                Vec::new(),
            );
        }
        if !(COLLAB_SCHEMA_MINIMUM_VERSION..=COLLAB_SCHEMA_MAXIMUM_VERSION)
            .contains(&self.schema_version)
        {
            return self.response(
                request,
                Some(contract),
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::SchemaVersionUnsupported),
                Vec::new(),
            );
        }
        if !supported_client_version(contract, &request.client_version) {
            return self.response(
                request,
                Some(contract),
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::ClientVersionUnsupported),
                Vec::new(),
            );
        }
        if request.protocols.iter().any(|protocol| {
            !contract
                .protocols
                .contains(&(protocol.id.as_str(), protocol.version))
        }) {
            return self.response(
                request,
                Some(contract),
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::ProtocolUnsupported),
                Vec::new(),
            );
        }

        let mut selected_features = Vec::with_capacity(request.features.len());
        let mut read_only = false;
        for feature in &request.features {
            let Some(feature_contract) = feature_contract(feature) else {
                return self.response(
                    request,
                    Some(contract),
                    CompatibilityOutcome::UpgradeRequired,
                    Some(CompatibilityReason::UnknownFeature),
                    Vec::new(),
                );
            };
            if !contract.features.contains(&feature_contract.id) {
                return self.response(
                    request,
                    Some(contract),
                    CompatibilityOutcome::UpgradeRequired,
                    Some(CompatibilityReason::FeatureUnavailable),
                    Vec::new(),
                );
            }
            read_only |= request.access == CompatibilityAccess::Write && !feature_contract.writes;
            selected_features.push(feature.clone());
        }

        self.response(
            request,
            Some(contract),
            if read_only {
                CompatibilityOutcome::ReadOnly
            } else {
                CompatibilityOutcome::Supported
            },
            read_only.then_some(CompatibilityReason::ReadOnlyFeature),
            selected_features,
        )
    }

    pub fn admit_write(
        &self,
        request: &CompatibilityRequest,
    ) -> Result<CompatibilityWriteAdmission, Box<CompatibilityResponse>> {
        if request.access != CompatibilityAccess::Write {
            return Err(Box::new(self.response(
                request,
                client_contract(&request.client_id),
                CompatibilityOutcome::UpgradeRequired,
                Some(CompatibilityReason::InvalidRequest),
                Vec::new(),
            )));
        }
        let response = self.negotiate(request);
        if response.outcome != CompatibilityOutcome::Supported {
            return Err(Box::new(response));
        }
        Ok(CompatibilityWriteAdmission {
            policy_version: response.policy_version,
            client_id: response.client_id,
        })
    }

    fn response(
        &self,
        _request: &CompatibilityRequest,
        contract: Option<&ClientContract>,
        outcome: CompatibilityOutcome,
        reason: Option<CompatibilityReason>,
        selected_features: Vec<String>,
    ) -> CompatibilityResponse {
        CompatibilityResponse {
            policy_version: COMPATIBILITY_POLICY_VERSION,
            outcome,
            error: (outcome == CompatibilityOutcome::UpgradeRequired)
                .then(|| "upgrade_required".to_owned()),
            reason,
            client_id: contract
                .map(|value| value.id.to_owned())
                .unwrap_or_else(|| "unknown".to_owned()),
            minimum_client_version: contract.map(|value| value.minimum_version.to_owned()),
            maximum_client_version: contract.map(|value| value.maximum_version.to_owned()),
            service_minimum_version: COLLAB_SERVICE_MINIMUM_VERSION.to_owned(),
            service_maximum_version: COLLAB_SERVICE_MAXIMUM_VERSION.to_owned(),
            accepted_protocols: contract
                .into_iter()
                .flat_map(|value| value.protocols)
                .map(|(id, version)| RequestedProtocol::new(*id, *version))
                .collect(),
            selected_features,
            schema: SchemaCompatibility {
                id: COLLAB_SCHEMA_ID.to_owned(),
                current_version: self.schema_version.to_string(),
                minimum_version: COLLAB_SCHEMA_MINIMUM_VERSION.to_string(),
                maximum_version: COLLAB_SCHEMA_MAXIMUM_VERSION.to_string(),
            },
            retryable: false,
        }
    }
}

pub fn http_router(policy: Arc<CompatibilityPolicy>) -> Router {
    Router::new()
        .route("/v1/collaboration/compatibility", post(http_negotiate))
        .layer(DefaultBodyLimit::max(MAX_NEGOTIATION_BODY_BYTES))
        .with_state(policy)
}

pub async fn http_negotiate(
    State(policy): State<Arc<CompatibilityPolicy>>,
    Json(request): Json<CompatibilityRequest>,
) -> (StatusCode, Json<CompatibilityResponse>) {
    let response = policy.negotiate(&request);
    (response.http_status(), Json(response))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "frame", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NostrCompatibilityFrame {
    Ok {
        response: CompatibilityResponse,
    },
    Closed {
        reason: String,
        response: CompatibilityResponse,
    },
}

pub fn negotiate_nostr(
    policy: &CompatibilityPolicy,
    request: &CompatibilityRequest,
) -> NostrCompatibilityFrame {
    let response = policy.negotiate(request);
    match response.outcome {
        CompatibilityOutcome::Supported => NostrCompatibilityFrame::Ok { response },
        CompatibilityOutcome::ReadOnly => NostrCompatibilityFrame::Closed {
            reason: "read-only: requested feature does not support writes".to_owned(),
            response,
        },
        CompatibilityOutcome::UpgradeRequired => NostrCompatibilityFrame::Closed {
            reason: format!(
                "upgrade-required: {}",
                response
                    .reason
                    .unwrap_or(CompatibilityReason::InvalidRequest)
                    .text()
            ),
            response,
        },
    }
}

fn valid_request_shape(request: &CompatibilityRequest) -> bool {
    valid_identifier(&request.client_id)
        && !request.client_version.is_empty()
        && request.client_version.len() <= MAX_VERSION_BYTES
        && Version::parse(&request.client_version).is_ok()
        && !request.protocols.is_empty()
        && request.protocols.len() <= MAX_PROTOCOLS
        && !request.features.is_empty()
        && request.features.len() <= MAX_FEATURES
        && request
            .protocols
            .iter()
            .all(|protocol| protocol.version > 0 && valid_identifier(&protocol.id))
        && request
            .features
            .iter()
            .all(|feature| valid_identifier(feature))
        && unique_protocols(&request.protocols)
        && unique_values(&request.features)
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
        })
}

fn unique_protocols(protocols: &[RequestedProtocol]) -> bool {
    let mut values = BTreeSet::new();
    protocols
        .iter()
        .all(|protocol| values.insert((protocol.id.as_str(), protocol.version)))
}

fn unique_values(values: &[String]) -> bool {
    let mut unique = BTreeSet::new();
    values.iter().all(|value| unique.insert(value.as_str()))
}

fn supported_client_version(contract: &ClientContract, value: &str) -> bool {
    if contract.minimum_version == contract.maximum_version {
        return value == contract.minimum_version;
    }
    let Ok(value) = Version::parse(value) else {
        return false;
    };
    let (Ok(minimum), Ok(maximum)) = (
        Version::parse(contract.minimum_version),
        Version::parse(contract.maximum_version),
    ) else {
        return false;
    };
    (minimum..=maximum).contains(&value)
}

struct ClientContract {
    id: &'static str,
    minimum_version: &'static str,
    maximum_version: &'static str,
    protocols: &'static [(&'static str, u32)],
    features: &'static [&'static str],
}

const COMMON_FEATURES: &[&str] = &[
    "invites",
    "communities",
    "channels",
    "messages",
    "direct-messages",
    "repository-browse",
    "repository-write",
    "review",
    "agents",
    "workflows",
    "moderation",
    "media",
];

const CLIENTS: &[ClientContract] = &[
    ClientContract {
        id: "zed-desktop",
        minimum_version: "1.16.2",
        maximum_version: "1.16.2",
        protocols: &[("collaboration-http", 1), ("zed-rpc", 68)],
        features: &[
            "invites",
            "communities",
            "channels",
            "messages",
            "direct-messages",
            "repository-browse",
            "repository-write",
            "review",
            "agents",
            "workflows",
            "moderation",
            "media",
            "pairing",
            "push",
            "huddles",
            "admin-lifecycle",
        ],
    },
    ClientContract {
        id: "buzz-desktop",
        minimum_version: "0.5.11",
        maximum_version: "0.5.11",
        protocols: &[("collaboration-http", 1), ("nostr-ingress", 1)],
        features: &[
            "invites",
            "communities",
            "channels",
            "messages",
            "direct-messages",
            "repository-browse",
            "repository-write",
            "review",
            "agents",
            "workflows",
            "moderation",
            "media",
            "pairing",
            "push",
            "huddles",
        ],
    },
    ClientContract {
        id: "buzz-mobile",
        minimum_version: "0.0.0+1",
        maximum_version: "0.0.0+1",
        protocols: &[
            ("collaboration-http", 1),
            ("nostr-ingress", 1),
            ("nip-ab", 1),
            ("nip44-payload", 2),
        ],
        features: &[
            "invites",
            "communities",
            "channels",
            "messages",
            "direct-messages",
            "media",
            "pairing",
            "push",
            "huddles",
        ],
    },
    ClientContract {
        id: "buzz-web",
        minimum_version: "0.1.0",
        maximum_version: "0.1.0",
        protocols: &[("collaboration-http", 1), ("nostr-ingress", 1)],
        features: &["invites", "communities", "repository-browse"],
    },
    ClientContract {
        id: "buzz-cli",
        minimum_version: "0.1.0",
        maximum_version: "0.1.0",
        protocols: &[("buzz-cli-forward", 1), ("collaboration-http", 1)],
        features: COMMON_FEATURES,
    },
    ClientContract {
        id: "buzz-admin-web",
        minimum_version: "0.1.0",
        maximum_version: "0.1.0",
        protocols: &[("collaboration-http", 1)],
        features: &["moderation", "admin-lifecycle"],
    },
];

fn client_contract(id: &str) -> Option<&'static ClientContract> {
    CLIENTS.iter().find(|contract| contract.id == id)
}

struct FeatureContract {
    id: &'static str,
    writes: bool,
}

const FEATURES: &[FeatureContract] = &[
    FeatureContract {
        id: "invites",
        writes: true,
    },
    FeatureContract {
        id: "communities",
        writes: true,
    },
    FeatureContract {
        id: "channels",
        writes: true,
    },
    FeatureContract {
        id: "messages",
        writes: true,
    },
    FeatureContract {
        id: "direct-messages",
        writes: true,
    },
    FeatureContract {
        id: "repository-browse",
        writes: false,
    },
    FeatureContract {
        id: "repository-write",
        writes: true,
    },
    FeatureContract {
        id: "review",
        writes: true,
    },
    FeatureContract {
        id: "agents",
        writes: true,
    },
    FeatureContract {
        id: "workflows",
        writes: true,
    },
    FeatureContract {
        id: "moderation",
        writes: true,
    },
    FeatureContract {
        id: "media",
        writes: true,
    },
    FeatureContract {
        id: "pairing",
        writes: true,
    },
    FeatureContract {
        id: "push",
        writes: true,
    },
    FeatureContract {
        id: "huddles",
        writes: true,
    },
    FeatureContract {
        id: "admin-lifecycle",
        writes: true,
    },
];

fn feature_contract(id: &str) -> Option<&'static FeatureContract> {
    FEATURES.iter().find(|contract| contract.id == id)
}
