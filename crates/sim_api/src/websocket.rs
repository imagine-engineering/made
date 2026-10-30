use std::{
    collections::{BTreeMap, VecDeque},
    sync::OnceLock,
};

use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::security::AuthenticatedPrincipal;

const WEBSOCKET_CATALOG_CSV: &str =
    include_str!("../../../.agents/specs/comfy-parity/catalogs/backend-websocket-events.csv");

pub const PREVIEW_IMAGE_EVENT_CODE: u32 = 1;
pub const UNENCODED_PREVIEW_IMAGE_EVENT_CODE: u32 = 2;
pub const TEXT_EVENT_CODE: u32 = 3;
pub const PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogDirection {
    ClientToServer,
    ServerToClient,
    InternalToServer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogWireKind {
    Json,
    Binary,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogAvailability {
    Active,
    Conditional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebSocketPayloadContract {
    FeatureFlagNegotiation,
    Status,
    FeatureFlags,
    Executing,
    Executed,
    ExecutionStart,
    ExecutionCached,
    ExecutionSuccess,
    ExecutionError,
    ExecutionInterrupted,
    Progress,
    ProgressState,
    Logs,
    AssetSeedStarted,
    AssetSeedProgress,
    AssetSeedPaused,
    AssetSeedResumed,
    AssetSeedFastComplete,
    AssetSeedEnrichComplete,
    AssetSeedCompleted,
    AssetSeedCancelled,
    AssetSeedError,
    PreviewImage,
    UnencodedPreviewImage,
    Text,
    PreviewImageWithMetadata,
}

impl WebSocketPayloadContract {
    const fn wire_kind(self) -> CatalogWireKind {
        match self {
            Self::PreviewImage | Self::Text | Self::PreviewImageWithMetadata => {
                CatalogWireKind::Binary
            }
            Self::UnencodedPreviewImage => CatalogWireKind::Internal,
            _ => CatalogWireKind::Json,
        }
    }

    const fn normative_schema(self) -> &'static str {
        match self {
            Self::FeatureFlagNegotiation => {
                r#"{type:"feature_flags",data:{feature:value}}; only the first text message is recognized for negotiation."#
            }
            Self::Status => r#"{type:"status",data:{status:{exec_info:{queue_remaining}},sid?}}"#,
            Self::FeatureFlags => r#"{type:"feature_flags",data:SERVER_FEATURE_FLAGS}"#,
            Self::Executing => "{node,display_node?,prompt_id?}; node null marks prompt terminal.",
            Self::Executed => "{node,display_node,output,prompt_id}",
            Self::ExecutionStart | Self::ExecutionSuccess => "{prompt_id,timestamp}",
            Self::ExecutionCached => "{nodes:[node_id],prompt_id,timestamp}",
            Self::ExecutionError => {
                "{prompt_id,node_id,node_type,executed,exception_message,exception_type,traceback,current_inputs,current_outputs,timestamp}"
            }
            Self::ExecutionInterrupted => "{prompt_id,node_id,node_type,executed,timestamp}",
            Self::Progress => "{value,max,prompt_id,node}",
            Self::ProgressState => {
                "{prompt_id,nodes:{node_id:{value,max,state,node_id,prompt_id,display_node_id,parent_node_id,real_node_id}}}"
            }
            Self::Logs => "{entries,size}",
            Self::AssetSeedStarted => "{roots?,progress?}",
            Self::AssetSeedProgress => "{scanned,total,created,skipped,...}",
            Self::AssetSeedPaused | Self::AssetSeedResumed => "{}",
            Self::AssetSeedFastComplete | Self::AssetSeedCompleted => "{scan summary}",
            Self::AssetSeedEnrichComplete => "{enrichment summary}",
            Self::AssetSeedCancelled => "{scan summary?}",
            Self::AssetSeedError => "{message,...}",
            Self::PreviewImage => {
                "big-endian uint32 event=1 + big-endian uint32 image_type (1 JPEG,2 PNG) + encoded image bytes"
            }
            Self::UnencodedPreviewImage => {
                "(image_format,PIL image,max_size); converted to wire PREVIEW_IMAGE rather than sending event 2."
            }
            Self::Text => {
                "big-endian uint32 event=3 + uint32 node_id byte length + UTF-8 node_id + arbitrary UTF-8/bytes text"
            }
            Self::PreviewImageWithMetadata => {
                "big-endian uint32 event=4 + uint32 metadata JSON byte length + UTF-8 JSON + JPEG/PNG bytes; metadata includes image_type and node identity."
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WebSocketEventDescriptor {
    pub feature_id: &'static str,
    pub direction: CatalogDirection,
    pub wire_kind: CatalogWireKind,
    pub event_type: &'static str,
    pub binary_code: Option<u32>,
    pub availability: CatalogAvailability,
    pub payload_contract: WebSocketPayloadContract,
}

pub const WEBSOCKET_EVENT_CATALOG: [WebSocketEventDescriptor; 26] = [
    descriptor(
        "COMFY-WS-001",
        CatalogDirection::ClientToServer,
        CatalogWireKind::Json,
        "feature_flags",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::FeatureFlagNegotiation,
    ),
    descriptor(
        "COMFY-WS-002",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "status",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::Status,
    ),
    descriptor(
        "COMFY-WS-003",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "feature_flags",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::FeatureFlags,
    ),
    descriptor(
        "COMFY-WS-004",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "executing",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::Executing,
    ),
    descriptor(
        "COMFY-WS-005",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "executed",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::Executed,
    ),
    descriptor(
        "COMFY-WS-006",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "execution_start",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ExecutionStart,
    ),
    descriptor(
        "COMFY-WS-007",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "execution_cached",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ExecutionCached,
    ),
    descriptor(
        "COMFY-WS-008",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "execution_success",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ExecutionSuccess,
    ),
    descriptor(
        "COMFY-WS-009",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "execution_error",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ExecutionError,
    ),
    descriptor(
        "COMFY-WS-010",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "execution_interrupted",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ExecutionInterrupted,
    ),
    descriptor(
        "COMFY-WS-011",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "progress",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::Progress,
    ),
    descriptor(
        "COMFY-WS-012",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "progress_state",
        None,
        CatalogAvailability::Active,
        WebSocketPayloadContract::ProgressState,
    ),
    descriptor(
        "COMFY-WS-013",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "logs",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::Logs,
    ),
    descriptor(
        "COMFY-WS-014",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.started",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedStarted,
    ),
    descriptor(
        "COMFY-WS-015",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.progress",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedProgress,
    ),
    descriptor(
        "COMFY-WS-016",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.paused",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedPaused,
    ),
    descriptor(
        "COMFY-WS-017",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.resumed",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedResumed,
    ),
    descriptor(
        "COMFY-WS-018",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.fast_complete",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedFastComplete,
    ),
    descriptor(
        "COMFY-WS-019",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.enrich_complete",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedEnrichComplete,
    ),
    descriptor(
        "COMFY-WS-020",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.completed",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedCompleted,
    ),
    descriptor(
        "COMFY-WS-021",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.cancelled",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedCancelled,
    ),
    descriptor(
        "COMFY-WS-022",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Json,
        "assets.seed.error",
        None,
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::AssetSeedError,
    ),
    descriptor(
        "COMFY-WS-023",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Binary,
        "PREVIEW_IMAGE",
        Some(PREVIEW_IMAGE_EVENT_CODE),
        CatalogAvailability::Active,
        WebSocketPayloadContract::PreviewImage,
    ),
    descriptor(
        "COMFY-WS-024",
        CatalogDirection::InternalToServer,
        CatalogWireKind::Internal,
        "UNENCODED_PREVIEW_IMAGE",
        Some(UNENCODED_PREVIEW_IMAGE_EVENT_CODE),
        CatalogAvailability::Active,
        WebSocketPayloadContract::UnencodedPreviewImage,
    ),
    descriptor(
        "COMFY-WS-025",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Binary,
        "TEXT",
        Some(TEXT_EVENT_CODE),
        CatalogAvailability::Active,
        WebSocketPayloadContract::Text,
    ),
    descriptor(
        "COMFY-WS-026",
        CatalogDirection::ServerToClient,
        CatalogWireKind::Binary,
        "PREVIEW_IMAGE_WITH_METADATA",
        Some(PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE),
        CatalogAvailability::Conditional,
        WebSocketPayloadContract::PreviewImageWithMetadata,
    ),
];

static DESCRIPTOR_CATALOG_VALIDATION: OnceLock<Result<(), String>> = OnceLock::new();
static NORMATIVE_WEBSOCKET_CATALOG: OnceLock<Result<Vec<NormativeWebSocketContract>, String>> =
    OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormativeWebSocketContract {
    pub feature_id: String,
    pub product: String,
    pub direction: CatalogDirection,
    pub wire_kind: CatalogWireKind,
    pub event_type: String,
    pub binary_code: Option<u32>,
    pub schema: String,
    pub availability: CatalogAvailability,
    pub evidence_level: String,
    pub confidence: String,
    pub trigger_success: String,
    pub error_recovery: String,
    pub ordering_concurrency: String,
    pub source_evidence: String,
    pub test_evidence: String,
    pub zed_status: String,
    pub parity_gap: String,
    pub acceptance_criteria: String,
}

pub fn normative_websocket_catalog()
-> Result<&'static [NormativeWebSocketContract], WebSocketContractError> {
    match NORMATIVE_WEBSOCKET_CATALOG.get_or_init(parse_normative_websocket_catalog) {
        Ok(catalog) => Ok(catalog),
        Err(error) => Err(WebSocketContractError::CatalogMismatch(error.clone())),
    }
}

fn parse_normative_websocket_catalog() -> Result<Vec<NormativeWebSocketContract>, String> {
    const EXPECTED_COLUMNS: [&str; 18] = [
        "feature_id",
        "product",
        "direction",
        "wire_kind",
        "event_type",
        "binary_code",
        "schema",
        "availability",
        "evidence_level",
        "confidence",
        "trigger_success",
        "error_recovery",
        "ordering_concurrency",
        "source_evidence",
        "test_evidence",
        "zed_status",
        "parity_gap",
        "acceptance_criteria",
    ];
    let rows = crate::http::parse_csv(WEBSOCKET_CATALOG_CSV).map_err(|error| error.to_string())?;
    let (header, data) = rows
        .split_first()
        .ok_or_else(|| "catalog has no header row".to_owned())?;
    if header.iter().map(String::as_str).collect::<Vec<_>>() != EXPECTED_COLUMNS {
        return Err("catalog columns do not match the normative WebSocket schema".into());
    }

    data.iter()
        .enumerate()
        .map(|(index, row)| {
            if row.len() != EXPECTED_COLUMNS.len() {
                return Err(format!("catalog row {} is incomplete", index + 2));
            }
            let value = |column: usize| row[column].clone();
            let direction = match row[2].as_str() {
                "client->server" => CatalogDirection::ClientToServer,
                "server->client" => CatalogDirection::ServerToClient,
                "internal->server" => CatalogDirection::InternalToServer,
                value => {
                    return Err(format!(
                        "catalog row {} has unknown direction {value}",
                        index + 2
                    ));
                }
            };
            let wire_kind = match row[3].as_str() {
                "JSON" => CatalogWireKind::Json,
                "binary" => CatalogWireKind::Binary,
                "internal event" => CatalogWireKind::Internal,
                value => {
                    return Err(format!(
                        "catalog row {} has unknown wire kind {value}",
                        index + 2
                    ));
                }
            };
            let binary_code = if row[5].is_empty() {
                None
            } else {
                Some(row[5].parse::<u32>().map_err(|error| {
                    format!("catalog row {} has invalid binary code: {error}", index + 2)
                })?)
            };
            let availability = match row[7].as_str() {
                "active" => CatalogAvailability::Active,
                "conditional" => CatalogAvailability::Conditional,
                value => {
                    return Err(format!(
                        "catalog row {} has unknown availability {value}",
                        index + 2
                    ));
                }
            };
            Ok(NormativeWebSocketContract {
                feature_id: value(0),
                product: value(1),
                direction,
                wire_kind,
                event_type: value(4),
                binary_code,
                schema: value(6),
                availability,
                evidence_level: value(8),
                confidence: value(9),
                trigger_success: value(10),
                error_recovery: value(11),
                ordering_concurrency: value(12),
                source_evidence: value(13),
                test_evidence: value(14),
                zed_status: value(15),
                parity_gap: value(16),
                acceptance_criteria: value(17),
            })
        })
        .collect()
}

fn validate_descriptor_catalog() -> Result<(), WebSocketContractError> {
    match DESCRIPTOR_CATALOG_VALIDATION.get_or_init(|| {
        let catalog = normative_websocket_catalog().map_err(|error| error.to_string())?;
        if catalog.len() != WEBSOCKET_EVENT_CATALOG.len() {
            return Err(format!(
                "catalog has {} rows; descriptors have {}",
                catalog.len(),
                WEBSOCKET_EVENT_CATALOG.len()
            ));
        }
        for (index, (contract, descriptor)) in catalog
            .iter()
            .zip(WEBSOCKET_EVENT_CATALOG.iter())
            .enumerate()
        {
            let fields_match = contract.feature_id == descriptor.feature_id
                && contract.direction == descriptor.direction
                && contract.wire_kind == descriptor.wire_kind
                && contract.event_type == descriptor.event_type
                && contract.binary_code == descriptor.binary_code
                && contract.availability == descriptor.availability
                && contract.schema == descriptor.payload_contract.normative_schema()
                && descriptor.wire_kind == descriptor.payload_contract.wire_kind();
            if !fields_match {
                return Err(format!(
                    "descriptor {} disagrees with catalog row {}",
                    descriptor.feature_id,
                    index + 2
                ));
            }
            let required_contract_fields = [
                ("product", contract.product.as_str()),
                ("schema", contract.schema.as_str()),
                ("trigger_success", contract.trigger_success.as_str()),
                ("error_recovery", contract.error_recovery.as_str()),
                (
                    "ordering_concurrency",
                    contract.ordering_concurrency.as_str(),
                ),
                ("acceptance_criteria", contract.acceptance_criteria.as_str()),
            ];
            if let Some((field, _)) = required_contract_fields
                .into_iter()
                .find(|(_, value)| value.trim().is_empty())
            {
                return Err(format!(
                    "catalog row {} has empty normative field {field}",
                    index + 2
                ));
            }
            if contract.product != "ComfyUI"
                || !contract.acceptance_criteria.contains(&contract.event_type)
                || !contract
                    .acceptance_criteria
                    .contains("no event may be consumed from or forwarded to another Comfy server")
            {
                return Err(format!(
                    "catalog row {} has an incomplete native acceptance contract",
                    index + 2
                ));
            }
        }
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(error) => Err(WebSocketContractError::CatalogMismatch(error.clone())),
    }
}

const fn descriptor(
    feature_id: &'static str,
    direction: CatalogDirection,
    wire_kind: CatalogWireKind,
    event_type: &'static str,
    binary_code: Option<u32>,
    availability: CatalogAvailability,
    payload_contract: WebSocketPayloadContract,
) -> WebSocketEventDescriptor {
    WebSocketEventDescriptor {
        feature_id,
        direction,
        wire_kind,
        event_type,
        binary_code,
        availability,
        payload_contract,
    }
}

pub fn websocket_event_descriptor(event_type: &str) -> Option<&'static WebSocketEventDescriptor> {
    WEBSOCKET_EVENT_CATALOG
        .iter()
        .find(|descriptor| descriptor.event_type == event_type)
}

pub fn websocket_event_descriptor_by_id(
    feature_id: &str,
) -> Option<&'static WebSocketEventDescriptor> {
    WEBSOCKET_EVENT_CATALOG
        .iter()
        .find(|descriptor| descriptor.feature_id == feature_id)
}

pub fn websocket_binary_event_descriptor(
    event_code: u32,
) -> Option<&'static WebSocketEventDescriptor> {
    WEBSOCKET_EVENT_CATALOG
        .iter()
        .find(|descriptor| descriptor.binary_code == Some(event_code))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ClientId(String);

impl ClientId {
    pub fn new(value: impl Into<String>) -> Result<Self, WebSocketContractError> {
        let value = value.into();
        if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
            return Err(WebSocketContractError::InvalidClientId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EventTarget {
    Broadcast,
    Client(ClientId),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeEventSource {
    Runtime,
    AssetSeeder,
    TerminalService,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventAssociation {
    pub prompt_id: Option<String>,
    pub node_id: Option<String>,
    pub attempt_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeJsonEvent {
    pub sequence: u64,
    pub event_type: String,
    pub data: Value,
    pub target: EventTarget,
    pub source: NativeEventSource,
    pub association: EventAssociation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreviewImageFormat {
    Jpeg,
    Png,
}

impl PreviewImageFormat {
    pub fn event_code(self) -> u32 {
        match self {
            Self::Jpeg => 1,
            Self::Png => 2,
        }
    }

    pub fn metadata_name(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Png => "png",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativePreviewEvent {
    pub sequence: u64,
    pub format: PreviewImageFormat,
    pub encoded_image: Vec<u8>,
    pub target: EventTarget,
    pub source: NativeEventSource,
    pub association: EventAssociation,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReconnectJsonEvent {
    pub event_type: String,
    pub data: Value,
    pub association: EventAssociation,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReconnectProjection {
    pub queue_remaining: u64,
    pub current_execution: Vec<ReconnectJsonEvent>,
    pub history_reconciliation: Vec<ReconnectJsonEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutboundWireKind {
    Text,
    Binary,
    Close,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutboundMessage {
    pub delivery_sequence: u64,
    pub source_sequence: Option<u64>,
    pub event_type: String,
    pub wire_kind: OutboundWireKind,
    pub payload: Vec<u8>,
    pub association: EventAssociation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishDisposition {
    Published,
    Duplicate,
    Stale,
    Suppressed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishReport {
    pub disposition: PublishDisposition,
    pub delivered_clients: usize,
    pub backpressured_clients: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticKind {
    MalformedInput,
    UnknownEvent,
    IgnoredNegotiation,
    StaleEvent,
    DuplicateEvent,
    Backpressure,
    FeatureDisabled,
    NotSubscribed,
    Disconnected,
    Shutdown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebSocketDiagnostic {
    pub kind: DiagnosticKind,
    pub client_id: Option<ClientId>,
    pub event_type: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebSocketLimits {
    pub max_message_bytes: usize,
    pub max_queued_messages_per_client: usize,
    pub max_diagnostics: usize,
    pub max_clients: usize,
    pub max_server_features: usize,
    pub max_source_sequence_scopes: usize,
}

impl Default for WebSocketLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: 8 * 1024 * 1024,
            max_queued_messages_per_client: 256,
            max_diagnostics: 256,
            max_clients: 256,
            max_server_features: 64,
            max_source_sequence_scopes: 1024,
        }
    }
}

impl WebSocketLimits {
    fn validate(&self) -> Result<(), WebSocketContractError> {
        if self.max_message_bytes < 16
            || self.max_queued_messages_per_client == 0
            || self.max_diagnostics == 0
            || self.max_clients == 0
            || self.max_server_features == 0
            || self.max_source_sequence_scopes == 0
        {
            return Err(WebSocketContractError::InvalidLimits);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FragmentKind {
    Text,
    Binary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputFragment {
    pub kind: FragmentKind,
    pub bytes: Vec<u8>,
    pub final_fragment: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputReport {
    pub complete_messages: usize,
    pub decoded_values: usize,
    pub feature_negotiated: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum WebSocketContractError {
    #[error("native WebSocket descriptors do not match the normative catalog: {0}")]
    CatalogMismatch(String),
    #[error("the WebSocket client identity is invalid")]
    InvalidClientId,
    #[error("WebSocket limits must be non-zero and permit a frame header")]
    InvalidLimits,
    #[error("WebSocket client {0} is not connected")]
    ClientNotConnected(String),
    #[error("WebSocket client {0} is already connected")]
    ClientAlreadyConnected(String),
    #[error("WebSocket client {0} is owned by a different authenticated principal")]
    PrincipalMismatch(String),
    #[error("the WebSocket reconnect projection cannot fit in the bounded client queue")]
    ReconnectProjectionTooLarge,
    #[error("the native WebSocket client limit was reached")]
    TooManyClients,
    #[error("the native WebSocket server-feature limit was reached")]
    TooManyServerFeatures,
    #[error("the native WebSocket source-sequence scope limit was reached")]
    TooManySourceSequenceScopes,
    #[error("the native WebSocket host is shutting down")]
    Shutdown,
    #[error("unknown WebSocket event type {0}")]
    UnknownEvent(String),
    #[error("WebSocket event {0} is not legal on this wire path")]
    IllegalWireEvent(String),
    #[error("malformed JSON WebSocket event: {0}")]
    MalformedJson(String),
    #[error("invalid payload for WebSocket event {event_type}: {reason}")]
    InvalidPayload { event_type: String, reason: String },
    #[error("malformed binary WebSocket event: {0}")]
    MalformedBinary(String),
    #[error("WebSocket message exceeds the configured byte limit")]
    MessageTooLarge,
    #[error("fragment kind changed before the message completed")]
    FragmentKindChanged,
    #[error("event association does not match its payload")]
    AssociationMismatch,
    #[error("WebSocket event {event_type} cannot originate from {event_source:?}")]
    InvalidEventSource {
        event_type: String,
        event_source: NativeEventSource,
    },
    #[error("WebSocket event {0} has an illegal client target")]
    InvalidEventTarget(String),
    #[error("source event sequences must be non-zero")]
    ZeroSequence,
}

#[derive(Clone, Debug)]
struct InputAssembler {
    kind: Option<FragmentKind>,
    bytes: Vec<u8>,
}

impl InputAssembler {
    fn new() -> Self {
        Self {
            kind: None,
            bytes: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.kind = None;
        self.bytes.clear();
    }

    fn push(
        &mut self,
        fragment: InputFragment,
        max_message_bytes: usize,
    ) -> Result<Option<(FragmentKind, Vec<u8>)>, WebSocketContractError> {
        if self.kind.is_some_and(|kind| kind != fragment.kind) {
            self.reset();
            return Err(WebSocketContractError::FragmentKindChanged);
        }
        if self.bytes.len().saturating_add(fragment.bytes.len()) > max_message_bytes {
            self.reset();
            return Err(WebSocketContractError::MessageTooLarge);
        }
        self.kind = Some(fragment.kind);
        self.bytes.extend(fragment.bytes);
        if !fragment.final_fragment {
            return Ok(None);
        }
        let kind = self.kind.take().ok_or_else(|| {
            WebSocketContractError::MalformedBinary("fragment has no message kind".into())
        })?;
        Ok(Some((kind, std::mem::take(&mut self.bytes))))
    }
}

#[derive(Clone, Debug)]
struct ClientSession {
    principal: AuthenticatedPrincipal,
    connected: bool,
    negotiation_seen: bool,
    features: BTreeMap<String, Value>,
    queue: VecDeque<OutboundMessage>,
    assembler: InputAssembler,
    logs_subscribed: bool,
}

impl ClientSession {
    fn new(principal: AuthenticatedPrincipal) -> Self {
        Self {
            principal,
            connected: true,
            negotiation_seen: false,
            features: BTreeMap::new(),
            queue: VecDeque::new(),
            assembler: InputAssembler::new(),
            logs_subscribed: false,
        }
    }

    fn supports_preview_metadata(&self) -> bool {
        self.features
            .get("supports_preview_metadata")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourceSequenceScope {
    source: NativeEventSource,
    attempt_id: Option<String>,
    prompt_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceSequenceState {
    last_sequence: u64,
    active: bool,
    last_touched: u64,
}

impl SourceSequenceScope {
    fn new(source: NativeEventSource, association: &EventAssociation) -> Self {
        Self {
            source,
            attempt_id: association.attempt_id.clone(),
            prompt_id: association.prompt_id.clone(),
        }
    }
}

pub struct NativeWebSocketEventBus {
    limits: WebSocketLimits,
    clients: BTreeMap<ClientId, ClientSession>,
    server_features: BTreeMap<String, Value>,
    diagnostics: VecDeque<WebSocketDiagnostic>,
    asset_seeder_enabled: bool,
    source_sequences: BTreeMap<SourceSequenceScope, SourceSequenceState>,
    next_source_touch: u64,
    next_delivery_sequence: u64,
    shutdown: bool,
}

impl NativeWebSocketEventBus {
    pub fn new(limits: WebSocketLimits) -> Result<Self, WebSocketContractError> {
        validate_descriptor_catalog()?;
        limits.validate()?;
        Ok(Self {
            limits,
            clients: BTreeMap::new(),
            server_features: BTreeMap::from([(
                "supports_preview_metadata".into(),
                Value::Bool(true),
            )]),
            diagnostics: VecDeque::new(),
            asset_seeder_enabled: false,
            source_sequences: BTreeMap::new(),
            next_source_touch: 1,
            next_delivery_sequence: 1,
            shutdown: false,
        })
    }

    pub fn set_server_feature(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), WebSocketContractError> {
        let name = name.into();
        if !self.server_features.contains_key(&name)
            && self.server_features.len() == self.limits.max_server_features
        {
            return Err(WebSocketContractError::TooManyServerFeatures);
        }
        self.server_features.insert(name, value);
        Ok(())
    }

    pub fn set_asset_seeder_enabled(&mut self, enabled: bool) {
        self.asset_seeder_enabled = enabled;
    }

    pub fn set_log_subscription(
        &mut self,
        client_id: &ClientId,
        subscribed: bool,
    ) -> Result<(), WebSocketContractError> {
        self.connected_client_mut(client_id)?.logs_subscribed = subscribed;
        Ok(())
    }

    #[cfg(test)]
    pub fn connect(
        &mut self,
        client_id: ClientId,
        projection: ReconnectProjection,
    ) -> Result<(), WebSocketContractError> {
        let principal = AuthenticatedPrincipal {
            identity: format!("test:{}", client_id.as_str()),
            scopes: Default::default(),
        };
        let session_id = client_id.clone();
        self.connect_authenticated_with_session_id(client_id, session_id, principal, projection)
    }

    pub fn connect_authenticated(
        &mut self,
        client_id: ClientId,
        principal: AuthenticatedPrincipal,
        projection: ReconnectProjection,
    ) -> Result<(), WebSocketContractError> {
        let session_id = client_id.clone();
        self.connect_authenticated_with_session_id(client_id, session_id, principal, projection)
    }

    pub fn connect_authenticated_with_session_id(
        &mut self,
        client_id: ClientId,
        session_id: ClientId,
        principal: AuthenticatedPrincipal,
        projection: ReconnectProjection,
    ) -> Result<(), WebSocketContractError> {
        if self.shutdown {
            return Err(WebSocketContractError::Shutdown);
        }
        if let Some(client) = self.clients.get(&client_id) {
            if client.principal.identity != principal.identity {
                return Err(WebSocketContractError::PrincipalMismatch(
                    client_id.as_str().into(),
                ));
            }
            if client.connected {
                return Err(WebSocketContractError::ClientAlreadyConnected(
                    client_id.as_str().into(),
                ));
            }
        }

        let eviction = if !self.clients.contains_key(&client_id)
            && self.clients.len() >= self.limits.max_clients
        {
            let disconnected = self
                .clients
                .iter()
                .find_map(|(client_id, client)| (!client.connected).then_some(client_id.clone()));
            if disconnected.is_none() {
                return Err(WebSocketContractError::TooManyClients);
            }
            disconnected
        } else {
            None
        };

        let previous_delivery_sequence = self.next_delivery_sequence;
        let previous_diagnostics = self.diagnostics.clone();
        let previous_session = self.clients.remove(&client_id);
        let evicted_session = eviction.and_then(|evicted_id| {
            self.clients
                .remove(&evicted_id)
                .map(|session| (evicted_id, session))
        });
        self.clients
            .insert(client_id.clone(), ClientSession::new(principal));

        let connection_result = (|| {
            let mut status_data = json!({
                "status": {"exec_info": {"queue_remaining": projection.queue_remaining}},
                "sid": session_id.as_str(),
            });
            validate_json_event("status", &status_data)?;
            require_reconnect_queue_space(self.enqueue_json_for_client(
                &client_id,
                None,
                "status",
                std::mem::take(&mut status_data),
                EventAssociation::default(),
            )?)?;
            for event in projection
                .current_execution
                .into_iter()
                .chain(projection.history_reconciliation)
            {
                validate_json_event(&event.event_type, &event.data)?;
                validate_association(&event.data, &event.association)?;
                require_reconnect_queue_space(self.enqueue_json_for_client(
                    &client_id,
                    None,
                    &event.event_type,
                    event.data,
                    event.association,
                )?)?;
            }
            Ok(())
        })();
        if let Err(error) = connection_result {
            self.clients.remove(&client_id);
            if let Some(previous_session) = previous_session {
                self.clients.insert(client_id, previous_session);
            }
            if let Some((evicted_id, evicted_session)) = evicted_session {
                self.clients.insert(evicted_id, evicted_session);
            }
            self.next_delivery_sequence = previous_delivery_sequence;
            self.diagnostics = previous_diagnostics;
            return Err(error);
        }
        Ok(())
    }

    pub fn authenticated_principal(
        &self,
        client_id: &ClientId,
    ) -> Result<&AuthenticatedPrincipal, WebSocketContractError> {
        self.clients
            .get(client_id)
            .filter(|client| client.connected)
            .map(|client| &client.principal)
            .ok_or_else(|| WebSocketContractError::ClientNotConnected(client_id.as_str().into()))
    }

    pub fn disconnect(&mut self, client_id: &ClientId) -> bool {
        let Some(client) = self.clients.get_mut(client_id) else {
            return false;
        };
        client.connected = false;
        client.negotiation_seen = false;
        client.features.clear();
        client.queue.clear();
        client.assembler.reset();
        true
    }

    pub fn forget_client(&mut self, client_id: &ClientId) -> bool {
        self.clients.remove(client_id).is_some()
    }

    pub fn cancel_delivery(&mut self, client_id: &ClientId) -> bool {
        let Some(client) = self.clients.get_mut(client_id) else {
            return false;
        };
        client.queue.clear();
        client.assembler.reset();
        true
    }

    pub fn process_input_fragment(
        &mut self,
        client_id: &ClientId,
        fragment: InputFragment,
    ) -> Result<InputReport, WebSocketContractError> {
        if self.shutdown {
            return Err(WebSocketContractError::Shutdown);
        }
        let max_message_bytes = self.limits.max_message_bytes;
        let complete = {
            let client = self.connected_client_mut(client_id)?;
            client.assembler.push(fragment, max_message_bytes)
        };
        let complete = match complete {
            Ok(complete) => complete,
            Err(error) => {
                self.record_diagnostic(WebSocketDiagnostic {
                    kind: DiagnosticKind::MalformedInput,
                    client_id: Some(client_id.clone()),
                    event_type: None,
                    detail: error.to_string(),
                });
                return Err(error);
            }
        };
        let Some((kind, bytes)) = complete else {
            return Ok(InputReport {
                complete_messages: 0,
                decoded_values: 0,
                feature_negotiated: false,
            });
        };
        match kind {
            FragmentKind::Text => self.process_text_message(client_id, &bytes),
            FragmentKind::Binary => {
                match decode_binary_message(&bytes, self.limits.max_message_bytes) {
                    Ok(_) => self.record_diagnostic(WebSocketDiagnostic {
                        kind: DiagnosticKind::UnknownEvent,
                        client_id: Some(client_id.clone()),
                        event_type: None,
                        detail: "clients do not send cataloged binary events".into(),
                    }),
                    Err(error) => self.record_diagnostic(WebSocketDiagnostic {
                        kind: DiagnosticKind::MalformedInput,
                        client_id: Some(client_id.clone()),
                        event_type: None,
                        detail: error.to_string(),
                    }),
                }
                Ok(InputReport {
                    complete_messages: 1,
                    decoded_values: 0,
                    feature_negotiated: false,
                })
            }
        }
    }

    pub(crate) fn publish_json(
        &mut self,
        event: NativeJsonEvent,
    ) -> Result<PublishReport, WebSocketContractError> {
        self.ensure_publishing(event.sequence)?;
        let descriptor = websocket_event_descriptor(&event.event_type)
            .ok_or_else(|| WebSocketContractError::UnknownEvent(event.event_type.clone()))?;
        if descriptor.wire_kind != CatalogWireKind::Json
            || descriptor.direction != CatalogDirection::ServerToClient
        {
            return Err(WebSocketContractError::IllegalWireEvent(event.event_type));
        }
        validate_event_source(&event.event_type, event.source)?;
        validate_event_target(&event.event_type, &event.target)?;
        validate_json_event(&event.event_type, &event.data)?;
        validate_association(&event.data, &event.association)?;
        let sequence_scope = SourceSequenceScope::new(event.source, &event.association);
        let disposition = self.sequence_disposition(&sequence_scope, event.sequence);
        if disposition != PublishDisposition::Published {
            self.record_sequence_diagnostic(
                disposition,
                &sequence_scope,
                event.sequence,
                &event.event_type,
            );
            return Ok(PublishReport {
                disposition,
                delivered_clients: 0,
                backpressured_clients: 0,
            });
        }
        self.record_source_sequence(
            sequence_scope,
            event.sequence,
            is_terminal(&event.event_type),
        )?;
        if event.event_type.starts_with("assets.seed.") && !self.asset_seeder_enabled {
            self.record_diagnostic(WebSocketDiagnostic {
                kind: DiagnosticKind::FeatureDisabled,
                client_id: None,
                event_type: Some(event.event_type),
                detail: "native asset seeder events are disabled".into(),
            });
            return Ok(PublishReport {
                disposition: PublishDisposition::Suppressed,
                delivered_clients: 0,
                backpressured_clients: 0,
            });
        }
        let recipients = self.recipients(&event.target);
        if event.event_type == "logs"
            && recipients.first().is_some_and(|client_id| {
                self.clients
                    .get(client_id)
                    .is_some_and(|client| !client.logs_subscribed)
            })
        {
            self.record_diagnostic(WebSocketDiagnostic {
                kind: DiagnosticKind::NotSubscribed,
                client_id: recipients.first().cloned(),
                event_type: Some(event.event_type),
                detail: "terminal logs require an active per-client subscription".into(),
            });
            return Ok(PublishReport {
                disposition: PublishDisposition::Suppressed,
                delivered_clients: 0,
                backpressured_clients: 0,
            });
        }
        let mut delivered_clients = 0;
        let mut backpressured_clients = 0;
        for client_id in recipients {
            let outcome = self.enqueue_json_for_client(
                &client_id,
                Some(event.sequence),
                &event.event_type,
                event.data.clone(),
                event.association.clone(),
            )?;
            match outcome {
                QueueOutcome::Queued => delivered_clients += 1,
                QueueOutcome::Backpressured => backpressured_clients += 1,
            }
        }
        Ok(PublishReport {
            disposition,
            delivered_clients,
            backpressured_clients,
        })
    }

    pub fn publish_preview(
        &mut self,
        event: NativePreviewEvent,
    ) -> Result<PublishReport, WebSocketContractError> {
        self.ensure_publishing(event.sequence)?;
        validate_event_source("PREVIEW_IMAGE", event.source)?;
        if !matches!(event.target, EventTarget::Client(_)) {
            return Err(WebSocketContractError::InvalidEventTarget(
                "PREVIEW_IMAGE".into(),
            ));
        }
        validate_preview_image(event.format, &event.encoded_image)?;
        if event.association.node_id.is_none() {
            return Err(WebSocketContractError::InvalidPayload {
                event_type: "PREVIEW_IMAGE".into(),
                reason: "a native preview must be associated with a node".into(),
            });
        }
        let sequence_scope = SourceSequenceScope::new(event.source, &event.association);
        let disposition = self.sequence_disposition(&sequence_scope, event.sequence);
        if disposition != PublishDisposition::Published {
            self.record_sequence_diagnostic(
                disposition,
                &sequence_scope,
                event.sequence,
                "PREVIEW_IMAGE",
            );
            return Ok(PublishReport {
                disposition,
                delivered_clients: 0,
                backpressured_clients: 0,
            });
        }
        self.record_source_sequence(sequence_scope, event.sequence, false)?;
        let recipients = self.recipients(&event.target);
        let mut delivered_clients = 0;
        let mut backpressured_clients = 0;
        for client_id in recipients {
            let supports_metadata = self
                .clients
                .get(&client_id)
                .is_some_and(ClientSession::supports_preview_metadata);
            let (event_type, payload) = if supports_metadata {
                (
                    "PREVIEW_IMAGE_WITH_METADATA",
                    encode_preview_with_metadata(
                        event.format,
                        &event.encoded_image,
                        &event.association,
                        event.metadata.clone(),
                    )?,
                )
            } else {
                (
                    "PREVIEW_IMAGE",
                    encode_preview_image(event.format, &event.encoded_image)?,
                )
            };
            let delivery_sequence = self.take_delivery_sequence();
            let message = OutboundMessage {
                delivery_sequence,
                source_sequence: Some(event.sequence),
                event_type: event_type.into(),
                wire_kind: OutboundWireKind::Binary,
                payload,
                association: event.association.clone(),
            };
            match self.enqueue_for_client(&client_id, message)? {
                QueueOutcome::Queued => delivered_clients += 1,
                QueueOutcome::Backpressured => backpressured_clients += 1,
            }
        }
        Ok(PublishReport {
            disposition,
            delivered_clients,
            backpressured_clients,
        })
    }

    pub fn publish_text(
        &mut self,
        sequence: u64,
        node_id: &str,
        text: &[u8],
        target: EventTarget,
        association: EventAssociation,
    ) -> Result<PublishReport, WebSocketContractError> {
        self.ensure_publishing(sequence)?;
        if association.node_id.as_deref() != Some(node_id) {
            return Err(WebSocketContractError::AssociationMismatch);
        }
        let sequence_scope = SourceSequenceScope::new(NativeEventSource::Runtime, &association);
        let disposition = self.sequence_disposition(&sequence_scope, sequence);
        if disposition != PublishDisposition::Published {
            self.record_sequence_diagnostic(disposition, &sequence_scope, sequence, "TEXT");
            return Ok(PublishReport {
                disposition,
                delivered_clients: 0,
                backpressured_clients: 0,
            });
        }
        let payload = encode_text_message(node_id, text, self.limits.max_message_bytes)?;
        self.record_source_sequence(sequence_scope, sequence, false)?;
        let recipients = self.recipients(&target);
        let mut delivered_clients = 0;
        let mut backpressured_clients = 0;
        for client_id in recipients {
            let delivery_sequence = self.take_delivery_sequence();
            let message = OutboundMessage {
                delivery_sequence,
                source_sequence: Some(sequence),
                event_type: "TEXT".into(),
                wire_kind: OutboundWireKind::Binary,
                payload: payload.clone(),
                association: association.clone(),
            };
            match self.enqueue_for_client(&client_id, message)? {
                QueueOutcome::Queued => delivered_clients += 1,
                QueueOutcome::Backpressured => backpressured_clients += 1,
            }
        }
        Ok(PublishReport {
            disposition,
            delivered_clients,
            backpressured_clients,
        })
    }

    pub fn drain_client(
        &mut self,
        client_id: &ClientId,
    ) -> Result<Vec<OutboundMessage>, WebSocketContractError> {
        let client = self.connected_client_mut(client_id)?;
        Ok(client.queue.drain(..).collect())
    }

    pub fn take_diagnostics(&mut self) -> Vec<WebSocketDiagnostic> {
        self.diagnostics.drain(..).collect()
    }

    pub fn shutdown(&mut self, reason: impl Into<String>) {
        if self.shutdown {
            return;
        }
        self.shutdown = true;
        let reason = reason.into();
        let clients: Vec<ClientId> = self
            .clients
            .iter()
            .filter_map(|(client_id, client)| client.connected.then_some(client_id.clone()))
            .collect();
        for client_id in clients {
            let delivery_sequence = self.take_delivery_sequence();
            if let Some(client) = self.clients.get_mut(&client_id) {
                client.queue.clear();
                client.queue.push_back(OutboundMessage {
                    delivery_sequence,
                    source_sequence: None,
                    event_type: "close".into(),
                    wire_kind: OutboundWireKind::Close,
                    payload: close_payload(1001, &reason),
                    association: EventAssociation::default(),
                });
            }
        }
        self.record_diagnostic(WebSocketDiagnostic {
            kind: DiagnosticKind::Shutdown,
            client_id: None,
            event_type: None,
            detail: reason,
        });
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown
    }

    fn process_text_message(
        &mut self,
        client_id: &ClientId,
        bytes: &[u8],
    ) -> Result<InputReport, WebSocketContractError> {
        let mut values = Vec::new();
        let stream = serde_json::Deserializer::from_slice(bytes).into_iter::<Value>();
        for value in stream {
            match value {
                Ok(value) => values.push(value),
                Err(error) => {
                    let client = self.connected_client_mut(client_id)?;
                    client.negotiation_seen = true;
                    self.record_diagnostic(WebSocketDiagnostic {
                        kind: DiagnosticKind::MalformedInput,
                        client_id: Some(client_id.clone()),
                        event_type: None,
                        detail: error.to_string(),
                    });
                    return Ok(InputReport {
                        complete_messages: 1,
                        decoded_values: values.len(),
                        feature_negotiated: false,
                    });
                }
            }
        }
        if values.is_empty() {
            let client = self.connected_client_mut(client_id)?;
            client.negotiation_seen = true;
            self.record_diagnostic(WebSocketDiagnostic {
                kind: DiagnosticKind::MalformedInput,
                client_id: Some(client_id.clone()),
                event_type: None,
                detail: "empty text message".into(),
            });
            return Ok(InputReport {
                complete_messages: 1,
                decoded_values: 0,
                feature_negotiated: false,
            });
        }

        let mut feature_negotiated = false;
        for value in &values {
            let negotiation_seen = self.connected_client_mut(client_id)?.negotiation_seen;
            if negotiation_seen {
                if value.get("type").and_then(Value::as_str) == Some("feature_flags") {
                    self.record_diagnostic(WebSocketDiagnostic {
                        kind: DiagnosticKind::IgnoredNegotiation,
                        client_id: Some(client_id.clone()),
                        event_type: Some("feature_flags".into()),
                        detail: "feature negotiation is accepted only from the first text value"
                            .into(),
                    });
                }
                continue;
            }
            let client = self.connected_client_mut(client_id)?;
            client.negotiation_seen = true;
            let event_type = value.get("type").and_then(Value::as_str);
            if event_type != Some("feature_flags") {
                self.record_diagnostic(WebSocketDiagnostic {
                    kind: DiagnosticKind::UnknownEvent,
                    client_id: Some(client_id.clone()),
                    event_type: event_type.map(str::to_owned),
                    detail: "the first client message was not feature_flags".into(),
                });
                continue;
            }
            let Some(data) = value.get("data").and_then(Value::as_object) else {
                self.record_diagnostic(WebSocketDiagnostic {
                    kind: DiagnosticKind::MalformedInput,
                    client_id: Some(client_id.clone()),
                    event_type: Some("feature_flags".into()),
                    detail: "feature_flags data must be an object".into(),
                });
                continue;
            };
            let allowed: BTreeMap<String, Value> = data
                .iter()
                .filter(|(name, value)| {
                    self.server_features.contains_key(name.as_str()) && value.is_boolean()
                })
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            self.connected_client_mut(client_id)?.features = allowed;
            let server_features = Value::Object(
                self.server_features
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            );
            self.enqueue_json_for_client(
                client_id,
                None,
                "feature_flags",
                server_features,
                EventAssociation::default(),
            )?;
            feature_negotiated = true;
        }
        Ok(InputReport {
            complete_messages: 1,
            decoded_values: values.len(),
            feature_negotiated,
        })
    }

    fn ensure_publishing(&self, sequence: u64) -> Result<(), WebSocketContractError> {
        if self.shutdown {
            return Err(WebSocketContractError::Shutdown);
        }
        if sequence == 0 {
            return Err(WebSocketContractError::ZeroSequence);
        }
        Ok(())
    }

    fn sequence_disposition(
        &self,
        scope: &SourceSequenceScope,
        sequence: u64,
    ) -> PublishDisposition {
        let state = self.source_sequences.get(scope);
        let previous = state.map(|state| state.last_sequence).unwrap_or(0);
        match sequence.cmp(&previous) {
            std::cmp::Ordering::Greater if state.is_none_or(|state| state.active) => {
                PublishDisposition::Published
            }
            std::cmp::Ordering::Greater => PublishDisposition::Stale,
            std::cmp::Ordering::Equal => PublishDisposition::Duplicate,
            std::cmp::Ordering::Less => PublishDisposition::Stale,
        }
    }

    fn record_source_sequence(
        &mut self,
        scope: SourceSequenceScope,
        sequence: u64,
        terminal: bool,
    ) -> Result<(), WebSocketContractError> {
        if !self.source_sequences.contains_key(&scope)
            && self.source_sequences.len() == self.limits.max_source_sequence_scopes
        {
            let eviction = self
                .source_sequences
                .iter()
                .filter(|(_, state)| !state.active)
                .min_by(|(left_scope, left_state), (right_scope, right_state)| {
                    left_state
                        .last_touched
                        .cmp(&right_state.last_touched)
                        .then_with(|| left_scope.cmp(right_scope))
                })
                .map(|(scope, _)| scope.clone());
            let Some(eviction) = eviction else {
                return Err(WebSocketContractError::TooManySourceSequenceScopes);
            };
            self.source_sequences.remove(&eviction);
        }
        let last_touched = self.next_source_touch;
        self.next_source_touch = self.next_source_touch.saturating_add(1);
        self.source_sequences.insert(
            scope,
            SourceSequenceState {
                last_sequence: sequence,
                active: !terminal,
                last_touched,
            },
        );
        Ok(())
    }

    fn record_sequence_diagnostic(
        &mut self,
        disposition: PublishDisposition,
        scope: &SourceSequenceScope,
        sequence: u64,
        event_type: &str,
    ) {
        let kind = match disposition {
            PublishDisposition::Duplicate => DiagnosticKind::DuplicateEvent,
            PublishDisposition::Stale => DiagnosticKind::StaleEvent,
            PublishDisposition::Published | PublishDisposition::Suppressed => return,
        };
        let previous = self
            .source_sequences
            .get(scope)
            .map(|state| state.last_sequence)
            .unwrap_or(0);
        self.record_diagnostic(WebSocketDiagnostic {
            kind,
            client_id: None,
            event_type: Some(event_type.into()),
            detail: format!("source sequence {sequence} did not advance {}", previous),
        });
    }

    fn recipients(&self, target: &EventTarget) -> Vec<ClientId> {
        match target {
            EventTarget::Broadcast => self
                .clients
                .iter()
                .filter_map(|(client_id, client)| client.connected.then_some(client_id.clone()))
                .collect(),
            EventTarget::Client(client_id) => self
                .clients
                .get(client_id)
                .filter(|client| client.connected)
                .map(|_| vec![client_id.clone()])
                .unwrap_or_default(),
        }
    }

    fn enqueue_json_for_client(
        &mut self,
        client_id: &ClientId,
        source_sequence: Option<u64>,
        event_type: &str,
        data: Value,
        association: EventAssociation,
    ) -> Result<QueueOutcome, WebSocketContractError> {
        let payload = encode_json_message(event_type, data)?;
        if payload.len() > self.limits.max_message_bytes {
            return Err(WebSocketContractError::MessageTooLarge);
        }
        let delivery_sequence = self.take_delivery_sequence();
        self.enqueue_for_client(
            client_id,
            OutboundMessage {
                delivery_sequence,
                source_sequence,
                event_type: event_type.into(),
                wire_kind: OutboundWireKind::Text,
                payload,
                association,
            },
        )
    }

    fn enqueue_for_client(
        &mut self,
        client_id: &ClientId,
        message: OutboundMessage,
    ) -> Result<QueueOutcome, WebSocketContractError> {
        let capacity = self.limits.max_queued_messages_per_client;
        let client = self.connected_client_mut(client_id)?;
        if client.queue.len() < capacity {
            client.queue.push_back(message);
            return Ok(QueueOutcome::Queued);
        }

        if is_coalescible(&message.event_type) {
            if let Some(index) = client
                .queue
                .iter()
                .position(|queued| coalesces_with(queued, &message))
            {
                client.queue.remove(index);
                client.queue.push_back(message);
                return Ok(QueueOutcome::Queued);
            }
        } else if let Some(index) = client
            .queue
            .iter()
            .position(|queued| is_coalescible(&queued.event_type))
        {
            client.queue.remove(index);
            client.queue.push_back(message);
            return Ok(QueueOutcome::Queued);
        }

        let event_type = message.event_type;
        let detail = if is_terminal(&event_type) {
            client.queue.clear();
            client.connected = false;
            "slow client was disconnected; reconnect projects authoritative terminal state"
        } else {
            "bounded client queue retained existing non-coalescible events"
        };
        self.record_diagnostic(WebSocketDiagnostic {
            kind: DiagnosticKind::Backpressure,
            client_id: Some(client_id.clone()),
            event_type: Some(event_type),
            detail: detail.into(),
        });
        Ok(QueueOutcome::Backpressured)
    }

    fn connected_client_mut(
        &mut self,
        client_id: &ClientId,
    ) -> Result<&mut ClientSession, WebSocketContractError> {
        self.clients
            .get_mut(client_id)
            .filter(|client| client.connected)
            .ok_or_else(|| WebSocketContractError::ClientNotConnected(client_id.as_str().into()))
    }

    fn take_delivery_sequence(&mut self) -> u64 {
        let sequence = self.next_delivery_sequence;
        self.next_delivery_sequence = self.next_delivery_sequence.saturating_add(1);
        sequence
    }

    fn record_diagnostic(&mut self, diagnostic: WebSocketDiagnostic) {
        if self.diagnostics.len() == self.limits.max_diagnostics {
            self.diagnostics.pop_front();
        }
        self.diagnostics.push_back(diagnostic);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueueOutcome {
    Queued,
    Backpressured,
}

fn require_reconnect_queue_space(outcome: QueueOutcome) -> Result<(), WebSocketContractError> {
    match outcome {
        QueueOutcome::Queued => Ok(()),
        QueueOutcome::Backpressured => Err(WebSocketContractError::ReconnectProjectionTooLarge),
    }
}

fn validate_event_source(
    event_type: &str,
    source: NativeEventSource,
) -> Result<(), WebSocketContractError> {
    let expected = if event_type.starts_with("assets.seed.") {
        NativeEventSource::AssetSeeder
    } else if event_type == "logs" {
        NativeEventSource::TerminalService
    } else {
        NativeEventSource::Runtime
    };
    if source == expected {
        Ok(())
    } else {
        Err(WebSocketContractError::InvalidEventSource {
            event_type: event_type.into(),
            event_source: source,
        })
    }
}

fn validate_event_target(
    event_type: &str,
    target: &EventTarget,
) -> Result<(), WebSocketContractError> {
    let valid = match event_type {
        "execution_start" | "execution_cached" | "executing" | "progress" | "progress_state"
        | "executed" | "execution_success" | "execution_error" | "logs" => {
            matches!(target, EventTarget::Client(_))
        }
        "execution_interrupted" => matches!(target, EventTarget::Broadcast),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(WebSocketContractError::InvalidEventTarget(
            event_type.into(),
        ))
    }
}

fn is_coalescible(event_type: &str) -> bool {
    matches!(
        event_type,
        "status" | "executing" | "progress" | "progress_state" | "assets.seed.progress"
    )
}

fn coalesces_with(queued: &OutboundMessage, incoming: &OutboundMessage) -> bool {
    is_coalescible(&incoming.event_type)
        && queued.event_type == incoming.event_type
        && queued.association.prompt_id == incoming.association.prompt_id
        && queued.association.attempt_id == incoming.association.attempt_id
        && queued.association.node_id == incoming.association.node_id
}

fn is_terminal(event_type: &str) -> bool {
    matches!(
        event_type,
        "execution_success"
            | "execution_error"
            | "execution_interrupted"
            | "assets.seed.completed"
            | "assets.seed.cancelled"
            | "assets.seed.error"
    )
}

pub fn encode_json_message(
    event_type: &str,
    data: Value,
) -> Result<Vec<u8>, WebSocketContractError> {
    validate_json_event(event_type, &data)?;
    serde_json::to_vec(&json!({"type": event_type, "data": data}))
        .map_err(|error| WebSocketContractError::MalformedJson(error.to_string()))
}

pub fn decode_json_message(bytes: &[u8]) -> Result<(String, Value), WebSocketContractError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| WebSocketContractError::MalformedJson(error.to_string()))?;
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| WebSocketContractError::MalformedJson("missing string type".into()))?;
    let data = value
        .get("data")
        .cloned()
        .ok_or_else(|| WebSocketContractError::MalformedJson("missing data".into()))?;
    validate_json_event(event_type, &data)?;
    Ok((event_type.into(), data))
}

pub fn validate_json_event(event_type: &str, data: &Value) -> Result<(), WebSocketContractError> {
    let descriptor = websocket_event_descriptor(event_type)
        .ok_or_else(|| WebSocketContractError::UnknownEvent(event_type.into()))?;
    if descriptor.wire_kind != CatalogWireKind::Json {
        return Err(WebSocketContractError::IllegalWireEvent(event_type.into()));
    }
    let object = data
        .as_object()
        .ok_or_else(|| invalid_payload(event_type, "data must be an object"))?;
    match descriptor.payload_contract {
        WebSocketPayloadContract::FeatureFlagNegotiation
        | WebSocketPayloadContract::FeatureFlags => {}
        WebSocketPayloadContract::Status => {
            require_number_path(
                data,
                &["status", "exec_info", "queue_remaining"],
                event_type,
            )?;
        }
        WebSocketPayloadContract::Executing => {
            require_present(object, "node", event_type)?;
            if !object["node"].is_null() && !object["node"].is_string() {
                return Err(invalid_payload(event_type, "node must be a string or null"));
            }
            optional_string(object, "display_node", event_type)?;
            optional_string(object, "prompt_id", event_type)?;
        }
        WebSocketPayloadContract::Executed => {
            require_string(object, "node", event_type)?;
            require_string(object, "display_node", event_type)?;
            require_present(object, "output", event_type)?;
            require_string(object, "prompt_id", event_type)?;
        }
        WebSocketPayloadContract::ExecutionStart | WebSocketPayloadContract::ExecutionSuccess => {
            require_string(object, "prompt_id", event_type)?;
            require_number(object, "timestamp", event_type)?;
        }
        WebSocketPayloadContract::ExecutionCached => {
            require_array(object, "nodes", event_type)?;
            require_string(object, "prompt_id", event_type)?;
            require_number(object, "timestamp", event_type)?;
        }
        WebSocketPayloadContract::ExecutionError => {
            for field in [
                "prompt_id",
                "node_id",
                "node_type",
                "exception_message",
                "exception_type",
            ] {
                require_string(object, field, event_type)?;
            }
            for field in ["executed", "traceback"] {
                require_array(object, field, event_type)?;
            }
            for field in ["current_inputs", "current_outputs"] {
                require_object(object, field, event_type)?;
            }
            require_number(object, "timestamp", event_type)?;
        }
        WebSocketPayloadContract::ExecutionInterrupted => {
            for field in ["prompt_id", "node_id", "node_type"] {
                require_string(object, field, event_type)?;
            }
            require_array(object, "executed", event_type)?;
            require_number(object, "timestamp", event_type)?;
        }
        WebSocketPayloadContract::Progress => {
            require_number(object, "value", event_type)?;
            require_number(object, "max", event_type)?;
            require_string(object, "prompt_id", event_type)?;
            require_string(object, "node", event_type)?;
        }
        WebSocketPayloadContract::ProgressState => {
            require_string(object, "prompt_id", event_type)?;
            validate_progress_state_nodes(object, event_type)?;
        }
        WebSocketPayloadContract::Logs => {
            require_array(object, "entries", event_type)?;
            require_number(object, "size", event_type)?;
        }
        WebSocketPayloadContract::AssetSeedProgress => {
            for field in ["scanned", "total", "created", "skipped"] {
                require_number(object, field, event_type)?;
            }
        }
        WebSocketPayloadContract::AssetSeedError => {
            require_string(object, "message", event_type)?;
        }
        WebSocketPayloadContract::AssetSeedStarted
        | WebSocketPayloadContract::AssetSeedPaused
        | WebSocketPayloadContract::AssetSeedResumed
        | WebSocketPayloadContract::AssetSeedFastComplete
        | WebSocketPayloadContract::AssetSeedEnrichComplete
        | WebSocketPayloadContract::AssetSeedCompleted
        | WebSocketPayloadContract::AssetSeedCancelled => {}
        WebSocketPayloadContract::PreviewImage
        | WebSocketPayloadContract::UnencodedPreviewImage
        | WebSocketPayloadContract::Text
        | WebSocketPayloadContract::PreviewImageWithMetadata => {
            return Err(WebSocketContractError::IllegalWireEvent(event_type.into()));
        }
    }
    Ok(())
}

fn require_present(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    if object.contains_key(field) {
        Ok(())
    } else {
        Err(invalid_payload(event_type, &format!("missing {field}")))
    }
}

fn require_string(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    if object.get(field).is_some_and(Value::is_string) {
        Ok(())
    } else {
        Err(invalid_payload(
            event_type,
            &format!("{field} must be a string"),
        ))
    }
}

fn optional_string(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    match object.get(field) {
        None | Some(Value::Null) | Some(Value::String(_)) => Ok(()),
        Some(_) => Err(invalid_payload(
            event_type,
            &format!("{field} must be a string when present"),
        )),
    }
}

fn validate_progress_state_nodes(
    object: &Map<String, Value>,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    let prompt_id = object
        .get("prompt_id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_payload(event_type, "prompt_id must be a string"))?;
    let nodes = object
        .get("nodes")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_payload(event_type, "nodes must be an object"))?;

    for (node_key, node_value) in nodes {
        let node = node_value.as_object().ok_or_else(|| {
            invalid_payload(event_type, &format!("nodes.{node_key} must be an object"))
        })?;
        require_number(node, "value", event_type)?;
        require_number(node, "max", event_type)?;
        let state = node
            .get("state")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_payload(event_type, "state must be a string"))?;
        if !matches!(state, "running" | "finished" | "error") {
            return Err(invalid_payload(
                event_type,
                "state must be running, finished, or error; pending nodes must be omitted",
            ));
        }
        for field in ["node_id", "prompt_id", "display_node_id", "real_node_id"] {
            require_string(node, field, event_type)?;
        }
        require_nullable_string(node, "parent_node_id", event_type)?;

        if node.get("node_id").and_then(Value::as_str) != Some(node_key.as_str()) {
            return Err(invalid_payload(
                event_type,
                &format!("nodes.{node_key}.node_id must match its nodes key"),
            ));
        }
        if node.get("prompt_id").and_then(Value::as_str) != Some(prompt_id) {
            return Err(invalid_payload(
                event_type,
                &format!("nodes.{node_key}.prompt_id must match the event prompt_id"),
            ));
        }
    }
    Ok(())
}

fn require_nullable_string(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    match object.get(field) {
        Some(Value::Null | Value::String(_)) => Ok(()),
        Some(_) => Err(invalid_payload(
            event_type,
            &format!("{field} must be a string or null"),
        )),
        None => Err(invalid_payload(event_type, &format!("missing {field}"))),
    }
}

fn require_number(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    if object.get(field).is_some_and(Value::is_number) {
        Ok(())
    } else {
        Err(invalid_payload(
            event_type,
            &format!("{field} must be a number"),
        ))
    }
}

fn require_array(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    if object.get(field).is_some_and(Value::is_array) {
        Ok(())
    } else {
        Err(invalid_payload(
            event_type,
            &format!("{field} must be an array"),
        ))
    }
}

fn require_object(
    object: &Map<String, Value>,
    field: &str,
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    if object.get(field).is_some_and(Value::is_object) {
        Ok(())
    } else {
        Err(invalid_payload(
            event_type,
            &format!("{field} must be an object"),
        ))
    }
}

fn require_number_path(
    value: &Value,
    path: &[&str],
    event_type: &str,
) -> Result<(), WebSocketContractError> {
    let mut current = value;
    for component in path {
        current = current
            .get(*component)
            .ok_or_else(|| invalid_payload(event_type, &format!("missing {}", path.join("."))))?;
    }
    if current.is_number() {
        Ok(())
    } else {
        Err(invalid_payload(
            event_type,
            &format!("{} must be a number", path.join(".")),
        ))
    }
}

fn invalid_payload(event_type: &str, reason: &str) -> WebSocketContractError {
    WebSocketContractError::InvalidPayload {
        event_type: event_type.into(),
        reason: reason.into(),
    }
}

fn validate_association(
    data: &Value,
    association: &EventAssociation,
) -> Result<(), WebSocketContractError> {
    if let (Some(payload_prompt), Some(associated_prompt)) = (
        data.get("prompt_id").and_then(Value::as_str),
        association.prompt_id.as_deref(),
    ) && payload_prompt != associated_prompt
    {
        return Err(WebSocketContractError::AssociationMismatch);
    }
    let payload_node = data
        .get("node_id")
        .or_else(|| data.get("node"))
        .and_then(Value::as_str);
    if let (Some(payload_node), Some(associated_node)) =
        (payload_node, association.node_id.as_deref())
        && payload_node != associated_node
    {
        return Err(WebSocketContractError::AssociationMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodedBinaryMessage {
    PreviewImage {
        format: PreviewImageFormat,
        encoded_image: Vec<u8>,
    },
    Text {
        node_id: String,
        text: Vec<u8>,
    },
    PreviewImageWithMetadata {
        format: PreviewImageFormat,
        encoded_image: Vec<u8>,
        metadata: Map<String, Value>,
        association: EventAssociation,
    },
}

pub fn legacy_preview_image_type_code(image_type: &str) -> u32 {
    if image_type.eq_ignore_ascii_case("png") {
        PreviewImageFormat::Png.event_code()
    } else {
        PreviewImageFormat::Jpeg.event_code()
    }
}

pub fn encode_preview_image(
    format: PreviewImageFormat,
    encoded_image: &[u8],
) -> Result<Vec<u8>, WebSocketContractError> {
    validate_preview_image(format, encoded_image)?;
    let mut message = Vec::with_capacity(8 + encoded_image.len());
    message.extend(PREVIEW_IMAGE_EVENT_CODE.to_be_bytes());
    message.extend(format.event_code().to_be_bytes());
    message.extend(encoded_image);
    Ok(message)
}

pub fn encode_preview_with_metadata(
    format: PreviewImageFormat,
    encoded_image: &[u8],
    association: &EventAssociation,
    mut metadata: Map<String, Value>,
) -> Result<Vec<u8>, WebSocketContractError> {
    validate_preview_image(format, encoded_image)?;
    let node_id = association.node_id.as_deref().ok_or_else(|| {
        invalid_payload(
            "PREVIEW_IMAGE_WITH_METADATA",
            "metadata preview requires node association",
        )
    })?;
    metadata.insert(
        "image_type".into(),
        Value::String(format.metadata_name().into()),
    );
    metadata.insert("node_id".into(), Value::String(node_id.into()));
    if let Some(prompt_id) = &association.prompt_id {
        metadata.insert("prompt_id".into(), Value::String(prompt_id.clone()));
    }
    let metadata = serde_json::to_vec(&Value::Object(metadata))
        .map_err(|error| WebSocketContractError::MalformedJson(error.to_string()))?;
    let metadata_length = u32::try_from(metadata.len()).map_err(|_| {
        WebSocketContractError::MalformedBinary("preview metadata is too large".into())
    })?;
    let mut message = Vec::with_capacity(8 + metadata.len() + encoded_image.len());
    message.extend(PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE.to_be_bytes());
    message.extend(metadata_length.to_be_bytes());
    message.extend(metadata);
    message.extend(encoded_image);
    Ok(message)
}

pub fn encode_text_message(
    node_id: &str,
    text: &[u8],
    max_message_bytes: usize,
) -> Result<Vec<u8>, WebSocketContractError> {
    if node_id.is_empty() {
        return Err(invalid_payload("TEXT", "node_id cannot be empty"));
    }
    let node_id_length = u32::try_from(node_id.len())
        .map_err(|_| WebSocketContractError::MalformedBinary("node_id is too large".into()))?;
    let message_length = 8usize
        .saturating_add(node_id.len())
        .saturating_add(text.len());
    if message_length > max_message_bytes {
        return Err(WebSocketContractError::MessageTooLarge);
    }
    let mut message = Vec::with_capacity(message_length);
    message.extend(TEXT_EVENT_CODE.to_be_bytes());
    message.extend(node_id_length.to_be_bytes());
    message.extend(node_id.as_bytes());
    message.extend(text);
    Ok(message)
}

pub fn decode_binary_message(
    bytes: &[u8],
    max_message_bytes: usize,
) -> Result<DecodedBinaryMessage, WebSocketContractError> {
    if bytes.len() > max_message_bytes {
        return Err(WebSocketContractError::MessageTooLarge);
    }
    if bytes.len() < 4 {
        return Err(WebSocketContractError::MalformedBinary(
            "missing event header".into(),
        ));
    }
    let event_code = read_u32(bytes, 0)?;
    match event_code {
        PREVIEW_IMAGE_EVENT_CODE => decode_preview_image(bytes),
        UNENCODED_PREVIEW_IMAGE_EVENT_CODE => Err(WebSocketContractError::IllegalWireEvent(
            "UNENCODED_PREVIEW_IMAGE".into(),
        )),
        TEXT_EVENT_CODE => decode_text(bytes),
        PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE => decode_preview_with_metadata(bytes),
        unknown => Err(WebSocketContractError::MalformedBinary(format!(
            "unknown binary event code {unknown}"
        ))),
    }
}

fn decode_preview_image(bytes: &[u8]) -> Result<DecodedBinaryMessage, WebSocketContractError> {
    if bytes.len() <= 8 {
        return Err(WebSocketContractError::MalformedBinary(
            "preview image is truncated".into(),
        ));
    }
    let format = decode_image_format(read_u32(bytes, 4)?)?;
    let encoded_image = bytes[8..].to_vec();
    validate_preview_image(format, &encoded_image)?;
    Ok(DecodedBinaryMessage::PreviewImage {
        format,
        encoded_image,
    })
}

fn decode_text(bytes: &[u8]) -> Result<DecodedBinaryMessage, WebSocketContractError> {
    if bytes.len() < 8 {
        return Err(WebSocketContractError::MalformedBinary(
            "text frame is truncated".into(),
        ));
    }
    let node_id_length = usize::try_from(read_u32(bytes, 4)?).map_err(|_| {
        WebSocketContractError::MalformedBinary("node id length is not representable".into())
    })?;
    let text_start = 8usize
        .checked_add(node_id_length)
        .ok_or_else(|| WebSocketContractError::MalformedBinary("node id length overflow".into()))?;
    if text_start > bytes.len() {
        return Err(WebSocketContractError::MalformedBinary(
            "node id length exceeds the frame".into(),
        ));
    }
    let node_id = std::str::from_utf8(&bytes[8..text_start])
        .map_err(|_| WebSocketContractError::MalformedBinary("node id is not UTF-8".into()))?;
    if node_id.is_empty() {
        return Err(WebSocketContractError::MalformedBinary(
            "node id cannot be empty".into(),
        ));
    }
    Ok(DecodedBinaryMessage::Text {
        node_id: node_id.into(),
        text: bytes[text_start..].to_vec(),
    })
}

fn decode_preview_with_metadata(
    bytes: &[u8],
) -> Result<DecodedBinaryMessage, WebSocketContractError> {
    if bytes.len() < 9 {
        return Err(WebSocketContractError::MalformedBinary(
            "metadata preview is truncated".into(),
        ));
    }
    let metadata_length = usize::try_from(read_u32(bytes, 4)?).map_err(|_| {
        WebSocketContractError::MalformedBinary("metadata length is not representable".into())
    })?;
    let image_start = 8usize.checked_add(metadata_length).ok_or_else(|| {
        WebSocketContractError::MalformedBinary("metadata length overflow".into())
    })?;
    if image_start >= bytes.len() {
        return Err(WebSocketContractError::MalformedBinary(
            "metadata length exceeds the frame or image is empty".into(),
        ));
    }
    let metadata_value: Value = serde_json::from_slice(&bytes[8..image_start])
        .map_err(|error| WebSocketContractError::MalformedJson(error.to_string()))?;
    let metadata = metadata_value.as_object().cloned().ok_or_else(|| {
        invalid_payload("PREVIEW_IMAGE_WITH_METADATA", "metadata must be an object")
    })?;
    let image_type = metadata
        .get("image_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid_payload(
                "PREVIEW_IMAGE_WITH_METADATA",
                "metadata.image_type must be a string",
            )
        })?;
    let format = match image_type.to_ascii_lowercase().as_str() {
        "jpeg" | "jpg" => PreviewImageFormat::Jpeg,
        "png" => PreviewImageFormat::Png,
        _ => {
            return Err(invalid_payload(
                "PREVIEW_IMAGE_WITH_METADATA",
                "metadata.image_type must be jpeg or png",
            ));
        }
    };
    let node_id = metadata
        .get("node_id")
        .or_else(|| metadata.get("node"))
        .and_then(Value::as_str)
        .filter(|node_id| !node_id.is_empty())
        .ok_or_else(|| {
            invalid_payload(
                "PREVIEW_IMAGE_WITH_METADATA",
                "metadata must include node identity",
            )
        })?;
    let prompt_id = metadata
        .get("prompt_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let encoded_image = bytes[image_start..].to_vec();
    validate_preview_image(format, &encoded_image)?;
    Ok(DecodedBinaryMessage::PreviewImageWithMetadata {
        format,
        encoded_image,
        association: EventAssociation {
            prompt_id,
            node_id: Some(node_id.into()),
            attempt_id: None,
        },
        metadata,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, WebSocketContractError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| WebSocketContractError::MalformedBinary("header offset overflow".into()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| WebSocketContractError::MalformedBinary("truncated 32-bit header".into()))?;
    let array: [u8; 4] = slice
        .try_into()
        .map_err(|_| WebSocketContractError::MalformedBinary("invalid 32-bit header".into()))?;
    Ok(u32::from_be_bytes(array))
}

fn decode_image_format(code: u32) -> Result<PreviewImageFormat, WebSocketContractError> {
    match code {
        1 => Ok(PreviewImageFormat::Jpeg),
        2 => Ok(PreviewImageFormat::Png),
        _ => Err(WebSocketContractError::MalformedBinary(format!(
            "unknown preview image type {code}"
        ))),
    }
}

fn validate_preview_image(
    format: PreviewImageFormat,
    encoded_image: &[u8],
) -> Result<(), WebSocketContractError> {
    let valid = match format {
        PreviewImageFormat::Jpeg => {
            encoded_image.starts_with(&[0xff, 0xd8, 0xff]) && encoded_image.ends_with(&[0xff, 0xd9])
        }
        PreviewImageFormat::Png => encoded_image.starts_with(b"\x89PNG\r\n\x1a\n"),
    };
    if valid {
        Ok(())
    } else {
        Err(WebSocketContractError::MalformedBinary(format!(
            "encoded bytes do not match declared {} preview format",
            format.metadata_name()
        )))
    }
}

fn close_payload(code: u16, reason: &str) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + reason.len());
    payload.extend(code.to_be_bytes());
    payload.extend(reason.as_bytes());
    payload
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn test_client(value: &str) -> ClientId {
        ClientId::new(value).expect("test client id should be valid")
    }

    fn test_principal(identity: &str, scopes: &[&str]) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal {
            identity: identity.into(),
            scopes: scopes.iter().map(|scope| (*scope).to_owned()).collect(),
        }
    }

    fn png() -> Vec<u8> {
        b"\x89PNG\r\n\x1a\nfixture".to_vec()
    }

    fn jpeg() -> Vec<u8> {
        vec![0xff, 0xd8, 0xff, 0x01, 0xff, 0xd9]
    }

    fn association(prompt_id: &str, node_id: &str) -> EventAssociation {
        EventAssociation {
            prompt_id: Some(prompt_id.into()),
            node_id: Some(node_id.into()),
            attempt_id: Some("attempt-1".into()),
        }
    }

    fn valid_json_payload(event_type: &str) -> Result<Value, String> {
        let payload = match event_type {
            "feature_flags" => json!({"supports_preview_metadata": true}),
            "status" => json!({"status":{"exec_info":{"queue_remaining":1}},"sid":"a"}),
            "executing" => json!({"node":"4","display_node":"4","prompt_id":"p"}),
            "executed" => json!({"node":"4","display_node":"4","output":null,"prompt_id":"p"}),
            "execution_start" | "execution_success" => json!({"prompt_id":"p","timestamp":1}),
            "execution_cached" => json!({"nodes":[],"prompt_id":"p","timestamp":1}),
            "execution_error" => json!({
                "prompt_id":"p","node_id":"4","node_type":"SaveImage","executed":[],
                "exception_message":"failure","exception_type":"NativeError","traceback":[],
                "current_inputs":{},"current_outputs":{},"timestamp":1
            }),
            "execution_interrupted" => json!({
                "prompt_id":"p","node_id":"4","node_type":"SaveImage","executed":[],"timestamp":1
            }),
            "progress" => json!({"value":1,"max":2,"prompt_id":"p","node":"4"}),
            "progress_state" => json!({
                "prompt_id":"p","nodes":{"4":{"value":1,"max":2,"state":"running",
                "node_id":"4","prompt_id":"p","display_node_id":"4","parent_node_id":null,
                "real_node_id":"4"}}
            }),
            "logs" => json!({"entries":[],"size":0}),
            "assets.seed.started" => json!({"roots":[]}),
            "assets.seed.progress" => {
                json!({"scanned":1,"total":2,"created":1,"skipped":0,"unknown":true})
            }
            "assets.seed.paused" | "assets.seed.resumed" => json!({}),
            "assets.seed.fast_complete"
            | "assets.seed.enrich_complete"
            | "assets.seed.completed"
            | "assets.seed.cancelled" => json!({"scanned":1,"unknown":"retained"}),
            "assets.seed.error" => json!({"message":"scan failed","errors":[]}),
            unexpected => return Err(format!("no JSON fixture for {unexpected}")),
        };
        Ok(payload)
    }

    fn exercise_catalog_row(
        index: usize,
        descriptor: &WebSocketEventDescriptor,
        contract: &NormativeWebSocketContract,
    ) -> Result<(), String> {
        let expected_feature_id = format!("COMFY-WS-{:03}", index + 1);
        if descriptor.feature_id != expected_feature_id {
            return Err(format!(
                "expected {expected_feature_id}, found {}",
                descriptor.feature_id
            ));
        }
        if contract.feature_id != descriptor.feature_id
            || contract.direction != descriptor.direction
            || contract.wire_kind != descriptor.wire_kind
            || contract.event_type != descriptor.event_type
            || contract.binary_code != descriptor.binary_code
            || contract.availability != descriptor.availability
        {
            return Err("parsed normative contract disagrees with production descriptor".into());
        }
        if websocket_event_descriptor_by_id(descriptor.feature_id) != Some(descriptor) {
            return Err("feature ID lookup did not return the catalog descriptor".into());
        }
        if let Some(event_code) = descriptor.binary_code
            && websocket_binary_event_descriptor(event_code) != Some(descriptor)
        {
            return Err("binary event-code lookup did not return the catalog descriptor".into());
        }

        match descriptor.wire_kind {
            CatalogWireKind::Json => {
                let payload = valid_json_payload(descriptor.event_type)?;
                let encoded = encode_json_message(descriptor.event_type, payload.clone())
                    .map_err(|error| error.to_string())?;
                let decoded = decode_json_message(&encoded).map_err(|error| error.to_string())?;
                if decoded != (descriptor.event_type.into(), payload) {
                    return Err("JSON framing did not round-trip exactly".into());
                }
            }
            CatalogWireKind::Binary => {
                let decoded = match descriptor.binary_code {
                    Some(PREVIEW_IMAGE_EVENT_CODE) => {
                        let encoded = encode_preview_image(PreviewImageFormat::Png, &png())
                            .map_err(|error| error.to_string())?;
                        decode_binary_message(&encoded, 1024)
                    }
                    Some(TEXT_EVENT_CODE) => {
                        let encoded = encode_text_message("4", b"status", 1024)
                            .map_err(|error| error.to_string())?;
                        decode_binary_message(&encoded, 1024)
                    }
                    Some(PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE) => {
                        let encoded = encode_preview_with_metadata(
                            PreviewImageFormat::Jpeg,
                            &jpeg(),
                            &association("p", "4"),
                            Map::new(),
                        )
                        .map_err(|error| error.to_string())?;
                        decode_binary_message(&encoded, 1024)
                    }
                    unexpected => {
                        return Err(format!("unexpected binary catalog code {unexpected:?}"));
                    }
                }
                .map_err(|error| error.to_string())?;
                let decoded_code = match decoded {
                    DecodedBinaryMessage::PreviewImage { .. } => PREVIEW_IMAGE_EVENT_CODE,
                    DecodedBinaryMessage::Text { .. } => TEXT_EVENT_CODE,
                    DecodedBinaryMessage::PreviewImageWithMetadata { .. } => {
                        PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE
                    }
                };
                if Some(decoded_code) != descriptor.binary_code {
                    return Err("binary framing decoded as a different event".into());
                }
            }
            CatalogWireKind::Internal => {
                if descriptor.binary_code != Some(UNENCODED_PREVIEW_IMAGE_EVENT_CODE)
                    || !matches!(
                        decode_binary_message(
                            &UNENCODED_PREVIEW_IMAGE_EVENT_CODE.to_be_bytes(),
                            1024
                        ),
                        Err(WebSocketContractError::IllegalWireEvent(_))
                    )
                {
                    return Err("internal preview event was exposed on the wire".into());
                }
            }
        }
        Ok(())
    }

    fn executable_test_outcome(identifier: &str, check: fn()) -> Value {
        let result = std::panic::catch_unwind(check);
        let error = result.err().map(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|value| (*value).into()))
                .unwrap_or_else(|| "validation panicked without a string payload".into())
        });
        json!({
            "id": identifier,
            "passed": error.is_none(),
            "error": error,
        })
    }

    #[test]
    pub(crate) fn val_ws_001_catalog_rows_have_exact_descriptors_and_working_contracts() {
        validate_descriptor_catalog().expect("normative catalog should validate");
        let normative_catalog =
            normative_websocket_catalog().expect("normative catalog should parse");
        assert_eq!(normative_catalog.len(), 26);
        assert_eq!(WEBSOCKET_EVENT_CATALOG.len(), 26);
        let catalog_cases = WEBSOCKET_EVENT_CATALOG
            .iter()
            .zip(normative_catalog)
            .enumerate()
            .map(|(index, (descriptor, contract))| {
                let result = exercise_catalog_row(index, descriptor, contract);
                json!({
                    "id": descriptor.feature_id,
                    "event_type": descriptor.event_type,
                    "passed": result.is_ok(),
                    "error": result.err(),
                })
            })
            .collect::<Vec<_>>();
        let protocol_cases = vec![
            executable_test_outcome(
                "fragmentation-and-first-message-negotiation",
                val_ws_001_fragmented_and_coalesced_negotiation_is_first_message_only,
            ),
            executable_test_outcome(
                "malformed-and-unknown-input-recovery",
                val_ws_001_malformed_and_unknown_input_remains_visible_and_later_frames_continue,
            ),
            executable_test_outcome(
                "targeting-ordering-duplicates-and-stale-events",
                val_ws_001_targeted_ordering_duplicate_and_stale_events_are_deterministic,
            ),
            executable_test_outcome(
                "attempt-scoped-source-sequences",
                val_ws_001_source_sequence_restarts_for_each_execution_attempt,
            ),
            executable_test_outcome(
                "reconnect-authoritative-projection",
                val_ws_001_reconnect_projects_status_execution_and_history_without_replaying_stale_source_events,
            ),
            executable_test_outcome(
                "preview-negotiation-and-association",
                val_ws_001_preview_negotiation_selects_metadata_frame_and_validates_association,
            ),
            executable_test_outcome(
                "conditional-events-and-log-subscription",
                val_ws_001_conditional_events_require_native_source_enablement_and_subscription,
            ),
            executable_test_outcome(
                "malformed-binary-contracts",
                val_ws_001_malformed_binary_header_length_format_and_internal_code_are_rejected,
            ),
            executable_test_outcome(
                "association-scoped-backpressure",
                val_ws_001_bounded_queue_coalesces_only_matching_progress_associations,
            ),
            executable_test_outcome(
                "bounded-source-sequences-and-server-features",
                val_ws_001_bounded_state_never_evicts_active_attempt_sequences,
            ),
            executable_test_outcome(
                "client-identity-lifecycle",
                val_ws_001_duplicate_connect_is_rejected_and_disconnected_sessions_evict_deterministically,
            ),
            executable_test_outcome(
                "authenticated-session-principal-ownership",
                val_ws_001_authenticated_session_principal_is_atomic_across_reconnect_and_disconnect,
            ),
            executable_test_outcome(
                "critical-backpressure",
                val_ws_001_bounded_queue_coalesces_progress_and_disconnects_on_critical_backpressure,
            ),
            executable_test_outcome(
                "disconnect-cancellation-and-shutdown",
                val_ws_001_disconnect_cancel_and_shutdown_stop_delivery_without_affecting_other_state,
            ),
            executable_test_outcome(
                "unknown-field-retention-and-schema-validation",
                val_ws_001_unknown_json_fields_are_retained_and_required_fields_are_checked,
            ),
            executable_test_outcome(
                "progress-state-typed-schema-validation",
                val_ws_001_progress_state_schema_is_bound_and_exhaustively_validated,
            ),
        ];
        let passed = catalog_cases
            .iter()
            .chain(&protocol_cases)
            .filter(|outcome| outcome["passed"] == Value::Bool(true))
            .count();
        let failed = catalog_cases.len() + protocol_cases.len() - passed;
        let fixture_sha256 = format!("{:x}", Sha256::digest(WEBSOCKET_CATALOG_CSV.as_bytes()));
        let artifact = json!({
            "schema_version": 1,
            "validation_id": "VAL-WS-001",
            "fixture": "catalogs/backend-websocket-events.csv",
            "fixture_sha256": fixture_sha256,
            "environment": {
                "backend": "native-rust-event-bus",
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
            },
            "catalog_cases": catalog_cases,
            "protocol_cases": protocol_cases,
            "passed": passed,
            "failed": failed,
            "skipped": [],
            "external_processes": 0,
            "proxy_or_forward_paths": 0,
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
            })
            .join("sim-parity");
        std::fs::create_dir_all(&target).expect("VAL-WS-001 artifact directory should exist");
        let mut artifact_bytes =
            serde_json::to_vec_pretty(&artifact).expect("VAL-WS-001 artifact should serialize");
        artifact_bytes.push(b'\n');
        std::fs::write(target.join("val-ws-001.json"), artifact_bytes)
            .expect("VAL-WS-001 artifact should be written");
        assert_eq!(failed, 0, "all recorded VAL-WS-001 checks must pass");
    }

    #[test]
    fn val_ws_001_fragmented_and_coalesced_negotiation_is_first_message_only() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("client-a");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.drain_client(&client).expect("status should drain");
        let first = br#"{"type":"feature_flags","data":{"supports_preview_"#;
        let second = br#"metadata":true}}{"type":"feature_flags","data":{"supports_preview_metadata":false}}"#;
        let report = bus
            .process_input_fragment(
                &client,
                InputFragment {
                    kind: FragmentKind::Text,
                    bytes: first.to_vec(),
                    final_fragment: false,
                },
            )
            .expect("first fragment should be accepted");
        assert_eq!(report.complete_messages, 0);
        let report = bus
            .process_input_fragment(
                &client,
                InputFragment {
                    kind: FragmentKind::Text,
                    bytes: second.to_vec(),
                    final_fragment: true,
                },
            )
            .expect("coalesced values should be decoded");
        assert_eq!(report.decoded_values, 2);
        assert!(report.feature_negotiated);
        let messages = bus.drain_client(&client).expect("response should drain");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].event_type, "feature_flags");
        assert!(
            bus.take_diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind == DiagnosticKind::IgnoredNegotiation)
        );
    }

    #[test]
    fn val_ws_001_malformed_and_unknown_input_remains_visible_and_later_frames_continue() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let malformed_client = test_client("malformed");
        bus.connect(malformed_client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        let report = bus
            .process_input_fragment(
                &malformed_client,
                InputFragment {
                    kind: FragmentKind::Text,
                    bytes: b"{".to_vec(),
                    final_fragment: true,
                },
            )
            .expect("malformed first message should be visible, not fatal");
        assert!(!report.feature_negotiated);
        let later = br#"{"type":"feature_flags","data":{"supports_preview_metadata":true}}"#;
        bus.process_input_fragment(
            &malformed_client,
            InputFragment {
                kind: FragmentKind::Text,
                bytes: later.to_vec(),
                final_fragment: true,
            },
        )
        .expect("later valid frame should continue processing");

        let unknown_client = test_client("unknown");
        bus.connect(unknown_client.clone(), ReconnectProjection::default())
            .expect("second client should connect");
        bus.process_input_fragment(
            &unknown_client,
            InputFragment {
                kind: FragmentKind::Text,
                bytes: br#"{"type":"future.event","data":{"retained":true}}"#.to_vec(),
                final_fragment: true,
            },
        )
        .expect("unknown frame should not stop the session");
        assert!(bus.take_diagnostics().iter().any(|diagnostic| {
            matches!(
                diagnostic.kind,
                DiagnosticKind::MalformedInput
                    | DiagnosticKind::IgnoredNegotiation
                    | DiagnosticKind::UnknownEvent
            )
        }));
    }

    #[test]
    fn val_ws_001_targeted_ordering_duplicate_and_stale_events_are_deterministic() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let first = test_client("first");
        let second = test_client("second");
        bus.connect(first.clone(), ReconnectProjection::default())
            .expect("first should connect");
        bus.connect(second.clone(), ReconnectProjection::default())
            .expect("second should connect");
        bus.drain_client(&first).expect("status should drain");
        bus.drain_client(&second).expect("status should drain");
        let report = bus
            .publish_json(NativeJsonEvent {
                sequence: 10,
                event_type: "execution_start".into(),
                data: json!({"prompt_id":"p","timestamp":1}),
                target: EventTarget::Client(first.clone()),
                source: NativeEventSource::Runtime,
                association: association("p", "4"),
            })
            .expect("targeted event should publish");
        assert_eq!(report.delivered_clients, 1);
        assert_eq!(
            bus.drain_client(&first).expect("first should drain").len(),
            1
        );
        assert!(
            bus.drain_client(&second)
                .expect("second should drain")
                .is_empty()
        );
        for (sequence, expected) in [
            (10, PublishDisposition::Duplicate),
            (9, PublishDisposition::Stale),
        ] {
            let report = bus
                .publish_json(NativeJsonEvent {
                    sequence,
                    event_type: "execution_success".into(),
                    data: json!({"prompt_id":"p","timestamp":2}),
                    target: EventTarget::Client(first.clone()),
                    source: NativeEventSource::Runtime,
                    association: association("p", "4"),
                })
                .expect("duplicate or stale event should be discarded safely");
            assert_eq!(report.disposition, expected);
            assert_eq!(report.delivered_clients, 0);
        }
    }

    #[test]
    fn val_ws_001_source_sequence_restarts_for_each_execution_attempt() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("attempt-sequences");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.drain_client(&client).expect("status should drain");

        for attempt_id in ["attempt-a", "attempt-b"] {
            let report = bus
                .publish_json(NativeJsonEvent {
                    sequence: 1,
                    event_type: "execution_start".into(),
                    data: json!({"prompt_id":"p","timestamp":1}),
                    target: EventTarget::Client(client.clone()),
                    source: NativeEventSource::Runtime,
                    association: EventAssociation {
                        prompt_id: Some("p".into()),
                        node_id: None,
                        attempt_id: Some(attempt_id.into()),
                    },
                })
                .expect("the first event in each attempt should publish");
            assert_eq!(report.disposition, PublishDisposition::Published);
        }

        let messages = bus.drain_client(&client).expect("events should drain");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].source_sequence, Some(1));
        assert_eq!(messages[1].source_sequence, Some(1));
    }

    #[test]
    fn val_ws_001_bounded_state_never_evicts_active_attempt_sequences() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits {
            max_source_sequence_scopes: 2,
            max_server_features: 1,
            ..WebSocketLimits::default()
        })
        .expect("bounded limits should be valid");
        let client = test_client("bounded-state");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.drain_client(&client).expect("status should drain");
        bus.set_server_feature("supports_preview_metadata", Value::Bool(false))
            .expect("an existing feature may be updated at capacity");
        assert_eq!(
            bus.set_server_feature("future_feature", Value::Bool(true)),
            Err(WebSocketContractError::TooManyServerFeatures)
        );

        let association_for = |attempt_id: &str| EventAssociation {
            prompt_id: Some(format!("prompt-{attempt_id}")),
            node_id: None,
            attempt_id: Some(attempt_id.into()),
        };
        for attempt_id in ["a", "b"] {
            bus.publish_json(NativeJsonEvent {
                sequence: 1,
                event_type: "execution_start".into(),
                data: json!({"prompt_id":format!("prompt-{attempt_id}"),"timestamp":1}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::Runtime,
                association: association_for(attempt_id),
            })
            .expect("active attempt should reserve a source-sequence scope");
        }
        assert!(matches!(
            bus.publish_json(NativeJsonEvent {
                sequence: 1,
                event_type: "execution_start".into(),
                data: json!({"prompt_id":"prompt-c","timestamp":1}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::Runtime,
                association: association_for("c"),
            }),
            Err(WebSocketContractError::TooManySourceSequenceScopes)
        ));

        bus.publish_json(NativeJsonEvent {
            sequence: 2,
            event_type: "execution_success".into(),
            data: json!({"prompt_id":"prompt-a","timestamp":2}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association_for("a"),
        })
        .expect("terminal event should make its sequence scope evictable");
        bus.publish_json(NativeJsonEvent {
            sequence: 1,
            event_type: "execution_start".into(),
            data: json!({"prompt_id":"prompt-c","timestamp":1}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association_for("c"),
        })
        .expect("the oldest terminal scope should be evicted deterministically");

        let duplicate = bus
            .publish_json(NativeJsonEvent {
                sequence: 1,
                event_type: "execution_start".into(),
                data: json!({"prompt_id":"prompt-b","timestamp":1}),
                target: EventTarget::Client(client),
                source: NativeEventSource::Runtime,
                association: association_for("b"),
            })
            .expect("active scope should remain tracked");
        assert_eq!(duplicate.disposition, PublishDisposition::Duplicate);
        assert_eq!(bus.source_sequences.len(), 2);
        assert!(
            bus.source_sequences
                .keys()
                .all(|scope| scope.attempt_id.as_deref() != Some("a"))
        );
    }

    #[test]
    fn val_ws_001_duplicate_connect_is_rejected_and_disconnected_sessions_evict_deterministically()
    {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits {
            max_clients: 2,
            ..WebSocketLimits::default()
        })
        .expect("bounded limits should be valid");
        let first = test_client("a");
        let second = test_client("b");
        bus.connect(first.clone(), ReconnectProjection::default())
            .expect("first client should connect");
        assert_eq!(
            bus.connect(first.clone(), ReconnectProjection::default()),
            Err(WebSocketContractError::ClientAlreadyConnected("a".into()))
        );
        assert_eq!(
            bus.drain_client(&first)
                .expect("the original session should remain intact")
                .len(),
            1
        );
        bus.connect(second.clone(), ReconnectProjection::default())
            .expect("second client should connect");
        assert!(bus.disconnect(&first));
        assert!(bus.disconnect(&second));

        let replacement = test_client("c");
        bus.connect(replacement.clone(), ReconnectProjection::default())
            .expect("a disconnected session should be evicted at capacity");
        assert!(!bus.clients.contains_key(&first));
        assert!(bus.clients.contains_key(&second));
        assert!(bus.clients.contains_key(&replacement));
    }

    #[test]
    fn val_ws_001_authenticated_session_principal_is_atomic_across_reconnect_and_disconnect() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("principal-session");
        let requested_session_id = test_client("requested-session");
        let original_principal = test_principal("principal-a", &["api:read"]);
        bus.connect_authenticated_with_session_id(
            client.clone(),
            requested_session_id.clone(),
            original_principal.clone(),
            ReconnectProjection::default(),
        )
        .expect("authenticated client should connect");
        assert_eq!(
            bus.authenticated_principal(&client)
                .expect("the live session should own its principal"),
            &original_principal
        );
        let status = bus
            .drain_client(&client)
            .expect("initial status should drain");
        let (_, status_data) = decode_json_message(&status[0].payload)
            .expect("initial status should use valid JSON framing");
        assert_eq!(status_data["sid"], requested_session_id.as_str());

        assert_eq!(
            bus.connect_authenticated(
                client.clone(),
                original_principal.clone(),
                ReconnectProjection {
                    queue_remaining: 99,
                    ..ReconnectProjection::default()
                },
            ),
            Err(WebSocketContractError::ClientAlreadyConnected(
                client.as_str().into()
            ))
        );
        assert!(
            bus.drain_client(&client)
                .expect("a duplicate connection must not replace session state")
                .is_empty()
        );
        assert_eq!(
            bus.connect_authenticated(
                client.clone(),
                test_principal("principal-b", &["api:read"]),
                ReconnectProjection::default(),
            ),
            Err(WebSocketContractError::PrincipalMismatch(
                client.as_str().into()
            ))
        );
        assert_eq!(
            bus.authenticated_principal(&client)
                .expect("a rejected duplicate must retain the original principal"),
            &original_principal
        );

        assert!(bus.disconnect(&client));
        assert!(matches!(
            bus.authenticated_principal(&client),
            Err(WebSocketContractError::ClientNotConnected(_))
        ));
        let disconnected = bus
            .clients
            .get(&client)
            .expect("disconnected session should remain available for authenticated reconnect");
        assert!(!disconnected.connected);
        assert_eq!(disconnected.principal, original_principal);
        assert!(disconnected.queue.is_empty());
        assert_eq!(
            bus.connect_authenticated(
                client.clone(),
                test_principal("principal-b", &["api:read"]),
                ReconnectProjection::default(),
            ),
            Err(WebSocketContractError::PrincipalMismatch(
                client.as_str().into()
            ))
        );
        let rejected_principal_reconnect = bus
            .clients
            .get(&client)
            .expect("principal mismatch must retain the disconnected session");
        assert!(!rejected_principal_reconnect.connected);
        assert_eq!(rejected_principal_reconnect.principal, original_principal);
        assert!(rejected_principal_reconnect.queue.is_empty());

        let delivery_sequence_before_failure = bus.next_delivery_sequence;
        let diagnostics_before_failure = bus.diagnostics.clone();
        let malformed_projection = ReconnectProjection {
            current_execution: vec![ReconnectJsonEvent {
                event_type: "progress_state".into(),
                data: json!({
                    "prompt_id":"prompt-1",
                    "nodes":{"4":{"value":1,"max":2}}
                }),
                association: association("prompt-1", "4"),
            }],
            ..ReconnectProjection::default()
        };
        assert!(matches!(
            bus.connect_authenticated(
                client.clone(),
                original_principal.clone(),
                malformed_projection,
            ),
            Err(WebSocketContractError::InvalidPayload { .. })
        ));
        let failed_reconnect = bus
            .clients
            .get(&client)
            .expect("failed reconnect must restore the disconnected session");
        assert!(!failed_reconnect.connected);
        assert_eq!(failed_reconnect.principal, original_principal);
        assert!(failed_reconnect.queue.is_empty());
        assert_eq!(bus.next_delivery_sequence, delivery_sequence_before_failure);
        assert_eq!(bus.diagnostics, diagnostics_before_failure);

        let refreshed_principal = test_principal("principal-a", &["api:read", "api:write"]);
        bus.connect_authenticated(
            client.clone(),
            refreshed_principal.clone(),
            ReconnectProjection::default(),
        )
        .expect("the same authenticated identity should reconnect atomically");
        assert_eq!(
            bus.authenticated_principal(&client)
                .expect("successful reconnect should install refreshed scopes"),
            &refreshed_principal
        );
        assert_eq!(
            bus.drain_client(&client)
                .expect("successful reconnect should enqueue one status")
                .len(),
            1
        );
        assert!(bus.disconnect(&client));
        assert!(matches!(
            bus.authenticated_principal(&client),
            Err(WebSocketContractError::ClientNotConnected(_))
        ));

        let mut bounded_bus = NativeWebSocketEventBus::new(WebSocketLimits {
            max_clients: 1,
            max_queued_messages_per_client: 1,
            ..WebSocketLimits::default()
        })
        .expect("bounded limits should be valid");
        let retained_client = test_client("retained");
        let retained_principal = test_principal("retained-principal", &[]);
        bounded_bus
            .connect_authenticated(
                retained_client.clone(),
                retained_principal.clone(),
                ReconnectProjection::default(),
            )
            .expect("retained client should connect");
        assert!(bounded_bus.disconnect(&retained_client));
        let bounded_delivery_sequence = bounded_bus.next_delivery_sequence;
        let bounded_diagnostics = bounded_bus.diagnostics.clone();
        let rejected_client = test_client("rejected");
        assert!(matches!(
            bounded_bus.connect_authenticated(
                rejected_client.clone(),
                test_principal("rejected-principal", &[]),
                ReconnectProjection {
                    current_execution: vec![
                        ReconnectJsonEvent {
                            event_type: "execution_start".into(),
                            data: json!({"prompt_id":"prompt-1","timestamp":1}),
                            association: association("prompt-1", "4"),
                        },
                        ReconnectJsonEvent {
                            event_type: "execution_cached".into(),
                            data: json!({"nodes":["4"],"prompt_id":"prompt-1","timestamp":2}),
                            association: association("prompt-1", "4"),
                        },
                    ],
                    ..ReconnectProjection::default()
                },
            ),
            Err(WebSocketContractError::ReconnectProjectionTooLarge)
        ));
        assert!(!bounded_bus.clients.contains_key(&rejected_client));
        let retained_session = bounded_bus
            .clients
            .get(&retained_client)
            .expect("failed connect must restore the disconnected eviction candidate");
        assert!(!retained_session.connected);
        assert_eq!(retained_session.principal, retained_principal);
        assert_eq!(
            bounded_bus.next_delivery_sequence,
            bounded_delivery_sequence
        );
        assert_eq!(bounded_bus.diagnostics, bounded_diagnostics);
    }

    #[test]
    fn val_ws_001_reconnect_projects_status_execution_and_history_without_replaying_stale_source_events()
     {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("reconnect");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.drain_client(&client).expect("status should drain");
        bus.publish_json(NativeJsonEvent {
            sequence: 3,
            event_type: "execution_start".into(),
            data: json!({"prompt_id":"p","timestamp":1}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association("p", "4"),
        })
        .expect("event should publish");
        assert!(bus.disconnect(&client));
        bus.connect(
            client.clone(),
            ReconnectProjection {
                queue_remaining: 2,
                current_execution: vec![ReconnectJsonEvent {
                    event_type: "executing".into(),
                    data: json!({"node":"4","display_node":"4","prompt_id":"p"}),
                    association: association("p", "4"),
                }],
                history_reconciliation: vec![ReconnectJsonEvent {
                    event_type: "executed".into(),
                    data: json!({"node":"3","display_node":"3","output":{},"prompt_id":"p"}),
                    association: association("p", "3"),
                }],
            },
        )
        .expect("client should reconnect");
        let messages = bus.drain_client(&client).expect("projection should drain");
        assert_eq!(
            messages
                .iter()
                .map(|message| message.event_type.as_str())
                .collect::<Vec<_>>(),
            ["status", "executing", "executed"]
        );
        let stale = bus
            .publish_json(NativeJsonEvent {
                sequence: 3,
                event_type: "execution_start".into(),
                data: json!({"prompt_id":"p","timestamp":1}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::Runtime,
                association: association("p", "4"),
            })
            .expect("stale replay should be discarded");
        assert_eq!(stale.disposition, PublishDisposition::Duplicate);
    }

    #[test]
    fn val_ws_001_preview_negotiation_selects_metadata_frame_and_validates_association() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let legacy = test_client("legacy");
        let modern = test_client("modern");
        bus.connect(legacy.clone(), ReconnectProjection::default())
            .expect("legacy should connect");
        bus.connect(modern.clone(), ReconnectProjection::default())
            .expect("modern should connect");
        bus.drain_client(&legacy).expect("status should drain");
        bus.drain_client(&modern).expect("status should drain");
        bus.process_input_fragment(
            &modern,
            InputFragment {
                kind: FragmentKind::Text,
                bytes: br#"{"type":"feature_flags","data":{"supports_preview_metadata":true}}"#
                    .to_vec(),
                final_fragment: true,
            },
        )
        .expect("modern client should negotiate");
        bus.drain_client(&modern)
            .expect("feature response should drain");
        let legacy_report = bus
            .publish_preview(NativePreviewEvent {
                sequence: 1,
                format: PreviewImageFormat::Png,
                encoded_image: png(),
                target: EventTarget::Client(legacy.clone()),
                source: NativeEventSource::Runtime,
                association: association("p-legacy", "4"),
                metadata: Map::from_iter([("future".into(), Value::Bool(true))]),
            })
            .expect("preview should publish");
        let modern_report = bus
            .publish_preview(NativePreviewEvent {
                sequence: 1,
                format: PreviewImageFormat::Png,
                encoded_image: png(),
                target: EventTarget::Client(modern.clone()),
                source: NativeEventSource::Runtime,
                association: association("p-modern", "4"),
                metadata: Map::from_iter([("future".into(), Value::Bool(true))]),
            })
            .expect("metadata preview should publish");
        assert_eq!(legacy_report.delivered_clients, 1);
        assert_eq!(modern_report.delivered_clients, 1);
        let legacy_message = bus.drain_client(&legacy).expect("legacy should drain");
        let modern_message = bus.drain_client(&modern).expect("modern should drain");
        assert_eq!(legacy_message[0].event_type, "PREVIEW_IMAGE");
        assert_eq!(modern_message[0].event_type, "PREVIEW_IMAGE_WITH_METADATA");
        let decoded = decode_binary_message(&modern_message[0].payload, 1024)
            .expect("metadata preview should decode");
        let DecodedBinaryMessage::PreviewImageWithMetadata { metadata, .. } = decoded else {
            panic!("expected metadata preview");
        };
        assert_eq!(metadata.get("future"), Some(&Value::Bool(true)));
    }

    #[test]
    fn val_ws_001_conditional_events_require_native_source_enablement_and_subscription() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("conditional");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.drain_client(&client).expect("status should drain");

        let disabled = bus
            .publish_json(NativeJsonEvent {
                sequence: 1,
                event_type: "assets.seed.started".into(),
                data: json!({"roots":[]}),
                target: EventTarget::Broadcast,
                source: NativeEventSource::AssetSeeder,
                association: EventAssociation::default(),
            })
            .expect("disabled seeder event should be safely suppressed");
        assert_eq!(disabled.disposition, PublishDisposition::Suppressed);
        bus.set_asset_seeder_enabled(true);
        let enabled = bus
            .publish_json(NativeJsonEvent {
                sequence: 2,
                event_type: "assets.seed.started".into(),
                data: json!({"roots":[]}),
                target: EventTarget::Broadcast,
                source: NativeEventSource::AssetSeeder,
                association: EventAssociation::default(),
            })
            .expect("enabled seeder event should publish");
        assert_eq!(enabled.delivered_clients, 1);
        bus.drain_client(&client)
            .expect("seeder event should drain");

        let unsubscribed = bus
            .publish_json(NativeJsonEvent {
                sequence: 3,
                event_type: "logs".into(),
                data: json!({"entries":[],"size":0}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::TerminalService,
                association: EventAssociation::default(),
            })
            .expect("unsubscribed logs should be safely suppressed");
        assert_eq!(unsubscribed.disposition, PublishDisposition::Suppressed);
        bus.set_log_subscription(&client, true)
            .expect("log subscription should be stored");
        let subscribed = bus
            .publish_json(NativeJsonEvent {
                sequence: 4,
                event_type: "logs".into(),
                data: json!({"entries":[],"size":0}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::TerminalService,
                association: EventAssociation::default(),
            })
            .expect("subscribed logs should publish");
        assert_eq!(subscribed.delivered_clients, 1);

        assert!(matches!(
            bus.publish_json(NativeJsonEvent {
                sequence: 5,
                event_type: "logs".into(),
                data: json!({"entries":[],"size":0}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::Runtime,
                association: EventAssociation::default(),
            }),
            Err(WebSocketContractError::InvalidEventSource { .. })
        ));
        assert!(matches!(
            bus.publish_json(NativeJsonEvent {
                sequence: 5,
                event_type: "progress".into(),
                data: json!({"value":1,"max":2,"prompt_id":"p","node":"4"}),
                target: EventTarget::Broadcast,
                source: NativeEventSource::Runtime,
                association: association("p", "4"),
            }),
            Err(WebSocketContractError::InvalidEventTarget(_))
        ));
    }

    #[test]
    fn val_ws_001_malformed_binary_header_length_format_and_internal_code_are_rejected() {
        assert_eq!(legacy_preview_image_type_code("PNG"), 2);
        assert_eq!(legacy_preview_image_type_code("webp"), 1);
        let cases = [
            Vec::new(),
            PREVIEW_IMAGE_EVENT_CODE.to_be_bytes().to_vec(),
            [PREVIEW_IMAGE_EVENT_CODE.to_be_bytes(), 9u32.to_be_bytes()].concat(),
            [
                PREVIEW_IMAGE_EVENT_CODE.to_be_bytes().as_slice(),
                PreviewImageFormat::Png
                    .event_code()
                    .to_be_bytes()
                    .as_slice(),
                b"not-png",
            ]
            .concat(),
            [TEXT_EVENT_CODE.to_be_bytes(), 100u32.to_be_bytes()].concat(),
            [
                PREVIEW_IMAGE_WITH_METADATA_EVENT_CODE.to_be_bytes(),
                100u32.to_be_bytes(),
            ]
            .concat(),
            UNENCODED_PREVIEW_IMAGE_EVENT_CODE.to_be_bytes().to_vec(),
            99u32.to_be_bytes().to_vec(),
        ];
        for case in cases {
            assert!(decode_binary_message(&case, 1024).is_err(), "case {case:?}");
        }
        let mismatch = [
            PREVIEW_IMAGE_EVENT_CODE.to_be_bytes().as_slice(),
            PreviewImageFormat::Jpeg
                .event_code()
                .to_be_bytes()
                .as_slice(),
            png().as_slice(),
        ]
        .concat();
        assert!(decode_binary_message(&mismatch, 1024).is_err());
    }

    #[test]
    fn val_ws_001_bounded_queue_coalesces_only_matching_progress_associations() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits {
            max_message_bytes: 1024,
            max_queued_messages_per_client: 3,
            max_diagnostics: 8,
            max_clients: 8,
            max_server_features: 8,
            max_source_sequence_scopes: 8,
        })
        .expect("limits should be valid");
        let client = test_client("association-backpressure");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        let association_for = |attempt_id: &str, node_id: &str| EventAssociation {
            prompt_id: Some(format!("prompt-{attempt_id}")),
            node_id: Some(node_id.into()),
            attempt_id: Some(attempt_id.into()),
        };
        let publish_progress =
            |bus: &mut NativeWebSocketEventBus, sequence, attempt_id: &str, node_id: &str| {
                bus.publish_json(NativeJsonEvent {
                    sequence,
                    event_type: "progress".into(),
                    data: json!({
                        "value":sequence,
                        "max":3,
                        "prompt_id":format!("prompt-{attempt_id}"),
                        "node":node_id,
                    }),
                    target: EventTarget::Client(client.clone()),
                    source: NativeEventSource::Runtime,
                    association: association_for(attempt_id, node_id),
                })
            };

        publish_progress(&mut bus, 1, "a", "1").expect("first progress should queue");
        publish_progress(&mut bus, 1, "b", "2").expect("second progress should queue");
        publish_progress(&mut bus, 2, "a", "1")
            .expect("matching progress should coalesce at capacity");
        let unrelated = publish_progress(&mut bus, 1, "c", "3")
            .expect("unrelated progress backpressure should be reported safely");
        assert_eq!(unrelated.backpressured_clients, 1);

        let messages = bus.drain_client(&client).expect("messages should drain");
        assert_eq!(messages.len(), 3);
        let progress = messages
            .iter()
            .filter(|message| message.event_type == "progress")
            .collect::<Vec<_>>();
        assert_eq!(progress.len(), 2);
        assert!(progress.iter().any(|message| {
            message.association.attempt_id.as_deref() == Some("a")
                && message.source_sequence == Some(2)
        }));
        assert!(progress.iter().any(|message| {
            message.association.attempt_id.as_deref() == Some("b")
                && message.source_sequence == Some(1)
        }));
        assert!(
            progress
                .iter()
                .all(|message| message.association.attempt_id.as_deref() != Some("c"))
        );
    }

    #[test]
    fn val_ws_001_bounded_queue_coalesces_progress_and_disconnects_on_critical_backpressure() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits {
            max_message_bytes: 1024,
            max_queued_messages_per_client: 2,
            max_diagnostics: 8,
            max_clients: 8,
            max_server_features: 8,
            max_source_sequence_scopes: 8,
        })
        .expect("limits should be valid");
        let client = test_client("slow");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        bus.publish_json(NativeJsonEvent {
            sequence: 1,
            event_type: "progress".into(),
            data: json!({"value":1,"max":3,"prompt_id":"p","node":"4"}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association("p", "4"),
        })
        .expect("progress should publish");
        bus.publish_json(NativeJsonEvent {
            sequence: 2,
            event_type: "progress".into(),
            data: json!({"value":2,"max":3,"prompt_id":"p","node":"4"}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association("p", "4"),
        })
        .expect("progress should coalesce");
        let messages = bus.drain_client(&client).expect("messages should drain");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].source_sequence, Some(2));

        bus.publish_json(NativeJsonEvent {
            sequence: 3,
            event_type: "execution_start".into(),
            data: json!({"prompt_id":"p","timestamp":1}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association("p", "4"),
        })
        .expect("first critical event should publish");
        bus.publish_json(NativeJsonEvent {
            sequence: 4,
            event_type: "executed".into(),
            data: json!({"node":"4","display_node":"4","output":{},"prompt_id":"p"}),
            target: EventTarget::Client(client.clone()),
            source: NativeEventSource::Runtime,
            association: association("p", "4"),
        })
        .expect("second critical event should publish");
        let report = bus
            .publish_json(NativeJsonEvent {
                sequence: 5,
                event_type: "execution_success".into(),
                data: json!({"prompt_id":"p","timestamp":2}),
                target: EventTarget::Client(client.clone()),
                source: NativeEventSource::Runtime,
                association: association("p", "4"),
            })
            .expect("terminal backpressure should be handled");
        assert_eq!(report.backpressured_clients, 1);
        assert!(matches!(
            bus.drain_client(&client),
            Err(WebSocketContractError::ClientNotConnected(_))
        ));
    }

    #[test]
    fn val_ws_001_disconnect_cancel_and_shutdown_stop_delivery_without_affecting_other_state() {
        let mut bus = NativeWebSocketEventBus::new(WebSocketLimits::default())
            .expect("default limits should be valid");
        let client = test_client("lifecycle");
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should connect");
        assert!(bus.cancel_delivery(&client));
        assert!(
            bus.drain_client(&client)
                .expect("queue should drain")
                .is_empty()
        );
        assert!(bus.disconnect(&client));
        assert!(matches!(
            bus.process_input_fragment(
                &client,
                InputFragment {
                    kind: FragmentKind::Text,
                    bytes: b"{}".to_vec(),
                    final_fragment: true,
                }
            ),
            Err(WebSocketContractError::ClientNotConnected(_))
        ));
        bus.connect(client.clone(), ReconnectProjection::default())
            .expect("client should reconnect");
        bus.shutdown("native host stopped");
        assert!(bus.is_shutdown());
        let messages = bus.drain_client(&client).expect("close should drain");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].wire_kind, OutboundWireKind::Close);
        assert!(matches!(
            bus.publish_json(NativeJsonEvent {
                sequence: 1,
                event_type: "execution_success".into(),
                data: json!({"prompt_id":"p","timestamp":1}),
                target: EventTarget::Broadcast,
                source: NativeEventSource::Runtime,
                association: association("p", "4"),
            }),
            Err(WebSocketContractError::Shutdown)
        ));
    }

    #[test]
    fn val_ws_001_unknown_json_fields_are_retained_and_required_fields_are_checked() {
        let payload = json!({
            "value":1,"max":2,"prompt_id":"p","node":"4",
            "future":{"nested":true}
        });
        let encoded =
            encode_json_message("progress", payload.clone()).expect("unknown fields should encode");
        let (_, decoded) = decode_json_message(&encoded).expect("message should decode");
        assert_eq!(decoded, payload);
        assert!(
            encode_json_message("progress", json!({"value":1,"max":2,"prompt_id":"p"})).is_err()
        );
        assert!(matches!(
            encode_json_message("future.event", json!({})),
            Err(WebSocketContractError::UnknownEvent(_))
        ));
    }

    #[test]
    fn val_ws_001_progress_state_schema_is_bound_and_exhaustively_validated() {
        let descriptor = websocket_event_descriptor("progress_state")
            .expect("progress_state must have a production descriptor");
        assert_eq!(
            descriptor.payload_contract,
            WebSocketPayloadContract::ProgressState
        );
        let normative_contract = normative_websocket_catalog()
            .expect("normative WebSocket catalog should parse")
            .iter()
            .find(|contract| contract.feature_id == "COMFY-WS-012")
            .expect("progress_state must have a normative catalog row");
        assert_eq!(
            normative_contract.schema,
            descriptor.payload_contract.normative_schema()
        );

        let valid = json!({
            "prompt_id":"prompt-1",
            "nodes":{
                "4":{
                    "value":1.5,
                    "max":2,
                    "state":"running",
                    "node_id":"4",
                    "prompt_id":"prompt-1",
                    "display_node_id":"4",
                    "parent_node_id":null,
                    "real_node_id":"4",
                    "future":{"retained":true}
                },
                "5":{
                    "value":2,
                    "max":2.0,
                    "state":"finished",
                    "node_id":"5",
                    "prompt_id":"prompt-1",
                    "display_node_id":"5",
                    "parent_node_id":"4",
                    "real_node_id":"4"
                },
                "6":{
                    "value":0,
                    "max":1,
                    "state":"error",
                    "node_id":"6",
                    "prompt_id":"prompt-1",
                    "display_node_id":"6",
                    "parent_node_id":null,
                    "real_node_id":"6"
                }
            },
            "future":"retained"
        });
        let encoded = encode_json_message("progress_state", valid.clone())
            .expect("all source-emitted progress states should validate");
        assert_eq!(
            decode_json_message(&encoded).expect("valid progress_state should decode"),
            ("progress_state".into(), valid.clone())
        );
        encode_json_message("progress_state", json!({"prompt_id":"prompt-1","nodes":{}}))
            .expect("an empty node map is valid when all nodes are pending");

        for field in [
            "value",
            "max",
            "state",
            "node_id",
            "prompt_id",
            "display_node_id",
            "parent_node_id",
            "real_node_id",
        ] {
            let mut missing = valid.clone();
            missing["nodes"]["4"]
                .as_object_mut()
                .expect("fixture node should be an object")
                .remove(field);
            assert!(
                encode_json_message("progress_state", missing).is_err(),
                "missing nodes.*.{field} must be rejected"
            );
        }

        for (field, wrong_value) in [
            ("value", json!("1")),
            ("max", Value::Null),
            ("state", json!(1)),
            ("node_id", json!(4)),
            ("prompt_id", Value::Null),
            ("display_node_id", Value::Null),
            ("parent_node_id", json!(false)),
            ("real_node_id", Value::Null),
        ] {
            let mut wrong_type = valid.clone();
            wrong_type["nodes"]["4"][field] = wrong_value;
            assert!(
                encode_json_message("progress_state", wrong_type).is_err(),
                "invalid nodes.*.{field} type must be rejected"
            );
        }

        for state in ["pending", "unknown", ""] {
            let mut invalid_state = valid.clone();
            invalid_state["nodes"]["4"]["state"] = json!(state);
            assert!(
                encode_json_message("progress_state", invalid_state).is_err(),
                "state {state:?} must not be emitted in progress_state"
            );
        }

        let mut mismatched_node = valid.clone();
        mismatched_node["nodes"]["4"]["node_id"] = json!("different");
        assert!(encode_json_message("progress_state", mismatched_node).is_err());
        let mut mismatched_prompt = valid.clone();
        mismatched_prompt["nodes"]["4"]["prompt_id"] = json!("different");
        assert!(encode_json_message("progress_state", mismatched_prompt).is_err());
        let mut non_object_node = valid;
        non_object_node["nodes"]["4"] = json!(1);
        assert!(encode_json_message("progress_state", non_object_node).is_err());
        assert!(
            encode_json_message("progress_state", json!({"prompt_id":"prompt-1","nodes":[]}),)
                .is_err()
        );
    }
}
