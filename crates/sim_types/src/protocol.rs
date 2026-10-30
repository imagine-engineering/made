use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const MAX_COMPATIBILITY_JSON_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RouteIdentity {
    pub method: HttpMethod,
    pub path: String,
    pub canonical_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouteContract {
    pub feature_id: String,
    pub identity: RouteIdentity,
    #[serde(default)]
    pub path_parameters: Vec<String>,
    #[serde(default)]
    pub query_parameters: Vec<String>,
    #[serde(default)]
    pub request_headers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(default)]
    pub content_types: Vec<String>,
    #[serde(default)]
    pub status_codes: Vec<u16>,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub permissions: Vec<String>,
    #[serde(default)]
    pub feature_gates: Vec<String>,
    #[serde(default)]
    pub side_effects: Vec<String>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityRequest {
    pub route: RouteIdentity,
    #[serde(default)]
    pub path_parameters: BTreeMap<String, String>,
    #[serde(default)]
    pub query: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub headers: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityResponse {
    pub status: u16,
    pub content_type: String,
    #[serde(default)]
    pub headers: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WebSocketEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Value,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinaryWebSocketFrame {
    PreviewImage { image_type: u32, bytes: Vec<u8> },
    UnencodedPreviewImage { bytes: Vec<u8> },
    Text { node_id: Vec<u8>, text: Vec<u8> },
    PreviewImageWithMetadata { metadata: Value, bytes: Vec<u8> },
    Unknown { code: u32, bytes: Vec<u8> },
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ProtocolError {
    #[error("frame is shorter than its required header")]
    Truncated,
    #[error("declared field length {declared} exceeds remaining frame bytes {remaining}")]
    InvalidLength { declared: usize, remaining: usize },
    #[error("metadata is not valid UTF-8 JSON: {0}")]
    InvalidMetadata(String),
    #[error("compatibility payload exceeds {MAX_COMPATIBILITY_JSON_BYTES} bytes")]
    Oversized,
    #[error("compatibility JSON is malformed: {0}")]
    InvalidJson(String),
}

pub fn decode_compatibility_json<'a, T>(bytes: &'a [u8]) -> Result<T, ProtocolError>
where
    T: Deserialize<'a>,
{
    if bytes.len() > MAX_COMPATIBILITY_JSON_BYTES {
        return Err(ProtocolError::Oversized);
    }
    serde_json::from_slice(bytes).map_err(|error| ProtocolError::InvalidJson(error.to_string()))
}

fn read_u32(bytes: &[u8]) -> Result<u32, ProtocolError> {
    let raw: [u8; 4] = bytes
        .get(..4)
        .ok_or(ProtocolError::Truncated)?
        .try_into()
        .map_err(|_| ProtocolError::Truncated)?;
    Ok(u32::from_be_bytes(raw))
}

fn split_sized(bytes: &[u8]) -> Result<(&[u8], &[u8]), ProtocolError> {
    let declared = read_u32(bytes)? as usize;
    let payload = bytes.get(4..).ok_or(ProtocolError::Truncated)?;
    if declared > payload.len() {
        return Err(ProtocolError::InvalidLength {
            declared,
            remaining: payload.len(),
        });
    }
    Ok(payload.split_at(declared))
}

pub fn decode_binary_websocket_frame(bytes: &[u8]) -> Result<BinaryWebSocketFrame, ProtocolError> {
    if bytes.len() > MAX_COMPATIBILITY_JSON_BYTES {
        return Err(ProtocolError::Oversized);
    }
    let code = read_u32(bytes)?;
    let payload = bytes.get(4..).ok_or(ProtocolError::Truncated)?;
    match code {
        1 => Ok(BinaryWebSocketFrame::PreviewImage {
            image_type: read_u32(payload)?,
            bytes: payload.get(4..).ok_or(ProtocolError::Truncated)?.to_vec(),
        }),
        2 => Ok(BinaryWebSocketFrame::UnencodedPreviewImage {
            bytes: payload.to_vec(),
        }),
        3 => {
            let (node_id, text) = split_sized(payload)?;
            Ok(BinaryWebSocketFrame::Text {
                node_id: node_id.to_vec(),
                text: text.to_vec(),
            })
        }
        4 => {
            let (metadata, image) = split_sized(payload)?;
            let metadata = serde_json::from_slice(metadata)
                .map_err(|error| ProtocolError::InvalidMetadata(error.to_string()))?;
            Ok(BinaryWebSocketFrame::PreviewImageWithMetadata {
                metadata,
                bytes: image.to_vec(),
            })
        }
        _ => Ok(BinaryWebSocketFrame::Unknown {
            code,
            bytes: payload.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_event_round_trip_preserves_unknown_fields() {
        let source = br#"{"type":"progress","data":{"value":1},"future":true}"#;
        let event: WebSocketEvent = serde_json::from_slice(source).expect("valid event");
        assert_eq!(event.unknown["future"], true);
        assert_eq!(
            serde_json::to_value(event).expect("serializable")["future"],
            true
        );
    }

    #[test]
    fn all_cataloged_binary_codes_decode() {
        let preview = [0, 0, 0, 1, 0, 0, 0, 2, 9];
        assert!(matches!(
            decode_binary_websocket_frame(&preview),
            Ok(BinaryWebSocketFrame::PreviewImage { image_type: 2, .. })
        ));
        assert_eq!(
            decode_binary_websocket_frame(&[0, 0, 0, 2, 9]),
            Ok(BinaryWebSocketFrame::UnencodedPreviewImage { bytes: vec![9] })
        );
        let text = [0, 0, 0, 3, 0, 0, 0, 1, b'n', b'o', b'k'];
        assert!(matches!(
            decode_binary_websocket_frame(&text),
            Ok(BinaryWebSocketFrame::Text { .. })
        ));
        let metadata = br#"{"image_type":2,"node_id":"4"}"#;
        let mut frame = 4_u32.to_be_bytes().to_vec();
        frame.extend_from_slice(&(metadata.len() as u32).to_be_bytes());
        frame.extend_from_slice(metadata);
        frame.push(9);
        assert!(matches!(
            decode_binary_websocket_frame(&frame),
            Ok(BinaryWebSocketFrame::PreviewImageWithMetadata { .. })
        ));
    }

    #[test]
    fn malformed_and_unknown_binary_frames_are_safe() {
        assert_eq!(
            decode_binary_websocket_frame(&[0, 0]),
            Err(ProtocolError::Truncated)
        );
        let unknown = [0, 0, 0, 99, 1, 2];
        assert_eq!(
            decode_binary_websocket_frame(&unknown),
            Ok(BinaryWebSocketFrame::Unknown {
                code: 99,
                bytes: vec![1, 2]
            })
        );
    }

    #[test]
    fn route_alias_and_schema_extensions_round_trip() {
        let source = br#"{
            "feature_id":"COMFY-API-0002",
            "identity":{"method":"GET","path":"/api/","canonical_path":"/","alias_of":"GET /"},
            "status_codes":[200],"content_types":["text/html"],"future_schema":{"v":2}
        }"#;
        let route: RouteContract = decode_compatibility_json(source).expect("valid route");
        assert_eq!(route.identity.alias_of.as_deref(), Some("GET /"));
        assert_eq!(route.unknown["future_schema"]["v"], 2);
    }

    #[test]
    fn bounded_json_rejects_malformed_and_oversized_input() {
        assert!(matches!(
            decode_compatibility_json::<WebSocketEvent>(b"{"),
            Err(ProtocolError::InvalidJson(_))
        ));
        let oversized = vec![b' '; MAX_COMPATIBILITY_JSON_BYTES + 1];
        assert_eq!(
            decode_compatibility_json::<Value>(&oversized),
            Err(ProtocolError::Oversized)
        );
    }

    #[test]
    fn generated_protocol_catalogs_are_covered_by_generic_contracts() {
        let routes =
            include_str!("../../../.agents/specs/comfy-parity/catalogs/backend-http-routes.csv");
        let mut route_count = 0;
        for line in routes.lines().skip(1) {
            let method = line.split(',').next().expect("method column");
            assert!(matches!(
                method,
                "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD"
            ));
            route_count += 1;
        }
        assert_eq!(route_count, 141);

        let events = include_str!(
            "../../../.agents/specs/comfy-parity/catalogs/backend-websocket-events.csv"
        );
        let mut event_count = 0;
        for line in events.lines().skip(1) {
            let mut columns = line.splitn(7, ',');
            let _feature_id = columns.next().expect("feature id column");
            let _product = columns.next().expect("product column");
            let _direction = columns.next().expect("direction column");
            let wire_kind = columns.next().expect("wire kind column");
            let _event_type = columns.next().expect("event type column");
            let binary_code = columns.next().expect("binary code column");
            columns.next().expect("schema and remaining columns");
            if wire_kind == "binary" || wire_kind == "internal event" {
                assert!(matches!(binary_code, "1" | "2" | "3" | "4"));
            }
            event_count += 1;
        }
        assert_eq!(event_count, 26);

        let formats =
            include_str!("../../../.agents/specs/comfy-parity/catalogs/cross-formats.csv");
        assert_eq!(formats.lines().skip(1).count(), 34);
    }
}
