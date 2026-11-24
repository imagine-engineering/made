use acp_thread::{
    CollaborationObserverAdapter, CollaborationObserverContext, CollaborationObserverError,
    CollaborationObserverPublish, NativeCollaborationObserverEvent, ToolCallStatus,
};
use agent_client_protocol::schema::v1 as acp;
use chrono::{TimeZone, Utc};
use serde_json::json;
use uuid::Uuid;

fn adapter() -> CollaborationObserverAdapter {
    CollaborationObserverAdapter::new(
        CollaborationObserverContext::new(
            Uuid::parse_str("52a85618-0f8f-4542-94ec-599e6e1c6f2e").expect("valid UUID"),
            acp::SessionId::new("native-session"),
            Some(3),
        )
        .expect("valid observer context"),
    )
}

fn timestamp(second: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 23, 12, 0, second)
        .single()
        .expect("valid timestamp")
}

fn frame(
    publish: CollaborationObserverPublish,
) -> nostr_compat::buzz_nips::agent_activity::ObserverTelemetry {
    match publish {
        CollaborationObserverPublish::Frame(frame) => frame,
        CollaborationObserverPublish::Duplicate => panic!("expected an observer frame"),
    }
}

#[test]
fn streams_protocol_and_action_updates_with_one_stable_item() {
    let mut adapter = adapter();
    let started = frame(
        adapter
            .publish(
                "source-1",
                "turn-1",
                timestamp(0),
                NativeCollaborationObserverEvent::TurnStarted,
            )
            .expect("turn starts"),
    );
    let protocol = json!({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "session/update",
        "params": { "content": "private transcript" },
    });
    let read = frame(
        adapter
            .publish(
                "source-2",
                "turn-1",
                timestamp(1),
                NativeCollaborationObserverEvent::ProtocolRead(&protocol),
            )
            .expect("protocol read streams"),
    );
    let action_id = acp::ToolCallId::new("action-1");
    let pending = frame(
        adapter
            .publish(
                "source-3",
                "turn-1",
                timestamp(2),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::Pending,
                },
            )
            .expect("pending action streams"),
    );
    let completed = frame(
        adapter
            .publish(
                "source-4",
                "turn-1",
                timestamp(3),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::Completed,
                },
            )
            .expect("terminal action update streams"),
    );

    assert_eq!((started.seq, started.kind.as_str()), (1, "turn_started"));
    assert_eq!((read.seq, read.kind.as_str()), (2, "acp_read"));
    assert_eq!(read.payload["method"], "session/update");
    assert_eq!(pending.payload["actionId"], completed.payload["actionId"]);
    assert_eq!(pending.payload["status"], "pending");
    assert_eq!(completed.payload["status"], "completed");
    assert_eq!(completed.seq, 4);
}

#[test]
fn publishes_terminal_and_cancelled_outcomes_once() {
    let mut completed_adapter = adapter();
    let completed = frame(
        completed_adapter
            .publish(
                "terminal-1",
                "turn-completed",
                timestamp(0),
                NativeCollaborationObserverEvent::SessionResolved(acp::StopReason::EndTurn),
            )
            .expect("completion streams"),
    );
    assert_eq!(completed.kind, "session_resolved");
    assert_eq!(completed.payload["stopReason"], "end_turn");
    assert_eq!(
        completed_adapter
            .publish(
                "terminal-2",
                "turn-completed",
                timestamp(1),
                NativeCollaborationObserverEvent::SessionResolved(acp::StopReason::EndTurn),
            )
            .expect_err("a turn has one terminal outcome"),
        CollaborationObserverError::TurnAlreadyResolved,
    );

    let mut cancelled_adapter = adapter();
    let cancelled = frame(
        cancelled_adapter
            .publish(
                "cancel-1",
                "turn-cancelled",
                timestamp(2),
                NativeCollaborationObserverEvent::SessionResolved(acp::StopReason::Cancelled),
            )
            .expect("cancellation streams"),
    );
    assert_eq!(cancelled.payload["stopReason"], "cancelled");
}

#[test]
fn redacts_protocol_content_and_debug_output() {
    let mut adapter = adapter();
    let protocol = json!({
        "jsonrpc": "2.0",
        "id": "secret-request-id",
        "method": "session/prompt",
        "params": {
            "prompt": "never publish this transcript",
            "authorization": "never publish this token",
        },
        "result": { "content": "never publish this response" },
    });
    let publish = adapter
        .publish(
            "redaction-source",
            "turn-redacted",
            timestamp(0),
            NativeCollaborationObserverEvent::ProtocolWritten(&protocol),
        )
        .expect("redacted frame streams");
    let serialized = serde_json::to_string(publish.frame().expect("observer frame serializes"))
        .expect("serialize observer frame");
    let debug = format!("{publish:?}");

    for secret in [
        "secret-request-id",
        "never publish this transcript",
        "never publish this token",
        "never publish this response",
    ] {
        assert!(!serialized.contains(secret));
        assert!(!debug.contains(secret));
    }
    assert!(debug.contains("<redacted>"));
}

#[test]
fn suppresses_transport_and_semantic_retries_without_consuming_sequence() {
    let mut adapter = adapter();
    let action_id = acp::ToolCallId::new("action-retry");
    let first = frame(
        adapter
            .publish(
                "retry-source",
                "turn-retry",
                timestamp(0),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::InProgress,
                },
            )
            .expect("first update streams"),
    );
    assert!(matches!(
        adapter
            .publish(
                "retry-source",
                "turn-retry",
                timestamp(1),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::InProgress,
                },
            )
            .expect("transport retry is idempotent"),
        CollaborationObserverPublish::Duplicate
    ));
    assert!(matches!(
        adapter
            .publish(
                "semantic-retry-source",
                "turn-retry",
                timestamp(2),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::InProgress,
                },
            )
            .expect("semantic retry is idempotent"),
        CollaborationObserverPublish::Duplicate
    ));
    let terminal = frame(
        adapter
            .publish(
                "retry-terminal",
                "turn-retry",
                timestamp(3),
                NativeCollaborationObserverEvent::ActionUpdated {
                    action_id: &action_id,
                    status: &ToolCallStatus::Completed,
                },
            )
            .expect("terminal update streams"),
    );
    assert_eq!(first.seq, 1);
    assert_eq!(terminal.seq, 2);
}
