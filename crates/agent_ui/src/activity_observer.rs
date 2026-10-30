use std::{error::Error, fmt};

use chrono::{DateTime, Utc};
use nostr_compat::agent_observer::{AgentObserverIngress, AgentObserverTelemetryKind};

use crate::activity_projection::{
    ActivityActor, ActivityContext, ActivityDetailHandle, ActivityItem, ActivityItemId,
    ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind, ActivityOutcome,
    ActivityOutcomeStatus, ActivityProjectionContractError, ActivitySemanticClass,
    ActivitySourceKind, ActivityVisibility,
};

#[derive(Clone, Debug)]
pub struct ObserverActivityProjectionContext {
    pub event_id: String,
    pub agent_actor: ActivityActor,
    pub context: ActivityContext,
    pub visibility: ActivityVisibility,
    pub projected_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObserverActivityProjectionError {
    Contract(ActivityProjectionContractError),
    InvalidEventId,
    InvalidTimestamp,
    TelemetryKindMismatch,
    SessionContextMismatch,
}

impl fmt::Display for ObserverActivityProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => error.fmt(formatter),
            Self::InvalidEventId => formatter.write_str("observer event ID must not be empty"),
            Self::InvalidTimestamp => {
                formatter.write_str("observer telemetry timestamp is invalid")
            }
            Self::TelemetryKindMismatch => {
                formatter.write_str("observer telemetry kind does not match parsed ingress")
            }
            Self::SessionContextMismatch => {
                formatter.write_str("observer telemetry session conflicts with activity context")
            }
        }
    }
}

impl Error for ObserverActivityProjectionError {}

impl From<ActivityProjectionContractError> for ObserverActivityProjectionError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Contract(error)
    }
}

pub fn project_agent_observer_activity(
    projection_context: &ObserverActivityProjectionContext,
    ingress: &AgentObserverIngress,
) -> Result<Option<ActivityItem>, ObserverActivityProjectionError> {
    let AgentObserverIngress::Telemetry {
        kind,
        channel_id,
        frame,
    } = ingress
    else {
        return Ok(None);
    };
    if projection_context.event_id.trim().is_empty() {
        return Err(ObserverActivityProjectionError::InvalidEventId);
    }
    if frame.kind != telemetry_kind_name(*kind) {
        return Err(ObserverActivityProjectionError::TelemetryKindMismatch);
    }

    let occurred_at = DateTime::parse_from_rfc3339(&frame.timestamp)
        .map_err(|_| ObserverActivityProjectionError::InvalidTimestamp)?
        .with_timezone(&Utc);
    let mut context = projection_context.context.clone();
    if let Some(session_id) = &frame.session_id {
        if context
            .session_id
            .as_ref()
            .is_some_and(|current| current != session_id)
        {
            return Err(ObserverActivityProjectionError::SessionContextMismatch);
        }
        context.session_id = Some(session_id.clone());
    }

    let semantics = observer_semantics(*kind, &frame.payload);
    let source_id = match kind {
        AgentObserverTelemetryKind::TurnStarted | AgentObserverTelemetryKind::SessionResolved => {
            lifecycle_source_id(
                &projection_context.event_id,
                &projection_context.agent_actor.id,
                frame.session_id.as_deref(),
                frame.turn_id.as_deref(),
            )
        }
        AgentObserverTelemetryKind::AcpRead | AgentObserverTelemetryKind::AcpWrite => {
            format!("observer/event/{}", projection_context.event_id)
        }
    };
    let id = ActivityItemId::new(ActivitySourceKind::Nostr, source_id)?;
    let object_id = frame.turn_id.clone().or_else(|| frame.session_id.clone());
    let mut links = Vec::new();
    if let Some(channel_id) = channel_id {
        links.push(ActivityLink::Entity {
            entity_kind: "channel".into(),
            entity_id: channel_id.to_string(),
        });
    }
    if let Some(session_id) = &frame.session_id {
        links.push(ActivityLink::Entity {
            entity_kind: "session".into(),
            entity_id: session_id.clone(),
        });
    }

    Ok(Some(ActivityItem {
        id,
        source_version: frame.seq,
        class: semantics.class,
        actor: projection_context.agent_actor.clone(),
        verb: semantics.verb.into(),
        object: ActivityObject {
            kind: semantics.object_kind,
            id: object_id,
            label: semantics.object_label.into(),
        },
        outcome: semantics.outcome,
        lifecycle: semantics.lifecycle,
        occurred_at,
        projected_at: projection_context.projected_at,
        context,
        visibility: projection_context.visibility,
        details: Some(ActivityDetailHandle::ProtocolEvent {
            event_id: projection_context.event_id.clone(),
        }),
        links,
    }))
}

struct ObserverSemantics {
    class: ActivitySemanticClass,
    verb: &'static str,
    object_kind: ActivityObjectKind,
    object_label: &'static str,
    outcome: ActivityOutcome,
    lifecycle: ActivityLifecycle,
}

fn observer_semantics(
    kind: AgentObserverTelemetryKind,
    payload: &serde_json::Value,
) -> ObserverSemantics {
    match kind {
        AgentObserverTelemetryKind::AcpRead => protocol_semantics("received"),
        AgentObserverTelemetryKind::AcpWrite => protocol_semantics("sent"),
        AgentObserverTelemetryKind::TurnStarted => ObserverSemantics {
            class: ActivitySemanticClass::Lifecycle,
            verb: "started",
            object_kind: ActivityObjectKind::Session,
            object_label: "agent turn",
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Agent turn started".into()),
            },
            lifecycle: ActivityLifecycle::Running,
        },
        AgentObserverTelemetryKind::SessionResolved => resolution_semantics(payload),
    }
}

fn protocol_semantics(verb: &'static str) -> ObserverSemantics {
    ObserverSemantics {
        class: ActivitySemanticClass::Generic,
        verb,
        object_kind: ActivityObjectKind::Other,
        object_label: "ACP protocol activity",
        outcome: ActivityOutcome {
            status: ActivityOutcomeStatus::Pending,
            summary: None,
        },
        lifecycle: ActivityLifecycle::Running,
    }
}

fn resolution_semantics(payload: &serde_json::Value) -> ObserverSemantics {
    let stop_reason = payload
        .as_object()
        .and_then(|payload| payload.get("stopReason"))
        .and_then(serde_json::Value::as_str);
    let (verb, status, lifecycle, summary) = match stop_reason {
        Some("end_turn") => (
            "completed",
            ActivityOutcomeStatus::Success,
            ActivityLifecycle::Succeeded,
            "Agent turn completed",
        ),
        Some("cancelled") => (
            "cancelled",
            ActivityOutcomeStatus::Cancelled,
            ActivityLifecycle::Cancelled,
            "Agent turn was cancelled",
        ),
        Some("max_tokens") => (
            "stopped",
            ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Failed,
            "Agent turn reached the token limit",
        ),
        Some("max_turn_requests") => (
            "stopped",
            ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Failed,
            "Agent turn reached the request limit",
        ),
        Some("refusal") => (
            "refused",
            ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Failed,
            "Agent refused to continue",
        ),
        Some("error") => (
            "failed",
            ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Failed,
            "Agent turn failed",
        ),
        _ => (
            "resolved",
            ActivityOutcomeStatus::Unknown,
            ActivityLifecycle::Succeeded,
            "Agent turn resolved",
        ),
    };
    ObserverSemantics {
        class: ActivitySemanticClass::Lifecycle,
        verb,
        object_kind: ActivityObjectKind::Session,
        object_label: "agent turn",
        outcome: ActivityOutcome {
            status,
            summary: Some(summary.into()),
        },
        lifecycle,
    }
}

fn lifecycle_source_id(
    event_id: &str,
    agent_id: &str,
    session_id: Option<&str>,
    turn_id: Option<&str>,
) -> String {
    match (session_id, turn_id) {
        (Some(session_id), Some(turn_id)) => {
            format!("observer/agent/{agent_id}/session/{session_id}/turn/{turn_id}")
        }
        _ => format!("observer/event/{event_id}"),
    }
}

const fn telemetry_kind_name(kind: AgentObserverTelemetryKind) -> &'static str {
    match kind {
        AgentObserverTelemetryKind::AcpRead => "acp_read",
        AgentObserverTelemetryKind::AcpWrite => "acp_write",
        AgentObserverTelemetryKind::TurnStarted => "turn_started",
        AgentObserverTelemetryKind::SessionResolved => "session_resolved",
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;
    use nostr_compat::agent_observer::AgentObserverIngress;
    use nostr_compat::buzz_nips::agent_activity::ObserverTelemetry;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::activity_projection::ActivityActorKind;
    use crate::activity_reducer::{ActivityReducer, ActivityReduction};

    fn projection_context(event_id: &str) -> ObserverActivityProjectionContext {
        ObserverActivityProjectionContext {
            event_id: event_id.into(),
            agent_actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Agent".into(),
            },
            context: ActivityContext {
                community_id: Some("community-1".into()),
                project_id: Some("project-1".into()),
                ..ActivityContext::default()
            },
            visibility: ActivityVisibility::Private,
            projected_at: Utc
                .with_ymd_and_hms(2026, 8, 23, 12, 0, 1)
                .single()
                .expect("valid fixture timestamp"),
        }
    }

    fn telemetry(
        kind: AgentObserverTelemetryKind,
        sequence: u64,
        payload: serde_json::Value,
    ) -> AgentObserverIngress {
        let channel_id = Uuid::from_u128(10);
        AgentObserverIngress::Telemetry {
            kind,
            channel_id: Some(channel_id),
            frame: ObserverTelemetry {
                seq: sequence,
                timestamp: "2026-08-23T12:00:00.500Z".into(),
                kind: telemetry_kind_name(kind).into(),
                agent_index: Some(0),
                channel_id: Some(channel_id.to_string()),
                session_id: Some("session-1".into()),
                turn_id: Some("turn-1".into()),
                payload,
            },
        }
    }

    #[test]
    fn activity_observer_maps_every_supported_kind_once_without_payload_content() {
        let fixtures = [
            (
                AgentObserverTelemetryKind::AcpRead,
                ActivitySemanticClass::Generic,
                ActivityLifecycle::Running,
            ),
            (
                AgentObserverTelemetryKind::AcpWrite,
                ActivitySemanticClass::Generic,
                ActivityLifecycle::Running,
            ),
            (
                AgentObserverTelemetryKind::TurnStarted,
                ActivitySemanticClass::Lifecycle,
                ActivityLifecycle::Running,
            ),
            (
                AgentObserverTelemetryKind::SessionResolved,
                ActivitySemanticClass::Lifecycle,
                ActivityLifecycle::Succeeded,
            ),
        ];

        for (index, (kind, class, lifecycle)) in fixtures.into_iter().enumerate() {
            let ingress = telemetry(
                kind,
                index as u64 + 1,
                json!({
                    "stopReason": "end_turn",
                    "ciphertext": "encrypted-secret-sentinel",
                    "params": { "prompt": "private-prompt-sentinel" }
                }),
            );
            let item = project_agent_observer_activity(
                &projection_context(&format!("event-{index}")),
                &ingress,
            )
            .expect("fixture should project")
            .expect("telemetry should produce one item");
            assert_eq!(item.class, class);
            assert_eq!(item.lifecycle, lifecycle);
            let serialized = serde_json::to_string(&item).expect("activity item should serialize");
            assert!(!serialized.contains("encrypted-secret-sentinel"));
            assert!(!serialized.contains("private-prompt-sentinel"));
            assert!(matches!(
                item.details,
                Some(ActivityDetailHandle::ProtocolEvent { .. })
            ));
        }
    }

    #[test]
    fn activity_observer_reduces_started_and_resolved_frames_to_one_turn() {
        let started = project_agent_observer_activity(
            &projection_context("event-started"),
            &telemetry(
                AgentObserverTelemetryKind::TurnStarted,
                10,
                json!({ "type": "turn_started" }),
            ),
        )
        .expect("started frame should project")
        .expect("started frame should be visible");
        let resolved = project_agent_observer_activity(
            &projection_context("event-resolved"),
            &telemetry(
                AgentObserverTelemetryKind::SessionResolved,
                11,
                json!({ "stopReason": "cancelled" }),
            ),
        )
        .expect("resolved frame should project")
        .expect("resolved frame should be visible");
        assert_eq!(started.id, resolved.id);

        let mut reducer = ActivityReducer::new();
        assert_eq!(
            reducer.reduce(started),
            Ok(ActivityReduction::Inserted { index: 0 })
        );
        assert_eq!(
            reducer.reduce(resolved),
            Ok(ActivityReduction::Updated { index: 0 })
        );
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].lifecycle, ActivityLifecycle::Cancelled);
    }

    #[test]
    fn activity_observer_maps_resolution_outcomes_without_exposing_unknown_values() {
        let fixtures = [
            (
                "max_tokens",
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                "max_turn_requests",
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                "refusal",
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                "future-secret-reason",
                ActivityLifecycle::Succeeded,
                ActivityOutcomeStatus::Unknown,
            ),
        ];
        for (index, (reason, lifecycle, status)) in fixtures.into_iter().enumerate() {
            let item = project_agent_observer_activity(
                &projection_context(&format!("resolution-{index}")),
                &telemetry(
                    AgentObserverTelemetryKind::SessionResolved,
                    index as u64 + 1,
                    json!({ "stopReason": reason }),
                ),
            )
            .expect("resolution should project")
            .expect("resolution should be visible");
            assert_eq!(item.lifecycle, lifecycle);
            assert_eq!(item.outcome.status, status);
            assert!(
                !item
                    .outcome
                    .summary
                    .as_deref()
                    .is_some_and(|summary| summary.contains(reason))
            );
        }
    }

    #[test]
    fn activity_observer_ignores_controls_and_rejects_mismatched_frames() {
        let channel_id = Uuid::from_u128(10);
        assert_eq!(
            project_agent_observer_activity(
                &projection_context("control-event"),
                &AgentObserverIngress::CancelTurn { channel_id },
            ),
            Ok(None)
        );
        assert_eq!(
            project_agent_observer_activity(
                &projection_context("future-event"),
                &AgentObserverIngress::Ignored,
            ),
            Ok(None)
        );

        let mut mismatched = telemetry(
            AgentObserverTelemetryKind::AcpRead,
            1,
            json!({ "method": "session/update" }),
        );
        let AgentObserverIngress::Telemetry { frame, .. } = &mut mismatched else {
            panic!("fixture should be telemetry")
        };
        frame.kind = "acp_write".into();
        assert_eq!(
            project_agent_observer_activity(&projection_context("mismatch"), &mismatched),
            Err(ObserverActivityProjectionError::TelemetryKindMismatch)
        );
    }
}
