use std::collections::HashSet;

use acp_thread::{
    AssistantMessage, AssistantMessageChunk, AuthorizationKind, ContentBlock, PermissionOptions,
    ToolCall, ToolCallStatus, UserMessage,
};
use agent_client_protocol::schema::v1 as acp;
use agent_ui::{
    activity_acp::{
        AcpActivityProjectionContext, AcpLifecycleActivity, AcpLifecycleKind,
        project_acp_assistant_message, project_acp_lifecycle, project_acp_user_message,
    },
    activity_actions::{NativeActionProjectionContext, project_native_tool_activity},
    activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityLifecycle,
        ActivitySemanticClass, ActivityVisibility,
    },
    activity_reducer::{ActivityReducer, ActivityReduction},
};
use chrono::{DateTime, Utc};
use futures::channel::oneshot;
use gpui::{AppContext as _, TestAppContext};
use markdown::Markdown;

const PROTOCOL_MANIFEST: &str =
    include_str!("../../../.agents/specs/collaborative-workspace/fixtures/protocol/manifest.json");

fn timestamp() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2026-08-14T12:00:00Z")
        .expect("fixture timestamp should parse")
        .with_timezone(&Utc)
}

fn actor(kind: ActivityActorKind, id: &str, label: &str) -> ActivityActor {
    ActivityActor {
        kind,
        id: id.into(),
        label: label.into(),
    }
}

fn acp_context() -> AcpActivityProjectionContext {
    AcpActivityProjectionContext {
        session_id: acp::SessionId::new("session-fixture"),
        human_actor: actor(ActivityActorKind::Human, "human-1", "Human"),
        agent_actor: actor(ActivityActorKind::Agent, "agent-1", "Agent"),
        context: ActivityContext {
            project_id: Some("project-1".into()),
            thread_id: Some("thread-1".into()),
            ..ActivityContext::default()
        },
        visibility: ActivityVisibility::Project,
        occurred_at: timestamp(),
        projected_at: timestamp(),
    }
}

fn action_context() -> NativeActionProjectionContext {
    NativeActionProjectionContext {
        session_id: acp::SessionId::new("session-fixture"),
        agent_actor: actor(ActivityActorKind::Agent, "agent-1", "Agent"),
        context: ActivityContext {
            project_id: Some("project-1".into()),
            thread_id: Some("thread-1".into()),
            ..ActivityContext::default()
        },
        visibility: ActivityVisibility::Project,
        occurred_at: timestamp(),
        projected_at: timestamp(),
    }
}

fn resource(uri: &str) -> ContentBlock {
    ContentBlock::ResourceLink {
        resource_link: acp::ResourceLink::new("fixture", uri),
    }
}

fn tool_call(
    id: &str,
    kind: acp::ToolKind,
    status: ToolCallStatus,
    label: &str,
    cx: &mut gpui::App,
) -> ToolCall {
    ToolCall {
        id: acp::ToolCallId::new(id),
        label: cx.new(|cx| Markdown::new_text(label.to_owned().into(), cx)),
        kind,
        content: Vec::new(),
        status,
        locations: Vec::new(),
        resolved_locations: Vec::new(),
        raw_input: None,
        raw_input_markdown: None,
        raw_output: None,
        tool_name: None,
        subagent_session_info: None,
        sandbox_authorization_details: None,
        sandbox_fallback_authorization_details: None,
        sandbox_not_applied: None,
    }
}

#[gpui::test]
fn collaborative_activity_projects_every_milestone_one_source_exactly_once(
    cx: &mut TestAppContext,
) {
    cx.update(|cx| {
        let human = UserMessage {
            protocol_id: Some(acp::MessageId::new("human-message-1")),
            client_id: None,
            is_optimistic: false,
            content: resource("file:///project/prompt.md"),
            chunks: Vec::new(),
            checkpoint: None,
            indented: false,
        };
        let assistant = AssistantMessage {
            chunks: vec![
                AssistantMessageChunk::Message {
                    id: Some(acp::MessageId::new("assistant-message-1")),
                    block: resource("file:///project/result.md"),
                },
                AssistantMessageChunk::Thought {
                    id: Some(acp::MessageId::new("assistant-thought-1")),
                    block: resource("file:///project/thought.md"),
                },
            ],
            indented: false,
            is_subagent_output: false,
        };

        let mut fixtures = vec![(
            "human_message",
            project_acp_user_message(&acp_context(), 0, 1, &human, cx)
                .expect("human message should project"),
        )];
        let mut assistant_items =
            project_acp_assistant_message(&acp_context(), 1, 1, &assistant, cx)
                .expect("assistant chunks should project")
                .into_iter();
        fixtures.push((
            "assistant_message",
            assistant_items
                .next()
                .expect("message chunk should produce one item"),
        ));
        fixtures.push((
            "assistant_thought",
            assistant_items
                .next()
                .expect("thought chunk should produce one item"),
        ));
        assert!(
            assistant_items.next().is_none(),
            "assistant fixture should not produce extra rows"
        );

        let tool_fixtures = [
            ("tool_read", acp::ToolKind::Read, "src/main.rs"),
            ("tool_edit", acp::ToolKind::Edit, "src/main.rs"),
            ("tool_delete", acp::ToolKind::Delete, "src/old.rs"),
            ("tool_move", acp::ToolKind::Move, "src/old.rs → src/new.rs"),
            ("tool_search", acp::ToolKind::Search, "ActivityItem"),
            ("tool_execute", acp::ToolKind::Execute, "cargo build"),
            ("tool_think", acp::ToolKind::Think, "next step"),
            ("tool_fetch", acp::ToolKind::Fetch, "https://example.test"),
            ("tool_switch_mode", acp::ToolKind::SwitchMode, "review"),
            ("tool_other", acp::ToolKind::Other, "custom tool"),
        ];
        for (name, kind, label) in tool_fixtures {
            let call = tool_call(name, kind, ToolCallStatus::Completed, label, cx);
            fixtures.push((
                name,
                project_native_tool_activity(&action_context(), 1, &call, cx)
                    .expect("registered tool kind should project"),
            ));
        }

        for (name, kind) in [
            ("lifecycle_started", AcpLifecycleKind::Started),
            ("lifecycle_idle", AcpLifecycleKind::Idle),
            (
                "lifecycle_completed",
                AcpLifecycleKind::Stopped(acp::StopReason::EndTurn),
            ),
        ] {
            fixtures.push((
                name,
                project_acp_lifecycle(
                    &acp_context(),
                    1,
                    &AcpLifecycleActivity {
                        event_id: name.into(),
                        kind,
                    },
                )
                .expect("lifecycle event should project"),
            ));
        }

        let expected_names = HashSet::from([
            "human_message",
            "assistant_message",
            "assistant_thought",
            "tool_read",
            "tool_edit",
            "tool_delete",
            "tool_move",
            "tool_search",
            "tool_execute",
            "tool_think",
            "tool_fetch",
            "tool_switch_mode",
            "tool_other",
            "lifecycle_started",
            "lifecycle_idle",
            "lifecycle_completed",
        ]);
        assert_eq!(
            fixtures
                .iter()
                .map(|(name, _)| *name)
                .collect::<HashSet<_>>(),
            expected_names,
            "the Milestone 1 source catalog must have no unmapped fixture"
        );

        let mut reducer = ActivityReducer::new();
        for (name, item) in &fixtures {
            assert!(
                matches!(
                    reducer
                        .reduce(item.clone())
                        .expect("first delivery should reduce"),
                    ActivityReduction::Inserted { .. }
                ),
                "{name} should insert once"
            );
            assert!(
                matches!(
                    reducer
                        .reduce(item.clone())
                        .expect("duplicate delivery should reduce"),
                    ActivityReduction::Duplicate { .. }
                ),
                "{name} should deduplicate on repeat delivery"
            );
        }
        assert_eq!(reducer.items().len(), fixtures.len());
        assert_eq!(
            reducer
                .items()
                .iter()
                .map(|item| item.id.clone())
                .collect::<HashSet<_>>()
                .len(),
            fixtures.len(),
            "every fixture must own a distinct canonical source identity"
        );
    });
}

#[gpui::test]
fn collaborative_activity_covers_empty_waiting_disconnected_and_error_states(
    cx: &mut TestAppContext,
) {
    let reducer = ActivityReducer::new();
    assert!(
        reducer.items().is_empty(),
        "an empty thread projects no rows"
    );

    cx.update(|cx| {
        let (respond_tx, _respond_rx) = oneshot::channel();
        let waiting_call = tool_call(
            "permission-1",
            acp::ToolKind::Execute,
            ToolCallStatus::WaitingForConfirmation {
                current_status: acp::ToolCallStatus::Pending,
                options: PermissionOptions::Flat(Vec::new()),
                respond_tx,
                kind: AuthorizationKind::PermissionGrant,
            },
            "cargo test",
            cx,
        );
        let waiting = project_native_tool_activity(&action_context(), 1, &waiting_call, cx)
            .expect("permission request should project");
        assert_eq!(waiting.class, ActivitySemanticClass::Permission);
        assert_eq!(waiting.lifecycle, ActivityLifecycle::WaitingForUser);

        let disconnected = project_acp_lifecycle(
            &acp_context(),
            1,
            &AcpLifecycleActivity {
                event_id: "disconnect-1".into(),
                kind: AcpLifecycleKind::Disconnected {
                    reason: Some("relay unavailable".into()),
                },
            },
        )
        .expect("disconnect should project");
        assert_eq!(disconnected.lifecycle, ActivityLifecycle::Disconnected);
        assert_eq!(
            disconnected.outcome.summary.as_deref(),
            Some("relay unavailable")
        );

        let failed = project_acp_lifecycle(
            &acp_context(),
            1,
            &AcpLifecycleActivity {
                event_id: "error-1".into(),
                kind: AcpLifecycleKind::Failed {
                    message: "agent process exited".into(),
                },
            },
        )
        .expect("failure should project");
        assert_eq!(failed.class, ActivitySemanticClass::Error);
        assert_eq!(failed.lifecycle, ActivityLifecycle::Failed);
        assert_eq!(
            failed.outcome.summary.as_deref(),
            Some("agent process exited")
        );
    });
}

#[test]
fn collaborative_activity_keeps_protocol_compatibility_fixtures_versioned() {
    let manifest: serde_json::Value =
        serde_json::from_str(PROTOCOL_MANIFEST).expect("protocol fixture manifest should parse");
    assert_eq!(manifest["schema_version"], 1);
    assert!(
        manifest["mixed_version_cases"]
            .as_array()
            .is_some_and(|cases| !cases.is_empty()),
        "compatibility fixtures must retain mixed-version coverage"
    );
    assert_eq!(
        manifest["authority"]["event_serialization"],
        "projects/buzz/crates/buzz-core/src/verification.rs"
    );
}
