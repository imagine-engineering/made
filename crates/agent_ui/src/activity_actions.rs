use acp_thread::{AuthorizationKind, ToolCall, ToolCallStatus};
use agent_client_protocol::schema::v1 as acp;
use chrono::{DateTime, Utc};
use gpui::App;

use crate::activity_projection::{
    ActivityActor, ActivityContext, ActivityDetailHandle, ActivityItem, ActivityItemId,
    ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind, ActivityOutcome,
    ActivityOutcomeStatus, ActivityProjectionContractError, ActivitySemanticClass,
    ActivitySourceKind, ActivityVisibility,
};

#[derive(Clone, Debug)]
pub struct NativeActionProjectionContext {
    pub session_id: acp::SessionId,
    pub agent_actor: ActivityActor,
    pub context: ActivityContext,
    pub visibility: ActivityVisibility,
    pub occurred_at: DateTime<Utc>,
    pub projected_at: DateTime<Utc>,
}

impl NativeActionProjectionContext {
    fn activity_context(&self) -> ActivityContext {
        let mut context = self.context.clone();
        context.session_id = Some(self.session_id.0.to_string());
        context
    }
}

pub fn project_native_tool_activity(
    projection_context: &NativeActionProjectionContext,
    source_version: u64,
    tool_call: &ToolCall,
    cx: &App,
) -> Result<ActivityItem, ActivityProjectionContractError> {
    let action_id = format!(
        "{}/tool/{}",
        projection_context.session_id.0, tool_call.id.0
    );
    let label = tool_call.label.read(cx).source().trim().to_owned();
    let label = if label.is_empty() {
        tool_call
            .tool_name
            .as_ref()
            .map_or_else(|| "tool call".into(), |name| name.to_string())
    } else {
        label
    };
    let semantics = action_semantics(tool_call.kind, &tool_call.status, &label);
    let id = ActivityItemId::new(ActivitySourceKind::NativeAction, action_id.clone())?;

    Ok(ActivityItem {
        id,
        source_version,
        class: semantics.class,
        actor: projection_context.agent_actor.clone(),
        verb: semantics.verb.into(),
        object: ActivityObject {
            kind: semantics.object_kind,
            id: Some(tool_call.id.0.to_string()),
            label,
        },
        outcome: semantics.outcome,
        lifecycle: semantics.lifecycle,
        occurred_at: projection_context.occurred_at,
        projected_at: projection_context.projected_at,
        context: projection_context.activity_context(),
        visibility: projection_context.visibility,
        details: Some(ActivityDetailHandle::AcpEntry {
            session_id: projection_context.session_id.0.to_string(),
            entry_id: action_id.clone(),
        }),
        links: vec![ActivityLink::Action { action_id }],
    })
}

struct NativeActionSemantics {
    class: ActivitySemanticClass,
    verb: &'static str,
    object_kind: ActivityObjectKind,
    lifecycle: ActivityLifecycle,
    outcome: ActivityOutcome,
}

fn action_semantics(
    kind: acp::ToolKind,
    status: &ToolCallStatus,
    label: &str,
) -> NativeActionSemantics {
    if let ToolCallStatus::WaitingForConfirmation { kind, .. } = status {
        return waiting_semantics(*kind);
    }
    if matches!(status, ToolCallStatus::Rejected) {
        return NativeActionSemantics {
            class: ActivitySemanticClass::Permission,
            verb: "was denied permission for",
            object_kind: ActivityObjectKind::Permission,
            lifecycle: ActivityLifecycle::Failed,
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some("Permission rejected".into()),
            },
        };
    }

    let (class, verb, object_kind) = kind_semantics(kind, label);
    let (lifecycle, outcome) = status_semantics(status);
    NativeActionSemantics {
        class,
        verb,
        object_kind,
        lifecycle,
        outcome,
    }
}

fn kind_semantics(
    kind: acp::ToolKind,
    label: &str,
) -> (ActivitySemanticClass, &'static str, ActivityObjectKind) {
    match kind {
        acp::ToolKind::Read => (
            ActivitySemanticClass::Generic,
            "read",
            ActivityObjectKind::File,
        ),
        acp::ToolKind::Edit => (
            ActivitySemanticClass::FileEdit,
            "edited",
            ActivityObjectKind::File,
        ),
        acp::ToolKind::Delete => (
            ActivitySemanticClass::FileEdit,
            "deleted",
            ActivityObjectKind::File,
        ),
        acp::ToolKind::Move => (
            ActivitySemanticClass::FileEdit,
            "moved",
            ActivityObjectKind::File,
        ),
        acp::ToolKind::Search => (
            ActivitySemanticClass::Generic,
            "searched",
            ActivityObjectKind::Tool,
        ),
        acp::ToolKind::Execute if is_test_command(label) => (
            ActivitySemanticClass::ShellCommand,
            "tested",
            ActivityObjectKind::TestSuite,
        ),
        acp::ToolKind::Execute => (
            ActivitySemanticClass::ShellCommand,
            "ran",
            ActivityObjectKind::Command,
        ),
        acp::ToolKind::Think => (
            ActivitySemanticClass::Thought,
            "thought about",
            ActivityObjectKind::Plan,
        ),
        acp::ToolKind::Fetch => (
            ActivitySemanticClass::Generic,
            "fetched",
            ActivityObjectKind::Tool,
        ),
        acp::ToolKind::SwitchMode => (
            ActivitySemanticClass::Lifecycle,
            "switched",
            ActivityObjectKind::Session,
        ),
        acp::ToolKind::Other => (
            ActivitySemanticClass::Generic,
            "used",
            ActivityObjectKind::Tool,
        ),
        _ => (
            ActivitySemanticClass::Generic,
            "used",
            ActivityObjectKind::Tool,
        ),
    }
}

fn status_semantics(status: &ToolCallStatus) -> (ActivityLifecycle, ActivityOutcome) {
    match status {
        ToolCallStatus::Pending => (
            ActivityLifecycle::Pending,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Pending".into()),
            },
        ),
        ToolCallStatus::InProgress => (
            ActivityLifecycle::Running,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Running".into()),
            },
        ),
        ToolCallStatus::Completed => (
            ActivityLifecycle::Succeeded,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: Some("Completed".into()),
            },
        ),
        ToolCallStatus::Failed => (
            ActivityLifecycle::Failed,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some("Tool failed".into()),
            },
        ),
        ToolCallStatus::Canceled => (
            ActivityLifecycle::Cancelled,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Cancelled,
                summary: Some("Cancelled".into()),
            },
        ),
        ToolCallStatus::Rejected => (
            ActivityLifecycle::Failed,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some("Permission rejected".into()),
            },
        ),
        ToolCallStatus::WaitingForConfirmation { .. } => (
            ActivityLifecycle::WaitingForUser,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Waiting for user".into()),
            },
        ),
    }
}

fn waiting_semantics(kind: AuthorizationKind) -> NativeActionSemantics {
    let (verb, summary) = match kind {
        AuthorizationKind::PermissionGrant => ("requested permission for", "Permission required"),
        AuthorizationKind::ActionChoice => ("requested a choice for", "Decision required"),
    };
    NativeActionSemantics {
        class: ActivitySemanticClass::Permission,
        verb,
        object_kind: ActivityObjectKind::Permission,
        lifecycle: ActivityLifecycle::WaitingForUser,
        outcome: ActivityOutcome {
            status: ActivityOutcomeStatus::Pending,
            summary: Some(summary.into()),
        },
    }
}

fn is_test_command(label: &str) -> bool {
    let command = label.trim_start();
    [
        "cargo test",
        "cargo nextest",
        "pytest",
        "python -m pytest",
        "npm test",
        "npm run test",
        "pnpm test",
        "pnpm run test",
        "yarn test",
        "go test",
        "swift test",
        "dotnet test",
    ]
    .iter()
    .any(|prefix| {
        command == *prefix
            || command
                .strip_prefix(prefix)
                .is_some_and(|suffix| suffix.starts_with(char::is_whitespace))
    })
}

#[cfg(test)]
mod tests {
    use acp_thread::{PermissionOptions, ToolCallStatus};
    use futures::channel::oneshot;
    use gpui::{AppContext as _, TestAppContext};
    use markdown::Markdown;

    use super::*;
    use crate::activity_projection::{ActivityActorKind, ActivitySemanticClass};

    fn projection_context() -> NativeActionProjectionContext {
        let timestamp = DateTime::parse_from_rfc3339("2026-08-14T12:00:00Z")
            .expect("test timestamp should parse")
            .with_timezone(&Utc);
        NativeActionProjectionContext {
            session_id: acp::SessionId::new("session-1"),
            agent_actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Agent".into(),
            },
            context: ActivityContext {
                project_id: Some("project-1".into()),
                thread_id: Some("thread-1".into()),
                ..ActivityContext::default()
            },
            visibility: ActivityVisibility::Project,
            occurred_at: timestamp,
            projected_at: timestamp,
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
    fn activity_action_mapping_maps_every_current_tool_kind_or_generic_fallback(
        cx: &mut TestAppContext,
    ) {
        let fixtures = [
            (
                acp::ToolKind::Read,
                "src/main.rs",
                ActivitySemanticClass::Generic,
                ActivityObjectKind::File,
            ),
            (
                acp::ToolKind::Edit,
                "src/main.rs",
                ActivitySemanticClass::FileEdit,
                ActivityObjectKind::File,
            ),
            (
                acp::ToolKind::Delete,
                "src/old.rs",
                ActivitySemanticClass::FileEdit,
                ActivityObjectKind::File,
            ),
            (
                acp::ToolKind::Move,
                "src/old.rs → src/new.rs",
                ActivitySemanticClass::FileEdit,
                ActivityObjectKind::File,
            ),
            (
                acp::ToolKind::Search,
                "ActivityItem",
                ActivitySemanticClass::Generic,
                ActivityObjectKind::Tool,
            ),
            (
                acp::ToolKind::Execute,
                "cargo test -p agent_ui",
                ActivitySemanticClass::ShellCommand,
                ActivityObjectKind::TestSuite,
            ),
            (
                acp::ToolKind::Think,
                "Review the ownership model",
                ActivitySemanticClass::Thought,
                ActivityObjectKind::Plan,
            ),
            (
                acp::ToolKind::Fetch,
                "Fetch issue details",
                ActivitySemanticClass::Generic,
                ActivityObjectKind::Tool,
            ),
            (
                acp::ToolKind::SwitchMode,
                "Switch to planning",
                ActivitySemanticClass::Lifecycle,
                ActivityObjectKind::Session,
            ),
            (
                acp::ToolKind::Other,
                "custom operation",
                ActivitySemanticClass::Generic,
                ActivityObjectKind::Tool,
            ),
        ];

        cx.update(|cx| {
            for (index, (kind, label, expected_class, expected_object_kind)) in
                fixtures.into_iter().enumerate()
            {
                let call = tool_call(
                    &format!("tool-{index}"),
                    kind,
                    ToolCallStatus::Completed,
                    label,
                    cx,
                );
                let item = project_native_tool_activity(&projection_context(), 1, &call, cx)
                    .expect("registered tool kind should project");
                assert_eq!(item.class, expected_class);
                assert_eq!(item.object.kind, expected_object_kind);
                assert_eq!(item.lifecycle, ActivityLifecycle::Succeeded);
                assert_eq!(item.outcome.status, ActivityOutcomeStatus::Success);
            }
        });
    }

    #[gpui::test]
    fn activity_action_mapping_maps_status_and_permission_outcomes(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let fixtures = [
                (
                    ToolCallStatus::Pending,
                    ActivityLifecycle::Pending,
                    ActivityOutcomeStatus::Pending,
                ),
                (
                    ToolCallStatus::InProgress,
                    ActivityLifecycle::Running,
                    ActivityOutcomeStatus::Pending,
                ),
                (
                    ToolCallStatus::Completed,
                    ActivityLifecycle::Succeeded,
                    ActivityOutcomeStatus::Success,
                ),
                (
                    ToolCallStatus::Failed,
                    ActivityLifecycle::Failed,
                    ActivityOutcomeStatus::Failure,
                ),
                (
                    ToolCallStatus::Rejected,
                    ActivityLifecycle::Failed,
                    ActivityOutcomeStatus::Failure,
                ),
                (
                    ToolCallStatus::Canceled,
                    ActivityLifecycle::Cancelled,
                    ActivityOutcomeStatus::Cancelled,
                ),
            ];
            for (index, (status, expected_lifecycle, expected_outcome)) in
                fixtures.into_iter().enumerate()
            {
                let call = tool_call(
                    &format!("status-{index}"),
                    acp::ToolKind::Execute,
                    status,
                    "cargo check",
                    cx,
                );
                let item = project_native_tool_activity(&projection_context(), 1, &call, cx)
                    .expect("tool status should project");
                assert_eq!(item.lifecycle, expected_lifecycle);
                assert_eq!(item.outcome.status, expected_outcome);
            }

            for (index, authorization_kind) in [
                AuthorizationKind::PermissionGrant,
                AuthorizationKind::ActionChoice,
            ]
            .into_iter()
            .enumerate()
            {
                let (respond_tx, _respond_rx) = oneshot::channel();
                let call = tool_call(
                    &format!("permission-{index}"),
                    acp::ToolKind::Edit,
                    ToolCallStatus::WaitingForConfirmation {
                        current_status: acp::ToolCallStatus::Pending,
                        options: PermissionOptions::Flat(Vec::new()),
                        respond_tx,
                        kind: authorization_kind,
                    },
                    "src/main.rs",
                    cx,
                );
                let item = project_native_tool_activity(&projection_context(), 1, &call, cx)
                    .expect("permission status should project");
                assert_eq!(item.class, ActivitySemanticClass::Permission);
                assert_eq!(item.object.kind, ActivityObjectKind::Permission);
                assert_eq!(item.lifecycle, ActivityLifecycle::WaitingForUser);
                assert_eq!(item.outcome.status, ActivityOutcomeStatus::Pending);
            }
        });
    }

    #[gpui::test]
    fn activity_action_mapping_updates_one_stable_item_and_links_canonical_action(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let pending = tool_call(
                "tool-1",
                acp::ToolKind::Edit,
                ToolCallStatus::Pending,
                "src/main.rs",
                cx,
            );
            let completed = tool_call(
                "tool-1",
                acp::ToolKind::Edit,
                ToolCallStatus::Completed,
                "src/main.rs",
                cx,
            );
            let pending = project_native_tool_activity(&projection_context(), 1, &pending, cx)
                .expect("pending tool should project");
            let completed = project_native_tool_activity(&projection_context(), 2, &completed, cx)
                .expect("completed tool should project");

            assert_eq!(pending.id, completed.id);
            assert_ne!(pending.source_version, completed.source_version);
            assert_eq!(completed.links.len(), 1);
            assert!(matches!(completed.links[0], ActivityLink::Action { .. }));
        });
    }

    #[test]
    fn activity_action_mapping_detects_test_commands_without_prefix_ambiguity() {
        for command in [
            "cargo test",
            "cargo test -p agent_ui",
            "python -m pytest tests/test_api.py",
            "pnpm run test --watch",
            "go test ./...",
        ] {
            assert!(is_test_command(command), "expected test command: {command}");
        }
        for command in ["cargo testable", "pytester", "go tester", "cargo check"] {
            assert!(
                !is_test_command(command),
                "unexpected test command: {command}"
            );
        }
    }
}
