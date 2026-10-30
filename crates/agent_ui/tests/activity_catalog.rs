#![cfg(feature = "multiplayer-tools")]

use std::{collections::HashSet, num::NonZeroU32};

use acp_thread::{
    AssistantMessage, AssistantMessageChunk, ContentBlock, ToolCall, ToolCallStatus, UserMessage,
};
use agent_client_protocol::schema::v1 as acp;
use agent_ui::{
    activity_acp::{
        AcpActivityProjectionContext, AcpLifecycleActivity, AcpLifecycleKind,
        project_acp_assistant_message, project_acp_lifecycle, project_acp_user_message,
    },
    activity_actions::{NativeActionProjectionContext, project_native_tool_activity},
    activity_collaboration::{
        CollaborationActivityScope, CollaborationActorPresentation, JobActivityContext,
        MessageActivityContext, project_job_activity, project_message_activity,
    },
    activity_git::{
        BranchCodeActivity, BranchCodeActivityKind, CodeActivityProjectionContext,
        CollaborationCodeActivity, GenericCodeActivity, ReviewDecisionActivity,
        project_code_activity,
    },
    activity_observer::{ObserverActivityProjectionContext, project_agent_observer_activity},
    activity_platform::{
        PLATFORM_ACTIVITY_SCHEMA_VERSION, PlatformActivity, PlatformActivityProjectionContext,
        PlatformActivityRecord, PlatformEventKind, RegisteredPlatformEventKind,
        project_platform_activity,
    },
    activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityItem, ActivitySourceKind,
        ActivityVisibility,
    },
    activity_reducer::{ActivityReducer, ActivityReduction},
};
use chrono::{DateTime, Utc};
use collaboration_domain::{
    AggregateId, AggregateVersion, BranchCollaborationIdentity, BranchGeneration, BranchRefName,
    BranchUpdateKind, CiCheckRunCompletionInput, CiCheckRunInput, CiCheckStatus, CiCheckSuite,
    CiCheckSuiteIdentity, CiLabel, CiOutputText, CiWorkflowLink, CommunityId, GitCommitId, Job,
    JobCommand, JobCommandKind, JobIdentity, Message, MessageAuthor, MessageContent,
    MessageLifecycleState, MessageRecordFields, MessageSource, NostrEventId, OperationId,
    PatchRevision, PatchRevisionNumber, PrincipalId, ReviewApproval, ReviewComment,
    ReviewCommentAnchor, ReviewCommentBody, ReviewDecision, ReviewDiffSide, ReviewFilePath,
    ReviewHunkId, ReviewIdentity,
};
use gpui::{AppContext as _, TestAppContext};
use markdown::Markdown;
use nostr_compat::{
    agent_observer::{AgentObserverIngress, AgentObserverTelemetryKind},
    buzz_nips::agent_activity::ObserverTelemetry,
};
use serde_json::{Value, json};
use uuid::Uuid;

const PROTOCOL_EVENTS: &str =
    include_str!("../../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json");
const PROTOCOL_MANIFEST: &str =
    include_str!("../../../.agents/specs/collaborative-workspace/fixtures/protocol/manifest.json");

struct CatalogFixture {
    name: String,
    family: &'static str,
    item: ActivityItem,
}

fn timestamp() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2026-08-24T12:00:00Z")
        .expect("fixture timestamp should parse")
        .with_timezone(&Utc)
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn version(value: u64) -> AggregateVersion {
    AggregateVersion::new(value).expect("fixture version should be positive")
}

fn actor(kind: ActivityActorKind, id: &str, label: &str) -> ActivityActor {
    ActivityActor {
        kind,
        id: id.into(),
        label: label.into(),
    }
}

fn collaboration_actor(
    value: u128,
    kind: ActivityActorKind,
    label: &str,
) -> CollaborationActorPresentation {
    CollaborationActorPresentation {
        principal_id: principal(value),
        kind,
        label: label.into(),
    }
}

fn acp_context() -> AcpActivityProjectionContext {
    AcpActivityProjectionContext {
        session_id: acp::SessionId::new("catalog-session"),
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
        session_id: acp::SessionId::new("catalog-session"),
        agent_actor: actor(ActivityActorKind::Agent, "agent-1", "Agent"),
        context: acp_context().context,
        visibility: ActivityVisibility::Project,
        occurred_at: timestamp(),
        projected_at: timestamp(),
    }
}

fn resource(uri: &str) -> ContentBlock {
    ContentBlock::ResourceLink {
        resource_link: acp::ResourceLink::new("catalog fixture", uri),
    }
}

fn tool_call(id: &str, kind: acp::ToolKind, label: &str, cx: &mut gpui::App) -> ToolCall {
    ToolCall {
        id: acp::ToolCallId::new(id),
        label: cx.new(|cx| Markdown::new_text(label.to_owned().into(), cx)),
        kind,
        content: Vec::new(),
        status: ToolCallStatus::Completed,
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

fn push_acp_fixtures(catalog: &mut Vec<CatalogFixture>, cx: &mut gpui::App) {
    let user = UserMessage {
        protocol_id: Some(acp::MessageId::new("catalog-human-message")),
        client_id: None,
        is_optimistic: false,
        content: resource("file:///project/prompt.md"),
        chunks: Vec::new(),
        checkpoint: None,
        indented: false,
    };
    catalog.push(CatalogFixture {
        name: "acp_human_message".into(),
        family: "acp",
        item: project_acp_user_message(&acp_context(), 0, 1, &user, cx)
            .expect("ACP user message should project"),
    });

    let assistant = AssistantMessage {
        chunks: vec![
            AssistantMessageChunk::Message {
                id: Some(acp::MessageId::new("catalog-assistant-message")),
                block: resource("file:///project/result.md"),
            },
            AssistantMessageChunk::Thought {
                id: Some(acp::MessageId::new("catalog-assistant-thought")),
                block: resource("file:///project/thought.md"),
            },
        ],
        indented: false,
        is_subagent_output: false,
    };
    let assistant_items = project_acp_assistant_message(&acp_context(), 1, 1, &assistant, cx)
        .expect("ACP assistant chunks should project");
    assert_eq!(assistant_items.len(), 2);
    for (name, item) in ["acp_assistant_message", "acp_assistant_thought"]
        .into_iter()
        .zip(assistant_items)
    {
        catalog.push(CatalogFixture {
            name: name.into(),
            family: "acp",
            item,
        });
    }

    for (name, kind, label) in [
        ("acp_tool_read", acp::ToolKind::Read, "src/main.rs"),
        ("acp_tool_edit", acp::ToolKind::Edit, "src/main.rs"),
        ("acp_tool_delete", acp::ToolKind::Delete, "src/old.rs"),
        (
            "acp_tool_move",
            acp::ToolKind::Move,
            "src/old.rs → src/new.rs",
        ),
        ("acp_tool_search", acp::ToolKind::Search, "ActivityItem"),
        ("acp_tool_execute", acp::ToolKind::Execute, "cargo check"),
        ("acp_tool_think", acp::ToolKind::Think, "next step"),
        (
            "acp_tool_fetch",
            acp::ToolKind::Fetch,
            "https://example.test",
        ),
        ("acp_tool_switch_mode", acp::ToolKind::SwitchMode, "review"),
        ("acp_tool_other", acp::ToolKind::Other, "custom tool"),
    ] {
        let call = tool_call(name, kind, label, cx);
        catalog.push(CatalogFixture {
            name: name.into(),
            family: "acp",
            item: project_native_tool_activity(&action_context(), 1, &call, cx)
                .expect("ACP tool fixture should project"),
        });
    }

    for (name, kind) in [
        ("acp_lifecycle_started", AcpLifecycleKind::Started),
        ("acp_lifecycle_idle", AcpLifecycleKind::Idle),
        (
            "acp_lifecycle_completed",
            AcpLifecycleKind::Stopped(acp::StopReason::EndTurn),
        ),
    ] {
        catalog.push(CatalogFixture {
            name: name.into(),
            family: "acp",
            item: project_acp_lifecycle(
                &acp_context(),
                1,
                &AcpLifecycleActivity {
                    event_id: name.into(),
                    kind,
                },
            )
            .expect("ACP lifecycle fixture should project"),
        });
    }
}

fn observer_kind_name(kind: AgentObserverTelemetryKind) -> &'static str {
    match kind {
        AgentObserverTelemetryKind::AcpRead => "acp_read",
        AgentObserverTelemetryKind::AcpWrite => "acp_write",
        AgentObserverTelemetryKind::TurnStarted => "turn_started",
        AgentObserverTelemetryKind::SessionResolved => "session_resolved",
    }
}

fn push_nip_ao_fixtures(catalog: &mut Vec<CatalogFixture>) {
    for (index, kind) in [
        AgentObserverTelemetryKind::AcpRead,
        AgentObserverTelemetryKind::AcpWrite,
        AgentObserverTelemetryKind::TurnStarted,
        AgentObserverTelemetryKind::SessionResolved,
    ]
    .into_iter()
    .enumerate()
    {
        let fixture_name = format!("nip_ao_{}", observer_kind_name(kind));
        let channel_id = Uuid::from_u128(1_000 + index as u128);
        let ingress = AgentObserverIngress::Telemetry {
            kind,
            channel_id: Some(channel_id),
            frame: ObserverTelemetry {
                seq: index as u64 + 1,
                timestamp: "2026-08-24T12:00:00Z".into(),
                kind: observer_kind_name(kind).into(),
                agent_index: Some(0),
                channel_id: Some(channel_id.to_string()),
                session_id: Some(format!("observer-session-{index}")),
                turn_id: Some(format!("observer-turn-{index}")),
                payload: match kind {
                    AgentObserverTelemetryKind::SessionResolved => {
                        json!({ "stopReason": "end_turn" })
                    }
                    _ => json!({}),
                },
            },
        };
        let item = project_agent_observer_activity(
            &ObserverActivityProjectionContext {
                event_id: format!("observer-event-{index}"),
                agent_actor: actor(ActivityActorKind::Agent, "agent-1", "Agent"),
                context: ActivityContext {
                    community_id: Some("community-1".into()),
                    project_id: Some("project-1".into()),
                    ..ActivityContext::default()
                },
                visibility: ActivityVisibility::Private,
                projected_at: timestamp(),
            },
            &ingress,
        )
        .expect("NIP-AO fixture should be valid")
        .expect("NIP-AO telemetry should produce one record");
        catalog.push(CatalogFixture {
            name: fixture_name,
            family: "nip_ao",
            item,
        });
    }
}

fn decode_event_id(value: &str) -> NostrEventId {
    assert_eq!(value.len(), 64, "fixture event ID should contain 32 bytes");
    let mut decoded = [0_u8; 32];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = char::from(pair[0])
            .to_digit(16)
            .expect("fixture event ID should be hexadecimal");
        let low = char::from(pair[1])
            .to_digit(16)
            .expect("fixture event ID should be hexadecimal");
        *output = u8::try_from(high * 16 + low).expect("two hexadecimal digits fit in one byte");
    }
    NostrEventId::from_bytes(decoded)
}

fn push_message_fixtures(catalog: &mut Vec<CatalogFixture>) {
    let manifest: Value =
        serde_json::from_str(PROTOCOL_MANIFEST).expect("protocol manifest should parse");
    let accepted = manifest["event_cases"]
        .as_array()
        .expect("protocol event cases should be an array")
        .iter()
        .filter(|case| case["expected"] == "accept")
        .map(|case| {
            case["event"]
                .as_str()
                .expect("accepted event should have a name")
                .to_owned()
        })
        .collect::<HashSet<_>>();
    assert_eq!(
        accepted,
        HashSet::from(["legacy_message".to_owned(), "v2_message".to_owned()])
    );
    let events: Value =
        serde_json::from_str(PROTOCOL_EVENTS).expect("protocol events should parse");
    let author = principal(2_000);
    for (index, name) in ["legacy_message", "v2_message"].into_iter().enumerate() {
        let event = &events["events"][name];
        let source = MessageSource {
            event_id: decode_event_id(
                event["id"]
                    .as_str()
                    .expect("message fixture should have an event ID"),
            ),
            event_created_at: event["created_at"]
                .as_u64()
                .expect("message fixture should have a timestamp"),
        };
        let message = Message::from_record(MessageRecordFields {
            community_id: CommunityId::from_uuid(Uuid::from_u128(2_001)),
            channel_id: aggregate(2_002),
            message_id: aggregate(2_100 + index as u128),
            author: MessageAuthor::principal(author),
            content: MessageContent::new(
                event["content"]
                    .as_str()
                    .expect("message fixture should have content"),
            )
            .expect("message fixture content should be valid"),
            lifecycle_state: MessageLifecycleState::Active,
            source,
            current_source: source,
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("canonical message fixture should hydrate");
        let item = project_message_activity(
            &message,
            &MessageActivityContext {
                author: CollaborationActorPresentation {
                    principal_id: author,
                    kind: ActivityActorKind::Human,
                    label: "Protocol fixture author".into(),
                },
                reply_to_event_id: None,
                projected_at: timestamp(),
                scope: CollaborationActivityScope::default(),
            },
        )
        .expect("canonical message fixture should project");
        catalog.push(CatalogFixture {
            name: format!("message_{name}"),
            family: "message",
            item,
        });
    }
}

fn commit(value: u64) -> GitCommitId {
    GitCommitId::parse(format!("{value:040x}")).expect("fixture commit should be valid")
}

fn branch() -> BranchCollaborationIdentity {
    BranchCollaborationIdentity::new(
        CommunityId::from_uuid(Uuid::from_u128(3_000)),
        aggregate(3_001),
        BranchRefName::parse("refs/heads/feature/activity-catalog")
            .expect("fixture branch should be valid"),
        BranchGeneration::FIRST,
    )
    .expect("fixture branch identity should be valid")
}

fn review_identity() -> ReviewIdentity {
    ReviewIdentity::new(aggregate(3_002), branch()).expect("fixture review should be valid")
}

fn revision() -> PatchRevision {
    PatchRevision {
        revision_id: aggregate(3_003),
        review: review_identity(),
        number: PatchRevisionNumber::FIRST,
        base_commit: commit(1),
        head_commit: commit(2),
        author_principal_id: principal(3_004),
        created_at_millis: 1_900_000_000_000,
    }
}

fn review_comment() -> ReviewComment {
    ReviewComment {
        comment_id: aggregate(3_005),
        review: review_identity(),
        author_principal_id: principal(3_006),
        body: ReviewCommentBody::new("Preserve the activity projection boundary")
            .expect("fixture review body should be valid"),
        anchor: ReviewCommentAnchor::new(
            PatchRevisionNumber::FIRST,
            commit(2),
            ReviewFilePath::new("src/activity.rs").expect("fixture path should be valid"),
            ReviewHunkId::parse("a".repeat(64)).expect("fixture hunk should be valid"),
            ReviewDiffSide::Head,
            NonZeroU32::new(20).expect("fixture line should be positive"),
            NonZeroU32::new(24).expect("fixture line should be positive"),
        )
        .expect("fixture review anchor should be valid"),
        created_at_millis: 1_900_000_001_000,
    }
}

fn review_approval(value: u128, decision: ReviewDecision) -> ReviewApproval {
    ReviewApproval {
        approval_id: aggregate(value),
        review: review_identity(),
        revision: PatchRevisionNumber::FIRST,
        head_commit: commit(2),
        approver_principal_id: principal(value + 10),
        decision,
        created_at_millis: 1_900_000_002_000 + value as u64,
    }
}

fn check_suite(value: u128, status: CiCheckStatus) -> CiCheckSuite {
    let revision = revision();
    let mut suite = CiCheckSuite::create(
        CiCheckSuiteIdentity::for_revision(aggregate(value), &revision)
            .expect("fixture suite identity should be valid"),
        CiWorkflowLink::new(
            aggregate(value + 100),
            aggregate(value + 200),
            CiLabel::from_untrusted("build and test")
                .expect("fixture workflow label should be valid"),
            None,
        )
        .expect("fixture workflow link should be valid"),
        1_900_000_003_000,
    );
    if status == CiCheckStatus::Pending {
        return suite;
    }
    let run_id = aggregate(value + 300);
    suite
        .add_run(
            AggregateVersion::FIRST,
            CiCheckRunInput {
                check_run_id: run_id,
                label: CiLabel::from_untrusted("tests")
                    .expect("fixture check label should be valid"),
                queued_at_millis: 1_900_000_004_000,
            },
        )
        .expect("fixture check run should be added");
    if status == CiCheckStatus::Running {
        suite
            .start_run(
                version(2),
                run_id,
                AggregateVersion::FIRST,
                1_900_000_005_000,
            )
            .expect("fixture check run should start");
        return suite;
    }
    suite
        .complete_run(
            version(2),
            run_id,
            AggregateVersion::FIRST,
            &commit(2),
            CiCheckRunCompletionInput {
                status,
                output: CiOutputText::from_untrusted("finished"),
                artifacts: Vec::new(),
                completed_at_millis: 1_900_000_006_000,
            },
        )
        .expect("fixture check run should complete");
    suite
}

fn push_git_fixtures(catalog: &mut Vec<CatalogFixture>) {
    let code_context = CodeActivityProjectionContext {
        actor_kind: ActivityActorKind::Human,
        actor_label: "Ada".into(),
        project_id: Some("project-1".into()),
        thread_id: Some("thread-1".into()),
        visibility: ActivityVisibility::Project,
        projected_at: timestamp(),
    };
    let branch = branch();
    let fixtures = vec![
        (
            "git_branch_created",
            CollaborationCodeActivity::Branch(BranchCodeActivity {
                event_id: aggregate(3_100),
                actor_principal_id: principal(3_004),
                branch: branch.clone(),
                version: version(1),
                occurred_at_millis: 1_900_000_010_000,
                kind: BranchCodeActivityKind::Created { commit: commit(1) },
            }),
        ),
        (
            "git_branch_fast_forwarded",
            CollaborationCodeActivity::Branch(BranchCodeActivity {
                event_id: aggregate(3_101),
                actor_principal_id: principal(3_004),
                branch: branch.clone(),
                version: version(2),
                occurred_at_millis: 1_900_000_011_000,
                kind: BranchCodeActivityKind::Updated {
                    previous_commit: commit(1),
                    current_commit: commit(2),
                    update_kind: BranchUpdateKind::FastForward,
                },
            }),
        ),
        (
            "git_branch_force_updated",
            CollaborationCodeActivity::Branch(BranchCodeActivity {
                event_id: aggregate(3_102),
                actor_principal_id: principal(3_004),
                branch: branch.clone(),
                version: version(3),
                occurred_at_millis: 1_900_000_012_000,
                kind: BranchCodeActivityKind::Updated {
                    previous_commit: commit(2),
                    current_commit: commit(3),
                    update_kind: BranchUpdateKind::Force,
                },
            }),
        ),
        (
            "git_branch_merged",
            CollaborationCodeActivity::Branch(BranchCodeActivity {
                event_id: aggregate(3_103),
                actor_principal_id: principal(3_004),
                branch: branch.clone(),
                version: version(4),
                occurred_at_millis: 1_900_000_013_000,
                kind: BranchCodeActivityKind::Merged {
                    source_commit: commit(3),
                    target_branch: BranchRefName::parse("refs/heads/main")
                        .expect("fixture target branch should be valid"),
                    result_commit: commit(4),
                },
            }),
        ),
        (
            "git_branch_deleted",
            CollaborationCodeActivity::Branch(BranchCodeActivity {
                event_id: aggregate(3_104),
                actor_principal_id: principal(3_004),
                branch,
                version: version(5),
                occurred_at_millis: 1_900_000_014_000,
                kind: BranchCodeActivityKind::Deleted { commit: commit(3) },
            }),
        ),
        (
            "git_patch_submitted",
            CollaborationCodeActivity::PatchSubmitted(revision()),
        ),
        (
            "git_review_commented",
            CollaborationCodeActivity::ReviewCommented(review_comment()),
        ),
        (
            "git_review_approved",
            CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity {
                approval: review_approval(3_200, ReviewDecision::Approve),
            }),
        ),
        (
            "git_review_changes_requested",
            CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity {
                approval: review_approval(3_201, ReviewDecision::RequestChanges),
            }),
        ),
        (
            "git_ci_pending",
            CollaborationCodeActivity::CiStatusChanged(check_suite(3_300, CiCheckStatus::Pending)),
        ),
        (
            "git_ci_running",
            CollaborationCodeActivity::CiStatusChanged(check_suite(3_301, CiCheckStatus::Running)),
        ),
        (
            "git_ci_success",
            CollaborationCodeActivity::CiStatusChanged(check_suite(3_302, CiCheckStatus::Success)),
        ),
        (
            "git_ci_failure",
            CollaborationCodeActivity::CiStatusChanged(check_suite(3_303, CiCheckStatus::Failure)),
        ),
        (
            "git_ci_cancelled",
            CollaborationCodeActivity::CiStatusChanged(check_suite(
                3_304,
                CiCheckStatus::Cancelled,
            )),
        ),
        (
            "git_future_kind",
            CollaborationCodeActivity::Unsupported(GenericCodeActivity {
                source_kind: ActivitySourceKind::Git,
                source_id: "git-future-kind".into(),
                source_version: 1,
                actor_id: "service-1".into(),
                community_id: CommunityId::from_uuid(Uuid::from_u128(3_000)),
                repository_id: aggregate(3_001),
                event_kind: "future_git_kind".into(),
                occurred_at_millis: 1_900_000_020_000,
            }),
        ),
    ];
    for (name, fixture) in fixtures {
        catalog.push(CatalogFixture {
            name: name.into(),
            family: "git",
            item: project_code_activity(&code_context, &fixture)
                .expect("Git fixture should project"),
        });
    }
}

fn job_command(job_index: u64, command_version: u64, kind: JobCommandKind) -> JobCommand {
    JobCommand::new(
        JobIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(4_000)),
            aggregate(4_100 + u128::from(job_index)),
        )
        .expect("fixture job identity should be valid"),
        OperationId::from_uuid(Uuid::from_u128(
            4_500 + u128::from(job_index * 10 + command_version),
        )),
        version(command_version),
        1_900_001_000_000 + job_index * 100 + command_version,
        kind,
    )
    .expect("fixture job command should be valid")
}

fn push_job_fixtures(catalog: &mut Vec<CatalogFixture>) {
    let requester = principal(4_001);
    let executor = principal(4_002);
    let context = JobActivityContext {
        requester: collaboration_actor(4_001, ActivityActorKind::Human, "Avery"),
        executor: collaboration_actor(4_002, ActivityActorKind::Agent, "Builder"),
        transition_actor: None,
        projected_at: timestamp(),
        scope: CollaborationActivityScope::default(),
    };
    for (job_index, target) in [
        "requested",
        "accepted",
        "in_progress",
        "completed",
        "cancelled",
        "failed",
    ]
    .into_iter()
    .enumerate()
    {
        let job_index = job_index as u64;
        let mut job = Job::request(job_command(
            job_index,
            1,
            JobCommandKind::Request {
                requester_principal_id: requester,
                target_executor_principal_id: executor,
            },
        ))
        .expect("fixture job request should apply");
        if target != "requested" {
            job.apply(job_command(
                job_index,
                2,
                JobCommandKind::Accept {
                    executor_principal_id: executor,
                },
            ))
            .expect("fixture job acceptance should apply");
        }
        if matches!(target, "in_progress" | "completed") {
            job.apply(job_command(
                job_index,
                3,
                JobCommandKind::Progress {
                    executor_principal_id: executor,
                },
            ))
            .expect("fixture job progress should apply");
        }
        match target {
            "completed" => {
                job.apply(job_command(
                    job_index,
                    4,
                    JobCommandKind::Result {
                        executor_principal_id: executor,
                    },
                ))
                .expect("fixture job result should apply");
            }
            "cancelled" => {
                job.apply(job_command(
                    job_index,
                    3,
                    JobCommandKind::Cancel {
                        actor_principal_id: requester,
                    },
                ))
                .expect("fixture job cancellation should apply");
            }
            "failed" => {
                job.apply(job_command(
                    job_index,
                    3,
                    JobCommandKind::Error {
                        actor_principal_id: executor,
                    },
                ))
                .expect("fixture job failure should apply");
            }
            _ => {}
        }
        catalog.push(CatalogFixture {
            name: format!("job_{target}"),
            family: "job",
            item: project_job_activity(&job, &context).expect("job fixture should project"),
        });
    }
}

fn push_platform_fixtures(catalog: &mut Vec<CatalogFixture>) {
    let context = PlatformActivityProjectionContext {
        actor_kind: ActivityActorKind::Service,
        actor_label: "Collaboration service".into(),
        project_id: Some("project-1".into()),
        thread_id: Some("thread-1".into()),
        session_id: None,
        visibility: ActivityVisibility::Community,
        projected_at: timestamp(),
    };
    for (index, kind) in RegisteredPlatformEventKind::ALL.into_iter().enumerate() {
        let record = PlatformActivityRecord {
            source_kind: kind.source_kind(),
            source_id: format!("platform-{index}"),
            source_version: 1,
            schema_version: PLATFORM_ACTIVITY_SCHEMA_VERSION,
            actor_id: "service-1".into(),
            community_id: Some(CommunityId::from_uuid(Uuid::from_u128(5_000))),
            event_kind: PlatformEventKind::Registered(kind),
            object_id: Some(format!("platform-object-{index}")),
            object_label: format!("Platform object {index}"),
            occurred_at_millis: 1_900_002_000_000 + index as u64,
        };
        catalog.push(CatalogFixture {
            name: format!("platform_{}", kind.catalog_name()),
            family: "platform",
            item: project_platform_activity(&context, &PlatformActivity::Platform(record))
                .expect("platform fixture should project"),
        });
    }
}

#[gpui::test]
fn activity_catalog_maps_every_fixture_exactly_once_without_blanks(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let mut catalog = Vec::new();
        push_acp_fixtures(&mut catalog, cx);
        push_nip_ao_fixtures(&mut catalog);
        push_message_fixtures(&mut catalog);
        push_git_fixtures(&mut catalog);
        push_job_fixtures(&mut catalog);
        push_platform_fixtures(&mut catalog);

        let expected_family_counts = [
            ("acp", 16),
            ("nip_ao", 4),
            ("message", 2),
            ("git", 15),
            ("job", 6),
            ("platform", 15),
        ];
        assert_eq!(catalog.len(), 58);
        for (family, expected_count) in expected_family_counts {
            assert_eq!(
                catalog
                    .iter()
                    .filter(|fixture| fixture.family == family)
                    .count(),
                expected_count,
                "{family} catalog count changed without updating the conformance inventory"
            );
        }

        let mut names = HashSet::new();
        let mut source_ids = HashSet::new();
        let mut reducer = ActivityReducer::new();
        for fixture in catalog {
            assert!(names.insert(fixture.name.clone()), "duplicate fixture name");
            assert!(
                source_ids.insert(fixture.item.id.clone()),
                "{} mapped onto another fixture's canonical row",
                fixture.name
            );
            assert!(!fixture.item.id.source_id().trim().is_empty());
            assert!(!fixture.item.actor.id.trim().is_empty());
            assert!(!fixture.item.actor.label.trim().is_empty());
            assert!(!fixture.item.verb.trim().is_empty());
            assert!(!fixture.item.object.label.trim().is_empty());
            assert!(
                fixture
                    .item
                    .outcome
                    .summary
                    .as_ref()
                    .is_none_or(|summary| !summary.trim().is_empty()),
                "{} produced a blank outcome",
                fixture.name
            );
            assert!(matches!(
                reducer
                    .reduce(fixture.item.clone())
                    .expect("first fixture delivery should reduce"),
                ActivityReduction::Inserted { .. }
            ));
            assert!(matches!(
                reducer
                    .reduce(fixture.item)
                    .expect("duplicate fixture delivery should reduce"),
                ActivityReduction::Duplicate { .. }
            ));
        }
        assert_eq!(names.len(), 58);
        assert_eq!(source_ids.len(), 58);
        assert_eq!(reducer.items().len(), 58);
    });
}
