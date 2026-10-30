use acp_thread::{
    CollaborationObserverAdapter, CollaborationObserverContext, CollaborationObserverPublish,
    NativeCollaborationObserverEvent, ToolCallStatus,
};
use agent::{
    buzz_tool_compat::{
        BuzzToolCompatibilityError, BuzzToolCompatibilityMapper, BuzzToolRequest,
        NativeBuzzToolRequest,
    },
    collaboration_mention::{
        CollaborationMentionError, CollaborationMentionRouter, CollaborationPromptDispatch,
        ResolvedCollaborationMentionTarget,
    },
    collaboration_session::{
        CollaborationExecutorId, CollaborationSessionIdentity, CollaborationSessionLease,
        CollaborationSessionRegistry, CollaborationSessionScope,
    },
};
use agent_client_protocol::schema::v1 as acp;
use chrono::{TimeZone, Utc};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus, Message,
    MessageAuthor, MessageContent, MessageLifecycleState, MessageRecordFields, MessageSource,
    NostrEventId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext,
    TrustedTenantRoute,
};
use serde_json::{Value, json};
use uuid::Uuid;

const COMMUNITY: u128 = 1;
const CHANNEL: u128 = 2;
const AUTHOR: u128 = 3;
const AGENT: u128 = 4;

#[derive(Clone, Debug, Eq, PartialEq)]
enum PromptOutcome {
    Accepted { event: u8, observer_sequence: u64 },
    Busy { event: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CleanupOutcome {
    event: u8,
    observer_sequence: u64,
    stop_reason: &'static str,
    resources: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToolOutcome {
    Native(&'static str),
    CurrentPlan,
    ReplacePlan,
    Denied,
    Cancelled,
}

#[derive(Default)]
struct FrozenBuzzHarness {
    session_active: bool,
    active_event: Option<u8>,
    observer_sequence: u64,
}

impl FrozenBuzzHarness {
    fn restart(&mut self) {
        self.session_active = true;
        self.observer_sequence = 1;
    }

    fn prompt(&mut self, event: u8) -> PromptOutcome {
        if self.active_event.is_some() {
            return PromptOutcome::Busy { event };
        }
        assert!(self.session_active, "reference session must be active");
        let observer_sequence = self.observer_sequence;
        self.observer_sequence += 1;
        self.active_event = Some(event);
        PromptOutcome::Accepted {
            event,
            observer_sequence,
        }
    }

    fn crash(&mut self) -> CleanupOutcome {
        let event = self
            .active_event
            .take()
            .expect("reference prompt is active");
        let observer_sequence = self.observer_sequence;
        self.observer_sequence += 1;
        self.session_active = false;
        CleanupOutcome {
            event,
            observer_sequence,
            stop_reason: "cancelled",
            resources: 0,
        }
    }

    fn tool(name: &str, denied: bool, cancelled: bool) -> ToolOutcome {
        if cancelled {
            return ToolOutcome::Cancelled;
        }
        if denied {
            return ToolOutcome::Denied;
        }
        match name {
            "shell" => ToolOutcome::Native("terminal"),
            "read_file" | "view_image" => ToolOutcome::Native("read_file"),
            "str_replace" => ToolOutcome::Native("edit_file"),
            "search" => ToolOutcome::Native("grep"),
            "tree" => ToolOutcome::Native("list_directory"),
            "todo-read" => ToolOutcome::CurrentPlan,
            "todo-write" => ToolOutcome::ReplacePlan,
            _ => panic!("unknown frozen Buzz tool"),
        }
    }
}

struct NativeHarness {
    sessions: CollaborationSessionRegistry,
    lease: Option<CollaborationSessionLease>,
    router: CollaborationMentionRouter,
    active_dispatch: Option<CollaborationPromptDispatch>,
    active_event: Option<u8>,
    observer: Option<CollaborationObserverAdapter>,
    generation: u128,
}

impl NativeHarness {
    fn new() -> Self {
        let mut harness = Self {
            sessions: CollaborationSessionRegistry::default(),
            lease: None,
            router: CollaborationMentionRouter::new(principal_id(AGENT)).expect("native router"),
            active_dispatch: None,
            active_event: None,
            observer: None,
            generation: 0,
        };
        harness.restart();
        harness
    }

    fn restart(&mut self) {
        self.generation += 1;
        let lease = self
            .sessions
            .resolve(
                session_identity(),
                CollaborationExecutorId::new(Uuid::from_u128(100 + self.generation))
                    .expect("executor"),
            )
            .expect("reserve native session")
            .lease()
            .clone();
        let session_id = acp::SessionId::new(format!("native-session-{}", self.generation));
        self.sessions
            .activate(&lease, session_id.clone())
            .expect("activate native session");
        self.observer = Some(CollaborationObserverAdapter::new(
            CollaborationObserverContext::new(Uuid::from_u128(CHANNEL), session_id, Some(0))
                .expect("observer context"),
        ));
        self.lease = Some(lease);
    }

    fn prompt(&mut self, event: u8) -> PromptOutcome {
        let message = message(event, &format!("prompt-{event}"));
        let authorization = AuthorizationFixture::new();
        let result = self.router.begin_prompt(
            &message,
            &ResolvedCollaborationMentionTarget::direct(principal_id(AGENT)),
            &authorization.request(&message),
            &self.sessions,
            self.lease.as_ref().expect("active lease"),
        );
        match result {
            Ok(dispatch) => {
                let turn_id = turn_id(event);
                let publish = self
                    .observer
                    .as_mut()
                    .expect("active observer")
                    .publish(
                        &format!("prompt-{event}"),
                        &turn_id,
                        timestamp(event),
                        NativeCollaborationObserverEvent::TurnStarted,
                    )
                    .expect("publish prompt start");
                let observer_sequence = publish.frame().expect("observer frame").seq;
                self.active_dispatch = Some(dispatch);
                self.active_event = Some(event);
                PromptOutcome::Accepted {
                    event,
                    observer_sequence,
                }
            }
            Err(CollaborationMentionError::BusySession) => PromptOutcome::Busy { event },
            Err(error) => panic!("unexpected native prompt outcome: {error}"),
        }
    }

    fn crash(&mut self) -> CleanupOutcome {
        let event = self.active_event.take().expect("native prompt is active");
        let dispatch = self
            .active_dispatch
            .take()
            .expect("native dispatch is active");
        self.router
            .abort_prompt(&dispatch)
            .expect("abort crashed prompt");
        assert_eq!(
            self.router.complete_prompt(&dispatch),
            Err(CollaborationMentionError::DispatchNotCurrent),
            "a stale completion cannot consume the retryable source event",
        );

        let turn_id = turn_id(event);
        let publish = self
            .observer
            .as_mut()
            .expect("active observer")
            .publish(
                &format!("crash-{event}"),
                &turn_id,
                timestamp(event.saturating_add(1)),
                NativeCollaborationObserverEvent::SessionResolved(acp::StopReason::Cancelled),
            )
            .expect("publish cancellation");
        let frame = publish.frame().expect("observer frame");
        let observer_sequence = frame.seq;
        assert_eq!(frame.kind, "session_resolved");
        assert_eq!(frame.payload["stopReason"], "cancelled");

        let cancellation = self
            .sessions
            .authorize_cancellation(self.lease.as_ref().expect("active lease"))
            .expect("authorize cancellation");
        self.sessions
            .complete_cancellation(&cancellation)
            .expect("release cancelled session");
        self.lease = None;
        self.observer = None;

        CleanupOutcome {
            event,
            observer_sequence,
            stop_reason: "cancelled",
            resources: self.resource_count(),
        }
    }

    fn resource_count(&self) -> usize {
        self.sessions.len() + usize::from(self.active_dispatch.is_some())
    }
}

#[test]
fn buzz_acp_conformance_reentrant_prompt_crash_and_queue_retry() {
    let mut legacy = FrozenBuzzHarness::default();
    legacy.restart();
    let mut native = NativeHarness::new();

    assert_eq!(native.prompt(1), legacy.prompt(1));
    assert_eq!(native.prompt(2), legacy.prompt(2));
    assert_eq!(native.crash(), legacy.crash());

    legacy.restart();
    native.restart();
    assert_eq!(native.prompt(1), legacy.prompt(1));
}

#[test]
fn buzz_acp_conformance_tools_permissions_and_cancellation() {
    let mapper = BuzzToolCompatibilityMapper::new("workspace").expect("compatibility mapper");
    for (legacy_name, native_name, arguments) in [
        ("shell", "shell", json!({"command":"cargo check"})),
        ("read_file", "read_file", json!({"path":"src/main.rs"})),
        (
            "str_replace",
            "str_replace",
            json!({"path":"src/main.rs", "old_str":"before", "new_str":"after"}),
        ),
        (
            "search",
            "search",
            json!({"regex":"SessionId", "path":"src"}),
        ),
        ("tree", "tree", json!({"path":"src", "depth":1})),
        (
            "view_image",
            "view_image",
            json!({"source":"assets/icon.png"}),
        ),
        ("todo-read", "todo", json!({})),
        (
            "todo-write",
            "todo",
            json!({"todos":[{"text":"validate cleanup", "done":false}]}),
        ),
    ] {
        assert_eq!(
            native_tool_outcome(&mapper, native_name, arguments, false, false),
            FrozenBuzzHarness::tool(legacy_name, false, false),
        );
    }

    assert_eq!(
        native_tool_outcome(
            &mapper,
            "shell",
            json!({"command":"git status"}),
            true,
            false,
        ),
        FrozenBuzzHarness::tool("shell", true, false),
    );
    assert_eq!(
        native_tool_outcome(&mapper, "shell", json!({"command":"sleep 30"}), false, true,),
        FrozenBuzzHarness::tool("shell", false, true),
    );
}

#[test]
fn buzz_acp_conformance_observer_output_and_resource_cleanup() {
    let session_id = acp::SessionId::new("observer-session");
    let mut observer = CollaborationObserverAdapter::new(
        CollaborationObserverContext::new(Uuid::from_u128(CHANNEL), session_id, Some(0))
            .expect("observer context"),
    );
    let action_id = acp::ToolCallId::new("action-1");

    let started = observer
        .publish(
            "source-start",
            "turn-1",
            timestamp(1),
            NativeCollaborationObserverEvent::TurnStarted,
        )
        .expect("turn starts");
    let pending = observer
        .publish(
            "source-action",
            "turn-1",
            timestamp(2),
            NativeCollaborationObserverEvent::ActionUpdated {
                action_id: &action_id,
                status: &ToolCallStatus::Pending,
            },
        )
        .expect("action starts");
    let duplicate = observer
        .publish(
            "source-action",
            "turn-1",
            timestamp(3),
            NativeCollaborationObserverEvent::ActionUpdated {
                action_id: &action_id,
                status: &ToolCallStatus::Pending,
            },
        )
        .expect("transport retry is suppressed");
    let cancelled = observer
        .publish(
            "source-cancel",
            "turn-1",
            timestamp(4),
            NativeCollaborationObserverEvent::SessionResolved(acp::StopReason::Cancelled),
        )
        .expect("turn cancels");

    assert_eq!(observer_trace(&started), (1, "turn_started", None, None));
    assert_eq!(
        observer_trace(&pending),
        (2, "acp_read", Some("pending"), None)
    );
    assert!(matches!(duplicate, CollaborationObserverPublish::Duplicate));
    assert_eq!(
        observer_trace(&cancelled),
        (3, "session_resolved", None, Some("cancelled"))
    );
    assert!(
        observer
            .publish(
                "source-after-terminal",
                "turn-1",
                timestamp(5),
                NativeCollaborationObserverEvent::TurnStarted,
            )
            .is_err(),
        "terminal turns cannot leak later observer activity",
    );
}

fn native_tool_outcome(
    mapper: &BuzzToolCompatibilityMapper,
    name: &str,
    arguments: Value,
    denied: bool,
    cancelled: bool,
) -> ToolOutcome {
    match mapper.map(
        BuzzToolRequest {
            name: name.to_owned(),
            arguments,
        },
        cancelled,
        |_| !denied,
    ) {
        Ok(NativeBuzzToolRequest::Tool(call)) => ToolOutcome::Native(call.tool_name),
        Ok(NativeBuzzToolRequest::CurrentPlan) => ToolOutcome::CurrentPlan,
        Ok(NativeBuzzToolRequest::ReplacePlan(_)) => ToolOutcome::ReplacePlan,
        Err(BuzzToolCompatibilityError::Denied) => ToolOutcome::Denied,
        Err(BuzzToolCompatibilityError::Cancelled) => ToolOutcome::Cancelled,
        Err(error) => panic!("unexpected native tool outcome: {error}"),
    }
}

fn observer_trace(
    publish: &CollaborationObserverPublish,
) -> (u64, &str, Option<&str>, Option<&str>) {
    let frame = publish.frame().expect("observer frame");
    (
        frame.seq,
        frame.kind.as_str(),
        frame.payload["status"].as_str(),
        frame.payload["stopReason"].as_str(),
    )
}

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(COMMUNITY))
}

fn aggregate_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn session_identity() -> CollaborationSessionIdentity {
    CollaborationSessionIdentity::new(
        Uuid::from_u128(COMMUNITY),
        CollaborationSessionScope::Channel {
            channel_id: Uuid::from_u128(CHANNEL),
        },
    )
    .expect("session identity")
}

fn message(event: u8, content: &str) -> Message {
    let source = MessageSource {
        event_id: NostrEventId::from_bytes([event; 32]),
        event_created_at: u64::from(event),
    };
    Message::from_record(MessageRecordFields {
        community_id: community_id(),
        channel_id: aggregate_id(CHANNEL),
        message_id: aggregate_id(1_000 + u128::from(event)),
        author: MessageAuthor::principal(principal_id(AUTHOR)),
        content: MessageContent::new(content).expect("message content"),
        lifecycle_state: MessageLifecycleState::Active,
        source,
        current_source: source,
        mutations: Vec::new(),
        version: AggregateVersion::FIRST,
    })
    .expect("message")
}

fn turn_id(event: u8) -> String {
    format!("turn-{event}")
}

fn timestamp(second: u8) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 23, 12, 0, u32::from(second))
        .single()
        .expect("timestamp")
}

struct AuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
}

impl AuthorizationFixture {
    fn new() -> Self {
        let scope = AuthorizationScope::new("messages:write").expect("scope");
        Self {
            tenant: TenantContext::establish(
                Some(
                    TrustedTenantRoute::from_listener(community_id(), "conformance-test")
                        .expect("tenant route"),
                ),
                &[],
            )
            .expect("tenant"),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id(AUTHOR),
                community_id(),
                ServiceAccountId::new(1),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            scope,
        }
    }

    fn request<'a>(&'a self, message: &Message) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Conversation,
                resource_id: message.fields().message_id,
                owner_principal_id: Some(principal_id(AUTHOR)),
                channel_id: Some(aggregate_id(CHANNEL)),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal_id(AUTHOR),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id: aggregate_id(CHANNEL),
                principal_id: principal_id(AUTHOR),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }
}
