use agent::{
    collaboration_mention::{
        CollaborationMentionError, CollaborationMentionRouter, ResolvedCollaborationMentionTarget,
    },
    collaboration_session::{
        CollaborationExecutorId, CollaborationSessionIdentity, CollaborationSessionRegistry,
        CollaborationSessionScope,
    },
};
use agent_client_protocol::schema::v1 as acp;
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus, Message,
    MessageAuthor, MessageContent, MessageLifecycleState, MessageRecordFields, MessageSource,
    NostrEventId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext,
    TrustedTenantRoute,
};
use uuid::Uuid;

const COMMUNITY: u128 = 1;
const CHANNEL: u128 = 2;
const AUTHOR: u128 = 3;
const AGENT: u128 = 4;

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(COMMUNITY))
}

fn aggregate_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn message(message_id: u128, event_byte: u8, content: &str) -> Message {
    let source = MessageSource {
        event_id: NostrEventId::from_bytes([event_byte; 32]),
        event_created_at: u64::from(event_byte),
    };
    Message::from_record(MessageRecordFields {
        community_id: community_id(),
        channel_id: aggregate_id(CHANNEL),
        message_id: aggregate_id(message_id),
        author: MessageAuthor::principal(principal_id(AUTHOR)),
        content: MessageContent::new(content).expect("content"),
        lifecycle_state: MessageLifecycleState::Active,
        source,
        current_source: source,
        mutations: Vec::new(),
        version: AggregateVersion::FIRST,
    })
    .expect("message")
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
                    TrustedTenantRoute::from_listener(community_id(), "mention-test")
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

fn active_session() -> (
    CollaborationSessionRegistry,
    agent::collaboration_session::CollaborationSessionLease,
) {
    let identity = CollaborationSessionIdentity::new(
        Uuid::from_u128(COMMUNITY),
        CollaborationSessionScope::Channel {
            channel_id: Uuid::from_u128(CHANNEL),
        },
    )
    .expect("identity");
    let executor = CollaborationExecutorId::new(Uuid::from_u128(5)).expect("executor");
    let mut sessions = CollaborationSessionRegistry::default();
    let lease = sessions
        .resolve(identity, executor)
        .expect("reserve")
        .lease()
        .clone();
    sessions
        .activate(&lease, acp::SessionId::new("native-session"))
        .expect("activate");
    (sessions, lease)
}

fn prompt_text(dispatch: &agent::collaboration_mention::CollaborationPromptDispatch) -> &str {
    let acp::ContentBlock::Text(text) = &dispatch.request().prompt[0] else {
        panic!("expected text prompt")
    };
    &text.text
}

#[test]
fn collaboration_mention_routes_direct_and_team_targets_to_native_prompts() {
    let fixture = AuthorizationFixture::new();
    let message = message(10, 1, "Please inspect the failing build");
    let (sessions, lease) = active_session();
    let mut router = CollaborationMentionRouter::new(principal_id(AGENT)).expect("router");

    let direct = router
        .begin_prompt(
            &message,
            &ResolvedCollaborationMentionTarget::direct(principal_id(AGENT)),
            &fixture.request(&message),
            &sessions,
            &lease,
        )
        .expect("direct prompt");
    assert_eq!(
        direct.request().session_id,
        acp::SessionId::new("native-session")
    );
    assert_eq!(prompt_text(&direct), "Please inspect the failing build");
    router.abort_prompt(&direct).expect("retryable abort");

    let team = ResolvedCollaborationMentionTarget::team(
        aggregate_id(20),
        [principal_id(AGENT), principal_id(30)],
    )
    .expect("team");
    let team_dispatch = router
        .begin_prompt(
            &message,
            &team,
            &fixture.request(&message),
            &sessions,
            &lease,
        )
        .expect("team prompt");
    assert_eq!(
        prompt_text(&team_dispatch),
        "Please inspect the failing build"
    );
}

#[test]
fn collaboration_mention_rejects_duplicate_events() {
    let fixture = AuthorizationFixture::new();
    let message = message(10, 1, "Run the focused tests");
    let (sessions, lease) = active_session();
    let target = ResolvedCollaborationMentionTarget::direct(principal_id(AGENT));
    let mut router = CollaborationMentionRouter::new(principal_id(AGENT)).expect("router");

    let dispatch = router
        .begin_prompt(
            &message,
            &target,
            &fixture.request(&message),
            &sessions,
            &lease,
        )
        .expect("prompt");
    assert!(matches!(
        router.begin_prompt(
            &message,
            &target,
            &fixture.request(&message),
            &sessions,
            &lease,
        ),
        Err(CollaborationMentionError::DuplicateEvent)
    ));
    router.complete_prompt(&dispatch).expect("complete");
    assert!(matches!(
        router.begin_prompt(
            &message,
            &target,
            &fixture.request(&message),
            &sessions,
            &lease,
        ),
        Err(CollaborationMentionError::DuplicateEvent)
    ));
}

#[test]
fn collaboration_mention_rejects_unauthorized_actors_before_routing() {
    let fixture = AuthorizationFixture::new();
    let message = message(10, 1, "Do not route this");
    let (sessions, lease) = active_session();
    let mut request = fixture.request(&message);
    request.community_membership = None;
    let mut router = CollaborationMentionRouter::new(principal_id(AGENT)).expect("router");

    assert!(matches!(
        router.begin_prompt(
            &message,
            &ResolvedCollaborationMentionTarget::direct(principal_id(AGENT)),
            &request,
            &sessions,
            &lease,
        ),
        Err(CollaborationMentionError::Unauthorized)
    ));
}

#[test]
fn collaboration_mention_rejects_a_second_event_while_session_is_busy() {
    let fixture = AuthorizationFixture::new();
    let first_message = message(10, 1, "First prompt");
    let second_message = message(11, 2, "Second prompt");
    let (sessions, lease) = active_session();
    let target = ResolvedCollaborationMentionTarget::direct(principal_id(AGENT));
    let mut router = CollaborationMentionRouter::new(principal_id(AGENT)).expect("router");

    let first_dispatch = router
        .begin_prompt(
            &first_message,
            &target,
            &fixture.request(&first_message),
            &sessions,
            &lease,
        )
        .expect("first prompt");
    assert!(matches!(
        router.begin_prompt(
            &second_message,
            &target,
            &fixture.request(&second_message),
            &sessions,
            &lease,
        ),
        Err(CollaborationMentionError::BusySession)
    ));
    router.abort_prompt(&first_dispatch).expect("abort first");
    router
        .begin_prompt(
            &second_message,
            &target,
            &fixture.request(&second_message),
            &sessions,
            &lease,
        )
        .expect("retry second");
}
