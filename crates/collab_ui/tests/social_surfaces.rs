#![cfg(feature = "multiplayer-tools")]

use std::collections::BTreeSet;

use agent_ui::activity_projection::{
    ActivityActor, ActivityActorKind, ActivityContext, ActivityItem, ActivityItemId,
    ActivityLifecycle, ActivityObject, ActivityObjectKind, ActivityOutcome, ActivityOutcomeStatus,
    ActivitySemanticClass, ActivitySourceKind, ActivityVisibility,
};
use chrono::{TimeZone as _, Utc};
use collab_ui::{
    forum::{ForumAuthorPresentation, ForumPermissions, ForumSnapshot, ForumView, ForumViewError},
    inbox_pulse::{
        InboxPulseError, InboxPulseFreshness, InboxPulseMode, InboxPulseRow, InboxPulseView,
    },
    message_timeline::MessageTimelineAuthorKind,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationDenial, AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind,
    AuthorizationScope, Channel, ChannelLifecycleState, ChannelMembership, ChannelName,
    ChannelRecordFields, ChannelType, ChannelVisibility, CommunityId, CommunityMembership,
    CustomEmoji, CustomEmojiError, CustomEmojiPalette, CustomEmojiSetRecord, CustomEmojiShortcode,
    Feedback, FeedbackBody, FeedbackCategory, FeedbackCommandOutcome, FeedbackCreateFields,
    FeedbackError, FeedbackStatus, FeedbackStatusReason, FeedbackStatusSource, ForumMessageInput,
    ForumProjection, InboxMessageInput, InboxProjection, InboxScope, ManualUnreadRegister,
    MembershipRole, MembershipStatus, Message, MessageAuthor, MessageContent, MessageRecordFields,
    MessageSource, NostrEventId, NostrPublicKey, OperationId, OwnerReadStateReplica, PrincipalId,
    PrincipalScopes, ReadContextId, ReadState, ReadStateCompleteness, ReadStateScope,
    ServiceAccountId, TenantContext, ThreadReference, TrustedTenantRoute,
};
use gpui::{AppContext as _, TestAppContext};
use uuid::Uuid;

fn community_id(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn aggregate_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn event_id(value: u8) -> NostrEventId {
    NostrEventId::from_bytes([value; 32])
}

fn source(value: u8, created_at: u64) -> MessageSource {
    MessageSource {
        event_id: event_id(value),
        event_created_at: created_at,
    }
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "social-surfaces-test")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant context")
}

fn scope(value: &str) -> AuthorizationScope {
    AuthorizationScope::new(value).expect("authorization scope")
}

fn principal(
    community_id: CommunityId,
    principal_id: PrincipalId,
    scopes: impl IntoIterator<Item = AuthorizationScope>,
) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(principal_id.as_uuid().as_u128() as u64),
        PrincipalScopes::new(scopes).expect("principal scopes"),
    )
}

fn authorization_request<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    required_scope: &'a AuthorizationScope,
    action: AuthorizationAction,
    resource_community_id: CommunityId,
    resource_kind: AuthorizationResourceKind,
    resource_id: AggregateId,
    channel_id: Option<AggregateId>,
    role: MembershipRole,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope,
        action,
        resource: AuthorizationResource {
            community_id: resource_community_id,
            kind: resource_kind,
            resource_id,
            owner_principal_id: None,
            channel_id,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(CommunityMembership {
            community_id: tenant.community_id(),
            principal_id: principal.principal_id(),
            role,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        current_channel_membership_version: channel_id.map(|_| AggregateVersion::FIRST),
        channel_membership: channel_id.map(|channel_id| ChannelMembership {
            community_id: tenant.community_id(),
            channel_id,
            principal_id: principal.principal_id(),
            role,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        delegation: None,
        now_millis: 100,
    }
}

fn message(
    community_id: CommunityId,
    channel_id: AggregateId,
    message_value: u128,
    event_value: u8,
    created_at: u64,
    author_principal_id: PrincipalId,
) -> Message {
    Message::from_record(MessageRecordFields {
        community_id,
        channel_id,
        message_id: aggregate_id(message_value),
        author: MessageAuthor::principal(author_principal_id),
        content: MessageContent::new(format!("canonical message {message_value}"))
            .expect("message content"),
        lifecycle_state: collaboration_domain::MessageLifecycleState::Active,
        source: source(event_value, created_at),
        current_source: source(event_value, created_at),
        mutations: Vec::new(),
        version: AggregateVersion::FIRST,
    })
    .expect("canonical message")
}

fn inbox_projection(scope: InboxScope, messages: &[Message]) -> InboxProjection {
    let context = ReadContextId::new("conversation:main").expect("read context");
    let read_scope = ReadStateScope::new(scope.community_id(), scope.viewer_principal_id());
    let read_state = ReadState::from_replicas(
        read_scope,
        ReadStateCompleteness::Complete,
        [OwnerReadStateReplica::new(
            read_scope,
            scope.viewer_principal_id(),
            [(context.clone(), 0)],
            Vec::<(ReadContextId, ManualUnreadRegister)>::new(),
        )
        .expect("read replica")],
    )
    .expect("read state");
    let mentions = BTreeSet::from([scope.viewer_principal_id()]);
    let inputs = messages
        .iter()
        .enumerate()
        .map(|(index, message)| InboxMessageInput {
            message,
            conversation_id: aggregate_id(70),
            read_context: &context,
            parent_read_context: None,
            sequence: u32::try_from(index + 1).expect("bounded fixture sequence"),
            mentioned_principal_ids: &mentions,
            reply_to_principal_id: None,
        });
    InboxProjection::build(scope, inputs, &read_state, []).expect("inbox projection")
}

fn activity(community_id: CommunityId, id: &str, second: u32) -> ActivityItem {
    let timestamp = Utc
        .with_ymd_and_hms(2026, 8, 22, 12, 0, second)
        .single()
        .expect("activity timestamp");
    ActivityItem {
        id: ActivityItemId::new(ActivitySourceKind::Nostr, id).expect("activity id"),
        source_version: 1,
        class: ActivitySemanticClass::Message,
        actor: ActivityActor {
            kind: ActivityActorKind::Human,
            id: format!("actor-{id}"),
            label: format!("Actor {id}"),
        },
        verb: "posted".into(),
        object: ActivityObject {
            kind: ActivityObjectKind::Message,
            id: None,
            label: format!("Activity {id}"),
        },
        outcome: ActivityOutcome {
            status: ActivityOutcomeStatus::Success,
            summary: None,
        },
        lifecycle: ActivityLifecycle::Succeeded,
        occurred_at: timestamp,
        projected_at: timestamp,
        context: ActivityContext {
            community_id: Some(community_id.to_string()),
            ..ActivityContext::default()
        },
        visibility: ActivityVisibility::Community,
        details: None,
        links: Vec::new(),
    }
}

fn forum_snapshot(
    community_id: CommunityId,
    channel_id: AggregateId,
    messages: &[Message],
) -> Result<ForumSnapshot, ForumViewError> {
    let channel = Channel::from_record(ChannelRecordFields {
        community_id,
        channel_id,
        name: ChannelName::new("discussions").expect("channel name"),
        channel_type: ChannelType::Forum,
        visibility: ChannelVisibility::Open,
        lifecycle_state: ChannelLifecycleState::Active,
        description: None,
        creator_principal_id: principal_id(10),
        expiration: None,
        version: AggregateVersion::FIRST,
    })
    .expect("forum channel");
    let tenant = tenant(community_id);
    let read_scope = scope("forum:read");
    let viewer = principal(community_id, principal_id(20), [read_scope.clone()]);
    let request = authorization_request(
        &tenant,
        &viewer,
        &read_scope,
        AuthorizationAction::Read,
        community_id,
        AuthorizationResourceKind::Channel,
        channel_id,
        Some(channel_id),
        MembershipRole::Member,
    );
    let inputs = messages.iter().map(|message| ForumMessageInput {
        message,
        author_public_key: NostrPublicKey::from_bytes(
            [message.fields().source.event_id.as_bytes()[0]; 32],
        ),
        reference: ThreadReference::TopLevel,
    });
    let projection =
        ForumProjection::build(&channel, &request, inputs, []).expect("canonical forum projection");
    ForumSnapshot::from_projection(
        &projection,
        messages.iter().map(|message| ForumAuthorPresentation {
            principal_id: message.fields().author.principal_id(),
            kind: MessageTimelineAuthorKind::Human,
            label: format!("Member {}", message.fields().source.event_id.as_bytes()[0]),
        }),
        20,
    )
}

fn emoji_record(
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
    source_value: u8,
    created_at: u64,
    asset: &str,
) -> CustomEmojiSetRecord {
    CustomEmojiSetRecord::new(
        community_id,
        owner_principal_id,
        source(source_value, created_at),
        vec![CustomEmoji::new("party", asset).expect("custom emoji")],
    )
    .expect("custom emoji record")
}

fn submitted_feedback(community_id: CommunityId) -> Feedback {
    let tenant = tenant(community_id);
    let submit_scope = scope("feedback:submit");
    let submitter = principal(community_id, principal_id(30), [submit_scope.clone()]);
    let request = authorization_request(
        &tenant,
        &submitter,
        &submit_scope,
        AuthorizationAction::Write,
        community_id,
        AuthorizationResourceKind::Community,
        AggregateId::from_uuid(community_id.as_uuid()),
        None,
        MembershipRole::Member,
    );
    Feedback::submit(
        FeedbackCreateFields {
            community_id,
            source: source(90, 10),
            category: Some(FeedbackCategory::NeedsWork),
            body: FeedbackBody::new("private reconnect context").expect("feedback body"),
        },
        &request,
    )
    .expect("submitted feedback")
}

fn update_feedback_status(feedback: &mut Feedback) {
    let community_id = feedback.fields().community_id;
    let tenant = tenant(community_id);
    let manage_scope = scope("feedback:manage");
    let operator = principal(community_id, principal_id(40), [manage_scope.clone()]);
    let request = authorization_request(
        &tenant,
        &operator,
        &manage_scope,
        AuthorizationAction::Manage,
        community_id,
        AuthorizationResourceKind::Administration,
        AggregateId::from_uuid(community_id.as_uuid()),
        None,
        MembershipRole::Admin,
    );
    assert_eq!(
        feedback.update_status(
            AggregateVersion::FIRST,
            FeedbackStatus::Reviewing,
            FeedbackStatusReason::Acknowledged,
            FeedbackStatusSource {
                operation_id: OperationId::from_uuid(Uuid::from_u128(500)),
                occurred_at: 20,
            },
            &request,
        ),
        Ok(FeedbackCommandOutcome::Applied)
    );
}

fn feedback_status(
    feedback: &Feedback,
    role: MembershipRole,
) -> Result<FeedbackStatus, FeedbackError> {
    let community_id = feedback.fields().community_id;
    let tenant = tenant(community_id);
    let manage_scope = scope("feedback:manage");
    let operator = principal(community_id, principal_id(40), [manage_scope.clone()]);
    let request = authorization_request(
        &tenant,
        &operator,
        &manage_scope,
        AuthorizationAction::Read,
        community_id,
        AuthorizationResourceKind::Administration,
        AggregateId::from_uuid(community_id.as_uuid()),
        None,
        role,
    );
    feedback.status_view(&request).map(|view| view.status)
}

#[gpui::test]
fn social_surfaces_rebuild_after_offline_reconnect(cx: &mut TestAppContext) {
    let community_id = community_id(1);
    let channel_id = aggregate_id(2);
    let inbox_scope = InboxScope::new(community_id, principal_id(20));
    let first_message = message(community_id, channel_id, 100, 1, 10, principal_id(11));
    let initial_messages = vec![first_message.clone()];
    let inbox_view = cx.new(|_| InboxPulseView::new(inbox_scope, 20));
    inbox_view
        .update(cx, |view, cx| {
            view.apply_snapshot(
                1,
                inbox_projection(inbox_scope, &initial_messages),
                vec![activity(community_id, "initial", 1)],
                cx,
            )
        })
        .expect("initial social snapshot");
    let initial_forum = cx.new(|_| {
        ForumView::new(
            forum_snapshot(community_id, channel_id, &initial_messages)
                .expect("initial forum snapshot"),
            ForumPermissions::default(),
        )
    });
    let owner_principal_id = principal_id(11);
    let first_emoji = emoji_record(
        community_id,
        owner_principal_id,
        10,
        10,
        "https://example.com/party-v1.png",
    );
    let initial_palette =
        CustomEmojiPalette::build(community_id, [first_emoji.clone()]).expect("emoji palette");
    let initial_feedback = submitted_feedback(community_id);

    inbox_view
        .update(cx, InboxPulseView::mark_stale)
        .expect("offline snapshot remains available");
    inbox_view
        .update(cx, InboxPulseView::mark_retrying)
        .expect("reconnect attempt remains available");
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.freshness()),
        InboxPulseFreshness::Retrying { revision: 1 }
    );
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.visible_rows().len()),
        1
    );
    assert_eq!(initial_forum.read_with(cx, |view, _| view.posts().len()), 1);
    let shortcode = CustomEmojiShortcode::new("party").expect("emoji shortcode");
    assert_eq!(
        initial_palette
            .get(&shortcode)
            .expect("initial emoji")
            .emoji
            .asset
            .as_str(),
        "https://example.com/party-v1.png"
    );
    assert_eq!(
        feedback_status(&initial_feedback, MembershipRole::Admin),
        Ok(FeedbackStatus::Submitted)
    );

    let second_message = message(community_id, channel_id, 101, 2, 20, principal_id(12));
    let reconnected_messages = vec![first_message, second_message];
    let second_emoji = emoji_record(
        community_id,
        owner_principal_id,
        11,
        20,
        "https://example.com/party-v2.png",
    );
    let mut authoritative_feedback =
        Feedback::from_record(initial_feedback.fields().clone()).expect("feedback record");
    update_feedback_status(&mut authoritative_feedback);

    let rebuilt_inbox = inbox_projection(inbox_scope, &reconnected_messages);
    assert_eq!(
        rebuilt_inbox,
        inbox_projection(inbox_scope, &reconnected_messages)
    );
    inbox_view
        .update(cx, |view, cx| {
            view.apply_snapshot(
                2,
                rebuilt_inbox,
                vec![
                    activity(community_id, "initial", 1),
                    activity(community_id, "reconnected", 2),
                ],
                cx,
            )
        })
        .expect("reconnected snapshot");
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.freshness()),
        InboxPulseFreshness::Fresh { revision: 2 }
    );
    let inbox_counts = inbox_view.read_with(cx, |view, _| {
        let [InboxPulseRow::Inbox(item)] = view.visible_rows() else {
            panic!("one rebuilt conversation expected");
        };
        (item.message_count(), item.unread_message_count())
    });
    assert_eq!(inbox_counts, (2, 2));
    inbox_view.update(cx, |view, cx| view.set_mode(InboxPulseMode::Pulse, cx));
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.visible_rows().len()),
        2
    );

    let reconnected_snapshot =
        forum_snapshot(community_id, channel_id, &reconnected_messages).expect("rebuilt forum");
    let reconnected_forum =
        cx.new(|_| ForumView::new(reconnected_snapshot, ForumPermissions::default()));
    let independently_rebuilt_forum = cx.new(|_| {
        ForumView::new(
            forum_snapshot(community_id, channel_id, &reconnected_messages)
                .expect("independent forum rebuild"),
            ForumPermissions::default(),
        )
    });
    assert_eq!(
        reconnected_forum.read_with(cx, |view, _| view.posts().len()),
        2
    );
    assert_eq!(
        reconnected_forum.read_with(cx, |view, _| view.posts().to_vec()),
        independently_rebuilt_forum.read_with(cx, |view, _| view.posts().to_vec())
    );

    let rebuilt_palette =
        CustomEmojiPalette::build(community_id, [first_emoji.clone(), second_emoji.clone()])
            .expect("rebuilt emoji palette");
    assert_eq!(
        rebuilt_palette,
        CustomEmojiPalette::build(community_id, [first_emoji, second_emoji])
            .expect("independent emoji rebuild")
    );
    assert_eq!(
        rebuilt_palette
            .get(&shortcode)
            .expect("replacement emoji")
            .emoji
            .asset
            .as_str(),
        "https://example.com/party-v2.png"
    );

    let reconnected_feedback =
        Feedback::from_record(authoritative_feedback.fields().clone()).expect("rebuilt feedback");
    assert_eq!(
        reconnected_feedback,
        Feedback::from_record(authoritative_feedback.fields().clone())
            .expect("independent feedback rebuild")
    );
    assert_eq!(
        feedback_status(&reconnected_feedback, MembershipRole::Admin),
        Ok(FeedbackStatus::Reviewing)
    );
}

#[gpui::test]
fn social_surfaces_fail_closed_without_replacing_trusted_state(cx: &mut TestAppContext) {
    let target_community_id = community_id(1);
    let channel_id = aggregate_id(2);
    let inbox_scope = InboxScope::new(target_community_id, principal_id(20));
    let trusted_messages = vec![message(
        target_community_id,
        channel_id,
        100,
        1,
        10,
        principal_id(11),
    )];
    let inbox_view = cx.new(|_| InboxPulseView::new(inbox_scope, 20));
    inbox_view
        .update(cx, |view, cx| {
            view.apply_snapshot(
                1,
                inbox_projection(inbox_scope, &trusted_messages),
                vec![activity(target_community_id, "trusted", 1)],
                cx,
            )
        })
        .expect("trusted inbox snapshot");
    let foreign_community_id = community_id(2);
    let foreign_scope = InboxScope::new(foreign_community_id, principal_id(20));
    assert_eq!(
        inbox_view.update(cx, |view, cx| {
            view.apply_snapshot(2, inbox_projection(foreign_scope, &[]), Vec::new(), cx)
        }),
        Err(InboxPulseError::ScopeMismatch)
    );
    assert_eq!(
        inbox_view.update(cx, |view, cx| {
            view.apply_snapshot(
                2,
                inbox_projection(inbox_scope, &trusted_messages),
                vec![activity(foreign_community_id, "foreign", 2)],
                cx,
            )
        }),
        Err(InboxPulseError::InvalidPulseItem)
    );
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.freshness()),
        InboxPulseFreshness::Fresh { revision: 1 }
    );
    assert_eq!(
        inbox_view.read_with(cx, |view, _| view.visible_rows().len()),
        1
    );

    let trusted_forum = cx.new(|_| {
        ForumView::new(
            forum_snapshot(target_community_id, channel_id, &trusted_messages)
                .expect("trusted forum snapshot"),
            ForumPermissions::default(),
        )
    });
    let channel = Channel::from_record(ChannelRecordFields {
        community_id: target_community_id,
        channel_id,
        name: ChannelName::new("discussions").expect("channel name"),
        channel_type: ChannelType::Forum,
        visibility: ChannelVisibility::Open,
        lifecycle_state: ChannelLifecycleState::Active,
        description: None,
        creator_principal_id: principal_id(10),
        expiration: None,
        version: AggregateVersion::FIRST,
    })
    .expect("forum channel");
    let tenant = tenant(target_community_id);
    let read_scope = scope("forum:read");
    let viewer = principal(target_community_id, principal_id(20), [read_scope.clone()]);
    let request = authorization_request(
        &tenant,
        &viewer,
        &read_scope,
        AuthorizationAction::Read,
        target_community_id,
        AuthorizationResourceKind::Channel,
        channel_id,
        Some(channel_id),
        MembershipRole::Member,
    );
    let projection = ForumProjection::build(
        &channel,
        &request,
        [ForumMessageInput {
            message: &trusted_messages[0],
            author_public_key: NostrPublicKey::from_bytes([1; 32]),
            reference: ThreadReference::TopLevel,
        }],
        [],
    )
    .expect("forum projection");
    assert!(matches!(
        ForumSnapshot::from_projection(&projection, [], 20),
        Err(ForumViewError::MissingPresentation(_))
    ));
    assert_eq!(trusted_forum.read_with(cx, |view, _| view.posts().len()), 1);

    let trusted_emoji = emoji_record(
        target_community_id,
        principal_id(11),
        10,
        10,
        "https://example.com/trusted.png",
    );
    let trusted_palette =
        CustomEmojiPalette::build(target_community_id, [trusted_emoji]).expect("trusted palette");
    let foreign_emoji = emoji_record(
        foreign_community_id,
        principal_id(11),
        11,
        20,
        "https://example.com/foreign.png",
    );
    assert_eq!(
        CustomEmojiPalette::build(target_community_id, [foreign_emoji]),
        Err(CustomEmojiError::CommunityMismatch)
    );
    assert_eq!(trusted_palette.entries().count(), 1);

    let feedback = submitted_feedback(target_community_id);
    assert_eq!(
        feedback_status(&feedback, MembershipRole::Member),
        Err(FeedbackError::Unauthorized(
            AuthorizationDenial::InsufficientRole
        ))
    );
    let debug = format!("{feedback:?}");
    assert!(!debug.contains("private reconnect context"));
    assert_eq!(feedback.fields().status, FeedbackStatus::Submitted);
}
