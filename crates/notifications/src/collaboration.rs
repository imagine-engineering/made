use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt,
    rc::Rc,
};

use collaboration_domain::{
    NotificationCandidate, NotificationDeliveryId, NotificationDevicePermissions,
    NotificationPrivacy, NotificationSuppression, NotificationSurfaceDecision, decide_notification,
};
use gpui::{App, SharedString, SystemNotification, SystemNotificationAction};
use sha2::{Digest as _, Sha256};
use workspace::collaborative_navigation::CollaborativeNavigationTarget;

const MAX_PENDING_TARGETS: usize = 500;
const MAX_TITLE_BYTES: usize = 128;
const MAX_BODY_BYTES: usize = 512;
const OPEN_ACTION_ID: &str = "open";
const PRIVATE_TITLE: &str = "New private activity";
const PRIVATE_BODY: &str = "Open Zed to view it.";

pub struct CollaborationNotificationRecord {
    candidate: NotificationCandidate,
    target: CollaborativeNavigationTarget,
    preview: CollaborationNotificationPreview,
}

enum CollaborationNotificationPreview {
    CommunityVisible {
        title: SharedString,
        body: SharedString,
    },
    Private,
}

impl CollaborationNotificationRecord {
    pub fn community_visible(
        candidate: NotificationCandidate,
        target: CollaborativeNavigationTarget,
        title: impl Into<SharedString>,
        body: impl Into<SharedString>,
    ) -> Result<Self, CollaborationNotificationRecordError> {
        if candidate.privacy != NotificationPrivacy::CommunityVisible {
            return Err(CollaborationNotificationRecordError::PrivacyMismatch);
        }
        let title = title.into();
        let body = body.into();
        if !valid_title(&title) || !valid_body(&body) {
            return Err(CollaborationNotificationRecordError::InvalidPreview);
        }
        Ok(Self {
            candidate,
            target,
            preview: CollaborationNotificationPreview::CommunityVisible { title, body },
        })
    }

    pub fn private(
        candidate: NotificationCandidate,
        target: CollaborativeNavigationTarget,
    ) -> Result<Self, CollaborationNotificationRecordError> {
        if !matches!(candidate.privacy, NotificationPrivacy::Private { .. }) {
            return Err(CollaborationNotificationRecordError::PrivacyMismatch);
        }
        Ok(Self {
            candidate,
            target,
            preview: CollaborationNotificationPreview::Private,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationNotificationRecordError {
    PrivacyMismatch,
    InvalidPreview,
}

impl fmt::Display for CollaborationNotificationRecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrivacyMismatch => formatter.write_str("notification privacy mismatch"),
            Self::InvalidPreview => formatter.write_str("invalid notification preview"),
        }
    }
}

impl std::error::Error for CollaborationNotificationRecordError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationNotificationDispatch {
    Posted(NotificationDeliveryId),
    Suppressed(NotificationSuppression),
}

struct PendingTargets {
    order: VecDeque<SharedString>,
    targets: HashMap<SharedString, CollaborativeNavigationTarget>,
}

impl PendingTargets {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            targets: HashMap::new(),
        }
    }

    fn insert(&mut self, tag: SharedString, target: CollaborativeNavigationTarget) {
        if !self.targets.contains_key(&tag) {
            if self.order.len() == MAX_PENDING_TARGETS
                && let Some(expired_tag) = self.order.pop_front()
            {
                self.targets.remove(&expired_tag);
            }
            self.order.push_back(tag.clone());
        }
        self.targets.insert(tag, target);
    }

    fn remove(&mut self, tag: &str) -> Option<CollaborativeNavigationTarget> {
        let target = self.targets.remove(tag)?;
        if let Some(index) = self.order.iter().position(|candidate| candidate == tag) {
            self.order.remove(index);
        }
        Some(target)
    }
}

pub struct CollaborationNotificationDispatcher {
    pending_targets: Rc<RefCell<PendingTargets>>,
}

impl CollaborationNotificationDispatcher {
    pub fn new(
        cx: &mut App,
        mut navigate: impl 'static + FnMut(CollaborativeNavigationTarget, &mut App) -> bool,
    ) -> Self {
        let pending_targets = Rc::new(RefCell::new(PendingTargets::new()));
        cx.on_system_notification_response({
            let pending_targets = pending_targets.clone();
            move |response, cx| {
                if response
                    .action_id
                    .as_ref()
                    .is_some_and(|action_id| action_id.as_ref() != OPEN_ACTION_ID)
                {
                    return;
                }
                let Some(target) = pending_targets.borrow_mut().remove(&response.tag) else {
                    return;
                };
                if !navigate(target, cx) {
                    log::info!("collaboration notification target is no longer available");
                }
            }
        });
        Self { pending_targets }
    }

    pub fn dispatch(
        &self,
        record: CollaborationNotificationRecord,
        permissions: NotificationDevicePermissions,
        already_delivered: impl Fn(&NotificationDeliveryId) -> bool,
        cx: &mut App,
    ) -> CollaborationNotificationDispatch {
        let decision = decide_notification(&record.candidate, permissions, already_delivered);
        let delivery_id = match decision.native {
            NotificationSurfaceDecision::Deliver(delivery_id) => delivery_id,
            NotificationSurfaceDecision::Suppress(suppression) => {
                return CollaborationNotificationDispatch::Suppressed(suppression);
            }
        };
        let tag = notification_tag(&delivery_id);
        let (title, body) = match record.preview {
            CollaborationNotificationPreview::CommunityVisible { title, body } => (title, body),
            CollaborationNotificationPreview::Private => {
                (PRIVATE_TITLE.into(), PRIVATE_BODY.into())
            }
        };
        self.pending_targets
            .borrow_mut()
            .insert(tag.clone(), record.target);
        cx.show_system_notification(SystemNotification {
            tag,
            title,
            body,
            actions: vec![SystemNotificationAction {
                id: OPEN_ACTION_ID.into(),
                label: "Open".into(),
            }],
        });
        CollaborationNotificationDispatch::Posted(delivery_id)
    }
}

fn notification_tag(delivery_id: &NotificationDeliveryId) -> SharedString {
    let source = delivery_id.source();
    let mut digest = Sha256::new();
    digest.update(source.community_id().as_uuid().as_bytes());
    digest.update(source_system_label(source.source_system()).as_bytes());
    digest.update(source.source_record_id().as_str().as_bytes());
    digest.update(delivery_id.recipient_principal_id().as_uuid().as_bytes());
    digest.update(b"native");
    let digest = digest.finalize();
    format!("collaboration-{digest:x}").into()
}

fn source_system_label(source_system: collaboration_domain::SourceSystem) -> &'static str {
    match source_system {
        collaboration_domain::SourceSystem::Zed => "zed",
        collaboration_domain::SourceSystem::Buzz => "buzz",
        collaboration_domain::SourceSystem::Nostr => "nostr",
        collaboration_domain::SourceSystem::Acp => "acp",
        collaboration_domain::SourceSystem::ExternalGit => "external-git",
    }
}

fn valid_title(title: &str) -> bool {
    !title.is_empty() && title.len() <= MAX_TITLE_BYTES && !title.chars().any(char::is_control)
}

fn valid_body(body: &str) -> bool {
    !body.is_empty()
        && body.len() <= MAX_BODY_BYTES
        && !body
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::HashSet};

    use collaboration_domain::{
        AggregateId, AggregateVersion, ChannelMembership, CommunityId, CommunityMembership,
        MembershipRole, MembershipStatus, NotificationMembership, NotificationPermission,
        NotificationReadState, NotificationReason, NotificationSourceId, PrincipalId,
        SourceRecordId, SourceSystem,
    };
    use gpui::{SystemNotificationResponse, TestAppContext};
    use uuid::Uuid;

    use super::*;

    fn community() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn recipient() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(2))
    }

    fn author() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(3))
    }

    fn channel() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(4))
    }

    fn candidate(privacy: NotificationPrivacy) -> NotificationCandidate {
        NotificationCandidate {
            source: NotificationSourceId::new(
                community(),
                SourceSystem::Nostr,
                SourceRecordId::new("secret-event-id").expect("valid source ID"),
            ),
            recipient_principal_id: recipient(),
            author_principal_id: author(),
            channel_id: Some(channel()),
            reason: NotificationReason::Mention,
            membership: NotificationMembership::channel(
                CommunityMembership {
                    community_id: community(),
                    principal_id: recipient(),
                    role: MembershipRole::Member,
                    status: MembershipStatus::Active,
                    version: AggregateVersion::FIRST,
                },
                ChannelMembership {
                    community_id: community(),
                    channel_id: channel(),
                    principal_id: recipient(),
                    role: MembershipRole::Member,
                    status: MembershipStatus::Active,
                    version: AggregateVersion::FIRST,
                },
            ),
            privacy,
            read_state: NotificationReadState::Unread,
            muted: false,
        }
    }

    fn permissions(native: NotificationPermission) -> NotificationDevicePermissions {
        NotificationDevicePermissions {
            native,
            push: NotificationPermission::Disabled,
        }
    }

    fn target() -> CollaborativeNavigationTarget {
        CollaborativeNavigationTarget::channel("channel-one")
    }

    fn init(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_app_identity("dev.zed.notifications.test", "Zed Tests"));
    }

    #[gpui::test]
    fn collaboration_notification_permission_denial_posts_nothing(cx: &mut TestAppContext) {
        init(cx);
        let dispatcher = cx.update(|cx| CollaborationNotificationDispatcher::new(cx, |_, _| true));
        let record = CollaborationNotificationRecord::community_visible(
            candidate(NotificationPrivacy::CommunityVisible),
            target(),
            "Mention",
            "A visible message",
        )
        .expect("valid record");

        assert_eq!(
            cx.update(|cx| dispatcher.dispatch(
                record,
                permissions(NotificationPermission::Denied),
                |_| false,
                cx,
            )),
            CollaborationNotificationDispatch::Suppressed(
                NotificationSuppression::PermissionDenied
            )
        );
        assert!(cx.shown_system_notifications().is_empty());
    }

    #[gpui::test]
    fn collaboration_notification_deduplicates_canonical_native_delivery(cx: &mut TestAppContext) {
        init(cx);
        let dispatcher = cx.update(|cx| CollaborationNotificationDispatcher::new(cx, |_, _| true));
        let delivered = RefCell::new(HashSet::new());
        let make_record = || {
            CollaborationNotificationRecord::community_visible(
                candidate(NotificationPrivacy::CommunityVisible),
                target(),
                "Mention",
                "A visible message",
            )
            .expect("valid record")
        };

        let first = cx.update(|cx| {
            dispatcher.dispatch(
                make_record(),
                permissions(NotificationPermission::Granted),
                |delivery_id| delivered.borrow().contains(delivery_id),
                cx,
            )
        });
        let CollaborationNotificationDispatch::Posted(delivery_id) = first else {
            panic!("first notification should post");
        };
        delivered.borrow_mut().insert(delivery_id);
        let second = cx.update(|cx| {
            dispatcher.dispatch(
                make_record(),
                permissions(NotificationPermission::Granted),
                |delivery_id| delivered.borrow().contains(delivery_id),
                cx,
            )
        });

        assert_eq!(
            second,
            CollaborationNotificationDispatch::Suppressed(NotificationSuppression::Duplicate)
        );
        assert_eq!(cx.shown_system_notifications().len(), 1);
    }

    #[gpui::test]
    fn collaboration_notification_private_preview_is_always_redacted(cx: &mut TestAppContext) {
        init(cx);
        let dispatcher = cx.update(|cx| CollaborationNotificationDispatcher::new(cx, |_, _| true));
        let record = CollaborationNotificationRecord::private(
            candidate(NotificationPrivacy::Private {
                recipient_is_participant: true,
            }),
            target(),
        )
        .expect("valid private record");
        cx.update(|cx| {
            dispatcher.dispatch(
                record,
                permissions(NotificationPermission::Granted),
                |_| false,
                cx,
            );
        });

        let notifications = cx.shown_system_notifications();
        assert_eq!(notifications.len(), 1);
        let notification = notifications.first().expect("shown notification");
        assert_eq!(notification.title, PRIVATE_TITLE);
        assert_eq!(notification.body, PRIVATE_BODY);
        let rendered = format!("{notification:?}");
        assert!(!rendered.contains("secret-event-id"));
    }

    #[gpui::test]
    fn collaboration_notification_missing_target_fails_safely(cx: &mut TestAppContext) {
        init(cx);
        let attempts = Rc::new(Cell::new(0));
        let dispatcher = cx.update(|cx| {
            CollaborationNotificationDispatcher::new(cx, {
                let attempts = attempts.clone();
                move |_, _| {
                    attempts.set(attempts.get() + 1);
                    false
                }
            })
        });
        let record = CollaborationNotificationRecord::community_visible(
            candidate(NotificationPrivacy::CommunityVisible),
            target(),
            "Mention",
            "A visible message",
        )
        .expect("valid record");
        cx.update(|cx| {
            dispatcher.dispatch(
                record,
                permissions(NotificationPermission::Granted),
                |_| false,
                cx,
            );
        });
        let notification = cx
            .shown_system_notifications()
            .into_iter()
            .next()
            .expect("shown notification");

        for _ in 0..2 {
            cx.simulate_system_notification_response(SystemNotificationResponse {
                tag: notification.tag.clone(),
                action_id: Some(OPEN_ACTION_ID.into()),
            });
        }
        assert_eq!(attempts.get(), 1);
    }
}
