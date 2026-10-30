use std::{error::Error, fmt};

use agent_ui::activity_projection::{
    ActivityDetailHandle, ActivityItem, ActivityLifecycle, ActivityObjectKind,
    ActivityOutcomeStatus, ActivitySourceKind,
};
use gpui::{Context, EventEmitter, Render, Role, Window};
use ui::{Button, ButtonStyle, LabelSize, prelude::*};
use util::ResultExt as _;
use workspace::{
    collaborative_review::CollaborativeReviewSlot,
    collaborative_review_actions::{
        CollaborativeReviewAction, CollaborativeReviewActionContext,
        CollaborativeReviewActionError, CollaborativeReviewActionRequest,
        CollaborativeReviewActionState, ExecutedCollaborativeReviewAction,
        route_collaborative_review_action,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitActivityCardStatus {
    Pending,
    Running,
    Success,
    Failure,
    Cancelled,
    Conflict,
    Stale,
    Deleted,
    Unknown,
}

impl GitActivityCardStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Success => "Completed",
            Self::Failure => "Failed",
            Self::Cancelled => "Cancelled",
            Self::Conflict => "Conflict",
            Self::Stale => "Stale",
            Self::Deleted => "Deleted",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitActivityCardEvent {
    NativeActionRequested(CollaborativeReviewActionRequest),
}

pub struct GitActivityCard {
    item: ActivityItem,
    action_context: Option<CollaborativeReviewActionContext>,
    details_expanded: bool,
}

impl GitActivityCard {
    pub fn new(item: ActivityItem) -> Result<Self, GitActivityCardError> {
        if !matches!(
            item.id.source_kind(),
            ActivitySourceKind::Git | ActivitySourceKind::Workflow | ActivitySourceKind::Ci
        ) || item.source_version == 0
            || item.actor.id.trim().is_empty()
            || item.actor.label.trim().is_empty()
            || item.verb.trim().is_empty()
            || item.object.label.trim().is_empty()
        {
            return Err(GitActivityCardError::InvalidActivity);
        }
        Ok(Self {
            item,
            action_context: None,
            details_expanded: false,
        })
    }

    pub fn with_review(
        mut self,
        action_context: CollaborativeReviewActionContext,
    ) -> Result<Self, GitActivityCardError> {
        if self.item.id.source_kind() != ActivitySourceKind::Git
            || !matches!(
                self.item.object.kind,
                ActivityObjectKind::Review | ActivityObjectKind::File
            )
            || !self.item.links.iter().any(|link| {
                matches!(
                    link,
                    agent_ui::activity_projection::ActivityLink::GitChange { .. }
                )
            })
        {
            return Err(GitActivityCardError::InvalidReviewActivity);
        }
        if action_context.source().slot() != CollaborativeReviewSlot::ProjectChanges {
            return Err(GitActivityCardError::InvalidActionContext);
        }
        self.action_context = Some(action_context);
        Ok(self)
    }

    pub fn item(&self) -> &ActivityItem {
        &self.item
    }

    pub fn status(&self) -> GitActivityCardStatus {
        if let Some(context) = &self.action_context {
            match context.state() {
                CollaborativeReviewActionState::Conflict => {
                    return GitActivityCardStatus::Conflict;
                }
                CollaborativeReviewActionState::Stale => return GitActivityCardStatus::Stale,
                CollaborativeReviewActionState::Rejected => {
                    return GitActivityCardStatus::Failure;
                }
                CollaborativeReviewActionState::Ready => {}
            }
        }
        match self.item.lifecycle {
            ActivityLifecycle::Pending => GitActivityCardStatus::Pending,
            ActivityLifecycle::Running => GitActivityCardStatus::Running,
            ActivityLifecycle::Failed | ActivityLifecycle::TimedOut => {
                GitActivityCardStatus::Failure
            }
            ActivityLifecycle::Cancelled => GitActivityCardStatus::Cancelled,
            ActivityLifecycle::Succeeded => match self.item.outcome.status {
                ActivityOutcomeStatus::Success | ActivityOutcomeStatus::NoChange => {
                    GitActivityCardStatus::Success
                }
                ActivityOutcomeStatus::Failure | ActivityOutcomeStatus::TimedOut => {
                    GitActivityCardStatus::Failure
                }
                ActivityOutcomeStatus::Cancelled => GitActivityCardStatus::Cancelled,
                ActivityOutcomeStatus::Pending => GitActivityCardStatus::Pending,
                ActivityOutcomeStatus::Unknown => GitActivityCardStatus::Unknown,
            },
            ActivityLifecycle::WaitingForUser
            | ActivityLifecycle::Idle
            | ActivityLifecycle::Disconnected
            | ActivityLifecycle::Suppressed => GitActivityCardStatus::Unknown,
        }
    }

    pub fn is_details_expanded(&self) -> bool {
        self.details_expanded
    }

    pub fn set_details_expanded(&mut self, expanded: bool, cx: &mut Context<Self>) {
        if self.details_expanded != expanded {
            self.details_expanded = expanded;
            cx.notify();
        }
    }

    pub fn available_actions(&self) -> Vec<CollaborativeReviewAction> {
        let Some(context) = &self.action_context else {
            return Vec::new();
        };
        if context.state() != CollaborativeReviewActionState::Ready {
            return Vec::new();
        }
        [
            CollaborativeReviewAction::Stage,
            CollaborativeReviewAction::Review,
        ]
        .into_iter()
        .filter(|action| context.is_available(*action))
        .collect()
    }

    pub fn action_request(
        &self,
        action: CollaborativeReviewAction,
    ) -> Result<CollaborativeReviewActionRequest, GitActivityCardError> {
        let context = self
            .action_context
            .as_ref()
            .ok_or(GitActivityCardError::ActionUnavailable)?;
        if !self.available_actions().contains(&action) {
            return Err(GitActivityCardError::ActionUnavailable);
        }
        Ok(CollaborativeReviewActionRequest::new(
            context.source(),
            action,
        ))
    }

    pub fn route_native_action(
        &self,
        request: CollaborativeReviewActionRequest,
        invoke_native: impl FnOnce(CollaborativeReviewAction) -> Result<(), String>,
    ) -> Result<ExecutedCollaborativeReviewAction, GitActivityCardError> {
        let context = self
            .action_context
            .as_ref()
            .ok_or(GitActivityCardError::ActionUnavailable)?;
        if !self.available_actions().contains(&request.action()) {
            return Err(GitActivityCardError::ActionUnavailable);
        }
        route_collaborative_review_action(context, request, invoke_native)
            .map_err(GitActivityCardError::Action)
    }

    fn toggle_details(&mut self, cx: &mut Context<Self>) {
        self.details_expanded = !self.details_expanded;
        cx.notify();
    }

    fn request_action(&mut self, action: CollaborativeReviewAction, cx: &mut Context<Self>) {
        if let Some(request) = self.action_request(action).log_err() {
            cx.emit(GitActivityCardEvent::NativeActionRequested(request));
        }
    }
}

impl EventEmitter<GitActivityCardEvent> for GitActivityCard {}

impl Render for GitActivityCard {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let status = self.status();
        let details_expanded = self.details_expanded;
        let details =
            details_expanded.then(|| detail_text(&self.item, self.action_context.as_ref()));
        let actions = self.available_actions();

        v_flex()
            .id("git-activity-card")
            .role(Role::Group)
            .aria_label(format!(
                "{}. {}. {}",
                headline(&self.item),
                status.label(),
                self.item
                    .outcome
                    .summary
                    .as_deref()
                    .unwrap_or("No outcome detail")
            ))
            .w_full()
            .p_3()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().colors().border)
            .bg(cx.theme().colors().editor_background)
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .gap_3()
                    .child(
                        v_flex()
                            .min_w_0()
                            .gap_0p5()
                            .child(div().text_ui(cx).child(headline(&self.item)))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(status.label()),
                            ),
                    )
                    .child(
                        Button::new(
                            "git-activity-details",
                            if details_expanded {
                                "Hide details"
                            } else {
                                "Show details"
                            },
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .aria_expanded(details_expanded)
                        .on_click(cx.listener(|this, _, _, cx| this.toggle_details(cx))),
                    ),
            )
            .when_some(self.item.outcome.summary.clone(), |this, summary| {
                this.child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().colors().text_muted)
                        .child(summary),
                )
            })
            .when_some(details, |this, details| {
                this.child(
                    div()
                        .p_2()
                        .rounded_sm()
                        .border_1()
                        .border_color(cx.theme().colors().border_variant)
                        .text_xs()
                        .text_color(cx.theme().colors().text_muted)
                        .child(details),
                )
            })
            .when(!actions.is_empty(), |this| {
                this.child(h_flex().gap_2().children(actions.into_iter().map(|action| {
                    Button::new(
                        ("git-activity-action", action_index(action)),
                        action_label(action),
                    )
                    .style(ButtonStyle::Subtle)
                    .label_size(LabelSize::Small)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.request_action(action, cx);
                    }))
                })))
            })
    }
}

#[derive(Debug)]
pub enum GitActivityCardError {
    InvalidActivity,
    InvalidReviewActivity,
    InvalidActionContext,
    ActionUnavailable,
    Action(CollaborativeReviewActionError),
}

impl fmt::Display for GitActivityCardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidActivity => formatter.write_str("Git activity item is invalid"),
            Self::InvalidReviewActivity => {
                formatter.write_str("Git activity item cannot resolve to a native review")
            }
            Self::InvalidActionContext => {
                formatter.write_str("native review action context does not match the diff outcome")
            }
            Self::ActionUnavailable => formatter.write_str("native review action is unavailable"),
            Self::Action(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl Error for GitActivityCardError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Action(error) => Some(error),
            _ => None,
        }
    }
}

fn headline(item: &ActivityItem) -> String {
    format!("{} {} {}", item.actor.label, item.verb, item.object.label)
}

fn action_label(action: CollaborativeReviewAction) -> &'static str {
    match action {
        CollaborativeReviewAction::Stage => "Stage",
        CollaborativeReviewAction::Review => "Review",
        CollaborativeReviewAction::Keep => "Keep",
        CollaborativeReviewAction::Reject => "Reject",
    }
}

fn action_index(action: CollaborativeReviewAction) -> usize {
    match action {
        CollaborativeReviewAction::Keep => 0,
        CollaborativeReviewAction::Reject => 1,
        CollaborativeReviewAction::Stage => 2,
        CollaborativeReviewAction::Review => 3,
    }
}

fn detail_text(
    item: &ActivityItem,
    action_context: Option<&CollaborativeReviewActionContext>,
) -> String {
    let source = match &item.details {
        Some(ActivityDetailHandle::GitChange {
            repository_id,
            change_id,
        }) => format!("Repository {repository_id}, change {change_id}"),
        Some(ActivityDetailHandle::WorkflowRun { run_id, step_id }) => {
            step_id.as_ref().map_or_else(
                || format!("Workflow run {run_id}"),
                |step_id| format!("Workflow run {run_id}, step {step_id}"),
            )
        }
        Some(ActivityDetailHandle::RawSource { item_id }) => {
            format!(
                "Raw event {:?}:{}",
                item_id.source_kind(),
                item_id.source_id()
            )
        }
        Some(_) | None => "No progressive Git detail is available".into(),
    };
    let review = action_context.map(|context| match context.state() {
        CollaborativeReviewActionState::Ready => "Native review ready",
        CollaborativeReviewActionState::Conflict => "Native review has conflicts",
        CollaborativeReviewActionState::Rejected => "Native review rejected",
        CollaborativeReviewActionState::Stale => "Native review changed",
    });
    review.map_or(source.clone(), |review| format!("{source}. {review}"))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use agent_ui::{
        activity_git::{
            CodeActivityProjectionContext, CollaborationCodeActivity, ReviewDecisionActivity,
            project_code_activity,
        },
        activity_projection::{ActivityActorKind, ActivityVisibility},
    };
    use collaboration_domain::{
        AggregateId, BranchCollaborationIdentity, BranchGeneration, BranchRefName, CiCheckStatus,
        CiCheckSuite, CiCheckSuiteIdentity, CiLabel, CiWorkflowLink, CommunityId, GitCommitId,
        PatchRevision, PatchRevisionNumber, PrincipalId, ReviewApproval, ReviewDecision,
        ReviewIdentity,
    };
    use gpui::TestAppContext;
    use uuid::Uuid;
    use workspace::{
        collaborative_review::CollaborativeReviewSlot,
        collaborative_review_actions::{
            CollaborativeReviewActionContext, CollaborativeReviewActionState,
        },
        collaborative_review_summary::CollaborativeReviewSummarySource,
    };

    use super::*;

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn commit(value: u64) -> GitCommitId {
        GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
    }

    fn review_identity() -> ReviewIdentity {
        ReviewIdentity::new(
            aggregate(3),
            BranchCollaborationIdentity::new(
                CommunityId::from_uuid(Uuid::from_u128(1)),
                aggregate(2),
                BranchRefName::parse("refs/heads/feature/cards").expect("valid branch"),
                BranchGeneration::FIRST,
            )
            .expect("valid branch identity"),
        )
        .expect("valid review identity")
    }

    fn revision() -> PatchRevision {
        PatchRevision {
            revision_id: aggregate(4),
            review: review_identity(),
            number: PatchRevisionNumber::FIRST,
            base_commit: commit(100),
            head_commit: commit(101),
            author_principal_id: PrincipalId::from_uuid(Uuid::from_u128(5)),
            created_at_millis: 1_900_000_000_000,
        }
    }

    fn projection_context() -> CodeActivityProjectionContext {
        CodeActivityProjectionContext {
            actor_kind: ActivityActorKind::Human,
            actor_label: "Ada".into(),
            project_id: Some("project-1".into()),
            thread_id: Some("thread-1".into()),
            visibility: ActivityVisibility::Project,
            projected_at: chrono::DateTime::from_timestamp_millis(1_900_000_010_000)
                .expect("valid timestamp"),
        }
    }

    fn approval_item() -> ActivityItem {
        project_code_activity(
            &projection_context(),
            &CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity {
                approval: ReviewApproval {
                    approval_id: aggregate(6),
                    review: review_identity(),
                    revision: PatchRevisionNumber::FIRST,
                    head_commit: commit(101),
                    approver_principal_id: PrincipalId::from_uuid(Uuid::from_u128(7)),
                    decision: ReviewDecision::Approve,
                    created_at_millis: 1_900_000_001_000,
                },
            }),
        )
        .expect("approval should project")
    }

    fn pending_ci_item() -> ActivityItem {
        let revision = revision();
        let suite = CiCheckSuite::create(
            CiCheckSuiteIdentity::for_revision(aggregate(8), &revision).expect("valid CI identity"),
            CiWorkflowLink::new(
                aggregate(9),
                aggregate(10),
                CiLabel::from_untrusted("build and test").expect("valid label"),
                None,
            )
            .expect("valid workflow"),
            1_900_000_002_000,
        );
        assert_eq!(suite.status(), CiCheckStatus::Pending);
        project_code_activity(
            &projection_context(),
            &CollaborationCodeActivity::CiStatusChanged(suite),
        )
        .expect("pending CI should project")
    }

    fn action_context(
        cx: &mut TestAppContext,
        state: CollaborativeReviewActionState,
    ) -> CollaborativeReviewActionContext {
        let provider_id = cx.update(|cx| cx.new(|_| ()).entity_id());
        CollaborativeReviewActionContext::new(
            CollaborativeReviewSummarySource::new(
                CollaborativeReviewSlot::ProjectChanges,
                provider_id,
                1,
            ),
            state,
            [
                CollaborativeReviewAction::Stage,
                CollaborativeReviewAction::Review,
            ],
        )
    }

    #[gpui::test]
    fn git_activity_card_shows_pending_ci_and_progressive_details(cx: &mut TestAppContext) {
        let card = GitActivityCard::new(pending_ci_item()).expect("valid CI card");
        assert_eq!(card.status(), GitActivityCardStatus::Pending);
        assert!(card.available_actions().is_empty());
        let card = cx.new(|_| card);
        card.update(cx, |card, cx| card.set_details_expanded(true, cx));
        assert!(card.read_with(cx, |card, _| card.is_details_expanded()));
    }

    #[gpui::test]
    fn git_activity_card_shows_immutable_approval(_cx: &mut TestAppContext) {
        let card = GitActivityCard::new(approval_item()).expect("valid approval card");
        assert_eq!(card.status(), GitActivityCardStatus::Success);
        assert_eq!(headline(card.item()), "Ada approved review revision 1");
        assert!(card.available_actions().is_empty());
    }

    #[gpui::test]
    fn git_activity_card_blocks_conflicting_review(cx: &mut TestAppContext) {
        let card = GitActivityCard::new(approval_item())
            .expect("valid approval card")
            .with_review(action_context(cx, CollaborativeReviewActionState::Conflict))
            .expect("matching conflict card");
        assert_eq!(card.status(), GitActivityCardStatus::Conflict);
        assert!(card.available_actions().is_empty());
    }

    #[gpui::test]
    fn git_activity_card_exposes_stale_review(cx: &mut TestAppContext) {
        let card = GitActivityCard::new(approval_item())
            .expect("valid approval card")
            .with_review(action_context(cx, CollaborativeReviewActionState::Stale))
            .expect("matching stale card");
        assert_eq!(card.status(), GitActivityCardStatus::Stale);
        assert!(card.available_actions().is_empty());
    }

    #[gpui::test]
    fn git_activity_card_routes_valid_native_actions(cx: &mut TestAppContext) {
        let card = GitActivityCard::new(approval_item())
            .expect("valid approval card")
            .with_review(action_context(cx, CollaborativeReviewActionState::Ready))
            .expect("matching ready card");
        assert_eq!(
            card.available_actions(),
            [
                CollaborativeReviewAction::Stage,
                CollaborativeReviewAction::Review
            ]
        );
        let invoked = Rc::new(Cell::new(None));
        let request = card
            .action_request(CollaborativeReviewAction::Stage)
            .expect("stage request");
        let executed = card
            .route_native_action(request, {
                let invoked = invoked.clone();
                move |action| {
                    invoked.set(Some(action));
                    Ok(())
                }
            })
            .expect("native stage should route");
        assert_eq!(executed.action(), CollaborativeReviewAction::Stage);
        assert_eq!(invoked.get(), Some(CollaborativeReviewAction::Stage));
    }
}
