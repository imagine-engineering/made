#![cfg(all(feature = "multiplayer-tools", feature = "test-support"))]

use std::{cell::Cell, num::NonZeroU32, rc::Rc};

use agent_ui::{
    activity_git::{
        CodeActivityProjectionContext, CollaborationCodeActivity, ReviewDecisionActivity,
        project_code_activity,
    },
    activity_projection::{ActivityActorKind, ActivityLink, ActivityVisibility},
};
use chrono::DateTime;
use collab_ui::git_activity::{GitActivityCard, GitActivityCardStatus};
use collaboration_domain::{
    AggregateId, AggregateVersion, BranchCollaboration, BranchCollaborationIdentity,
    BranchGeneration, BranchLifecycleState, BranchRefName, CiCheckRunCompletionInput,
    CiCheckRunInput, CiCheckStatus, CiCheckSuite, CiCheckSuiteIdentity, CiLabel, CiOutputText,
    CiWorkflowLink, CommunityId, GitCommitId, MergeEligibility, PatchRevisionInput,
    PatchRevisionNumber, PrincipalId, Review, ReviewCommentAnchor, ReviewCommentBody,
    ReviewCommentInput, ReviewDecision, ReviewDecisionInput, ReviewDiffSide, ReviewError,
    ReviewFilePath, ReviewHunkId, ReviewIdentity,
};
use gpui::{AppContext as _, TestAppContext};
use uuid::Uuid;
use workspace::{
    collaborative_review::CollaborativeReviewSlot,
    collaborative_review_actions::{
        CollaborativeReviewAction, CollaborativeReviewActionContext, CollaborativeReviewActionState,
    },
    collaborative_review_summary::CollaborativeReviewSummarySource,
};

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn commit(value: u64) -> GitCommitId {
    GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
}

fn branch_identity() -> BranchCollaborationIdentity {
    BranchCollaborationIdentity::new(
        CommunityId::from_uuid(Uuid::from_u128(1)),
        aggregate(2),
        BranchRefName::parse("refs/heads/agent/review-cards").expect("valid branch"),
        BranchGeneration::FIRST,
    )
    .expect("valid branch identity")
}

fn review_identity() -> ReviewIdentity {
    ReviewIdentity::new(aggregate(3), branch_identity()).expect("valid review identity")
}

fn projection_context(
    actor_kind: ActivityActorKind,
    actor_label: &str,
) -> CodeActivityProjectionContext {
    CodeActivityProjectionContext {
        actor_kind,
        actor_label: actor_label.into(),
        project_id: Some("project-patch-review".into()),
        thread_id: Some("thread-patch-review".into()),
        visibility: ActivityVisibility::Project,
        projected_at: DateTime::from_timestamp_millis(1_900_000_100_000).expect("valid timestamp"),
    }
}

fn action_context(
    source: CollaborativeReviewSummarySource,
    state: CollaborativeReviewActionState,
) -> CollaborativeReviewActionContext {
    CollaborativeReviewActionContext::new(
        source,
        state,
        [
            CollaborativeReviewAction::Stage,
            CollaborativeReviewAction::Review,
        ],
    )
}

#[gpui::test]
fn patch_review_merge_converges_and_blocks_unsafe_variants(cx: &mut TestAppContext) {
    let base_commit = commit(100);
    let head_commit = commit(101);
    let merge_commit = commit(102);
    let active_branch =
        BranchCollaboration::create(branch_identity(), head_commit.clone()).expect("active branch");
    let mut review = Review::open(
        review_identity(),
        1,
        PatchRevisionInput {
            revision_id: aggregate(4),
            base_commit: base_commit.clone(),
            head_commit: head_commit.clone(),
            author_principal_id: principal(5),
            created_at_millis: 1_900_000_001_000,
        },
    )
    .expect("open review");
    let revision = review.current_revision().expect("current revision").clone();

    let patch_item = project_code_activity(
        &projection_context(ActivityActorKind::Agent, "Patch agent"),
        &CollaborationCodeActivity::PatchSubmitted(revision.clone()),
    )
    .expect("project patch activity");
    let patch_card = GitActivityCard::new(patch_item).expect("valid patch card");
    assert_eq!(patch_card.status(), GitActivityCardStatus::Success);
    assert!(patch_card.available_actions().is_empty());

    review
        .add_comment(
            review.fields().version,
            ReviewCommentInput {
                comment_id: aggregate(6),
                author_principal_id: principal(7),
                body: ReviewCommentBody::new("Please retain the canonical error path")
                    .expect("valid comment"),
                anchor: ReviewCommentAnchor::new(
                    PatchRevisionNumber::FIRST,
                    head_commit.clone(),
                    ReviewFilePath::new("src/lib.rs").expect("valid review path"),
                    ReviewHunkId::parse("a".repeat(64)).expect("valid hunk"),
                    ReviewDiffSide::Head,
                    NonZeroU32::new(10).expect("nonzero line"),
                    NonZeroU32::new(14).expect("nonzero line"),
                )
                .expect("valid comment anchor"),
                created_at_millis: 1_900_000_002_000,
            },
        )
        .expect("record review comment");
    let comment = review
        .fields()
        .comments
        .last()
        .expect("recorded comment")
        .clone();
    let comment_item = project_code_activity(
        &projection_context(ActivityActorKind::Human, "Reviewer"),
        &CollaborationCodeActivity::ReviewCommented(comment),
    )
    .expect("project comment activity");
    let review_link = comment_item
        .links
        .iter()
        .find(|link| matches!(link, ActivityLink::GitChange { .. }))
        .expect("timeline carries a native Git-change link");
    assert!(matches!(
        review_link,
        ActivityLink::GitChange { repository_id, change_id }
            if repository_id == &aggregate(2).to_string() && !change_id.is_empty()
    ));

    let source = cx.update(|cx| {
        CollaborativeReviewSummarySource::new(
            CollaborativeReviewSlot::ProjectChanges,
            cx.new(|_| ()).entity_id(),
            1,
        )
    });
    let ready_card = GitActivityCard::new(comment_item.clone())
        .expect("valid comment card")
        .with_review(action_context(
            source,
            CollaborativeReviewActionState::Ready,
        ))
        .expect("exact native review card");
    assert_eq!(
        ready_card.available_actions(),
        [
            CollaborativeReviewAction::Stage,
            CollaborativeReviewAction::Review
        ]
    );
    let request = ready_card
        .action_request(CollaborativeReviewAction::Review)
        .expect("native review action");
    assert_eq!(request.source(), source);
    let invoked_action = Rc::new(Cell::new(None));
    let executed = ready_card
        .route_native_action(request, {
            let invoked_action = invoked_action.clone();
            move |action| {
                invoked_action.set(Some(action));
                Ok(())
            }
        })
        .expect("route exact native review action");
    assert_eq!(executed.source(), source);
    assert_eq!(
        invoked_action.get(),
        Some(CollaborativeReviewAction::Review)
    );
    let ready_card = cx.new(|_| ready_card);
    ready_card.update(cx, |card, cx| card.set_details_expanded(true, cx));
    assert!(ready_card.read_with(cx, |card, _| card.is_details_expanded()));

    let mut ci_suite = CiCheckSuite::create(
        CiCheckSuiteIdentity::for_revision(aggregate(8), &revision).expect("CI identity"),
        CiWorkflowLink::new(
            aggregate(9),
            aggregate(10),
            CiLabel::from_untrusted("build and test").expect("CI label"),
            None,
        )
        .expect("workflow link"),
        1_900_000_003_000,
    );
    let pending_ci = GitActivityCard::new(
        project_code_activity(
            &projection_context(ActivityActorKind::Service, "CI"),
            &CollaborationCodeActivity::CiStatusChanged(ci_suite.clone()),
        )
        .expect("project pending CI"),
    )
    .expect("pending CI card");
    assert_eq!(pending_ci.status(), GitActivityCardStatus::Pending);
    let check_run_id = aggregate(11);
    ci_suite
        .add_run(
            AggregateVersion::FIRST,
            CiCheckRunInput {
                check_run_id,
                label: CiLabel::from_untrusted("tests").expect("run label"),
                queued_at_millis: 1_900_000_004_000,
            },
        )
        .expect("queue CI run");
    ci_suite
        .start_run(
            ci_suite.fields().version,
            check_run_id,
            AggregateVersion::FIRST,
            1_900_000_005_000,
        )
        .expect("start CI run");
    let running_ci = GitActivityCard::new(
        project_code_activity(
            &projection_context(ActivityActorKind::Service, "CI"),
            &CollaborationCodeActivity::CiStatusChanged(ci_suite.clone()),
        )
        .expect("project running CI"),
    )
    .expect("running CI card");
    assert_eq!(running_ci.status(), GitActivityCardStatus::Running);
    let running_version = ci_suite
        .fields()
        .runs
        .first()
        .expect("running check")
        .version;
    ci_suite
        .complete_run(
            ci_suite.fields().version,
            check_run_id,
            running_version,
            &head_commit,
            CiCheckRunCompletionInput {
                status: CiCheckStatus::Success,
                output: CiOutputText::from_untrusted("all checks passed"),
                artifacts: Vec::new(),
                completed_at_millis: 1_900_000_006_000,
            },
        )
        .expect("complete CI run");
    let passed_ci = GitActivityCard::new(
        project_code_activity(
            &projection_context(ActivityActorKind::Service, "CI"),
            &CollaborationCodeActivity::CiStatusChanged(ci_suite.clone()),
        )
        .expect("project successful CI"),
    )
    .expect("successful CI card");
    assert_eq!(passed_ci.status(), GitActivityCardStatus::Success);

    review
        .record_decision(
            review.fields().version,
            ReviewDecisionInput {
                approval_id: aggregate(12),
                revision: PatchRevisionNumber::FIRST,
                head_commit: head_commit.clone(),
                approver_principal_id: principal(13),
                decision: ReviewDecision::Approve,
                created_at_millis: 1_900_000_007_000,
            },
        )
        .expect("approve revision");
    let approval = review
        .fields()
        .approvals
        .last()
        .expect("recorded approval")
        .clone();
    let approval_card = GitActivityCard::new(
        project_code_activity(
            &projection_context(ActivityActorKind::Human, "Approver"),
            &CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity { approval }),
        )
        .expect("project approval"),
    )
    .expect("approval card");
    assert_eq!(approval_card.status(), GitActivityCardStatus::Success);
    assert!(approval_card.available_actions().is_empty());
    let readiness = review
        .merge_readiness(PatchRevisionNumber::FIRST, &head_commit)
        .expect("current merge readiness");
    assert_eq!(readiness.eligibility, MergeEligibility::Ready);
    assert_eq!(ci_suite.status(), CiCheckStatus::Success);

    let mut merged_branch = active_branch.clone();
    merged_branch
        .merge(
            AggregateVersion::FIRST,
            &head_commit,
            BranchRefName::parse("refs/heads/dev").expect("valid merge target"),
            merge_commit.clone(),
        )
        .expect("merge approved green revision");
    assert_eq!(
        merged_branch.fields().lifecycle_state,
        BranchLifecycleState::Merged
    );
    let merge = merged_branch.fields().merge.as_ref().expect("merge record");
    assert_eq!(merge.source_commit(), &head_commit);
    assert_eq!(merge.result_commit(), &merge_commit);

    let mut newer_review = review.clone();
    newer_review
        .submit_revision(
            newer_review.fields().version,
            PatchRevisionNumber::FIRST,
            PatchRevisionInput {
                revision_id: aggregate(14),
                base_commit,
                head_commit: commit(103),
                author_principal_id: principal(5),
                created_at_millis: 1_900_000_008_000,
            },
        )
        .expect("submit replacement revision");
    assert!(matches!(
        newer_review.merge_readiness(PatchRevisionNumber::FIRST, &head_commit),
        Err(ReviewError::StaleRevision { .. })
    ));
    let stale_card = GitActivityCard::new(comment_item.clone())
        .expect("valid stale comment card")
        .with_review(action_context(
            source,
            CollaborativeReviewActionState::Stale,
        ))
        .expect("stale review card");
    assert_eq!(stale_card.status(), GitActivityCardStatus::Stale);
    assert!(stale_card.available_actions().is_empty());
    assert!(
        stale_card
            .action_request(CollaborativeReviewAction::Review)
            .is_err()
    );

    let conflict_card = GitActivityCard::new(comment_item)
        .expect("valid conflicting comment card")
        .with_review(action_context(
            source,
            CollaborativeReviewActionState::Conflict,
        ))
        .expect("conflicting review card");
    assert_eq!(conflict_card.status(), GitActivityCardStatus::Conflict);
    assert!(conflict_card.available_actions().is_empty());
    assert!(
        conflict_card
            .action_request(CollaborativeReviewAction::Stage)
            .is_err()
    );
    assert_eq!(
        active_branch.fields().lifecycle_state,
        BranchLifecycleState::Active
    );
    assert!(active_branch.fields().merge.is_none());
}
