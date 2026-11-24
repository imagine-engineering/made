use std::{cell::Cell, path::Path, rc::Rc};

use fs::FakeFs;
use git::{
    repository::repo_path,
    status::{StageStatus, StatusCode},
};
use gpui::{AppContext as _, Empty, TestAppContext};
use project::Project;
use serde_json::json;
use settings::SettingsStore;
use util::path;
use workspace::{
    collaborative_review::{CollaborativeReviewHost, CollaborativeReviewSlot},
    collaborative_review_actions::{
        CollaborativeReviewAction, CollaborativeReviewActionContext,
        CollaborativeReviewActionError, CollaborativeReviewActionRequest,
        CollaborativeReviewActionState, route_collaborative_review_action,
    },
    collaborative_review_summary::CollaborativeReviewSummarySource,
};

#[gpui::test]
async fn collaborative_review(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        cx.set_global(db::AppDatabase::test_new());
        theme_settings::init(theme::LoadThemes::JustBase, cx);
    });

    let file_system = FakeFs::new(cx.executor());
    file_system
        .insert_tree(
            path!("/project"),
            json!({
                ".git": {},
                "src": {
                    "lib.rs": "pub fn value() -> u32 { 2 }\n",
                    "main.rs": "fn main() {}\n",
                },
            }),
        )
        .await;
    let dot_git = Path::new(path!("/project/.git"));
    file_system.set_head_and_index_for_repo(
        dot_git,
        &[
            ("src/lib.rs", "pub fn value() -> u32 { 1 }\n".into()),
            ("src/main.rs", "fn main() {}\n".into()),
        ],
    );

    let project = Project::test(file_system.clone(), [Path::new(path!("/project"))], cx).await;
    project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    cx.run_until_parked();

    let repository = project.read_with(cx, |project, cx| {
        let repository = project
            .repositories(cx)
            .values()
            .next()
            .cloned()
            .expect("the repository fixture should be discovered");
        repository
    });
    assert_eq!(
        repository.read_with(cx, |repository, _| {
            repository
                .status_for_path(&repo_path("src/lib.rs"))
                .expect("the changed file should have Git status")
                .status
                .staging()
        }),
        StageStatus::Unstaged
    );

    let review_view = cx.new(|_| Empty);
    let mut host = CollaborativeReviewHost::new(project.clone());
    host.register(
        CollaborativeReviewSlot::ProjectChanges,
        &project,
        review_view.clone().into(),
    )
    .expect("the canonical project review provider should register");
    assert_eq!(
        host.visible_view(true)
            .expect("expanded review should expose the provider")
            .entity_id(),
        review_view.entity_id()
    );
    assert!(host.visible_view(false).is_none());
    assert_eq!(
        host.selected_view()
            .expect("collapse should retain the selected provider")
            .entity_id(),
        review_view.entity_id()
    );
    assert_eq!(
        host.visible_view(true)
            .expect("restoring review should recover the same provider")
            .entity_id(),
        review_view.entity_id()
    );

    let source = CollaborativeReviewSummarySource::new(
        CollaborativeReviewSlot::ProjectChanges,
        review_view.entity_id(),
        1,
    );
    let action_context = CollaborativeReviewActionContext::new(
        source,
        CollaborativeReviewActionState::Ready,
        [CollaborativeReviewAction::Stage],
    );
    route_collaborative_review_action(
        &action_context,
        CollaborativeReviewActionRequest::new(source, CollaborativeReviewAction::Stage),
        |_| {
            file_system.set_index_for_repo(
                dot_git,
                &[
                    ("src/lib.rs", "pub fn value() -> u32 { 2 }\n".into()),
                    ("src/main.rs", "fn main() {}\n".into()),
                ],
            );
            Ok(())
        },
    )
    .expect("a current stage action should invoke the native Git update");
    project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    cx.run_until_parked();
    assert_eq!(
        repository.read_with(cx, |repository, _| {
            repository
                .status_for_path(&repo_path("src/lib.rs"))
                .expect("the staged file should retain Git status")
                .status
        }),
        StatusCode::Modified.index()
    );

    let next_source = CollaborativeReviewSummarySource::new(
        CollaborativeReviewSlot::ProjectChanges,
        review_view.entity_id(),
        2,
    );
    let stale_action_invoked = Rc::new(Cell::new(false));
    let stale_action_flag = stale_action_invoked.clone();
    assert_eq!(
        route_collaborative_review_action(
            &CollaborativeReviewActionContext::new(
                next_source,
                CollaborativeReviewActionState::Ready,
                [CollaborativeReviewAction::Stage],
            ),
            CollaborativeReviewActionRequest::new(source, CollaborativeReviewAction::Stage),
            move |_| {
                stale_action_flag.set(true);
                Ok(())
            },
        ),
        Err(CollaborativeReviewActionError::StaleRevision)
    );
    assert!(!stale_action_invoked.get());
}
