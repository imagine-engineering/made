use collaboration_domain::RepositoryCoordinate;
use fs::FakeFs;
use gpui::TestAppContext;
use project::{Project, git_store::RepositoryId};
use serde_json::json;
use util::path;

use crate::init_test;

#[gpui::test]
async fn collaboration_repository_identity_survives_reopen_and_remote_change_and_rejects_missing_repository(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/root"),
        json!({
            "repository": {
                ".git": {},
                "main.rs": "fn main() {}",
            },
        }),
    )
    .await;
    let git_directory = path!("/root/repository/.git");
    fs.set_remote_for_repo(
        git_directory.as_ref(),
        "origin",
        "git@example.com:owner/original.git",
    );

    let first_project = Project::test(fs.clone(), [path!("/root/repository").as_ref()], cx).await;
    first_project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    let (first_repository_id, first_identity, first_remote) =
        first_project.read_with(cx, |project, cx| {
            let (repository_id, repository) = project.repositories(cx).iter().next().unwrap();
            (
                *repository_id,
                project
                    .collaboration_repository_identity(*repository_id, cx)
                    .unwrap(),
                repository.read(cx).remote_origin_url.clone(),
            )
        });
    assert_eq!(first_identity.path(), path!("/root/repository"));
    assert_eq!(
        first_remote.as_deref(),
        Some("git@example.com:owner/original.git")
    );

    drop(first_project);
    fs.set_remote_for_repo(
        git_directory.as_ref(),
        "origin",
        "git@example.com:other/moved.git",
    );
    let reopened_project =
        Project::test(fs.clone(), [path!("/root/repository").as_ref()], cx).await;
    reopened_project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;

    reopened_project.read_with(cx, |project, cx| {
        let (reopened_repository_id, repository) = project.repositories(cx).iter().next().unwrap();
        let reopened_identity = project
            .collaboration_repository_identity(*reopened_repository_id, cx)
            .unwrap();
        assert_eq!(reopened_identity, first_identity);
        assert_eq!(
            repository.read(cx).remote_origin_url.as_deref(),
            Some("git@example.com:other/moved.git")
        );
        let hosted_coordinate = RepositoryCoordinate::parse(
            &format!("30617:{}:owner/repository", "11".repeat(32)),
            Some("wss://relay.example.com".to_owned()),
        )
        .unwrap();
        let binding = project
            .collaboration_repository_binding(
                *reopened_repository_id,
                hosted_coordinate.clone(),
                cx,
            )
            .unwrap();
        assert_eq!(binding.repository_identity(), &first_identity);
        assert_eq!(binding.hosted_coordinate(), &hosted_coordinate);
        assert_eq!(
            project.collaboration_repository_identity(
                RepositoryId::from_proto(first_repository_id.to_proto() + 10_000),
                cx,
            ),
            Err(
                project::collaboration_repository::CollaborationRepositoryError::RepositoryNotFound(
                    RepositoryId::from_proto(first_repository_id.to_proto() + 10_000),
                )
            )
        );
    });
}
