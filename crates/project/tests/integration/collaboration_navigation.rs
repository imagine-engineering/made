use collaboration_domain::{
    AggregateId, ChannelLifecycleState, CommunityId, MessageSource, NostrEventId, NostrPublicKey,
    ProjectChannelReference, ProjectGroup, RepositoryCoordinate,
};
use fs::FakeFs;
use gpui::TestAppContext;
use project::{
    Project,
    collaboration_navigation::{
        CollaborationChannelNavigationBinding, CollaborationNavigationResolutionError,
        CollaborationProjectNavigationLifecycle,
    },
};
use serde_json::json;
use std::path::{Path, PathBuf};
use util::path;

use crate::init_test;

fn repository_coordinate(owner: u8, discriminator: &str) -> RepositoryCoordinate {
    RepositoryCoordinate::parse(
        &format!(
            "30617:{}:{discriminator}",
            format!("{owner:02x}").repeat(32)
        ),
        None,
    )
    .unwrap()
}

fn project_group(
    repository: RepositoryCoordinate,
    channel_reference: Option<String>,
) -> ProjectGroup {
    ProjectGroup::from_signed_metadata(
        NostrPublicKey::from_bytes([0xaa; 32]),
        MessageSource {
            event_id: NostrEventId::from_bytes([0x11; 32]),
            event_created_at: 100,
        },
        "platform",
        Some("Platform".into()),
        None,
        [repository],
        channel_reference,
        Some("listed"),
    )
    .unwrap()
}

#[gpui::test]
async fn collaboration_navigation_marks_a_missing_local_clone_unavailable(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/root"),
        json!({
            "bound": { ".git": {}, "bound.rs": "" },
            "open": { ".git": {}, "open.rs": "" },
        }),
    )
    .await;

    let bound_project = Project::test(fs.clone(), [path!("/root/bound").as_ref()], cx).await;
    bound_project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    let coordinate = repository_coordinate(0xbb, "owner/bound");
    let binding = bound_project.read_with(cx, |project, cx| {
        let repository_id = *project.repositories(cx).keys().next().unwrap();
        project
            .collaboration_repository_binding(repository_id, coordinate.clone(), cx)
            .unwrap()
    });

    let open_project = Project::test(fs, [path!("/root/open").as_ref()], cx).await;
    open_project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    let target = open_project.read_with(cx, |project, cx| {
        project
            .resolve_collaboration_navigation(
                &project_group(coordinate, None),
                CollaborationProjectNavigationLifecycle::Active,
                &[binding],
                None,
                cx,
            )
            .unwrap()
    });

    assert_eq!(target.repositories().len(), 1);
    assert!(!target.repositories()[0].is_available());
    assert!(target.repositories()[0].local_targets().is_empty());
    assert_eq!(
        target
            .native_project()
            .path_list()
            .ordered_paths()
            .collect::<Vec<_>>(),
        vec![path!("/root/open")]
    );
}

#[gpui::test]
async fn collaboration_navigation_resolves_every_open_linked_worktree_and_active_channel(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/root"),
        json!({
            "repository": {
                ".git": {},
                "main.rs": "",
            },
        }),
    )
    .await;
    let linked_path = PathBuf::from(path!("/root/worktrees/feature"));
    fs.add_linked_worktree_for_repo(
        Path::new(path!("/root/repository/.git")),
        false,
        git::repository::Worktree {
            path: linked_path.clone(),
            ref_name: Some("refs/heads/feature".into()),
            sha: "abc123".into(),
            is_main: false,
            is_bare: false,
        },
    )
    .await;

    let project = Project::test(
        fs,
        [path!("/root/repository").as_ref(), linked_path.as_path()],
        cx,
    )
    .await;
    project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    let coordinate = repository_coordinate(0xbb, "owner/repository");
    let community_id = CommunityId::new();
    let channel_id = AggregateId::new();
    let reference = ProjectChannelReference::new(channel_id.to_string()).unwrap();
    let channel_binding = CollaborationChannelNavigationBinding::new(
        reference,
        community_id,
        channel_id,
        ChannelLifecycleState::Active,
    );
    let target = project.read_with(cx, |project, cx| {
        let repository_id = *project.repositories(cx).keys().next().unwrap();
        let binding = project
            .collaboration_repository_binding(repository_id, coordinate.clone(), cx)
            .unwrap();
        project
            .resolve_collaboration_navigation(
                &project_group(coordinate, Some(channel_id.to_string())),
                CollaborationProjectNavigationLifecycle::Active,
                &[binding],
                Some(&channel_binding),
                cx,
            )
            .unwrap()
    });

    let repository = &target.repositories()[0];
    assert!(repository.is_available());
    assert_eq!(repository.local_targets().len(), 2);
    assert_eq!(
        repository
            .local_targets()
            .iter()
            .map(|target| target.work_directory().clone())
            .collect::<Vec<_>>(),
        vec![PathBuf::from(path!("/root/repository")), linked_path]
    );
    assert!(
        repository
            .local_targets()
            .iter()
            .all(|target| target.worktrees().len() == 1)
    );
    let channel = target.channel().unwrap();
    assert_eq!(channel.community_id(), community_id);
    assert_eq!(channel.channel_id(), channel_id);
    assert_eq!(
        target
            .native_project()
            .path_list()
            .ordered_paths()
            .collect::<Vec<_>>(),
        vec![path!("/root/repository"), path!("/root/repository")]
    );
}

#[gpui::test]
async fn collaboration_navigation_rejects_an_archived_project_group(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/root"), json!({ "project": {} }))
        .await;
    let project = Project::test(fs, [path!("/root/project").as_ref()], cx).await;
    let group = project_group(repository_coordinate(0xbb, "owner/repository"), None);

    assert_eq!(
        project.read_with(cx, |project, cx| {
            project.resolve_collaboration_navigation(
                &group,
                CollaborationProjectNavigationLifecycle::Archived,
                &[],
                None,
                cx,
            )
        }),
        Err(CollaborationNavigationResolutionError::ArchivedProjectGroup)
    );
}
