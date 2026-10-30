use collaboration_domain::{
    MessageSource, NostrEventId, NostrPublicKey, ProjectGroup, RepositoryCoordinate,
};
use fs::{FakeFs, Fs};
use gpui::TestAppContext;
use project::{
    Project,
    collaboration_navigation::{
        CollaborationChannelNavigationTarget, CollaborationLocalRepositoryNavigationTarget,
        CollaborationProjectNavigationLifecycle, CollaborationProjectNavigationTarget,
        CollaborationRepositoryNavigationTarget, CollaborationWorktreeNavigationTarget,
    },
};
use serde_json::json;
use settings::SettingsStore;
use util::path;

macro_rules! assert_not_impl {
    ($type:ty: $trait:path) => {{
        trait AmbiguousIfImpl<A> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
        struct Invalid;
        impl<T: ?Sized + $trait> AmbiguousIfImpl<Invalid> for T {}
        let _ = <$type as AmbiguousIfImpl<_>>::check;
    }};
}

fn init_test(cx: &mut TestAppContext) {
    zlog::init_test();
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
    });
}

fn coordinate(owner: u8) -> RepositoryCoordinate {
    RepositoryCoordinate::parse(
        &format!(
            "30617:{}:owner/repository",
            format!("{owner:02x}").repeat(32)
        ),
        Some("wss://external.example".into()),
    )
    .unwrap()
}

fn signed_group(repository: RepositoryCoordinate) -> ProjectGroup {
    ProjectGroup::from_signed_metadata(
        NostrPublicKey::from_bytes([0xaa; 32]),
        MessageSource {
            event_id: NostrEventId::from_bytes([0x11; 32]),
            event_created_at: 100,
        },
        "stranger-group",
        None,
        None,
        [repository],
        None,
        Some("listed"),
    )
    .unwrap()
}

#[test]
fn project_group_permissions_expose_no_git_filesystem_or_external_host_capability() {
    assert_not_impl!(CollaborationProjectNavigationTarget: git::repository::GitRepository);
    assert_not_impl!(CollaborationRepositoryNavigationTarget: git::repository::GitRepository);
    assert_not_impl!(CollaborationLocalRepositoryNavigationTarget: git::repository::GitRepository);
    assert_not_impl!(CollaborationWorktreeNavigationTarget: git::repository::GitRepository);
    assert_not_impl!(CollaborationChannelNavigationTarget: git::repository::GitRepository);

    assert_not_impl!(CollaborationProjectNavigationTarget: fs::Fs);
    assert_not_impl!(CollaborationRepositoryNavigationTarget: fs::Fs);
    assert_not_impl!(CollaborationLocalRepositoryNavigationTarget: fs::Fs);
    assert_not_impl!(CollaborationWorktreeNavigationTarget: fs::Fs);
    assert_not_impl!(CollaborationChannelNavigationTarget: fs::Fs);

    assert_not_impl!(CollaborationProjectNavigationTarget: http_client::HttpClient);
    assert_not_impl!(CollaborationRepositoryNavigationTarget: http_client::HttpClient);
    assert_not_impl!(CollaborationLocalRepositoryNavigationTarget: http_client::HttpClient);
    assert_not_impl!(CollaborationWorktreeNavigationTarget: http_client::HttpClient);
    assert_not_impl!(CollaborationChannelNavigationTarget: http_client::HttpClient);
}

#[gpui::test]
async fn project_group_permissions_cross_owner_resolution_has_no_git_or_filesystem_side_effects(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/root"),
        json!({
            "repository": {
                ".git": {},
                "protected.txt": "repository owner content",
            },
        }),
    )
    .await;
    let git_directory = path!("/root/repository/.git");
    let remote = "git@external.example:owner/repository.git";
    fs.set_remote_for_repo(git_directory.as_ref(), "origin", remote);
    let protected_path = path!("/root/repository/protected.txt");
    let before = fs.load(protected_path.as_ref()).await.unwrap();

    let project = Project::test(fs.clone(), [path!("/root/repository").as_ref()], cx).await;
    project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;
    let repository = coordinate(0xbb);
    let target = project.read_with(cx, |project, cx| {
        let repository_id = *project.repositories(cx).keys().next().unwrap();
        let binding = project
            .collaboration_repository_binding(repository_id, repository.clone(), cx)
            .unwrap();
        project
            .resolve_collaboration_navigation(
                &signed_group(repository),
                CollaborationProjectNavigationLifecycle::Active,
                &[binding],
                None,
                cx,
            )
            .unwrap()
    });

    assert_ne!(
        target.project_identity().signer_public_key(),
        target.repositories()[0]
            .hosted_coordinate()
            .owner_public_key()
    );
    assert_eq!(fs.load(protected_path.as_ref()).await.unwrap(), before);
    project.read_with(cx, |project, cx| {
        let repository = project.repositories(cx).values().next().unwrap().read(cx);
        assert_eq!(repository.remote_origin_url.as_deref(), Some(remote));
    });
}
