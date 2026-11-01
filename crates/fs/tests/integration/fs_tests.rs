mod fake_git_repo_tests;

use std::{
    collections::BTreeSet,
    io::{Read, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use futures::{FutureExt, StreamExt};

use fs::*;
use gpui::{BackgroundExecutor, TestAppContext};
use serde_json::json;
use tempfile::TempDir;
use util::path;

#[gpui::test]
async fn real_fs_confined_file_rejects_replaced_paths(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    // Windows metadata lookup waits for OS file I/O outside GPUI's test scheduler.
    cx.executor().allow_parking();

    let temporary = tempfile::tempdir().expect("temporary worktree");
    let root = temporary.path().join("root");
    let report_path = Path::new("module/target/surefire-reports/TEST-Example.xml");
    let report = root.join(report_path);
    std::fs::create_dir_all(report.parent().expect("report directory"))
        .expect("create report directory");
    std::fs::write(&report, b"<testsuite/>").expect("write report");
    let fs = RealFs::new(None, executor);
    let root_inode = fs
        .metadata(&root)
        .await
        .expect("read root metadata")
        .expect("root exists")
        .inode;
    let file_inode = fs
        .metadata(&report)
        .await
        .expect("read report metadata")
        .expect("report exists")
        .inode;
    let mut reader = fs
        .open_sync_confined_file(&root, report_path, root_inode, file_inode)
        .await
        .expect("open current report");
    let mut content = String::new();
    reader.read_to_string(&mut content).expect("read report");
    assert_eq!(content, "<testsuite/>");
    drop(reader);

    assert!(
        fs.open_sync_confined_file(&root, Path::new("../outside.xml"), root_inode, file_inode)
            .await
            .is_err()
    );
    assert!(
        fs.open_sync_confined_file(&root, &report, root_inode, file_inode)
            .await
            .is_err()
    );

    #[cfg(unix)]
    {
        let outside = temporary.path().join("outside");
        std::fs::create_dir(&outside).expect("create outside directory");
        let outside_report = outside.join("TEST-Escape.xml");
        std::fs::write(&outside_report, b"<testsuite name=\"outside\"/>")
            .expect("write outside report");
        std::os::unix::fs::symlink(&outside, root.join("linked")).expect("link outside directory");
        let outside_inode = fs
            .metadata(&outside_report)
            .await
            .expect("read outside metadata")
            .expect("outside report exists")
            .inode;
        assert!(
            fs.open_sync_confined_file(
                &root,
                Path::new("linked/TEST-Escape.xml"),
                root_inode,
                outside_inode,
            )
            .await
            .is_err()
        );
    }

    #[cfg(windows)]
    {
        let outside = temporary.path().join("outside");
        std::fs::create_dir(&outside).expect("create outside directory");
        let outside_report = outside.join("TEST-Escape.xml");
        std::fs::write(&outside_report, b"<testsuite name=\"outside\"/>")
            .expect("write outside report");
        let junction = root.join("linked");
        let output = std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(&junction)
            .arg(&outside)
            .output()
            .expect("create parent junction");
        assert!(
            output.status.success(),
            "parent junction creation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let outside_inode = fs
            .metadata(&outside_report)
            .await
            .expect("read outside metadata")
            .expect("outside report exists")
            .inode;
        assert!(
            fs.open_sync_confined_file(
                &root,
                Path::new("linked/TEST-Escape.xml"),
                root_inode,
                outside_inode,
            )
            .await
            .is_err()
        );
        std::fs::remove_dir(&junction).expect("remove parent junction");
    }

    std::fs::rename(&report, report.with_extension("old")).expect("move original report");
    std::fs::write(&report, b"<testsuite name=\"replacement\"/>").expect("replace report");
    assert!(
        fs.open_sync_confined_file(&root, report_path, root_inode, file_inode)
            .await
            .is_err()
    );

    let replaced_root = temporary.path().join("moved-root");
    std::fs::rename(&root, &replaced_root).expect("move original root");
    std::fs::create_dir_all(report.parent().expect("replacement report directory"))
        .expect("create replacement root");
    std::fs::write(&report, b"<testsuite name=\"new-root\"/>")
        .expect("write replacement root report");
    let new_file_inode = fs
        .metadata(&report)
        .await
        .expect("read replacement metadata")
        .expect("replacement report exists")
        .inode;
    assert!(
        fs.open_sync_confined_file(&root, report_path, root_inode, new_file_inode)
            .await
            .is_err()
    );
}

#[gpui::test]
async fn test_fake_fs(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/root"),
        json!({
            "dir1": {
                "a": "A",
                "b": "B"
            },
            "dir2": {
                "c": "C",
                "dir3": {
                    "d": "D"
                }
            }
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/dir1/a")),
            PathBuf::from(path!("/root/dir1/b")),
            PathBuf::from(path!("/root/dir2/c")),
            PathBuf::from(path!("/root/dir2/dir3/d")),
        ]
    );

    fs.create_symlink(path!("/root/dir2/link-to-dir3").as_ref(), "./dir3".into())
        .await
        .unwrap();

    assert_eq!(
        fs.canonicalize(path!("/root/dir2/link-to-dir3").as_ref())
            .await
            .unwrap(),
        PathBuf::from(path!("/root/dir2/dir3")),
    );
    assert_eq!(
        fs.canonicalize(path!("/root/dir2/link-to-dir3/d").as_ref())
            .await
            .unwrap(),
        PathBuf::from(path!("/root/dir2/dir3/d")),
    );
    assert_eq!(
        fs.load(path!("/root/dir2/link-to-dir3/d").as_ref())
            .await
            .unwrap(),
        "D",
    );
}

#[gpui::test]
async fn fake_fs_ordinary_alias_cycles_return_errors(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    let root = Path::new(path!("/root"));
    fs.insert_tree(root, json!({"sentinel": "ordinary content"}))
        .await;
    for (name, target) in [
        ("self", PathBuf::from("self")),
        ("pair-a", PathBuf::from("pair-b")),
        ("pair-b", PathBuf::from("pair-a")),
        ("expanding", PathBuf::from("expanding/suffix")),
        ("absolute", root.join("absolute")),
    ] {
        fs.create_symlink(&root.join(name), target)
            .await
            .expect("create cyclic alias");
    }
    for name in ["self", "pair-a", "pair-b", "expanding", "absolute"] {
        let path = root.join(name);
        let errors = [
            fs.canonicalize(&path)
                .await
                .expect_err("canonicalize must reject cycle"),
            fs.load(&path).await.expect_err("load must reject cycle"),
            fs.load_bytes(&path)
                .await
                .expect_err("load_bytes must reject cycle"),
            fs.metadata(&path)
                .await
                .expect_err("metadata must reject cycle"),
        ];
        for error in errors {
            assert!(
                format!("{error:#}").contains("symlink cycle"),
                "{path:?}: {error:#}"
            );
            assert!(
                error
                    .chain()
                    .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
                    .all(|error| error.kind() != std::io::ErrorKind::NotFound)
            );
        }
        assert!(!fs.is_file(&path).await);
        assert!(!fs.is_dir(&path).await);
        assert_eq!(
            fs.load(&root.join("sentinel"))
                .await
                .expect("same FakeFs remains usable"),
            "ordinary content"
        );
    }
}

#[gpui::test]
async fn fake_fs_ordinary_alias_read_link_does_not_follow_final_alias(
    executor: BackgroundExecutor,
) {
    let fs = FakeFs::new(executor);
    let root = Path::new(path!("/root"));
    fs.insert_tree(root, json!({"sentinel": "ordinary content"}))
        .await;
    fs.create_symlink(&root.join("self"), PathBuf::from("self"))
        .await
        .expect("create final alias");
    assert_eq!(
        fs.read_link(&root.join("self"))
            .await
            .expect("final alias is not followed"),
        PathBuf::from("self")
    );
    let error = fs
        .read_link(&root.join("self/child"))
        .await
        .expect_err("cyclic ancestor must be followed and rejected");
    assert!(format!("{error:#}").contains("symlink cycle"));
    assert_eq!(
        fs.load(&root.join("sentinel"))
            .await
            .expect("resolver releases state"),
        "ordinary content"
    );
}

#[gpui::test]
async fn fake_fs_ordinary_alias_chains_have_no_strict_follow_limit(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    let root = Path::new(path!("/root"));
    fs.insert_tree(root, json!({"sentinel": "ordinary content"}))
        .await;
    for length in [41, 128] {
        for index in 0..length {
            let name = format!("chain-{length}-{index}");
            let target = if index + 1 == length {
                PathBuf::from("sentinel")
            } else {
                PathBuf::from(format!("chain-{length}-{}", index + 1))
            };
            fs.create_symlink(&root.join(name), target)
                .await
                .expect("create finite chain");
        }
        let path = root.join(format!("chain-{length}-0"));
        assert_eq!(
            fs.canonicalize(&path)
                .await
                .expect("finite chain canonicalizes"),
            root.join("sentinel")
        );
        assert_eq!(
            fs.load(&path).await.expect("finite chain loads"),
            "ordinary content"
        );
        assert_eq!(
            fs.load_bytes(&path)
                .await
                .expect("finite chain loads bytes"),
            b"ordinary content"
        );
        assert!(fs.is_file(&path).await);
    }
}

#[gpui::test]
async fn fake_fs_ordinary_alias_targets_can_be_reused_after_completion(
    executor: BackgroundExecutor,
) {
    let fs = FakeFs::new(executor);
    let root = Path::new(path!("/root"));
    fs.insert_tree(root, json!({"sentinel": "ordinary content"}))
        .await;
    fs.create_symlink(&root.join("self"), PathBuf::from("."))
        .await
        .expect("create current-directory alias");
    fs.create_symlink(&root.join("nested"), PathBuf::from("self/self"))
        .await
        .expect("create nested completed-target reuse");
    fs.create_symlink(&root.join("absolute"), root.to_path_buf())
        .await
        .expect("create absolute directory alias");
    let mut repeated = root.to_path_buf();
    for _ in 0..128 {
        repeated.push("self");
    }
    repeated.push("sentinel");
    for path in [
        repeated,
        root.join("nested/nested/sentinel"),
        root.join("absolute/self/absolute/sentinel"),
    ] {
        assert_eq!(
            fs.canonicalize(&path)
                .await
                .expect("completed alias may be reused"),
            root.join("sentinel")
        );
        assert_eq!(
            fs.load(&path)
                .await
                .expect("completed aliases preserve content"),
            "ordinary content"
        );
        assert_eq!(
            fs.load_bytes(&path)
                .await
                .expect("completed aliases preserve bytes"),
            b"ordinary content"
        );
    }
}

#[gpui::test]
async fn fake_fs_ordinary_alias_missing_outcomes_remain_api_specific(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    let root = Path::new(path!("/root"));
    fs.insert_tree(root, json!({"sentinel": "ordinary content"}))
        .await;
    fs.create_symlink(&root.join("broken"), PathBuf::from("missing-target"))
        .await
        .expect("create missing target alias");
    for path in [
        root.join("missing"),
        root.join("missing-parent/child"),
        root.join("broken"),
        root.join("sentinel/child"),
    ] {
        for error in [
            fs.load(&path).await.expect_err("missing load fails"),
            fs.load_bytes(&path)
                .await
                .expect_err("missing byte load fails"),
        ] {
            assert_eq!(
                error
                    .downcast_ref::<std::io::Error>()
                    .expect("typed missing error")
                    .kind(),
                std::io::ErrorKind::NotFound
            );
        }
        assert!(
            fs.metadata(&path)
                .await
                .expect("missing metadata is not an error")
                .is_none()
        );
        let error = fs
            .canonicalize(&path)
            .await
            .expect_err("missing canonicalize fails");
        assert!(format!("{error:#}").contains("path does not exist"));
        assert!(!fs.is_file(&path).await);
        assert!(!fs.is_dir(&path).await);
    }
    assert_eq!(
        fs.read_link(&root.join("broken"))
            .await
            .expect("broken final link remains readable"),
        PathBuf::from("missing-target")
    );
    let error = fs
        .read_link(&root.join("missing"))
        .await
        .expect_err("missing final link fails");
    assert!(format!("{error:#}").contains("path does not exist"));
    assert_eq!(
        fs.load(&root.join("sentinel"))
            .await
            .expect("ordinary control after missing paths"),
        "ordinary content"
    );
}

#[gpui::test]
async fn test_copy_recursive_with_single_file(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/outer"),
        json!({
            "a": "A",
            "b": "B",
            "inner": {}
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/b")),
        ]
    );

    let source = Path::new(path!("/outer/a"));
    let target = Path::new(path!("/outer/a copy"));
    copy_recursive(fs.as_ref(), source, target, Default::default())
        .await
        .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/a copy")),
            PathBuf::from(path!("/outer/b")),
        ]
    );

    let source = Path::new(path!("/outer/a"));
    let target = Path::new(path!("/outer/inner/a copy"));
    copy_recursive(fs.as_ref(), source, target, Default::default())
        .await
        .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/a copy")),
            PathBuf::from(path!("/outer/b")),
            PathBuf::from(path!("/outer/inner/a copy")),
        ]
    );
}

#[gpui::test]
async fn test_copy_recursive_with_single_dir(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/outer"),
        json!({
            "a": "A",
            "empty": {},
            "non-empty": {
                "b": "B",
            }
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/non-empty/b")),
        ]
    );
    assert_eq!(
        fs.directories(false),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/outer")),
            PathBuf::from(path!("/outer/empty")),
            PathBuf::from(path!("/outer/non-empty")),
        ]
    );

    let source = Path::new(path!("/outer/empty"));
    let target = Path::new(path!("/outer/empty copy"));
    copy_recursive(fs.as_ref(), source, target, Default::default())
        .await
        .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/non-empty/b")),
        ]
    );
    assert_eq!(
        fs.directories(false),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/outer")),
            PathBuf::from(path!("/outer/empty")),
            PathBuf::from(path!("/outer/empty copy")),
            PathBuf::from(path!("/outer/non-empty")),
        ]
    );

    let source = Path::new(path!("/outer/non-empty"));
    let target = Path::new(path!("/outer/non-empty copy"));
    copy_recursive(fs.as_ref(), source, target, Default::default())
        .await
        .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/a")),
            PathBuf::from(path!("/outer/non-empty/b")),
            PathBuf::from(path!("/outer/non-empty copy/b")),
        ]
    );
    assert_eq!(
        fs.directories(false),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/outer")),
            PathBuf::from(path!("/outer/empty")),
            PathBuf::from(path!("/outer/empty copy")),
            PathBuf::from(path!("/outer/non-empty")),
            PathBuf::from(path!("/outer/non-empty copy")),
        ]
    );
}

#[gpui::test]
async fn test_copy_recursive(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/outer"),
        json!({
            "inner1": {
                "a": "A",
                "b": "B",
                "inner3": {
                    "d": "D",
                },
                "inner4": {}
            },
            "inner2": {
                "c": "C",
            }
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/inner3/d")),
        ]
    );
    assert_eq!(
        fs.directories(false),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/outer")),
            PathBuf::from(path!("/outer/inner1")),
            PathBuf::from(path!("/outer/inner2")),
            PathBuf::from(path!("/outer/inner1/inner3")),
            PathBuf::from(path!("/outer/inner1/inner4")),
        ]
    );

    let source = Path::new(path!("/outer"));
    let target = Path::new(path!("/outer/inner1/outer"));
    copy_recursive(fs.as_ref(), source, target, Default::default())
        .await
        .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/inner3/d")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner1/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/inner3/d")),
        ]
    );
    assert_eq!(
        fs.directories(false),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/outer")),
            PathBuf::from(path!("/outer/inner1")),
            PathBuf::from(path!("/outer/inner2")),
            PathBuf::from(path!("/outer/inner1/inner3")),
            PathBuf::from(path!("/outer/inner1/inner4")),
            PathBuf::from(path!("/outer/inner1/outer")),
            PathBuf::from(path!("/outer/inner1/outer/inner1")),
            PathBuf::from(path!("/outer/inner1/outer/inner2")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/inner3")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/inner4")),
        ]
    );
}

#[gpui::test]
async fn test_copy_recursive_with_overwriting(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/outer"),
        json!({
            "inner1": {
                "a": "A",
                "b": "B",
                "outer": {
                    "inner1": {
                        "a": "B"
                    }
                }
            },
            "inner2": {
                "c": "C",
            }
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/a")),
        ]
    );
    assert_eq!(
        fs.load(path!("/outer/inner1/outer/inner1/a").as_ref())
            .await
            .unwrap(),
        "B",
    );

    let source = Path::new(path!("/outer"));
    let target = Path::new(path!("/outer/inner1/outer"));
    copy_recursive(
        fs.as_ref(),
        source,
        target,
        CopyOptions {
            overwrite: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner1/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/outer/inner1/a")),
        ]
    );
    assert_eq!(
        fs.load(path!("/outer/inner1/outer/inner1/a").as_ref())
            .await
            .unwrap(),
        "A"
    );
}

#[gpui::test]
async fn test_copy_recursive_with_ignoring(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/outer"),
        json!({
            "inner1": {
                "a": "A",
                "b": "B",
                "outer": {
                    "inner1": {
                        "a": "B"
                    }
                }
            },
            "inner2": {
                "c": "C",
            }
        }),
    )
    .await;

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/a")),
        ]
    );
    assert_eq!(
        fs.load(path!("/outer/inner1/outer/inner1/a").as_ref())
            .await
            .unwrap(),
        "B",
    );

    let source = Path::new(path!("/outer"));
    let target = Path::new(path!("/outer/inner1/outer"));
    copy_recursive(
        fs.as_ref(),
        source,
        target,
        CopyOptions {
            ignore_if_exists: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/a")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/b")),
            PathBuf::from(path!("/outer/inner1/outer/inner2/c")),
            PathBuf::from(path!("/outer/inner1/outer/inner1/outer/inner1/a")),
        ]
    );
    assert_eq!(
        fs.load(path!("/outer/inner1/outer/inner1/a").as_ref())
            .await
            .unwrap(),
        "B"
    );
}

#[gpui::test]
async fn test_realfs_atomic_write(executor: BackgroundExecutor) {
    // With the file handle still open, the file should be replaced
    // https://github.com/imagine-engineering/made/issues/30054
    let fs = RealFs::new(None, executor);
    let temp_dir = TempDir::new().unwrap();
    let file_to_be_replaced = temp_dir.path().join("file.txt");
    let mut file = std::fs::File::create_new(&file_to_be_replaced).unwrap();
    file.write_all(b"Hello").unwrap();
    // drop(file);  // We still hold the file handle here
    let content = std::fs::read_to_string(&file_to_be_replaced).unwrap();
    assert_eq!(content, "Hello");
    gpui::block_on(fs.atomic_write(file_to_be_replaced.clone(), "World".into())).unwrap();
    let content = std::fs::read_to_string(&file_to_be_replaced).unwrap();
    assert_eq!(content, "World");
}

#[gpui::test]
async fn test_realfs_atomic_write_non_existing_file(executor: BackgroundExecutor) {
    let fs = RealFs::new(None, executor);
    let temp_dir = TempDir::new().unwrap();
    let file_to_be_replaced = temp_dir.path().join("file.txt");
    gpui::block_on(fs.atomic_write(file_to_be_replaced.clone(), "Hello".into())).unwrap();
    let content = std::fs::read_to_string(&file_to_be_replaced).unwrap();
    assert_eq!(content, "Hello");
}

#[gpui::test]
#[cfg(target_os = "windows")]
async fn test_realfs_canonicalize(executor: BackgroundExecutor) {
    use util::paths::SanitizedPath;

    let fs = RealFs::new(None, executor);
    let temp_dir = TempDir::new().unwrap();
    let file = temp_dir.path().join("test (1).txt");
    let file = SanitizedPath::new(&file);
    std::fs::write(&file, "test").unwrap();

    let canonicalized = fs.canonicalize(file.as_path()).await;
    assert!(canonicalized.is_ok());
}

#[gpui::test]
async fn test_rename(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/root"),
        json!({
            "src": {
                "file_a.txt": "content a",
                "file_b.txt": "content b"
            }
        }),
    )
    .await;

    fs.rename(
        Path::new(path!("/root/src/file_a.txt")),
        Path::new(path!("/root/src/new/renamed_a.txt")),
        RenameOptions {
            create_parents: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Assert that the `file_a.txt` file was being renamed and moved to a
    // different directory that did not exist before.
    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/src/file_b.txt")),
            PathBuf::from(path!("/root/src/new/renamed_a.txt")),
        ]
    );

    let result = fs
        .rename(
            Path::new(path!("/root/src/file_b.txt")),
            Path::new(path!("/root/src/old/renamed_b.txt")),
            RenameOptions {
                create_parents: false,
                ..Default::default()
            },
        )
        .await;

    // Assert that the `file_b.txt` file was not renamed nor moved, as
    // `create_parents` was set to `false`.
    // different directory that did not exist before.
    assert!(result.is_err());
    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/src/file_b.txt")),
            PathBuf::from(path!("/root/src/new/renamed_a.txt")),
        ]
    );
}

#[gpui::test]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
async fn test_realfs_parallel_rename_without_overwrite_preserves_losing_source(
    executor: BackgroundExecutor,
) {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let source_a = root.join("dir_a/shared.txt");
    let source_b = root.join("dir_b/shared.txt");
    let target = root.join("shared.txt");

    std::fs::create_dir_all(source_a.parent().unwrap()).unwrap();
    std::fs::create_dir_all(source_b.parent().unwrap()).unwrap();
    std::fs::write(&source_a, "from a").unwrap();
    std::fs::write(&source_b, "from b").unwrap();

    let fs = RealFs::new(None, executor);
    let (first_result, second_result) = futures::future::join(
        fs.rename(&source_a, &target, RenameOptions::default()),
        fs.rename(&source_b, &target, RenameOptions::default()),
    )
    .await;

    assert_ne!(first_result.is_ok(), second_result.is_ok());
    assert!(target.exists());
    assert_eq!(source_a.exists() as u8 + source_b.exists() as u8, 1);
}

#[gpui::test]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
async fn test_realfs_rename_ignore_if_exists_leaves_source_and_target_unchanged(
    executor: BackgroundExecutor,
) {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let source = root.join("source.txt");
    let target = root.join("target.txt");

    std::fs::write(&source, "from source").unwrap();
    std::fs::write(&target, "from target").unwrap();

    let fs = RealFs::new(None, executor);
    let result = fs
        .rename(
            &source,
            &target,
            RenameOptions {
                ignore_if_exists: true,
                ..Default::default()
            },
        )
        .await;

    assert!(result.is_ok());

    assert_eq!(std::fs::read_to_string(&source).unwrap(), "from source");
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "from target");
}

#[gpui::test]
#[cfg(unix)]
async fn test_realfs_broken_symlink_metadata(executor: BackgroundExecutor) {
    let tempdir = TempDir::new().unwrap();
    let path = tempdir.path();
    let fs = RealFs::new(None, executor);
    let symlink_path = path.join("symlink");
    gpui::block_on(fs.create_symlink(&symlink_path, PathBuf::from("file_a.txt"))).unwrap();
    let metadata = fs
        .metadata(&symlink_path)
        .await
        .expect("metadata call succeeds")
        .expect("metadata returned");
    assert!(metadata.is_symlink);
    assert!(!metadata.is_dir);
    assert!(!metadata.is_fifo);
    assert!(!metadata.is_executable);
    // don't care about len or mtime on symlinks?
}

#[gpui::test]
#[cfg(unix)]
async fn test_realfs_symlink_loop_metadata(executor: BackgroundExecutor) {
    let tempdir = TempDir::new().unwrap();
    let path = tempdir.path();
    let fs = RealFs::new(None, executor);
    let symlink_path = path.join("symlink");
    gpui::block_on(fs.create_symlink(&symlink_path, PathBuf::from("symlink"))).unwrap();
    let metadata = fs
        .metadata(&symlink_path)
        .await
        .expect("metadata call succeeds")
        .expect("metadata returned");
    assert!(metadata.is_symlink);
    assert!(!metadata.is_dir);
    assert!(!metadata.is_fifo);
    assert!(!metadata.is_executable);
    // don't care about len or mtime on symlinks?
}

#[gpui::test]
async fn test_fake_fs_trash(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/root"),
        json!({
            "src": {
                "file_c.txt": "File C",
                "file_d.txt": "File D"
            },
            "file_a.txt": "File A",
            "file_b.txt": "File B",
        }),
    )
    .await;

    // Trashing a file.
    let path = path!("/root/file_a.txt").as_ref();
    fs.trash(path, Default::default())
        .await
        .expect("should be able to trash {path:?}");

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/file_b.txt")),
            PathBuf::from(path!("/root/src/file_c.txt")),
            PathBuf::from(path!("/root/src/file_d.txt"))
        ]
    );

    // Trashing a directory.
    let path = path!("/root/src").as_ref();
    fs.trash(
        path,
        RemoveOptions {
            recursive: true,
            ..Default::default()
        },
    )
    .await
    .expect("should be able to trash {path:?}");

    assert_eq!(fs.files(), vec![PathBuf::from(path!("/root/file_b.txt"))]);
}

#[gpui::test]
async fn test_fake_fs_restore(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(
        path!("/root"),
        json!({
            "src": {
                "file_a.txt": "File A",
                "file_b.txt": "File B",
            },
            "file_c.txt": "File C",
        }),
    )
    .await;

    // Attempt deleting a file, asserting that the filesystem no longer reports
    // it as part of its list of files, restore it and verify that the list of
    // files and trash has been updated accordingly.
    let path = path!("/root/src/file_a.txt").as_ref();
    let trashed_entry = fs.trash(path, Default::default()).await.unwrap();

    assert_eq!(
        fs.original_path_for_trash_id(trashed_entry),
        Some(path.to_path_buf())
    );

    fs.restore(trashed_entry).await.unwrap();
    assert_eq!(fs.original_path_for_trash_id(trashed_entry), None);

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/file_c.txt")),
            PathBuf::from(path!("/root/src/file_a.txt")),
            PathBuf::from(path!("/root/src/file_b.txt"))
        ]
    );

    // Deleting and restoring a directory should also remove all of its files
    // but create a single trashed entry, which should be removed after
    // restoration.
    let options = RemoveOptions {
        recursive: true,
        ..Default::default()
    };
    let path = path!("/root/src/").as_ref();
    let trashed_entry = fs.trash(path, options).await.unwrap();

    assert_eq!(fs.files(), vec![PathBuf::from(path!("/root/file_c.txt"))]);

    fs.restore(trashed_entry).await.unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/file_c.txt")),
            PathBuf::from(path!("/root/src/file_a.txt")),
            PathBuf::from(path!("/root/src/file_b.txt"))
        ]
    );

    // A collision error should be returned in case a file is being restored to
    // a path where a file already exists.
    let path = path!("/root/src/file_a.txt").as_ref();
    let trashed_entry = fs.trash(path, Default::default()).await.unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/file_c.txt")),
            PathBuf::from(path!("/root/src/file_b.txt"))
        ]
    );

    fs.write(path, "New File A".as_bytes()).await.unwrap();

    assert_eq!(
        fs.files(),
        vec![
            PathBuf::from(path!("/root/file_c.txt")),
            PathBuf::from(path!("/root/src/file_a.txt")),
            PathBuf::from(path!("/root/src/file_b.txt"))
        ]
    );

    let file_contents = fs.files_with_contents(path);
    assert!(fs.restore(trashed_entry).await.is_err());
    assert_eq!(
        file_contents,
        vec![(PathBuf::from(path), b"New File A".to_vec())]
    );

    // A collision error should be returned in case a directory is being
    // restored to a path where a directory already exists.
    let options = RemoveOptions {
        recursive: true,
        ..Default::default()
    };
    let path = path!("/root/src/").as_ref();
    let trashed_entry = fs.trash(path, options).await.unwrap();

    assert_eq!(fs.files(), vec![PathBuf::from(path!("/root/file_c.txt"))]);

    fs.create_dir(path).await.unwrap();

    assert_eq!(fs.files(), vec![PathBuf::from(path!("/root/file_c.txt"))]);

    let result = fs.restore(trashed_entry).await;
    assert!(result.is_err());

    assert_eq!(fs.files(), vec![PathBuf::from(path!("/root/file_c.txt"))]);
}

/// Create a directory symlink (`link` -> `target`) in a cross-platform way.
///
/// Returns `Err` when the platform cannot create symlinks (e.g. Windows without
/// the create-symlink privilege), so callers can skip a scenario gracefully
/// rather than failing the whole test.
fn make_dir_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_dir(target, link)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symlinks are not supported on this platform",
        ))
    }
}

/// Waits up to `timeout` for `events` to deliver something that covers the
/// written file: either a path event whose path satisfies `path_matches`, or a
/// `Rescan` (which tells the consumer to re-scan this watcher's whole tree, so
/// it would discover the file anyway). Returns `false` if nothing relevant
/// arrives before the timeout.
async fn watcher_delivered_event(
    events: &mut (impl futures::Stream<Item = Vec<PathEvent>> + Unpin),
    executor: &BackgroundExecutor,
    timeout: Duration,
    path_matches: &(dyn Fn(&Path) -> bool + Send + Sync),
) -> bool {
    let timeout = executor.timer(timeout).fuse();
    futures::pin_mut!(timeout);
    loop {
        futures::select_biased! {
            batch = events.next().fuse() => {
                let Some(batch) = batch else { return false };
                let covered = batch.iter().any(|event| {
                    path_matches(&event.path) || event.kind == Some(PathEventKind::Rescan)
                });
                if covered {
                    return true;
                }
            }
            _ = timeout => return false,
        }
    }
}

/// Exercises a spread of real watchers whose registered watch path is spelled
/// differently from the path the OS reports events under. Each scenario watches
/// some directory and then mutates the on-disk file; a correct watcher must
/// deliver an event (or a rescan) for every scenario.
///
/// This asserts the residual path-aliasing bugs that real-casing the watch root
/// at add-time does NOT fix. The headline failure is `symlink_ancestor`:
/// watching a path that traverses a symlinked ancestor. On macOS FSEvents
/// reports events under the resolved real path, which no longer has the
/// symlinked prefix the watch root was registered with, so the events are
/// filtered out and never delivered.
///
/// Platform notes (the test runs everywhere but scenarios self-skip when they
/// cannot apply):
/// - Case scenarios require a case-insensitive filesystem (macOS/Windows
///   default; case-sensitive Linux/APFS skip them).
/// - Symlink scenarios require symlink creation (skipped on Windows without the
///   privilege).
/// - `symlink_ancestor` fails on macOS (FSEvents canonicalizes) but is expected
///   to pass on Linux (notify reconstructs paths from the watch path you pass),
///   which is itself a useful demonstration that this is an FSEvents-specific
///   bug.
#[gpui::test]
async fn test_realfs_watch_aliased_watch_paths_deliver_events(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();

    let fs = RealFs::new(None, executor.clone());
    let temp_dir = TempDir::new().expect("create temp dir");
    let root = temp_dir.path().to_path_buf();
    let latency = Duration::from_millis(10);

    // Probe the real filesystem for case sensitivity rather than guessing from
    // the platform.
    std::fs::create_dir_all(root.join("CaseProbe")).expect("create case probe dir");
    let case_insensitive = root.join("caseprobe").exists();
    eprintln!("filesystem is case-insensitive: {case_insensitive}");

    struct Scenario {
        name: &'static str,
        events: Pin<Box<dyn Send + futures::Stream<Item = Vec<PathEvent>>>>,
        _watcher: Arc<dyn Watcher>,
        path_matches: Box<dyn Fn(&Path) -> bool + Send + Sync>,
        action: Option<Box<dyn FnOnce() + Send>>,
    }

    let mut scenarios: Vec<Scenario> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    // --- Headline residual bug: watch path traverses a symlinked ancestor. ---
    {
        let real = root.join("ancestor_real");
        let inner = real.join("inner");
        std::fs::create_dir_all(&inner).expect("create symlinked-ancestor target");
        let link = root.join("ancestor_link");
        match make_dir_symlink(&real, &link) {
            Ok(()) => {
                let (events, watcher) = fs.watch(&link.join("inner"), latency).await;
                let file = inner.join("symlink_ancestor.txt");
                scenarios.push(Scenario {
                    name: "symlink_ancestor",
                    events,
                    _watcher: watcher,
                    path_matches: Box::new(|path| {
                        path.ends_with(Path::new("inner/symlink_ancestor.txt"))
                    }),
                    action: Some(Box::new(move || {
                        std::fs::write(&file, b"x").expect("write symlink-ancestor file");
                    })),
                });
            }
            Err(error) => skipped.push(format!("symlink_ancestor (cannot symlink: {error})")),
        }
    }

    // --- Control: watching a symlinked root IS handled (RealFs::watch follows
    //     the root symlink and also watches the target). ---
    {
        let real = root.join("root_real");
        std::fs::create_dir_all(&real).expect("create symlinked-root target");
        let link = root.join("root_link");
        match make_dir_symlink(&real, &link) {
            Ok(()) => {
                let (events, watcher) = fs.watch(&link, latency).await;
                let file = real.join("symlink_root.txt");
                scenarios.push(Scenario {
                    name: "symlink_root",
                    events,
                    _watcher: watcher,
                    path_matches: Box::new(|path| path.ends_with(Path::new("symlink_root.txt"))),
                    action: Some(Box::new(move || {
                        std::fs::write(&file, b"x").expect("write symlink-root file");
                    })),
                });
            }
            Err(error) => skipped.push(format!("symlink_root (cannot symlink: {error})")),
        }
    }

    // --- Control: wrong-case watch root (the originally-reported bug, which the
    //     real-casing fix already addresses). ---
    if case_insensitive {
        let real = root.join("CaseAlpha");
        std::fs::create_dir_all(&real).expect("create wrong-case root");
        let lower = PathBuf::from(real.to_string_lossy().to_lowercase());
        let (events, watcher) = fs.watch(&lower, latency).await;
        let file = real.join("alpha.txt");
        scenarios.push(Scenario {
            name: "wrong_case_root",
            events,
            _watcher: watcher,
            path_matches: Box::new(|path| path.ends_with(Path::new("alpha.txt"))),
            action: Some(Box::new(move || {
                std::fs::write(&file, b"x").expect("write wrong-case-root file");
            })),
        });
    } else {
        skipped.push("wrong_case_root (case-sensitive fs)".to_owned());
    }

    // --- Control: wrong-case nested watch path. ---
    if case_insensitive {
        let real = root.join("CaseBravo").join("Inner");
        std::fs::create_dir_all(&real).expect("create wrong-case nested dir");
        let lower = PathBuf::from(real.to_string_lossy().to_lowercase());
        let (events, watcher) = fs.watch(&lower, latency).await;
        let file = real.join("bravo.txt");
        scenarios.push(Scenario {
            name: "nested_wrong_case",
            events,
            _watcher: watcher,
            path_matches: Box::new(|path| path.ends_with(Path::new("bravo.txt"))),
            action: Some(Box::new(move || {
                std::fs::write(&file, b"x").expect("write nested-wrong-case file");
            })),
        });
    } else {
        skipped.push("nested_wrong_case (case-sensitive fs)".to_owned());
    }

    // --- Residual bug: the watched root is renamed to a different casing after
    //     the watch is established, so later events arrive under a spelling the
    //     registered (old-case) root no longer matches. ---
    if case_insensitive {
        let real = root.join("CaseEcho");
        std::fs::create_dir_all(&real).expect("create case-rename dir");
        let (events, watcher) = fs.watch(&real, latency).await;
        let renamed = root.join("CASEECHO");
        let file = renamed.join("echo.txt");
        scenarios.push(Scenario {
            name: "case_rename_root",
            events,
            _watcher: watcher,
            path_matches: Box::new(|path| path.ends_with(Path::new("echo.txt"))),
            action: Some(Box::new(move || {
                std::fs::rename(&real, &renamed).expect("case-only rename of watched root");
                std::fs::write(&file, b"x").expect("write case-rename file");
            })),
        });
    } else {
        skipped.push("case_rename_root (case-sensitive fs)".to_owned());
    }

    // Let every watch settle before mutating, then perform all mutations.
    executor.timer(Duration::from_millis(250)).await;
    for scenario in &mut scenarios {
        if let Some(action) = scenario.action.take() {
            action();
        }
    }

    let mut failures = Vec::new();
    for scenario in &mut scenarios {
        let delivered = watcher_delivered_event(
            &mut scenario.events,
            &executor,
            Duration::from_secs(3),
            scenario.path_matches.as_ref(),
        )
        .await;
        eprintln!("scenario {}: delivered={delivered}", scenario.name);
        if !delivered {
            failures.push(scenario.name);
        }
    }

    for name in &skipped {
        eprintln!("scenario skipped: {name}");
    }

    assert!(
        failures.is_empty(),
        "watchers failed to deliver events for {failures:?} (skipped: {skipped:?})"
    );
}

#[gpui::test]
#[ignore = "stress test; run explicitly when needed"]
async fn test_realfs_watch_stress_reports_missed_paths(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    const FILE_COUNT: usize = 32000;
    cx.executor().allow_parking();

    let fs = RealFs::new(None, executor.clone());
    let temp_dir = TempDir::new().expect("create temp dir");
    let root = temp_dir.path();

    let mut file_paths = Vec::with_capacity(FILE_COUNT);
    let mut expected_paths = BTreeSet::new();

    for index in 0..FILE_COUNT {
        let dir_path = root.join(format!("dir-{index:04}"));
        let file_path = dir_path.join("file.txt");
        fs.create_dir(&dir_path).await.expect("create watched dir");
        fs.write(&file_path, b"before")
            .await
            .expect("create initial file");
        expected_paths.insert(file_path.clone());
        file_paths.push(file_path);
    }

    let (mut events, watcher) = fs.watch(root, Duration::from_millis(10)).await;
    let _watcher = watcher;

    for file_path in &expected_paths {
        _watcher
            .add(file_path.parent().expect("file has parent"))
            .expect("add explicit directory watch");
    }

    for (index, file_path) in file_paths.iter().enumerate() {
        let content = format!("after-{index}");
        fs.write(file_path, content.as_bytes())
            .await
            .expect("modify watched file");
    }

    let mut changed_paths = BTreeSet::new();
    let mut rescan_count: u32 = 0;
    let timeout = executor.timer(Duration::from_secs(10)).fuse();

    futures::pin_mut!(timeout);

    let mut ticks = 0;
    while ticks < 1000 {
        if let Some(batch) = events.next().fuse().now_or_never().flatten() {
            for event in batch {
                if event.kind == Some(PathEventKind::Rescan) {
                    rescan_count += 1;
                }
                if expected_paths.contains(&event.path) {
                    changed_paths.insert(event.path);
                }
            }
            if changed_paths.len() == expected_paths.len() {
                break;
            }
            ticks = 0;
        } else {
            ticks += 1;
            executor.timer(Duration::from_millis(10)).await;
        }
    }

    let missed_paths: BTreeSet<_> = expected_paths.difference(&changed_paths).cloned().collect();

    eprintln!(
        "realfs watch stress: expected={}, observed={}, missed={}, rescan={}",
        expected_paths.len(),
        changed_paths.len(),
        missed_paths.len(),
        rescan_count
    );

    assert!(
        missed_paths.is_empty() || rescan_count > 0,
        "missed {} paths without rescan being reported",
        missed_paths.len()
    );
}

#[gpui::test]
async fn restore_can_be_retried_after_collision(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.background_executor.clone());
    let path = path!("/root/a.txt");
    let remove_options = RemoveOptions::default();
    fs.insert_tree(path!("/root"), json!({ "a.txt": "original"}))
        .await;

    // We'll first trash the `a.txt` file so we can hold onto its `TrashId`,
    // allowing us to later attempt restoring it again, ensuring that it didn't
    // get removed from the trash state, even if restoring failed.
    let trash_id = fs.trash(path.as_ref(), remove_options).await.unwrap();

    fs.insert_file(path, "conflicting".into()).await;
    let err = fs.restore(trash_id).await.unwrap_err();
    assert!(matches!(err, TrashRestoreError::Collision { .. }));

    fs.remove_file(path.as_ref(), remove_options).await.unwrap();
    let restored_path = fs.restore(trash_id).await.unwrap();
    assert_eq!(fs.load(restored_path.as_path()).await.unwrap(), "original");

    assert!(matches!(
        fs.restore(trash_id).await.unwrap_err(),
        TrashRestoreError::AlreadyRestored
    ));
}

#[cfg(unix)]
#[gpui::test]
async fn test_realfs_confined_directory_creation_rejects_admission_replacements(
    executor: BackgroundExecutor,
) {
    use std::os::unix::fs::{MetadataExt as _, symlink};
    let directory = TempDir::new().expect("confined fixture");
    let base = std::fs::canonicalize(directory.path()).expect("canonical fixture");
    let root = base.join("root");
    let outside = base.join("outside");
    std::fs::create_dir(&root).expect("root");
    std::fs::create_dir(&outside).expect("outside");
    let filesystem = RealFs::new(None, executor);
    let root_inode = std::fs::metadata(&root).expect("root identity").ino();
    filesystem
        .create_dir_confined(&root, Path::new("target"), root_inode, root_inode)
        .await
        .expect("fresh confined root");
    let existing_inode = std::fs::metadata(root.join("target"))
        .expect("created target")
        .ino();
    filesystem
        .create_dir_confined(&root, Path::new("target"), root_inode, root_inode)
        .await
        .expect("adopt existing ordinary root");
    assert_eq!(
        std::fs::metadata(root.join("target"))
            .expect("retained target")
            .ino(),
        existing_inode
    );
    symlink(&outside, root.join("linked-target")).expect("target alias after admission");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new("linked-target"), root_inode, root_inode)
            .await
            .is_err()
    );
    assert!(
        std::fs::symlink_metadata(root.join("linked-target"))
            .expect("preserved alias")
            .file_type()
            .is_symlink()
    );
    let oversized = "x".repeat(4097);
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new(&oversized), root_inode, root_inode)
            .await
            .is_err()
    );
    let too_deep = vec!["x"; 33].join("/");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new(&too_deep), root_inode, root_inode)
            .await
            .is_err()
    );
    assert!(!root.join("x").exists());
    let parent = root.join("parent");
    std::fs::create_dir(&parent).expect("parent");
    let parent_inode = std::fs::metadata(&parent).expect("admitted parent").ino();
    std::fs::rename(&parent, root.join("retained-parent")).expect("replace after admission");
    symlink(&outside, &parent).expect("raced parent alias");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new("parent/escape"), root_inode, parent_inode)
            .await
            .is_err()
    );
    assert!(!outside.join("escape").exists());
    std::fs::remove_file(&parent).expect("remove only fixture alias");
    std::fs::create_dir(&parent).expect("replacement inode");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new("parent/escape"), root_inode, parent_inode)
            .await
            .is_err()
    );
    assert!(!parent.join("escape").exists());
    std::fs::write(root.join("collision"), b"retained").expect("unrelated file");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new("collision"), root_inode, root_inode)
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read(root.join("collision")).expect("preserved collision"),
        b"retained"
    );
    std::fs::rename(&root, base.join("retained-root")).expect("root changed after admission");
    std::fs::create_dir(&root).expect("new root");
    assert!(
        filesystem
            .create_dir_confined(&root, Path::new("target"), root_inode, root_inode)
            .await
            .is_err()
    );
    assert!(!root.join("target").exists());
    assert!(
        filesystem
            .create_dir_confined(
                &root,
                Path::new("../outside/escape"),
                root_inode,
                root_inode
            )
            .await
            .is_err()
    );
    assert!(!outside.join("escape").exists());
}

#[cfg(unix)]
#[gpui::test]
async fn real_fs_confined_owned_file_preserves_original_and_rejects_escape(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use std::os::{fd::AsRawFd as _, unix::fs::MetadataExt as _};
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("owned confined file fixture");
    let root = fixture.path().join("root");
    std::fs::create_dir_all(root.join("nested")).expect("root directories");
    let relative = Path::new("nested/helper");
    let path = root.join(relative);
    std::fs::write(&path, b"original").expect("original bytes");
    let filesystem = RealFs::new(None, executor);
    let root_inode = std::fs::metadata(&root).expect("root metadata").ino();
    let file_inode = std::fs::metadata(&path).expect("file metadata").ino();
    let mut file = filesystem
        .open_sync_confined_file_handle(&root, relative, root_inode, file_inode)
        .await
        .expect("owned confined file");
    assert_eq!(file.metadata().expect("owned metadata").ino(), file_inode);
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) };
    assert!(flags >= 0 && flags & libc::FD_CLOEXEC != 0);
    let replacement = fixture.path().join("replacement");
    std::fs::write(&replacement, b"replacement").expect("replacement bytes");
    std::fs::rename(&replacement, &path).expect("replace pathname");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("retained descriptor read");
    assert_eq!(bytes, b"original");
    assert!(
        filesystem
            .open_sync_confined_file_handle(&root, relative, root_inode, file_inode)
            .await
            .is_err(),
        "old inode cannot adopt replacement"
    );
    let replacement_inode = std::fs::metadata(&path).expect("new metadata").ino();
    assert!(
        filesystem
            .open_sync_confined_file_handle(&root, relative, root_inode ^ 1, replacement_inode)
            .await
            .is_err()
    );
    for invalid in [
        Path::new("../replacement"),
        path.as_path(),
        Path::new("nested"),
    ] {
        assert!(
            filesystem
                .open_sync_confined_file_handle(&root, invalid, root_inode, replacement_inode)
                .await
                .is_err()
        );
    }
    let outside = fixture.path().join("outside");
    std::fs::write(&outside, b"outside").expect("outside bytes");
    std::os::unix::fs::symlink(&outside, root.join("linked-file")).expect("file symlink");
    std::os::unix::fs::symlink(fixture.path(), root.join("linked-parent"))
        .expect("directory symlink");
    let outside_inode = std::fs::metadata(&outside).expect("outside metadata").ino();
    for invalid in [Path::new("linked-file"), Path::new("linked-parent/outside")] {
        assert!(
            filesystem
                .open_sync_confined_file_handle(&root, invalid, root_inode, outside_inode)
                .await
                .is_err()
        );
    }
    let alias = fixture.path().join("root-alias");
    std::os::unix::fs::symlink(&root, &alias).expect("root symlink");
    assert!(
        filesystem
            .open_sync_confined_file_handle(&alias, relative, root_inode, replacement_inode)
            .await
            .is_err()
    );
}

#[gpui::test]
async fn fake_fs_confined_owned_file_is_unsupported_while_reader_stays_simulated(
    executor: BackgroundExecutor,
) {
    let filesystem = FakeFs::new(executor);
    filesystem
        .insert_tree("/root", serde_json::json!({"helper": "simulated"}))
        .await;
    let root = filesystem
        .metadata(Path::new("/root"))
        .await
        .expect("fake root metadata")
        .expect("fake root");
    let file = filesystem
        .metadata(Path::new("/root/helper"))
        .await
        .expect("fake file metadata")
        .expect("fake file");
    let error = filesystem
        .open_sync_confined_file_handle(
            Path::new("/root"),
            Path::new("helper"),
            root.inode,
            file.inode,
        )
        .await
        .expect_err("fake cannot mint owned file authority");
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .expect("typed unsupported error")
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    let mut reader = filesystem
        .open_sync_confined_file(
            Path::new("/root"),
            Path::new("helper"),
            root.inode,
            file.inode,
        )
        .await
        .expect("existing simulated reader");
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).expect("simulated bytes");
    assert_eq!(bytes, b"simulated");
}

async fn check_retained_root_contract(filesystem: &dyn Fs, root: &Path) {
    filesystem
        .create_dir(&root.join("nested"))
        .await
        .expect("fixture directories");
    filesystem
        .write(&root.join("nested/settings.json"), b"{}\n")
        .await
        .expect("fixture settings");
    let admission = filesystem
        .capture_confined_root(root)
        .await
        .expect("admit original root");
    let clone = admission.clone();
    assert!(admission.same_admission(&clone));
    let readmission = filesystem
        .capture_confined_root(root)
        .await
        .expect("explicit readmission");
    assert!(!admission.same_admission(&readmission));
    let handle = admission.file_handle();
    let handle_owners = Arc::strong_count(&handle);
    let additional_clone = admission.clone();
    assert_eq!(Arc::strong_count(&handle), handle_owners);
    drop(additional_clone);
    let present = filesystem
        .read_confined_file(&admission, root, Path::new("nested/settings.json"), 3)
        .await
        .expect("bounded original read");
    assert_eq!(present.bytes, b"{}\n");
    assert_eq!(present.metadata.len, 3);
    for limit in [0, 2, usize::MAX] {
        assert!(
            filesystem
                .read_confined_file(&admission, root, Path::new("nested/settings.json"), limit)
                .await
                .is_err()
        );
    }
    for missing in ["nested/missing.json", "missing/child/settings.json"] {
        assert!(
            filesystem
                .is_confined_file_absent(&admission, root, Path::new(missing))
                .await
                .expect("confirmed absence")
        );
        assert!(
            filesystem
                .read_confined_file(&admission, root, Path::new(missing), 32)
                .await
                .is_err()
        );
    }
    for present in ["nested", "nested/settings.json"] {
        assert!(
            !filesystem
                .is_confined_file_absent(&admission, root, Path::new(present))
                .await
                .expect("existing entry")
        );
    }
    filesystem
        .create_symlink(&root.join("dangling-file"), root.join("uncreated"))
        .await
        .expect("dangling final link");
    assert!(
        !filesystem
            .is_confined_file_absent(&admission, root, Path::new("dangling-file"))
            .await
            .expect("final alias is present")
    );
    assert!(
        filesystem
            .read_confined_file(&admission, root, Path::new("dangling-file"), 32)
            .await
            .is_err()
    );
    for (name, target) in [
        ("linked-parent", root.join("nested")),
        ("dangling-parent", root.join("uncreated")),
    ] {
        filesystem
            .create_symlink(&root.join(name), target)
            .await
            .expect("parent alias");
        assert!(
            filesystem
                .is_confined_file_absent(&admission, root, &Path::new(name).join("missing.json"))
                .await
                .is_err()
        );
        assert!(
            filesystem
                .read_confined_file(&admission, root, &Path::new(name).join("settings.json"), 32)
                .await
                .is_err()
        );
    }
    for invalid in [
        Path::new("../escape"),
        root,
        Path::new(""),
        Path::new("nested/settings.json/child"),
    ] {
        assert!(
            filesystem
                .is_confined_file_absent(&admission, root, invalid)
                .await
                .is_err()
        );
        assert!(
            filesystem
                .read_confined_file(&admission, root, invalid, 32)
                .await
                .is_err()
        );
    }
    for invalid in ["x".repeat(4097), vec!["x"; 257].join("/")] {
        assert!(
            filesystem
                .is_confined_file_absent(&admission, root, Path::new(&invalid))
                .await
                .is_err()
        );
        assert!(
            filesystem
                .read_confined_file(&admission, root, Path::new(&invalid), 32)
                .await
                .is_err()
        );
    }
    let file_path = root.join("nested/settings.json");
    let file_admission = filesystem
        .capture_confined_root(&file_path)
        .await
        .expect("admit ordinary file");
    assert!(!file_admission.metadata().is_dir);
    assert_eq!(
        filesystem
            .read_confined_file(&file_admission, &file_path, Path::new(""), 3)
            .await
            .expect("file-root bytes")
            .bytes,
        b"{}\n"
    );
    assert!(
        filesystem
            .read_confined_file(&file_admission, &file_path, Path::new("child"), 3)
            .await
            .is_err()
    );
    assert!(
        filesystem
            .is_confined_file_absent(&file_admission, &file_path, Path::new("missing"))
            .await
            .is_err()
    );
    let retained = root.with_extension("retained");
    filesystem
        .rename(root, &retained, RenameOptions::default())
        .await
        .expect("move original root");
    filesystem
        .create_dir(&root.join("nested"))
        .await
        .expect("replacement root");
    filesystem
        .write(&root.join("nested/settings.json"), b"replacement canary")
        .await
        .expect("replacement canary");
    assert!(
        filesystem
            .read_confined_file(&admission, root, Path::new("nested/settings.json"), 64)
            .await
            .is_err()
    );
    assert!(
        filesystem
            .is_confined_file_absent(&admission, root, Path::new("missing"))
            .await
            .is_err()
    );
    assert_eq!(
        filesystem
            .read_confined_file(&admission, &retained, Path::new("nested/settings.json"), 3)
            .await
            .expect("same admitted object at accepted new name")
            .bytes,
        b"{}\n"
    );
    drop(admission);
    drop(clone);
    assert_eq!(Arc::strong_count(&handle), 1);
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_contract(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("retained-root fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("root");
    let filesystem = RealFs::new(None, executor);
    check_retained_root_contract(&filesystem, &root).await;
}

#[gpui::test]
async fn test_fake_fs_retained_root_contract(cx: &mut TestAppContext) {
    let filesystem = FakeFs::new(cx.executor());
    check_retained_root_contract(filesystem.as_ref(), Path::new(path!("/retained-root"))).await;
}

#[gpui::test]
async fn test_fake_fs_retained_root_rejects_another_domain(cx: &mut TestAppContext) {
    let first = FakeFs::new(cx.executor());
    let second = FakeFs::new(cx.executor());
    let root = Path::new(path!("/domain-root"));
    for filesystem in [&first, &second] {
        filesystem.create_dir(root).await.expect("domain root");
        filesystem
            .write(&root.join("settings.json"), b"domain canary")
            .await
            .expect("domain canary");
    }
    let first_admission = first
        .capture_confined_root(root)
        .await
        .expect("first domain admission");
    let second_admission = second
        .capture_confined_root(root)
        .await
        .expect("second domain admission");
    assert_eq!(
        first_admission.metadata().inode,
        second_admission.metadata().inode
    );
    assert!(
        second
            .read_confined_file(&first_admission, root, Path::new("settings.json"), 64)
            .await
            .is_err()
    );
    assert!(
        second
            .is_confined_file_absent(&first_admission, root, Path::new("missing"))
            .await
            .is_err()
    );
    assert_eq!(
        second
            .read_confined_file(&second_admission, root, Path::new("settings.json"), 64)
            .await
            .expect("same-domain control")
            .bytes,
        b"domain canary"
    );
}

fn require_fake_confined_link_limit<T>(result: anyhow::Result<T>) {
    let error = match result {
        Ok(_) => panic!("strict named-root resolution must reject link exhaustion"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "Fake filesystem symbolic link limit exceeded"
    );
}

#[gpui::test]
async fn test_fake_fs_retained_root_named_root_cycles_are_errors(cx: &mut TestAppContext) {
    let filesystem = FakeFs::new(cx.executor());
    let real_root = Path::new(path!("/bounded-link-real/owned"));
    filesystem
        .create_dir(real_root)
        .await
        .expect("ordinary root");
    filesystem
        .write(&real_root.join("settings.json"), b"trusted")
        .await
        .expect("ordinary bytes");
    let admission = filesystem
        .capture_confined_root(real_root)
        .await
        .expect("ordinary admission");
    let self_link = Path::new(path!("/bounded-link-self"));
    filesystem
        .create_symlink(self_link, self_link.to_path_buf())
        .await
        .expect("self-link fixture");
    let first = Path::new(path!("/bounded-link-first"));
    let second = Path::new(path!("/bounded-link-second"));
    filesystem
        .create_symlink(first, second.to_path_buf())
        .await
        .expect("first cyclic link");
    filesystem
        .create_symlink(second, first.to_path_buf())
        .await
        .expect("second cyclic link");
    for named_root in [self_link.join("owned"), first.join("owned")] {
        require_fake_confined_link_limit(filesystem.capture_confined_root(&named_root).await);
        require_fake_confined_link_limit(
            filesystem
                .read_confined_file(&admission, &named_root, Path::new("settings.json"), 32)
                .await,
        );
        require_fake_confined_link_limit(
            filesystem
                .is_confined_file_absent(&admission, &named_root, Path::new("missing.json"))
                .await,
        );
        assert_eq!(
            filesystem
                .read_confined_file(&admission, real_root, Path::new("settings.json"), 32)
                .await
                .expect("state and admission lock remain usable after each rejected cycle")
                .bytes,
            b"trusted"
        );
        assert!(
            filesystem
                .is_confined_file_absent(&admission, real_root, Path::new("missing.json"))
                .await
                .expect("ordinary absence after rejected cycle")
        );
    }
    let missing_root = Path::new(path!("/bounded-link-missing/owned"));
    for error in [
        filesystem
            .capture_confined_root(missing_root)
            .await
            .err()
            .expect("missing admission"),
        filesystem
            .read_confined_file(&admission, missing_root, Path::new("settings.json"), 32)
            .await
            .err()
            .expect("missing named-root read"),
        filesystem
            .is_confined_file_absent(&admission, missing_root, Path::new("missing.json"))
            .await
            .err()
            .expect("missing named root cannot authorize absence"),
    ] {
        assert_eq!(error.to_string(), "Confined root is missing");
    }
}

#[gpui::test]
async fn test_fake_fs_retained_root_finite_ancestor_alias_is_preserved(cx: &mut TestAppContext) {
    let filesystem = FakeFs::new(cx.executor());
    let root_parent = Path::new(path!("/finite-link-real"));
    let real_root = root_parent.join("owned");
    filesystem
        .create_dir(&real_root)
        .await
        .expect("ordinary root");
    filesystem
        .write(&real_root.join("settings.json"), b"trusted")
        .await
        .expect("ordinary bytes");
    let admission = filesystem
        .capture_confined_root(&real_root)
        .await
        .expect("ordinary admission");
    let alias_parent = Path::new(path!("/finite-link-alias"));
    filesystem
        .create_symlink(alias_parent, root_parent.to_path_buf())
        .await
        .expect("finite ancestor alias");
    let alias_root = alias_parent.join("owned");
    let alias_admission = filesystem
        .capture_confined_root(&alias_root)
        .await
        .expect("final root is ordinary beneath one ancestor alias");
    assert_eq!(admission.metadata().inode, alias_admission.metadata().inode);
    assert!(!admission.same_admission(&alias_admission));
    for token in [&admission, &alias_admission] {
        assert_eq!(
            filesystem
                .read_confined_file(token, &alias_root, Path::new("settings.json"), 32)
                .await
                .expect("finite alias read")
                .bytes,
            b"trusted"
        );
        assert!(
            !filesystem
                .is_confined_file_absent(token, &alias_root, Path::new("settings.json"))
                .await
                .expect("existing file through alias")
        );
        assert!(
            filesystem
                .is_confined_file_absent(token, &alias_root, Path::new("missing.json"))
                .await
                .expect("true descendant absence through alias")
        );
    }
    assert_eq!(
        filesystem
            .canonicalize(&alias_root)
            .await
            .expect("ordinary resolver retains finite aliases"),
        real_root
    );
}

#[gpui::test]
async fn test_fake_fs_retained_root_link_budget_does_not_change_ordinary_resolution(
    cx: &mut TestAppContext,
) {
    let filesystem = FakeFs::new(cx.executor());
    let root_parent = Path::new(path!("/link-budget-real"));
    let real_root = root_parent.join("owned");
    filesystem
        .create_dir(&real_root)
        .await
        .expect("ordinary root");
    filesystem
        .write(&real_root.join("settings.json"), b"trusted")
        .await
        .expect("ordinary bytes");
    let admission = filesystem
        .capture_confined_root(&real_root)
        .await
        .expect("ordinary admission");
    let links = (0..41)
        .map(|index| Path::new(path!("/")).join(format!("link-budget-{index}")))
        .collect::<Vec<_>>();
    for (index, link) in links.iter().enumerate() {
        let target = links
            .get(index + 1)
            .cloned()
            .unwrap_or_else(|| root_parent.to_path_buf());
        filesystem
            .create_symlink(link, target)
            .await
            .expect("finite chain fixture");
    }
    let forty = links.get(1).expect("forty-link boundary").join("owned");
    let boundary = filesystem
        .capture_confined_root(&forty)
        .await
        .expect("exact finite-link budget is allowed");
    assert_eq!(
        filesystem
            .read_confined_file(&boundary, &forty, Path::new("settings.json"), 32)
            .await
            .expect("boundary read")
            .bytes,
        b"trusted"
    );
    assert!(
        filesystem
            .is_confined_file_absent(&boundary, &forty, Path::new("missing.json"))
            .await
            .expect("boundary absence")
    );
    let forty_one = links.first().expect("over-budget head").join("owned");
    require_fake_confined_link_limit(filesystem.capture_confined_root(&forty_one).await);
    require_fake_confined_link_limit(
        filesystem
            .read_confined_file(&admission, &forty_one, Path::new("settings.json"), 32)
            .await,
    );
    require_fake_confined_link_limit(
        filesystem
            .is_confined_file_absent(&admission, &forty_one, Path::new("missing.json"))
            .await,
    );
    assert_eq!(
        filesystem
            .canonicalize(&forty_one)
            .await
            .expect("ordinary resolver preserves preexisting finite-chain behavior"),
        real_root
    );
}

#[cfg(unix)]
async fn require_native_checkpoint<T: Unpin + 'static>(
    reached: futures::channel::oneshot::Receiver<()>,
    operation: &mut gpui::Task<anyhow::Result<T>>,
    label: &str,
) {
    match futures::future::select(reached, operation).await {
        futures::future::Either::Left((receipt, _)) => receipt.expect(label),
        futures::future::Either::Right((result, _)) => match result {
            Ok(_) => panic!("Native operation completed before {label}"),
            Err(error) => panic!("Native operation failed before {label}: {error:#}"),
        },
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_handle_last_owner_drop(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("native lifetime fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("root");
    std::fs::create_dir(&root).expect("root");
    let filesystem: Arc<dyn Fs> = Arc::new(RealFs::new(None, executor));
    let admission = filesystem
        .capture_confined_root(&root)
        .await
        .expect("admission");
    let cloned = admission.clone();
    let handle = admission.file_handle();
    let weak = Arc::downgrade(&handle);
    drop(admission);
    drop(cloned);
    assert_eq!(
        handle
            .current_path(&filesystem)
            .expect("retained handle path"),
        root
    );
    assert!(weak.upgrade().is_some());
    drop(handle);
    assert!(
        weak.upgrade().is_none(),
        "last external handle releases native File owner"
    );
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_file_and_named_root_replacements(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use std::os::unix::fs::{MetadataExt as _, symlink};
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("native replacement fixture");
    let base = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
    let root = base.join("root");
    std::fs::create_dir(&root).expect("root");
    let original = root.join("settings.json");
    std::fs::write(&original, b"original").expect("original");
    let filesystem = RealFs::new(None, executor);
    let admission = filesystem
        .capture_confined_root(&root)
        .await
        .expect("directory admission");
    let file_admission = filesystem
        .capture_confined_root(&original)
        .await
        .expect("file admission");
    std::fs::rename(&original, root.join("retained-file")).expect("retain original file");
    std::fs::write(&original, b"replacement").expect("same-name different file");
    assert_ne!(
        std::fs::metadata(&original)
            .expect("replacement metadata")
            .ino(),
        file_admission.metadata().inode
    );
    assert!(
        filesystem
            .read_confined_file(&file_admission, &original, Path::new(""), 64)
            .await
            .is_err()
    );
    assert_eq!(
        filesystem
            .read_confined_file(
                &file_admission,
                &root.join("retained-file"),
                Path::new(""),
                64
            )
            .await
            .expect("same file new name")
            .bytes,
        b"original"
    );
    assert_eq!(
        filesystem
            .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
            .await
            .expect("directory token admits current ordinary descendant")
            .bytes,
        b"replacement"
    );
    std::fs::rename(&root, base.join("moved-root")).expect("retain original root");
    symlink(base.join("moved-root"), &root).expect("final named-root alias");
    assert!(
        filesystem
            .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
            .await
            .is_err()
    );
    assert!(
        filesystem
            .is_confined_file_absent(&admission, &root, Path::new("missing"))
            .await
            .is_err()
    );
    assert_eq!(
        filesystem
            .read_confined_file(
                &admission,
                &base.join("moved-root"),
                Path::new("settings.json"),
                64
            )
            .await
            .expect("ordinary accepted moved name")
            .bytes,
        b"replacement"
    );
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_native_encoding_and_nonregular_controls(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use std::{
        ffi::{CString, OsString},
        os::unix::{
            ffi::{OsStrExt as _, OsStringExt as _},
            net::UnixListener,
        },
    };
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("native boundary fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("root");
    std::fs::create_dir(&root).expect("root");
    let filesystem = RealFs::new(None, executor);
    let admission = filesystem
        .capture_confined_root(&root)
        .await
        .expect("admission");
    let native_name = PathBuf::from("设置-λ.json");
    std::fs::write(root.join(&native_name), [0xff, 0x00])
        .expect("Unicode name with raw non-UTF8 bytes");
    assert_eq!(
        filesystem
            .read_confined_file(&admission, &root, &native_name, 2)
            .await
            .expect("native byte semantics")
            .bytes,
        [0xff, 0x00]
    );
    assert!(
        filesystem
            .read_confined_file(&admission, &root, &native_name, 1)
            .await
            .is_err()
    );
    assert!(
        !filesystem
            .is_confined_file_absent(&admission, &root, &native_name)
            .await
            .expect("native name exists")
    );
    let nul = PathBuf::from(OsString::from_vec(vec![b'f', 0, b'x']));
    assert!(
        filesystem
            .read_confined_file(&admission, &root, &nul, 32)
            .await
            .is_err()
    );
    assert!(
        filesystem
            .is_confined_file_absent(&admission, &root, &nul)
            .await
            .is_err()
    );
    let socket = UnixListener::bind(root.join("socket")).expect("socket fixture");
    let fifo = CString::new(root.join("fifo").as_os_str().as_bytes()).expect("fixture has no NUL");
    // The CString remains alive with a terminating NUL throughout the POSIX call.
    let status = unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) };
    assert_eq!(
        status,
        0,
        "fifo fixture: {}",
        std::io::Error::last_os_error()
    );
    for name in ["socket", "fifo"] {
        assert!(
            filesystem
                .capture_confined_root(&root.join(name))
                .await
                .is_err()
        );
        assert!(
            filesystem
                .read_confined_file(&admission, &root, Path::new(name), 32)
                .await
                .is_err()
        );
        assert!(
            !filesystem
                .is_confined_file_absent(&admission, &root, Path::new(name))
                .await
                .expect("nonregular object exists")
        );
    }
    std::fs::write(root.join("empty"), []).expect("empty regular file");
    assert!(
        filesystem
            .read_confined_file(&admission, &root, Path::new("empty"), 0)
            .await
            .expect("empty within zero bound")
            .bytes
            .is_empty()
    );
    drop(socket);
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_revalidates_substitution_at_read_checkpoint(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    for mode in ["directory-file", "parent", "root", "file-root"] {
        let fixture = TempDir::new().expect("read substitution fixture");
        let root = std::fs::canonicalize(fixture.path())
            .expect("canonical fixture")
            .join("root");
        std::fs::create_dir_all(root.join("nested")).expect("root and parent");
        let path = root.join("nested/settings.json");
        std::fs::write(&path, b"original").expect("original");
        let filesystem = Arc::new(RealFs::new(None, executor.clone()));
        let named_root = if mode == "file-root" {
            path.clone()
        } else {
            root.clone()
        };
        let relative = if mode == "file-root" {
            Path::new("")
        } else {
            Path::new("nested/settings.json")
        };
        let admission = filesystem
            .capture_confined_root(&named_root)
            .await
            .expect("admission");
        let (reached, resume) = admission
            .pause_native_operation_for_test(NativeConfinedCheckpoint::AfterReadBytes)
            .expect("install checkpoint");
        let mut operation = executor.spawn({
            let filesystem = filesystem.clone();
            let admission = admission.clone();
            let named_root = named_root.clone();
            async move {
                filesystem
                    .read_confined_file(&admission, &named_root, relative, 64)
                    .await
            }
        });
        require_native_checkpoint(reached, &mut operation, "after reading original bytes").await;
        match mode {
            "parent" => {
                std::fs::rename(root.join("nested"), root.join("retained-parent"))
                    .expect("retain observed parent");
                std::fs::create_dir(root.join("nested")).expect("replacement parent");
            }
            "root" => {
                std::fs::rename(&root, root.with_extension("retained"))
                    .expect("retain admitted root");
                std::fs::create_dir_all(root.join("nested")).expect("replacement root");
            }
            _ => std::fs::rename(&path, root.join("nested/retained"))
                .expect("retain opened original"),
        }
        std::fs::write(&path, b"replacement canary").expect("same-name substitute");
        resume.send(()).expect("release read");
        assert!(
            operation.await.is_err(),
            "no bytes may be returned under a changed binding: {mode}"
        );
        let subsequent = filesystem
            .read_confined_file(&admission, &named_root, relative, 64)
            .await;
        if matches!(mode, "root" | "file-root") {
            assert!(
                subsequent.is_err(),
                "root identity remains incompatible: {mode}"
            );
        } else {
            assert_eq!(
                subsequent
                    .expect("new operation admits fresh ordinary descendant")
                    .bytes,
                b"replacement canary"
            );
        }
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_rejects_content_growth_after_pre_read_stat(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("content mutation fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("root");
    std::fs::create_dir(&root).expect("root");
    let path = root.join("settings.json");
    std::fs::write(&path, b"old").expect("original");
    let filesystem = Arc::new(RealFs::new(None, executor.clone()));
    let admission = filesystem
        .capture_confined_root(&root)
        .await
        .expect("admission");
    let (reached, resume) = admission
        .pause_native_operation_for_test(NativeConfinedCheckpoint::BeforeReadBytes)
        .expect("install checkpoint");
    let mut operation = executor.spawn({
        let filesystem = filesystem.clone();
        let admission = admission.clone();
        let root = root.clone();
        async move {
            filesystem
                .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
                .await
        }
    });
    require_native_checkpoint(
        reached,
        &mut operation,
        "initial metadata and byte-limit check",
    )
    .await;
    std::fs::write(&path, b"longer content").expect("change same inode length deterministically");
    resume.send(()).expect("release read");
    let error = operation.await.err().expect("changed length must reject");
    assert_eq!(
        error.to_string(),
        "Confined file changed during its bounded read"
    );
    assert_eq!(
        filesystem
            .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
            .await
            .expect("subsequent read and lock usable")
            .bytes,
        b"longer content"
    );
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_absence_revalidates_ancestor_binding(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    for (replace_root, missing_ancestor) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let fixture = TempDir::new().expect("absence mutation fixture");
        let root = std::fs::canonicalize(fixture.path())
            .expect("canonical fixture")
            .join("root");
        std::fs::create_dir_all(root.join("nested")).expect("ordinary ancestors");
        let filesystem = Arc::new(RealFs::new(None, executor.clone()));
        let admission = filesystem
            .capture_confined_root(&root)
            .await
            .expect("admission");
        let (reached, resume) = admission
            .pause_native_operation_for_test(NativeConfinedCheckpoint::AfterAbsenceLookup)
            .expect("install checkpoint");
        let relative = if missing_ancestor {
            Path::new("nested/uncreated/child.json")
        } else {
            Path::new("nested/missing.json")
        };
        let mut operation = executor.spawn({
            let filesystem = filesystem.clone();
            let admission = admission.clone();
            let root = root.clone();
            async move {
                filesystem
                    .is_confined_file_absent(&admission, &root, relative)
                    .await
            }
        });
        require_native_checkpoint(reached, &mut operation, "anchored absence lookup").await;
        if replace_root {
            std::fs::rename(&root, root.with_extension("retained")).expect("retain admitted root");
            std::fs::create_dir_all(root.join("nested")).expect("new root and parent");
        } else {
            std::fs::rename(root.join("nested"), root.join("retained-parent"))
                .expect("retain observed parent");
            std::fs::create_dir(root.join("nested")).expect("same-name replacement parent");
        }
        if missing_ancestor {
            std::fs::create_dir(root.join("nested/uncreated")).expect("replacement missing parent");
        }
        std::fs::write(root.join(relative), b"replacement canary").expect("canary");
        resume.send(()).expect("release absence check");
        assert!(
            operation.await.is_err(),
            "changed root/ancestor cannot authorize absence"
        );
        if !replace_root {
            assert!(
                !filesystem
                    .is_confined_file_absent(&admission, &root, relative)
                    .await
                    .expect("current parent presents canary")
            );
        }
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_retained_root_cancel_at_checkpoint_releases_owners_and_lock(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("native cancellation fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("root");
    std::fs::create_dir(&root).expect("root");
    std::fs::write(root.join("settings.json"), b"stable").expect("original");
    let filesystem = Arc::new(RealFs::new(None, executor.clone()));
    let admission = filesystem
        .capture_confined_root(&root)
        .await
        .expect("admission");
    let weak = Arc::downgrade(&admission.file_handle());
    let (reached, resume) = admission
        .pause_native_operation_for_test(NativeConfinedCheckpoint::BeforeReadBytes)
        .expect("install checkpoint");
    let mut operation = executor.spawn({
        let filesystem = filesystem.clone();
        let admission = admission.clone();
        let root = root.clone();
        async move {
            filesystem
                .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
                .await
        }
    });
    require_native_checkpoint(reached, &mut operation, "open file and admission lock").await;
    drop(operation);
    cx.executor().run_until_parked();
    assert!(
        resume.is_canceled(),
        "native paused receiver actually dropped"
    );
    assert_eq!(
        filesystem
            .read_confined_file(&admission, &root, Path::new("settings.json"), 64)
            .await
            .expect("next operation acquires released lock")
            .bytes,
        b"stable"
    );
    drop(admission);
    assert!(
        weak.upgrade().is_none(),
        "cancelled operation retained no admission File owner"
    );
}

#[cfg(unix)]
async fn check_logical_absence_contract(filesystem: &dyn Fs, root: &Path) {
    filesystem
        .create_dir(&root.join("nested"))
        .await
        .expect("fixture directory");
    filesystem
        .atomic_write(root.join("present"), String::new())
        .await
        .expect("fixture file");
    for (name, target) in [
        ("relative", PathBuf::from("nested")),
        ("absolute", root.join("nested")),
        ("chained", PathBuf::from("relative")),
        ("broken", PathBuf::from("missing-target")),
        ("broken-parent", PathBuf::from("missing-target/../nested")),
        ("file-alias", PathBuf::from("present")),
        ("file-dot", PathBuf::from("present/.")),
        ("self-cycle", PathBuf::from("self-cycle")),
        ("first-cycle", PathBuf::from("second-cycle")),
        ("second-cycle", PathBuf::from("first-cycle")),
        ("nested/up", PathBuf::from("..")),
    ] {
        filesystem
            .create_symlink(&root.join(name), target)
            .await
            .expect("fixture alias");
    }
    for suffix in [
        "missing",
        "uncreated/child.json",
        "relative/missing",
        "absolute/missing",
        "chained/missing",
        "nested/up/nested/missing",
        "nested/../missing",
        "nested/./missing",
    ] {
        let logical = root.join(suffix);
        let receipt = filesystem
            .observe_path_absence(&logical)
            .await
            .expect("stable original suffix missing")
            .expect("fresh absence receipt");
        assert_eq!(receipt.logical_path().as_os_str(), logical.as_os_str());
        assert_eq!(format!("{receipt:?}"), "AbsentPath { .. }");
    }
    for suffix in [
        "present",
        "nested",
        "relative",
        "absolute",
        "chained",
        "file-alias",
    ] {
        assert!(
            filesystem
                .observe_path_absence(&root.join(suffix))
                .await
                .expect("existing target observation")
                .is_none(),
            "{suffix}"
        );
    }
    for suffix in [
        "broken",
        "broken/settings.json",
        "broken-parent/settings.json",
        "present/child",
        "present/.",
        "present/",
        "file-dot",
        "self-cycle",
        "first-cycle",
    ] {
        assert!(
            filesystem
                .observe_path_absence(&root.join(suffix))
                .await
                .is_err(),
            "{suffix}"
        );
    }
    let load_error = filesystem
        .load(&root.join("broken/settings.json"))
        .await
        .expect_err("ordinary read of broken target");
    assert_eq!(
        load_error
            .downcast_ref::<std::io::Error>()
            .expect("ordinary I/O error")
            .kind(),
        std::io::ErrorKind::NotFound
    );
    for index in (0..40).rev() {
        let target = if index == 39 {
            PathBuf::from("nested")
        } else {
            PathBuf::from(format!("link-{}", index + 1))
        };
        filesystem
            .create_symlink(&root.join(format!("link-{index}")), target)
            .await
            .expect("finite alias");
    }
    assert!(
        filesystem
            .observe_path_absence(&root.join("link-0/missing"))
            .await
            .expect("forty aliases")
            .is_some()
    );
    filesystem
        .create_symlink(&root.join("overflow"), PathBuf::from("link-0"))
        .await
        .expect("forty-first alias");
    assert!(
        filesystem
            .observe_path_absence(&root.join("overflow/missing"))
            .await
            .is_err()
    );
    assert!(
        filesystem
            .observe_path_absence(Path::new("relative-source"))
            .await
            .is_err()
    );
    assert!(
        filesystem
            .observe_path_absence(Path::new("//unknown/source"))
            .await
            .is_err()
    );
}

#[cfg(unix)]
#[gpui::test]
async fn test_fake_fs_logical_absence_contract(cx: &mut TestAppContext) {
    let filesystem = FakeFs::new(cx.executor());
    check_logical_absence_contract(filesystem.as_ref(), Path::new("/logical-source")).await;
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_logical_absence_contract(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("logical source fixture");
    let root = std::fs::canonicalize(fixture.path())
        .expect("canonical fixture")
        .join("source");
    let filesystem = RealFs::new(None, executor);
    check_logical_absence_contract(&filesystem, &root).await;
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_logical_absence_revalidates_held_route(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    for mutation in [
        "alias",
        "same-target-alias",
        "parent",
        "target",
        "missing-parent",
    ] {
        let fixture = TempDir::new().expect("route mutation fixture");
        let root = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
        std::fs::create_dir(root.join("nested")).expect("original parent");
        std::os::unix::fs::symlink("nested", root.join("alias")).expect("original alias");
        let filesystem = Arc::new(RealFs::new(None, executor.clone()));
        let logical = root.join(if mutation == "missing-parent" {
            "alias/uncreated/child"
        } else {
            "alias/missing"
        });
        let (reached, resume) = filesystem
            .pause_path_absence_for_test()
            .expect("held first missing lookup");
        let mut operation = executor.spawn({
            let filesystem = filesystem.clone();
            let logical = logical.clone();
            async move { filesystem.observe_path_absence(&logical).await }
        });
        require_native_checkpoint(reached, &mut operation, "actual logical absence lookup").await;
        match mutation {
            "alias" => {
                std::fs::create_dir(root.join("replacement")).expect("new target");
                std::fs::remove_file(root.join("alias")).expect("remove original alias");
                std::os::unix::fs::symlink("replacement", root.join("alias"))
                    .expect("replace alias");
            }
            "same-target-alias" => {
                std::fs::remove_file(root.join("alias")).expect("remove original alias");
                std::os::unix::fs::symlink("nested", root.join("alias"))
                    .expect("same-byte replacement alias");
            }
            "parent" => {
                std::fs::rename(root.join("nested"), root.join("retained"))
                    .expect("retain observed parent");
                std::fs::create_dir(root.join("nested")).expect("same-name new parent");
            }
            "target" => std::fs::write(&logical, b"appeared").expect("target appears"),
            "missing-parent" => {
                std::fs::create_dir(root.join("nested/uncreated")).expect("missing binding appears")
            }
            _ => unreachable!(),
        }
        resume.send(()).expect("release observed mutation");
        assert!(
            operation.await.is_err(),
            "{mutation} must not authorize defaults"
        );
        let current = filesystem
            .observe_path_absence(&logical)
            .await
            .expect("fresh corrected observation");
        assert_eq!(current.is_none(), mutation == "target");
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_fake_fs_logical_absence_revalidates_held_route(cx: &mut TestAppContext) {
    for mutation in ["alias", "same-target-alias", "parent", "target"] {
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree("/source", json!({ "nested": {}, "replacement": {} }))
            .await;
        filesystem
            .insert_symlink("/source/alias", PathBuf::from("nested"))
            .await;
        let logical = PathBuf::from("/source/alias/missing");
        let (reached, resume) = filesystem
            .pause_path_absence_for_test()
            .expect("first lookup checkpoint");
        let mut operation = cx.executor().spawn({
            let filesystem = filesystem.clone();
            let logical = logical.clone();
            async move { filesystem.observe_path_absence(&logical).await }
        });
        require_native_checkpoint(reached, &mut operation, "coherent FakeFs missing lookup").await;
        match mutation {
            "alias" => {
                filesystem
                    .insert_symlink("/source/alias", PathBuf::from("replacement"))
                    .await
            }
            "same-target-alias" => {
                filesystem
                    .insert_symlink("/source/alias", PathBuf::from("nested"))
                    .await
            }
            "parent" => {
                filesystem
                    .rename(
                        Path::new("/source/nested"),
                        Path::new("/source/retained"),
                        RenameOptions::default(),
                    )
                    .await
                    .expect("retain old parent");
                filesystem
                    .create_dir(Path::new("/source/nested"))
                    .await
                    .expect("new parent");
            }
            "target" => filesystem.insert_file(&logical, b"appeared".to_vec()).await,
            _ => unreachable!(),
        }
        resume
            .send(())
            .expect("release after actual mutation without held state lock");
        assert!(operation.await.is_err(), "{mutation}");
        assert_eq!(
            filesystem
                .observe_path_absence(&logical)
                .await
                .expect("current retry")
                .is_none(),
            mutation == "target"
        );
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_logical_absence_native_names_and_nonregular_objects(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use std::{
        ffi::{CString, OsString},
        os::unix::{
            ffi::{OsStrExt, OsStringExt},
            net::UnixListener,
        },
    };
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("native logical fixture");
    let root = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
    let filesystem = RealFs::new(None, executor);
    #[cfg(target_os = "macos")]
    let name = OsString::from("设置-λ");
    #[cfg(not(target_os = "macos"))]
    let name = OsString::from_vec(vec![b'n', b'a', b'm', b'e', 0xff]);
    let logical = root.join(&name).join("missing");
    std::fs::create_dir(root.join(name)).expect("representable native directory");
    #[cfg(target_os = "macos")]
    {
        // APFS rejects creation with EILSEQ even when lookup reports a missing name.
        let invalid_name = root.join(OsString::from_vec(vec![b'n', b'a', b'm', b'e', 0xff]));
        let error = filesystem
            .observe_path_absence(&invalid_name)
            .await
            .expect_err("unsupported encoding must not produce an absence receipt");
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .expect("explicit unsupported encoding")
                .kind(),
            std::io::ErrorKind::Unsupported
        );
    }
    let receipt = filesystem
        .observe_path_absence(&logical)
        .await
        .expect("native lookup")
        .expect("missing original suffix");
    assert_eq!(
        receipt.logical_path().as_os_str().as_bytes(),
        logical.as_os_str().as_bytes()
    );
    assert!(
        filesystem
            .observe_path_absence(&root.join(OsString::from_vec(vec![b'n', 0, b'x'])))
            .await
            .is_err()
    );
    let socket = UnixListener::bind(root.join("socket")).expect("socket fixture");
    let fifo = CString::new(root.join("fifo").as_os_str().as_bytes()).expect("NUL-free fixture");
    // The CString keeps its terminating NUL alive for the POSIX fixture call.
    assert_eq!(
        unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) },
        0,
        "fifo fixture: {}",
        std::io::Error::last_os_error()
    );
    for suffix in ["socket", "fifo"] {
        assert!(
            filesystem
                .observe_path_absence(&root.join(suffix))
                .await
                .expect("existing special object")
                .is_none()
        );
        assert!(
            filesystem
                .observe_path_absence(&root.join(suffix).join("child"))
                .await
                .is_err()
        );
    }
    for unsupported in [
        "/proc/self/fd/999999/source",
        "/dev/fd/999999/source",
        "/sys/unknown-source",
    ] {
        assert!(
            filesystem
                .observe_path_absence(Path::new(unsupported))
                .await
                .is_err()
        );
    }
    assert!(
        filesystem
            .observe_path_absence(&root.join("x".repeat(4097)))
            .await
            .is_err()
    );
    drop(socket);
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_logical_absence_releases_handles_on_cancel_and_completion(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("logical lifetime fixture");
    let root = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
    std::fs::create_dir(root.join("nested")).expect("target directory");
    std::os::unix::fs::symlink("nested", root.join("alias")).expect("finite alias");
    let filesystem = Arc::new(RealFs::new(None, executor.clone()));
    let logical = root.join("alias/missing");
    for cancel in [true, false] {
        let (reached, resume) = filesystem
            .pause_path_absence_for_test()
            .expect("hold observation");
        let mut operation = executor.spawn({
            let filesystem = filesystem.clone();
            let logical = logical.clone();
            async move { filesystem.observe_path_absence(&logical).await }
        });
        require_native_checkpoint(reached, &mut operation, "actual retained logical handles").await;
        let handles = filesystem.path_absence_handles_for_test();
        assert!(!handles.is_empty());
        assert!(handles.iter().all(|handle| handle.upgrade().is_some()));
        if cancel {
            drop(operation);
            cx.run_until_parked();
            assert!(
                resume.send(()).is_err(),
                "cancelled observation releases its pause"
            );
        } else {
            resume.send(()).expect("release successful observation");
            assert!(operation.await.expect("completed observation").is_some());
        }
        assert!(
            handles.iter().all(|handle| handle.upgrade().is_none()),
            "all captured directory and alias owners must drop"
        );
        assert!(
            filesystem
                .observe_path_absence(&logical)
                .await
                .expect("subsequent observation")
                .is_some()
        );
    }
}

#[cfg(unix)]
#[gpui::test]
async fn test_real_fs_logical_absence_native_permission_denial(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use std::os::unix::fs::PermissionsExt;
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("permission fixture");
    let root = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
    let guarded = root.join("guarded");
    std::fs::create_dir(&guarded).expect("guarded directory");
    let filesystem = RealFs::new(None, executor);
    std::fs::set_permissions(&guarded, std::fs::Permissions::from_mode(0o000))
        .expect("deny own disposable directory");
    let native_read = std::fs::read_to_string(guarded.join("missing"));
    let observation = filesystem
        .observe_path_absence(&guarded.join("missing"))
        .await;
    std::fs::set_permissions(&guarded, std::fs::Permissions::from_mode(0o700))
        .expect("restore fixture access before assertions");
    assert_eq!(
        native_read.expect_err("actual native denial").kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert!(
        observation.is_err(),
        "permission denial must not become a missing receipt"
    );
    assert!(
        filesystem
            .observe_path_absence(&guarded.join("missing"))
            .await
            .expect("permission correction")
            .is_some()
    );
}

#[cfg(target_os = "macos")]
#[gpui::test]
async fn test_real_fs_logical_absence_native_acl_denial(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let fixture = TempDir::new().expect("ACL fixture");
    let root = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
    let guarded = root.join("guarded");
    std::fs::create_dir(&guarded).expect("guarded directory");
    let deny = smol::process::Command::new("/bin/chmod")
        .args(["+a#", "0", "everyone deny read,search"])
        .arg(&guarded)
        .output()
        .await
        .expect("install fixture ACL");
    assert!(
        deny.status.success(),
        "ACL installation: {}",
        String::from_utf8_lossy(&deny.stderr)
    );
    let filesystem = RealFs::new(None, executor);
    let native_read = std::fs::read_to_string(guarded.join("missing"));
    let observation = filesystem
        .observe_path_absence(&guarded.join("missing"))
        .await;
    let restore = smol::process::Command::new("/bin/chmod")
        .args(["-a#", "0"])
        .arg(&guarded)
        .output()
        .await
        .expect("remove only inserted fixture ACL");
    assert!(
        restore.status.success(),
        "ACL removal: {}",
        String::from_utf8_lossy(&restore.stderr)
    );
    assert_eq!(
        native_read.expect_err("actual ACL denial").kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert!(observation.is_err(), "native ACL denial is not absence");
    assert!(
        filesystem
            .observe_path_absence(&guarded.join("missing"))
            .await
            .expect("ACL correction")
            .is_some()
    );
}

async fn assert_copy_overwrite_semantics(root: &Path, fs: &dyn Fs) {
    let source = root.join("source.txt");
    let target = root.join("target.txt");
    #[cfg(unix)]
    let original = fs
        .metadata(&target)
        .await
        .expect("target metadata")
        .expect("target exists");
    fs.copy_file(
        &source,
        &target,
        CopyOptions {
            overwrite: true,
            ignore_if_exists: false,
        },
    )
    .await
    .expect("overwrite target");
    assert_eq!(fs.load(&target).await.expect("copied bytes"), "from source");
    #[cfg(unix)]
    assert_eq!(
        fs.metadata(&target)
            .await
            .expect("copied metadata")
            .expect("target remains")
            .inode,
        original.inode
    );
}

async fn assert_copy_options_semantics(root: &Path, fs: &dyn Fs) {
    let source = root.join("source.txt");
    let target = root.join("target.txt");
    assert!(
        fs.copy_file(&source, &target, CopyOptions::default())
            .await
            .is_err()
    );
    fs.copy_file(
        &source,
        &target,
        CopyOptions {
            overwrite: false,
            ignore_if_exists: true,
        },
    )
    .await
    .expect("ignore existing target");
    assert_eq!(
        fs.load(&target).await.expect("untouched target"),
        "from target"
    );
    assert!(
        fs.copy_file(
            &source,
            &root.join("dir"),
            CopyOptions {
                overwrite: true,
                ignore_if_exists: false
            }
        )
        .await
        .is_err()
    );
    assert!(fs.is_dir(&root.join("dir")).await);
    assert_eq!(
        fs.load(&root.join("dir/inner.txt"))
            .await
            .expect("directory contents remain"),
        "inner"
    );
    fs.copy_file(&source, &root.join("new.txt"), CopyOptions::default())
        .await
        .expect("new target");
    assert_eq!(
        fs.load(&root.join("new.txt")).await.expect("new bytes"),
        "from source"
    );
}

async fn assert_remove_file_unlinks_aliases(root: &Path, fs: &dyn Fs) {
    for (name, target) in [
        ("dir-link", "dir"),
        ("file-link", "source.txt"),
        ("dangling-link", "missing"),
    ] {
        let link = root.join(name);
        fs.create_symlink(&link, root.join(target))
            .await
            .expect("create fixture alias");
        fs.remove_file(&link, RemoveOptions::default())
            .await
            .expect("unlink alias");
        assert!(
            fs.read_link(&link).await.is_err(),
            "alias must be removed, including dangling aliases"
        );
        assert!(
            fs.metadata(&link)
                .await
                .expect("removed link metadata")
                .is_none()
        );
    }
    assert_eq!(
        fs.load(&root.join("source.txt"))
            .await
            .expect("file target remains"),
        "from source"
    );
    assert_eq!(
        fs.load(&root.join("dir/inner.txt"))
            .await
            .expect("directory target remains"),
        "inner"
    );
    assert!(
        fs.remove_file(&root.join("dir"), RemoveOptions::default())
            .await
            .is_err()
    );
}

async fn copy_remove_fixture(fs: &FakeFs) {
    fs.insert_tree(
        path!("/root"),
        json!({"source.txt":"from source","target.txt":"from target","dir":{"inner.txt":"inner"}}),
    )
    .await;
}

#[gpui::test]
async fn test_fake_fs_copy_overwrite_semantics(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    copy_remove_fixture(&fs).await;
    assert_copy_overwrite_semantics(Path::new(path!("/root")), fs.as_ref()).await;
}

#[gpui::test]
async fn test_fake_fs_copy_options_semantics(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    copy_remove_fixture(&fs).await;
    assert_copy_options_semantics(Path::new(path!("/root")), fs.as_ref()).await;
}

#[gpui::test]
async fn test_fake_fs_copy_ignore_emits_no_event(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    copy_remove_fixture(&fs).await;
    let root = Path::new(path!("/root"));
    let (mut events, _watcher) = fs.watch(root, Duration::ZERO).await;
    fs.copy_file(
        &root.join("source.txt"),
        &root.join("target.txt"),
        CopyOptions {
            overwrite: false,
            ignore_if_exists: true,
        },
    )
    .await
    .expect("ignore existing");
    fs.create_file(&root.join("sentinel"), CreateOptions::default())
        .await
        .expect("create event sentinel");
    let next = events.next().await.expect("sentinel event");
    assert_eq!(next.len(), 1);
    assert_eq!(next.first().expect("one event").path, root.join("sentinel"));
}

#[gpui::test]
async fn test_fake_fs_remove_file_unlinks_aliases(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    copy_remove_fixture(&fs).await;
    assert_remove_file_unlinks_aliases(Path::new(path!("/root")), fs.as_ref()).await;
}

#[gpui::test]
async fn test_realfs_copy_and_remove_semantics(cx: &mut TestAppContext) {
    cx.executor().allow_parking();
    let directory = TempDir::new().expect("temporary fixture");
    let root = directory.path();
    std::fs::write(root.join("source.txt"), "from source").expect("source");
    std::fs::write(root.join("target.txt"), "from target").expect("target");
    std::fs::create_dir(root.join("dir")).expect("directory");
    std::fs::write(root.join("dir/inner.txt"), "inner").expect("directory content");
    let fs = RealFs::new(None, cx.executor());
    assert_copy_options_semantics(root, &fs).await;
    assert_copy_overwrite_semantics(root, &fs).await;
    #[cfg(unix)]
    assert_remove_file_unlinks_aliases(root, &fs).await;
}

async fn assert_copy_follows_destination_aliases(root: &Path, fs: &dyn Fs) {
    let source = root.join("source.txt");
    let alias = root.join("alias");
    fs.create_symlink(&alias, PathBuf::from("target.txt"))
        .await
        .expect("relative target alias");
    assert!(
        fs.copy_file(&source, &alias, CopyOptions::default())
            .await
            .is_err()
    );
    fs.copy_file(
        &source,
        &alias,
        CopyOptions {
            overwrite: false,
            ignore_if_exists: true,
        },
    )
    .await
    .expect("ignore existing alias target");
    assert_eq!(
        fs.load(&root.join("target.txt"))
            .await
            .expect("unchanged target"),
        "from target"
    );
    fs.copy_file(
        &source,
        &alias,
        CopyOptions {
            overwrite: true,
            ignore_if_exists: false,
        },
    )
    .await
    .expect("overwrite alias target");
    assert_eq!(
        fs.read_link(&alias).await.expect("alias retained"),
        PathBuf::from("target.txt")
    );
    assert_eq!(
        fs.load(&root.join("target.txt"))
            .await
            .expect("changed target"),
        "from source"
    );
    let dangling = root.join("dangling");
    fs.create_symlink(&dangling, PathBuf::from("created-via-alias.txt"))
        .await
        .expect("dangling alias");
    fs.copy_file(&source, &dangling, CopyOptions::default())
        .await
        .expect("create through dangling alias");
    assert_eq!(
        fs.read_link(&dangling)
            .await
            .expect("dangling alias retained"),
        PathBuf::from("created-via-alias.txt")
    );
    assert_eq!(
        fs.load(&root.join("created-via-alias.txt"))
            .await
            .expect("created target"),
        "from source"
    );
    fs.create_symlink(&root.join("chain"), PathBuf::from("alias"))
        .await
        .expect("alias chain");
    fs.copy_file(
        &source,
        &root.join("chain"),
        CopyOptions {
            overwrite: true,
            ignore_if_exists: false,
        },
    )
    .await
    .expect("copy through alias chain");
    assert_eq!(
        fs.read_link(&root.join("chain"))
            .await
            .expect("chain retained"),
        PathBuf::from("alias")
    );
    fs.create_symlink(&root.join("cycle-a"), PathBuf::from("cycle-b"))
        .await
        .expect("first cycle alias");
    fs.create_symlink(&root.join("cycle-b"), PathBuf::from("cycle-a"))
        .await
        .expect("second cycle alias");
    assert!(
        fs.copy_file(
            &source,
            &root.join("cycle-a"),
            CopyOptions {
                overwrite: true,
                ignore_if_exists: false
            }
        )
        .await
        .is_err()
    );
    assert_eq!(
        fs.load(&source).await.expect("source remains"),
        "from source"
    );
}

#[gpui::test]
async fn test_fake_fs_copy_destination_aliases(executor: BackgroundExecutor) {
    let fs = FakeFs::new(executor);
    copy_remove_fixture(&fs).await;
    assert_copy_follows_destination_aliases(Path::new(path!("/root")), fs.as_ref()).await;
}

#[cfg(unix)]
#[gpui::test]
async fn test_realfs_copy_destination_aliases(cx: &mut TestAppContext) {
    cx.executor().allow_parking();
    let directory = TempDir::new().expect("temporary fixture");
    let root = directory.path();
    std::fs::write(root.join("source.txt"), "from source").expect("source");
    std::fs::write(root.join("target.txt"), "from target").expect("target");
    let fs = RealFs::new(None, cx.executor());
    assert_copy_follows_destination_aliases(root, &fs).await;
}

#[cfg(target_os = "macos")]
#[gpui::test]
async fn test_realfs_macos_system_alias_watch_delivery(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    let directory = tempfile::Builder::new()
        .prefix("copper-system-alias-watch-")
        .tempdir_in("/tmp")
        .expect("disposable native alias fixture");
    let alias_root = directory.path();
    let canonical_root = std::fs::canonicalize(alias_root).expect("canonical native root");
    assert!(canonical_root.starts_with("/private/tmp"));
    for name in ["context-a.rs", "context-b.rs", "deleted-settings.json"] {
        std::fs::write(alias_root.join(name), b"before").expect("initial fixture file");
    }
    let filesystem = RealFs::new(None, executor.clone());
    let (mut events, _root_watcher) = filesystem
        .watch(&canonical_root, Duration::from_millis(10))
        .await;
    let (_first_events, _first_alias_watcher) = filesystem
        .watch(&alias_root.join("context-a.rs"), Duration::from_millis(10))
        .await;
    let (_second_events, _second_alias_watcher) = filesystem
        .watch(&alias_root.join("context-b.rs"), Duration::from_millis(10))
        .await;
    executor.timer(Duration::from_millis(250)).await;
    for name in [
        "written-settings.json",
        "deleted-settings.json",
        "atomic-settings.json",
    ] {
        match name {
            "written-settings.json" => {
                std::fs::write(alias_root.join(name), b"after").expect("write through system alias")
            }
            "deleted-settings.json" => std::fs::remove_file(canonical_root.join(name))
                .expect("remove canonical fixture file"),
            _ => {
                let temporary = alias_root.join("atomic-settings.new");
                std::fs::write(&temporary, b"atomic").expect("atomic fixture content");
                std::fs::rename(temporary, alias_root.join(name))
                    .expect("atomic alias replacement");
            }
        }
        let expected = canonical_root.join(name);
        let timeout = executor.timer(Duration::from_secs(10)).fuse();
        futures::pin_mut!(timeout);
        let delivered = loop {
            futures::select_biased! {
                batch = events.next().fuse() => {
                    let Some(batch) = batch else { break false };
                    // A broad rescan cannot prove delivery of this particular native event.
                    if batch.iter().any(|event| event.path == expected && event.kind != Some(PathEventKind::Rescan)) {
                        break true;
                    }
                }
                _ = timeout => break false,
            }
        };
        assert!(delivered, "actual native event missing for {name}");
    }
    assert_eq!(
        std::fs::read(canonical_root.join("written-settings.json")).expect("written bytes"),
        b"after"
    );
    assert_eq!(
        std::fs::read(canonical_root.join("atomic-settings.json")).expect("atomic bytes"),
        b"atomic"
    );
    assert!(!canonical_root.join("deleted-settings.json").exists());
}
