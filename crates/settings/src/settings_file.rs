use crate::{settings_content::SettingsContent, settings_store::SettingsStore};
use collections::HashSet;
use fs::{Fs, PathEventKind};
use futures::{StreamExt, channel::mpsc};
use gpui::{App, BackgroundExecutor, ReadGlobal};
use std::{path::PathBuf, sync::Arc, time::Duration};

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;

    use gpui::TestAppContext;
    use serde_json::json;
    use std::path::Path;

    #[gpui::test]
    async fn test_watch_config_dir_reloads_tracked_file_on_rescan(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        let fs = FakeFs::new(cx.background_executor.clone());
        let config_dir = PathBuf::from("/root/config");
        let settings_path = PathBuf::from("/root/config/settings.json");

        fs.insert_tree(
            Path::new("/root"),
            json!({
                "config": {
                    "settings.json": "A"
                }
            }),
        )
        .await;

        let mut rx = watch_config_dir(
            &cx.background_executor,
            fs.clone(),
            config_dir.clone(),
            HashSet::from_iter([settings_path.clone()]),
        );

        assert_eq!(rx.next().await.as_deref(), Some("A"));
        cx.run_until_parked();

        fs.pause_events();
        fs.insert_file(&settings_path, b"B".to_vec()).await;
        fs.clear_buffered_events();

        fs.emit_fs_event(&settings_path, Some(PathEventKind::Rescan));
        fs.unpause_events_and_flush();
        assert_eq!(rx.next().await.as_deref(), Some("B"));

        fs.pause_events();
        fs.insert_file(&settings_path, b"A".to_vec()).await;
        fs.clear_buffered_events();

        fs.emit_fs_event(&config_dir, Some(PathEventKind::Rescan));
        fs.unpause_events_and_flush();
        assert_eq!(rx.next().await.as_deref(), Some("A"));
    }

    #[gpui::test]
    async fn test_watch_config_file_reloads_when_parent_dir_is_symlink(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let config_settings_path = PathBuf::from("/root/.config/zed/settings.json");
        let target_settings_path = PathBuf::from("/root/dotfiles/zed/settings.json");

        fs.insert_tree(
            Path::new("/root"),
            json!({
                ".config": {},
                "dotfiles": {
                    "zed": {
                        "settings.json": "A"
                    }
                }
            }),
        )
        .await;

        fs.create_symlink(
            Path::new("/root/.config/zed"),
            PathBuf::from("/root/dotfiles/zed"),
        )
        .await
        .unwrap();

        let (mut rx, _task) =
            watch_config_file(&cx.background_executor, fs.clone(), config_settings_path);
        assert_eq!(rx.next().await.as_deref(), Some("A"));

        fs.insert_file(&target_settings_path, b"B".to_vec()).await;
        assert_eq!(rx.next().await.as_deref(), Some("B"));
    }
    #[test]
    fn config_read_debug_does_not_expose_settings_or_paths() {
        let path = PathBuf::from("/private/config-path-canary/settings.json");
        let mut current = CurrentConfigFileRead::default();
        let identity = current.begin(&path);
        let events = [
            ConfigFileReadEvent::Pending {
                identity: identity.clone(),
            },
            ConfigFileReadEvent::Finished {
                identity: identity.clone(),
                read_path: path.clone(),
                result: Ok("settings-content-canary".to_owned()),
            },
            ConfigFileReadEvent::Finished {
                identity,
                read_path: path,
                result: Err(Arc::new(anyhow::anyhow!("configuration-error-canary"))),
            },
        ];
        for event in events {
            let diagnostic = format!("{event:?}");
            assert!(diagnostic.contains("ConfigFileReadEvent::"));
            for private_value in [
                "config-path-canary",
                "settings-content-canary",
                "configuration-error-canary",
            ] {
                assert!(!diagnostic.contains(private_value), "{diagnostic}");
            }
        }
    }

    async fn next_config_read(
        receiver: &mut mpsc::UnboundedReceiver<ConfigFileReadEvent>,
    ) -> (ConfigFileReadIdentity, PathBuf, ConfigFileReadResult) {
        let identity = match receiver.next().await.expect("pending read event") {
            ConfigFileReadEvent::Pending { identity } => identity,
            ConfigFileReadEvent::Finished { .. } => panic!("finished without preceding pending"),
        };
        match receiver.next().await.expect("finished read event") {
            ConfigFileReadEvent::Finished {
                identity: finished,
                read_path,
                result,
            } => {
                assert_eq!(finished, identity, "same actual read identity");
                (identity, read_path, result)
            }
            ConfigFileReadEvent::Pending { .. } => panic!("second pending before read completion"),
        }
    }

    #[gpui::test]
    async fn test_config_read_events_keep_missing_error_distinct_from_empty_content(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        fs.insert_tree("/root", json!({"empty.json": ""})).await;
        for (name, missing) in [("missing.json", true), ("empty.json", false)] {
            let path = PathBuf::from("/root").join(name);
            let (mut receiver, task) = watch_config_file_with_read_events(
                &cx.background_executor,
                fs.clone(),
                path.clone(),
            );
            let (identity, read_path, result) = next_config_read(&mut receiver).await;
            assert_eq!(identity.configured_path(), path.as_path());
            assert_eq!(read_path, path);
            if missing {
                let error = result.expect_err("actual missing lookup remains an error");
                assert_eq!(
                    error
                        .downcast_ref::<std::io::Error>()
                        .expect("retained FakeFs I/O provenance")
                        .kind(),
                    std::io::ErrorKind::NotFound
                );
            } else {
                assert_eq!(result.expect("actual empty file read"), "");
            }
            assert!(
                identity.is_current(),
                "identity is current, not a permission result"
            );
            assert_eq!(fs.watched_paths(), vec![path]);
            drop(task);
            cx.run_until_parked();
            assert!(!identity.is_current());
            assert!(receiver.next().await.is_none());
            assert!(fs.watched_paths().is_empty());
        }
    }

    #[gpui::test]
    async fn test_config_read_events_keep_later_failure_and_correction(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "A"})).await;
        let (mut receiver, task) =
            watch_config_file_with_read_events(&cx.background_executor, fs.clone(), path.clone());
        let (initial, _, result) = next_config_read(&mut receiver).await;
        assert_eq!(result.expect("initial actual contents"), "A");
        fs.pause_events();
        fs.remove_file(&path, Default::default())
            .await
            .expect("remove actual fixture file");
        fs.clear_buffered_events();
        fs.emit_fs_event(&path, Some(PathEventKind::Rescan));
        fs.unpause_events_and_flush();
        let (failed, failed_path, result) = next_config_read(&mut receiver).await;
        assert_ne!(failed, initial);
        assert!(!initial.is_current());
        assert!(failed.is_current());
        assert_eq!(failed_path, path);
        assert!(
            result.is_err(),
            "failed current I/O cannot reuse old good A"
        );
        fs.insert_file(&path, b"C".to_vec()).await;
        let (corrected, _, result) = next_config_read(&mut receiver).await;
        assert_ne!(corrected, failed);
        assert!(!failed.is_current());
        assert_eq!(result.expect("actual corrected read"), "C");
        assert!(corrected.is_current());
        assert_eq!(fs.watched_paths(), vec![path]);
        drop(task);
        cx.run_until_parked();
        assert!(!corrected.is_current());
        assert!(fs.watched_paths().is_empty());
    }

    #[gpui::test]
    async fn test_config_read_events_preserve_utf8_failure(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({})).await;
        fs.insert_file(&path, vec![0xff]).await;
        let (mut receiver, task) =
            watch_config_file_with_read_events(&cx.background_executor, fs.clone(), path.clone());
        let (_, read_path, result) = next_config_read(&mut receiver).await;
        assert_eq!(read_path, path);
        let error = result.expect_err("actual FakeFs UTF8 conversion fails");
        assert!(error.downcast_ref::<std::string::FromUtf8Error>().is_some());
        drop(task);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn test_config_read_new_begin_invalidates_queued_old_finished(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "A"})).await;
        let (mut receiver, task) =
            watch_config_file_with_read_events(&cx.background_executor, fs.clone(), path.clone());
        let old = match receiver.next().await.expect("initial pending") {
            ConfigFileReadEvent::Pending { identity } => identity,
            _ => panic!("initial event must be pending"),
        };
        cx.run_until_parked();
        fs.insert_file(&path, b"B".to_vec()).await;
        cx.run_until_parked();
        assert!(
            !old.is_current(),
            "new producer begin invalidates before its pending event is consumed"
        );
        match receiver.next().await.expect("queued old completion") {
            ConfigFileReadEvent::Finished {
                identity, result, ..
            } => {
                assert_eq!(identity, old);
                assert_eq!(result.expect("old actual A read"), "A");
                assert!(
                    !identity.is_current(),
                    "queued old success is not current authority"
                );
            }
            _ => panic!("old completion order retained"),
        }
        let (current, _, result) = next_config_read(&mut receiver).await;
        assert_eq!(result.expect("new actual B read"), "B");
        assert!(current.is_current());
        drop(task);
        cx.run_until_parked();
        assert!(!current.is_current());
    }

    #[gpui::test]
    async fn test_config_read_events_keep_parent_symlink_source_provenance(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let configured_path = PathBuf::from("/root/.config/zed/settings.json");
        let target_path = PathBuf::from("/root/dotfiles/zed/settings.json");
        fs.insert_tree(
            "/root",
            json!({".config": {}, "dotfiles": {"zed": {"settings.json": "A"}}}),
        )
        .await;
        fs.create_symlink(
            Path::new("/root/.config/zed"),
            PathBuf::from("/root/dotfiles/zed"),
        )
        .await
        .expect("ordinary supported dotfiles alias");
        let (mut receiver, task) = watch_config_file_with_read_events(
            &cx.background_executor,
            fs.clone(),
            configured_path.clone(),
        );
        let (initial, read_path, result) = next_config_read(&mut receiver).await;
        assert_eq!(initial.configured_path(), configured_path.as_path());
        assert_eq!(read_path, target_path);
        assert_eq!(result.expect("initial target read"), "A");
        assert_eq!(fs.watched_paths(), vec![target_path.clone()]);
        fs.insert_file(&target_path, b"B".to_vec()).await;
        let (later, read_path, result) = next_config_read(&mut receiver).await;
        assert!(!initial.is_current());
        assert_eq!(later.configured_path(), configured_path.as_path());
        assert_eq!(read_path, target_path);
        assert_eq!(result.expect("changed target read"), "B");
        drop(task);
        cx.run_until_parked();
        assert!(!later.is_current());
    }

    #[gpui::test]
    async fn test_config_read_string_adapter_keeps_fallback_and_omits_later_error(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({})).await;
        let (mut receiver, task) =
            watch_config_file(&cx.background_executor, fs.clone(), path.clone());
        assert_eq!(
            receiver.next().await.as_deref(),
            Some(""),
            "legacy initial failure fallback"
        );
        fs.insert_file(&path, b"A".to_vec()).await;
        assert_eq!(receiver.next().await.as_deref(), Some("A"));
        fs.remove_file(&path, Default::default())
            .await
            .expect("later actual read failure");
        cx.run_until_parked();
        fs.insert_file(&path, b"C".to_vec()).await;
        assert_eq!(
            receiver.next().await.as_deref(),
            Some("C"),
            "positive correction barrier proves no hidden empty fallback from later error"
        );
        assert_eq!(fs.watched_paths(), vec![path]);
        drop(task);
        cx.run_until_parked();
        assert!(receiver.next().await.is_none());
        assert!(fs.watched_paths().is_empty());
    }

    #[gpui::test]
    async fn test_config_read_errors_do_not_classify_directory_or_alias_as_absent(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        fs.insert_tree("/root", json!({"directory": {}, "ordinary": "file"}))
            .await;
        fs.insert_symlink("/root/dangling.json", PathBuf::from("/missing/target.json"))
            .await;
        for path in [
            PathBuf::from("/root/directory"),
            PathBuf::from("/root/missing-parent/settings.json"),
            PathBuf::from("/root/ordinary/settings.json"),
            PathBuf::from("/root/dangling.json"),
        ] {
            let (mut receiver, task) = watch_config_file_with_read_events(
                &cx.background_executor,
                fs.clone(),
                path.clone(),
            );
            let (identity, _, result) = next_config_read(&mut receiver).await;
            assert_eq!(identity.configured_path(), path.as_path());
            assert!(
                result.is_err(),
                "non-file and unresolved source remain actual read errors: {path:?}"
            );
            assert!(
                identity.is_current(),
                "current identity never changes error into absence or permission"
            );
            assert_eq!(fs.watched_paths().len(), 1);
            drop(task);
            cx.run_until_parked();
            assert!(!identity.is_current());
            assert!(receiver.next().await.is_none());
            assert!(fs.watched_paths().is_empty());
        }
    }
    #[gpui::test]
    async fn test_config_read_cycles_remain_errors_and_release_the_worker(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        fs.insert_tree("/root", json!({"settings.json": "ordinary"}))
            .await;
        for (name, target) in [
            ("self", "self"),
            ("pair-a", "pair-b"),
            ("pair-b", "pair-a"),
            ("expanding", "expanding/suffix"),
        ] {
            fs.create_symlink(&PathBuf::from("/root").join(name), PathBuf::from(target))
                .await
                .expect("create cyclic fixture alias");
        }
        for name in ["self", "pair-a", "expanding"] {
            let path = PathBuf::from("/root").join(name);
            let (mut receiver, task) = watch_config_file_with_read_events(
                &cx.background_executor,
                fs.clone(),
                path.clone(),
            );
            let (identity, read_path, result) = next_config_read(&mut receiver).await;
            assert_eq!(identity.configured_path(), path.as_path());
            assert_eq!(read_path, path);
            let error = result.expect_err("cyclic sources never become absence or defaults");
            assert!(format!("{error:#}").contains("symlink cycle"));
            assert!(
                error
                    .chain()
                    .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
                    .all(|error| error.kind() != std::io::ErrorKind::NotFound)
            );
            assert_eq!(fs.watched_paths().len(), 1);
            drop(task);
            cx.run_until_parked();
            assert!(!identity.is_current());
            assert!(receiver.next().await.is_none());
            assert!(fs.watched_paths().is_empty());
        }
        let (mut receiver, task) = watch_config_file_with_read_events(
            &cx.background_executor,
            fs.clone(),
            PathBuf::from("/root/settings.json"),
        );
        let (_, _, result) = next_config_read(&mut receiver).await;
        assert_eq!(
            result.expect("same Fs remains usable after cycle errors"),
            "ordinary"
        );
        drop(task);
        cx.run_until_parked();
        assert!(receiver.next().await.is_none());
        assert!(fs.watched_paths().is_empty());
    }

    fn config_read_barrier(
        injected_error: Option<std::io::ErrorKind>,
    ) -> (
        ConfigFileReadBarrier,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (entered_sender, entered_receiver) = futures::channel::oneshot::channel();
        let (resume_sender, resume_receiver) = futures::channel::oneshot::channel();
        (
            ConfigFileReadBarrier {
                entered: entered_sender,
                resume: resume_receiver,
                injected_error,
            },
            entered_receiver,
            resume_sender,
        )
    }

    async fn pending_config_read(
        receiver: &mut mpsc::UnboundedReceiver<ConfigFileReadEvent>,
    ) -> ConfigFileReadIdentity {
        match receiver.next().await.expect("pending actual worker read") {
            ConfigFileReadEvent::Pending { identity } => identity,
            ConfigFileReadEvent::Finished { .. } => panic!("completion before pending"),
        }
    }

    async fn finished_config_read(
        receiver: &mut mpsc::UnboundedReceiver<ConfigFileReadEvent>,
        expected: &ConfigFileReadIdentity,
    ) -> (PathBuf, ConfigFileReadResult) {
        match receiver.next().await.expect("actual worker completion") {
            ConfigFileReadEvent::Finished {
                identity,
                read_path,
                result,
            } => {
                assert_eq!(&identity, expected);
                (read_path, result)
            }
            ConfigFileReadEvent::Pending { .. } => panic!("new pending before held read completed"),
        }
    }

    #[gpui::test]
    async fn test_config_read_pending_precedes_held_actual_load(cx: &mut TestAppContext) {
        use futures::FutureExt as _;
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "before"}))
            .await;
        fs.pause_events();
        let (barrier, entered, resume) = config_read_barrier(None);
        let hooks = ConfigFileWatchHooks {
            before_load: Some(barrier),
            ..Default::default()
        };
        let (mut receiver, task) = watch_config_file_with_test_hooks(
            &cx.background_executor,
            fs.clone(),
            path.clone(),
            hooks,
        );
        let identity = pending_config_read(&mut receiver).await;
        entered
            .await
            .expect("same worker reaches its actual before-load barrier");
        assert!(identity.is_current());
        assert_eq!(fs.watched_paths(), vec![path.clone()]);
        assert!(
            receiver.next().now_or_never().is_none(),
            "no Finished while actual read is held"
        );
        fs.insert_file(&path, b"after".to_vec()).await;
        fs.clear_buffered_events();
        fs.unpause_events_and_flush();
        resume.send(()).expect("resume same live worker");
        let (read_path, result) = finished_config_read(&mut receiver, &identity).await;
        assert_eq!(read_path, path);
        assert_eq!(
            result.expect("actual Fs load after release"),
            "after",
            "not a preloaded before value"
        );
        assert!(identity.is_current());
        drop(task);
        cx.run_until_parked();
        assert!(!identity.is_current());
        assert!(receiver.next().await.is_none());
        assert!(fs.watched_paths().is_empty());
    }

    #[gpui::test]
    async fn test_config_read_cancellation_while_held_has_no_finished_or_watch(
        cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "available"}))
            .await;
        let (barrier, entered, resume) = config_read_barrier(None);
        let hooks = ConfigFileWatchHooks {
            before_load: Some(barrier),
            ..Default::default()
        };
        let (mut receiver, task) = watch_config_file_with_test_hooks(
            &cx.background_executor,
            fs.clone(),
            path.clone(),
            hooks,
        );
        let identity = pending_config_read(&mut receiver).await;
        entered
            .await
            .expect("actual worker parked after Pending and watch registration");
        assert!(identity.is_current());
        assert_eq!(fs.watched_paths(), vec![path.clone()]);
        assert!(receiver.next().now_or_never().is_none());
        drop(task);
        cx.run_until_parked();
        assert!(
            !identity.is_current(),
            "cancelled current read is not retained authority"
        );
        assert!(
            receiver.next().await.is_none(),
            "no Finished before the cancelled stream closes"
        );
        assert!(
            resume.send(()).is_err(),
            "cancelled worker dropped the held read receiver"
        );
        assert!(fs.watched_paths().is_empty());
        fs.insert_file(&path, b"later".to_vec()).await;
        cx.run_until_parked();
        assert!(
            receiver.next().await.is_none(),
            "later event cannot revive cancelled worker"
        );
        assert!(fs.watched_paths().is_empty());
    }

    #[gpui::test]
    async fn test_config_read_controlled_stream_eof_invalidates_after_actual_read(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "actual"}))
            .await;
        let (end_sender, end_receiver) = futures::channel::oneshot::channel();
        let hooks = ConfigFileWatchHooks {
            end_events: Some(end_receiver),
            ..Default::default()
        };
        let (mut receiver, task) = watch_config_file_with_test_hooks(
            &cx.background_executor,
            fs.clone(),
            path.clone(),
            hooks,
        );
        let (identity, read_path, result) = next_config_read(&mut receiver).await;
        assert_eq!(read_path, path);
        assert_eq!(result.expect("real initial Fs load"), "actual");
        assert!(identity.is_current());
        assert_eq!(fs.watched_paths(), vec![path]);
        end_sender
            .send(())
            .expect("explicitly terminate wrapped actual event stream");
        task.await;
        assert!(
            !identity.is_current(),
            "EOF returns from the real worker loop with Task still owned"
        );
        assert!(receiver.next().await.is_none());
        assert!(fs.watched_paths().is_empty());
    }

    #[gpui::test]
    async fn test_config_read_injected_permission_error_preserves_provenance_and_recovers(
        cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        cx.executor().allow_parking();
        let fs = FakeFs::new(cx.background_executor.clone());
        let path = PathBuf::from("/root/settings.json");
        fs.insert_tree("/root", json!({"settings.json": "actually-readable"}))
            .await;
        let (barrier, entered, resume) =
            config_read_barrier(Some(std::io::ErrorKind::PermissionDenied));
        let hooks = ConfigFileWatchHooks {
            before_load: Some(barrier),
            ..Default::default()
        };
        let (mut receiver, task) = watch_config_file_with_test_hooks(
            &cx.background_executor,
            fs.clone(),
            path.clone(),
            hooks,
        );
        let failed = pending_config_read(&mut receiver).await;
        entered
            .await
            .expect("same worker held at controlled test I/O boundary");
        assert!(receiver.next().now_or_never().is_none());
        resume.send(()).expect("release controlled failure");
        let (read_path, result) = finished_config_read(&mut receiver, &failed).await;
        assert_eq!(read_path, path);
        let error = result.expect_err("injected test I/O must not become empty successful String");
        let error = error
            .downcast_ref::<std::io::Error>()
            .expect("original typed I/O provenance survives Arc and event");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            error.to_string(),
            "controlled watched-configuration read failure"
        );
        assert!(
            failed.is_current(),
            "current failed read is not permission or readiness"
        );
        assert_eq!(
            fs.load(&path)
                .await
                .expect("control proving this is not native/FakeFs ACL evidence"),
            "actually-readable"
        );
        assert_eq!(fs.watched_paths(), vec![path.clone()]);
        fs.insert_file(&path, b"corrected".to_vec()).await;
        let (corrected, read_path, result) = next_config_read(&mut receiver).await;
        assert_eq!(read_path, path);
        assert!(!failed.is_current());
        assert_ne!(failed, corrected);
        assert_eq!(
            result.expect("actual later Fs read in same worker"),
            "corrected"
        );
        assert!(corrected.is_current());
        assert_eq!(fs.watched_paths(), vec![path]);
        drop(task);
        cx.run_until_parked();
        assert!(!corrected.is_current());
        assert!(receiver.next().await.is_none());
        assert!(fs.watched_paths().is_empty());
    }
}

pub const EMPTY_THEME_NAME: &str = "empty-theme";

/// Settings for visual tests that use proper fonts instead of Courier.
/// Uses Helvetica Neue for UI (sans-serif) and Menlo for code (monospace),
/// which are available on all macOS systems.
#[cfg(any(test, feature = "test-support"))]
pub fn visual_test_settings() -> String {
    let mut value =
        crate::parse_json_with_comments::<serde_json::Value>(crate::default_settings().as_ref())
            .unwrap();
    util::merge_non_null_json_value_into(
        serde_json::json!({
            "ui_font_family": ".SystemUIFont",
            "ui_font_features": {},
            "ui_font_size": 14,
            "ui_font_fallback": [],
            "buffer_font_family": "Menlo",
            "buffer_font_features": {},
            "buffer_font_size": 14,
            "buffer_font_fallbacks": [],
            "theme": EMPTY_THEME_NAME,
        }),
        &mut value,
    );
    value.as_object_mut().unwrap().remove("languages");
    serde_json::to_string(&value).unwrap()
}

#[cfg(any(test, feature = "test-support"))]
pub fn test_settings() -> &'static str {
    static CACHED: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        let mut value = crate::parse_json_with_comments::<serde_json::Value>(
            crate::default_settings().as_ref(),
        )
        .unwrap();
        #[cfg(not(target_os = "windows"))]
        util::merge_non_null_json_value_into(
            serde_json::json!({
                "format_on_save": "on",
                "ui_font_family": "Courier",
                "ui_font_features": {},
                "ui_font_size": 14,
                "ui_font_fallback": [],
                "buffer_font_family": "Courier",
                "buffer_font_features": {},
                "buffer_font_size": 14,
                "buffer_font_fallbacks": [],
                "theme": EMPTY_THEME_NAME,
            }),
            &mut value,
        );
        #[cfg(target_os = "windows")]
        util::merge_non_null_json_value_into(
            serde_json::json!({
                "format_on_save": "on",
                "ui_font_family": "Courier New",
                "ui_font_features": {},
                "ui_font_size": 14,
                "ui_font_fallback": [],
                "buffer_font_family": "Courier New",
                "buffer_font_features": {},
                "buffer_font_size": 14,
                "buffer_font_fallbacks": [],
                "theme": EMPTY_THEME_NAME,
            }),
            &mut value,
        );
        value.as_object_mut().unwrap().remove("languages");
        serde_json::to_string(&value).unwrap()
    });
    &CACHED
}

#[derive(Clone)]
pub struct ConfigFileReadIdentity(Arc<ConfigFileReadIdentityState>);

struct ConfigFileReadIdentityState {
    configured_path: PathBuf,
    current: std::sync::atomic::AtomicBool,
}

impl ConfigFileReadIdentity {
    pub fn configured_path(&self) -> &std::path::Path {
        &self.0.configured_path
    }

    /// Current identifies the producer read, not successful I/O, parsing or policy authority.
    pub fn is_current(&self) -> bool {
        self.0.current.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn invalidate(&self) {
        self.0
            .current
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl PartialEq for ConfigFileReadIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ConfigFileReadIdentity {}

impl std::fmt::Debug for ConfigFileReadIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfigFileReadIdentity")
            .finish_non_exhaustive()
    }
}

pub type ConfigFileReadResult = Result<String, Arc<anyhow::Error>>;

pub enum ConfigFileReadEvent {
    Pending {
        identity: ConfigFileReadIdentity,
    },
    Finished {
        identity: ConfigFileReadIdentity,
        read_path: PathBuf,
        result: ConfigFileReadResult,
    },
}

impl std::fmt::Debug for ConfigFileReadEvent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending { identity } => formatter
                .debug_struct("ConfigFileReadEvent::Pending")
                .field("identity", identity)
                .finish_non_exhaustive(),
            Self::Finished {
                identity, result, ..
            } => formatter
                .debug_struct("ConfigFileReadEvent::Finished")
                .field("identity", identity)
                .field("read_succeeded", &result.is_ok())
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Default)]
struct CurrentConfigFileRead(Option<ConfigFileReadIdentity>);

impl CurrentConfigFileRead {
    fn begin(&mut self, configured_path: &std::path::Path) -> ConfigFileReadIdentity {
        if let Some(previous) = self.0.take() {
            previous.invalidate();
        }
        let identity = ConfigFileReadIdentity(Arc::new(ConfigFileReadIdentityState {
            configured_path: configured_path.to_path_buf(),
            current: std::sync::atomic::AtomicBool::new(true),
        }));
        self.0 = Some(identity.clone());
        identity
    }
}

impl Drop for CurrentConfigFileRead {
    fn drop(&mut self) {
        if let Some(identity) = self.0.take() {
            identity.invalidate();
        }
    }
}

pub fn watch_config_file_with_read_events(
    executor: &BackgroundExecutor,
    fs: Arc<dyn Fs>,
    path: PathBuf,
) -> (mpsc::UnboundedReceiver<ConfigFileReadEvent>, gpui::Task<()>) {
    let (sender, receiver) = mpsc::unbounded();
    let task = watch_config_file_worker(
        executor,
        fs,
        path,
        move |event, _initial| match sender.unbounded_send(event) {
            Ok(()) => true,
            Err(_receiver_closed) => false,
        },
        ConfigFileWatchHooks::default(),
    );
    (receiver, task)
}

pub fn watch_config_file(
    executor: &BackgroundExecutor,
    fs: Arc<dyn Fs>,
    path: PathBuf,
) -> (mpsc::UnboundedReceiver<String>, gpui::Task<()>) {
    let (sender, receiver) = mpsc::unbounded();
    let task = watch_config_file_worker(
        executor,
        fs,
        path,
        move |event, initial| {
            let ConfigFileReadEvent::Finished { result, .. } = event else {
                return true;
            };
            let content = match result {
                Ok(content) => content,
                Err(error) => {
                    log::warn!("Could not read watched configuration: {error:#}");
                    if !initial {
                        return true;
                    }
                    String::new()
                }
            };
            match sender.unbounded_send(content) {
                Ok(()) => true,
                Err(_receiver_closed) => false,
            }
        },
        ConfigFileWatchHooks::default(),
    );
    (receiver, task)
}

fn watch_config_file_worker(
    executor: &BackgroundExecutor,
    fs: Arc<dyn Fs>,
    configured_path: PathBuf,
    mut emit: impl FnMut(ConfigFileReadEvent, bool) -> bool + Send + 'static,
    mut test_hooks: ConfigFileWatchHooks,
) -> gpui::Task<()> {
    executor.spawn(async move {
        let mut current = CurrentConfigFileRead::default();
        let identity = current.begin(&configured_path);
        if !emit(ConfigFileReadEvent::Pending { identity: identity.clone() }, true) {
            return;
        }
        let read_path = match fs.canonicalize(&configured_path).await {
            Ok(path) => path,
            Err(error) => {
                log::debug!("Could not canonicalize watched configuration; retaining configured path: {error:#}");
                configured_path.clone()
            }
        };
        let (events, _watcher) = fs.watch(&read_path, Duration::from_millis(100)).await;
        #[cfg(test)]
        let events = match test_hooks.end_events.take() {
            Some(end_events) => events.take_until(async move {
                if let Err(error) = end_events.await {
                    log::debug!("Configuration EOF test control closed: {error}");
                }
            }).boxed(),
            None => events,
        };
        futures::pin_mut!(events);
        let result = read_watched_config_file(&fs, &read_path, &mut test_hooks).await;
        if !emit(ConfigFileReadEvent::Finished { identity, read_path: read_path.clone(), result }, true) {
            return;
        }
        while events.next().await.is_some() {
            let identity = current.begin(&configured_path);
            if !emit(ConfigFileReadEvent::Pending { identity: identity.clone() }, false) {
                return;
            }
            let result = read_watched_config_file(&fs, &read_path, &mut test_hooks).await;
            if !emit(ConfigFileReadEvent::Finished { identity, read_path: read_path.clone(), result }, false) {
                return;
            }
        }
    })
}

#[derive(Default)]
struct ConfigFileWatchHooks {
    #[cfg(test)]
    before_load: Option<ConfigFileReadBarrier>,
    #[cfg(test)]
    end_events: Option<futures::channel::oneshot::Receiver<()>>,
}

#[cfg(test)]
struct ConfigFileReadBarrier {
    entered: futures::channel::oneshot::Sender<()>,
    resume: futures::channel::oneshot::Receiver<()>,
    injected_error: Option<std::io::ErrorKind>,
}

async fn read_watched_config_file(
    fs: &Arc<dyn Fs>,
    read_path: &std::path::Path,
    test_hooks: &mut ConfigFileWatchHooks,
) -> ConfigFileReadResult {
    #[cfg(not(test))]
    let _ = test_hooks;
    #[cfg(test)]
    if let Some(barrier) = test_hooks.before_load.take() {
        if barrier.entered.send(()).is_err() {
            return Err(Arc::new(anyhow::anyhow!(
                "Configuration read test observer closed"
            )));
        }
        if let Err(error) = barrier.resume.await {
            return Err(Arc::new(
                anyhow::Error::new(error).context("Configuration read test barrier closed"),
            ));
        }
        if let Some(error_kind) = barrier.injected_error {
            return Err(Arc::new(anyhow::Error::new(std::io::Error::new(
                error_kind,
                "controlled watched-configuration read failure",
            ))));
        }
    }
    fs.load(read_path).await.map_err(Arc::new)
}

#[cfg(test)]
fn watch_config_file_with_test_hooks(
    executor: &BackgroundExecutor,
    fs: Arc<dyn Fs>,
    path: PathBuf,
    test_hooks: ConfigFileWatchHooks,
) -> (mpsc::UnboundedReceiver<ConfigFileReadEvent>, gpui::Task<()>) {
    let (sender, receiver) = mpsc::unbounded();
    let task = watch_config_file_worker(
        executor,
        fs,
        path,
        move |event, _initial| match sender.unbounded_send(event) {
            Ok(()) => true,
            Err(_receiver_closed) => false,
        },
        test_hooks,
    );
    (receiver, task)
}

pub fn watch_config_dir(
    executor: &BackgroundExecutor,
    fs: Arc<dyn Fs>,
    dir_path: PathBuf,
    config_paths: HashSet<PathBuf>,
) -> mpsc::UnboundedReceiver<String> {
    let (tx, rx) = mpsc::unbounded();
    executor
        .spawn(async move {
            for file_path in &config_paths {
                if fs.metadata(file_path).await.is_ok_and(|v| v.is_some())
                    && let Ok(contents) = fs.load(file_path).await
                    && tx.unbounded_send(contents).is_err()
                {
                    return;
                }
            }

            let (events, _) = fs.watch(&dir_path, Duration::from_millis(100)).await;
            futures::pin_mut!(events);

            while let Some(event_batch) = events.next().await {
                for event in event_batch {
                    if config_paths.contains(&event.path) {
                        match event.kind {
                            Some(PathEventKind::Removed) => {
                                if tx.unbounded_send(String::new()).is_err() {
                                    return;
                                }
                            }
                            Some(PathEventKind::Created) | Some(PathEventKind::Changed) => {
                                if let Ok(contents) = fs.load(&event.path).await
                                    && tx.unbounded_send(contents).is_err()
                                {
                                    return;
                                }
                            }
                            Some(PathEventKind::Rescan) => {
                                for file_path in &config_paths {
                                    if let Ok(contents) = fs.load(file_path).await
                                        && tx.unbounded_send(contents).is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            _ => {}
                        }
                    } else if matches!(event.kind, Some(PathEventKind::Rescan))
                        && event.path == dir_path
                    {
                        for file_path in &config_paths {
                            if let Ok(contents) = fs.load(file_path).await
                                && tx.unbounded_send(contents).is_err()
                            {
                                return;
                            }
                        }
                    }
                }
            }
        })
        .detach();

    rx
}

pub fn update_settings_file(
    fs: Arc<dyn Fs>,
    cx: &App,
    update: impl 'static + Send + FnOnce(&mut SettingsContent, &App),
) {
    SettingsStore::global(cx).update_settings_file(fs, update)
}

pub fn update_settings_file_with_completion(
    fs: Arc<dyn Fs>,
    cx: &App,
    update: impl 'static + Send + FnOnce(&mut SettingsContent, &App),
) -> futures::channel::oneshot::Receiver<anyhow::Result<()>> {
    SettingsStore::global(cx).update_settings_file_with_completion(fs, update)
}
