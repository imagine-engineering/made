use std::{path::Path, sync::Arc};

use fs::FakeFs;
use gpui::{
    BorrowAppContext as _, TestAppContext, VisualContext as _, VisualTestContext, black, px, size,
    white,
};
use project::Project;
use settings::SettingsStore;
use theme::GlobalTheme;

use workspace::{
    AppState, MultiWorkspace, SwitchToCollaborativeWorkspace, SwitchToEditorWorkspace, Workspace,
    WorkspaceId, WorkspacePresentation, collaborative_navigation::CollaborativeNavigationTarget,
};

fn init_test(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        cx.set_global(db::AppDatabase::test_new());
        theme_settings::init(theme::LoadThemes::JustBase, cx);
    });
}

fn settle_collaborative_layout(cx: &mut VisualTestContext) {
    cx.run_until_parked();
    cx.debug_bounds("COLLABORATIVE-LAYOUT");
    cx.refresh()
        .expect("collaborative regression fixture should refresh");
    cx.run_until_parked();
}

#[gpui::test]
async fn collaborative_workspace_reparents_native_status_bar(cx: &mut TestAppContext) {
    init_test(cx);
    let file_system = FakeFs::new(cx.executor());
    let project = Project::test(file_system, [], cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
    let status_bar_id = multi_workspace.read_with(cx, |multi_workspace, cx| {
        multi_workspace
            .workspace()
            .read(cx)
            .status_bar()
            .entity_id()
    });

    cx.dispatch_action(SwitchToCollaborativeWorkspace);
    for width in [1_400., 602., 1_400.] {
        cx.simulate_resize(size(px(width), px(900.)));
        settle_collaborative_layout(cx);
        settle_collaborative_layout(cx);
        let timeline = cx
            .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
            .expect("timeline");
        let footer = cx
            .debug_bounds("WORKSPACE-STATUS-BAR")
            .expect("native status bar");
        let composer = cx.debug_bounds("COLLABORATIVE-COMPOSER").expect("composer");
        assert_eq!(footer.left(), timeline.left());
        assert_eq!(footer.right(), timeline.right());
        assert!(footer.top() >= composer.bottom());
        assert!((footer.bottom() - timeline.bottom()).abs() <= px(1.));
        if let Some(review) = cx.debug_bounds("COLLABORATIVE-REVIEW-REGION") {
            assert_eq!(review.bottom(), timeline.bottom());
        }
    }

    cx.update(|_, cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.status_bar.get_or_insert_default().show = Some(false);
            });
        });
    });
    settle_collaborative_layout(cx);
    assert!(cx.debug_bounds("WORKSPACE-STATUS-BAR").is_none());
    assert!(cx.debug_bounds("COLLABORATIVE-STATUS-BAR").is_none());

    cx.dispatch_action(SwitchToEditorWorkspace);
    cx.update(|_, cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.status_bar.get_or_insert_default().show = Some(true);
            });
        });
    });
    cx.run_until_parked();
    assert!(cx.debug_bounds("COLLABORATIVE-STATUS-BAR").is_none());
    assert!(cx.debug_bounds("WORKSPACE-STATUS-BAR").is_some());
    assert_eq!(
        status_bar_id,
        multi_workspace.read_with(cx, |multi_workspace, cx| {
            multi_workspace
                .workspace()
                .read(cx)
                .status_bar()
                .entity_id()
        })
    );
}

#[gpui::test]
async fn collaborative_status_keeps_execution_details_inside_narrow_footer(
    cx: &mut TestAppContext,
) {
    use workspace::collaborative_participants::{
        CollaborativeExecutionLocation, CollaborativeExecutionPhase, CollaborativeExecutionStatus,
        CollaborativeParticipantProvider, CollaborativeParticipantProviderState,
        CollaborativeParticipantViewData,
    };

    init_test(cx);
    let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    let provider = CollaborativeParticipantProvider::new(
        project,
        workspace.entity_id(),
        CollaborativeParticipantProviderState::Ready(CollaborativeParticipantViewData {
            execution: Some(CollaborativeExecutionStatus {
                phase: CollaborativeExecutionPhase::WaitingForUser,
                model: Some("openai-subscribed/gpt-5.6-sol".into()),
                runtime: Some("Zed Agent".into()),
                location: CollaborativeExecutionLocation::Remote(Some("development-host".into())),
            }),
            ..Default::default()
        }),
    );
    workspace
        .update(cx, |workspace, cx| {
            workspace.register_collaborative_participant_provider(provider, cx)
        })
        .expect("participant provider");
    cx.dispatch_action(SwitchToCollaborativeWorkspace);
    for width in [602., 420., 1_400.] {
        cx.simulate_resize(size(px(width), px(900.)));
        settle_collaborative_layout(cx);
        settle_collaborative_layout(cx);
        let footer = cx
            .debug_bounds("WORKSPACE-STATUS-BAR")
            .expect("native footer");
        let execution = cx
            .debug_bounds("COLLABORATIVE-PARTICIPANT-STATUS")
            .expect("execution details");
        assert!(
            footer.contains(&execution.origin),
            "execution starts outside footer at {width}: {execution:?} / {footer:?}"
        );
        assert!(
            execution.right() <= footer.right(),
            "execution overflows footer at {width}: {execution:?} / {footer:?}"
        );
        assert!(execution.bottom() <= footer.bottom());
    }
}

#[gpui::test]
async fn collaborative_workspace_theme_zoom_and_narrow_window(cx: &mut TestAppContext) {
    init_test(cx);

    let file_system = FakeFs::new(cx.executor());
    let project = Project::test(file_system, [], cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));

    cx.dispatch_action(SwitchToCollaborativeWorkspace);
    cx.simulate_resize(size(px(1_400.), px(900.)));
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);
    assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_some());

    let initial_rem_size = cx.update(|window, _| window.rem_size());
    cx.update(|window, cx| {
        let mut high_contrast = GlobalTheme::theme(cx).as_ref().clone();
        high_contrast.id = "collaborative-high-contrast-test".to_owned();
        high_contrast.name = "Collaborative High Contrast Test".into();
        let colors = &mut high_contrast.styles.colors;
        colors.background = black();
        colors.panel_background = black();
        colors.surface_background = black();
        colors.title_bar_background = black();
        colors.status_bar_background = black();
        colors.border = white();
        colors.border_variant = white();
        colors.text = white();
        colors.text_muted = white().opacity(0.85);
        GlobalTheme::update_theme(cx, Arc::new(high_contrast));
        theme_settings::adjust_ui_font_size(cx, |font_size| font_size + px(4.));
        theme_settings::setup_ui_font(window, cx);
    });
    settle_collaborative_layout(cx);

    cx.update(|window, cx| {
        assert!(window.rem_size() > initial_rem_size);
        let theme = GlobalTheme::theme(cx);
        assert_eq!(theme.name.as_ref(), "Collaborative High Contrast Test");
        assert_eq!(theme.colors().background, black());
        assert_eq!(theme.colors().text, white());
    });
    assert!(multi_workspace.read_with(cx, |multi_workspace, cx| {
        multi_workspace
            .workspace()
            .read(cx)
            .collaborative_title_bar(cx)
            .is_some()
    }));
    for selector in [
        "COLLABORATIVE-TIMELINE-REGION",
        "COLLABORATIVE-REVIEW-REGION",
        "COLLABORATIVE-COMPOSER",
        "COLLABORATIVE-PARTICIPANT-STATUS",
    ] {
        assert!(
            cx.debug_bounds(selector).is_some(),
            "zoomed high-contrast fixture should retain {selector}"
        );
    }

    cx.simulate_resize(size(px(602.), px(640.)));
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);
    assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_none());
    assert_eq!(
        cx.debug_bounds("COLLABORATIVE-TIMELINE-REGION"),
        cx.debug_bounds("COLLABORATIVE-LAYOUT")
    );
    cx.simulate_resize(size(px(1_400.), px(900.)));
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);
    assert!(
        cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_some(),
        "widening should restore a still-requested review pane"
    );
}

#[gpui::test]
async fn collaborative_workspace_restart_restores_presentation_state(cx: &mut TestAppContext) {
    init_test(cx);

    let app_state = cx.update(AppState::test);
    cx.update(|cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.workspace.workspace_presentation =
                    Some(WorkspacePresentation::Collaborative);
            });
        });
    });
    let project = Project::test(app_state.fs.clone(), [], cx).await;
    let workspace_id = WorkspaceId::from_i64(10_700);
    let (workspace, cx) = cx.add_window_view({
        let project = project.clone();
        let app_state = app_state.clone();
        move |window, cx| Workspace::new(Some(workspace_id), project.clone(), app_state, window, cx)
    });

    cx.simulate_resize(size(px(1_400.), px(900.)));
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);
    assert_eq!(
        workspace.read_with(cx, |workspace, _| workspace.workspace_presentation()),
        WorkspacePresentation::Collaborative
    );

    let initial_review_width = cx
        .debug_bounds("COLLABORATIVE-REVIEW-REGION")
        .expect("wide restart fixture should expose review")
        .size
        .width;
    let resize_handle = cx
        .debug_bounds("COLLABORATIVE-REVIEW-RESIZE-HANDLE")
        .expect("wide restart fixture should expose review resizing");
    let drag_start = resize_handle.center();
    let drag_end = gpui::point(drag_start.x - px(72.), drag_start.y);
    cx.simulate_mouse_down(
        drag_start,
        gpui::MouseButton::Left,
        gpui::Modifiers::default(),
    );
    cx.simulate_mouse_move(
        gpui::point(drag_start.x - px(8.), drag_start.y),
        gpui::MouseButton::Left,
        gpui::Modifiers::default(),
    );
    cx.simulate_mouse_move(
        drag_end,
        gpui::MouseButton::Left,
        gpui::Modifiers::default(),
    );
    cx.simulate_mouse_up(
        drag_end,
        gpui::MouseButton::Left,
        gpui::Modifiers::default(),
    );
    settle_collaborative_layout(cx);
    let resized_review_width = cx
        .debug_bounds("COLLABORATIVE-REVIEW-REGION")
        .expect("resized review should remain visible")
        .size
        .width;
    assert!(resized_review_width > initial_review_width + px(60.));

    workspace.update_in(cx, |workspace, window, cx| {
        workspace
            .navigate_collaborative_to(
                CollaborativeNavigationTarget::thread("restart-thread-a"),
                |_| true,
                window,
                cx,
            )
            .expect("first restart target should navigate");
        workspace
            .navigate_collaborative_to(
                CollaborativeNavigationTarget::thread("restart-thread-b"),
                |_| true,
                window,
                cx,
            )
            .expect("second restart target should navigate");
        workspace
            .navigate_collaborative_backward(|_| true, window, cx)
            .expect("restart fixture should retain backward navigation");
    });
    assert!(workspace.read_with(cx, |workspace, cx| {
        workspace.collaborative_title_bar(cx).is_some()
    }));
    workspace.update(cx, |workspace, cx| {
        workspace.toggle_collaborative_review_for_tests(cx);
    });
    settle_collaborative_layout(cx);

    assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_none());

    workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.flush_serialization(window, cx)
        })
        .await;
    cx.run_until_parked();

    let restored = cx.replace_root_view({
        let project = project.clone();
        let app_state = app_state.clone();
        move |window, cx| Workspace::new(Some(workspace_id), project.clone(), app_state, window, cx)
    });
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);

    restored.read_with(cx, |restored, _| {
        assert_eq!(
            restored.workspace_presentation(),
            WorkspacePresentation::Collaborative
        );
        assert_eq!(restored.project().entity_id(), project.entity_id());
        assert_eq!(
            restored.collaborative_navigation().current(),
            Some(&CollaborativeNavigationTarget::thread("restart-thread-a"))
        );
        assert!(!restored.collaborative_navigation().can_go_backward());
        assert!(restored.collaborative_navigation().can_go_forward());
    });
    assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_none());

    assert!(restored.read_with(cx, |workspace, cx| {
        workspace.collaborative_title_bar(cx).is_some()
    }));
    restored.update(cx, |workspace, cx| {
        workspace.toggle_collaborative_review_for_tests(cx);
    });
    settle_collaborative_layout(cx);
    settle_collaborative_layout(cx);
    assert_eq!(
        cx.debug_bounds("COLLABORATIVE-REVIEW-REGION")
            .expect("restored review should expand")
            .size
            .width,
        resized_review_width,
        "restored review should use its persisted resized width"
    );
}

#[test]
fn collaborative_workspace_reduced_motion_and_theme_token_contract() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace crate should live under the repository crates directory");
    for relative_directory in [
        "crates/workspace/src",
        "crates/sidebar/src",
        "crates/agent_ui/src",
    ] {
        for entry in std::fs::read_dir(repository_root.join(relative_directory))
            .unwrap_or_else(|error| panic!("failed to inspect {relative_directory}: {error}"))
        {
            let path = entry
                .expect("collaborative source entry should read")
                .path();
            let is_collaborative_rust = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("collaborative_") && name.ends_with(".rs"));
            if !is_collaborative_rust {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            for forbidden in [".animation(", ".with_animation(", "Animation::"] {
                assert!(
                    !source.contains(forbidden),
                    "{} schedules motion through {forbidden}; add a reduced-motion policy first",
                    path.display()
                );
            }
            for forbidden in ["hsla(", "rgba(", "rgb("] {
                assert!(
                    !source.contains(forbidden),
                    "{} hardcodes {forbidden} instead of a Zed theme token",
                    path.display()
                );
            }
        }
    }
}
