// Allow blocking process commands in this binary - it's a synchronous test runner
#![allow(clippy::disallowed_methods)]

//! Visual Test Runner
//!
//! This binary runs visual regression tests for Zed's UI. It captures screenshots
//! of real Zed windows and compares them against baseline images.
//!
//! **Note: This tool is macOS-only** because it uses `VisualTestAppContext` which
//! depends on the macOS Metal renderer for accurate screenshot capture.
//!
//! ## How It Works
//!
//! This tool uses `VisualTestAppContext` which combines:
//! - Real Metal/compositor rendering for accurate screenshots
//! - Deterministic task scheduling via TestDispatcher
//! - Controllable time via `advance_clock` for testing time-based behaviors
//!
//! This approach:
//! - Does NOT require Screen Recording permission
//! - Does NOT require the window to be visible on screen
//! - Captures raw GPUI output without system window chrome
//! - Is fully deterministic - tooltips, animations, etc. work reliably
//!
//! ## Usage
//!
//! Run the visual tests:
//!   cargo run -p zed --bin zed_visual_test_runner --features visual-tests
//!
//! Update baseline images (when UI intentionally changes):
//!   UPDATE_BASELINE=1 cargo run -p zed --bin zed_visual_test_runner --features visual-tests
//!
//! ## Environment Variables
//!
//!   UPDATE_BASELINE - Set to update baseline images instead of comparing
//!   VISUAL_TEST_OUTPUT_DIR - Directory to save test output (default: target/visual_tests)
//!   COLLABORATIVE_VISUAL_THEME_OVERRIDES - JSON theme overrides for controlled appearance checks
//!   COLLABORATIVE_VISUAL_MULTI_ACP - Render native mode/model/boolean configuration controls

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
mod workspace_status;

// Stub main for non-macOS platforms
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Visual test runner is only supported on macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    // Set ZED_STATELESS early to prevent file system access to real config directories
    // This must be done before any code accesses zed_env_vars::ZED_STATELESS
    // SAFETY: We're at the start of main(), before any threads are spawned
    unsafe {
        std::env::set_var("ZED_STATELESS", "1");
    }

    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let update_baseline = std::env::var("UPDATE_BASELINE").is_ok();

    // Create a temporary directory for test files
    // Canonicalize the path to resolve symlinks (on macOS, /var -> /private/var)
    // which prevents "path does not exist" errors during worktree scanning
    // Use keep() to prevent auto-cleanup - background worktree tasks may still be running
    // when tests complete, so we let the OS clean up temp directories on process exit
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path
        .canonicalize()
        .expect("Failed to canonicalize temp directory");
    // RealFs-backed settings actions need a writable profile even though
    // stateless mode keeps the canonical test databases in memory.
    paths::set_custom_data_dir(&canonical_temp.join("user-data").to_string_lossy());
    std::fs::create_dir_all(paths::config_dir()).expect("Failed to create test settings directory");
    let project_path = canonical_temp.join(if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
        "gpui"
    } else {
        "project"
    });
    std::fs::create_dir_all(&project_path).expect("Failed to create project directory");

    // Create test files in the real filesystem
    create_test_files(&project_path);
    if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
        prepare_collaborative_visual_git_repository(&project_path)
            .expect("Failed to prepare collaborative visual Git repository");
    }

    let test_result = std::panic::catch_unwind(|| run_visual_tests(project_path, update_baseline));

    // Note: We don't delete temp_path here because background worktree tasks may still
    // be running. The directory will be cleaned up when the process exits or by the OS.

    match test_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("Visual tests failed: {}", e);
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Visual tests panicked");
            std::process::exit(1);
        }
    }
}

// All macOS-specific imports grouped together
#[cfg(target_os = "macos")]
use {
    acp_thread::{AgentConnection, StubAgentConnection},
    agent_client_protocol::schema::v1 as acp,
    agent_servers::{AgentServer, AgentServerDelegate},
    anyhow::{Context as _, Result},
    assets::Assets,
    editor::display_map::DisplayRow,
    feature_flags::FeatureFlagAppExt as _,
    git_ui::project_diff::ProjectDiff,
    gpui::{
        App, AppContext as _, BorrowAppContext as _, Bounds, Entity, Modifiers,
        VisualTestAppContext, WindowBounds, WindowHandle, WindowOptions, point, px, size,
    },
    image::{GenericImage as _, RgbaImage},
    project::{AgentId, Project},
    project_panel::ProjectPanel,
    settings::{NotifyWhenAgentWaiting, PlaySoundWhenAgentDone, Settings as _},
    settings_ui::SettingsWindow,
    std::{
        any::Any,
        path::{Path, PathBuf},
        rc::Rc,
        sync::Arc,
        time::Duration,
    },
    util::ResultExt as _,
    workspace::{AppState, MultiWorkspace, Workspace},
    zed_actions::OpenSettingsAt,
};

// All macOS-specific constants grouped together
#[cfg(target_os = "macos")]
mod constants {
    use std::time::Duration;

    /// Baseline images are stored relative to this file
    pub const BASELINE_DIR: &str = "crates/zed/test_fixtures/visual_tests";

    /// Embedded test image (Zed app icon) for visual tests.
    pub const EMBEDDED_TEST_IMAGE: &[u8] = include_bytes!("../resources/app-icon.png");

    /// Threshold for image comparison (0.0 to 1.0)
    /// Images must match at least this percentage to pass
    pub const MATCH_THRESHOLD: f64 = 0.99;

    /// Tooltip show delay - must match TOOLTIP_SHOW_DELAY in gpui/src/elements/div.rs
    pub const TOOLTIP_SHOW_DELAY: Duration = Duration::from_millis(500);
}

#[cfg(target_os = "macos")]
use constants::*;

#[cfg(target_os = "macos")]
fn run_visual_tests(project_path: PathBuf, update_baseline: bool) -> Result<()> {
    // Create the visual test context with deterministic task scheduling
    // Use real Assets so that SVG icons render properly
    let mut cx = VisualTestAppContext::with_asset_source(
        gpui_platform::current_platform(false),
        Arc::new(Assets),
    );

    // Load embedded fonts (IBM Plex Sans, Lilex, etc.) so UI renders with correct fonts
    cx.update(|cx| {
        Assets.load_fonts(cx).unwrap();
    });

    // Initialize settings store with real default settings (not test settings)
    // Test settings use Courier font, but we want the real Zed fonts for visual tests
    cx.update(|cx| {
        settings::init(cx);
    });

    // Create AppState using the test initialization
    let app_state = cx.update(|cx| init_app_state(cx));

    // Set the global app state so settings_ui and other subsystems can find it
    cx.update(|cx| {
        AppState::set_global(app_state.clone(), cx);
    });

    // Initialize all Zed subsystems
    cx.update(|cx| {
        gpui_tokio::init(cx);
        let themes = if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
            theme::LoadThemes::All(Box::new(Assets))
        } else {
            theme::LoadThemes::JustBase
        };
        theme_settings::init(themes, cx);
        if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(
                        &serde_json::json!({
                            "theme": collaborative_visual_theme(),
                            "ui_font_size": 16,
                            "agent_ui_font_size": 14,
                            "agent_buffer_font_size": 14,
                            "buffer_font_size": 14,
                        })
                        .to_string(),
                        cx,
                    )
                    .expect("collaborative visual settings should parse");
            });
            app_state.user_store.update(cx, |user_store, cx| {
                user_store.set_current_user_for_tests(
                    Arc::new(client::User {
                        legacy_id: 1,
                        username: "local-developer".into(),
                        avatar_uri: "https://visual-test.invalid/avatar.png".into(),
                        name: Some("Local Developer".to_owned()),
                    }),
                    cx,
                );
            });
            cx.http_client()
                .as_fake()
                .replace_handler(|old_handler, request| async move {
                    if request.uri() == "https://visual-test.invalid/avatar.png" {
                        return Ok(http_client::Response::builder()
                            .status(200)
                            .body(http_client::AsyncBody::from(
                                include_bytes!("../test_fixtures/collaborative-avatar.png")
                                    .to_vec(),
                            ))
                            .expect("visual avatar response should build"));
                    }
                    let avatar = match request.uri().to_string().as_str() {
                        "https://visual-test.invalid/maya.png" => Some(
                            include_bytes!("../test_fixtures/collaborative-avatar-maya.png")
                                .as_slice(),
                        ),
                        "https://visual-test.invalid/theo.png" => Some(
                            include_bytes!("../test_fixtures/collaborative-avatar-theo.png")
                                .as_slice(),
                        ),
                        _ => None,
                    };
                    if let Some(avatar) = avatar {
                        return Ok(http_client::Response::builder()
                            .status(200)
                            .body(http_client::AsyncBody::from(avatar.to_vec()))
                            .expect("controlled participant avatar response"));
                    }
                    old_handler(request).await
                });
            channel::init(&app_state.client, app_state.user_store.clone(), cx);
        }
        app_state
            .languages
            .set_theme(theme::ActiveTheme::theme(cx).clone());
        cx.observe_global::<theme::GlobalTheme>({
            let languages = app_state.languages.clone();
            move |cx| languages.set_theme(theme::ActiveTheme::theme(cx).clone())
        })
        .detach();
        client::init(&app_state.client, cx);
        audio::init(cx);
        workspace::init(app_state.clone(), cx);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        command_palette::init(cx);
        editor::init(cx);
        call::init(app_state.client.clone(), app_state.user_store.clone(), cx);
        title_bar::init(cx);
        project_panel::init(cx);
        outline_panel::init(cx);
        terminal_view::init(cx);
        image_viewer::init(cx);
        search::init(cx);
        lsp_locations::init(cx);
        cx.set_global(workspace::PaneSearchBarCallbacks {
            setup_search_bar: |languages, toolbar, window, cx| {
                let search_bar = cx.new(|cx| search::BufferSearchBar::new(languages, window, cx));
                toolbar.update(cx, |toolbar, cx| {
                    toolbar.add_item(search_bar, window, cx);
                });
            },
            wrap_div_with_search_actions: search::buffer_search::register_pane_search_actions,
        });
        prompt_store::init(cx);
        let prompt_builder = prompt_store::PromptBuilder::load(app_state.fs.clone(), false, cx);
        language_model::init(cx);
        client::RefreshLlmTokenListener::register(
            app_state.client.clone(),
            app_state.user_store.clone(),
            cx,
        );
        language_models::init(app_state.user_store.clone(), app_state.client.clone(), cx);
        git_ui::init(cx);
        project::AgentRegistryStore::init_global(
            cx,
            app_state.fs.clone(),
            app_state.client.http_client(),
        );
        agent_ui::init(
            app_state.fs.clone(),
            prompt_builder,
            app_state.languages.clone(),
            true,
            false,
            false,
            cx,
        );
        settings_ui::init(cx);
        #[cfg(feature = "multiplayer-tools")]
        if std::env::var_os("COLLABORATIVE_VISUAL_ONLY").is_some() {
            let app_state = app_state.clone();
            cx.observe_new(move |workspace: &mut Workspace, window, cx| {
                if let Some(window) = window {
                    workspace_status::initialize(&app_state, workspace, window, cx);
                }
            })
            .detach();
        }

        // This fixture omits unrelated application initializers. Load the real
        // asset while requiring every action used by its keyboard workflows.
        let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
            "keymaps/default-macos.json",
            cx,
        )
        .expect("production macOS keymap should parse");
        for action in [
            "menu::Cancel",
            "menu::SelectLast",
            "menu::SelectPrevious",
            "menu::Confirm",
            "agent::OpenTaskContextMenu",
            "editor::Tab",
            "editor::Undo",
            "workspace::FocusNextPart",
            "workspace::FocusPreviousPart",
        ] {
            assert!(
                bindings
                    .iter()
                    .any(|binding| binding.action().name() == action),
                "production keymap is missing visual workflow action {action}"
            );
        }
        cx.bind_keys(bindings);

        // Disable agent notifications during visual tests to avoid popup windows
        agent_settings::AgentSettings::override_global(
            agent_settings::AgentSettings {
                notify_when_agent_waiting: NotifyWhenAgentWaiting::Never,
                play_sound_when_agent_done: PlaySoundWhenAgentDone::Never,
                ..agent_settings::AgentSettings::get_global(cx).clone()
            },
            cx,
        );
    });

    // Run until all initialization tasks complete
    cx.run_until_parked();

    if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
        cx.update(|cx| {
            let theme = theme::ThemeRegistry::global(cx)
                .get(collaborative_visual_theme())
                .expect("Fractal Neutral should be available to visual tests");
            theme::GlobalTheme::update_theme(cx, theme);
            let settings = theme_settings::ThemeSettings::get_global(cx);
            assert_eq!(settings.ui_font.family.as_ref(), "Figtree");
            assert_eq!(settings.agent_ui_font_family().as_ref(), "Figtree");
            assert_ne!(settings.agent_buffer_font_family().as_ref(), "Figtree");
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    if std::env::var("COLLABORATIVE_VISUAL_ONLY").is_ok() {
        let mut failures = Vec::new();
        for variant in [
            CollaborativeVisualVariant::Expanded,
            CollaborativeVisualVariant::AttachedReference,
            CollaborativeVisualVariant::RowActionsReference,
            CollaborativeVisualVariant::TaskMenuReference,
            CollaborativeVisualVariant::Collapsed,
            CollaborativeVisualVariant::Narrow,
            CollaborativeVisualVariant::Compact,
            CollaborativeVisualVariant::EnlargedCompact,
        ] {
            let result = run_collaborative_workspace_visual_tests(
                project_path.clone(),
                app_state.clone(),
                &mut cx,
                update_baseline,
                variant,
            )
            .inspect_err(|error| eprintln!("Collaborative native workflow failed: {error:#}"));
            if result.is_err() {
                // Early setup or capture errors can leave window-owned entities alive.
                for window in cx.update(|cx| cx.windows()) {
                    cx.update_window(window, |_, window, _| window.remove_window())
                        .log_err();
                }
                cx.run_until_parked();
                for _ in 0..15 {
                    cx.advance_clock(Duration::from_millis(100));
                    cx.run_until_parked();
                }
            }
            if let Err(error) = result {
                failures.push(format!("{variant:?}: {error:#}"));
            }
        }
        anyhow::ensure!(failures.is_empty(), "{}", failures.join("\n"));
        return Ok(());
    }

    // Open workspace window
    let window_size = size(px(1280.0), px(800.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    // Create a project for the workspace
    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let workspace_window: WindowHandle<Workspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    })
                },
            )
        })
        .context("Failed to open workspace window")?;

    cx.run_until_parked();

    // Add the test project as a worktree
    let add_worktree_task = workspace_window
        .update(&mut cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                project.find_or_create_worktree(&project_path, true, cx)
            })
        })
        .context("Failed to start adding worktree")?;

    // Use block_test to wait for the worktree task
    // block_test runs both foreground and background tasks, which is needed because
    // worktree creation spawns foreground tasks via cx.spawn
    // Allow parking since filesystem operations happen outside the test dispatcher
    cx.background_executor.allow_parking();
    let worktree_result = cx.foreground_executor.block_test(add_worktree_task);
    cx.background_executor.forbid_parking();
    worktree_result.context("Failed to add worktree")?;

    cx.run_until_parked();

    // Create and add the project panel
    let (weak_workspace, async_window_cx) = workspace_window
        .update(&mut cx, |workspace, window, cx| {
            (workspace.weak_handle(), window.to_async(cx))
        })
        .context("Failed to get workspace handle")?;

    cx.background_executor.allow_parking();
    let panel = cx
        .foreground_executor
        .block_test(ProjectPanel::load(weak_workspace, async_window_cx))
        .context("Failed to load project panel")?;
    cx.background_executor.forbid_parking();

    workspace_window
        .update(&mut cx, |workspace, window, cx| {
            workspace.add_panel(panel, window, cx);
        })
        .log_err();

    cx.run_until_parked();

    // Open the project panel
    workspace_window
        .update(&mut cx, |workspace, window, cx| {
            workspace.open_panel::<ProjectPanel>(window, cx);
        })
        .log_err();

    cx.run_until_parked();

    // Open main.rs in the editor
    let open_file_task = workspace_window
        .update(&mut cx, |workspace, window, cx| {
            let worktree = workspace.project().read(cx).worktrees(cx).next();
            if let Some(worktree) = worktree {
                let worktree_id = worktree.read(cx).id();
                let rel_path: std::sync::Arc<util::rel_path::RelPath> =
                    util::rel_path::rel_path("src/main.rs").into();
                let project_path: project::ProjectPath = (worktree_id, rel_path).into();
                Some(workspace.open_path(project_path, None, true, window, cx))
            } else {
                None
            }
        })
        .log_err()
        .flatten();

    if let Some(task) = open_file_task {
        cx.background_executor.allow_parking();
        let block_result = cx.foreground_executor.block_test(task);
        cx.background_executor.forbid_parking();
        if let Ok(item) = block_result {
            workspace_window
                .update(&mut cx, |workspace, window, cx| {
                    let pane = workspace.active_pane().clone();
                    pane.update(cx, |pane, cx| {
                        if let Some(index) = pane.index_for_item(item.as_ref()) {
                            pane.activate_item(index, true, true, window, cx);
                        }
                    });
                })
                .log_err();
        }
    }

    cx.run_until_parked();

    // Request a window refresh
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })
    .log_err();

    cx.run_until_parked();

    // Track test results
    let mut passed = 0;
    let mut failed = 0;
    let mut updated = 0;

    // Run Test 1: Project Panel (with project panel visible)
    println!("\n--- Test 1: project_panel ---");
    match run_visual_test(
        "project_panel",
        workspace_window.into(),
        &mut cx,
        update_baseline,
    ) {
        Ok(TestResult::Passed) => {
            println!("✓ project_panel: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ project_panel: Baseline updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ project_panel: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 2: Workspace with Editor
    println!("\n--- Test 2: workspace_with_editor ---");

    // Close project panel for this test
    workspace_window
        .update(&mut cx, |workspace, window, cx| {
            workspace.close_panel::<ProjectPanel>(window, cx);
        })
        .log_err();

    cx.run_until_parked();

    match run_visual_test(
        "workspace_with_editor",
        workspace_window.into(),
        &mut cx,
        update_baseline,
    ) {
        Ok(TestResult::Passed) => {
            println!("✓ workspace_with_editor: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ workspace_with_editor: Baseline updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ workspace_with_editor: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test: ThreadItem branch names visual test
    println!("\n--- Test: thread_item_branch_names ---");
    match run_thread_item_branch_name_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ thread_item_branch_names: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ thread_item_branch_names: Baseline updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ thread_item_branch_names: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 3: Multi-workspace sidebar visual tests
    println!("\n--- Test 3: multi_workspace_sidebar ---");
    match run_multi_workspace_sidebar_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ multi_workspace_sidebar: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ multi_workspace_sidebar: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ multi_workspace_sidebar: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 4: Error wrapping visual tests
    println!("\n--- Test 4: error_message_wrapping ---");
    match run_error_wrapping_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ error_message_wrapping: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ error_message_wrapping: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ error_message_wrapping: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 5: Agent Thread View tests
    #[cfg(feature = "visual-tests")]
    {
        println!("\n--- Test 5: agent_thread_with_image (collapsed + expanded) ---");
        match run_agent_thread_view_test(app_state.clone(), &mut cx, update_baseline) {
            Ok(TestResult::Passed) => {
                println!("✓ agent_thread_with_image (collapsed + expanded): PASSED");
                passed += 1;
            }
            Ok(TestResult::BaselineUpdated(_)) => {
                println!("✓ agent_thread_with_image: Baselines updated (collapsed + expanded)");
                updated += 1;
            }
            Err(e) => {
                eprintln!("✗ agent_thread_with_image: FAILED - {}", e);
                failed += 1;
            }
        }
    }

    // Run Test 6: Breakpoint Hover visual tests
    println!("\n--- Test 6: breakpoint_hover (3 variants) ---");
    match run_breakpoint_hover_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ breakpoint_hover: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ breakpoint_hover: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ breakpoint_hover: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 7: Diff Review Button visual tests
    println!("\n--- Test 7: diff_review_button (3 variants) ---");
    match run_diff_review_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ diff_review_button: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ diff_review_button: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ diff_review_button: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 8: ThreadItem icon decorations visual tests
    println!("\n--- Test 8: thread_item_icon_decorations ---");
    match run_thread_item_icon_decorations_visual_tests(app_state.clone(), &mut cx, update_baseline)
    {
        Ok(TestResult::Passed) => {
            println!("✓ thread_item_icon_decorations: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ thread_item_icon_decorations: Baseline updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ thread_item_icon_decorations: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test: Sidebar with duplicate project names
    println!("\n--- Test: sidebar_duplicate_names ---");
    match run_sidebar_duplicate_project_names_visual_tests(
        app_state.clone(),
        &mut cx,
        update_baseline,
    ) {
        Ok(TestResult::Passed) => {
            println!("✓ sidebar_duplicate_names: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ sidebar_duplicate_names: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ sidebar_duplicate_names: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 9: Tool Permissions Settings UI visual test
    println!("\n--- Test 9: tool_permissions_settings ---");
    match run_tool_permissions_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ tool_permissions_settings: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ tool_permissions_settings: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ tool_permissions_settings: FAILED - {}", e);
            failed += 1;
        }
    }

    // Run Test 10: Settings UI sub-page auto-open visual tests
    println!("\n--- Test 10: settings_ui_subpage_auto_open (2 variants) ---");
    match run_settings_ui_subpage_visual_tests(app_state.clone(), &mut cx, update_baseline) {
        Ok(TestResult::Passed) => {
            println!("✓ settings_ui_subpage_auto_open: PASSED");
            passed += 1;
        }
        Ok(TestResult::BaselineUpdated(_)) => {
            println!("✓ settings_ui_subpage_auto_open: Baselines updated");
            updated += 1;
        }
        Err(e) => {
            eprintln!("✗ settings_ui_subpage_auto_open: FAILED - {}", e);
            failed += 1;
        }
    }

    // Clean up the main workspace's worktree to stop background scanning tasks
    // This prevents "root path could not be canonicalized" errors when main() drops temp_dir
    workspace_window
        .update(&mut cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                let worktree_ids: Vec<_> =
                    project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                for id in worktree_ids {
                    project.remove_worktree(id, cx);
                }
            });
        })
        .log_err();

    cx.run_until_parked();

    // Close the main window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    // Run until all cleanup tasks complete
    cx.run_until_parked();

    // Give background tasks time to finish, including scrollbar hide timers (1 second)
    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Print summary
    println!("\n=== Test Summary ===");
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    if updated > 0 {
        println!("Baselines Updated: {}", updated);
    }

    if failed > 0 {
        eprintln!("\n=== Visual Tests FAILED ===");
        Err(anyhow::anyhow!("{} tests failed", failed))
    } else {
        println!("\n=== All Visual Tests PASSED ===");
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn visual_sum_tree_source(cached: bool) -> String {
    let field = if cached {
        "summary_cache: SummaryCache<T::Summary>"
    } else {
        "summary: T::Summary"
    };
    let binding = if cached { "summary_cache" } else { "summary" };
    let update = if cached {
        "summary_cache.invalidate();"
    } else {
        "*summary = compute_summary(children);"
    };
    format!(
        "{}enum Node<T: Item> {{\n    Internal {{\n        height: u8,\n        {field},\n        children: Vec<SumTree<T>>,\n    }},\n    Leaf {{\n        items: Vec<T>,\n    }},\n}}\n{}pub fn push(&mut self, item: T) {{\n    match Arc::make_mut(&mut self.0) {{\n        Node::Internal {{\n            {binding},\n            children,\n            ..\n        }} => {{\n            {update}\n            children\n                .last_mut()\n                .expect(\"internal node has children\")\n                .push(item);\n        }}\n        Node::Leaf {{ items }} => items.push(item),\n    }}\n}}\n",
        "\n".repeat(20),
        "\n".repeat(55)
    )
}

#[cfg(target_os = "macos")]
fn visual_cursor_source(cached: bool) -> String {
    let summary = if cached {
        "child.cached_summary()"
    } else {
        "child.compute_summary()"
    };
    format!(
        "let mut end = self.position.clone();\nend.add_summary(&{summary});\nif end.cmp(target).is_ge() {{\n    self.seek_internal(target);\n}}\n"
    )
}

#[cfg(target_os = "macos")]
fn prepare_collaborative_visual_git_repository(project_path: &Path) -> Result<()> {
    fn run_git(project_path: &Path, arguments: &[&str]) -> Result<()> {
        let status = std::process::Command::new("git")
            .args(arguments)
            .current_dir(project_path)
            .status()?;
        anyhow::ensure!(status.success(), "git {} failed", arguments.join(" "));
        Ok(())
    }

    std::fs::write(project_path.join(".gitignore"), "/target\n/Cargo.lock\n")?;
    std::fs::create_dir_all(project_path.join("src/tree"))?;
    std::fs::write(
        project_path.join("src/sum_tree.rs"),
        visual_sum_tree_source(false),
    )?;
    std::fs::write(
        project_path.join("src/tree/cursor.rs"),
        visual_cursor_source(false),
    )?;
    run_git(project_path, &["init"])?;
    run_git(project_path, &["config", "user.name", "Visual Test"])?;
    run_git(
        project_path,
        &["config", "user.email", "visual-test@example.invalid"],
    )?;
    run_git(project_path, &["add", "."])?;
    run_git(project_path, &["commit", "-m", "Initial visual fixture"])?;
    std::fs::write(
        project_path.join("src/sum_tree.rs"),
        visual_sum_tree_source(true),
    )?;
    std::fs::write(
        project_path.join("src/tree/cursor.rs"),
        visual_cursor_source(true),
    )?;
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
#[derive(Clone, Copy, Debug)]
enum CollaborativeVisualVariant {
    Expanded,
    AttachedReference,
    RowActionsReference,
    TaskMenuReference,
    Collapsed,
    Narrow,
    Compact,
    EnlargedCompact,
}

#[cfg(target_os = "macos")]
fn collaborative_visual_theme() -> &'static str {
    if std::env::var("COLLABORATIVE_VISUAL_THEME").as_deref() == Ok("dark") {
        "Fractal Neutral Dark"
    } else {
        "Fractal Neutral Light"
    }
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn run_collaborative_workspace_visual_tests(
    project_path: PathBuf,
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    _update_baseline: bool,
    variant: CollaborativeVisualVariant,
) -> Result<()> {
    use agent_ui::AgentPanel;
    use gpui::Focusable as _;
    use theme::ActiveTheme as _;
    cx.update(|cx| {
        if let Some(store) = agent_ui::thread_metadata_store::ThreadMetadataStore::try_global(cx) {
            store.update(cx, |store, cx| {
                let previous = store
                    .entries()
                    .map(|entry| entry.thread_id)
                    .collect::<Vec<_>>();
                for thread_id in previous {
                    store.delete(thread_id, cx);
                }
            });
        }
    });
    cx.run_until_parked();
    let enlarged = matches!(variant, CollaborativeVisualVariant::EnlargedCompact);
    let compact = enlarged || matches!(variant, CollaborativeVisualVariant::Compact);
    let font_size = if enlarged { 20 } else { 16 };
    let mut appearance_settings = serde_json::json!({
        "theme": collaborative_visual_theme(), "ui_font_size": font_size,
        "agent_ui_font_size": font_size - 2, "agent_buffer_font_size": font_size - 2,
        "buffer_font_size": font_size - 2,
        "unstable.ui_density": if compact { "compact" } else { "default" },
    });
    let requested_theme_overrides = std::env::var("COLLABORATIVE_VISUAL_THEME_OVERRIDES")
        .ok()
        .map(|overrides| serde_json::from_str::<settings::ThemeStyleContent>(&overrides))
        .transpose()?;
    if let Some(overrides) = &requested_theme_overrides {
        appearance_settings["experimental.theme_overrides"] = serde_json::to_value(overrides)?;
    }
    if let Ok(binary) = std::env::var("COLLABORATIVE_VISUAL_RUST_ANALYZER") {
        anyhow::ensure!(
            std::path::Path::new(&binary).is_absolute() && std::path::Path::new(&binary).is_file(),
            "controlled Rust language server must name an existing absolute binary path"
        );
        appearance_settings["lsp"] = serde_json::json!({
            "rust-analyzer": { "binary": { "path": binary } }
        });
    }
    let appearance_settings = appearance_settings.to_string();
    std::fs::write(paths::settings_file(), &appearance_settings)?;
    // The visual context's generic update does not flush GPUI effects. This entry point
    // lets theme observers apply the settings before the capture window is constructed.
    cx.update_global::<settings::SettingsStore, _>(|store, cx| {
        store
            .set_user_settings(&appearance_settings, cx)
            .expect("visual appearance settings should parse");
    });
    cx.run_until_parked();

    cx.read(|cx| {
        anyhow::ensure!(
            theme::theme_settings(cx).ui_density(cx)
                == if compact {
                    theme::UiDensity::Compact
                } else {
                    theme::UiDensity::Default
                },
            "native capture did not activate the requested density"
        );
        let settings = theme_settings::ThemeSettings::get_global(cx);
        if let Some(overrides) = &requested_theme_overrides {
            anyhow::ensure!(
                settings.experimental_theme_overrides.as_ref() == Some(overrides),
                "capture settings lost the requested theme overrides"
            );
            let expected_theme = settings.apply_theme_overrides(cx.theme().clone());
            anyhow::ensure!(
                cx.theme().window_background_appearance()
                    == expected_theme.window_background_appearance()
                    && cx.theme().colors().editor_background
                        == expected_theme.colors().editor_background
                    && cx.theme().colors().editor_subheader_background
                        == expected_theme.colors().editor_subheader_background,
                "capture theme has not applied the requested review backgrounds"
            );
        }
        anyhow::ensure!(settings.ui_font.family.as_ref() == "Figtree", "unexpected effective UI font");
        anyhow::ensure!(settings.ui_font_size(cx) == px(font_size as f32), "unexpected effective UI size");
        anyhow::ensure!(settings.agent_ui_font_size(cx) == px((font_size - 2) as f32), "unexpected effective agent size");
        println!("Effective appearance: theme={}, UI=Figtree {:?}, agent={} {:?}, code={} {:?}, density={:?}",
            collaborative_visual_theme(), settings.ui_font_size(cx), settings.agent_ui_font_family(), settings.agent_ui_font_size(cx),
            settings.agent_buffer_font_family(), settings.agent_buffer_font_size(cx), theme::theme_settings(cx).ui_density(cx));
        println!(
            "Effective review backgrounds: window={:?}, editor={:?}, subheader={:?}",
            cx.theme().window_background_appearance(),
            cx.theme().colors().editor_background,
            cx.theme().colors().editor_subheader_background,
        );
        anyhow::Ok(())
    })?;

    let create_project = |cx: &mut App| {
        Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    };
    let project = cx.update(create_project);
    let add_worktree = project.update(cx, |project, cx| {
        project.find_or_create_worktree(&project_path, true, cx)
    });
    cx.background_executor.allow_parking();
    cx.foreground_executor.block_test(add_worktree)?;
    let rust_language = cx
        .foreground_executor
        .block_test(app_state.languages.language_for_name("Rust"))?;
    let syntax_probe = language::Rope::from("fn main() { let count: u8 = 1; }");
    anyhow::ensure!(
        !rust_language
            .highlight_text(&syntax_probe, 0..syntax_probe.len())
            .is_empty(),
        "native Rust grammar has no effective syntax-theme mapping"
    );
    cx.background_executor.forbid_parking();
    cx.run_until_parked();

    cx.update(|cx| {
        channel::ChannelStore::global(cx).update(cx, |store, cx| {
            store.set_channels_for_tests([], cx);
            store.set_favorite_channel_ids(Vec::new(), cx);
        });
    });

    let bounds = Bounds {
        origin: point(px(0.), px(0.)),
        size: match variant {
            CollaborativeVisualVariant::Expanded
            | CollaborativeVisualVariant::Compact
            | CollaborativeVisualVariant::EnlargedCompact => size(px(965.), px(631.)),
            CollaborativeVisualVariant::Narrow => size(px(602.), px(640.)),
            CollaborativeVisualVariant::Collapsed => size(px(964.), px(649.)),
            CollaborativeVisualVariant::AttachedReference => size(px(960.), px(649.)),
            CollaborativeVisualVariant::RowActionsReference => size(px(983.), px(676.)),
            CollaborativeVisualVariant::TaskMenuReference => size(px(963.), px(651.)),
        },
    };
    let window: WindowHandle<MultiWorkspace> = cx.update(|cx| {
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                focus: false,
                show: false,
                ..Default::default()
            },
            |window, cx| {
                let workspace = cx
                    .new(|cx| Workspace::new(None, project.clone(), app_state.clone(), window, cx));
                cx.new(|cx| MultiWorkspace::new(workspace, window, cx))
            },
        )
    })?;
    cx.run_until_parked();

    for name in ["zed", "cosmic-text"] {
        let path = project_path
            .parent()
            .context("fixture parent directory")?
            .join(name);
        std::fs::create_dir_all(&path)?;
        let other_project = cx.update(create_project);
        let load = other_project.update(cx, |project, cx| {
            project.find_or_create_worktree(&path, true, cx)
        });
        cx.background_executor.allow_parking();
        cx.foreground_executor.block_test(load)?;
        cx.background_executor.forbid_parking();
        cx.run_until_parked();
        window.update(cx, |multi_workspace, window, cx| {
            let other_workspace =
                cx.new(|cx| Workspace::new(None, other_project, app_state.clone(), window, cx));
            let key = other_workspace.read(cx).project_group_key(cx);
            multi_workspace.add(other_workspace, window, cx);
            if let Some(state) = multi_workspace.group_state_by_key_mut(&key) {
                state.expanded = false;
            }
            cx.notify();
        })?;
    }

    let sidebar = cx.update_window(window.into(), |root, window, cx| {
        let multi_workspace: Entity<MultiWorkspace> = root
            .downcast()
            .map_err(|_| anyhow::anyhow!("collaborative root is not a MultiWorkspace"))?;
        Ok::<_, anyhow::Error>(cx.new(|cx| sidebar::Sidebar::new(multi_workspace, window, cx)))
    })??;
    window.update(cx, |multi_workspace, _, cx| {
        multi_workspace.register_sidebar(sidebar.clone(), cx);
    })?;

    let (workspace, weak_workspace, async_window_context) =
        window.update(cx, |multi_workspace, window, cx| {
            window.set_rem_size(px(font_size as f32));
            let workspace = multi_workspace.workspace().clone();
            multi_workspace.add(workspace.clone(), window, cx);
            (
                workspace.clone(),
                workspace.downgrade(),
                window.to_async(cx),
            )
        })?;
    cx.background_executor.allow_parking();
    let agent_panel = cx.foreground_executor.block_test(AgentPanel::load(
        weak_workspace.clone(),
        async_window_context.clone(),
    ))?;
    let git_panel = cx
        .foreground_executor
        .block_test(git_ui::git_panel::GitPanel::load(
            weak_workspace,
            async_window_context,
        ))?;
    cx.background_executor.forbid_parking();
    window.update(cx, |_, window, cx| {
        workspace.update(cx, |workspace, cx| {
            workspace.add_panel(agent_panel.clone(), window, cx);
            workspace.add_panel(git_panel.clone(), window, cx);
        });
    })?;

    if matches!(variant, CollaborativeVisualVariant::Expanded) {
        window.update(cx, |_, window, cx| {
            workspace.focus_handle(cx).dispatch_action(
                &workspace::SwitchToCollaborativeWorkspace,
                window,
                cx,
            );
        })?;
        cx.run_until_parked();
        cx.resize_native_window(window.into(), size(px(1000.), px(682.)))?;
        if let Some(directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR") {
            std::fs::create_dir_all(&directory)?;
            let actual = cx.capture_screenshot(window.into())?;
            anyhow::ensure!(
                actual.dimensions() == (2000, 1364),
                "empty native viewport does not match Image 1"
            );
            actual.save(PathBuf::from(directory).join("user-current-empty.png"))?;
        }
        cx.resize_native_window(window.into(), bounds.size)?;
    }

    let multi_acp = std::env::var_os("COLLABORATIVE_VISUAL_MULTI_ACP").is_some();
    let connection = StubAgentConnection::new().with_supports_load_session(true);
    let connection = if multi_acp {
        connection.with_session_config_options(vec![
            acp::SessionConfigOption::select(
                "mode",
                "Mode",
                "automatic",
                vec![
                    acp::SessionConfigSelectOption::new("automatic", "Automatic"),
                    acp::SessionConfigSelectOption::new("manual", "Manual"),
                ],
            )
            .category(acp::SessionConfigOptionCategory::Mode),
            acp::SessionConfigOption::select(
                "model",
                "Model",
                "large",
                vec![
                    acp::SessionConfigSelectOption::new("large", "A model with a long name"),
                    acp::SessionConfigSelectOption::new("small", "Small model"),
                ],
            )
            .category(acp::SessionConfigOptionCategory::Model),
            acp::SessionConfigOption::boolean("web_search", "Web search", false),
        ])
    } else {
        connection
    };
    let diff_path = project_path.join("src/sum_tree.rs");
    let initial_updates = vec![
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
            r#"```rust
let mut end = self.position.clone();
end.add_summary(&child.cached_summary());
```

Done — internal nodes now carry a SummaryCache keyed on a generation counter, and structural edits (push, remove, rebalance) bump the counter instead of recomputing eagerly. Cursor::seek reads through the cache, so it only recomputes summaries for nodes whose subtree actually changed — repeated seeks over an unedited region are pure cache hits. The focused tests cover invalidation on push and on rebalancing."#
                .into(),
        )),
        acp::SessionUpdate::ToolCall(
            acp::ToolCall::new("workspace-diff", "Updated 2 files")
                .kind(acp::ToolKind::Edit)
                .status(acp::ToolCallStatus::Completed)
                .content(vec![
                    acp::ToolCallContent::Diff(acp::Diff::new(diff_path, visual_sum_tree_source(true)).old_text(visual_sum_tree_source(false))),
                    acp::ToolCallContent::Diff(acp::Diff::new(project_path.join("src/tree/cursor.rs"), visual_cursor_source(true)).old_text(visual_cursor_source(false))),
                ]),
        ),
        acp::SessionUpdate::ToolCall(
            acp::ToolCall::new("validation-output", "Ran focused Rust validation")
                .kind(acp::ToolKind::Execute)
                .status(acp::ToolCallStatus::Completed)
                .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                    acp::ContentBlock::Text(acp::TextContent::new(
                        "cargo test -p workspace\n4 focused tests passed in 2.1s".to_owned(),
                    )),
                ))]),
        ),
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
            "The native project diff is ready for review.".into(),
        )),
    ];
    let agent_server: Rc<dyn AgentServer> = Rc::new(StubAgentServer::new(connection.clone()));
    let previous_thread_id = cx.read(|cx| agent_panel.read(cx).active_thread_id(cx));
    window.update(cx, |_, window, cx| {
        agent_panel.update(cx, |panel, cx| {
            panel.open_external_thread_with_server(agent_server, window, cx);
        });
    })?;
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }
    let active_thread_id = cx.read(|cx| agent_panel.read(cx).active_thread_id(cx));
    anyhow::ensure!(
        active_thread_id.is_some() && active_thread_id != previous_thread_id,
        "collaborative timeline did not activate a distinct native ACP thread"
    );

    let thread_view = cx
        .read(|cx| agent_panel.read(cx).active_thread_view_for_tests().cloned())
        .context("collaborative visual agent thread view is unavailable")?;
    let thread = cx
        .read(|cx| {
            thread_view
                .read(cx)
                .active_thread()
                .map(|active| active.read(cx).thread.clone())
        })
        .context("collaborative visual ACP thread is unavailable")?;
    if matches!(variant, CollaborativeVisualVariant::Expanded) {
        let (composer_registration, timeline_registration, participant_registration, review_registration) = window.update(cx, |_, window, cx| {
            let composer =
                agent_ui::collaborative_composer::CollaborativeComposerAdapter::from_agent_panel(
                    &agent_panel,
                    &workspace,
                    cx,
                )?;
            let timeline =
                agent_ui::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                    &agent_panel,
                    &workspace,
                    cx,
                )?;
            let participants = agent_ui::collaborative_participants::CollaborativeParticipantAdapter::from_agent_panel(
                &agent_panel, &workspace, cx,
            )?;
            let review = agent_ui::collaborative_review::CollaborativeAgentReviewAdapter::new(
                Some(thread.clone()), &workspace, window, cx,
            )?;
            let thread_id = agent_panel.read(cx).active_thread_id(cx)
                .context("empty draft must have a canonical task identity")?;
            workspace.update(cx, |workspace, cx| -> Result<_> {
                workspace.navigate_collaborative_to(
                    workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(thread_id.to_key_string()),
                    |_| true, window, cx,
                )?;
                Ok((
                    composer.register_in_workspace(workspace, cx)?,
                    timeline.register_in_workspace(workspace, cx)?,
                    workspace.register_collaborative_participant_provider(participants.into_provider(), cx)?,
                    review.register_in_workspace(workspace, cx)?,
                ))
            })
        })??;
        cx.resize_native_window(window.into(), size(px(1000.), px(682.)))?;
        cx.read(|cx| {
            anyhow::ensure!(
                theme::ActiveTheme::theme(cx).name.as_ref() == collaborative_visual_theme(),
                "empty draft capture lost its requested appearance"
            );
            anyhow::Ok(())
        })?;
        let composer_bounds = cx
            .debug_bounds(window.into(), "COLLABORATIVE-COMPOSER")?
            .context("empty draft composer must be rendered")?;
        anyhow::ensure!(
            composer_bounds.size.height > px(0.),
            "empty draft composer has no height"
        );
        if let Some(directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR") {
            let actual = cx.capture_screenshot(window.into())?;
            anyhow::ensure!(
                actual.dimensions() == (2000, 1364),
                "empty draft viewport does not match Image 1"
            );
            actual.save(PathBuf::from(directory).join("user-current-empty-draft.png"))?;
        }
        window.update(cx, |_, _, cx| {
            workspace.update(cx, |workspace, cx| {
                assert!(
                    workspace.unregister_collaborative_composer_provider(composer_registration, cx)
                );
                assert!(
                    workspace.unregister_collaborative_timeline_provider(timeline_registration, cx)
                );
                assert!(
                    workspace.unregister_collaborative_participant_provider(
                        participant_registration,
                        cx
                    )
                );
                assert!(
                    workspace.unregister_collaborative_review_provider(review_registration, cx)
                );
            });
        })?;
        cx.resize_native_window(window.into(), bounds.size)?;
    }
    let visual_prompt = "Cache SumTree node summaries across seeks";
    connection.set_next_prompt_updates(initial_updates);
    let send = thread.update(cx, |thread, cx| {
        thread.set_provisional_title(visual_prompt.into(), cx);
        thread.send(vec!["Cursor::seek recomputes each internal node’s summary on every descent — for a rope of a few MB that is most of our seek time. Can we cache the summaries on the nodes and invalidate only the edited path?".into()], cx)
    });
    cx.background_executor.allow_parking();
    cx.foreground_executor.block_test(send)?;
    cx.background_executor.forbid_parking();
    cx.run_until_parked();

    let terminal_tool_call_id = acp::ToolCallId::new("terminal-output");
    let terminal_id = acp::TerminalId::new("collaborative-visual-terminal");
    thread.update(cx, |thread, cx| {
        thread.register_test_terminal(
            terminal_id.clone(),
            "cargo test -p workspace".to_owned(),
            Some(project_path.clone()),
            b"running 4 tests\ntest collaborative_layout ... ok\ntest result: ok\n".to_vec(),
            cx,
        );
        thread
            .handle_session_update(
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new(terminal_tool_call_id.clone(), "cargo test -p workspace")
                        .kind(acp::ToolKind::Execute)
                        .status(acp::ToolCallStatus::Completed)
                        .content(vec![acp::ToolCallContent::Terminal(acp::Terminal::new(
                            terminal_id,
                        ))]),
                ),
                cx,
            )
            .expect("visual terminal tool call should be accepted");
    });
    cx.run_until_parked();
    thread.update(cx, |thread, cx| {
        thread
            .handle_session_update(
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new("failed-check", "Checked hosted collaboration health")
                        .kind(acp::ToolKind::Fetch)
                        .status(acp::ToolCallStatus::Failed)
                        .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                            acp::ContentBlock::Text(acp::TextContent::new(
                                "Local hosted collaboration service is not running; retry after starting Compose."
                                    .to_owned(),
                            )),
                        ))]),
                ),
                cx,
            )
            .expect("visual failed tool call should be accepted");
    });
    cx.run_until_parked();
    thread_view.update(cx, |conversation, cx| {
        conversation.expand_tool_call(acp::ToolCallId::new("workspace-diff"), cx);
        conversation.expand_tool_call(acp::ToolCallId::new("validation-output"), cx);
        conversation.expand_tool_call(terminal_tool_call_id, cx);
        conversation.expand_tool_call(acp::ToolCallId::new("failed-check"), cx);
    });
    cx.run_until_parked();

    for (prompt, updates) in [
        (
            "Let’s back that claim with data before we merge — add a criterion benchmark that seeks randomly through a tree of a million items, and share before/after numbers. I’d also like one seek-after-push case so we can see the invalidation cost on the spine.",
            vec![
                acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
                    "Following project state through the native workspace adapters.".into(),
                )),
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new("read-source", "Read the workspace composition")
                        .kind(acp::ToolKind::Read)
                        .status(acp::ToolCallStatus::Completed),
                ),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "The benchmark now exercises repeated random seeks and a seek immediately after a push. Cached summaries avoid recomputing unchanged subtrees; the push case invalidates only the path to the edited leaf. The raw run output and benchmark configuration are available in the tool details."
                        .into(),
                )),
            ],
        ),
        (
            "Run the focused Rust validation",
            vec![
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new("cargo-test", "Ran focused Rust validation")
                        .kind(acp::ToolKind::Execute)
                        .status(acp::ToolCallStatus::Completed),
                ),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "Formatting, compilation, and the focused workspace tests completed successfully. The validation covered the multiplayer feature boundary, Rust tooling integration, persisted layout restoration, and the native project review adapter."
                        .into(),
                )),
            ],
        ),
        (
            "Prepare the native change for review",
            vec![
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new("edit-workspace", "Updated the workspace session model")
                        .kind(acp::ToolKind::Edit)
                        .status(acp::ToolCallStatus::Completed),
                ),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "The focused structural change is visible in the native review pane and ready for review. It keeps project data in the Git owner, activity in the ACP thread, and selection in the workspace navigation model while the responsive layout presents those sources coherently."
                        .into(),
                )),
            ],
        ),
    ] {
        println!("  Extending collaborative timeline: {prompt}");
        connection.set_next_prompt_updates(updates);
        let send = thread.update(cx, |thread, cx| thread.send(vec![prompt.into()], cx));
        cx.background_executor.allow_parking();
        cx.foreground_executor.block_test(send)?;
        cx.background_executor.forbid_parking();
        cx.run_until_parked();
        println!("  Extended collaborative timeline: {prompt}");
    }

    for (index, label) in [
        "Trace summary propagation",
        "Review randomized buffer coverage",
        "Check cursor invalidation callers",
    ]
    .into_iter()
    .enumerate()
    {
        let session_id = acp::SessionId::new(format!("visual-subtask-{index}"));
        let mut call = acp::ToolCall::new(format!("spawn-subtask-{index}"), label)
            .status(acp::ToolCallStatus::Completed);
        call.meta = Some(serde_json::from_value(serde_json::json!({
            "subagent_session_info": { "session_id": session_id, "message_start_index": 0 }
        }))?);
        thread.update(cx, |thread, cx| -> Result<()> {
            thread.handle_session_update(acp::SessionUpdate::ToolCall(call), cx)?;
            thread.subagent_spawned(session_id, cx);
            Ok(())
        })?;
        cx.run_until_parked();
    }

    let worktree_paths = cx.read(|cx| project.read(cx).worktree_paths(cx));
    let mut metadata = [
        "Trace dropped keystrokes in GPUI input",
        "Review element cache invalidation rules",
        "Prototype a paint bounds inspector",
        "How deferred elements are drawn",
        "Benchmark text layout after font fallback",
        "Fix IME candidate window positioning",
        "Add scroll wheel phase events on macOS",
        "Simplify Entity subscription cleanup",
        "Reproduce stale text after window restore",
        "Audit focus restoration across modals",
        "Document the element lifecycle",
        "Summarize feedback on my first GPUI change",
        "Find a small rendering issue to work on",
        "Explain retained mode rendering",
        "Compare Metal frame capture tools",
        "Draft release notes for accessibility fixes",
    ]
    .into_iter()
    .enumerate()
    .map(
        |(index, title)| agent_ui::thread_metadata_store::ThreadMetadata {
            execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
            thread_id: agent_ui::ThreadId::new(),
            session_id: Some(acp::SessionId::new(format!("visual-history-{index}"))),
            agent_id: project::AgentId("stub".into()),
            title: Some(title.into()),
            title_override: None,
            updated_at: chrono::DateTime::from_timestamp(
                chrono::Utc::now().timestamp() - (index as i64 + 1) * 60,
                0,
            )
            .expect("valid controlled timestamp"),
            created_at: None,
            interacted_at: None,
            worktree_paths: if (5..11).contains(&index) {
                worktree_paths.clone()
            } else {
                Default::default()
            },
            remote_connection: None,
            archived: false,
        },
    )
    .collect::<Vec<_>>();
    let menu_metadata = metadata
        .iter_mut()
        .find(|entry| entry.display_title().as_ref() == "Review element cache invalidation rules")
        .context("menu task fixture")?;
    let menu_session = acp::SessionId::new(format!(
        "visual-menu-{}",
        menu_metadata.thread_id.to_key_string()
    ));
    menu_metadata.session_id = Some(menu_session.clone());
    menu_metadata.agent_id = agent::ZED_AGENT_ID.clone();
    let mut stored: agent::DbThread = serde_json::from_value(serde_json::json!({
        "title": menu_metadata.display_title(), "messages": [], "updated_at": menu_metadata.updated_at,
    }))?;
    stored
        .messages
        .push(Arc::new(agent::Message::User(agent::UserMessage {
            id: acp_thread::ClientUserMessageId::new(),
            content: vec![agent::UserMessageContent::Text(
                "Review the element cache invalidation rules and retain the parent task context."
                    .into(),
            )]
            .into(),
        })));
    let save = cx.update(|cx| {
        agent::ThreadStore::global(cx).update(cx, |store, cx| {
            store.save_thread(menu_session, stored, Default::default(), cx)
        })
    });
    cx.background_executor.allow_parking();
    let saved = cx.foreground_executor.block_test(save);
    cx.background_executor.forbid_parking();
    saved?;
    cx.update(|cx| {
        agent_ui::thread_metadata_store::ThreadMetadataStore::global(cx).update(cx, |store, cx| {
            store.save_all(metadata, cx);
        });
    });
    cx.run_until_parked();

    window.update(cx, |_, window, cx| {
        let mut pinned = agent_ui::thread_metadata_store::ThreadMetadataStore::global(cx)
            .read(cx)
            .entries()
            .filter(|metadata| {
                [
                    "Trace dropped",
                    "Review element",
                    "Prototype a",
                    "How deferred",
                    "Benchmark text",
                ]
                .iter()
                .any(|prefix| metadata.display_title().starts_with(prefix))
            })
            .map(|metadata| {
                (
                    std::cmp::Reverse(metadata.updated_at),
                    workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(
                        metadata.thread_id.to_key_string(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        pinned.sort_by_key(|(updated_at, _)| *updated_at);
        workspace.update(cx, |workspace, cx| {
            for (_, target) in pinned {
                if !workspace.collaborative_navigation().is_pinned(&target) {
                    workspace
                        .toggle_collaborative_pin(target, window, cx)
                        .expect("controlled task should pin");
                }
            }
        });
    })?;

    let (
        composer_registration,
        timeline_registration,
        participant_registration,
        project_review_registration,
        agent_review_registration,
    ) = window.update(cx, |_, window, cx| {
        let composer = agent_ui::collaborative_composer::CollaborativeComposerAdapter::from_agent_panel(
            &agent_panel,
            &workspace,
            cx,
        )?;
        let timeline = agent_ui::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
            &agent_panel,
            &workspace,
            cx,
        )?;
        let participants = agent_ui::collaborative_participants::CollaborativeParticipantAdapter::from_agent_panel(
            &agent_panel,
            &workspace,
            cx,
        )?;
        let user_store = app_state.user_store.clone();
        let participants = participants.with_room_state_reader(move |cx| {
            let mut room_participants = user_store.read(cx).current_user().map(|user| {
                workspace::collaborative_participants::CollaborativeParticipant::human(
                    &user, workspace::collaborative_participants::CollaborativeParticipantPresence::Online,
                )
            }).into_iter().collect::<Vec<_>>();
            for (legacy_id, name, avatar) in [(2, "Maya", "maya"), (3, "Theo", "theo")] {
                let user = client::User { legacy_id, username: name.into(), name: Some(name.to_owned()),
                    avatar_uri: format!("https://visual-test.invalid/{avatar}.png").into() };
                room_participants.push(workspace::collaborative_participants::CollaborativeParticipant::human(
                    &user, workspace::collaborative_participants::CollaborativeParticipantPresence::Online,
                ));
            }
            (
                room_participants,
                workspace::collaborative_participants::CollaborativeConnectionState::Connected,
            )
        });
        let project_review = git_ui::collaborative_review::CollaborativeProjectReviewAdapter::from_workspace_or_create(
            &workspace,
            window,
            cx,
        )?;
        let agent_review = agent_ui::collaborative_review::CollaborativeAgentReviewAdapter::new(
            Some(thread.clone()),
            &workspace,
            window,
            cx,
        )?;
        let registrations = workspace.update(cx, |workspace, cx| -> Result<_> {
            let composer_registration = composer.register_in_workspace(workspace, cx)?;
            let timeline_registration = timeline.register_in_workspace(workspace, cx)?;
            let participant_registration = workspace.register_collaborative_participant_provider(
                participants.into_provider(),
                cx,
            )?;
            let project_review_registration =
                project_review.register_in_workspace(workspace, cx)?;
            let agent_review_registration = agent_review.register_in_workspace(workspace, cx)?;
            Ok((
                composer_registration,
                timeline_registration,
                participant_registration,
                project_review_registration,
                agent_review_registration,
            ))
        })?;
        workspace.focus_handle(cx).dispatch_action(
            &workspace::SwitchToCollaborativeWorkspace,
            window,
            cx,
        );
        Ok::<_, anyhow::Error>(registrations)
    })??;
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    let visual_thread_id = window.update(cx, |_, window, cx| {
        let thread_id = agent_panel.read(cx).active_thread_id(cx);
        if let Some(thread_id) = thread_id {
            workspace.update(cx, |workspace, cx| {
                workspace.navigate_collaborative_to(
                    workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(
                        thread_id.to_key_string(),
                    ),
                    |_| true,
                    window,
                    cx,
                )
            })?;
        }
        workspace.focus_handle(cx).dispatch_action(
            &workspace::collaborative_review::SelectProjectReview,
            window,
            cx,
        );
        if matches!(variant, CollaborativeVisualVariant::Collapsed) {
            window.dispatch_action(
                Box::new(workspace::collaborative_review::ToggleCollaborativeReview),
                cx,
            );
        }
        Ok::<_, anyhow::Error>(thread_id)
    })??;
    for _ in 0..20 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(50));
        cx.advance_clock(Duration::from_millis(50));
        cx.run_until_parked();
    }
    cx.read(|cx| {
        anyhow::ensure!(
            workspace.read(cx).collaborative_review().selected_slot()
                == Some(workspace::collaborative_review::CollaborativeReviewSlot::ProjectChanges),
            "project review action did not change the displayed review owner"
        );
        Ok::<_, anyhow::Error>(())
    })?;
    let review_requested = cx.read(|cx| workspace.read(cx).collaborative_review_view(cx).is_some());
    anyhow::ensure!(
        review_requested != matches!(variant, CollaborativeVisualVariant::Collapsed),
        "collaborative review state does not match the requested visual variant"
    );
    for (entry_index, selector) in [
        (1, "AGENT-THREAD-MARKDOWN-CODE-BLOCK"),
        (2, "AGENT-THREAD-TOOL-DIFF"),
        (3, "AGENT-THREAD-TOOL-CONTENT"),
        (5, "AGENT-THREAD-TOOL-TERMINAL"),
        (6, "AGENT-THREAD-TOOL-FAILED"),
    ] {
        thread_view.update(cx, |conversation, cx| {
            conversation.scroll_thread_to_entry_for_tests(entry_index, cx);
        });
        cx.update_window(window.into(), |_, window, _| window.refresh())?;
        cx.run_until_parked();
        if cx.debug_bounds(window.into(), selector)?.is_none() {
            if let Some(directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR") {
                std::fs::create_dir_all(&directory)?;
                cx.capture_screenshot(window.into())?
                    .save(PathBuf::from(directory).join("failed-native-content.png"))?;
            }
            anyhow::bail!(
                "collaborative visual scenario did not render {selector} at ACP entry {entry_index}"
            );
        }
    }
    thread_view.update(cx, |conversation, cx| {
        for id in ["validation-output", "terminal-output"] {
            conversation.collapse_tool_call_for_tests(&acp::ToolCallId::new(id), cx);
        }
        conversation.scroll_thread_to_entry_for_tests(
            if matches!(
                variant,
                CollaborativeVisualVariant::RowActionsReference
                    | CollaborativeVisualVariant::TaskMenuReference
            ) {
                2
            } else {
                1
            },
            cx,
        );
    });
    cx.update_window(window.into(), |_, window, _| window.refresh())?;
    cx.run_until_parked();
    for _ in 0..3 {
        window.update(cx, |_, window, cx| {
            window.simulate_next_frame(cx);
            window.refresh();
        })?;
        cx.run_until_parked();
    }
    cx.read(|cx| -> Result<()> {
        anyhow::ensure!(
            theme::ActiveTheme::theme(cx).name.as_ref() == collaborative_visual_theme(),
            "populated capture lost its requested appearance"
        );
        let settings = theme_settings::ThemeSettings::get_global(cx);
        anyhow::ensure!(
            settings.ui_font_size(cx) == px(font_size as f32),
            "capture-time UI font size changed"
        );
        anyhow::ensure!(
            settings.agent_ui_font_size(cx) == px((font_size - 2) as f32),
            "capture-time agent font size changed"
        );
        Ok(())
    })?;
    match variant {
        CollaborativeVisualVariant::AttachedReference
        | CollaborativeVisualVariant::RowActionsReference
        | CollaborativeVisualVariant::TaskMenuReference => {
            capture_collaborative_accessibility_state(
                match variant {
                    CollaborativeVisualVariant::RowActionsReference => "row-actions-reference",
                    CollaborativeVisualVariant::TaskMenuReference => "task-menu-reference",
                    _ => "attached-reference",
                },
                bounds.size,
                window.into(),
                cx,
            )
        }
        CollaborativeVisualVariant::Expanded | CollaborativeVisualVariant::Collapsed => {
            capture_collaborative_reference_comparison(
                if matches!(variant, CollaborativeVisualVariant::Expanded) {
                    "screenshot-1"
                } else {
                    "screenshot-2"
                },
                window.into(),
                cx,
            )
        }
        CollaborativeVisualVariant::Narrow
        | CollaborativeVisualVariant::Compact
        | CollaborativeVisualVariant::EnlargedCompact => capture_collaborative_accessibility_state(
            if matches!(variant, CollaborativeVisualVariant::Narrow) {
                "narrow"
            } else if matches!(variant, CollaborativeVisualVariant::Compact) {
                "compact"
            } else {
                "enlarged-compact"
            },
            bounds.size,
            window.into(),
            cx,
        ),
    }?;

    if matches!(variant, CollaborativeVisualVariant::Expanded) {
        for (name, viewport) in [
            ("user-final-populated", size(px(961.), px(647.))),
            ("resized-narrow", size(px(602.), px(640.))),
            ("resized-expanded", bounds.size),
        ] {
            cx.resize_native_window(window.into(), viewport)?;
            capture_collaborative_accessibility_state(name, viewport, window.into(), cx)?;
            anyhow::ensure!(
                cx.debug_bounds(window.into(), "COLLABORATIVE-REVIEW-REGION")?
                    .is_some()
                    == (name != "resized-narrow"),
                "{name}: review visibility did not recover across native window resizing"
            );
        }
    }

    if let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR") {
        let target = cx
            .read(|cx| {
                agent_ui::thread_metadata_store::ThreadMetadataStore::global(cx)
                    .read(cx)
                    .entries()
                    .find(|metadata| {
                        metadata.display_title().as_ref()
                            == "Review element cache invalidation rules"
                    })
                    .map(|metadata| metadata.thread_id)
            })
            .context("non-active task-menu fixture")?;
        anyhow::ensure!(
            Some(target) != visual_thread_id,
            "menu fixture must be a non-active task"
        );
        let row = cx
            .debug_bounds(
                window.into(),
                &format!("COLLABORATIVE-TASK-ROW-{}", target.to_key_string()),
            )?
            .context("non-active task row must be visible")?;
        cx.simulate_mouse_down(
            window.into(),
            row.center(),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        cx.simulate_mouse_up(
            window.into(),
            row.center(),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        for _ in 0..3 {
            window.update(cx, |_, window, cx| {
                window.simulate_next_frame(cx);
                window.refresh();
            })?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "MENU_ITEM-Unpin")?.is_some(),
            "menu must resolve pinned state from the target task"
        );
        anyhow::ensure!(
            cx.read(|cx| agent_panel.read(cx).active_thread_id(cx)) == visual_thread_id,
            "opening a task menu changed the active conversation"
        );
        let name = match variant {
            CollaborativeVisualVariant::Expanded => "expanded",
            CollaborativeVisualVariant::Collapsed => "collapsed",
            CollaborativeVisualVariant::AttachedReference => "attached-reference",
            CollaborativeVisualVariant::RowActionsReference => "row-actions-reference",
            CollaborativeVisualVariant::TaskMenuReference => "task-menu-reference",
            CollaborativeVisualVariant::Narrow => "narrow",
            CollaborativeVisualVariant::Compact => "compact",
            CollaborativeVisualVariant::EnlargedCompact => "enlarged-compact",
        };
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join(format!("menu-non-active-{name}.png")))?;
        cx.simulate_keystrokes(window.into(), "escape");
        cx.run_until_parked();
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "MENU_ITEM-Unpin")?.is_none(),
            "Escape must dismiss the task context menu"
        );
        if matches!(variant, CollaborativeVisualVariant::TaskMenuReference) {
            for label in ["Copy content", "Copy JSON"] {
                cx.update(|cx| {
                    cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                        "awaiting task export".to_owned(),
                    ))
                });
                cx.simulate_mouse_down(
                    window.into(),
                    row.center(),
                    gpui::MouseButton::Right,
                    Modifiers::default(),
                );
                cx.simulate_mouse_up(
                    window.into(),
                    row.center(),
                    gpui::MouseButton::Right,
                    Modifiers::default(),
                );
                for _ in 0..3 {
                    window.update(cx, |_, window, cx| {
                        window.simulate_next_frame(cx);
                    })?;
                    cx.run_until_parked();
                }
                let operation = cx
                    .debug_bounds(window.into(), &format!("MENU_ITEM-{label}"))?
                    .context("native transcript export action")?;
                cx.simulate_click(window.into(), operation.center(), Modifiers::default());
                let mut exported = None;
                for _ in 0..30 {
                    cx.run_until_parked();
                    exported = cx
                        .read_from_clipboard()
                        .and_then(|item| item.text())
                        .filter(|text| text != "awaiting task export");
                    if exported.is_some() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                    cx.advance_clock(Duration::from_millis(10));
                }
                let exported = exported.context("native task export did not complete")?;
                if label == "Copy JSON" {
                    let transcript: agent::DbThread = serde_json::from_str(&exported)?;
                    anyhow::ensure!(
                        transcript.title.as_ref() == "Review element cache invalidation rules"
                            && transcript.messages.len() == 1,
                        "structured export must contain the non-active task transcript"
                    );
                } else {
                    anyhow::ensure!(
                        exported.contains("retain the parent task context"),
                        "content export must contain the non-active task transcript"
                    );
                }
                anyhow::ensure!(
                    cx.read(|cx| agent_panel.read(cx).active_thread_id(cx)) == visual_thread_id,
                    "export changed the active conversation"
                );
                println!("Native non-active task {label}: transcript contents verified");
            }
            let opened_menu = std::rc::Rc::new(std::cell::RefCell::new(None));
            let _menu_observation = cx.update(|cx| {
                let opened_menu = opened_menu.clone();
                cx.observe_new::<ui::ContextMenu>(move |_, _, cx| {
                    *opened_menu.borrow_mut() = Some(cx.entity());
                })
            });
            cx.simulate_mouse_down(
                window.into(),
                row.center(),
                gpui::MouseButton::Right,
                Modifiers::default(),
            );
            cx.simulate_mouse_up(
                window.into(),
                row.center(),
                gpui::MouseButton::Right,
                Modifiers::default(),
            );
            for _ in 0..3 {
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
            }
            let menu = opened_menu
                .borrow_mut()
                .take()
                .context("opened native task menu")?;
            let contracts = cx.read(|cx| menu.read(cx).accessibility_contracts());
            let projects_index = contracts
                .iter()
                .position(|entry| entry.label.as_ref() == "Projects")
                .context("native Projects submenu")?;
            let steps = contracts
                .iter()
                .skip(projects_index + 1)
                .filter(|entry| !entry.disabled)
                .count();
            cx.simulate_keystrokes(window.into(), "end");
            cx.run_until_parked();
            for _ in 0..steps {
                cx.simulate_keystrokes(window.into(), "up");
                cx.run_until_parked();
            }
            cx.simulate_keystrokes(window.into(), "enter");
            cx.run_until_parked();
            for _ in 0..3 {
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
            }
            anyhow::ensure!(
                cx.debug_bounds(window.into(), "MENU_ITEM-Add to gpui")?
                    .is_some(),
                "keyboard project submenu did not render"
            );
            cx.capture_screenshot(window.into())?
                .save(PathBuf::from(&output_directory).join("menu-projects-keyboard.png"))?;
            cx.simulate_keystrokes(window.into(), "escape");
            cx.run_until_parked();
            cx.simulate_keystrokes(window.into(), "escape");
            cx.run_until_parked();
            anyhow::ensure!(
                cx.debug_bounds(window.into(), "MENU_ITEM-Unpin")?.is_none(),
                "Escape must dismiss both menu levels"
            );
            anyhow::ensure!(
                cx.read(|cx| agent_panel.read(cx).active_thread_id(cx)) == visual_thread_id,
                "project submenu changed the active conversation"
            );
            println!("Native non-active task Projects: keyboard submenu and dismissal verified");
        }
    }

    if matches!(variant, CollaborativeVisualVariant::Expanded)
        && let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
    {
        let native_review = cx
            .read(|cx| {
                workspace
                    .read(cx)
                    .collaborative_review_view(cx)
                    .and_then(|view| view.downcast::<git_ui::project_diff::ProjectDiff>().ok())
            })
            .context("native ProjectDiff should be selected")?;
        let review_editor = cx.read(|cx| {
            native_review
                .read(cx)
                .editor(cx)
                .read(cx)
                .rhs_editor()
                .clone()
        });
        window.update(cx, |_, window, cx| {
            review_editor.update(cx, |editor, cx| {
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections
                        .select_ranges([language::Point::new(0, 0)..language::Point::new(0, 0)]);
                });
            });
            review_editor.focus_handle(cx).dispatch_action(
                &editor::actions::OpenExcerpts,
                window,
                cx,
            );
        })?;
        cx.run_until_parked();
        let source_editor = cx
            .read(|cx| workspace.read(cx).active_item_as::<editor::Editor>(cx))
            .context("OpenExcerpts should activate the source editor")?;
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "NATIVE-WORKSPACE-PANE")?
                .is_some(),
            "native source pane must be painted before keyboard input"
        );
        window.update(cx, |_, window, cx| {
            println!(
                "Native source keyboard: focus={:?}, source={:?}, contexts={:?}, Tab={:?}",
                window.focused(cx),
                source_editor.focus_handle(cx),
                window.context_stack(),
                window.highest_precedence_binding_for_action(&editor::actions::Tab)
            );
            anyhow::ensure!(
                source_editor.focus_handle(cx).contains_focused(window, cx),
                "OpenExcerpts must focus the native source editor"
            );
            anyhow::Ok(())
        })??;
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join("review-source-before-keyboard.png"))?;
        let source_selection = window.update(cx, |_, window, cx| {
            source_editor.update(cx, |editor, cx| {
                let selection = editor
                    .selections
                    .newest::<language::Point>(&editor.display_snapshot(cx));
                // Tab at an existing indent can correctly move the caret without editing.
                // Use the unindented start of this file to require a real text edit.
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections
                        .select_ranges([language::Point::new(0, 0)..language::Point::new(0, 0)]);
                });
                selection.range()
            })
        })?;
        let source_before_tab = cx.read(|cx| source_editor.read(cx).text(cx));
        cx.simulate_keystrokes(window.into(), "tab");
        cx.run_until_parked();
        anyhow::ensure!(
            cx.read(|cx| source_editor.read(cx).text(cx)) != source_before_tab,
            "Tab must edit the source file, not escape to another collaborative region"
        );
        cx.simulate_keystrokes(window.into(), "cmd-z");
        cx.run_until_parked();
        anyhow::ensure!(
            cx.read(|cx| source_editor.read(cx).text(cx)) == source_before_tab,
            "native Undo should restore the file after the keyboard check"
        );
        window.update(cx, |_, window, cx| {
            source_editor.update(cx, |editor, cx| {
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections.select_ranges([source_selection]);
                });
            });
        })?;
        for (name, viewport) in [
            ("review-source-file", bounds.size),
            ("review-source-file-narrow", size(px(420.), px(800.))),
            ("review-source-file-restored", bounds.size),
        ] {
            cx.resize_native_window(window.into(), viewport)?;
            for _ in 0..3 {
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
            }
            anyhow::ensure!(
                cx.debug_bounds(window.into(), "NATIVE-WORKSPACE-PANE")?
                    .is_some(),
                "OpenExcerpts must render the canonical source pane: {name}"
            );
            cx.capture_screenshot(window.into())?
                .save(PathBuf::from(&output_directory).join(format!("{name}.png")))?;
        }
        window.update(cx, |_, window, cx| {
            workspace.focus_handle(cx).dispatch_action(
                &workspace::collaborative_review::ShowCollaborativeReview,
                window,
                cx,
            );
        })?;
        cx.run_until_parked();
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "NATIVE-WORKSPACE-PANE")?
                .is_none(),
            "returning to review should unmount, not destroy, the native editor pane"
        );
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join("review-source-return.png"))?;

        // Closing the last item opened from review must release the native pane instead of
        // leaving an empty pane where the retained review provider belongs.
        let native_pane = cx.read(|cx| workspace.read(cx).active_pane().clone());
        let source_index = cx
            .read(|cx| {
                native_pane
                    .read(cx)
                    .items()
                    .position(|item| item.item_id() == source_editor.entity_id())
            })
            .context("the source editor must remain a retained native item")?;
        window.update(cx, |_, window, cx| {
            native_pane.update(cx, |pane, cx| {
                pane.activate_item(source_index, true, true, window, cx);
            });
        })?;
        cx.run_until_parked();
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        // `PaneGroup` paints the pane through a cached `AnyView`, so an unchanged remount does not
        // re-record its debug bounds. Assert the presentation state the region renders from.
        anyhow::ensure!(
            cx.read(|cx| {
                let workspace = workspace.read(cx);
                workspace.collaborative_native_editor_active(cx)
                    && workspace.active_item_as::<editor::Editor>(cx) == Some(source_editor.clone())
            }),
            "reactivating the retained source item must present it in the review region again"
        );
        window.update(cx, |_, window, cx| {
            use gpui::TaskExt as _;
            native_pane.update(cx, |pane, cx| {
                pane.close_all_items(
                    &workspace::pane::CloseAllItems {
                        save_intent: Some(workspace::SaveIntent::Skip),
                        close_pinned: true,
                    },
                    window,
                    cx,
                )
                .detach_and_log_err(cx);
            })
        })?;
        cx.run_until_parked();
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "NATIVE-WORKSPACE-PANE")?
                .is_none()
                && !cx.read(|cx| workspace.read(cx).collaborative_native_editor_active(cx)),
            "closing every review-opened item must release the native pane"
        );
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "COLLABORATIVE-REVIEW-REGION")?
                .is_some()
                && cx
                    .debug_bounds(window.into(), "COLLABORATIVE-TIMELINE-REGION")?
                    .is_some(),
            "closing every review-opened item must restore the review and conversation regions"
        );
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join("review-source-closed.png"))?;

        let project_review_id = cx.read(|cx| {
            workspace
                .read(cx)
                .collaborative_review()
                .selected_view()
                .map(|view| view.entity_id())
        });
        window.update(cx, |_, window, cx| {
            workspace.focus_handle(cx).dispatch_action(
                &workspace::collaborative_review::ToggleCollaborativeReview,
                window,
                cx,
            );
        })?;
        cx.run_until_parked();
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "COLLABORATIVE-REVIEW-REGION")?
                .is_none(),
            "review should collapse before footer activation"
        );
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join("footer-before-diff.png"))?;
        let diff_button = cx
            .debug_bounds(window.into(), "NATIVE-PROJECT-DIFF-STATUS")?
            .context("footer diff control missing")?;
        println!("Footer diff click at {diff_button:?}");
        cx.simulate_click(window.into(), diff_button.center(), Modifiers::default());
        cx.run_until_parked();
        for _ in 0..3 {
            window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
            cx.run_until_parked();
        }
        anyhow::ensure!(
            cx.debug_bounds(window.into(), "COLLABORATIVE-REVIEW-REGION")?
                .is_some(),
            "footer diff action must reveal the native project review; selected/visible: {:?}",
            cx.read(|cx| (
                workspace.read(cx).collaborative_review().selected_slot(),
                workspace
                    .read(cx)
                    .collaborative_review_view(cx)
                    .map(|view| view.entity_id())
            ))
        );
        anyhow::ensure!(
            cx.read(|cx| workspace
                .read(cx)
                .collaborative_review_view(cx)
                .map(|view| view.entity_id()))
                == project_review_id,
            "footer must reuse the exact native ProjectDiff view"
        );
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(&output_directory).join("footer-diff-reopened.png"))?;
        for (trigger, menu, name) in [
            (
                "NATIVE-PROJECT-PICKER",
                "NATIVE-RECENT-PROJECTS-MENU",
                "footer-project-menu",
            ),
            (
                "NATIVE-WORKTREE-PICKER",
                "NATIVE-WORKTREE-MENU",
                "footer-worktree-menu",
            ),
            (
                "NATIVE-BRANCH-PICKER",
                "NATIVE-GIT-PICKER-MENU",
                "footer-branch-menu",
            ),
            if multi_acp {
                (
                    "ACP-CONFIG-OPTION-mode",
                    "NATIVE-FOCUSED-ACP-OPTION",
                    "footer-acp-menu",
                )
            } else {
                (
                    "COLLABORATIVE-MODEL-CONTROLS",
                    "NATIVE-FOCUSED-MODEL-OPTION",
                    "footer-model-menu",
                )
            },
        ] {
            let button = cx
                .debug_bounds(window.into(), trigger)?
                .with_context(|| format!("missing {trigger}"))?;
            let footer = cx
                .debug_bounds(window.into(), "WORKSPACE-STATUS-BAR")?
                .context("footer missing")?;
            anyhow::ensure!(
                footer.contains(&button.center()),
                "{trigger} cannot be reached in the footer"
            );
            cx.simulate_click(window.into(), button.center(), Modifiers::default());
            cx.run_until_parked();
            for _ in 0..3 {
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
            }
            let menu_bounds = cx
                .debug_bounds(window.into(), menu)?
                .with_context(|| format!("{trigger} did not open {menu}"))?;
            anyhow::ensure!(
                menu_bounds.bottom() <= button.top() + px(1.),
                "{menu} must open above the footer button: {menu_bounds:?}, button {button:?}"
            );
            cx.capture_screenshot(window.into())?
                .save(PathBuf::from(&output_directory).join(format!("{name}.png")))?;
            cx.simulate_keystrokes(window.into(), "escape");
            cx.run_until_parked();
            anyhow::ensure!(
                cx.debug_bounds(window.into(), menu)?.is_none(),
                "Escape did not dismiss {menu}"
            );
            anyhow::ensure!(
                cx.read(|cx| agent_panel.read(cx).active_thread_id(cx)) == visual_thread_id,
                "{trigger} changed the active conversation"
            );
            println!(
                "Native footer: {trigger} opened {menu} above its button and dismissed without changing conversation"
            );
        }
        if multi_acp {
            cx.activate_native_window(window.into())?;
            for _ in 0..3 {
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
            }
            anyhow::ensure!(
                window.update(cx, |_, window, _| window.is_window_active())?,
                "native ACP focus-scroll validation requires an active host window"
            );
            let native_thread = cx
                .read(|cx| thread_view.read(cx).active_thread().cloned())
                .context("native ACP thread missing before keyboard navigation")?;
            let mut mode_focused = false;
            for _ in 0..64 {
                window.update(cx, |_, window, cx| window.focus_next(cx))?;
                cx.run_until_parked();
                window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                cx.run_until_parked();
                mode_focused = window.update(cx, |_, window, cx| {
                    native_thread
                        .read(cx)
                        .focused_collaborative_config_option(window, cx)
                        .is_some_and(|id| id.0.as_ref() == "mode")
                })?;
                if mode_focused {
                    break;
                }
            }
            anyhow::ensure!(
                mode_focused,
                "native mode control is not keyboard reachable"
            );
            for (config_id, selector, name) in [
                (
                    "model",
                    "ACP-CONFIG-OPTION-model",
                    "footer-acp-model-focused",
                ),
                (
                    "web_search",
                    "NATIVE-SWITCH-CONTROL",
                    "footer-acp-boolean-focused",
                ),
            ] {
                window.update(cx, |_, window, cx| window.focus_next(cx))?;
                cx.run_until_parked();
                for _ in 0..3 {
                    window.update(cx, |_, window, cx| window.simulate_next_frame(cx))?;
                    cx.run_until_parked();
                }
                anyhow::ensure!(
                    window.update(cx, |_, window, cx| {
                        native_thread
                            .read(cx)
                            .focused_collaborative_config_option(window, cx)
                            .is_some_and(|id| id.0.as_ref() == config_id)
                    })?,
                    "keyboard navigation did not focus native option {config_id}"
                );
                let row = cx
                    .debug_bounds(window.into(), "ACP-SINGLE-ROW-CONFIG-OPTIONS")?
                    .context("native configuration row missing after keyboard navigation")?;
                let control = cx
                    .debug_bounds(window.into(), selector)?
                    .with_context(|| format!("missing focused control {selector}"))?;
                anyhow::ensure!(
                    row.contains(&control.origin)
                        && control.right() <= row.right()
                        && control.bottom() <= row.bottom(),
                    "keyboard navigation must reveal {selector}: row={row:?}, control={control:?}"
                );
                cx.capture_screenshot(window.into())?
                    .save(PathBuf::from(&output_directory).join(format!("{name}.png")))?;
                println!("Multi-ACP keyboard navigation revealed {selector}: {control:?}");
            }
        }
    }

    if matches!(variant, CollaborativeVisualVariant::Expanded)
        && let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
    {
        thread_view.update(cx, |conversation, cx| {
            conversation.scroll_thread_to_entry_for_tests(6, cx);
        });
        window.update(cx, |_, window, cx| {
            let native_thread = thread_view
                .read(cx)
                .active_thread()
                .context("visual conversation should have an active native thread")?;
            let editor = native_thread.read(cx).message_editor.clone();
            window.focus(&editor.focus_handle(cx), cx);
            window.refresh();
            anyhow::Ok(())
        })??;
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(output_directory).join("activity-error-focus.png"))?;
    }

    if matches!(variant, CollaborativeVisualVariant::Expanded)
        && let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
    {
        let output_directory = PathBuf::from(output_directory);
        let row = cx
            .debug_bounds(window.into(), "COLLABORATIVE-NAVIGATION-PROJECT-ROW")?
            .context("native project navigation row should render")?;
        cx.simulate_mouse_move(
            window.into(),
            row.center(),
            None,
            gpui::Modifiers::default(),
        );
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(output_directory.join("navigation-hover.png"))?;
        cx.simulate_mouse_down(
            window.into(),
            row.center(),
            gpui::MouseButton::Left,
            gpui::Modifiers::default(),
        );
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(output_directory.join("navigation-pressed.png"))?;
        // Release outside the row so this state capture does not navigate away
        // from the ACP conversation or change any persisted project selection.
        cx.simulate_mouse_up(
            window.into(),
            gpui::point(px(0.), px(0.)),
            gpui::MouseButton::Left,
            gpui::Modifiers::default(),
        );
        cx.run_until_parked();
        window.update(cx, |_, window, _| window.blur())?;
        cx.simulate_keystrokes(window.into(), "tab");
        window.update(cx, |_, window, cx| window.focus_next(cx))?;
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(output_directory.join("navigation-keyboard-focus.png"))?;

        let composer_focus = window.update(cx, |_, window, cx| {
            let native_thread = thread_view
                .read(cx)
                .active_thread()
                .context("visual conversation should have an active native thread")?;
            let focus = native_thread.read(cx).message_editor.focus_handle(cx);
            window.focus(&focus, cx);
            anyhow::Ok(focus)
        })??;
        cx.simulate_keystrokes(window.into(), "f6");
        cx.run_until_parked();
        anyhow::ensure!(
            window.update(cx, |_, window, _| !composer_focus.is_focused(window))?,
            "native F6 must leave the composer for the visible review region"
        );
        cx.capture_screenshot(window.into())?
            .save(output_directory.join("review-region-keyboard-focus.png"))?;
        cx.simulate_keystrokes(window.into(), "shift-f6");
        cx.run_until_parked();
        anyhow::ensure!(
            window.update(cx, |_, window, _| composer_focus.is_focused(window))?,
            "native Shift-F6 must return to the retained composer"
        );
        cx.capture_screenshot(window.into())?
            .save(output_directory.join("composer-region-keyboard-focus.png"))?;
    }

    if matches!(variant, CollaborativeVisualVariant::EnlargedCompact)
        && let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
    {
        let projects = cx
            .debug_bounds(window.into(), "COLLABORATIVE-RAIL-PROJECTS")?
            .context("project navigation section should render")?;
        cx.simulate_event(
            window.into(),
            gpui::ScrollWheelEvent {
                position: projects.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), -projects.size.height)),
                touch_phase: gpui::TouchPhase::Moved,
                ..Default::default()
            },
        );
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(PathBuf::from(output_directory).join("navigation-enlarged-children.png"))?;
    }

    if matches!(
        variant,
        CollaborativeVisualVariant::Expanded
            | CollaborativeVisualVariant::Compact
            | CollaborativeVisualVariant::EnlargedCompact
            | CollaborativeVisualVariant::Narrow
    ) && let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
    {
        let output_directory = PathBuf::from(output_directory);
        let suffix = match variant {
            CollaborativeVisualVariant::Expanded => "normal",
            CollaborativeVisualVariant::Compact => "compact",
            CollaborativeVisualVariant::EnlargedCompact => "enlarged",
            _ => "narrow",
        };
        cx.update(|cx| {
            channel::ChannelStore::global(cx).update(cx, |store, cx| {
                let mut channels = vec![(client::ChannelId(700), "Design community".into(), None)];
                channels.extend((701..717).map(|id| {
                    (
                        client::ChannelId(id),
                        format!("Workspace channel {}", id - 700).into(),
                        Some(client::ChannelId(700)),
                    )
                }));
                store.set_channels_for_tests(channels, cx);
                store.update_latest_notes_version(
                    client::ChannelId(701),
                    1,
                    &Default::default(),
                    cx,
                );
            });
        });
        cx.run_until_parked();
        let rail = cx
            .debug_bounds(window.into(), "COLLABORATIVE-RAIL-SCROLL")?
            .context("navigation scroll surface")?;
        cx.simulate_event(
            window.into(),
            gpui::ScrollWheelEvent {
                position: rail.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(-5000.))),
                touch_phase: gpui::TouchPhase::Moved,
                ..Default::default()
            },
        );
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(output_directory.join(format!("hierarchical-channels-{suffix}.png")))?;
        if !matches!(variant, CollaborativeVisualVariant::Narrow) {
            window.update(cx, |_, window, cx| {
                thread
                    .read(cx)
                    .action_log()
                    .clone()
                    .update(cx, |log, cx| log.keep_all_edits(None, cx));
                workspace.focus_handle(cx).dispatch_action(
                    &workspace::collaborative_review::SelectAgentReview,
                    window,
                    cx,
                );
            })?;
            cx.run_until_parked();
            for _ in 0..10 {
                cx.advance_clock(Duration::from_millis(100));
                cx.run_until_parked();
            }
            cx.capture_screenshot(window.into())?
                .save(output_directory.join(format!("review-empty-{suffix}.png")))?;
            let button = cx
                .debug_bounds(window.into(), "AGENT-REVIEW-COMPOSER-ACTION")?
                .context("review composer action")?;
            cx.simulate_mouse_down(
                window.into(),
                button.center(),
                gpui::MouseButton::Left,
                Default::default(),
            );
            cx.simulate_mouse_up(
                window.into(),
                button.center(),
                gpui::MouseButton::Left,
                Default::default(),
            );
            cx.run_until_parked();
            window.update(cx, |_, window, cx| -> Result<()> {
                let view = agent_panel
                    .read(cx)
                    .active_thread_view(cx)
                    .context("review selected its task")?;
                anyhow::ensure!(
                    view.read(cx).thread == thread,
                    "review selected another canonical thread"
                );
                anyhow::ensure!(
                    view.read(cx)
                        .message_editor
                        .focus_handle(cx)
                        .contains_focused(window, cx),
                    "review did not focus the canonical composer"
                );
                Ok(())
            })??;
            cx.capture_screenshot(window.into())?
                .save(output_directory.join(format!("review-composer-focus-{suffix}.png")))?;
            if matches!(variant, CollaborativeVisualVariant::Expanded) {
                let editor = cx
                    .read(|cx| {
                        agent_panel
                            .read(cx)
                            .active_thread_view(cx)
                            .map(|view| view.read(cx).message_editor.clone())
                    })
                    .context("native composer for recovery capture")?;
                for (prompt, failure, image_name) in [
                    (
                        "Check failure recovery",
                        true,
                        "composer-transport-error.png",
                    ),
                    (
                        "Continue after the transport failure",
                        false,
                        "composer-transport-recovered.png",
                    ),
                ] {
                    window.update(cx, |_, window, cx| {
                        editor.update(cx, |editor, cx| editor.set_text(prompt, window, cx));
                    })?;
                    cx.run_until_parked();
                    let send = cx
                        .debug_bounds(window.into(), "ICON-Send")?
                        .context("native send control after failure")?;
                    cx.simulate_mouse_down(
                        window.into(),
                        send.center(),
                        gpui::MouseButton::Left,
                        Default::default(),
                    );
                    cx.simulate_mouse_up(
                        window.into(),
                        send.center(),
                        gpui::MouseButton::Left,
                        Default::default(),
                    );
                    cx.run_until_parked();
                    let session_id = cx.read(|cx| {
                        anyhow::ensure!(
                            thread.read(cx).status() == acp_thread::ThreadStatus::Generating,
                            "native send did not start the recovery turn"
                        );
                        anyhow::Ok(thread.read(cx).session_id().clone())
                    })?;
                    // Real Git checkpoint processes must finish before ACP receives the prompt;
                    // advancing GPUI's simulated clock alone cannot make those processes finish.
                    let transport_deadline = std::time::Instant::now() + Duration::from_secs(30);
                    while std::time::Instant::now() < transport_deadline {
                        if connection.has_pending_turn(&session_id) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(10));
                        cx.advance_clock(Duration::from_millis(100));
                        cx.run_until_parked();
                    }
                    if !connection.has_pending_turn(&session_id) {
                        eprintln!(
                            "Native recovery prompt did not finish Git checkpoint preparation"
                        );
                        cx.capture_screenshot(window.into())?.save(
                            output_directory.join("composer-transport-pending-timeout.png"),
                        )?;
                    }
                    anyhow::ensure!(
                        connection.has_pending_turn(&session_id),
                        "native Send did not reach the pending transport request"
                    );
                    if failure {
                        connection.fail_turn(
                            &session_id,
                            anyhow::anyhow!("Audit transport disconnected"),
                        );
                    } else {
                        connection.end_turn(session_id, acp::StopReason::EndTurn);
                    }
                    let completion_deadline = std::time::Instant::now() + Duration::from_secs(30);
                    while std::time::Instant::now() < completion_deadline {
                        std::thread::sleep(Duration::from_millis(10));
                        cx.run_until_parked();
                        if cx.read(|cx| thread.read(cx).status()) == acp_thread::ThreadStatus::Idle
                        {
                            break;
                        }
                    }
                    cx.capture_screenshot(window.into())?
                        .save(output_directory.join(image_name))?;
                    anyhow::ensure!(
                        cx.read(|cx| thread.read(cx).status()) == acp_thread::ThreadStatus::Idle,
                        "native turn did not settle after failure/recovery"
                    );
                    anyhow::ensure!(
                        cx.debug_bounds(window.into(), "ACP-THREAD-ERROR")?
                            .is_some()
                            == failure,
                        "native error visibility did not follow failure/recovery"
                    );
                    let review = cx
                        .debug_bounds(window.into(), "COLLABORATIVE-REVIEW-REGION")?
                        .context("review during native recovery")?;
                    let review_message = cx
                        .debug_bounds(window.into(), "AGENT-REVIEW-EMPTY-MESSAGE")?
                        .context("review message during native recovery")?;
                    anyhow::ensure!(
                        review.contains(&review_message.origin)
                            && review_message.right() <= review.right()
                            && review_message.bottom() <= review.bottom(),
                        "review message overflows its native pane during recovery"
                    );
                    println!(
                        "Native composer: {image_name} verified through Send and ACP completion"
                    );
                }
                thread.update(cx, |thread, cx| {
                    thread.handle_session_update(
                        acp::SessionUpdate::ToolCall(
                            acp::ToolCall::new("empty-review-recovery", "Check review changes")
                                .status(acp::ToolCallStatus::Failed),
                        ),
                        cx,
                    )
                })?;
                cx.run_until_parked();
                cx.capture_screenshot(window.into())?
                    .save(output_directory.join("review-empty-error.png"))?;
                thread.update(cx, |thread, cx| {
                    thread.handle_session_update(
                        acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
                            "empty-review-recovery",
                            acp::ToolCallUpdateFields::new().status(acp::ToolCallStatus::Completed),
                        )),
                        cx,
                    )
                })?;
                cx.run_until_parked();
                cx.capture_screenshot(window.into())?
                    .save(output_directory.join("review-empty-recovered.png"))?;
                cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                    store.update_user_settings(cx, |content| {
                        content.theme.experimental_theme_overrides = Some(serde_json::from_value(serde_json::json!({
                            "text.accent": if collaborative_visual_theme().ends_with("Dark") { "#A9E6E0" } else { "#005F73" }
                        })).expect("controlled custom accent"));
                    });
                });
                cx.run_until_parked();
                cx.capture_screenshot(window.into())?
                    .save(output_directory.join("review-empty-custom-accent.png"))?;
            }
        }
    }

    workspace.update(cx, |workspace, cx| -> Result<()> {
        anyhow::ensure!(
            workspace.unregister_collaborative_composer_provider(composer_registration, cx),
            "collaborative visual composer registration was replaced before cleanup"
        );
        anyhow::ensure!(
            workspace.unregister_collaborative_timeline_provider(timeline_registration, cx),
            "collaborative visual timeline registration was replaced before cleanup"
        );
        anyhow::ensure!(
            workspace.unregister_collaborative_participant_provider(participant_registration, cx),
            "collaborative visual participant registration was replaced before cleanup"
        );
        anyhow::ensure!(
            workspace.unregister_collaborative_review_provider(project_review_registration, cx),
            "collaborative visual project review registration was replaced before cleanup"
        );
        anyhow::ensure!(
            workspace.unregister_collaborative_review_provider(agent_review_registration, cx),
            "collaborative visual agent review registration was replaced before cleanup"
        );
        Ok(())
    })?;

    if let Some(thread_id) = visual_thread_id {
        window.update(cx, |_, window, cx| {
            agent_panel.update(cx, |panel, cx| {
                panel.remove_thread_without_activating_draft(thread_id, window, cx);
            });
        })?;
    }
    cx.run_until_parked();

    workspace.update(cx, |workspace, cx| {
        let project = workspace.project().clone();
        project.update(cx, |project, cx| {
            let worktree_ids = project
                .worktrees(cx)
                .map(|worktree| worktree.read(cx).id())
                .collect::<Vec<_>>();
            for worktree_id in worktree_ids {
                project.remove_worktree(worktree_id, cx);
            }
        });
    });
    cx.run_until_parked();

    cx.update_window(window.into(), |_, window, _| window.remove_window())?;
    cx.run_until_parked();

    drop(thread);
    drop(thread_view);
    drop(agent_panel);
    drop(git_panel);
    drop(workspace);
    drop(sidebar);
    drop(project);
    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(())
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn capture_collaborative_accessibility_state(
    name: &str,
    viewport: gpui::Size<gpui::Pixels>,
    window: gpui::AnyWindowHandle,
    cx: &mut VisualTestAppContext,
) -> Result<()> {
    let Some(output_directory) = std::env::var_os("VISUAL_TEST_OUTPUT_DIR") else {
        return Ok(());
    };
    let output_directory = PathBuf::from(output_directory);
    std::fs::create_dir_all(&output_directory)?;
    let composer = cx
        .debug_bounds(window, "COLLABORATIVE-COMPOSER")?
        .context("resized composer should render")?;
    let send = cx
        .debug_bounds(window, "ICON-Send")?
        .context("resized native send control should render")?;
    let composer_footer = cx
        .debug_bounds(window, "COLLABORATIVE-COMPOSER-FOOTER")?
        .context("resized native composer footer should render")?;
    let model_controls = cx
        .debug_bounds(window, "COLLABORATIVE-MODEL-CONTROLS")?
        .context("resized native model controls should render")?;
    if std::env::var_os("COLLABORATIVE_VISUAL_MULTI_ACP").is_some() {
        let row = cx
            .debug_bounds(window, "ACP-SINGLE-ROW-CONFIG-OPTIONS")?
            .context("multi-ACP fixture did not mount the native configuration row")?;
        let button_height = cx.update_window(window, |_, window, _| {
            ui::ButtonSize::Default.rems() * window.rem_size()
        })?;
        anyhow::ensure!(
            row.size.height <= button_height && row.size.width <= model_controls.size.width,
            "multi-ACP row exceeds its native footer allocation: {row:?}, controls={model_controls:?}"
        );
        for selector in [
            "ACP-CONFIG-OPTION-mode",
            "ACP-CONFIG-OPTION-model",
            "ACP-CONFIG-OPTION-web_search",
        ] {
            anyhow::ensure!(
                cx.debug_bounds(window, selector)?.is_some(),
                "missing native configuration selector: {selector}"
            );
        }
        println!(
            "Multi-ACP production footer: three native configuration selectors, one row {row:?}"
        );
    }
    let actual = cx.capture_screenshot(window)?;
    actual.save(output_directory.join(format!("{name}.png")))?;
    anyhow::ensure!(
        actual.width() == (f32::from(viewport.width) * 2.) as u32
            && actual.height() == (f32::from(viewport.height) * 2.) as u32,
        "{name}: native capture must retain the requested viewport without resampling"
    );
    anyhow::ensure!(
        composer_footer.contains(&send.origin)
            && send.right() <= composer_footer.right()
            && send.bottom() <= composer_footer.bottom()
            && composer_footer.right() <= composer.right()
            && composer_footer.bottom() <= viewport.height,
        "{name}: send control is clipped: send={send:?}, footer={composer_footer:?}, composer={composer:?}, viewport={viewport:?}"
    );
    anyhow::ensure!(
        composer.right() <= viewport.width && composer.bottom() <= viewport.height,
        "{name}: composer {composer:?} extends beyond {viewport:?}"
    );
    anyhow::ensure!(
        model_controls.size.width >= px(48.)
            && composer_footer.contains(&model_controls.origin)
            && model_controls.right() <= composer_footer.right()
            && model_controls.bottom() <= composer_footer.bottom(),
        "{name}: model controls must retain a visible native hit target: {model_controls:?} / {composer_footer:?}"
    );
    let project_controls = cx
        .debug_bounds(window, "COLLABORATIVE-PROJECT-CONTROLS")?
        .context("native project controls should remain visible")?;
    let status_overflow = cx
        .debug_bounds(window, "COLLABORATIVE-STATUS-OVERFLOW")?
        .context("native status overflow should remain visible")?;
    anyhow::ensure!(
        project_controls.contains(&status_overflow.origin)
            && status_overflow.right() <= project_controls.right()
            && status_overflow.bottom() <= project_controls.bottom(),
        "{name}: status overflow is clipped by project controls: {status_overflow:?} / {project_controls:?}"
    );
    let account = cx
        .debug_bounds(window, "COLLABORATIVE-ACCOUNT-SURFACE")?
        .context("account surface should remain visible")?;
    let footer = cx
        .debug_bounds(window, "WORKSPACE-STATUS-BAR")?
        .context("native status bar should remain visible")?;
    if let Some(execution) = cx.debug_bounds(window, "COLLABORATIVE-PARTICIPANT-STATUS")? {
        let execution_icon = cx
            .debug_bounds(window, "COLLABORATIVE-PARTICIPANT-STATUS-ICON")?
            .context("compact execution icon should remain visible")?;
        anyhow::ensure!(
            execution.contains(&execution_icon.origin)
                && execution_icon.right() <= execution.right()
                && execution_icon.bottom() <= execution.bottom(),
            "{name}: compact execution icon is clipped: {execution_icon:?} / {execution:?}"
        );
        anyhow::ensure!(
            footer.contains(&execution.origin)
                && execution.right() <= footer.right()
                && execution.bottom() <= footer.bottom(),
            "{name}: execution details extend outside the native footer: {execution:?} / {footer:?}"
        );
    }
    let timeline = cx
        .debug_bounds(window, "COLLABORATIVE-TIMELINE-REGION")?
        .context("timeline should remain visible")?;
    anyhow::ensure!(
        footer.left() == timeline.left()
            && footer.right() == timeline.right()
            && footer.top() >= composer.bottom()
            && (footer.bottom() - timeline.bottom()).abs() <= gpui::px(1.),
        "{name}: native status bar must stay below the composer within the conversation column"
    );
    for selector in [
        "NATIVE-PROJECT-PICKER",
        "NATIVE-WORKTREE-PICKER",
        "NATIVE-BRANCH-PICKER",
        "NATIVE-PROJECT-DIFF-STATUS",
    ] {
        let control = cx
            .debug_bounds(window, selector)?
            .with_context(|| format!("{name}: missing {selector}"))?;
        if footer.size.width >= px(650.) && selector == "NATIVE-BRANCH-PICKER" {
            let branch_content_width = cx.update_window(window, |_, window, cx| {
                let mut font = theme_settings::ThemeSettings::get_global(cx)
                    .ui_font
                    .clone();
                font.weight = gpui::FontWeight::MEDIUM;
                let label = window.text_system().shape_line(
                    "main".into(),
                    ui::TextSize::Small.rems(cx).to_pixels(window.rem_size()),
                    &[gpui::TextRun {
                        len: "main".len(),
                        font,
                        ..Default::default()
                    }],
                    None,
                );
                label.width
                    + ui::DynamicSpacing::Base08
                        .rems(cx)
                        .to_pixels(window.rem_size())
                        * 2.
                    + px(2.)
            })?;
            anyhow::ensure!(
                (control.size.width - branch_content_width).abs() <= px(2.)
                    && model_controls.size.width >= px(120.),
                "{name}: wide footer must fit the complete branch label without excess allocation and retain model space: branch={control:?}, expected={branch_content_width:?}, model={model_controls:?}"
            );
        }
        anyhow::ensure!(
            control.left() >= footer.left()
                && control.right() <= footer.right()
                && control.top() >= footer.top()
                && control.bottom() <= footer.bottom(),
            "{name}: {selector} is clipped: {control:?}, footer {footer:?}"
        );
    }
    if let Some(review) = cx.debug_bounds(window, "COLLABORATIVE-REVIEW-REGION")? {
        let header = cx
            .debug_bounds(window, "COLLABORATIVE-REVIEW-HEADER")?
            .context("expanded review should have its own title region")?;
        anyhow::ensure!(
            header.left() == review.left()
                && header.right() == review.right()
                && header.bottom() <= review.top(),
            "{name}: review title {header:?} is not aligned above its pane {review:?}"
        );
        anyhow::ensure!(
            review.bottom() == timeline.bottom(),
            "{name}: expanded review should reach the bottom alongside the conversation footer"
        );
    } else {
        anyhow::ensure!(
            cx.debug_bounds(window, "COLLABORATIVE-REVIEW-HEADER")?
                .is_none(),
            "{name}: collapsed review must release its title region"
        );
    }
    let tasks = cx
        .debug_bounds(window, "COLLABORATIVE-RAIL-SCROLL")?
        .context("task navigation should remain visible")?;
    anyhow::ensure!(
        account.bottom() <= viewport.height && tasks.bottom() <= account.top(),
        "{name}: navigation overlaps or clips the account controls"
    );
    if name == "narrow" {
        anyhow::ensure!(
            cx.debug_bounds(window, "COLLABORATIVE-REVIEW-REGION")?
                .is_none(),
            "narrow window should collapse the review pane"
        );
    }
    println!(
        "  Collaborative {name}: native {}×{} capture; composer and send control visible",
        actual.width(),
        actual.height()
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn capture_collaborative_reference_comparison(
    name: &str,
    window: gpui::AnyWindowHandle,
    cx: &mut VisualTestAppContext,
) -> Result<()> {
    cx.update_window(window, |_, window, _| window.refresh())?;
    cx.run_until_parked();
    for selector in [
        "PLATFORM-TITLE-BAR",
        "COLLABORATIVE-NATIVE-ACP-TIMELINE",
        "COLLABORATIVE-PARTICIPANT-AVATAR",
        "COLLABORATIVE-NATIVE-ACP-COMPOSER",
        "ICON-Send",
    ] {
        anyhow::ensure!(
            cx.debug_bounds(window, selector)?.is_some(),
            "{name} did not render required native semantic surface {selector}"
        );
    }
    let timeline_bounds = cx
        .debug_bounds(window, "COLLABORATIVE-TIMELINE-REGION")?
        .context("collaborative timeline bounds are unavailable")?;
    let composer_bounds = cx
        .debug_bounds(window, "COLLABORATIVE-COMPOSER")?
        .context("collaborative composer bounds are unavailable")?;
    let send_bounds = cx
        .debug_bounds(window, "ICON-Send")?
        .context("native send button bounds are unavailable")?;
    let footer_bounds = cx
        .debug_bounds(window, "COLLABORATIVE-COMPOSER-FOOTER")?
        .context("native composer footer bounds are unavailable")?;
    anyhow::ensure!(
        footer_bounds.contains(&send_bounds.origin)
            && send_bounds.bottom() <= footer_bounds.bottom()
            && send_bounds.right() <= footer_bounds.right()
            && footer_bounds.right() <= timeline_bounds.right()
            && footer_bounds.bottom() <= timeline_bounds.bottom()
            && composer_bounds.bottom() <= footer_bounds.top()
            && composer_bounds.right() == timeline_bounds.right(),
        "native composer footer controls are clipped, overlap the input, or extend into the review pane: send={send_bounds:?}, footer={footer_bounds:?}, composer={composer_bounds:?}, timeline={timeline_bounds:?}"
    );
    let geometry_report = if name == "screenshot-1" {
        let review_bounds = cx
            .debug_bounds(window, "COLLABORATIVE-REVIEW-REGION")?
            .context("expanded collaborative review bounds are unavailable")?;
        let physical_review_boundary = f32::from(review_bounds.origin.x) * 2.;
        anyhow::ensure!(
            (physical_review_boundary - 1221.).abs() <= 2.,
            "expanded review boundary {physical_review_boundary:.1}px is outside 1221±2px"
        );
        format!(
            "    geometry: expanded review boundary {physical_review_boundary:.1}px (reference 1221px, tolerance ±2px)\n"
        )
    } else {
        anyhow::ensure!(
            cx.debug_bounds(window, "COLLABORATIVE-REVIEW-REGION")?
                .is_none(),
            "collapsed visual variant still renders the review region"
        );
        let physical_timeline_right = f32::from(timeline_bounds.right()) * 2.;
        anyhow::ensure!(
            (physical_timeline_right - 1928.).abs() <= 2.,
            "collapsed timeline right edge {physical_timeline_right:.1}px does not consume the released width"
        );
        format!(
            "    geometry: review absent; timeline right edge {physical_timeline_right:.1}px (viewport 1928px, tolerance ±2px)\n"
        )
    };
    let viewport = cx.update_window(window, |_, window, _| window.viewport_size())?;
    anyhow::ensure!(
        composer_bounds.bottom() <= viewport.height && composer_bounds.right() <= viewport.width,
        "{name}: composer extends beyond the visible viewport"
    );
    let native_actual = cx.capture_screenshot(window)?;
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("zed crate should live below the workspace root")?;
    let evidence_directory = std::env::var_os("VISUAL_TEST_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("test-results/collaborative-workspace"));
    std::fs::create_dir_all(&evidence_directory)?;
    record_collaborative_host_window_evidence(name, &evidence_directory)?;
    let reference_path = workspace_root
        .join(".agents/specs/collaborative-workspace/screenshots")
        .join(format!("{name}.png"));
    let reference = image::open(&reference_path)?.to_rgba8();
    native_actual.save(evidence_directory.join(format!("native-{name}.png")))?;
    anyhow::ensure!(
        native_actual.dimensions() == reference.dimensions(),
        "native capture must match the reference viewport without resampling"
    );
    let actual = native_actual;
    actual.save(evidence_directory.join(format!("actual-{name}.png")))?;
    let comparison = compare_images(&actual, &reference);
    let structural_similarity = collaborative_structural_similarity(&actual, &reference);
    comparison
        .diff_image
        .save(evidence_directory.join(format!("diff-{name}.png")))?;
    let amplified_diff = RgbaImage::from_fn(reference.width(), reference.height(), |x, y| {
        let actual_pixel = actual.get_pixel(x, y);
        let reference_pixel = reference.get_pixel(x, y);
        image::Rgba([
            actual_pixel[0]
                .abs_diff(reference_pixel[0])
                .saturating_mul(4),
            actual_pixel[1]
                .abs_diff(reference_pixel[1])
                .saturating_mul(4),
            actual_pixel[2]
                .abs_diff(reference_pixel[2])
                .saturating_mul(4),
            255,
        ])
    });
    amplified_diff.save(evidence_directory.join(format!("amplified-diff-{name}.png")))?;
    let mut side_by_side = RgbaImage::new(reference.width() * 2, reference.height());
    side_by_side.copy_from(&reference, 0, 0)?;
    side_by_side.copy_from(&actual, reference.width(), 0)?;
    side_by_side.save(evidence_directory.join(format!("side-by-side-{name}.png")))?;
    let mut region_report = validate_collaborative_regions(name, &actual, &reference)?;
    region_report.push_str(&geometry_report);
    std::fs::write(
        evidence_directory.join(format!("region-metrics-{name}.txt")),
        &region_report,
    )?;
    print!("{region_report}");
    println!(
        "  Collaborative {name}: {:.2}% structural similarity, {:.2}% exact-pixel match ({} differing pixels)",
        structural_similarity * 100.,
        comparison.match_percentage * 100.,
        comparison.diff_pixel_count
    );
    // The references predate Fractal, so palette-dependent pixel similarity is
    // diagnostic. Native content, contrast and geometry remain mandatory above.
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn record_collaborative_host_window_evidence(name: &str, evidence_directory: &Path) -> Result<()> {
    const CAPTURE_VARIABLE: &str = "COLLABORATIVE_NATIVE_WINDOW_CAPTURE";
    if let Some(path) = std::env::var_os(CAPTURE_VARIABLE) {
        let capture = image::open(&path)?.to_rgba8();
        anyhow::ensure!(
            capture.width() > 0 && capture.height() > 0,
            "native host-window evidence is empty"
        );
        capture.save(evidence_directory.join(format!("host-window-{name}.png")))?;
        return Ok(());
    }

    std::fs::write(
        evidence_directory.join("host-window-chrome-evidence.md"),
        format!(
            "# Collaborative Workspace host-window evidence\n\n\
The deterministic `actual-{name}.png` capture is GPUI content raster output. It deliberately \
does not contain the macOS-owned traffic-light controls. The visual runner separately asserts \
that `PlatformTitleBar` owns the title-bar surface.\n\n\
For full-window evidence, grant the terminal Screen Recording permission, capture the visible \
Rust-product window through the macOS screenshot UI, and rerun with \
`{CAPTURE_VARIABLE}=/absolute/path/to/capture.png`. The runner validates and copies that native \
capture to `host-window-{name}.png`; it never synthesizes window controls.\n"
        ),
    )?;
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
#[derive(Clone, Copy)]
struct CollaborativeRasterMetrics {
    contrast: f64,
    edge_density: f64,
    foreground_coverage: f64,
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn collaborative_raster_metrics(
    image: &RgbaImage,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> CollaborativeRasterMetrics {
    let left = left.min(image.width());
    let right = right.clamp(left + 1, image.width());
    let top = top.min(image.height());
    let bottom = bottom.clamp(top + 1, image.height());
    let mut luminance_sum = 0.;
    let mut luminance_squared_sum = 0.;
    let mut edge_count = 0u64;
    let mut buckets = std::collections::HashMap::<u16, u64>::new();
    let mut count = 0u64;
    for y in top..bottom {
        for x in left..right {
            let pixel = image.get_pixel(x, y);
            let luminance = 0.2126 * f64::from(pixel[0])
                + 0.7152 * f64::from(pixel[1])
                + 0.0722 * f64::from(pixel[2]);
            luminance_sum += luminance;
            luminance_squared_sum += luminance * luminance;
            let bucket = (u16::from(pixel[0] / 16) << 8)
                | (u16::from(pixel[1] / 16) << 4)
                | u16::from(pixel[2] / 16);
            *buckets.entry(bucket).or_default() += 1;
            if x > left {
                let previous = image.get_pixel(x - 1, y);
                let difference = u16::from(pixel[0].abs_diff(previous[0]))
                    + u16::from(pixel[1].abs_diff(previous[1]))
                    + u16::from(pixel[2].abs_diff(previous[2]));
                edge_count += u64::from(difference > 36);
            }
            if y > top {
                let previous = image.get_pixel(x, y - 1);
                let difference = u16::from(pixel[0].abs_diff(previous[0]))
                    + u16::from(pixel[1].abs_diff(previous[1]))
                    + u16::from(pixel[2].abs_diff(previous[2]));
                edge_count += u64::from(difference > 36);
            }
            count += 1;
        }
    }
    let count = count.max(1) as f64;
    let mean = luminance_sum / count;
    let contrast = (luminance_squared_sum / count - mean * mean).max(0.).sqrt() / 255.;
    let dominant_pixels = buckets.values().copied().max().unwrap_or_default() as f64;
    CollaborativeRasterMetrics {
        contrast,
        edge_density: edge_count as f64 / (count * 2.),
        foreground_coverage: 1. - dominant_pixels / count,
    }
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn validate_collaborative_regions(
    name: &str,
    actual: &RgbaImage,
    reference: &RgbaImage,
) -> Result<String> {
    let width = reference.width();
    let height = reference.height();
    let regions = if name == "screenshot-1" {
        vec![
            ("rail", 0, 0, 452, height),
            ("top-bar", 452, 0, width, 126),
            ("timeline", 452, 126, 1220, height.saturating_sub(96)),
            ("review", 1220, 126, width, height.saturating_sub(96)),
            (
                "composer-status",
                452,
                height.saturating_sub(112),
                width,
                height,
            ),
        ]
    } else {
        vec![
            ("rail", 0, 0, 450, height),
            ("top-bar", 450, 0, width, 60),
            ("timeline", 450, 60, width, height.saturating_sub(58)),
            ("timeline-right", 1220, 60, width, height.saturating_sub(58)),
            (
                "composer-status",
                450,
                height.saturating_sub(96),
                width,
                height,
            ),
        ]
    };
    let mut report = format!("  {name} region metrics (actual/reference):\n");
    for (label, left, top, right, bottom) in regions {
        let actual_metrics = collaborative_raster_metrics(actual, left, top, right, bottom);
        let reference_metrics = collaborative_raster_metrics(reference, left, top, right, bottom);
        report.push_str(&format!(
            "    {label}: contrast {:.3}/{:.3}, edges {:.3}/{:.3}, coverage {:.3}/{:.3}\n",
            actual_metrics.contrast,
            reference_metrics.contrast,
            actual_metrics.edge_density,
            reference_metrics.edge_density,
            actual_metrics.foreground_coverage,
            reference_metrics.foreground_coverage,
        ));
        anyhow::ensure!(
            actual_metrics.contrast >= (reference_metrics.contrast * 0.35).max(0.01),
            "{name} {label} is washed out or lacks native contrast"
        );
        anyhow::ensure!(
            actual_metrics.edge_density >= (reference_metrics.edge_density * 0.30).max(0.002),
            "{name} {label} is empty or too sparsely populated"
        );
        anyhow::ensure!(
            actual_metrics.foreground_coverage
                >= (reference_metrics.foreground_coverage * 0.30).max(0.02),
            "{name} {label} lacks the required populated native surface"
        );
    }
    Ok(report)
}

#[cfg(all(target_os = "macos", feature = "multiplayer-tools"))]
fn collaborative_structural_similarity(actual: &RgbaImage, reference: &RgbaImage) -> f64 {
    const COLUMNS: u32 = 32;
    const ROWS: u32 = 24;

    fn luminance(pixel: &image::Rgba<u8>) -> f64 {
        (0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2]))
            / 255.
    }

    fn cell_features(image: &RgbaImage, column: u32, row: u32) -> (f64, f64) {
        let left = column * image.width() / COLUMNS;
        let right = ((column + 1) * image.width() / COLUMNS).max(left + 1);
        let top = row * image.height() / ROWS;
        let bottom = ((row + 1) * image.height() / ROWS).max(top + 1);
        let mut luminance_sum = 0.;
        let mut edge_sum = 0.;
        let mut count = 0u64;
        for y in top..bottom.min(image.height()) {
            for x in left..right.min(image.width()) {
                let value = luminance(image.get_pixel(x, y));
                luminance_sum += value;
                if x > left {
                    edge_sum += (value - luminance(image.get_pixel(x - 1, y))).abs();
                }
                if y > top {
                    edge_sum += (value - luminance(image.get_pixel(x, y - 1))).abs();
                }
                count += 1;
            }
        }
        let count = count.max(1) as f64;
        (luminance_sum / count, (edge_sum / (count * 2.)).min(1.))
    }

    let mut score = 0.;
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let (actual_luminance, actual_edges) = cell_features(actual, column, row);
            let (reference_luminance, reference_edges) = cell_features(reference, column, row);
            let luminance_similarity = 1. - (actual_luminance - reference_luminance).abs();
            let edge_similarity = 1. - (actual_edges - reference_edges).abs();
            score += 0.35 * luminance_similarity + 0.65 * edge_similarity;
        }
    }
    score / f64::from(COLUMNS * ROWS)
}

#[cfg(target_os = "macos")]
enum TestResult {
    Passed,
    BaselineUpdated(PathBuf),
}

#[cfg(target_os = "macos")]
fn run_visual_test(
    test_name: &str,
    window: gpui::AnyWindowHandle,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    // Ensure all pending work is done
    cx.run_until_parked();

    // Refresh the window to ensure it's fully rendered
    cx.update_window(window, |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture the screenshot using direct texture capture
    let screenshot = cx.capture_screenshot(window)?;

    // Get paths
    let baseline_path = get_baseline_path(test_name);
    let output_dir = std::env::var("VISUAL_TEST_OUTPUT_DIR")
        .unwrap_or_else(|_| "target/visual_tests".to_string());
    let output_path = PathBuf::from(&output_dir).join(format!("{}.png", test_name));

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // Always save the current screenshot
    screenshot.save(&output_path)?;
    println!("  Screenshot saved to: {}", output_path.display());

    if update_baseline {
        // Update the baseline
        if let Some(parent) = baseline_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        screenshot.save(&baseline_path)?;
        println!("  Baseline updated: {}", baseline_path.display());
        return Ok(TestResult::BaselineUpdated(baseline_path));
    }

    // Compare with baseline
    if !baseline_path.exists() {
        return Err(anyhow::anyhow!(
            "Baseline not found: {}. Run with UPDATE_BASELINE=1 to create it.",
            baseline_path.display()
        ));
    }

    let baseline = image::open(&baseline_path)?.to_rgba8();
    let comparison = compare_images(&screenshot, &baseline);

    println!(
        "  Match: {:.2}% ({} different pixels)",
        comparison.match_percentage * 100.0,
        comparison.diff_pixel_count
    );

    if comparison.match_percentage >= MATCH_THRESHOLD {
        Ok(TestResult::Passed)
    } else {
        // Save diff image
        let diff_path = PathBuf::from(&output_dir).join(format!("{}_diff.png", test_name));
        comparison.diff_image.save(&diff_path)?;
        println!("  Diff image saved to: {}", diff_path.display());

        Err(anyhow::anyhow!(
            "Image mismatch: {:.2}% match (threshold: {:.2}%)",
            comparison.match_percentage * 100.0,
            MATCH_THRESHOLD * 100.0
        ))
    }
}

#[cfg(target_os = "macos")]
fn get_baseline_path(test_name: &str) -> PathBuf {
    // Get the workspace root (where Cargo.toml is)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    workspace_root
        .join(BASELINE_DIR)
        .join(format!("{}.png", test_name))
}

#[cfg(target_os = "macos")]
struct ImageComparison {
    match_percentage: f64,
    diff_image: RgbaImage,
    diff_pixel_count: u32,
    #[allow(dead_code)]
    total_pixels: u32,
}

#[cfg(target_os = "macos")]
fn compare_images(actual: &RgbaImage, expected: &RgbaImage) -> ImageComparison {
    let width = actual.width().max(expected.width());
    let height = actual.height().max(expected.height());
    let total_pixels = width * height;

    let mut diff_image = RgbaImage::new(width, height);
    let mut matching_pixels = 0u32;

    for y in 0..height {
        for x in 0..width {
            let actual_pixel = if x < actual.width() && y < actual.height() {
                *actual.get_pixel(x, y)
            } else {
                image::Rgba([0, 0, 0, 0])
            };

            let expected_pixel = if x < expected.width() && y < expected.height() {
                *expected.get_pixel(x, y)
            } else {
                image::Rgba([0, 0, 0, 0])
            };

            if pixels_are_similar(&actual_pixel, &expected_pixel) {
                matching_pixels += 1;
                // Semi-transparent green for matching pixels
                diff_image.put_pixel(x, y, image::Rgba([0, 255, 0, 64]));
            } else {
                // Bright red for differing pixels
                diff_image.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
            }
        }
    }

    let match_percentage = matching_pixels as f64 / total_pixels as f64;
    let diff_pixel_count = total_pixels - matching_pixels;

    ImageComparison {
        match_percentage,
        diff_image,
        diff_pixel_count,
        total_pixels,
    }
}

#[cfg(target_os = "macos")]
fn pixels_are_similar(a: &image::Rgba<u8>, b: &image::Rgba<u8>) -> bool {
    const TOLERANCE: i16 = 2;
    (a.0[0] as i16 - b.0[0] as i16).abs() <= TOLERANCE
        && (a.0[1] as i16 - b.0[1] as i16).abs() <= TOLERANCE
        && (a.0[2] as i16 - b.0[2] as i16).abs() <= TOLERANCE
        && (a.0[3] as i16 - b.0[3] as i16).abs() <= TOLERANCE
}

#[cfg(target_os = "macos")]
fn create_test_files(project_path: &Path) {
    // Create src directory
    let src_dir = project_path.join("src");
    std::fs::create_dir_all(&src_dir).expect("Failed to create src directory");

    // Create main.rs
    let main_rs = r#"fn main() {
    println!("Hello, world!");

    let x = 42;
    let y = x * 2;

    if y > 50 {
        println!("y is greater than 50");
    } else {
        println!("y is not greater than 50");
    }

    for i in 0..10 {
        println!("i = {}", i);
    }
}

fn helper_function(a: i32, b: i32) -> i32 {
    a + b
}

struct MyStruct {
    field1: String,
    field2: i32,
}

impl MyStruct {
    fn new(name: &str, value: i32) -> Self {
        Self {
            field1: name.to_string(),
            field2: value,
        }
    }

    fn get_value(&self) -> i32 {
        self.field2
    }
}
"#;
    std::fs::write(src_dir.join("main.rs"), main_rs).expect("Failed to write main.rs");

    // Create lib.rs
    let lib_rs = r#"//! A sample library for visual testing

pub mod utils;

/// A public function in the library
pub fn library_function() -> String {
    "Hello from lib".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(library_function(), "Hello from lib");
    }
}
"#;
    std::fs::write(src_dir.join("lib.rs"), lib_rs).expect("Failed to write lib.rs");

    // Create utils.rs
    let utils_rs = r#"//! Utility functions

/// Format a number with commas
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Calculate fibonacci number
pub fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
"#;
    std::fs::write(src_dir.join("utils.rs"), utils_rs).expect("Failed to write utils.rs");

    // Create Cargo.toml
    let cargo_toml = r#"[package]
name = "test_project"
version = "0.1.0"
edition = "2021"

[dependencies]
"#;
    std::fs::write(project_path.join("Cargo.toml"), cargo_toml)
        .expect("Failed to write Cargo.toml");

    // Create README.md
    let readme = r#"# Test Project

This is a test project for visual testing of Zed.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```bash
cargo run
```
"#;
    std::fs::write(project_path.join("README.md"), readme).expect("Failed to write README.md");
}

#[cfg(target_os = "macos")]
fn init_app_state(cx: &mut App) -> Arc<AppState> {
    use fs::Fs;
    use node_runtime::NodeRuntime;
    use session::Session;
    use settings::SettingsStore;

    if !cx.has_global::<SettingsStore>() {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
    }

    // Use the real filesystem instead of FakeFs so we can access actual files on disk
    let fs: Arc<dyn Fs> = Arc::new(fs::RealFs::new(None, cx.background_executor().clone()));
    <dyn Fs>::set_global(fs.clone(), cx);

    let languages = Arc::new(language::LanguageRegistry::test(
        cx.background_executor().clone(),
    ));
    let clock = Arc::new(clock::FakeSystemClock::new());
    let http_client = http_client::FakeHttpClient::with_404_response();
    let client = client::Client::new(clock, http_client, cx);
    let session = cx.new(|cx| session::AppSession::new(Session::test(), cx));
    let user_store = cx.new(|cx| client::UserStore::new(client.clone(), cx));
    let workspace_store = cx.new(|cx| workspace::WorkspaceStore::new(client.clone(), cx));
    let node_runtime = NodeRuntime::unavailable();

    theme_settings::init(theme::LoadThemes::JustBase, cx);
    client::init(&client, cx);
    languages::init(languages.clone(), fs.clone(), node_runtime.clone(), cx);

    let app_state = Arc::new(AppState {
        client,
        fs,
        languages,
        user_store,
        workspace_store,
        node_runtime,
        build_window_options: |_, _| Default::default(),
        session,
    });
    AppState::set_global(app_state.clone(), cx);
    app_state
}

/// Runs visual tests for breakpoint hover states in the editor gutter.
///
/// This test captures three states:
/// 1. Gutter with line numbers, no breakpoint hover (baseline)
/// 2. Gutter with breakpoint hover indicator (gray circle)
/// 3. Gutter with breakpoint hover AND tooltip
#[cfg(target_os = "macos")]
fn run_breakpoint_hover_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    // Create a temporary directory with a simple test file
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path.canonicalize()?;
    let project_path = canonical_temp.join("project");
    std::fs::create_dir_all(&project_path)?;

    // Create a simple file with a few lines
    let src_dir = project_path.join("src");
    std::fs::create_dir_all(&src_dir)?;

    let test_content = r#"fn main() {
    println!("Hello");
    let x = 42;
}
"#;
    std::fs::write(src_dir.join("test.rs"), test_content)?;

    // Create a small window - just big enough to show gutter and a few lines
    let window_size = size(px(300.0), px(200.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    // Create project
    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    // Open workspace window
    let workspace_window: WindowHandle<Workspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    })
                },
            )
        })
        .context("Failed to open breakpoint test window")?;

    cx.run_until_parked();

    // Add the project as a worktree
    let add_worktree_task = workspace_window
        .update(cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                project.find_or_create_worktree(&project_path, true, cx)
            })
        })
        .context("Failed to start adding worktree")?;

    cx.background_executor.allow_parking();
    let worktree_result = cx.foreground_executor.block_test(add_worktree_task);
    cx.background_executor.forbid_parking();
    worktree_result.context("Failed to add worktree")?;

    cx.run_until_parked();

    // Open the test file
    let open_file_task = workspace_window
        .update(cx, |workspace, window, cx| {
            let worktree = workspace.project().read(cx).worktrees(cx).next();
            if let Some(worktree) = worktree {
                let worktree_id = worktree.read(cx).id();
                let rel_path: std::sync::Arc<util::rel_path::RelPath> =
                    util::rel_path::rel_path("src/test.rs").into();
                let project_path: project::ProjectPath = (worktree_id, rel_path).into();
                Some(workspace.open_path(project_path, None, true, window, cx))
            } else {
                None
            }
        })
        .log_err()
        .flatten();

    if let Some(task) = open_file_task {
        cx.background_executor.allow_parking();
        cx.foreground_executor.block_test(task).log_err();
        cx.background_executor.forbid_parking();
    }

    cx.run_until_parked();

    // Wait for the editor to fully load
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Test 1: Gutter visible with line numbers, no breakpoint hover
    let test1_result = run_visual_test(
        "breakpoint_hover_none",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 2: Breakpoint hover indicator (circle) visible
    // The gutter is on the left side. We need to position the mouse over the gutter area
    // for line 1. The breakpoint indicator appears in the leftmost part of the gutter.
    //
    // The breakpoint hover requires multiple steps:
    // 1. Draw to register mouse listeners
    // 2. Mouse move to trigger gutter_hovered and create GutterHoverButton
    // 3. Wait 200ms for is_active to become true
    // 4. Draw again to render the indicator
    //
    // The gutter_position should be in the gutter area to trigger the gutter hover button.
    // The button_position should be directly over the breakpoint icon button for tooltip hover.
    // Based on debug output: button is at origin=(3.12, 66.5) with size=(14, 16)
    let gutter_position = point(px(30.0), px(85.0));
    let button_position = point(px(10.0), px(75.0)); // Center of the breakpoint button

    // Step 1: Initial draw to register mouse listeners
    cx.update_window(workspace_window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })?;
    cx.run_until_parked();

    // Step 2: Simulate mouse move into gutter area
    cx.simulate_mouse_move(
        workspace_window.into(),
        gutter_position,
        None,
        Modifiers::default(),
    );

    // Step 3: Advance clock past 200ms debounce
    cx.advance_clock(Duration::from_millis(300));
    cx.run_until_parked();

    // Step 4: Draw again to pick up the indicator state change
    cx.update_window(workspace_window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })?;
    cx.run_until_parked();

    // Step 5: Another mouse move to keep hover state active
    cx.simulate_mouse_move(
        workspace_window.into(),
        gutter_position,
        None,
        Modifiers::default(),
    );

    // Step 6: Final draw
    cx.update_window(workspace_window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })?;
    cx.run_until_parked();

    let test2_result = run_visual_test(
        "breakpoint_hover_circle",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 3: Breakpoint hover with tooltip visible
    // The tooltip delay is 500ms (TOOLTIP_SHOW_DELAY constant)
    // We need to position the mouse directly over the breakpoint button for the tooltip to show.
    // The button hitbox is approximately at (3.12, 66.5) with size (14, 16).

    // Move mouse directly over the button to trigger tooltip hover
    cx.simulate_mouse_move(
        workspace_window.into(),
        button_position,
        None,
        Modifiers::default(),
    );

    // Draw to register the button's tooltip hover listener
    cx.update_window(workspace_window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })?;
    cx.run_until_parked();

    // Move mouse over button again to trigger tooltip scheduling
    cx.simulate_mouse_move(
        workspace_window.into(),
        button_position,
        None,
        Modifiers::default(),
    );

    // Advance clock past TOOLTIP_SHOW_DELAY (500ms)
    cx.advance_clock(TOOLTIP_SHOW_DELAY + Duration::from_millis(100));
    cx.run_until_parked();

    // Draw to render the tooltip
    cx.update_window(workspace_window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })?;
    cx.run_until_parked();

    // Refresh window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    let test3_result = run_visual_test(
        "breakpoint_hover_tooltip",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Clean up: remove worktrees to stop background scanning
    workspace_window
        .update(cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                let worktree_ids: Vec<_> =
                    project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                for id in worktree_ids {
                    project.remove_worktree(id, cx);
                }
            });
        })
        .log_err();

    cx.run_until_parked();

    // Close the window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    // Give background tasks time to finish
    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Return combined result
    match (&test1_result, &test2_result, &test3_result) {
        (TestResult::Passed, TestResult::Passed, TestResult::Passed) => Ok(TestResult::Passed),
        (TestResult::BaselineUpdated(p), _, _)
        | (_, TestResult::BaselineUpdated(p), _)
        | (_, _, TestResult::BaselineUpdated(p)) => Ok(TestResult::BaselineUpdated(p.clone())),
    }
}

/// Runs visual tests for the settings UI sub-page auto-open feature.
///
/// This test verifies that when opening settings via OpenSettingsAt with a path
/// that maps to a single SubPageLink, the sub-page is automatically opened.
///
/// This test captures two states:
/// 1. Settings opened with a path that maps to multiple items (no auto-open)
/// 2. Settings opened with a path that maps to a single SubPageLink (auto-opens sub-page)
#[cfg(target_os = "macos")]
fn run_settings_ui_subpage_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    // Create a workspace window for dispatching actions
    let window_size = size(px(1280.0), px(800.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let workspace_window: WindowHandle<MultiWorkspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    let workspace = cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    });
                    cx.new(|cx| MultiWorkspace::new(workspace, window, cx))
                },
            )
        })
        .context("Failed to open workspace window")?;

    cx.run_until_parked();

    // Test 1: Open settings with a path that maps to multiple items (e.g., "agent")
    // This should NOT auto-open a sub-page since multiple items match
    workspace_window
        .update(cx, |_workspace, window, cx| {
            window.dispatch_action(
                Box::new(OpenSettingsAt {
                    path: "agent".to_string(),
                    target: None,
                }),
                cx,
            );
        })
        .context("Failed to dispatch OpenSettingsAt for multiple items")?;

    cx.run_until_parked();

    // Find the settings window
    let settings_window_1 = cx
        .update(|cx| {
            cx.windows()
                .into_iter()
                .find_map(|window| window.downcast::<SettingsWindow>())
        })
        .context("Settings window not found")?;

    // Refresh and capture screenshot
    cx.update_window(settings_window_1.into(), |_, window, _cx| {
        window.refresh();
    })?;
    cx.run_until_parked();

    let test1_result = run_visual_test(
        "settings_ui_no_auto_open",
        settings_window_1.into(),
        cx,
        update_baseline,
    )?;

    // Close the settings window
    cx.update_window(settings_window_1.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();
    cx.run_until_parked();

    // Test 2: Open settings with a path that maps to a single SubPageLink
    // "edit_predictions.providers" maps to the "Configure Providers" SubPageLink
    // This should auto-open the sub-page
    workspace_window
        .update(cx, |_workspace, window, cx| {
            window.dispatch_action(
                Box::new(OpenSettingsAt {
                    path: "edit_predictions.providers".to_string(),
                    target: None,
                }),
                cx,
            );
        })
        .context("Failed to dispatch OpenSettingsAt for single SubPageLink")?;

    cx.run_until_parked();

    // Find the new settings window
    let settings_window_2 = cx
        .update(|cx| {
            cx.windows()
                .into_iter()
                .find_map(|window| window.downcast::<SettingsWindow>())
        })
        .context("Settings window not found for sub-page test")?;

    // Refresh and capture screenshot
    cx.update_window(settings_window_2.into(), |_, window, _cx| {
        window.refresh();
    })?;
    cx.run_until_parked();

    let test2_result = run_visual_test(
        "settings_ui_subpage_auto_open",
        settings_window_2.into(),
        cx,
        update_baseline,
    )?;

    // Clean up: close the settings window
    cx.update_window(settings_window_2.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();
    cx.run_until_parked();

    // Clean up: close the workspace window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();
    cx.run_until_parked();

    // Give background tasks time to finish
    for _ in 0..5 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Return combined result
    match (&test1_result, &test2_result) {
        (TestResult::Passed, TestResult::Passed) => Ok(TestResult::Passed),
        (TestResult::BaselineUpdated(p), _) | (_, TestResult::BaselineUpdated(p)) => {
            Ok(TestResult::BaselineUpdated(p.clone()))
        }
    }
}

/// Runs visual tests for the diff review button in git diff views.
///
/// This test captures three states:
/// 1. Diff view with feature flag enabled (button visible)
/// 2. Diff view with feature flag disabled (no button)
/// 3. Regular editor with feature flag enabled (no button - only shows in diff views)
#[cfg(target_os = "macos")]
fn run_diff_review_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    // Create a temporary directory with test files and a real git repo
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path.canonicalize()?;
    let project_path = canonical_temp.join("project");
    std::fs::create_dir_all(&project_path)?;

    // Initialize a real git repository
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(&project_path)
        .output()?;

    // Configure git user for commits
    std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(&project_path)
        .output()?;
    std::process::Command::new("git")
        .args(["config", "user.name", "Test User"])
        .current_dir(&project_path)
        .output()?;

    // Create a test file with original content
    let original_content = "// Original content\n";
    std::fs::write(project_path.join("thread-view.tsx"), original_content)?;

    // Commit the original file
    std::process::Command::new("git")
        .args(["add", "thread-view.tsx"])
        .current_dir(&project_path)
        .output()?;
    std::process::Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(&project_path)
        .output()?;

    // Modify the file to create a diff
    let modified_content = r#"import { ScrollArea } from 'components';
import { ButtonAlt, Tooltip } from 'ui';
import { Message, FileEdit } from 'types';
import { AiPaneTabContext } from 'context';
"#;
    std::fs::write(project_path.join("thread-view.tsx"), modified_content)?;

    // Create window for the diff view - sized to show just the editor
    let window_size = size(px(600.0), px(400.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    // Create project
    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    // Add the test directory as a worktree
    let add_worktree_task = project.update(cx, |project, cx| {
        project.find_or_create_worktree(&project_path, true, cx)
    });

    cx.background_executor.allow_parking();
    cx.foreground_executor
        .block_test(add_worktree_task)
        .log_err();
    cx.background_executor.forbid_parking();

    cx.run_until_parked();

    // Wait for worktree to be fully scanned and git status to be detected
    for _ in 0..5 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Test 1: Diff view with feature flag enabled
    // Enable the feature flag
    cx.update(|cx| {
        cx.update_flags(true, vec!["diff-review".to_string()]);
    });

    let workspace_window: WindowHandle<Workspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    })
                },
            )
        })
        .context("Failed to open diff review test window")?;

    cx.run_until_parked();

    // Create and add the ProjectDiff using the public deploy_at method
    workspace_window
        .update(cx, |workspace, window, cx| {
            ProjectDiff::deploy_at(workspace, None, window, cx);
        })
        .log_err();

    // Wait for diff to render
    for _ in 0..5 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 1: Diff with flag enabled
    let test1_result = run_visual_test(
        "diff_review_button_enabled",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 2: Diff view with feature flag disabled
    // Disable the feature flag
    cx.update(|cx| {
        cx.update_flags(false, vec![]);
    });

    // Refresh window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Capture Test 2: Diff with flag disabled
    let test2_result = run_visual_test(
        "diff_review_button_disabled",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 3: Regular editor with flag enabled (should NOT show button)
    // Re-enable the feature flag
    cx.update(|cx| {
        cx.update_flags(true, vec!["diff-review".to_string()]);
    });

    // Create a new window with just a regular editor
    let regular_window: WindowHandle<Workspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    })
                },
            )
        })
        .context("Failed to open regular editor window")?;

    cx.run_until_parked();

    // Open a regular file (not a diff view)
    let open_file_task = regular_window
        .update(cx, |workspace, window, cx| {
            let worktree = workspace.project().read(cx).worktrees(cx).next();
            if let Some(worktree) = worktree {
                let worktree_id = worktree.read(cx).id();
                let rel_path: std::sync::Arc<util::rel_path::RelPath> =
                    util::rel_path::rel_path("thread-view.tsx").into();
                let project_path: project::ProjectPath = (worktree_id, rel_path).into();
                Some(workspace.open_path(project_path, None, true, window, cx))
            } else {
                None
            }
        })
        .log_err()
        .flatten();

    if let Some(task) = open_file_task {
        cx.background_executor.allow_parking();
        cx.foreground_executor.block_test(task).log_err();
        cx.background_executor.forbid_parking();
    }

    // Wait for file to open
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 3: Regular editor with flag enabled (no button)
    let test3_result = run_visual_test(
        "diff_review_button_regular_editor",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 4: Show the diff review overlay on the regular editor
    regular_window
        .update(cx, |workspace, window, cx| {
            // Get the first editor from the workspace
            let editors: Vec<_> = workspace.items_of_type::<editor::Editor>(cx).collect();
            if let Some(editor) = editors.into_iter().next() {
                editor.update(cx, |editor, cx| {
                    editor.show_diff_review_overlay(DisplayRow(1)..DisplayRow(1), window, cx);
                });
            }
        })
        .log_err();

    // Wait for overlay to render
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 4: Regular editor with overlay shown
    let test4_result = run_visual_test(
        "diff_review_overlay_shown",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 5: Type text into the diff review prompt and submit it
    // First, get the prompt editor from the overlay and type some text
    regular_window
        .update(cx, |workspace, window, cx| {
            let editors: Vec<_> = workspace.items_of_type::<editor::Editor>(cx).collect();
            if let Some(editor) = editors.into_iter().next() {
                editor.update(cx, |editor, cx| {
                    // Get the prompt editor from the overlay and insert text
                    if let Some(prompt_editor) = editor.diff_review_prompt_editor().cloned() {
                        prompt_editor.update(cx, |prompt_editor: &mut editor::Editor, cx| {
                            prompt_editor.insert(
                                "This change needs better error handling",
                                window,
                                cx,
                            );
                        });
                    }
                });
            }
        })
        .log_err();

    // Wait for text to be inserted
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 5: Diff review overlay with typed text
    let test5_result = run_visual_test(
        "diff_review_overlay_with_text",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 6: Submit a comment to store it locally
    regular_window
        .update(cx, |workspace, window, cx| {
            let editors: Vec<_> = workspace.items_of_type::<editor::Editor>(cx).collect();
            if let Some(editor) = editors.into_iter().next() {
                editor.update(cx, |editor, cx| {
                    // Submit the comment that was typed in test 5
                    editor.submit_diff_review_comment(window, cx);
                });
            }
        })
        .log_err();

    // Wait for comment to be stored
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 6: Overlay with one stored comment
    let test6_result = run_visual_test(
        "diff_review_one_comment",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 7: Add more comments to show multiple comments expanded
    regular_window
        .update(cx, |workspace, window, cx| {
            let editors: Vec<_> = workspace.items_of_type::<editor::Editor>(cx).collect();
            if let Some(editor) = editors.into_iter().next() {
                editor.update(cx, |editor, cx| {
                    // Add second comment
                    if let Some(prompt_editor) = editor.diff_review_prompt_editor().cloned() {
                        prompt_editor.update(cx, |pe, cx| {
                            pe.insert("Second comment about imports", window, cx);
                        });
                    }
                    editor.submit_diff_review_comment(window, cx);

                    // Add third comment
                    if let Some(prompt_editor) = editor.diff_review_prompt_editor().cloned() {
                        prompt_editor.update(cx, |pe, cx| {
                            pe.insert("Third comment about naming conventions", window, cx);
                        });
                    }
                    editor.submit_diff_review_comment(window, cx);
                });
            }
        })
        .log_err();

    // Wait for comments to be stored
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 7: Overlay with multiple comments expanded
    let test7_result = run_visual_test(
        "diff_review_multiple_comments_expanded",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Test 8: Collapse the comments section
    regular_window
        .update(cx, |workspace, _window, cx| {
            let editors: Vec<_> = workspace.items_of_type::<editor::Editor>(cx).collect();
            if let Some(editor) = editors.into_iter().next() {
                editor.update(cx, |editor, cx| {
                    // Toggle collapse using the public method
                    editor.set_diff_review_comments_expanded(false, cx);
                });
            }
        })
        .log_err();

    // Wait for UI to update
    for _ in 0..3 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh window
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture Test 8: Comments collapsed
    let test8_result = run_visual_test(
        "diff_review_comments_collapsed",
        regular_window.into(),
        cx,
        update_baseline,
    )?;

    // Clean up: remove worktrees to stop background scanning
    workspace_window
        .update(cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                let worktree_ids: Vec<_> =
                    project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                for id in worktree_ids {
                    project.remove_worktree(id, cx);
                }
            });
        })
        .log_err();

    cx.run_until_parked();

    // Close windows
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();
    cx.update_window(regular_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    // Give background tasks time to finish
    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Return combined result
    let all_results = [
        &test1_result,
        &test2_result,
        &test3_result,
        &test4_result,
        &test5_result,
        &test6_result,
        &test7_result,
        &test8_result,
    ];

    // Combine results: if any test updated a baseline, return BaselineUpdated;
    // otherwise return Passed. The exhaustive match ensures the compiler
    // verifies we handle all TestResult variants.
    let result = all_results
        .iter()
        .fold(TestResult::Passed, |acc, r| match r {
            TestResult::Passed => acc,
            TestResult::BaselineUpdated(p) => TestResult::BaselineUpdated(p.clone()),
        });
    Ok(result)
}

/// A stub AgentServer for visual testing that returns a pre-programmed connection.
#[derive(Clone)]
#[cfg(target_os = "macos")]
struct StubAgentServer {
    connection: StubAgentConnection,
}

#[cfg(target_os = "macos")]
impl StubAgentServer {
    fn new(connection: StubAgentConnection) -> Self {
        Self { connection }
    }
}

#[cfg(target_os = "macos")]
impl AgentServer for StubAgentServer {
    fn logo(&self) -> ui::IconName {
        ui::IconName::ZedAssistant
    }

    fn agent_id(&self) -> AgentId {
        "Visual Test Agent".into()
    }

    fn connect(
        &self,
        _delegate: AgentServerDelegate,
        _project: Entity<Project>,
        _cx: &mut App,
    ) -> gpui::Task<gpui::Result<Rc<dyn AgentConnection>>> {
        gpui::Task::ready(Ok(Rc::new(self.connection.clone())))
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

#[cfg(all(target_os = "macos", feature = "visual-tests"))]
fn run_agent_thread_view_test(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    use agent::{AgentTool, ToolInput};
    use agent_ui::AgentPanel;

    // Create a temporary directory with the test image
    // Canonicalize to resolve symlinks (on macOS, /var -> /private/var)
    // Use keep() to prevent auto-cleanup - we'll clean up manually after stopping background tasks
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path.canonicalize()?;
    let project_path = canonical_temp.join("project");
    std::fs::create_dir_all(&project_path)?;
    let image_path = project_path.join("test-image.png");
    std::fs::write(&image_path, EMBEDDED_TEST_IMAGE)?;

    // Create a project with the test image
    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    // Add the test directory as a worktree
    let add_worktree_task = project.update(cx, |project, cx| {
        project.find_or_create_worktree(&project_path, true, cx)
    });

    cx.background_executor.allow_parking();
    let (worktree, _) = cx
        .foreground_executor
        .block_test(add_worktree_task)
        .context("Failed to add worktree")?;
    cx.background_executor.forbid_parking();

    cx.run_until_parked();

    let worktree_name = cx.read(|cx| worktree.read(cx).root_name_str().to_string());

    // Create the necessary entities for the ReadFileTool
    let action_log = cx.update(|cx| cx.new(|_| action_log::ActionLog::new(project.clone())));

    // Create the ReadFileTool
    let tool = Arc::new(agent::ReadFileTool::new(project.clone(), action_log, true));

    // Create a test event stream to capture tool output
    let (event_stream, mut event_receiver) = agent::ToolCallEventStream::test();

    // Run the real ReadFileTool to get the actual image content
    let input = agent::ReadFileToolInput {
        path: format!("{}/test-image.png", worktree_name),
        start_line: None,
        end_line: None,
    };
    let run_task = cx.update(|cx| {
        tool.clone()
            .run(ToolInput::resolved(input), event_stream, cx)
    });

    cx.background_executor.allow_parking();
    let run_result = cx.foreground_executor.block_test(run_task);
    cx.background_executor.forbid_parking();
    run_result.map_err(|e| match e {
        language_model::LanguageModelToolResultContent::Text(text) => {
            anyhow::anyhow!("ReadFileTool failed: {text}")
        }
        other => anyhow::anyhow!("ReadFileTool failed: {other:?}"),
    })?;

    cx.run_until_parked();

    // Collect the events from the tool execution
    let mut tool_content: Vec<acp::ToolCallContent> = Vec::new();
    let mut tool_locations: Vec<acp::ToolCallLocation> = Vec::new();

    while let Ok(event) = event_receiver.try_recv() {
        if let Ok(agent::ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(
            update,
        ))) = event
        {
            if let Some(content) = update.fields.content {
                tool_content.extend(content);
            }
            if let Some(locations) = update.fields.locations {
                tool_locations.extend(locations);
            }
        }
    }

    if tool_content.is_empty() {
        return Err(anyhow::anyhow!("ReadFileTool did not produce any content"));
    }

    // Create stub connection with the real tool output
    let connection = StubAgentConnection::new();
    connection.set_next_prompt_updates(vec![acp::SessionUpdate::ToolCall(
        acp::ToolCall::new(
            "read_file",
            format!("Read file `{}/test-image.png`", worktree_name),
        )
        .kind(acp::ToolKind::Read)
        .status(acp::ToolCallStatus::Completed)
        .locations(tool_locations)
        .content(tool_content),
    )]);

    let stub_agent: Rc<dyn AgentServer> = Rc::new(StubAgentServer::new(connection));

    // Create a window sized for the agent panel
    let window_size = size(px(500.0), px(900.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let workspace_window: WindowHandle<Workspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    })
                },
            )
        })
        .context("Failed to open agent window")?;

    cx.run_until_parked();

    // Load the AgentPanel
    let (weak_workspace, async_window_cx) = workspace_window
        .update(cx, |workspace, window, cx| {
            (workspace.weak_handle(), window.to_async(cx))
        })
        .context("Failed to get workspace handle")?;

    cx.background_executor.allow_parking();
    let panel = cx
        .foreground_executor
        .block_test(AgentPanel::load(weak_workspace, async_window_cx))
        .context("Failed to load AgentPanel")?;
    cx.background_executor.forbid_parking();

    cx.update_window(workspace_window.into(), |_, _window, cx| {
        workspace_window
            .update(cx, |workspace, window, cx| {
                workspace.add_panel(panel.clone(), window, cx);
                workspace.open_panel::<AgentPanel>(window, cx);
            })
            .log_err();
    })?;

    cx.run_until_parked();

    // Inject the stub server and open the stub thread
    cx.update_window(workspace_window.into(), |_, window, cx| {
        panel.update(cx, |panel, cx| {
            panel.open_external_thread_with_server(stub_agent.clone(), window, cx);
        });
    })?;

    cx.run_until_parked();

    // Get the thread view and send a message
    let thread_view = cx
        .read(|cx| panel.read(cx).active_thread_view_for_tests().cloned())
        .ok_or_else(|| anyhow::anyhow!("No active thread view"))?;

    let thread = cx
        .read(|cx| {
            thread_view
                .read(cx)
                .active_thread()
                .map(|active| active.read(cx).thread.clone())
        })
        .ok_or_else(|| anyhow::anyhow!("Thread not available"))?;

    // Send the message to trigger the image response
    let send_future = thread.update(cx, |thread, cx| {
        thread.send(vec!["Show me the Zed logo".into()], cx)
    });

    cx.background_executor.allow_parking();
    let send_result = cx.foreground_executor.block_test(send_future);
    cx.background_executor.forbid_parking();
    send_result.context("Failed to send message")?;

    cx.run_until_parked();

    // Get the tool call ID for expanding later
    let tool_call_id = cx
        .read(|cx| {
            thread.read(cx).entries().iter().find_map(|entry| {
                if let acp_thread::AgentThreadEntry::ToolCall(tool_call) = entry {
                    Some(tool_call.id.clone())
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| anyhow::anyhow!("Expected a ToolCall entry in thread"))?;

    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture the COLLAPSED state
    let collapsed_result = run_visual_test(
        "agent_thread_with_image_collapsed",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Now expand the tool call so the image is visible
    thread_view.update(cx, |view, cx| {
        view.expand_tool_call(tool_call_id, cx);
    });

    cx.run_until_parked();

    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture the EXPANDED state
    let expanded_result = run_visual_test(
        "agent_thread_with_image_expanded",
        workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Remove the worktree from the project to stop background scanning tasks
    // This prevents "root path could not be canonicalized" errors when we clean up
    workspace_window
        .update(cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            project.update(cx, |project, cx| {
                let worktree_ids: Vec<_> =
                    project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                for id in worktree_ids {
                    project.remove_worktree(id, cx);
                }
            });
        })
        .log_err();

    cx.run_until_parked();

    // Close the window
    // Note: This may cause benign "editor::scroll window not found" errors from scrollbar
    // auto-hide timers that were scheduled before the window was closed. These errors
    // don't affect test results.
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    // Run until all cleanup tasks complete
    cx.run_until_parked();

    // Give background tasks time to finish, including scrollbar hide timers (1 second)
    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Note: We don't delete temp_path here because background worktree tasks may still
    // be running. The directory will be cleaned up when the process exits.

    match (&collapsed_result, &expanded_result) {
        (TestResult::Passed, TestResult::Passed) => Ok(TestResult::Passed),
        (TestResult::BaselineUpdated(p), _) | (_, TestResult::BaselineUpdated(p)) => {
            Ok(TestResult::BaselineUpdated(p.clone()))
        }
    }
}

/// Visual test for the Tool Permissions Settings UI page
///
/// Takes a screenshot showing the tool config page with matched patterns and verdict.
#[cfg(target_os = "macos")]
fn run_tool_permissions_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    _update_baseline: bool,
) -> Result<TestResult> {
    use agent_settings::{AgentSettings, CompiledRegex, ToolPermissions, ToolRules};
    use collections::HashMap;
    use settings::ToolPermissionMode;
    use zed_actions::OpenSettingsAt;

    // Set up tool permissions with "hi" as both always_deny and always_allow for terminal
    cx.update(|cx| {
        let mut tools = HashMap::default();
        tools.insert(
            Arc::from("terminal"),
            ToolRules {
                default: None,
                always_allow: vec![CompiledRegex::new("hi", false).unwrap()],
                always_deny: vec![CompiledRegex::new("hi", false).unwrap()],
                always_confirm: vec![],
                invalid_patterns: vec![],
            },
        );
        let mut settings = AgentSettings::get_global(cx).clone();
        settings.tool_permissions = ToolPermissions {
            default: ToolPermissionMode::Confirm,
            tools,
        };
        AgentSettings::override_global(settings, cx);
    });

    // Create a minimal workspace to dispatch the settings action from
    let window_size = size(px(900.0), px(700.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let workspace_window: WindowHandle<MultiWorkspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    let workspace = cx.new(|cx| {
                        Workspace::new(None, project.clone(), app_state.clone(), window, cx)
                    });
                    cx.new(|cx| MultiWorkspace::new(workspace, window, cx))
                },
            )
        })
        .context("Failed to open workspace window for settings test")?;

    cx.run_until_parked();

    // Dispatch the OpenSettingsAt action to open settings at the tool_permissions path
    workspace_window
        .update(cx, |_workspace, window, cx| {
            window.dispatch_action(
                Box::new(OpenSettingsAt {
                    path: "agent.tool_permissions".to_string(),
                    target: None,
                }),
                cx,
            );
        })
        .context("Failed to dispatch OpenSettingsAt action")?;

    cx.run_until_parked();

    // Give the settings window time to open and render
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(50));
        cx.run_until_parked();
    }

    // Find the settings window - it should be the newest window (last in the list)
    let all_windows = cx.update(|cx| cx.windows());
    let settings_window = all_windows.last().copied().context("No windows found")?;

    let output_dir = std::env::var("VISUAL_TEST_OUTPUT_DIR")
        .unwrap_or_else(|_| "target/visual_tests".to_string());
    std::fs::create_dir_all(&output_dir).log_err();

    // Navigate to the tool permissions sub-page using the public API
    let settings_window_handle = settings_window
        .downcast::<settings_ui::SettingsWindow>()
        .context("Failed to downcast to SettingsWindow")?;

    settings_window_handle
        .update(cx, |settings_window, window, cx| {
            settings_window.navigate_to_sub_page("agent.tool_permissions", window, cx);
        })
        .context("Failed to navigate to tool permissions sub-page")?;

    cx.run_until_parked();

    // Give the sub-page time to render
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(50));
        cx.run_until_parked();
    }

    // Now navigate into a specific tool (Terminal) to show the tool config page
    settings_window_handle
        .update(cx, |settings_window, window, cx| {
            settings_window.push_dynamic_sub_page(
                "Terminal",
                "Configure Tool Rules",
                None,
                true,
                settings_ui::pages::render_terminal_tool_config,
                window,
                cx,
            );
        })
        .context("Failed to navigate to Terminal tool config")?;

    cx.run_until_parked();

    // Give the tool config page time to render
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(50));
        cx.run_until_parked();
    }

    // Refresh and redraw so the "Test Your Rules" input is present
    cx.update_window(settings_window, |_, window, cx| {
        window.draw(cx).clear(cx);
    })
    .log_err();
    cx.run_until_parked();

    cx.update_window(settings_window, |_, window, _cx| {
        window.refresh();
    })
    .log_err();
    cx.run_until_parked();

    // Focus the first tab stop in the window (the "Test Your Rules" editor
    // has tab_index(0) and tab_stop(true)) and type "hi" into it.
    cx.update_window(settings_window, |_, window, cx| {
        window.focus_next(cx);
    })
    .log_err();
    cx.run_until_parked();

    cx.simulate_input(settings_window, "hi");

    // Let the UI update with the matched patterns
    for _ in 0..5 {
        cx.advance_clock(Duration::from_millis(50));
        cx.run_until_parked();
    }

    // Refresh and redraw
    cx.update_window(settings_window, |_, window, cx| {
        window.draw(cx).clear(cx);
    })
    .log_err();
    cx.run_until_parked();

    cx.update_window(settings_window, |_, window, _cx| {
        window.refresh();
    })
    .log_err();
    cx.run_until_parked();

    // Save screenshot: Tool config page with "hi" typed and matched patterns visible
    let tool_config_output_path =
        PathBuf::from(&output_dir).join("tool_permissions_test_rules.png");

    if let Ok(screenshot) = cx.capture_screenshot(settings_window) {
        screenshot.save(&tool_config_output_path).log_err();
        println!(
            "Screenshot (test rules) saved to: {}",
            tool_config_output_path.display()
        );
    }

    // Clean up - close the settings window
    cx.update_window(settings_window, |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    // Close the workspace window
    cx.update_window(workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    // Give background tasks time to finish
    for _ in 0..5 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Return success - we're just capturing screenshots, not comparing baselines
    Ok(TestResult::Passed)
}

#[cfg(target_os = "macos")]
fn run_multi_workspace_sidebar_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    // Create temporary directories to act as worktrees for active workspaces
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path.canonicalize()?;

    let workspace1_dir = canonical_temp.join("private-test-remote");
    let workspace2_dir = canonical_temp.join("zed");
    std::fs::create_dir_all(&workspace1_dir)?;
    std::fs::create_dir_all(&workspace2_dir)?;

    // Create both projects upfront so we can build both workspaces during
    // window creation, before the MultiWorkspace entity exists.
    // This avoids a re-entrant read panic that occurs when Workspace::new
    // tries to access the window root (MultiWorkspace) while it's being updated.
    let project1 = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let project2 = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let window_size = size(px(1280.0), px(800.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    // Open a MultiWorkspace window with both workspaces created at construction time
    let multi_workspace_window: WindowHandle<MultiWorkspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    let workspace1 = cx.new(|cx| {
                        Workspace::new(None, project1.clone(), app_state.clone(), window, cx)
                    });
                    let workspace2 = cx.new(|cx| {
                        Workspace::new(None, project2.clone(), app_state.clone(), window, cx)
                    });
                    cx.new(|cx| {
                        let mut multi_workspace = MultiWorkspace::new(workspace1, window, cx);
                        multi_workspace.activate(workspace2, None, window, cx);
                        multi_workspace
                    })
                },
            )
        })
        .context("Failed to open MultiWorkspace window")?;

    cx.run_until_parked();

    // Add worktree to workspace 1 (index 0) so it shows as "private-test-remote"
    let add_worktree1_task = multi_workspace_window
        .update(cx, |multi_workspace, _window, cx| {
            let workspace1 = multi_workspace.workspaces().next().unwrap();
            let project = workspace1.read(cx).project().clone();
            project.update(cx, |project, cx| {
                project.find_or_create_worktree(&workspace1_dir, true, cx)
            })
        })
        .context("Failed to start adding worktree 1")?;

    cx.background_executor.allow_parking();
    cx.foreground_executor
        .block_test(add_worktree1_task)
        .context("Failed to add worktree 1")?;
    cx.background_executor.forbid_parking();

    cx.run_until_parked();

    // Add worktree to workspace 2 (index 1) so it shows as "zed"
    let add_worktree2_task = multi_workspace_window
        .update(cx, |multi_workspace, _window, cx| {
            let workspace2 = multi_workspace.workspaces().nth(1).unwrap();
            let project = workspace2.read(cx).project().clone();
            project.update(cx, |project, cx| {
                project.find_or_create_worktree(&workspace2_dir, true, cx)
            })
        })
        .context("Failed to start adding worktree 2")?;

    cx.background_executor.allow_parking();
    cx.foreground_executor
        .block_test(add_worktree2_task)
        .context("Failed to add worktree 2")?;
    cx.background_executor.forbid_parking();

    cx.run_until_parked();

    // Switch to workspace 1 so it's highlighted as active (index 0)
    multi_workspace_window
        .update(cx, |multi_workspace, window, cx| {
            let workspace = multi_workspace.workspaces().next().unwrap().clone();
            multi_workspace.activate(workspace, None, window, cx);
        })
        .context("Failed to activate workspace 1")?;

    cx.run_until_parked();

    // Create the sidebar outside the MultiWorkspace update to avoid a
    // re-entrant read panic (Sidebar::new reads the MultiWorkspace).
    let sidebar = cx
        .update_window(multi_workspace_window.into(), |root_view, window, cx| {
            let multi_workspace_handle: Entity<MultiWorkspace> = root_view.downcast().unwrap();
            cx.new(|cx| sidebar::Sidebar::new(multi_workspace_handle, window, cx))
        })
        .context("Failed to create sidebar")?;

    multi_workspace_window
        .update(cx, |multi_workspace, _window, cx| {
            multi_workspace.register_sidebar(sidebar.clone(), cx);
        })
        .context("Failed to register sidebar")?;

    cx.run_until_parked();

    // Save test threads to the ThreadStore for each workspace
    let save_tasks = multi_workspace_window
        .update(cx, |multi_workspace, _window, cx| {
            let thread_store = agent::ThreadStore::global(cx);
            let workspaces: Vec<_> = multi_workspace.workspaces().cloned().collect();
            let mut tasks = Vec::new();

            for (index, workspace) in workspaces.iter().enumerate() {
                let workspace_ref = workspace.read(cx);
                let mut paths = Vec::new();
                for worktree in workspace_ref.worktrees(cx) {
                    let worktree_ref = worktree.read(cx);
                    if worktree_ref.is_visible() {
                        paths.push(worktree_ref.abs_path().to_path_buf());
                    }
                }
                let path_list = util::path_list::PathList::new(&paths);

                let (session_id, title, updated_at) = match index {
                    0 => (
                        "visual-test-thread-0",
                        "Refine thread view scrolling behavior",
                        chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2024, 6, 15, 10, 30, 0)
                            .unwrap(),
                    ),
                    1 => (
                        "visual-test-thread-1",
                        "Add line numbers option to FileEditBlock",
                        chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2024, 6, 15, 11, 0, 0)
                            .unwrap(),
                    ),
                    _ => continue,
                };

                let task = thread_store.update(cx, |store, cx| {
                    store.save_thread(
                        acp::SessionId::new(Arc::from(session_id)),
                        agent::DbThread {
                            recipe_scan: false,
                            doctor_repair_proposal: false,
                            goal: None,
                            title: title.to_string().into(),
                            messages: Vec::new(),
                            updated_at,
                            detailed_summary: None,
                            initial_project_snapshot: None,
                            cumulative_token_usage: Default::default(),
                            request_token_usage: Default::default(),
                            model: None,
                            profile: None,
                            subagent_context: None,
                            speed: None,
                            thinking_enabled: false,
                            thinking_effort: None,
                            ui_scroll_position: None,
                            draft_prompt: None,
                            sandboxed_terminal_temp_dir: None,
                            sandbox_grants: Default::default(),
                        },
                        path_list,
                        cx,
                    )
                });
                tasks.push(task);
            }
            tasks
        })
        .context("Failed to create test threads")?;

    cx.background_executor.allow_parking();
    for task in save_tasks {
        cx.foreground_executor
            .block_test(task)
            .context("Failed to save test thread")?;
    }
    cx.background_executor.forbid_parking();

    cx.run_until_parked();

    // Open the sidebar
    multi_workspace_window
        .update(cx, |multi_workspace, window, cx| {
            multi_workspace.toggle_sidebar(window, cx);
        })
        .context("Failed to toggle sidebar")?;

    // Let rendering settle
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh the window
    cx.update_window(multi_workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    // Capture: sidebar open with active workspaces and recent projects
    let test_result = run_visual_test(
        "multi_workspace_sidebar_open",
        multi_workspace_window.into(),
        cx,
        update_baseline,
    )?;

    // Clean up worktrees
    multi_workspace_window
        .update(cx, |multi_workspace, _window, cx| {
            for workspace in multi_workspace.workspaces() {
                let project = workspace.read(cx).project().clone();
                project.update(cx, |project, cx| {
                    let worktree_ids: Vec<_> =
                        project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                    for id in worktree_ids {
                        project.remove_worktree(id, cx);
                    }
                });
            }
        })
        .log_err();

    cx.run_until_parked();

    // Close the window
    cx.update_window(multi_workspace_window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(test_result)
}

#[cfg(target_os = "macos")]
struct ErrorWrappingTestView;

#[cfg(target_os = "macos")]
impl gpui::Render for ErrorWrappingTestView {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::{Button, Callout, IconName, LabelSize, Severity, prelude::*, v_flex};

        let long_error_message = "Rate limit reached for gpt-5.2-codex in organization \
            org-QmYpir6k6dkULKU1XUSN6pal on tokens per min (TPM): Limit 500000, Used 442480, \
            Requested 59724. Please try again in 264ms. Visit \
            https://platform.openai.com/account/rate-limits to learn more.";

        let retry_description = "Retrying. Next attempt in 4 seconds (Attempt 1 of 2).";

        v_flex()
            .size_full()
            .bg(cx.theme().colors().background)
            .p_4()
            .gap_4()
            .child(
                Callout::new()
                    .icon(IconName::Warning)
                    .severity(Severity::Warning)
                    .title(long_error_message)
                    .description(retry_description),
            )
            .child(
                Callout::new()
                    .severity(Severity::Error)
                    .icon(IconName::XCircle)
                    .title("An Error Happened")
                    .description(long_error_message)
                    .actions_slot(Button::new("dismiss", "Dismiss").label_size(LabelSize::Small)),
            )
            .child(
                Callout::new()
                    .severity(Severity::Error)
                    .icon(IconName::XCircle)
                    .title(long_error_message)
                    .actions_slot(Button::new("retry", "Retry").label_size(LabelSize::Small)),
            )
    }
}

#[cfg(target_os = "macos")]
struct ThreadItemBranchNameTestView;

#[cfg(target_os = "macos")]
impl gpui::Render for ThreadItemBranchNameTestView {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::{
            IconName, Label, LabelSize, ThreadItem, ThreadItemWorktreeInfo, WorktreeKind,
            prelude::*,
        };

        let section_label = |text: &str| {
            Label::new(text.to_string())
                .size(LabelSize::Small)
                .color(Color::Muted)
        };

        let container = || {
            v_flex()
                .w_80()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().panel_background)
        };

        v_flex()
            .size_full()
            .bg(cx.theme().colors().background)
            .p_4()
            .gap_3()
            .child(
                Label::new("ThreadItem Branch Names")
                    .size(LabelSize::Large)
                    .color(Color::Default),
            )
            .child(section_label(
                "Linked worktree with branch (worktree / branch)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-linked-branch", "Fix scrolling behavior")
                        .icon(IconName::AiClaude)
                        .timestamp("5m")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("jade-glen".into()),
                            full_path: "/worktrees/jade-glen/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: Some("fix-scrolling".into()),
                        }]),
                ),
            )
            .child(section_label(
                "Linked worktree without branch (detached HEAD)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-linked-no-branch", "Review worktree cleanup")
                        .icon(IconName::AiClaude)
                        .timestamp("1h")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("focal-arrow".into()),
                            full_path: "/worktrees/focal-arrow/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: None,
                        }]),
                ),
            )
            .child(section_label("Main worktree with branch (nothing shown)"))
            .child(
                container().child(
                    ThreadItem::new("ti-main-branch", "Request for Long Classic Poem")
                        .icon(IconName::ZedAgent)
                        .timestamp("2d")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("zed".into()),
                            full_path: "/projects/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Main,
                            branch_name: Some("main".into()),
                        }]),
                ),
            )
            .child(section_label(
                "Main worktree without branch (nothing shown)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-main-no-branch", "Simple greeting thread")
                        .icon(IconName::ZedAgent)
                        .timestamp("3d")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("zed".into()),
                            full_path: "/projects/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Main,
                            branch_name: None,
                        }]),
                ),
            )
            .child(section_label("Linked worktree where name matches branch"))
            .child(
                container().child(
                    ThreadItem::new("ti-same-name", "Implement feature")
                        .icon(IconName::AiClaude)
                        .timestamp("6d")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("stoic-reed".into()),
                            full_path: "/worktrees/stoic-reed/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: Some("stoic-reed".into()),
                        }]),
                ),
            )
            .child(section_label(
                "Manually opened linked worktree (main_path resolves to original repo)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-manual-linked", "Robust Git Worktree Rollback")
                        .icon(IconName::ZedAgent)
                        .timestamp("40m")
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("focal-arrow".into()),
                            full_path: "/worktrees/focal-arrow/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: Some("persist-worktree-3-wiring".into()),
                        }]),
                ),
            )
            .child(section_label(
                "Linked worktree + branch + diff stats + timestamp",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-linked-full", "Full metadata with diff stats")
                        .icon(IconName::AiClaude)
                        .timestamp("3w")
                        .added(42)
                        .removed(17)
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("jade-glen".into()),
                            full_path: "/worktrees/jade-glen/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: Some("feature-branch".into()),
                        }]),
                ),
            )
            .child(section_label("Long branch name truncation with diff stats"))
            .child(
                container().child(
                    ThreadItem::new("ti-long-branch", "Overflow test with very long branch")
                        .icon(IconName::AiClaude)
                        .timestamp("2d")
                        .added(108)
                        .removed(53)
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("my-project".into()),
                            full_path: "/worktrees/my-project/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Linked,
                            branch_name: Some(
                                "fix-very-long-branch-name-that-should-truncate".into(),
                            ),
                        }]),
                ),
            )
            .child(section_label(
                "Main worktree with branch + diff stats + timestamp (branch hidden)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-main-full", "Main worktree with everything")
                        .icon(IconName::ZedAgent)
                        .timestamp("5m")
                        .added(23)
                        .removed(8)
                        .worktrees(vec![ThreadItemWorktreeInfo {
                            worktree_name: Some("zed".into()),
                            full_path: "/projects/zed".into(),
                            highlight_positions: Vec::new(),
                            kind: WorktreeKind::Main,
                            branch_name: Some("sidebar-show-branch-name".into()),
                        }]),
                ),
            )
    }
}

#[cfg(target_os = "macos")]
fn run_thread_item_branch_name_visual_tests(
    _app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    let window_size = size(px(400.0), px(1150.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let window = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |_window, cx| cx.new(|_| ThreadItemBranchNameTestView),
            )
        })
        .context("Failed to open thread item branch name test window")?;

    cx.run_until_parked();

    cx.update_window(window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    let test_result = run_visual_test(
        "thread_item_branch_names",
        window.into(),
        cx,
        update_baseline,
    )?;

    cx.update_window(window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(test_result)
}

#[cfg(target_os = "macos")]
struct ThreadItemIconDecorationsTestView;

#[cfg(target_os = "macos")]
impl gpui::Render for ThreadItemIconDecorationsTestView {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::{IconName, Label, LabelSize, ThreadItem, prelude::*};

        let section_label = |text: &str| {
            Label::new(text.to_string())
                .size(LabelSize::Small)
                .color(Color::Muted)
        };

        let container = || {
            v_flex()
                .w_80()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().panel_background)
        };

        v_flex()
            .size_full()
            .bg(cx.theme().colors().background)
            .p_4()
            .gap_3()
            .child(
                Label::new("ThreadItem Icon Decorations")
                    .size(LabelSize::Large)
                    .color(Color::Default),
            )
            .child(section_label("No decoration (default idle)"))
            .child(
                container()
                    .child(ThreadItem::new("ti-none", "Default idle thread").timestamp("1:00 AM")),
            )
            .child(section_label("Blue dot (notified)"))
            .child(
                container().child(
                    ThreadItem::new("ti-done", "Generation completed successfully")
                        .timestamp("1:05 AM")
                        .notified(true),
                ),
            )
            .child(section_label("Yellow triangle (waiting for confirmation)"))
            .child(
                container().child(
                    ThreadItem::new("ti-waiting", "Waiting for user confirmation")
                        .timestamp("1:10 AM")
                        .status(ui::AgentThreadStatus::WaitingForConfirmation),
                ),
            )
            .child(section_label("Red X (error)"))
            .child(
                container().child(
                    ThreadItem::new("ti-error", "Failed to connect to server")
                        .timestamp("1:15 AM")
                        .status(ui::AgentThreadStatus::Error),
                ),
            )
            .child(section_label("Spinner (running)"))
            .child(
                container().child(
                    ThreadItem::new("ti-running", "Generating response...")
                        .icon(IconName::AiClaude)
                        .timestamp("1:20 AM")
                        .status(ui::AgentThreadStatus::Running),
                ),
            )
            .child(section_label(
                "Spinner + yellow triangle (waiting for confirmation)",
            ))
            .child(
                container().child(
                    ThreadItem::new("ti-running-waiting", "Running but needs confirmation")
                        .icon(IconName::AiClaude)
                        .timestamp("1:25 AM")
                        .status(ui::AgentThreadStatus::WaitingForConfirmation),
                ),
            )
    }
}

#[cfg(target_os = "macos")]
fn run_thread_item_icon_decorations_visual_tests(
    _app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    let window_size = size(px(400.0), px(600.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let window = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |_window, cx| cx.new(|_| ThreadItemIconDecorationsTestView),
            )
        })
        .context("Failed to open thread item icon decorations test window")?;

    cx.run_until_parked();

    cx.update_window(window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    let test_result = run_visual_test(
        "thread_item_icon_decorations",
        window.into(),
        cx,
        update_baseline,
    )?;

    cx.update_window(window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(test_result)
}

#[cfg(target_os = "macos")]
fn run_error_wrapping_visual_tests(
    _app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    let window_size = size(px(500.0), px(400.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let window = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |_window, cx| cx.new(|_| ErrorWrappingTestView),
            )
        })
        .context("Failed to open error wrapping test window")?;

    cx.run_until_parked();

    cx.update_window(window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    let test_result =
        run_visual_test("error_message_wrapping", window.into(), cx, update_baseline)?;

    cx.update_window(window.into(), |_, window, _cx| {
        window.remove_window();
    })
    .log_err();

    cx.run_until_parked();

    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(test_result)
}

#[cfg(target_os = "macos")]
/// Helper to create a project, add a worktree at the given path, and return the project.
fn create_project_with_worktree(
    worktree_dir: &Path,
    app_state: &Arc<AppState>,
    cx: &mut VisualTestAppContext,
) -> Result<Entity<Project>> {
    let project = cx.update(|cx| {
        project::Project::local(
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            None,
            project::LocalProjectFlags {
                init_worktree_trust: false,
                ..Default::default()
            },
            cx,
        )
    });

    let add_task = cx.update(|cx| {
        project.update(cx, |project, cx| {
            project.find_or_create_worktree(worktree_dir, true, cx)
        })
    });

    cx.background_executor.allow_parking();
    cx.foreground_executor
        .block_test(add_task)
        .context("Failed to add worktree")?;
    cx.background_executor.forbid_parking();

    cx.run_until_parked();
    Ok(project)
}

#[cfg(target_os = "macos")]
fn open_sidebar_test_window(
    projects: Vec<Entity<Project>>,
    app_state: &Arc<AppState>,
    cx: &mut VisualTestAppContext,
) -> Result<WindowHandle<MultiWorkspace>> {
    anyhow::ensure!(!projects.is_empty(), "need at least one project");

    let window_size = size(px(400.0), px(600.0));
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: window_size,
    };

    let mut projects_iter = projects.into_iter();
    let first_project = projects_iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("need at least one project"))?;
    let remaining: Vec<_> = projects_iter.collect();

    let multi_workspace_window: WindowHandle<MultiWorkspace> = cx
        .update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    focus: false,
                    show: false,
                    ..Default::default()
                },
                |window, cx| {
                    let first_ws = cx.new(|cx| {
                        Workspace::new(None, first_project.clone(), app_state.clone(), window, cx)
                    });
                    cx.new(|cx| {
                        let mut mw = MultiWorkspace::new(first_ws, window, cx);
                        for project in remaining {
                            let ws = cx.new(|cx| {
                                Workspace::new(None, project, app_state.clone(), window, cx)
                            });
                            mw.activate(ws, None, window, cx);
                        }
                        mw
                    })
                },
            )
        })
        .context("Failed to open MultiWorkspace window")?;

    cx.run_until_parked();

    // Create the sidebar outside the MultiWorkspace update to avoid a
    // re-entrant read panic (Sidebar::new reads the MultiWorkspace).
    let sidebar = cx
        .update_window(multi_workspace_window.into(), |root_view, window, cx| {
            let mw_handle: Entity<MultiWorkspace> = root_view
                .downcast()
                .map_err(|_| anyhow::anyhow!("Failed to downcast root view to MultiWorkspace"))?;
            Ok::<_, anyhow::Error>(cx.new(|cx| sidebar::Sidebar::new(mw_handle, window, cx)))
        })
        .context("Failed to create sidebar")??;

    multi_workspace_window
        .update(cx, |mw, _window, cx| {
            mw.register_sidebar(sidebar.clone(), cx);
        })
        .context("Failed to register sidebar")?;

    cx.run_until_parked();

    // Open the sidebar
    multi_workspace_window
        .update(cx, |mw, window, cx| {
            mw.toggle_sidebar(window, cx);
        })
        .context("Failed to toggle sidebar")?;

    // Let rendering settle
    for _ in 0..10 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    // Refresh the window
    cx.update_window(multi_workspace_window.into(), |_, window, _cx| {
        window.refresh();
    })?;

    cx.run_until_parked();

    Ok(multi_workspace_window)
}

#[cfg(target_os = "macos")]
fn cleanup_sidebar_test_window(
    window: WindowHandle<MultiWorkspace>,
    cx: &mut VisualTestAppContext,
) -> Result<()> {
    window.update(cx, |mw, _window, cx| {
        for workspace in mw.workspaces() {
            let project = workspace.read(cx).project().clone();
            project.update(cx, |project, cx| {
                let ids: Vec<_> = project.worktrees(cx).map(|wt| wt.read(cx).id()).collect();
                for id in ids {
                    project.remove_worktree(id, cx);
                }
            });
        }
    })?;

    cx.run_until_parked();

    cx.update_window(window.into(), |_, window, _cx| {
        window.remove_window();
    })?;

    cx.run_until_parked();

    for _ in 0..15 {
        cx.advance_clock(Duration::from_millis(100));
        cx.run_until_parked();
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn run_sidebar_duplicate_project_names_visual_tests(
    app_state: Arc<AppState>,
    cx: &mut VisualTestAppContext,
    update_baseline: bool,
) -> Result<TestResult> {
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let canonical_temp = temp_path.canonicalize()?;

    // Create directory structure where every leaf directory is named "zed" but
    // lives at a distinct path. This lets us test that the sidebar correctly
    // disambiguates projects whose names would otherwise collide.
    //
    //   code/zed/       — project1 (single worktree)
    //   code/foo/zed/   — project2 (single worktree)
    //   code/bar/zed/   — project3, first worktree
    //   code/baz/zed/   — project3, second worktree
    //
    // No two projects share a worktree path, so ProjectGroupBuilder will
    // place each in its own group.
    let code_zed = canonical_temp.join("code").join("zed");
    let foo_zed = canonical_temp.join("code").join("foo").join("zed");
    let bar_zed = canonical_temp.join("code").join("bar").join("zed");
    let baz_zed = canonical_temp.join("code").join("baz").join("zed");
    std::fs::create_dir_all(&code_zed)?;
    std::fs::create_dir_all(&foo_zed)?;
    std::fs::create_dir_all(&bar_zed)?;
    std::fs::create_dir_all(&baz_zed)?;

    cx.update(|cx| {
        cx.update_flags(true, vec!["agent-v2".to_string()]);
    });

    let mut has_baseline_update = None;

    // Two single-worktree projects whose leaf name is "zed"
    {
        let project1 = create_project_with_worktree(&code_zed, &app_state, cx)?;
        let project2 = create_project_with_worktree(&foo_zed, &app_state, cx)?;

        let window = open_sidebar_test_window(vec![project1, project2], &app_state, cx)?;

        let result = run_visual_test(
            "sidebar_two_projects_same_leaf_name",
            window.into(),
            cx,
            update_baseline,
        );

        cleanup_sidebar_test_window(window, cx)?;
        match result? {
            TestResult::Passed => {}
            TestResult::BaselineUpdated(path) => {
                has_baseline_update = Some(path);
            }
        }
    }

    // Three projects, third has two worktrees (all leaf names "zed")
    //
    // project1: code/zed
    // project2: code/foo/zed
    // project3: code/bar/zed + code/baz/zed
    //
    // Each project has a unique set of worktree paths, so they form
    // separate groups. The sidebar must disambiguate all three.
    {
        let project1 = create_project_with_worktree(&code_zed, &app_state, cx)?;
        let project2 = create_project_with_worktree(&foo_zed, &app_state, cx)?;

        let project3 = create_project_with_worktree(&bar_zed, &app_state, cx)?;
        let add_second_worktree = cx.update(|cx| {
            project3.update(cx, |project, cx| {
                project.find_or_create_worktree(&baz_zed, true, cx)
            })
        });
        cx.background_executor.allow_parking();
        cx.foreground_executor
            .block_test(add_second_worktree)
            .context("Failed to add second worktree to project 3")?;
        cx.background_executor.forbid_parking();
        cx.run_until_parked();

        let window = open_sidebar_test_window(vec![project1, project2, project3], &app_state, cx)?;

        let result = run_visual_test(
            "sidebar_three_projects_with_multi_worktree",
            window.into(),
            cx,
            update_baseline,
        );

        cleanup_sidebar_test_window(window, cx)?;
        match result? {
            TestResult::Passed => {}
            TestResult::BaselineUpdated(path) => {
                has_baseline_update = Some(path);
            }
        }
    }

    if let Some(path) = has_baseline_update {
        Ok(TestResult::BaselineUpdated(path))
    } else {
        Ok(TestResult::Passed)
    }
}
