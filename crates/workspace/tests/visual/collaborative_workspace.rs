const COLLABORATIVE_VISUAL_CONTRACT: &str =
    include_str!("../fixtures/collaborative_workspace/visual-contract.json");

use sha2::{Digest as _, Sha256};

#[gpui::test]
async fn collaborative_workspace_structural_contract(cx: &mut TestAppContext) {
    let contract: serde_json::Value = serde_json::from_str(COLLABORATIVE_VISUAL_CONTRACT)
        .expect("collaborative visual contract should be valid JSON");
    assert_eq!(contract["version"].as_u64(), Some(1));
    assert_eq!(
        contract["approval"]["basis"].as_str(),
        Some("user-provided reference artifacts")
    );

    let project = init_test_project("/visual-collaborative-project", cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    setup_sidebar_closed(&multi_workspace, cx);
    cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
    cx.run_until_parked();

    let cases = contract["cases"]
        .as_array()
        .expect("visual contract should define cases");
    assert_eq!(cases.len(), 2);
    for case in cases {
        let id = case["id"].as_str().expect("visual case should have an id");
        let width = case["viewport"]["width"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .expect("visual case width should fit u32");
        let height = case["viewport"]["height"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .expect("visual case height should fit u32");
        assert_reference_png_dimensions(
            case["reference"]["path"]
                .as_str()
                .expect("visual case should name its reference"),
            case["reference"]["sha256"]
                .as_str()
                .expect("visual case should pin its reference hash"),
            width,
            height,
        );

        cx.simulate_resize(gpui::size(px(width as f32), px(height as f32)));
        cx.run_until_parked();
        cx.debug_bounds("COLLABORATIVE-LAYOUT");
        cx.refresh()
            .expect("visual fixture should refresh after resize");
        cx.run_until_parked();

        let review_visible = case["review_visible"]
            .as_bool()
            .expect("visual case should define review visibility");
        let currently_visible = cx.debug_bounds("COLLABORATIVE-REVIEW-REGION").is_some();
        if currently_visible != review_visible {
            cx.dispatch_action(workspace::ToggleCollaborativeReview);
            cx.run_until_parked();
            cx.debug_bounds("COLLABORATIVE-LAYOUT");
            cx.refresh()
                .expect("visual fixture should refresh after layout toggle");
            cx.run_until_parked();
        }

        for selector in case["required_selectors"]
            .as_array()
            .expect("visual case should list required selectors")
        {
            let selector = selector
                .as_str()
                .expect("required selector should be a string");
            if selector == "COLLABORATIVE-TOP-BAR" {
                assert!(workspace.read_with(cx, |workspace, cx| {
                    workspace.collaborative_title_bar(cx).is_some()
                }));
                continue;
            }
            assert!(
                collaborative_visual_bounds(cx, selector).is_some(),
                "{id}: required visual region {selector} should render"
            );
        }
        for selector in case["absent_selectors"]
            .as_array()
            .expect("visual case should list absent selectors")
        {
            let selector = selector
                .as_str()
                .expect("absent selector should be a string");
            assert!(
                collaborative_visual_bounds(cx, selector).is_none(),
                "{id}: visual region {selector} should be absent"
            );
        }

        assert_collaborative_visual_geometry(cx, width, height, review_visible, id);
    }
}

fn collaborative_visual_bounds(
    cx: &mut gpui::VisualTestContext,
    selector: &str,
) -> Option<gpui::Bounds<gpui::Pixels>> {
    match selector {
        "COLLABORATIVE-RAIL" => cx.debug_bounds("COLLABORATIVE-RAIL"),
        "COLLABORATIVE-TOP-BAR" => cx.debug_bounds("COLLABORATIVE-TOP-BAR"),
        "COLLABORATIVE-TIMELINE-REGION" => cx.debug_bounds("COLLABORATIVE-TIMELINE-REGION"),
        "COLLABORATIVE-REVIEW-REGION" => cx.debug_bounds("COLLABORATIVE-REVIEW-REGION"),
        "COLLABORATIVE-COMPOSER" => cx.debug_bounds("COLLABORATIVE-COMPOSER"),
        "COLLABORATIVE-PARTICIPANT-STATUS" => cx.debug_bounds("COLLABORATIVE-PARTICIPANT-STATUS"),
        unknown => panic!("unknown collaborative visual selector {unknown}"),
    }
}

fn assert_reference_png_dimensions(
    path: &str,
    expected_sha256: &str,
    expected_width: u32,
    expected_height: u32,
) {
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("sidebar crate should live under the repository crates directory");
    let bytes = std::fs::read(repository_root.join(path))
        .unwrap_or_else(|error| panic!("failed to read visual reference {path}: {error}"));
    assert!(bytes.len() >= 24, "visual reference {path} is truncated");
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "{path} is not a PNG");
    let width = u32::from_be_bytes(
        bytes[16..20]
            .try_into()
            .expect("PNG width bytes should exist"),
    );
    let height = u32::from_be_bytes(
        bytes[20..24]
            .try_into()
            .expect("PNG height bytes should exist"),
    );
    assert_eq!((width, height), (expected_width, expected_height));
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_sha256);
}

fn assert_collaborative_visual_geometry(
    cx: &mut gpui::VisualTestContext,
    viewport_width: u32,
    viewport_height: u32,
    review_visible: bool,
    id: &str,
) {
    let rail = cx
        .debug_bounds("COLLABORATIVE-RAIL")
        .expect("collaborative rail should render");
    let layout = cx
        .debug_bounds("COLLABORATIVE-LAYOUT")
        .expect("collaborative layout should render");
    let timeline = cx
        .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
        .expect("collaborative timeline should render");
    let composer = cx
        .debug_bounds("COLLABORATIVE-COMPOSER")
        .expect("collaborative composer should render");
    let project_status = cx
        .debug_bounds("COLLABORATIVE-PARTICIPANT-STATUS")
        .expect("collaborative status should render");
    let status_bar = cx
        .debug_bounds("WORKSPACE-STATUS-BAR")
        .expect("the native status toolbar should render below the composer");

    assert_eq!(rail.left(), px(0.), "{id}: rail should anchor left");
    assert!(
        rail.size.width >= px(200.),
        "{id}: rail should retain density"
    );
    assert!(
        rail.size.width <= px(viewport_width as f32 * 0.35),
        "{id}: rail should leave room for the timeline"
    );
    for bounds in [layout, timeline, composer] {
        assert!(
            (f32::from(bounds.left()) - f32::from(rail.right())).abs() <= 1.0,
            "{id}: main surface should follow the rail border"
        );
    }
    assert_eq!(layout.right(), px(viewport_width as f32));
    assert_eq!(composer.right(), timeline.right());
    assert_eq!(composer.bottom(), status_bar.top());
    assert_eq!(status_bar.bottom(), timeline.bottom());
    assert_eq!(status_bar.left(), timeline.left());
    assert_eq!(status_bar.right(), timeline.right());
    assert!(status_bar.contains(&project_status.center()));
    assert!(composer.top() > timeline.top());
    assert!(project_status.bottom() <= px(viewport_height as f32));
    assert_eq!(timeline.left(), layout.left());
    assert_eq!(timeline.top(), layout.top());
    assert_eq!(timeline.bottom(), layout.bottom());

    if review_visible {
        let review = cx
            .debug_bounds("COLLABORATIVE-REVIEW-REGION")
            .expect("expanded fixture should render review");
        assert!(timeline.right() <= review.left());
        assert_eq!(review.right(), layout.right());
        assert_eq!(review.top(), layout.top());
        assert_eq!(review.bottom(), layout.bottom());
        assert!(review.size.width >= px(320.));
    } else {
        assert_eq!(
            timeline, layout,
            "{id}: collapsed timeline should fill layout"
        );
    }
}
